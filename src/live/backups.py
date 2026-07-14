"""SQLite-safe, versioned Phase-4 backup and verified restore operations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .alerts import AlertManager
from .ledger import Ledger, LedgerError
from .phase4_models import Phase4Error, canonical_bytes
from .phase4_store import Phase4Store


_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_BACKUP_ID = re.compile(r"^backup-[0-9]{8}T[0-9]{6}-[0-9a-f]{8}$")


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise Phase4Error("Backup content must be a regular non-symlink file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _assert_config_has_no_credentials(payload: Any, prefix: str = "") -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key).lower()
            path = f"{prefix}.{key}" if prefix else str(key)
            if any(token in lowered for token in ("api_secret", "api_key", "password", "credential", "private_key")):
                raise Phase4Error(f"Critical config backup refuses credential-like field {path}")
            _assert_config_has_no_credentials(value, path)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _assert_config_has_no_credentials(value, f"{prefix}[{index}]")


class BackupManager:
    def __init__(
        self,
        ledger: Ledger | None,
        backup_root: str | Path,
        *,
        retention: int = 30,
        alerts: AlertManager | None = None,
    ) -> None:
        if ledger is not None and ledger.path == ":memory:":
            raise Phase4Error("Automatic backups require a file-backed execution ledger")
        if retention <= 0:
            raise ValueError("Backup retention must be positive")
        self.ledger = ledger
        self.store = Phase4Store(ledger) if ledger is not None else None
        raw_root = Path(backup_root).expanduser()
        if raw_root.is_symlink():
            raise Phase4Error("Backup root may not be a symbolic link")
        self.backup_root = raw_root.resolve()
        if ledger is not None and self.backup_root == Path(ledger.path).resolve().parent:
            raise Phase4Error("Backup root must be dedicated and separate from the ledger directory")
        self.retention = retention
        self.alerts = alerts

    def create(self, critical_configs: tuple[str | Path, ...] = ()) -> dict:
        if self.ledger is None or self.store is None:
            raise Phase4Error("Backup creation requires an open execution ledger")
        if self.backup_root.exists():
            root_stat = self.backup_root.stat()
            if (
                not stat.S_ISDIR(root_stat.st_mode)
                or (hasattr(os, "geteuid") and root_stat.st_uid != os.geteuid())
                or stat.S_IMODE(root_stat.st_mode) & 0o077
            ):
                raise Phase4Error(
                    "Existing backup root must be an operator-owned private directory"
                )
            unexpected = [
                entry for entry in self.backup_root.iterdir()
                if not (
                    entry.is_dir() and not entry.is_symlink()
                    and _BACKUP_ID.fullmatch(entry.name)
                )
            ]
            if unexpected:
                raise Phase4Error("Existing backup root is not a dedicated Phase-4 backup directory")
        else:
            self.backup_root.mkdir(parents=True, mode=0o700)
        now = self.ledger.clock()
        backup_id = f"backup-{now.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        temp_dir = Path(tempfile.mkdtemp(prefix=f".{backup_id}-", dir=self.backup_root))
        final_dir = self.backup_root / backup_id
        try:
            db_path = temp_dir / "execution-ledger.sqlite3"
            destination = sqlite3.connect(str(db_path))
            try:
                self.ledger.conn.backup(destination)
                destination.commit()
            finally:
                destination.close()
            os.chmod(db_path, 0o400)
            self._verify_database(db_path)
            config_entries = []
            config_dir = temp_dir / "config"
            for index, raw_path in enumerate(critical_configs):
                unresolved_source = Path(raw_path).expanduser()
                if unresolved_source.is_symlink():
                    raise Phase4Error("Critical config must not be a symbolic link")
                source = unresolved_source.resolve()
                if source.name.startswith(".env"):
                    raise Phase4Error("Critical config must be a regular non-.env file")
                try:
                    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                    descriptor = os.open(str(unresolved_source), flags)
                    try:
                        source_stat = os.fstat(descriptor)
                        if not stat.S_ISREG(source_stat.st_mode):
                            raise Phase4Error("Critical config must be a regular file")
                        with os.fdopen(descriptor, "rb", closefd=False) as handle:
                            raw_config = handle.read(1_000_001)
                    finally:
                        os.close(descriptor)
                    if len(raw_config) > 1_000_000:
                        raise Phase4Error("Critical config exceeds 1 MB")
                    payload = json.loads(raw_config.decode("utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise Phase4Error("Critical config must be valid JSON") from exc
                _assert_config_has_no_credentials(payload)
                config_dir.mkdir(mode=0o700, exist_ok=True)
                copied = config_dir / f"{index:02d}-{source.name}"
                copied.write_bytes(raw_config)
                os.chmod(copied, 0o400)
                config_entries.append({
                    "source_name": source.name,
                    "backup_name": copied.name,
                    "sha256": _sha256(copied),
                })
            manifest = {
                "backup_schema_version": 1,
                "backup_id": backup_id,
                "created_at": now.isoformat(),
                "ledger_file": db_path.name,
                "ledger_sha256": _sha256(db_path),
                "critical_configs": config_entries,
                "contains_credentials": False,
            }
            manifest_bytes = canonical_bytes(manifest)
            manifest_path = temp_dir / "manifest.json"
            manifest_path.write_bytes(manifest_bytes + b"\n")
            os.chmod(manifest_path, 0o400)
            manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
            hash_path = temp_dir / "manifest.sha256"
            hash_path.write_text(manifest_hash + "\n", encoding="ascii")
            os.chmod(hash_path, 0o400)
            durable_files = [db_path, manifest_path, hash_path]
            if config_dir.exists():
                durable_files.extend(config_dir.iterdir())
            for durable_file in durable_files:
                _fsync_file(durable_file)
            if config_dir.exists():
                _fsync_directory(config_dir)
            _fsync_directory(temp_dir)
            os.rename(temp_dir, final_dir)
            _fsync_directory(self.backup_root)
            row = self.store.record_backup(
                backup_id, manifest["ledger_sha256"], manifest_hash, str(final_dir)
            )
            self._enforce_retention()
            return row
        except Exception as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if self.alerts:
                self.alerts.emit(
                    "critical", "backup_failure", f"Backup failed: {type(exc).__name__}",
                    entity_id=backup_id, dedupe_key="automatic-backup-failure",
                )
            raise

    @staticmethod
    def _verify_database(path: Path) -> None:
        try:
            # SQLite backups inherit the source's WAL journal-mode marker.
            # ``immutable=1`` is appropriate for this mode-0400 verification
            # copy and prevents SQLite from trying to create WAL sidecars.
            connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
            try:
                quick = connection.execute("PRAGMA quick_check").fetchone()
                foreign = connection.execute("PRAGMA foreign_key_check").fetchone()
                if quick is None or quick[0] != "ok" or foreign is not None:
                    raise Phase4Error("Backup database integrity verification failed")
            finally:
                connection.close()
        except sqlite3.Error as exc:
            raise Phase4Error("Backup database cannot be opened") from exc

    def verify(self, backup_dir: str | Path) -> dict:
        raw_directory = Path(backup_dir).expanduser()
        if raw_directory.is_symlink() or not raw_directory.is_dir():
            raise Phase4Error("Backup directory must be a regular directory, not a symlink")
        directory = raw_directory.resolve()
        try:
            manifest_raw = (directory / "manifest.json").read_bytes().rstrip(b"\n")
            manifest = json.loads(manifest_raw)
            expected_manifest_hash = (directory / "manifest.sha256").read_text(
                encoding="ascii"
            ).strip()
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise Phase4Error("Backup manifest cannot be read") from exc
        actual_manifest_hash = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
        if not _HEX64.fullmatch(expected_manifest_hash) or expected_manifest_hash != actual_manifest_hash:
            raise Phase4Error("Backup manifest hash verification failed")
        expected_keys = {
            "backup_schema_version", "backup_id", "created_at", "ledger_file",
            "ledger_sha256", "critical_configs", "contains_credentials",
        }
        if not isinstance(manifest, dict) or set(manifest) != expected_keys:
            raise Phase4Error("Backup manifest schema is invalid")
        if (
            manifest["backup_schema_version"] != 1
            or not isinstance(manifest["backup_id"], str)
            or not _BACKUP_ID.fullmatch(manifest["backup_id"])
            or manifest["ledger_file"] != "execution-ledger.sqlite3"
            or manifest["contains_credentials"] is not False
            or not isinstance(manifest["critical_configs"], list)
            or not _HEX64.fullmatch(str(manifest["ledger_sha256"]))
        ):
            raise Phase4Error("Backup manifest safety fields are invalid")
        db_path = directory / str(manifest["ledger_file"])
        if _sha256(db_path) != manifest["ledger_sha256"]:
            raise Phase4Error("Backup ledger hash verification failed")
        for entry in manifest.get("critical_configs", []):
            if (
                not isinstance(entry, dict)
                or set(entry) != {"source_name", "backup_name", "sha256"}
                or not isinstance(entry["backup_name"], str)
                or Path(entry["backup_name"]).name != entry["backup_name"]
                or not _HEX64.fullmatch(str(entry["sha256"]))
            ):
                raise Phase4Error("Backup critical-config manifest entry is invalid")
            path = directory / "config" / entry["backup_name"]
            if _sha256(path) != entry["sha256"]:
                raise Phase4Error("Backup critical-config hash verification failed")
        self._verify_database(db_path)
        return manifest

    def restore(
        self,
        backup_dir: str | Path,
        destination: str | Path,
        *,
        active_ledger_path: str | Path | None = None,
        confirm_replace: bool = False,
    ) -> Path:
        manifest = self.verify(backup_dir)
        source = Path(backup_dir).expanduser().resolve() / manifest["ledger_file"]
        raw_target = Path(destination).expanduser()
        if raw_target.is_symlink():
            raise Phase4Error("Restore destination may not be a symbolic link")
        target = raw_target.resolve()
        active = Path(active_ledger_path).expanduser().resolve() if active_ledger_path else None
        replacing_active = active is not None and target == active
        if (target.exists() or replacing_active) and not confirm_replace:
            raise Phase4Error("Replacing an existing/active ledger requires explicit confirmation")
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            source_descriptor = os.open(str(source), flags)
            try:
                with os.fdopen(source_descriptor, "rb", closefd=False) as source_handle:
                    with temp_path.open("wb") as target_handle:
                        shutil.copyfileobj(source_handle, target_handle)
                        target_handle.flush()
                        os.fsync(target_handle.fileno())
            finally:
                os.close(source_descriptor)
            os.chmod(temp_path, 0o600)
            if _sha256(temp_path) != manifest["ledger_sha256"]:
                raise Phase4Error("Restored temporary ledger hash verification failed")
            self._verify_database(temp_path)
            os.replace(temp_path, target)
            os.chmod(target, 0o600)
            _fsync_file(target)
            _fsync_directory(target.parent)
            return target
        except Exception:
            try:
                temp_path.unlink()
            except OSError:
                pass
            raise

    def _enforce_retention(self) -> None:
        directories = sorted(
            (
                path for path in self.backup_root.glob("backup-*")
                if (
                    path.is_dir() and not path.is_symlink()
                    and _BACKUP_ID.fullmatch(path.name)
                )
            ),
            key=lambda path: path.name,
        )
        for expired in directories[:-self.retention]:
            shutil.rmtree(expired)
            if self.ledger is None:
                raise Phase4Error("Backup retention requires an open execution ledger")
            self.ledger.record_audit(
                "backup_expired_by_retention", "backup", expired.name,
                {"retention": self.retention},
            )
