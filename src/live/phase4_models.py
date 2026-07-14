"""Phase-4 policy, signed target snapshots, and canonical serialization.

The signature is an HMAC-SHA256 over a domain-separated content hash using an
operator-owned local key file.  The key is never accepted as a CLI value,
stored in a snapshot, or written to the ledger.  HMAC is deliberately offered
behind a small interface so an asymmetric or hardware-backed signer can replace
it without changing the publisher or execution workflow.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from .deployment import DeploymentConfig
from .models import ZERO, as_decimal, ensure_aware, json_safe


PHASE4_CONFIG_SCHEMA_VERSION = 1
TARGET_SNAPSHOT_SCHEMA_VERSION = 1
SIGNATURE_DOMAIN = b"wallst-strategy-lab/phase4-target/v1\0"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class Phase4Error(RuntimeError):
    """A Phase-4 policy, publication, or supervision invariant failed."""


class OperationMode(str, Enum):
    OBSERVE = "observe"
    SHADOW = "shadow"
    PAPER_MANUAL = "paper_manual"
    PAPER_SUPERVISED = "paper_supervised"

    @property
    def can_submit_paper(self) -> bool:
        return self in {self.PAPER_MANUAL, self.PAPER_SUPERVISED}


def canonical_bytes(payload: Any) -> bytes:
    """Serialize deterministically, rejecting NaN/Infinity and key ambiguity."""
    try:
        return json.dumps(
            json_safe(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Phase4Error("Snapshot content is not canonically serializable") from exc


def content_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def account_fingerprint(account_id: str, system_id: str) -> str:
    if not account_id or not system_id:
        raise Phase4Error("Account and system identifiers are required for fingerprinting")
    return hashlib.sha256(
        f"wallst-strategy-lab:{system_id}:{account_id}".encode("utf-8")
    ).hexdigest()


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value.strip()):
        raise Phase4Error(f"{label} must match {_IDENTIFIER.pattern}")
    return value.strip()


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool):
        raise Phase4Error(f"{label} must be numeric")
    try:
        return as_decimal(value)
    except (ArithmeticError, TypeError, ValueError) as exc:
        raise Phase4Error(f"{label} must be a finite decimal") from exc


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Phase4Error(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class Phase4Policy:
    system_id: str
    mode: OperationMode = OperationMode.OBSERVE
    publisher_identity: str = "phase4-publisher"
    require_signing: bool = True
    signing_key_id: str = "phase4-local-v1"
    # Seven calendar days covers a Friday month-end followed by a Monday
    # execution (and exchange holidays). Execution remains pinned to the exact
    # signed next session and fresh quotes; this is not an open-ended signal.
    snapshot_ttl_seconds: int = 604800
    max_quote_spread_bps: Decimal = Decimal("50")
    max_price_deviation_bps: Decimal = Decimal("100")
    max_snapshot_age_seconds: int = 604800
    max_open_order_age_seconds: int = 1800
    min_trade_notional: Decimal = Decimal("25")
    max_cash_deployment_pct: Decimal = Decimal("1")
    max_aggregate_ticker_weight: Decimal = Decimal("0.25")
    dirty_worktree_policy: str = "reject"
    automatic_backup_directory: str = "phase4_backups"
    backup_retention: int = 30
    critical_alert_escalation_seconds: int = 900
    schema_version: int = PHASE4_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "system_id", _identifier(self.system_id, "system_id"))
        object.__setattr__(
            self, "publisher_identity", _identifier(self.publisher_identity, "publisher_identity")
        )
        object.__setattr__(self, "signing_key_id", _identifier(self.signing_key_id, "signing_key_id"))
        if not isinstance(self.mode, OperationMode):
            object.__setattr__(self, "mode", OperationMode(str(self.mode)))
        for name in (
            "max_quote_spread_bps", "max_price_deviation_bps", "min_trade_notional",
            "max_cash_deployment_pct", "max_aggregate_ticker_weight",
        ):
            object.__setattr__(self, name, _decimal(getattr(self, name), name))
        for name in (
            "snapshot_ttl_seconds", "max_snapshot_age_seconds", "max_open_order_age_seconds",
            "backup_retention", "critical_alert_escalation_seconds",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if self.schema_version != PHASE4_CONFIG_SCHEMA_VERSION:
            raise Phase4Error("Unsupported Phase-4 configuration schema")
        if self.dirty_worktree_policy not in {"reject", "allow_labelled"}:
            raise Phase4Error("dirty_worktree_policy must be reject or allow_labelled")
        if (
            not isinstance(self.automatic_backup_directory, str)
            or not self.automatic_backup_directory.strip()
            or "\x00" in self.automatic_backup_directory
        ):
            raise Phase4Error("automatic_backup_directory must be a non-empty path")
        if self.max_quote_spread_bps > Decimal("500"):
            raise Phase4Error("max_quote_spread_bps may not exceed 500")
        if self.max_price_deviation_bps > Decimal("500"):
            raise Phase4Error("max_price_deviation_bps may not exceed 500")
        if self.snapshot_ttl_seconds > 7 * 24 * 60 * 60:
            raise Phase4Error("snapshot_ttl_seconds may not exceed 7 calendar days")
        if self.max_snapshot_age_seconds > 7 * 24 * 60 * 60:
            raise Phase4Error("max_snapshot_age_seconds may not exceed 7 calendar days")
        if self.snapshot_ttl_seconds > self.max_snapshot_age_seconds:
            raise Phase4Error("snapshot TTL may not exceed maximum snapshot age")
        if not ZERO < self.max_cash_deployment_pct <= Decimal("1"):
            raise Phase4Error("max_cash_deployment_pct must be in (0, 1]")
        if not ZERO < self.max_aggregate_ticker_weight <= Decimal("1"):
            raise Phase4Error("max_aggregate_ticker_weight must be in (0, 1]")
        if self.min_trade_notional < ZERO:
            raise Phase4Error("min_trade_notional cannot be negative")

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Phase4Policy":
        expected = {
            "schema_version", "system_id", "mode", "publisher_identity", "require_signing",
            "signing_key_id", "snapshot_ttl_seconds", "max_quote_spread_bps",
            "max_price_deviation_bps", "max_snapshot_age_seconds",
            "max_open_order_age_seconds", "min_trade_notional", "max_cash_deployment_pct",
            "max_aggregate_ticker_weight", "dirty_worktree_policy", "backup_retention",
            "automatic_backup_directory", "critical_alert_escalation_seconds",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise Phase4Error(
                f"Phase-4 config keys do not match schema; missing={sorted(expected-set(payload))}, "
                f"unknown={sorted(set(payload)-expected)}"
            )
        if not isinstance(payload["require_signing"], bool):
            raise Phase4Error("require_signing must be a JSON boolean")
        return cls(
            system_id=payload["system_id"],
            mode=OperationMode(str(payload["mode"])),
            publisher_identity=payload["publisher_identity"],
            require_signing=payload["require_signing"],
            signing_key_id=payload["signing_key_id"],
            snapshot_ttl_seconds=payload["snapshot_ttl_seconds"],
            max_quote_spread_bps=payload["max_quote_spread_bps"],
            max_price_deviation_bps=payload["max_price_deviation_bps"],
            max_snapshot_age_seconds=payload["max_snapshot_age_seconds"],
            max_open_order_age_seconds=payload["max_open_order_age_seconds"],
            min_trade_notional=payload["min_trade_notional"],
            max_cash_deployment_pct=payload["max_cash_deployment_pct"],
            max_aggregate_ticker_weight=payload["max_aggregate_ticker_weight"],
            dirty_worktree_policy=payload["dirty_worktree_policy"],
            automatic_backup_directory=payload["automatic_backup_directory"],
            backup_retention=payload["backup_retention"],
            critical_alert_escalation_seconds=payload["critical_alert_escalation_seconds"],
            schema_version=payload["schema_version"],
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "Phase4Policy":
        from .deployment import load_strict_json

        return cls.from_payload(load_strict_json(path))

    def to_payload(self) -> dict[str, Any]:
        return json_safe({
            "schema_version": self.schema_version,
            "system_id": self.system_id,
            "mode": self.mode.value,
            "publisher_identity": self.publisher_identity,
            "require_signing": self.require_signing,
            "signing_key_id": self.signing_key_id,
            "snapshot_ttl_seconds": self.snapshot_ttl_seconds,
            "max_quote_spread_bps": self.max_quote_spread_bps,
            "max_price_deviation_bps": self.max_price_deviation_bps,
            "max_snapshot_age_seconds": self.max_snapshot_age_seconds,
            "max_open_order_age_seconds": self.max_open_order_age_seconds,
            "min_trade_notional": self.min_trade_notional,
            "max_cash_deployment_pct": self.max_cash_deployment_pct,
            "max_aggregate_ticker_weight": self.max_aggregate_ticker_weight,
            "dirty_worktree_policy": self.dirty_worktree_policy,
            "automatic_backup_directory": self.automatic_backup_directory,
            "backup_retention": self.backup_retention,
            "critical_alert_escalation_seconds": self.critical_alert_escalation_seconds,
        })

    def validate_deployment(self, deployment: DeploymentConfig) -> None:
        weights = dict(deployment.sleeve_weights)
        expected = {
            "momentum": Decimal("0.60"),
            "sector_rotation": Decimal("0.35"),
            "regime_switch": Decimal("0.05"),
        }
        if weights != expected:
            raise Phase4Error("Phase 4 requires the exact 60/35/5 registered sleeve mix")
        if deployment.risk_limits.max_symbol_exposure_pct > self.max_aggregate_ticker_weight:
            raise Phase4Error("Deployment symbol limit exceeds Phase-4 aggregate ticker limit")
        if deployment.risk_limits.min_cash_buffer < ZERO:
            raise Phase4Error("Deployment cash buffer is invalid")


class SnapshotSigner(Protocol):
    key_id: str
    algorithm: str

    def sign(self, digest: str) -> str: ...

    def verify(self, digest: str, signature: str) -> bool: ...


class HMACFileSigner:
    """Operator-controlled local HMAC key with strict file checks."""

    algorithm = "hmac-sha256"

    def __init__(self, key_path: str | Path, key_id: str) -> None:
        self.key_id = _identifier(key_id, "signing key ID")
        path = Path(key_path).expanduser()
        if path.is_symlink():
            raise Phase4Error("Signing key may not be a symbolic link")
        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(str(path), flags)
            with os.fdopen(fd, "rb") as handle:
                file_stat = os.fstat(handle.fileno())
                if not stat.S_ISREG(file_stat.st_mode):
                    raise Phase4Error("Signing key must be a regular file")
                if hasattr(os, "geteuid") and file_stat.st_uid != os.geteuid():
                    raise Phase4Error("Signing key must be owned by the current operator")
                mode = stat.S_IMODE(file_stat.st_mode)
                if mode & 0o077:
                    raise Phase4Error("Signing key permissions must not grant group/other access")
                key = handle.read(4097)
        except OSError as exc:
            raise Phase4Error("Cannot read operator signing key") from exc
        if len(key) < 32 or len(key) > 4096:
            raise Phase4Error("Signing key must contain 32-4096 bytes")
        self._key = key

    def sign(self, digest: str) -> str:
        if not _HEX64.fullmatch(digest):
            raise Phase4Error("Content digest must be lowercase SHA-256 hex")
        return hmac.new(self._key, SIGNATURE_DOMAIN + digest.encode("ascii"), hashlib.sha256).hexdigest()

    def verify(self, digest: str, signature: str) -> bool:
        return bool(_HEX64.fullmatch(signature)) and hmac.compare_digest(
            self.sign(digest), signature
        )


@dataclass(frozen=True)
class PublishedTargetSnapshot:
    snapshot_id: str
    content_hash: str
    content: dict[str, Any]
    signature_algorithm: str | None
    signature_key_id: str | None
    signature: str | None
    schema_version: int = TARGET_SNAPSHOT_SCHEMA_VERSION

    @classmethod
    def create(
        cls, content: dict[str, Any], *, signer: SnapshotSigner | None = None
    ) -> "PublishedTargetSnapshot":
        digest = content_hash(content)
        signature = signer.sign(digest) if signer is not None else None
        return cls(
            snapshot_id=f"snapshot-{digest[:24]}",
            content_hash=digest,
            content=json.loads(canonical_bytes(content)),
            signature_algorithm=signer.algorithm if signer is not None else None,
            signature_key_id=signer.key_id if signer is not None else None,
            signature=signature,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "snapshot_schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "content_hash": self.content_hash,
            "signature": {
                "signed": self.signature is not None,
                "algorithm": self.signature_algorithm,
                "key_id": self.signature_key_id,
                "value": self.signature,
            },
            # Return an isolated canonical copy so callers inspecting or
            # redacting an envelope cannot mutate the frozen in-memory
            # snapshot through a shared nested dict reference.
            "content": json.loads(canonical_bytes(self.content)),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PublishedTargetSnapshot":
        if not isinstance(payload, dict) or set(payload) != {
            "snapshot_schema_version", "snapshot_id", "content_hash", "signature", "content"
        }:
            raise Phase4Error("Target snapshot envelope does not match schema")
        if payload["snapshot_schema_version"] != TARGET_SNAPSHOT_SCHEMA_VERSION:
            raise Phase4Error("Unsupported target snapshot schema")
        sig = payload["signature"]
        if not isinstance(sig, dict) or set(sig) != {"signed", "algorithm", "key_id", "value"}:
            raise Phase4Error("Target snapshot signature block is invalid")
        if not isinstance(sig["signed"], bool):
            raise Phase4Error("Target snapshot signed flag must be boolean")
        expected_hash = content_hash(payload["content"])
        if not hmac.compare_digest(str(payload["content_hash"]), expected_hash):
            raise Phase4Error("Target snapshot content hash is invalid")
        expected_id = f"snapshot-{expected_hash[:24]}"
        if payload["snapshot_id"] != expected_id:
            raise Phase4Error("Target snapshot ID is invalid")
        if sig["signed"] != bool(sig["value"]):
            raise Phase4Error("Target snapshot signature metadata is inconsistent")
        if not sig["signed"] and any(
            value is not None for value in (sig["algorithm"], sig["key_id"], sig["value"])
        ):
            raise Phase4Error("Unsigned target snapshot may not carry signature metadata")
        if sig["signed"] and (
            sig["algorithm"] != "hmac-sha256"
            or not isinstance(sig["key_id"], str)
            or not _HEX64.fullmatch(str(sig["value"]))
        ):
            raise Phase4Error("Target snapshot signature is malformed")
        return cls(
            snapshot_id=expected_id,
            content_hash=expected_hash,
            content=json.loads(canonical_bytes(payload["content"])),
            signature_algorithm=sig["algorithm"],
            signature_key_id=sig["key_id"],
            signature=sig["value"],
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "PublishedTargetSnapshot":
        from .deployment import load_strict_json

        # Snapshot hashes cover native JSON number types. Deployment/policy
        # configs intentionally parse decimal literals as Decimal, but doing
        # that to a signed envelope would change its canonical representation.
        return cls.from_payload(load_strict_json(path, parse_floats_as_decimal=False))

    def verify(self, policy: Phase4Policy, signer: SnapshotSigner | None, *, now: datetime) -> None:
        parsed_now = ensure_aware(now, "snapshot verification time")
        expected_hash = content_hash(self.content)
        if not hmac.compare_digest(self.content_hash, expected_hash):
            raise Phase4Error("Target snapshot content hash is invalid")
        if self.snapshot_id != f"snapshot-{expected_hash[:24]}":
            raise Phase4Error("Target snapshot ID is invalid")
        if self.content.get("target_snapshot_schema_version") != TARGET_SNAPSHOT_SCHEMA_VERSION:
            raise Phase4Error("Target snapshot content schema is invalid")
        try:
            created = ensure_aware(
                datetime.fromisoformat(str(self.content["creation_timestamp"]).replace("Z", "+00:00")),
                "snapshot creation_timestamp",
            )
            expires = ensure_aware(
                datetime.fromisoformat(str(self.content["expiration_time"]).replace("Z", "+00:00")),
                "snapshot expiration_time",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise Phase4Error("Snapshot timestamps are invalid") from exc
        age = (parsed_now - created).total_seconds()
        lifetime = (expires - created).total_seconds()
        if age < -2:
            raise Phase4Error("Snapshot creation time is in the future")
        if lifetime <= 0 or lifetime > policy.snapshot_ttl_seconds:
            raise Phase4Error("Target snapshot expiration exceeds policy")
        if age > policy.max_snapshot_age_seconds or parsed_now >= expires:
            raise Phase4Error("Target snapshot is expired")
        if policy.require_signing and self.signature is None:
            raise Phase4Error("Unsigned target snapshot is not executable under this policy")
        if self.signature is not None:
            if signer is None:
                raise Phase4Error("A signer is required to verify this target snapshot")
            if self.signature_key_id != policy.signing_key_id or signer.key_id != policy.signing_key_id:
                raise Phase4Error("Target snapshot signing key ID does not match policy")
            if self.signature_algorithm != signer.algorithm or not signer.verify(
                self.content_hash, self.signature
            ):
                raise Phase4Error("Target snapshot signature verification failed")
