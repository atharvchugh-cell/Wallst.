"""Reproducibility manifest for lab runs.

Every lab run writes a `run_manifest.json` capturing everything needed to
reproduce it byte-for-byte: git SHA, schema version, interpreter/dependency
versions, the full lab + portfolio configuration, universe definition,
data-cache fingerprint, seeds, and sha256 hashes of the artifacts the run
produced. Writes are atomic (write-temp-then-rename, same convention as
src/data.py) so a crashed run can never leave a half-written manifest.

The manifest also carries the HOLDOUT LEDGER: the holdout period is declared
up front, and the first time any lab code evaluates it, the timestamp is
recorded (see experiments.py). A manifest whose design sections changed after
the holdout was first evaluated is self-incriminating by construction.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from .. import config

MANIFEST_SCHEMA_VERSION = 1


def git_sha(repo_root: str | None = None) -> str | None:
    """Current HEAD sha, or None when git is unavailable (e.g. an exported
    tree). Never raises -- the manifest records what it can prove."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root or os.getcwd(),
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def dependency_versions() -> dict[str, str]:
    import matplotlib
    import numpy
    import yfinance

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": numpy.__version__,
        "matplotlib": matplotlib.__version__,
        "yfinance": yfinance.__version__,
    }


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_fingerprint(cache_dir: str = config.CACHE_DIR, tickers: list[str] | None = None) -> dict:
    """A cheap, deterministic fingerprint of the price cache actually visible
    to this run: per-ticker cached ranges + last refresh (from the .meta.json
    sidecars), plus one digest over all of it. Does NOT hash the CSV bodies
    (62MB+) -- the metadata identifies the snapshot; a changed snapshot always
    changes cached_end/last_full_refresh."""
    d = Path(cache_dir)
    entries: list[dict] = []
    if d.exists():
        for meta_path in sorted(d.glob("*.meta.json")):
            ticker = meta_path.name[: -len(".meta.json")]
            if tickers is not None and ticker not in tickers:
                continue
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception:
                entries.append({"ticker": ticker, "error": "unreadable meta"})
                continue
            entries.append({
                "ticker": ticker,
                "cached_start": meta.get("cached_start"),
                "cached_end": meta.get("cached_end"),
                "last_full_refresh": meta.get("last_full_refresh"),
                "yfinance_version": meta.get("yfinance_version"),
            })
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return {"cache_dir": str(d), "num_tickers": len(entries), "digest": digest, "entries": entries}


def atomic_write_json(path: str | Path, payload: dict) -> None:
    """Write JSON atomically: serialize to a temp file in the destination
    directory, then rename over the target. A failure can leave a stray temp
    file but never a truncated artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(fd)
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=False, default=str)
        shutil.move(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".txt.tmp")
    os.close(fd)
    try:
        with open(tmp, "w") as f:
            f.write(text)
        shutil.move(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


class ManifestError(Exception):
    pass


def build_manifest(
    *,
    run_kind: str,
    lab_config_dict: dict,
    lab_config_hash: str,
    weights: list[tuple[str, float]],
    capital: float,
    cost_bps: float,
    fractional_shares: bool,
    start: str,
    end: str,
    universe_mode: str,
    universe: list[str],
    universe_info: dict | None,
    strategy_params: dict[str, dict],
    warmup_days: dict[str, int],
    seeds: dict[str, int] | None = None,
    cash_yield_annual: float = 0.0,
    holdout: dict | None = None,
    cache_dir: str = config.CACHE_DIR,
    cache_tickers: list[str] | None = None,
    extra: dict | None = None,
) -> dict:
    """Assemble the run manifest. Artifact hashes are attached later via
    `attach_artifact_hashes` once the artifacts exist on disk."""
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_kind": run_kind,
        "generated_at": pd.Timestamp.now().isoformat(),
        "git_sha": git_sha(),
        "versions": dependency_versions(),
        "lab_config": lab_config_dict,
        "lab_config_hash": lab_config_hash,
        "portfolio": {
            "weights": {name: w for name, w in weights},
            "capital": capital,
            "cost_bps": cost_bps,
            "fractional_shares": fractional_shares,
            "start": str(start),
            "end": str(end),
            "benchmark": config.BENCHMARK_TICKER,
            "execution_lag": "signal at close of T, fill at close of next trading day (T+1)",
            "fill_price": "adjusted close (auto_adjust=True)",
            "cash_yield_annual": cash_yield_annual,
        },
        "universe": {
            "mode": universe_mode,
            "tickers": list(universe),
            "snapshot_info": universe_info,
            "survivorship_note": (
                "Stock universes are CURRENT-snapshot, survivorship-biased lists, "
                "not point-in-time constituents. See docs/RED_TEAM.md A1."
            ),
        },
        "strategy_params": strategy_params,
        "warmup_calendar_days": warmup_days,
        "seeds": seeds or {},
        "data_cache": cache_fingerprint(cache_dir, tickers=cache_tickers),
        "holdout": holdout or {"declared": None, "first_evaluated_at": None},
        "artifact_hashes": {},
    }
    if extra:
        manifest.update(extra)
    return manifest


def attach_artifact_hashes(manifest: dict, run_dir: str | Path,
                           exclude: tuple[str, ...] = ("run_manifest.json",)) -> dict:
    """sha256 every artifact in `run_dir` (excluding the manifest itself,
    which cannot contain its own hash) and record them in the manifest."""
    run_dir = Path(run_dir)
    hashes = {}
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.name not in exclude and not p.name.endswith(".tmp"):
            hashes[str(p.relative_to(run_dir))] = sha256_file(p)
    manifest["artifact_hashes"] = hashes
    return manifest


def write_manifest(manifest: dict, run_dir: str | Path) -> Path:
    path = Path(run_dir) / "run_manifest.json"
    atomic_write_json(path, manifest)
    return path


def load_manifest(path: str | Path) -> dict:
    """Load and structurally validate a manifest; corrupt or wrong-schema
    manifests FAIL CLOSED with a clear error rather than being half-used."""
    try:
        with open(path) as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ManifestError(f"Cannot read manifest {path}: {e}") from e
    if not isinstance(manifest, dict):
        raise ManifestError(f"Manifest {path} is not a JSON object")
    version = manifest.get("schema_version")
    if version != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"Manifest {path} has schema_version {version!r}; this code expects "
            f"{MANIFEST_SCHEMA_VERSION}. Refusing to interpret a mismatched schema."
        )
    return manifest
