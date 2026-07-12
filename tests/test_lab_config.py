"""LabConfig and manifest correctness: defaults are all-off, unknown keys
fail closed, invalid values fail clearly, hashes are stable, manifests are
atomic and fail closed on corruption."""

import json

import pytest

from src.lab.lab_config import (
    DrawdownThrottleConfig,
    DynamicAllocationConfig,
    LabConfig,
    LabConfigError,
    VolTargetConfig,
)
from src.lab.manifest import (
    MANIFEST_SCHEMA_VERSION,
    ManifestError,
    atomic_write_json,
    cache_fingerprint,
    load_manifest,
    sha256_file,
)


# --- Defaults ---------------------------------------------------------------------

def test_default_config_is_all_off_and_valid():
    cfg = LabConfig()
    cfg.validate()
    assert cfg.enabled_features() == []
    assert not cfg.any_behavior_change()
    assert cfg.trace is True  # observational, behavior-neutral


def test_round_trip_preserves_hash():
    cfg = LabConfig()
    cfg.vol_target.enabled = True
    cfg.vol_target.target_annual_vol = 0.15
    restored = LabConfig.from_dict(cfg.to_dict())
    assert restored.config_hash() == cfg.config_hash()
    assert restored.vol_target.target_annual_vol == 0.15


def test_hash_changes_when_config_changes():
    a = LabConfig()
    b = LabConfig()
    b.drawdown_throttle.enabled = True
    assert a.config_hash() != b.config_hash()


# --- Fail-closed deserialization ----------------------------------------------------

def test_unknown_top_level_key_fails_closed():
    with pytest.raises(LabConfigError, match="Unknown lab config key"):
        LabConfig.from_dict({"vol_targget": {"enabled": True}})


def test_unknown_section_key_fails_closed():
    with pytest.raises(LabConfigError, match="Unknown key"):
        LabConfig.from_dict({"vol_target": {"enabeld": True}})


def test_non_dict_section_rejected():
    with pytest.raises(LabConfigError):
        LabConfig.from_dict({"vol_target": "on"})


def test_from_dict_validates_values():
    with pytest.raises(LabConfigError, match="target_annual_vol"):
        LabConfig.from_dict({"vol_target": {"enabled": True, "target_annual_vol": -1.0}})


# --- Validation rules ----------------------------------------------------------------

def test_vol_target_rejects_leverage():
    cfg = VolTargetConfig(enabled=True, max_exposure=1.5)
    with pytest.raises(LabConfigError, match="leverage"):
        cfg.validate()


def test_throttle_levels_must_be_ordered_and_negative():
    with pytest.raises(LabConfigError):
        DrawdownThrottleConfig(enabled=True, levels=((0.08, 0.75),)).validate()
    with pytest.raises(LabConfigError, match="ordered"):
        DrawdownThrottleConfig(enabled=True, levels=((-0.15, 0.5), (-0.08, 0.75))).validate()


def test_allocation_bounds_must_be_feasible():
    cfg = DynamicAllocationConfig(
        enabled=True,
        bounds={"momentum": (0.8, 0.9), "sector_rotation": (0.5, 0.9), "regime_switch": (0.0, 0.1)},
    )
    with pytest.raises(LabConfigError, match="infeasible"):
        cfg.validate()


def test_cash_yield_rejects_implausible_rate():
    cfg = LabConfig()
    cfg.cash_yield.annual_rate = 0.5
    with pytest.raises(LabConfigError):
        cfg.validate()


# --- Manifest ------------------------------------------------------------------------

def test_atomic_write_and_load_round_trip(tmp_path):
    payload = {"schema_version": MANIFEST_SCHEMA_VERSION, "hello": "world"}
    path = tmp_path / "run_manifest.json"
    atomic_write_json(path, payload)
    loaded = load_manifest(path)
    assert loaded["hello"] == "world"
    # No temp files left behind.
    assert list(tmp_path.glob("*.tmp")) == []


def test_corrupt_manifest_fails_closed(tmp_path):
    path = tmp_path / "run_manifest.json"
    path.write_text("{not json")
    with pytest.raises(ManifestError, match="Cannot read"):
        load_manifest(path)


def test_wrong_schema_version_fails_closed(tmp_path):
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps({"schema_version": 999}))
    with pytest.raises(ManifestError, match="schema_version"):
        load_manifest(path)


def test_missing_manifest_fails_closed(tmp_path):
    with pytest.raises(ManifestError):
        load_manifest(tmp_path / "does_not_exist.json")


def test_cache_fingerprint_is_deterministic(tmp_path):
    (tmp_path / "AAA.meta.json").write_text(json.dumps({
        "cached_start": "2020-01-01", "cached_end": "2024-12-31",
        "last_full_refresh": "2026-07-10", "yfinance_version": "1.2.0",
    }))
    fp1 = cache_fingerprint(str(tmp_path))
    fp2 = cache_fingerprint(str(tmp_path))
    assert fp1["digest"] == fp2["digest"]
    assert fp1["num_tickers"] == 1
    # Changing the cache changes the digest.
    (tmp_path / "BBB.meta.json").write_text(json.dumps({"cached_end": "2024-12-31"}))
    assert cache_fingerprint(str(tmp_path))["digest"] != fp1["digest"]


def test_sha256_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    assert sha256_file(f) == sha256_file(f)
    g = tmp_path / "b.txt"
    g.write_text("hello!")
    assert sha256_file(f) != sha256_file(g)


# --- Reason-code taxonomy -------------------------------------------------------------

def test_reason_codes_are_centralized_and_validated():
    from src.lab.reasons import REASON_CODES, UnknownReasonCode, human_reason, validate_code

    assert "SELECTED_TOP_K" in REASON_CODES
    assert "BELOW_TREND_FILTER" in REASON_CODES
    assert "REGIME_RISK_OFF" in REASON_CODES
    assert validate_code("CASH_FALLBACK") == "CASH_FALLBACK"
    with pytest.raises(UnknownReasonCode):
        validate_code("MADE_UP_CODE")
    # Templates render with missing fields as n/a, never raising.
    text = human_reason("BELOW_TREND_FILTER", {"close": 12.0})
    assert "12.0" in text and "n/a" in text
    with pytest.raises(UnknownReasonCode):
        human_reason("NOPE", {})
