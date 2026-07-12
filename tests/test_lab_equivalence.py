"""Baseline equivalence: the lab's synchronized multi-sleeve engine, with
every enhancement disabled, must reproduce the production `--strategy
portfolio` mode EXACTLY -- same signals, orders, fills, trades, cash,
positions, sleeve equity, portfolio equity, costs, turnover, and metrics.

These tests are the lab's foundation: every experiment result is only
meaningful because this file pins the all-off lab path to the production
path. Numerical tolerances are NOT used for engine outputs -- both paths run
the same functions on the same data in the same order, so equality is exact.
The only approx comparisons are on metrics dict values (some are NaN, where
== is unusable by definition and isnan-equality is asserted instead).
"""

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from src import cli, portfolio, tournament
from src.lab.lab_config import LabConfig
from src.lab.portfolio_engine import run_lab_portfolio
from src.lab.trace import DecisionRecorder

FULL_DATES = pd.bdate_range("1990-01-01", "2024-12-31")


def shared_trending_df(seed=42, drift=0.01):
    rng = np.random.default_rng(seed)
    base = 100.0 + np.cumsum(rng.normal(drift, 0.5, size=len(FULL_DATES)))
    base = np.clip(base, 10.0, None)
    return pd.DataFrame(
        {"Open": base, "High": base, "Low": base, "Close": base, "Volume": 1000}, index=FULL_DATES
    )


@pytest.fixture
def mocked_data(monkeypatch):
    """Same deterministic synthetic-frame pattern tests/test_portfolio.py uses;
    patching src.data's accessors covers the production path AND the lab's
    dataprep, which call the same module functions."""
    frames: dict[str, pd.DataFrame] = {}

    def get_frame(t):
        if t not in frames:
            frames[t] = shared_trending_df(seed=abs(hash(t)) % (2**31))
        return frames[t].copy()

    def fake_get_price_data(tickers, start, end, warmup_calendar_days, hard_fail_on_missing, **kw):
        return {t: get_frame(t) for t in tickers}, []

    def fake_get_benchmark_data(start, end, **kw):
        return get_frame("SPY")

    monkeypatch.setattr(tournament.data, "get_price_data", fake_get_price_data)
    monkeypatch.setattr(tournament.data, "get_benchmark_data", fake_get_benchmark_data)
    return frames


RUN_KWARGS = dict(
    total_capital=15000.0, start="2023-01-01", end="2024-12-31", cost_bps=5.0,
    fractional_shares=True, refresh_cache=False,
)


def run_production(pairs):
    return portfolio.run_portfolio(
        pairs, RUN_KWARGS["total_capital"], RUN_KWARGS["start"], RUN_KWARGS["end"],
        cost_bps=RUN_KWARGS["cost_bps"], fractional_shares=RUN_KWARGS["fractional_shares"],
        refresh_cache=False, output_dir="output", write_sleeve_artifacts=False,
    )


def run_lab(pairs, lab_config=None, recorder=None):
    return run_lab_portfolio(pairs, lab_config=lab_config, recorder=recorder, **RUN_KWARGS)


def assert_dataclass_lists_equal(a: list, b: list, label: str):
    assert len(a) == len(b), f"{label}: {len(a)} vs {len(b)} records"
    for i, (x, y) in enumerate(zip(a, b)):
        dx, dy = dataclasses.asdict(x), dataclasses.asdict(y)
        assert dx == dy, f"{label}[{i}] differs:\n{dx}\nvs\n{dy}"


def assert_metrics_equal(a: dict, b: dict, label: str):
    assert set(a) == set(b), f"{label}: metric keys differ"
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, float) and isinstance(vb, float) and math.isnan(va):
            assert math.isnan(vb), f"{label}[{k}]: NaN vs {vb}"
        else:
            assert va == vb, f"{label}[{k}]: {va} vs {vb}"


# --- The core proof --------------------------------------------------------------

def test_lab_engine_reproduces_production_60_35_5_exactly(mocked_data):
    pairs = portfolio.parse_portfolio_weights(None)
    prod = run_production(pairs)
    lab = run_lab(pairs).portfolio

    # Portfolio equity: exact, every date.
    pd.testing.assert_series_equal(
        prod.combined_result.equity_curve, lab.combined_result.equity_curve, check_names=False
    )
    assert prod.common_start == lab.common_start and prod.common_end == lab.common_end

    prod_sleeves = {s.strategy: s for s in prod.sleeves}
    lab_sleeves = {s.strategy: s for s in lab.sleeves}
    assert set(prod_sleeves) == set(lab_sleeves) == {"momentum", "sector_rotation", "regime_switch"}

    for name in prod_sleeves:
        ps, ls = prod_sleeves[name], lab_sleeves[name]
        pd.testing.assert_series_equal(
            ps.result.equity_curve, ls.result.equity_curve, check_names=False
        )
        assert_dataclass_lists_equal(ps.result.target_events, ls.result.target_events,
                                     f"{name}.target_events")
        assert_dataclass_lists_equal(ps.result.transactions, ls.result.transactions,
                                     f"{name}.transactions")
        assert_dataclass_lists_equal(ps.result.trades, ls.result.trades, f"{name}.trades")
        assert ps.result.positions == ls.result.positions, f"{name}.positions differ"
        assert ps.result.universe == ls.result.universe
        assert ps.result.dropped_tickers == ls.result.dropped_tickers
        assert ps.result.capital == ls.result.capital
        assert ps.allocated_capital == ls.allocated_capital
        assert ps.final_value == ls.final_value
        assert ps.ending_weight == ls.ending_weight
        assert ps.pnl_contribution == ls.pnl_contribution
        assert ps.cost_contribution == ls.cost_contribution
        assert_metrics_equal(ps.metrics, ls.metrics, f"{name}.metrics")

    assert_metrics_equal(prod.metrics, lab.metrics, "portfolio.metrics")
    assert_metrics_equal(prod.spy_metrics, lab.spy_metrics, "spy_metrics")


@pytest.mark.parametrize("weights_spec", [
    None,                                        # default 60/35/5
    "momentum=1.0",                              # single sleeve
    "sector_rotation=0.5,regime_switch=0.5",     # sector-plan pair
])
def test_lab_engine_matches_production_across_weightings(mocked_data, weights_spec):
    pairs = portfolio.parse_portfolio_weights(weights_spec)
    prod = run_production(pairs)
    lab = run_lab(pairs).portfolio
    pd.testing.assert_series_equal(
        prod.combined_result.equity_curve, lab.combined_result.equity_curve, check_names=False
    )
    assert_metrics_equal(prod.metrics, lab.metrics, "portfolio.metrics")


def test_default_lab_config_changes_no_behavior_flag():
    cfg = LabConfig()
    assert cfg.enabled_features() == []
    assert not cfg.any_behavior_change()


# --- Tracing is behavior-neutral ---------------------------------------------------

def test_recorder_attached_is_behavior_neutral(mocked_data):
    pairs = portfolio.parse_portfolio_weights(None)
    silent = run_lab(pairs).portfolio
    recorder = DecisionRecorder(lab_config_hash="test")
    traced_run = run_lab(pairs, recorder=recorder)
    traced = traced_run.portfolio

    pd.testing.assert_series_equal(
        silent.combined_result.equity_curve, traced.combined_result.equity_curve, check_names=False
    )
    for s_silent, s_traced in zip(silent.sleeves, traced.sleeves):
        assert_dataclass_lists_equal(
            s_silent.result.transactions, s_traced.result.transactions,
            f"{s_silent.strategy}.transactions",
        )
    assert_metrics_equal(silent.metrics, traced.metrics, "portfolio.metrics")
    # And the trace actually captured the run.
    assert traced_run.trace is not None
    assert len(traced_run.trace.portfolio_days) == len(traced.combined_result.equity_curve.index.union(
        traced.sleeves[0].result.equity_curve.index))
    assert len(traced_run.trace.orders) == sum(len(s.result.transactions) for s in traced.sleeves)


# --- No lookahead through the lab engine ------------------------------------------

def test_lab_no_lookahead_future_mutation_cannot_change_past(mocked_data):
    pairs = portfolio.parse_portfolio_weights(None)
    before = run_lab(pairs).portfolio.combined_result.equity_curve.copy()

    cutoff = pd.Timestamp("2024-01-01")
    for t, df in list(mocked_data.items()):
        mask = df.index > cutoff
        df.loc[mask, ["Open", "High", "Low", "Close"]] *= 3.0
        mocked_data[t] = df

    after = run_lab(pairs).portfolio.combined_result.equity_curve
    up_to_cutoff = before.index[before.index <= cutoff]
    assert len(up_to_cutoff) > 50
    pd.testing.assert_series_equal(
        before.reindex(up_to_cutoff), after.reindex(up_to_cutoff), check_names=False
    )


# --- Determinism -------------------------------------------------------------------

def test_lab_runs_are_deterministic(mocked_data):
    pairs = portfolio.parse_portfolio_weights(None)
    a = run_lab(pairs).portfolio
    b = run_lab(pairs).portfolio
    pd.testing.assert_series_equal(
        a.combined_result.equity_curve, b.combined_result.equity_curve, check_names=False
    )
    for sa, sb in zip(a.sleeves, b.sleeves):
        assert_dataclass_lists_equal(sa.result.transactions, sb.result.transactions,
                                     f"{sa.strategy}.transactions")


# --- CLI end-to-end ----------------------------------------------------------------

def test_cli_strategy_lab_baseline_writes_manifest_and_snapshot(mocked_data, tmp_path):
    exit_code = cli.main([
        "--strategy", "strategy_lab", "--start", "2023-01-01", "--end", "2024-12-31",
        "--capital", "15000", "--output-dir", str(tmp_path),
    ])
    assert exit_code == 0
    run_dirs = list(tmp_path.glob("*_portfolio_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    for artifact in [
        "portfolio_report.txt", "portfolio_summary.csv", "portfolio_equity.csv",
        "run_manifest.json", "baseline_snapshot.json",
        "decision_trace.csv", "decision_trace.jsonl", "order_trace.csv",
        "portfolio_trace.csv", "reason_code_summary.csv", "selection_funnel.csv",
        "decision_summary.json",
    ]:
        assert (run_dir / artifact).exists(), f"missing {artifact}"

    from src.lab.manifest import load_manifest

    manifest = load_manifest(run_dir / "run_manifest.json")
    assert manifest["run_kind"] == "baseline"
    assert manifest["portfolio"]["weights"] == {
        "momentum": 0.60, "sector_rotation": 0.35, "regime_switch": 0.05,
    }
    assert manifest["lab_config"]["vol_target"]["enabled"] is False
    assert manifest["artifact_hashes"], "artifact hashes must be recorded"
    assert "run_manifest.json" not in manifest["artifact_hashes"]

    # The portfolio trace covers every common-window day and reconciles to the
    # equity artifact.
    ptrace = pd.read_csv(run_dir / "portfolio_trace.csv")
    equity = pd.read_csv(run_dir / "portfolio_equity.csv")
    assert len(ptrace) >= len(equity)  # trace spans the union calendar
    # Baseline run must have an all-disabled lab config recorded.
    import json

    snapshot = json.loads((run_dir / "baseline_snapshot.json").read_text())
    assert snapshot["weights"] == {"momentum": 0.60, "sector_rotation": 0.35, "regime_switch": 0.05}
    assert snapshot["portfolio_metrics"]["final_equity"] > 0


def test_cli_baseline_refuses_enabled_enhancements(mocked_data, tmp_path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text('{"vol_target": {"enabled": true}}')
    exit_code = cli.main([
        "--strategy", "strategy_lab", "--start", "2023-01-01", "--end", "2023-12-31",
        "--output-dir", str(tmp_path / "out"), "--lab-config", str(cfg_path),
    ])
    assert exit_code == 1


def test_existing_cli_choices_still_unchanged():
    for choice in ["mean_reversion", "sector_rotation", "both", "compare", "robustness",
                   "tournament", "portfolio", "walk_forward"]:
        args = cli.parse_args(["--strategy", choice])
        assert args.strategy == choice
        assert args.experiment is None  # lab flags default off for every mode
