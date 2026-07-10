"""Tests for --strategy walk_forward: out-of-sample validation of the weighted
portfolio across (train, test) folds.

The hard properties proven here:
  - a fold's training window always ends before its test window begins (no
    train/test overlap),
  - future test data cannot change an earlier fold's selection or results
    (no lookahead across folds),
  - the stitched equity curve contains only test-period dates,
  - capital carries forward correctly (folds compound; stitched final == last
    fold final),
  - transaction costs are included and aggregated,
  - fixed-parameter mode performs NO parameter selection,
  - and existing CLI modes/defaults are undisturbed.
"""

import numpy as np
import pandas as pd
import pytest

from src import cli, portfolio, tournament, walk_forward

FULL_DATES = pd.bdate_range("1990-01-01", "2024-12-31")


def shared_trending_df(seed=42, drift=0.02):
    rng = np.random.default_rng(seed)
    base = 100.0 + np.cumsum(rng.normal(drift, 0.5, size=len(FULL_DATES)))
    base = np.clip(base, 10.0, None)
    return pd.DataFrame(
        {"Open": base, "High": base, "Low": base, "Close": base, "Volume": 1000}, index=FULL_DATES
    )


@pytest.fixture
def mocked_data(monkeypatch):
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
    monkeypatch.setattr(portfolio.data, "get_benchmark_data", fake_get_benchmark_data)
    monkeypatch.setattr(walk_forward.data, "get_benchmark_data", fake_get_benchmark_data)
    return frames


def _run(pairs, start, end, capital=15000.0, train=2, test=1, step=1, expanding=True, optimize=False):
    return walk_forward.run_walk_forward(
        pairs, capital, start, end, cost_bps=5.0, fractional_shares=True, refresh_cache=False,
        output_dir="output", train_years=train, test_years=test, step_years=step,
        expanding=expanding, optimize=optimize,
    )


DEFAULT_PAIRS = [("momentum", 0.60), ("sector_rotation", 0.35), ("regime_switch", 0.05)]


# --- Fold generation ------------------------------------------------------------

def test_generate_folds_train_never_overlaps_test():
    folds = walk_forward.generate_folds("2015-01-01", "2024-12-31", 3, 1, 1, expanding=True)
    assert len(folds) == 7  # test years 2018..2024
    for train_start, train_end, test_start, test_end in folds:
        assert train_end < test_start                 # no overlap
        assert train_start <= train_end
        assert test_start <= test_end


def test_generate_folds_expanding_vs_rolling():
    exp = walk_forward.generate_folds("2015-01-01", "2020-12-31", 3, 1, 1, expanding=True)
    roll = walk_forward.generate_folds("2015-01-01", "2020-12-31", 3, 1, 1, expanding=False)
    # Expanding: train_start is always the global start.
    assert all(ts == pd.Timestamp("2015-01-01") for ts, _te, _s, _e in exp)
    # Rolling: train window is a fixed 3 years wide.
    for train_start, train_end, _s, _e in roll:
        assert (train_end - train_start).days > 3 * 365 - 5  # ~3 years wide


def test_generate_folds_drops_incomplete_trailing_test():
    # 2015->2018-06 can't fit a full 2018 test year after 3y training.
    folds = walk_forward.generate_folds("2015-01-01", "2018-06-30", 3, 1, 1)
    assert folds == []  # no complete test year -> no folds


def test_run_walk_forward_raises_when_span_too_short(mocked_data):
    with pytest.raises(walk_forward.WalkForwardError, match="No complete walk-forward folds"):
        _run(DEFAULT_PAIRS, "2020-01-01", "2021-06-30", train=2, test=1)


# --- Core invariants ------------------------------------------------------------

def test_training_dates_never_overlap_test_dates_in_a_real_run(mocked_data):
    wf = _run(DEFAULT_PAIRS, "2015-01-01", "2019-12-31", train=2)
    assert len(wf.folds) == 3
    for f in wf.folds:
        assert f.train_end < f.test_start


def test_capital_carries_forward_correctly(mocked_data):
    wf = _run(DEFAULT_PAIRS, "2015-01-01", "2019-12-31", capital=15000.0, train=2)
    assert wf.folds[0].start_capital == pytest.approx(15000.0)
    for prev, nxt in zip(wf.folds, wf.folds[1:]):
        # Each fold starts with the prior fold's ending equity -- compounding.
        assert nxt.start_capital == pytest.approx(prev.final_equity)
    # Stitched final equity == last fold's final equity.
    assert float(wf.stitched_equity.iloc[-1]) == pytest.approx(wf.folds[-1].final_equity)
    # And the aggregate total return is exactly final/initial - 1 (telescoped compounding).
    assert wf.aggregate["stitched_total_return"] == pytest.approx(
        wf.folds[-1].final_equity / 15000.0 - 1.0
    )


def test_stitched_equity_uses_test_periods_only(mocked_data):
    wf = _run(DEFAULT_PAIRS, "2015-01-01", "2019-12-31", train=2)
    for d in wf.stitched_equity.index:
        assert any(f.test_start <= d <= f.test_end for f in wf.folds), f"{d} not in any test window"
    # No stitched date lands in the training-only region before the first test.
    first_test_start = wf.folds[0].test_start
    assert (wf.stitched_equity.index >= first_test_start).all()
    # Stitched dates are strictly increasing (folds concatenated in order, no dupes).
    assert wf.stitched_equity.index.is_monotonic_increasing
    assert not wf.stitched_equity.index.duplicated().any()


def test_transaction_costs_are_included_and_aggregated(mocked_data):
    wf = _run(DEFAULT_PAIRS, "2015-01-01", "2019-12-31", train=2)
    for f in wf.folds:
        assert f.transaction_costs is not None and f.transaction_costs > 0
    assert wf.aggregate["total_transaction_costs"] == pytest.approx(
        sum(f.transaction_costs for f in wf.folds)
    )


def test_all_required_fold_and_aggregate_fields_present(mocked_data):
    wf = _run(DEFAULT_PAIRS, "2015-01-01", "2019-12-31", train=2)
    f = wf.folds[0]
    for val in [f.train_start, f.train_end, f.test_start, f.test_end, f.test_return,
                f.spy_return, f.excess_return, f.max_drawdown, f.transaction_costs]:
        assert val is not None
    for key in ["stitched_total_return", "stitched_cagr", "stitched_max_drawdown",
                "pct_folds_beating_spy", "pct_folds_profitable", "num_folds"]:
        assert key in wf.aggregate
    assert 0.0 <= wf.aggregate["pct_folds_beating_spy"] <= 1.0
    assert 0.0 <= wf.aggregate["pct_folds_profitable"] <= 1.0


# --- Fixed mode performs no selection -------------------------------------------

def test_fixed_mode_performs_no_parameter_selection(mocked_data, monkeypatch):
    """In fixed mode the variant-selection routine must never be called, and
    every fold's selected variant is the sentinel 'fixed'."""
    def _boom(*a, **k):
        raise AssertionError("_select_variants_for_fold must NOT run in fixed mode")

    monkeypatch.setattr(walk_forward, "_select_variants_for_fold", _boom)
    wf = _run(DEFAULT_PAIRS, "2015-01-01", "2019-12-31", train=2, optimize=False)
    for f in wf.folds:
        assert set(f.selected_variants.values()) == {"fixed"}


def test_optimize_mode_selects_and_freezes_a_variant_per_sleeve(mocked_data):
    # Two sleeves keeps the per-variant training runs bounded/fast.
    wf = _run([("momentum", 0.6), ("sector_rotation", 0.4)], "2016-01-01", "2019-12-31",
              train=2, optimize=True)
    assert len(wf.folds) >= 1
    for f in wf.folds:
        # Each sleeve got a concrete variant label (never the fixed-mode sentinel).
        assert set(f.selected_variants) == {"momentum", "sector_rotation"}
        assert "fixed" not in f.selected_variants.values()


# --- No lookahead across folds --------------------------------------------------

def test_future_test_data_cannot_change_an_earlier_folds_results(mocked_data):
    """Mutate price data for dates AFTER fold 0's test period and re-run; fold 0's
    results (which depend only on data through its own test end) must be
    identical."""
    wf_before = _run(DEFAULT_PAIRS, "2015-01-01", "2019-12-31", train=2)
    fold0 = wf_before.folds[0]
    fold0_equity = fold0.portfolio_result.combined_result.equity_curve.copy()
    fold0_return = fold0.test_return

    cutoff = fold0.test_end
    for t, df in list(mocked_data.items()):
        mask = df.index > cutoff
        df.loc[mask, ["Open", "High", "Low", "Close"]] *= 4.0
        mocked_data[t] = df

    wf_after = _run(DEFAULT_PAIRS, "2015-01-01", "2019-12-31", train=2)
    fold0_after = wf_after.folds[0]
    pd.testing.assert_series_equal(
        fold0_after.portfolio_result.combined_result.equity_curve, fold0_equity, check_names=False
    )
    assert fold0_after.test_return == pytest.approx(fold0_return)


def test_future_test_data_cannot_change_an_earlier_folds_selection(mocked_data):
    """Same as above but for optimize mode: fold 0's variant selection is made
    on its training window (which ends before its test period), so it cannot be
    swayed by any later data."""
    pairs = [("momentum", 0.6), ("sector_rotation", 0.4)]
    wf_before = _run(pairs, "2016-01-01", "2019-12-31", train=2, optimize=True)
    fold0_selection = dict(wf_before.folds[0].selected_variants)

    cutoff = wf_before.folds[0].test_end
    for t, df in list(mocked_data.items()):
        mask = df.index > cutoff
        df.loc[mask, ["Open", "High", "Low", "Close"]] *= 4.0
        mocked_data[t] = df

    wf_after = _run(pairs, "2016-01-01", "2019-12-31", train=2, optimize=True)
    assert wf_after.folds[0].selected_variants == fold0_selection


# --- CLI end-to-end -------------------------------------------------------------

def test_cli_walk_forward_writes_all_artifacts(mocked_data, tmp_path):
    exit_code = cli.main([
        "--strategy", "walk_forward", "--start", "2015-01-01", "--end", "2019-12-31",
        "--capital", "15000", "--output-dir", str(tmp_path), "--walk-forward-train-years", "2",
    ])
    assert exit_code == 0
    dirs = list(tmp_path.glob("*_walk_forward_*"))
    assert len(dirs) == 1
    run_dir = dirs[0]
    for artifact in ["walk_forward_report.txt", "walk_forward_folds.csv",
                     "walk_forward_equity.csv", "walk_forward.json"]:
        assert (run_dir / artifact).exists(), f"missing {artifact}"

    folds = pd.read_csv(run_dir / "walk_forward_folds.csv")
    for col in ["train_start", "train_end", "test_start", "test_end", "portfolio_test_return",
                "spy_test_return", "excess_return", "max_drawdown", "sharpe_ratio",
                "transaction_costs", "selected_variants"]:
        assert col in folds.columns
    # train_end strictly before test_start in every row.
    assert (pd.to_datetime(folds["train_end"]) < pd.to_datetime(folds["test_start"])).all()

    text = (run_dir / "walk_forward_report.txt").read_text()
    assert "survivorship" in text.lower()
    assert "does NOT fix survivorship bias" in text
    assert "Stitched total return" in text


def test_cli_walk_forward_rejects_bad_weights(mocked_data, tmp_path):
    exit_code = cli.main([
        "--strategy", "walk_forward", "--start", "2015-01-01", "--end", "2019-12-31",
        "--output-dir", str(tmp_path), "--portfolio-weights", "momentum=0.6,sector_rotation=0.5",
    ])
    assert exit_code == 1
    assert list(tmp_path.glob("*_walk_forward_*")) == []


# --- Existing modes/defaults undisturbed ---------------------------------------

def test_existing_strategy_choices_and_defaults_unchanged():
    for choice in ["mean_reversion", "sector_rotation", "both", "compare", "robustness",
                   "tournament", "portfolio"]:
        args = cli.parse_args(["--strategy", choice])
        assert args.strategy == choice
        # New walk-forward flags default without disturbing other modes.
        assert args.walk_forward_optimize is False
        assert args.walk_forward_window == "expanding"
    args = cli.parse_args(["--strategy", "walk_forward"])
    assert args.strategy == "walk_forward"
    assert args.walk_forward_train_years == 3
    assert args.walk_forward_test_years == 1
    assert args.walk_forward_step_years == 1
