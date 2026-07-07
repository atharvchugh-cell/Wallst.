import numpy as np
import pandas as pd
import pytest

from src import cli, config, data


def make_price_df(dates, close=100.0):
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1000}, index=dates
    )


def test_min_universe_check_hard_fails_on_degraded_universe(monkeypatch):
    dates = pd.bdate_range("2023-06-01", periods=260)  # covers warmup + the 2024 window below

    # Simulate a data outage that only leaves 3 of the ~25 configured tickers
    # usable -- well under the 80% minimum -- while the benchmark (SPY) itself
    # fetches fine, so the run should hard-fail via the min-universe check
    # rather than silently producing a near-empty backtest.
    surviving = {"AAPL": make_price_df(dates), "MSFT": make_price_df(dates), "AMZN": make_price_df(dates)}

    def fake_get_price_data(tickers, start, end, warmup_calendar_days, hard_fail_on_missing, **kwargs):
        dropped = [(t, "simulated fetch failure") for t in tickers if t not in surviving]
        return dict(surviving), dropped

    def fake_get_benchmark_data(start, end, **kwargs):
        return make_price_df(dates)

    monkeypatch.setattr(cli.data, "get_price_data", fake_get_price_data)
    monkeypatch.setattr(cli.data, "get_benchmark_data", fake_get_benchmark_data)

    with pytest.raises(data.FetchError, match="Only 3/"):
        cli.run_mean_reversion_sleeve(
            "2024-01-01", "2024-12-31", capital=7500.0, cost_bps=5.0,
            fractional_shares=True, refresh_cache=False, output_dir="output",
        )


def test_compare_strategy_produces_comparison_artifacts(tmp_path, monkeypatch):
    # One shared, gently-trending synthetic price series used for every
    # ticker mean_reversion/sector_rotation/SPY ask for -- covers 1990
    # onward so sector rotation's effective-start clipping never kicks in
    # and mean reversion's warmup is always satisfied.
    full_dates = pd.bdate_range("1990-01-01", "2024-12-31")
    rng = np.random.default_rng(42)
    base = 100.0 + np.cumsum(rng.normal(0.01, 0.5, size=len(full_dates)))
    base = np.clip(base, 10.0, None)
    shared_df = pd.DataFrame(
        {"Open": base, "High": base, "Low": base, "Close": base, "Volume": 1000}, index=full_dates
    )

    def fake_get_price_data(tickers, start, end, warmup_calendar_days, hard_fail_on_missing, **kwargs):
        return {t: shared_df.copy() for t in tickers}, []

    def fake_get_benchmark_data(start, end, **kwargs):
        return shared_df.copy()

    monkeypatch.setattr(cli.data, "get_price_data", fake_get_price_data)
    monkeypatch.setattr(cli.data, "get_benchmark_data", fake_get_benchmark_data)

    exit_code = cli.main([
        "--strategy", "compare", "--start", "2022-01-01", "--end", "2024-12-31",
        "--capital", "2000", "--output-dir", str(tmp_path),
    ])
    assert exit_code == 0

    compare_dirs = list(tmp_path.glob("*_comparison_*"))
    assert len(compare_dirs) == 1
    run_dir = compare_dirs[0]
    assert (run_dir / "comparison.csv").exists()
    assert (run_dir / "comparison.txt").exists()
    assert (run_dir / "annual_returns.csv").exists()
    assert (run_dir / "monthly_returns.csv").exists()
    assert (run_dir / "comparison.json").exists()

    comparison_df = pd.read_csv(run_dir / "comparison.csv")
    assert list(comparison_df.columns) == ["metric", "mean_reversion", "sector_rotation", "both", "SPY"]

    # --compare-years defaults to every calendar year spanned by --start/--end.
    annual_df = pd.read_csv(run_dir / "annual_returns.csv")
    assert list(annual_df["year"]) == [2022, 2023, 2024]

    # The three individual-sleeve report dirs (mean_reversion, sector_rotation,
    # both) are still produced too -- compare is a superset, not a replacement.
    assert len(list(tmp_path.glob("*_mean_reversion_*"))) == 1
    assert len(list(tmp_path.glob("*_sector_rotation_*"))) == 1
    assert len(list(tmp_path.glob("*_both_*"))) == 1


def test_compare_strategy_respects_explicit_compare_years(tmp_path, monkeypatch):
    full_dates = pd.bdate_range("1990-01-01", "2024-12-31")
    rng = np.random.default_rng(7)
    base = 100.0 + np.cumsum(rng.normal(0.01, 0.5, size=len(full_dates)))
    base = np.clip(base, 10.0, None)
    shared_df = pd.DataFrame(
        {"Open": base, "High": base, "Low": base, "Close": base, "Volume": 1000}, index=full_dates
    )

    def fake_get_price_data(tickers, start, end, warmup_calendar_days, hard_fail_on_missing, **kwargs):
        return {t: shared_df.copy() for t in tickers}, []

    def fake_get_benchmark_data(start, end, **kwargs):
        return shared_df.copy()

    monkeypatch.setattr(cli.data, "get_price_data", fake_get_price_data)
    monkeypatch.setattr(cli.data, "get_benchmark_data", fake_get_benchmark_data)

    exit_code = cli.main([
        "--strategy", "compare", "--start", "2022-01-01", "--end", "2024-12-31",
        "--capital", "2000", "--output-dir", str(tmp_path), "--compare-years", "2023",
    ])
    assert exit_code == 0
    run_dir = list(tmp_path.glob("*_comparison_*"))[0]
    annual_df = pd.read_csv(run_dir / "annual_returns.csv")
    assert list(annual_df["year"]) == [2023]
