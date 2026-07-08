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


def test_robustness_strategy_cli_produces_expected_artifacts(tmp_path, monkeypatch):
    full_dates = pd.bdate_range("1990-01-01", "2024-12-31")
    rng = np.random.default_rng(11)
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

    # Two small windows via --robustness-windows (not the 5 full default
    # windows) to keep this test fast -- the default-window LIST itself is
    # covered separately in tests/test_robustness.py.
    exit_code = cli.main([
        "--strategy", "robustness", "--start", "2022-01-01", "--end", "2023-12-31",
        "--capital", "2000", "--output-dir", str(tmp_path),
        "--robustness-windows", "2022-01-01:2022-12-31,2023-01-01:2023-12-31",
    ])
    assert exit_code == 0

    robustness_dirs = list(tmp_path.glob("*_robustness_*"))
    assert len(robustness_dirs) == 1
    run_dir = robustness_dirs[0]
    assert (run_dir / "robustness_summary.csv").exists()
    assert (run_dir / "robustness_summary.txt").exists()
    assert (run_dir / "robustness_rankings.csv").exists()
    assert (run_dir / "robustness_heatmap_data.csv").exists()
    assert (run_dir / "robustness_summary.json").exists()

    summary_df = pd.read_csv(run_dir / "robustness_summary.csv")
    assert set(summary_df["window"]) == {"2022-2022", "2023-2023"}
    assert set(summary_df["allocation"]) == {label for label, _, _ in cli.ALLOCATION_MIXES} | {"SPY"}

    # Individual sleeve reports are still written per window (2 windows x
    # {mean_reversion, sector_rotation} = 4 dirs), since robustness mode
    # reuses run_mean_reversion_sleeve/run_sector_rotation_sleeve unmodified
    # rather than duplicating their strategy-running logic.
    assert len(list(tmp_path.glob("*_mean_reversion_*"))) == 2
    assert len(list(tmp_path.glob("*_sector_rotation_*"))) == 2


def _shared_price_fakes(seed=1):
    full_dates = pd.bdate_range("1990-01-01", "2024-12-31")
    rng = np.random.default_rng(seed)
    base = 100.0 + np.cumsum(rng.normal(0.01, 0.5, size=len(full_dates)))
    base = np.clip(base, 10.0, None)
    shared_df = pd.DataFrame(
        {"Open": base, "High": base, "Low": base, "Close": base, "Volume": 1000}, index=full_dates
    )

    def fake_get_price_data(tickers, start, end, warmup_calendar_days, hard_fail_on_missing, **kwargs):
        return {t: shared_df.copy() for t in tickers}, []

    def fake_get_benchmark_data(start, end, **kwargs):
        return shared_df.copy()

    return fake_get_price_data, fake_get_benchmark_data


def test_universe_us_50b_threads_into_mean_reversion_sleeve(tmp_path, monkeypatch):
    # resolve_mean_reversion_universe is mocked so this test doesn't hit
    # Nasdaq Trader / yfinance for a live us_50b build -- the point here is
    # only to verify cli.py actually THREADS whatever it resolves through
    # into the mean-reversion sleeve, not to re-test src/universe.py's own
    # fetch/filter logic (covered in tests/test_universe.py).
    fake_tickers = ["ZZZ1", "ZZZ2", "ZZZ3", "ZZZ4", "ZZZ5"]
    fake_resolution = cli.universe_module.UniverseResolution(
        tickers=fake_tickers,
        info={
            "mode": "us_50b", "num_selected": len(fake_tickers), "num_candidates": 500,
            "num_dropped_lookup_failed": 12, "min_market_cap": 50.5e9, "max_market_cap": 3.1e12,
            "cache_file": "data_cache/universe_us_50b.csv", "snapshot_date": "2026-07-08T00:00:00+00:00",
        },
    )
    monkeypatch.setattr(cli.universe_module, "resolve_mean_reversion_universe", lambda **kwargs: fake_resolution)

    fake_get_price_data, fake_get_benchmark_data = _shared_price_fakes()
    monkeypatch.setattr(cli.data, "get_price_data", fake_get_price_data)
    monkeypatch.setattr(cli.data, "get_benchmark_data", fake_get_benchmark_data)

    exit_code = cli.main([
        "--strategy", "mean_reversion", "--universe", "us_50b", "--start", "2022-01-01",
        "--end", "2023-12-31", "--capital", "2000", "--output-dir", str(tmp_path),
    ])
    assert exit_code == 0

    run_dir = list(tmp_path.glob("*_mean_reversion_*"))[0]
    import json
    metrics = json.loads((run_dir / "metrics.json").read_text())
    # The sleeve ran with the mocked us_50b ticker list, NOT the config default.
    assert set(metrics["universe"]) == set(fake_tickers)
    assert metrics["universe_info"]["mode"] == "us_50b"
    assert metrics["universe_info"]["num_dropped_lookup_failed"] == 12

    txt = (run_dir / "report.txt").read_text()
    assert "Universe mode: us_50b" in txt
    assert "CURRENT SNAPSHOT" in txt


def test_universe_default_unchanged_when_no_universe_flag(tmp_path, monkeypatch):
    # Without --universe, mean_reversion must use exactly config.MEAN_REVERSION_UNIVERSE,
    # byte-for-byte -- this is the reproducibility guarantee the feature promises.
    fake_get_price_data, fake_get_benchmark_data = _shared_price_fakes(seed=2)
    monkeypatch.setattr(cli.data, "get_price_data", fake_get_price_data)
    monkeypatch.setattr(cli.data, "get_benchmark_data", fake_get_benchmark_data)

    exit_code = cli.main([
        "--strategy", "mean_reversion", "--start", "2022-01-01", "--end", "2023-12-31",
        "--capital", "2000", "--output-dir", str(tmp_path),
    ])
    assert exit_code == 0

    run_dir = list(tmp_path.glob("*_mean_reversion_*"))[0]
    import json
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert set(metrics["universe"]) == set(config.MEAN_REVERSION_UNIVERSE)
    assert metrics["universe_info"]["mode"] == "default"
    assert metrics["universe_info"]["cache_file"] is None

    txt = (run_dir / "report.txt").read_text()
    assert "Universe mode: default" in txt
    # The current-snapshot caveat is specific to non-default universes.
    assert "CURRENT SNAPSHOT" not in txt
