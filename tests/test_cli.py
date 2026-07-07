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
