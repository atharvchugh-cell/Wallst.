import numpy as np
import pandas as pd
import pytest

from src.engine import Trade
from src.metrics import max_drawdown, cagr, win_rate, sharpe_ratio, is_short_period


def test_max_drawdown_known_path():
    equity = pd.Series([100.0, 150.0, 75.0, 140.0], index=pd.bdate_range("2024-01-01", periods=4))
    assert max_drawdown(equity) == pytest.approx(75.0 / 150.0 - 1.0)  # -0.5


def test_cagr_known_growth_over_one_year():
    # cagr() truncates the elapsed span to whole days via Timedelta.days, so
    # use an exact 365-day span (not a fractional 365.25) for a precise check.
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01") + pd.Timedelta(days=365)])
    equity = pd.Series([100.0, 200.0], index=idx)
    expected = 2.0 ** (365.25 / 365) - 1.0
    assert cagr(equity) == pytest.approx(expected, abs=1e-6)


def test_cagr_flat_equity_is_zero():
    equity = pd.Series([100.0, 100.0, 100.0], index=pd.bdate_range("2024-01-01", periods=3))
    assert cagr(equity) == pytest.approx(0.0)


def test_win_rate_only_counts_full_exit_trades():
    trades = [
        Trade("s", "A", "full_exit", pd.Timestamp("2024-01-01"), 1, 10, 5, 5.0, 1.0, "r", 1),
        Trade("s", "A", "full_exit", pd.Timestamp("2024-01-02"), 1, 10, 5, -5.0, -1.0, "r", 1),
        Trade("s", "A", "full_exit", pd.Timestamp("2024-01-03"), 1, 10, 5, 3.0, 0.6, "r", 1),
        Trade("s", "A", "partial_sell", pd.Timestamp("2024-01-04"), 1, 10, 5, 100.0, 20.0, "r", 1),
    ]
    assert win_rate(trades) == pytest.approx(2 / 3)  # partial_sell excluded


def test_win_rate_no_trades_is_zero():
    assert win_rate([]) == 0.0


def test_sharpe_zero_std_returns_nan_not_zero():
    equity = pd.Series([100.0] * 10, index=pd.bdate_range("2024-01-01", periods=10))  # zero daily return std
    result = sharpe_ratio(equity)
    assert pd.isna(result)


def test_sharpe_positive_series_finite():
    equity = pd.Series(np.linspace(100, 200, 50) + np.sin(np.arange(50)), index=pd.bdate_range("2024-01-01", periods=50))
    result = sharpe_ratio(equity)
    assert np.isfinite(result)


def test_is_short_period_warning_threshold():
    assert is_short_period(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-30")) is True  # 29 days < 90
    assert is_short_period(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")) is False  # 365 days
