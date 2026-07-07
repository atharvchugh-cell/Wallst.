import numpy as np
import pandas as pd
import pytest

from src.engine import Trade
from src.metrics import (
    max_drawdown, max_drawdown_duration_days, cagr, total_return, win_rate, sharpe_ratio,
    sortino_ratio, calmar_ratio, is_short_period,
)


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


def test_win_rate_net_by_default_flips_a_barely_profitable_gross_trade():
    # Gross P&L is positive (a "win") but costs eat the whole thing and then
    # some -- net win rate must count this as a loss, gross win rate as a win.
    trades = [
        Trade(
            "s", "A", "full_exit", pd.Timestamp("2024-01-01"), 1, 10, 5,
            realized_pnl=1.0, realized_return_pct=0.2, reason="r", holding_days=1,
            realized_pnl_net=-0.5, realized_return_pct_net=-0.1,
        ),
    ]
    assert win_rate(trades, net=True) == 0.0
    assert win_rate(trades, net=False) == 1.0


def test_total_return_uses_start_capital_when_provided():
    # First recorded equity point already reflects day-0 transaction costs
    # (995 instead of the true 1000 starting cash). Passing start_capital
    # should measure growth against the true starting cash, not that
    # already-discounted first point.
    equity = pd.Series([995.0, 1100.0], index=pd.bdate_range("2024-01-01", periods=2))
    assert total_return(equity) == pytest.approx(1100.0 / 995.0 - 1.0)
    assert total_return(equity, start_capital=1000.0) == pytest.approx(1100.0 / 1000.0 - 1.0)


def test_cagr_uses_start_capital_when_provided():
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01") + pd.Timedelta(days=365)])
    equity = pd.Series([995.0, 1990.0], index=idx)
    default_cagr = cagr(equity)
    capital_basis_cagr = cagr(equity, start_capital=1000.0)
    assert capital_basis_cagr != pytest.approx(default_cagr)
    assert capital_basis_cagr == pytest.approx((1990.0 / 1000.0) ** (365.25 / 365) - 1.0, abs=1e-6)


def test_max_drawdown_duration_known_path():
    # Peak at day0 (100), underwater through day2, new peak (recovery) day3.
    equity = pd.Series(
        [100.0, 80.0, 90.0, 110.0],
        index=[pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08")],
    )
    # Underwater from day0 (the peak) through day2 (last day below peak) -> day0 to day5 = 4 days.
    assert max_drawdown_duration_days(equity) == 4


def test_max_drawdown_duration_zero_when_never_underwater():
    equity = pd.Series([100.0, 110.0, 120.0], index=pd.bdate_range("2024-01-01", periods=3))
    assert max_drawdown_duration_days(equity) == 0


def test_sortino_nan_when_no_downside_days():
    equity = pd.Series([100.0, 101.0, 102.0, 103.0], index=pd.bdate_range("2024-01-01", periods=4))
    assert pd.isna(sortino_ratio(equity))


def test_calmar_ratio_nan_when_no_drawdown():
    equity = pd.Series([100.0, 101.0, 102.0], index=pd.bdate_range("2024-01-01", periods=3))
    assert pd.isna(calmar_ratio(equity))


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
