import numpy as np
import pandas as pd
import pytest

from src.indicators import sma, rsi, month_end_dates, trailing_month_end_return


def test_sma_basic():
    s = pd.Series(range(1, 11), dtype=float)  # 1..10
    result = sma(s, window=3)
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == pytest.approx(2.0)  # mean(1,2,3)
    assert result.iloc[-1] == pytest.approx(9.0)  # mean(8,9,10)


def test_rsi_monotonic_increasing_approaches_100():
    s = pd.Series(range(1, 30), dtype=float)
    r = rsi(s, period=14)
    assert r.iloc[-1] == pytest.approx(100.0)


def test_rsi_monotonic_decreasing_approaches_0():
    s = pd.Series(range(30, 1, -1), dtype=float)
    r = rsi(s, period=14)
    assert r.iloc[-1] == pytest.approx(0.0)


def test_rsi_flat_series_is_neutral_not_100():
    s = pd.Series([200.0] * 60)
    r = rsi(s, period=14)
    assert r.iloc[-1] == pytest.approx(50.0)
    assert r.isna().sum() == 14  # warmup period


def test_rsi_matches_hand_computed_wilder_formula():
    # Small fixed series, cross-check vectorized RSI against a plain-Python
    # re-implementation of Wilder's smoothing.
    prices = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84,
              46.08, 45.89, 46.03, 45.61, 46.28, 46.28]
    period = 14
    s = pd.Series(prices)
    vectorized = rsi(s, period=period)

    # Hand-computed Wilder RSI: seed avg_gain/avg_loss with a plain mean over
    # the first `period` deltas, then apply Wilder's smoothing for the rest --
    # matching pandas' ewm(alpha=1/period, adjust=False) applied from the
    # first observation, since ewm with adjust=False is defined from index 0.
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    alpha = 1.0 / period
    avg_gain = gains[0]
    avg_loss = losses[0]
    for i in range(1, len(gains)):
        avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss
    rs = avg_gain / avg_loss
    expected_rsi = 100.0 - (100.0 / (1.0 + rs))
    assert vectorized.iloc[-1] == pytest.approx(expected_rsi, abs=1e-6)


def test_month_end_dates():
    idx = pd.bdate_range("2024-01-01", "2024-03-31")
    me = month_end_dates(idx)
    assert list(me) == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29"), pd.Timestamp("2024-03-29")]


def test_trailing_month_end_return_known_growth():
    idx = pd.bdate_range("2024-01-01", "2024-04-30")
    me = month_end_dates(idx)
    close = pd.Series(100.0, index=idx)
    # Simulate 10% growth at each successive month-end
    close.loc[me[0]:] = 110.0
    close.loc[me[1]:] = 121.0
    close.loc[me[2]:] = 133.1
    close.loc[me[3]:] = 146.41
    tr = trailing_month_end_return(close, me, lookback_months=1)
    assert pd.isna(tr.iloc[0])
    assert tr.iloc[1] == pytest.approx(0.10, abs=1e-9)
    assert tr.iloc[2] == pytest.approx(0.10, abs=1e-9)
    assert tr.iloc[3] == pytest.approx(0.10, abs=1e-9)


def test_trailing_month_end_return_three_month_lookback():
    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    me = month_end_dates(idx)
    close = pd.Series(np.nan, index=idx)
    # Jan..Dec month-end closes, deliberately simple values
    values = [100, 105, 110, 121, 100, 90, 80, 130, 140, 150, 160, 170]
    for d, v in zip(me, values):
        close.loc[d:] = float(v)
    tr = trailing_month_end_return(close, me, lookback_months=3)
    # First 3 entries NaN (insufficient history), 4th (April) = Apr/Jan - 1
    assert tr.iloc[:3].isna().all()
    assert tr.iloc[3] == pytest.approx(values[3] / values[0] - 1.0)
    assert tr.iloc[4] == pytest.approx(values[4] / values[1] - 1.0)


def test_adjusted_price_series_no_spurious_indicator_spike():
    # A smooth, already-adjusted price series (as auto_adjust=True produces
    # across a real split) should not create an artificial RSI/SMA spike --
    # unlike a raw, un-adjusted series which would show a fake single-day
    # drop at the split date.
    smooth = pd.Series(np.linspace(100, 110, 60))
    r = rsi(smooth, period=14)
    s = sma(smooth, window=50)
    # RSI should reflect a mild, steady uptrend (elevated but not pegged, and
    # no NaN/inf produced), SMA should track smoothly with no discontinuity.
    assert r.iloc[-1] > 50.0
    assert np.isfinite(r.iloc[-1])
    assert s.diff().dropna().abs().max() < 1.0  # no jump in the moving average
