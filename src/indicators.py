"""Indicator calculations. All are computed once, vectorized, over a ticker's
full history -- rolling/EWM formulas are causal (a value at date d depends
only on data <= d), so precomputing over the whole series and later
restricting *visibility* at query time (see market_view.py) is both efficient
and still safe against lookahead.
"""

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI: an exponential moving average of gains/losses with
    alpha = 1/period (not a plain rolling mean). This is the textbook
    convention and affects expected values in tests."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    result = 100.0 - (100.0 / (1.0 + rs))
    result[(avg_loss == 0) & (avg_gain > 0)] = 100.0  # all gains, no losses -> RSI 100
    result[(avg_loss == 0) & (avg_gain == 0)] = 50.0  # perfectly flat -> RSI neutral, not 100
    return result


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last trading date of each (year, month) present in `index`, sorted ascending."""
    if len(index) == 0:
        return pd.DatetimeIndex([])
    s = pd.Series(index, index=index)
    last_per_month = s.groupby([index.year, index.month]).max()
    return pd.DatetimeIndex(sorted(last_per_month.values))


def trailing_month_end_return(
    close: pd.Series, month_ends: pd.DatetimeIndex, lookback_months: int
) -> pd.Series:
    """Month-end close vs. month-end close `lookback_months` *completed
    calendar months* prior (NOT a fixed trading-day lag). Returned as a
    Series indexed by `month_ends`; the first `lookback_months` entries are
    NaN (insufficient prior history)."""
    values = close.reindex(month_ends)
    return values / values.shift(lookback_months) - 1.0
