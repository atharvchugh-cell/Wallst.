"""Restricted data accessor handed to strategies. This is what turns
"don't read tomorrow's price" from a code-review convention into something
structurally impossible to violate by accident: any read dated after the
current walk date raises LookaheadError.
"""

from __future__ import annotations

import pandas as pd


class LookaheadError(Exception):
    """Raised when strategy code requests price/indicator data dated after
    the current walk date."""


class MarketDataView:
    def __init__(
        self,
        frames: dict[str, pd.DataFrame],
        as_of: pd.Timestamp,
        calendar: pd.DatetimeIndex,
    ):
        self._frames = frames
        self._as_of = pd.Timestamp(as_of)
        self._calendar = calendar

    @property
    def as_of(self) -> pd.Timestamp:
        return self._as_of

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return self._calendar

    @property
    def tickers(self) -> list[str]:
        return list(self._frames.keys())

    def _check(self, ticker: str, date: pd.Timestamp) -> None:
        if ticker not in self._frames:
            raise KeyError(f"Unknown ticker: {ticker}")
        if date > self._as_of:
            raise LookaheadError(
                f"{ticker}: requested data for {date.date()} while walk is at {self._as_of.date()}"
            )

    def close(self, ticker: str, date: pd.Timestamp | None = None) -> float:
        d = pd.Timestamp(date) if date is not None else self._as_of
        self._check(ticker, d)
        return float(self._frames[ticker].loc[d, "Close"])

    def indicator(self, ticker: str, name: str, date: pd.Timestamp | None = None) -> float:
        d = pd.Timestamp(date) if date is not None else self._as_of
        self._check(ticker, d)
        value = self._frames[ticker].loc[d, name]
        return float(value) if pd.notna(value) else float("nan")

    def history(self, ticker: str, lookback: int | None = None, column: str = "Close") -> pd.Series:
        if ticker not in self._frames:
            raise KeyError(f"Unknown ticker: {ticker}")
        series = self._frames[ticker].loc[: self._as_of, column]
        return series.tail(lookback) if lookback is not None else series

    def has_data(self, ticker: str, date: pd.Timestamp | None = None) -> bool:
        d = pd.Timestamp(date) if date is not None else self._as_of
        if ticker not in self._frames or d > self._as_of:
            return False
        return d in self._frames[ticker].index and pd.notna(self._frames[ticker].loc[d, "Close"])

    # --- Calendar navigation -------------------------------------------------
    # Safe to expose beyond as_of: which dates markets are open is public
    # knowledge in advance and carries no price information, unlike the
    # methods above.
    def next_trading_day(self, date: pd.Timestamp | None = None) -> pd.Timestamp | None:
        d = pd.Timestamp(date) if date is not None else self._as_of
        pos = self._calendar.searchsorted(d, side="right")
        if pos >= len(self._calendar):
            return None
        return self._calendar[pos]

    def is_month_end(self, date: pd.Timestamp | None = None) -> bool:
        """True if `date` is the last trading day of its calendar month.
        Returns False for the last date in the whole dataset even if it might
        be a true month-end in reality -- we can't confirm without a
        following trading day in the fetched range, so this strategy simply
        won't rebalance on the literal final day of a --end-truncated window."""
        d = pd.Timestamp(date) if date is not None else self._as_of
        pos = self._calendar.get_loc(d)
        if pos == len(self._calendar) - 1:
            return False
        next_d = self._calendar[pos + 1]
        return (d.year, d.month) != (next_d.year, next_d.month)
