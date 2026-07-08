"""Sector rotation: rank the 11 SPDR sector ETFs by trailing 3-completed-
calendar-month total return, hold the top-K equal-weighted, rebalance
monthly. Every universe member emits a TargetEvent at each rebalance --
including top-K members whose weight fraction is unchanged, since price
drift since the last rebalance means their actual dollar weight has moved
and the point of a monthly rebalance is to correct that.
"""

from __future__ import annotations

import pandas as pd

from .. import config
from ..indicators import month_end_dates, trailing_month_end_return
from ..market_view import MarketDataView
from .base import Strategy, TargetEvent


class SectorRotationStrategy(Strategy):
    name = "sector_rotation"
    family = "momentum"

    def __init__(
        self,
        universe: list[str] | None = None,
        lookback_months: int = config.SECTOR_LOOKBACK_MONTHS,
        top_k: int = config.SECTOR_TOP_K,
    ):
        super().__init__()
        if lookback_months <= 0:
            raise ValueError(f"lookback_months must be > 0, got {lookback_months}")
        resolved_universe = list(universe) if universe is not None else list(config.SECTOR_ETFS)
        if top_k <= 0 or top_k > len(resolved_universe):
            raise ValueError(f"top_k must be in [1, {len(resolved_universe)}], got {top_k}")
        self.universe = resolved_universe
        self.lookback_months = lookback_months
        self.top_k = top_k
        self._last_rebalance_month: tuple[int, int] | None = None

    def reset(self) -> None:
        super().reset()
        self._last_rebalance_month = None

    def describe(self) -> dict:
        info = super().describe()
        info["params"] = {
            "lookback_months": self.lookback_months,
            "top_k": self.top_k,
        }
        info["assumptions"] = [
            "Always fully invested in the top-K sectors, even in a broad bear market "
            "(ranks 'least bad' sectors rather than de-risking).",
            f"top_k={self.top_k} was re-tuned mid-development (see config.py comments) -- "
            "treat single-window results as in-sample.",
            "Universe is the 11 fixed SPDR sector ETFs; earliest usable start is bounded "
            "by XLC's 2018 inception + the lookback.",
        ]
        return info

    def prepare(
        self, price_data: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex, start: pd.Timestamp
    ) -> dict[str, pd.DataFrame]:
        enriched: dict[str, pd.DataFrame] = {}
        for ticker in self.universe:
            if ticker not in price_data:
                raise ValueError(
                    f"Sector rotation requires all {len(self.universe)} ETFs; "
                    f"{ticker} was not available. This strategy hard-fails rather "
                    f"than silently ranking a reduced universe."
                )
            df = price_data[ticker].copy()
            m_ends = month_end_dates(df.index)
            trailing = trailing_month_end_return(df["Close"], m_ends, self.lookback_months)
            col = pd.Series(index=df.index, dtype=float)
            col.loc[m_ends] = trailing.values
            df["TrailingReturn_3M"] = col
            enriched[ticker] = df
        return enriched

    def _rebalance_events(
        self,
        signal_date: pd.Timestamp,
        market: MarketDataView,
        fill_date: pd.Timestamp | None,
    ) -> list[TargetEvent]:
        returns = {}
        for ticker in self.universe:
            val = market.indicator(ticker, "TrailingReturn_3M", signal_date)
            if pd.notna(val):
                returns[ticker] = val
        ranked = sorted(returns.items(), key=lambda kv: kv[1], reverse=True)
        top_k_tickers = {t for t, _ in ranked[: self.top_k]}
        events = []
        for ticker in self.universe:
            if ticker not in returns:
                continue
            weight = (1.0 / self.top_k) if ticker in top_k_tickers else 0.0
            close = market.close(ticker, signal_date)
            events.append(
                TargetEvent(
                    strategy=self.name, ticker=ticker, signal_date=signal_date, fill_date=fill_date,
                    target_weight=weight, sizing_price=close, reason="rebalance",
                )
            )
        return events

    def initial_events(self, market: MarketDataView, sleeve_equity: float) -> list[TargetEvent]:
        ref_ticker = self.universe[0]
        hist = market.history(ref_ticker, column="TrailingReturn_3M").dropna()
        if hist.empty:
            return []
        warmup_signal_date = hist.index[-1]
        events = self._rebalance_events(warmup_signal_date, market, fill_date=None)
        self._last_rebalance_month = (warmup_signal_date.year, warmup_signal_date.month)
        return events

    def on_day(self, day: pd.Timestamp, market: MarketDataView, sleeve_equity: float) -> list[TargetEvent]:
        if not market.is_month_end(day):
            return []
        ym = (day.year, day.month)
        if ym == self._last_rebalance_month:
            return []
        fill_date = market.next_trading_day(day)
        if fill_date is None:
            return []
        events = self._rebalance_events(day, market, fill_date=fill_date)
        self._last_rebalance_month = ym
        return events
