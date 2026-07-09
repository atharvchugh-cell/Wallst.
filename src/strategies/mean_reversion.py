"""Mean reversion: buy oversold large caps (RSI < 30), exit on mean
reversion (close > SMA-50 or RSI >= 50), a delayed close-to-close stop-loss,
or a max-holding-day timeout. Fixed-slot sizing: each open position gets
1/max_concurrent_positions of the sleeve.

NOTE ON SURVIVORSHIP BIAS: `universe` defaults to today's large-cap
survivors, not a point-in-time constituent list. This strategy's backtest
results are survivorship-biased research, not a general validation -- see
README.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .. import config
from ..indicators import rsi, sma
from ..market_view import MarketDataView
from .base import Strategy, TargetEvent


@dataclass
class _PositionState:
    status: str = "flat"  # flat | pending_entry | open | pending_exit
    pending_fill_date: pd.Timestamp | None = None
    entry_fill_price: float | None = None
    entry_fill_date: pd.Timestamp | None = None
    days_held: int = 0


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"
    family = "mean_reversion"

    def __init__(
        self,
        universe: list[str] | None = None,
        rsi_period: int = config.RSI_PERIOD,
        rsi_entry: float = config.RSI_ENTRY_THRESHOLD,
        rsi_exit: float = config.RSI_EXIT_THRESHOLD,
        sma_period: int = config.SMA_PERIOD,
        stop_loss_pct: float = config.STOP_LOSS_PCT,
        max_holding_days: int = config.MAX_HOLDING_DAYS,
        max_concurrent_positions: int = config.MAX_CONCURRENT_POSITIONS,
    ):
        super().__init__()
        if max_concurrent_positions <= 0:
            raise ValueError(f"max_concurrent_positions must be > 0, got {max_concurrent_positions}")
        self.universe = list(universe) if universe is not None else list(config.MEAN_REVERSION_UNIVERSE)
        self.rsi_period = rsi_period
        self.rsi_entry = rsi_entry
        self.rsi_exit = rsi_exit
        self.sma_period = sma_period
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        self.max_concurrent_positions = max_concurrent_positions
        self._state: dict[str, _PositionState] = {}

    def reset(self) -> None:
        super().reset()
        self._state = {t: _PositionState() for t in self.universe}

    def prepare(
        self, price_data: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex, start: pd.Timestamp
    ) -> dict[str, pd.DataFrame]:
        enriched: dict[str, pd.DataFrame] = {}
        kept_universe = []
        for ticker in self.universe:
            if ticker not in price_data:
                self.dropped_tickers.append((ticker, "no price data fetched"))
                continue
            df = price_data[ticker].copy()
            df["RSI_14"] = rsi(df["Close"], self.rsi_period)
            df["SMA_50"] = sma(df["Close"], self.sma_period)
            # A ticker must have valid indicators by the first walk day, or it
            # can never generate a signal within the requested window.
            first_walk_day = calendar[0] if len(calendar) else pd.Timestamp(start)
            if first_walk_day in df.index and pd.isna(df.loc[first_walk_day, "SMA_50"]):
                self.dropped_tickers.append(
                    (ticker, f"insufficient warmup history for SMA-{self.sma_period} by {first_walk_day.date()}")
                )
                continue
            enriched[ticker] = df
            kept_universe.append(ticker)
        self.universe = kept_universe
        self._state = {t: _PositionState() for t in self.universe}
        return enriched

    def describe(self) -> dict:
        info = super().describe()
        info["params"] = {
            "rsi_period": self.rsi_period,
            "rsi_entry_threshold": self.rsi_entry,
            "rsi_exit_threshold": self.rsi_exit,
            "sma_period": self.sma_period,
            "stop_loss_pct": self.stop_loss_pct,
            "max_holding_days": self.max_holding_days,
            "max_concurrent_positions": self.max_concurrent_positions,
        }
        info["assumptions"] = [
            "Universe is survivorship-biased (today's large caps, not point-in-time).",
            "Default thresholds were re-tuned mid-development against historical data "
            "(see config.py comments) -- treat single-window results as in-sample.",
            "The stop-loss is a delayed close-to-close exit rule, not an intraday stop.",
            "Short holding periods mean gains are mostly short-term (higher tax rates).",
        ]
        return info

    def _entry_allowed(self, ticker: str, day: pd.Timestamp, market: MarketDataView) -> bool:
        """Entry-eligibility hook, called once per RSI-qualified candidate on
        its signal day. The BASELINE strategy always returns True -- this
        exists solely so subclasses (mean_reversion_filtered) can veto
        entries with additional filters without duplicating any of the
        entry/exit/slot machinery. Exits are deliberately NOT routed through
        any hook: a risk-reducing exit must never be vetoed by a filter."""
        return True

    def on_day(self, day: pd.Timestamp, market: MarketDataView, sleeve_equity: float) -> list[TargetEvent]:
        events: list[TargetEvent] = []

        # 1. Confirm fills scheduled for today. Reading market.close(ticker, day)
        # here is safe: day == market.as_of, it is not future data.
        for ticker in self.universe:
            st = self._state[ticker]
            if st.pending_fill_date == day:
                if st.status == "pending_entry":
                    st.status = "open"
                    st.entry_fill_price = market.close(ticker, day)
                    st.entry_fill_date = day
                    st.days_held = 0
                elif st.status == "pending_exit":
                    st.status = "flat"
                    st.entry_fill_price = None
                    st.entry_fill_date = None
                    st.days_held = 0
                st.pending_fill_date = None

        # 2. Increment days_held for positions that were already open coming into today.
        for ticker in self.universe:
            st = self._state[ticker]
            if st.status == "open" and st.entry_fill_date is not None and st.entry_fill_date < day:
                st.days_held += 1

        # 3. Exit decision (priority order: stop-loss, timeout, SMA, RSI).
        for ticker in self.universe:
            st = self._state[ticker]
            if st.status != "open" or not market.has_data(ticker, day):
                continue
            close = market.close(ticker, day)
            reason = None
            if close <= st.entry_fill_price * (1.0 + self.stop_loss_pct):
                reason = "exit_stop_loss"
            elif st.days_held >= self.max_holding_days:
                reason = "exit_timeout"
            else:
                sma_val = market.indicator(ticker, "SMA_50", day)
                if pd.notna(sma_val) and close > sma_val:
                    reason = "exit_sma"
                else:
                    rsi_val = market.indicator(ticker, "RSI_14", day)
                    if pd.notna(rsi_val) and rsi_val >= self.rsi_exit:
                        reason = "exit_rsi"
            if reason is not None:
                fill_date = market.next_trading_day(day)
                if fill_date is None:
                    continue  # backtest window ends before this exit could fill
                events.append(
                    TargetEvent(
                        strategy=self.name, ticker=ticker, signal_date=day, fill_date=fill_date,
                        target_weight=0.0, sizing_price=close, reason=reason,
                    )
                )
                st.status = "pending_exit"
                st.pending_fill_date = fill_date

        # 4. Entry decision: only for flat tickers, only while slots remain.
        occupied = sum(1 for st in self._state.values() if st.status in ("open", "pending_entry", "pending_exit"))
        free_slots = self.max_concurrent_positions - occupied
        if free_slots > 0:
            candidates = []
            for ticker in self.universe:
                st = self._state[ticker]
                if st.status != "flat" or not market.has_data(ticker, day):
                    continue
                rsi_val = market.indicator(ticker, "RSI_14", day)
                if pd.notna(rsi_val) and rsi_val < self.rsi_entry and self._entry_allowed(ticker, day, market):
                    candidates.append((rsi_val, ticker))
            candidates.sort(key=lambda pair: pair[0])  # lowest RSI (most oversold) first
            for rsi_val, ticker in candidates[:free_slots]:
                st = self._state[ticker]  # re-fetch: `st` from the loop above is stale here
                fill_date = market.next_trading_day(day)
                if fill_date is None:
                    break
                close = market.close(ticker, day)
                weight = 1.0 / self.max_concurrent_positions
                events.append(
                    TargetEvent(
                        strategy=self.name, ticker=ticker, signal_date=day, fill_date=fill_date,
                        target_weight=weight, sizing_price=close, reason="entry_rsi",
                    )
                )
                st.status = "pending_entry"
                st.pending_fill_date = fill_date

        return events
