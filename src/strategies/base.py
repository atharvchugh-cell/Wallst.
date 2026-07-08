"""Shared strategy interface.

Strategies are stateful objects walked forward one trading day at a time by
the engine. On each day they receive a MarketDataView bounded at that day
(structurally cannot see later dates -- see market_view.py) and the sleeve's
current equity, and return zero or more TargetEvents describing a desired
change in a ticker's target weight. Sizing (`requested_notional`) is filled
in by the engine immediately after collecting each day's events, using that
same day's equity -- strategies never see or use a future day's price or
equity to decide or size a trade.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import pandas as pd

from ..market_view import MarketDataView


@dataclass
class TargetEvent:
    strategy: str
    ticker: str
    signal_date: pd.Timestamp
    fill_date: pd.Timestamp | None   # None only transiently (initial_events()); the
                                       # engine assigns the real fill_date before queuing
    target_weight: float
    sizing_price: float               # informational only -- NOT used for share math
    reason: str | None = None
    requested_notional: float | None = None  # frozen by the engine at signal time
    sizing_date: pd.Timestamp = field(init=False)

    def __post_init__(self) -> None:
        self.sizing_date = self.signal_date


class Strategy(abc.ABC):
    name: str
    universe: list[str]
    # Human-readable strategy-family label for tournament grouping/reporting
    # (e.g. "mean_reversion", "momentum", "regime_switch"). Purely metadata.
    family: str = "unspecified"
    # Tickers the strategy needs PRICE DATA for but must never trade (e.g.
    # SPY as a market-regime signal). The tournament data-fetch layer unions
    # these with `universe`; they are structurally untradable because every
    # entry/exit loop iterates `self.universe` only, and these are not in it.
    signal_tickers: list[str] = []

    def __init__(self) -> None:
        self.dropped_tickers: list[tuple[str, str]] = []

    def describe(self) -> dict:
        """Metadata for tournament reporting: name/family/universe size plus
        subclass-supplied `params` (every knob and its value -- nothing
        hidden) and `assumptions` (plain-language statements a reader must
        accept before trusting this strategy's results). Subclasses override
        and extend; the base implementation reports what it can see."""
        return {
            "name": self.name,
            "family": self.family,
            "universe_size": len(self.universe),
            "signal_tickers": list(self.signal_tickers),
            "params": {},
            "assumptions": [],
        }

    def reset(self) -> None:
        """Reset all internal mutable state before a run. Subclasses with
        their own state must override and call super().reset()."""
        self.dropped_tickers = []

    def prepare(
        self,
        price_data: dict[str, pd.DataFrame],
        calendar: pd.DatetimeIndex,
        start: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        """One-time vectorized precompute of indicator columns, attached to
        (copies of) the raw OHLCV frames. `calendar` is the walk calendar
        ([start, end]); `price_data` includes the pre-start warmup buffer.
        Default: no extra columns."""
        return price_data

    def initial_events(self, market: MarketDataView, sleeve_equity: float) -> list[TargetEvent]:
        """TargetEvents decided using pre-start warmup data alone (e.g. sector
        rotation's last-month-end-before-start rebalance). `market` is bounded
        at the last available date before the walk's first day. Leave
        `fill_date=None`; the engine assigns the walk's first day uniformly.
        Default: none."""
        return []

    @abc.abstractmethod
    def on_day(self, day: pd.Timestamp, market: MarketDataView, sleeve_equity: float) -> list[TargetEvent]:
        """Called once per trading day, in chronological order. Must not use
        `market` to read anything dated after `day` -- structurally enforced,
        since MarketDataView raises LookaheadError on such reads."""
        raise NotImplementedError
