"""Filtered mean reversion: the baseline MeanReversionStrategy with two
additional ENTRY-ONLY filters. Every inherited threshold (RSI entry/exit,
SMA period, stop, timeout, slots, sizing) is unchanged -- the point is to
isolate the value of the filters, not to re-tune the baseline.

1. Market-regime gate: no NEW entries while SPY closes below its 200-day
   SMA. Buying oversold stocks works when dips are noise around an uptrend;
   in a sustained downtrend "oversold" keeps getting more oversold (2008,
   2022). Exits are never gated -- risk-reducing exits must not be blocked.
2. Falling-knife guard: skip a candidate whose trailing 5-trading-day return
   is <= -15%. RSI < entry-threshold after an orderly pullback is a dip; a
   -15% week is an event (earnings blowup, fraud, guidance cut) where mean
   reversion has no edge. Round, severe, disclosed heuristic -- not fitted.
   Scope limitation, stated plainly: the guard rejects entries into an
   ESTABLISHED crash. Day 1 of what later becomes a crash is
   indistinguishable from an ordinary dip without foresight, so a first-day
   entry can still happen (and is then managed by the inherited stop-loss).

SPY is a SIGNAL-ONLY ticker (`signal_tickers`): its price data is required
(prepare() hard-fails without it), but it is structurally untradable because
it is not in `self.universe`, which is the only set the inherited entry/exit
loops iterate.

Both filters read only same-day-or-earlier data through the MarketDataView,
so the no-lookahead guarantee is inherited unchanged.
"""

from __future__ import annotations

import pandas as pd

from .. import config
from ..indicators import sma
from ..market_view import MarketDataView
from .mean_reversion import MeanReversionStrategy


class FilteredMeanReversionStrategy(MeanReversionStrategy):
    name = "mean_reversion_filtered"
    family = "mean_reversion"
    signal_tickers = [config.REGIME_TICKER]

    def __init__(
        self,
        universe: list[str] | None = None,
        regime_sma_period: int = config.REGIME_SMA_PERIOD,
        knife_lookback_days: int = config.FILTERED_MR_KNIFE_LOOKBACK_TRADING_DAYS,
        knife_return_threshold: float = config.FILTERED_MR_KNIFE_RETURN_THRESHOLD,
        **baseline_kwargs,
    ):
        super().__init__(universe=universe, **baseline_kwargs)
        if regime_sma_period <= 0:
            raise ValueError(f"regime_sma_period must be > 0, got {regime_sma_period}")
        if knife_lookback_days <= 0:
            raise ValueError(f"knife_lookback_days must be > 0, got {knife_lookback_days}")
        if knife_return_threshold >= 0:
            raise ValueError(f"knife_return_threshold must be < 0, got {knife_return_threshold}")
        self.regime_ticker = config.REGIME_TICKER
        self.regime_sma_period = regime_sma_period
        self.knife_lookback_days = knife_lookback_days
        self.knife_return_threshold = knife_return_threshold

    def describe(self) -> dict:
        info = super().describe()
        info["params"].update({
            "regime_ticker": self.regime_ticker,
            "regime_sma_period": self.regime_sma_period,
            "knife_lookback_days": self.knife_lookback_days,
            "knife_return_threshold": self.knife_return_threshold,
        })
        info["assumptions"] = [
            "Identical to baseline mean reversion EXCEPT two entry-only filters; every "
            "inherited threshold is unchanged, so differences vs. the baseline isolate "
            "the filters' effect.",
            "Regime gate (SPY > 200-day SMA) will miss the sharpest V-bottom entries "
            "(e.g. April 2020) by design -- it trades upside for downside protection.",
            "Falling-knife threshold (-15% in 5 days) is a disclosed heuristic, not a "
            "fitted parameter; sensitivity variants are reported, never auto-selected.",
        ] + [a for a in info["assumptions"] if "stop-loss" in a or "survivorship" in a.lower()
             or "tax" in a.lower() or "re-tuned" in a]
        return info

    def prepare(
        self, price_data: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex, start: pd.Timestamp
    ) -> dict[str, pd.DataFrame]:
        if self.regime_ticker not in price_data:
            raise ValueError(
                f"{self.name} requires {self.regime_ticker} price data for its market-regime "
                f"filter (declared in signal_tickers); it was not fetched. Refusing to run "
                f"without the filter rather than silently degrading to the unfiltered baseline."
            )
        enriched = super().prepare(price_data, calendar, start)
        # Per-universe-ticker knife-guard column (trailing N-day return).
        for ticker in self.universe:
            df = enriched[ticker]
            df["KnifeReturn"] = df["Close"] / df["Close"].shift(self.knife_lookback_days) - 1.0
        # Regime frame: SPY with its regime SMA. Added to the enriched dict so
        # the MarketDataView can serve it; never in self.universe, so it is
        # never iterated for entries/exits and the engine never trades it.
        regime_df = price_data[self.regime_ticker].copy()
        regime_df["RegimeSMA"] = sma(regime_df["Close"], self.regime_sma_period)
        enriched[self.regime_ticker] = regime_df
        return enriched

    def _entry_allowed(self, ticker: str, day: pd.Timestamp, market: MarketDataView) -> bool:
        # 1. Market-regime gate: SPY must close above its regime SMA today.
        if not market.has_data(self.regime_ticker, day):
            return False  # can't confirm risk-on -> no new risk (conservative)
        regime_sma = market.indicator(self.regime_ticker, "RegimeSMA", day)
        if pd.isna(regime_sma) or market.close(self.regime_ticker, day) <= regime_sma:
            return False
        # 2. Falling-knife guard: reject crash-shaped candidates.
        knife = market.indicator(ticker, "KnifeReturn", day)
        if pd.isna(knife) or knife <= self.knife_return_threshold:
            return False
        return True
