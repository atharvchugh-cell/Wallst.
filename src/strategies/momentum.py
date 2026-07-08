"""Cross-sectional momentum / trend following on large-cap stocks.

Signal (month-end only, same month-end mechanism as sector rotation): rank
the universe by trailing 126-trading-day (~6-month) total return. A ticker
is ELIGIBLE only if, at the signal date, (a) its close is above its 200-day
SMA (trend filter) and (b) its trailing 126-day return is positive (absolute
momentum). Hold the top K=5 eligible tickers equal-weighted at 1/K; anything
held that drops out gets target weight 0 at the same rebalance. If fewer
than K tickers are eligible, the remaining weight stays in CASH -- in a
broad downtrend this strategy de-risks by construction instead of being
forced fully invested.

Parameter provenance (see docs/TOURNAMENT_DESIGN.md section 3.1): 6-month
momentum is the canonical Jegadeesh-Titman horizon and the 200-day SMA is
the canonical long-term trend line (Faber) -- both predate this repo by
decades and were chosen for that canonical status, NOT tuned on this data.

There are deliberately NO intramonth exits or stops: monthly-cadence-only
decisions keep the strategy simple, auditable, and low-turnover, and adding
daily stops would be a second (confounded) change relative to the sector-
rotation control. Every held/target ticker re-emits a TargetEvent at each
rebalance so price drift is corrected (same convention as sector rotation).

SURVIVORSHIP BIAS: the default universe is the same survivorship-biased
large-cap list mean reversion uses (today's survivors, not point-in-time).
Momentum results on it are upward-biased -- past winners that later died
aren't in the list to be picked. Stated in describe() and every report.
"""

from __future__ import annotations

import pandas as pd

from .. import config
from ..indicators import month_end_dates, sma
from ..market_view import MarketDataView
from .base import Strategy, TargetEvent


class MomentumStrategy(Strategy):
    name = "momentum"
    family = "momentum"

    def __init__(
        self,
        universe: list[str] | None = None,
        lookback_trading_days: int = config.MOMENTUM_LOOKBACK_TRADING_DAYS,
        top_k: int = config.MOMENTUM_TOP_K,
        trend_sma_period: int = config.MOMENTUM_TREND_SMA_PERIOD,
    ):
        super().__init__()
        if lookback_trading_days <= 0:
            raise ValueError(f"lookback_trading_days must be > 0, got {lookback_trading_days}")
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        if trend_sma_period <= 0:
            raise ValueError(f"trend_sma_period must be > 0, got {trend_sma_period}")
        self.universe = list(universe) if universe is not None else list(config.MEAN_REVERSION_UNIVERSE)
        self.lookback_trading_days = lookback_trading_days
        self.top_k = top_k
        self.trend_sma_period = trend_sma_period
        self._last_rebalance_month: tuple[int, int] | None = None
        # Tickers currently held (or pending) at a nonzero target -- tracked so
        # a rebalance can zero out holdings that fell out of the top-K without
        # emitting no-op zero-weight events for names that were never held.
        self._held: set[str] = set()

    def reset(self) -> None:
        super().reset()
        self._last_rebalance_month = None
        self._held = set()

    def describe(self) -> dict:
        info = super().describe()
        info["params"] = {
            "lookback_trading_days": self.lookback_trading_days,
            "top_k": self.top_k,
            "trend_sma_period": self.trend_sma_period,
            "rebalance": "monthly (last trading day of month)",
        }
        info["assumptions"] = [
            "Universe is survivorship-biased (today's large caps) -- momentum results on "
            "such a list are upward-biased, since past winners that later died are absent.",
            "6-month lookback and 200-day SMA are canonical literature conventions, not "
            "tuned on this data; sensitivity variants are reported, never auto-selected.",
            "No intramonth exits or stops -- a crash between month-ends is ridden until "
            "the next rebalance.",
            "Cash earns 0% during defensive fallback (conservative; no T-bill yield).",
        ]
        return info

    def prepare(
        self, price_data: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex, start: pd.Timestamp
    ) -> dict[str, pd.DataFrame]:
        enriched: dict[str, pd.DataFrame] = {}
        kept_universe = []
        first_walk_day = calendar[0] if len(calendar) else pd.Timestamp(start)
        for ticker in self.universe:
            if ticker not in price_data:
                self.dropped_tickers.append((ticker, "no price data fetched"))
                continue
            df = price_data[ticker].copy()
            df["TrendSMA"] = sma(df["Close"], self.trend_sma_period)
            df["MomentumReturn"] = df["Close"] / df["Close"].shift(self.lookback_trading_days) - 1.0
            # Both indicators must be valid by the first walk day, or the
            # ticker can never produce a signal inside the requested window
            # (a frame that doesn't even reach the first walk day fails for
            # the same reason -- ~330 trading days of post-inception history
            # are needed before either indicator is defined).
            if (
                first_walk_day not in df.index
                or pd.isna(df.loc[first_walk_day, "TrendSMA"])
                or pd.isna(df.loc[first_walk_day, "MomentumReturn"])
            ):
                self.dropped_tickers.append(
                    (ticker, f"insufficient warmup history for SMA-{self.trend_sma_period}/"
                             f"{self.lookback_trading_days}d momentum by {first_walk_day.date()}")
                )
                continue
            enriched[ticker] = df
            kept_universe.append(ticker)
        self.universe = kept_universe
        return enriched

    def _rebalance_events(
        self, signal_date: pd.Timestamp, market: MarketDataView, fill_date: pd.Timestamp | None
    ) -> list[TargetEvent]:
        eligible: list[tuple[float, str]] = []
        for ticker in self.universe:
            if not market.has_data(ticker, signal_date):
                continue
            mom = market.indicator(ticker, "MomentumReturn", signal_date)
            trend = market.indicator(ticker, "TrendSMA", signal_date)
            close = market.close(ticker, signal_date)
            if pd.isna(mom) or pd.isna(trend):
                continue
            if mom > 0 and close > trend:
                eligible.append((mom, ticker))
        eligible.sort(key=lambda pair: pair[0], reverse=True)
        selected = {t for _mom, t in eligible[: self.top_k]}

        events: list[TargetEvent] = []
        # Weight is ALWAYS 1/top_k per selected name, never 1/len(selected):
        # with fewer than top_k eligible names the remainder stays in cash by
        # design (defensive fallback), rather than concentrating harder into
        # whatever survived the filters.
        for ticker in sorted(selected | self._held):
            if not market.has_data(ticker, signal_date):
                continue
            weight = (1.0 / self.top_k) if ticker in selected else 0.0
            events.append(
                TargetEvent(
                    strategy=self.name, ticker=ticker, signal_date=signal_date, fill_date=fill_date,
                    target_weight=weight, sizing_price=market.close(ticker, signal_date),
                    reason="rebalance" if weight > 0 else "rebalance_exit",
                )
            )
        self._held = selected
        return events

    def initial_events(self, market: MarketDataView, sleeve_equity: float) -> list[TargetEvent]:
        if not self.universe:
            return []
        ref_hist = market.history(self.universe[0], column="Close")
        if ref_hist.empty:
            return []
        m_ends = month_end_dates(ref_hist.index)
        if len(m_ends) == 0:
            return []
        warmup_signal_date = m_ends[-1]
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
