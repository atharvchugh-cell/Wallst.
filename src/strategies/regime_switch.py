"""Regime-aware hybrid: sector rotation when the broad market is in an
uptrend, 100% cash when it is not.

- Risk-on (SPY close > 200-day SMA at the month-end signal date): behave
  EXACTLY like sector rotation -- the ranking/weighting code path is
  inherited, not copied, so any fix to sector rotation automatically applies
  here and the two cannot silently diverge.
- Risk-off (SPY <= its 200-day SMA): target weight 0 on all 11 sector ETFs,
  i.e. 100% cash. Cash earns nothing (conservative: no synthetic T-bill
  yield is credited).

The regime is evaluated ONLY at month-end signal dates, same cadence as the
rotation itself -- simple, auditable, no intramonth whipsaw trading and no
hidden extra turnover. No new tuned parameters: reuses sector rotation's
lookback/top-K unchanged and the shared config.REGIME_SMA_PERIOD (the same
single regime definition mean_reversion_filtered uses).

Known cost, stated up front: a 200-day SMA regime filter is a blunt
instrument. It exits AFTER a crash is underway and re-enters AFTER a
recovery is underway -- it sacrifices upside around V-bottoms (2020) in
exchange for avoiding deep sustained bears (2008, 2022). Whether that trade
is worth it is exactly what the tournament's regime windows measure.
"""

from __future__ import annotations

import pandas as pd

from .. import config
from ..indicators import sma
from ..market_view import MarketDataView
from .base import TargetEvent
from .sector_rotation import SectorRotationStrategy


class RegimeSwitchStrategy(SectorRotationStrategy):
    name = "regime_switch"
    family = "regime_switch"
    signal_tickers = [config.REGIME_TICKER]

    def __init__(
        self,
        universe: list[str] | None = None,
        regime_sma_period: int = config.REGIME_SMA_PERIOD,
        **rotation_kwargs,
    ):
        super().__init__(universe=universe, **rotation_kwargs)
        if regime_sma_period <= 0:
            raise ValueError(f"regime_sma_period must be > 0, got {regime_sma_period}")
        self.regime_ticker = config.REGIME_TICKER
        self.regime_sma_period = regime_sma_period

    def describe(self) -> dict:
        info = super().describe()
        info["params"].update({
            "regime_ticker": self.regime_ticker,
            "regime_sma_period": self.regime_sma_period,
            "risk_off_allocation": "100% cash (earns 0%)",
        })
        info["assumptions"] = [
            "Risk-on behavior is EXACTLY sector rotation (inherited code path, same "
            "lookback/top-K); differences vs. sector rotation isolate the regime filter.",
            "The 200-day SMA regime filter exits after crashes start and re-enters after "
            "recoveries start -- it will look bad around V-bottoms (2020) and good in "
            "sustained bears (2022). Judge it across BOTH window types.",
            "Cash earns 0% while risk-off (no T-bill yield credited -- conservative).",
            "Regime is checked only at month-end signal dates (no intramonth de-risking).",
        ]
        return info

    def prepare(
        self, price_data: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex, start: pd.Timestamp
    ) -> dict[str, pd.DataFrame]:
        if self.regime_ticker not in price_data:
            raise ValueError(
                f"{self.name} requires {self.regime_ticker} price data for its market-regime "
                f"filter (declared in signal_tickers); it was not fetched. Refusing to run "
                f"without the filter rather than silently degrading to plain sector rotation."
            )
        enriched = super().prepare(price_data, calendar, start)
        regime_df = price_data[self.regime_ticker].copy()
        regime_df["RegimeSMA"] = sma(regime_df["Close"], self.regime_sma_period)
        enriched[self.regime_ticker] = regime_df
        return enriched

    def _regime_state(
        self, signal_date: pd.Timestamp, market: MarketDataView
    ) -> tuple[bool, float | None, float | None]:
        """Return (is_risk_on, regime_close, regime_sma). Reads the regime
        ticker's close/SMA once; callers (and the trace) reuse the values so
        there is never a duplicate read. Conservative default: if the regime
        signal is unavailable, treat it as risk-off (values None)."""
        if not market.has_data(self.regime_ticker, signal_date):
            return False, None, None
        regime_sma = market.indicator(self.regime_ticker, "RegimeSMA", signal_date)
        if pd.isna(regime_sma):
            return False, None, None
        regime_close = market.close(self.regime_ticker, signal_date)
        return (regime_close > regime_sma), regime_close, regime_sma

    def _risk_on(self, signal_date: pd.Timestamp, market: MarketDataView) -> bool:
        return self._regime_state(signal_date, market)[0]

    def _rebalance_events(
        self, signal_date: pd.Timestamp, market: MarketDataView, fill_date: pd.Timestamp | None
    ) -> list[TargetEvent]:
        is_on, regime_close, regime_sma = self._regime_state(signal_date, market)
        if self.recorder is not None:
            self._trace(
                decision_date=signal_date, ticker=self.regime_ticker, tradable=False,
                reason_code="REGIME_RISK_ON" if is_on else "REGIME_RISK_OFF",
                regime_state="risk_on" if is_on else "risk_off", eligible=None, selected=None,
                regime_ticker=self.regime_ticker, regime_close=regime_close, regime_sma=regime_sma,
                regime_sma_period=self.regime_sma_period, close=regime_close,
                signal_date=signal_date, fill_date=fill_date,
            )
        if is_on:
            return super()._rebalance_events(signal_date, market, fill_date)
        # Risk-off: everything to cash. Emitting zero-weight targets for all
        # ETFs (rather than only held ones) is safe -- the engine treats a
        # zero-target on a flat ticker as a no-op below MIN_TRADE_NOTIONAL --
        # and keeps this override state-free, mirroring sector rotation's own
        # emit-for-every-member convention.
        events: list[TargetEvent] = []
        for ticker in self.universe:
            if not market.has_data(ticker, signal_date):
                continue
            if self.recorder is not None:
                self._trace(
                    decision_date=signal_date, ticker=ticker, reason_code="REGIME_RISK_OFF",
                    regime_state="risk_off", eligible=False, selected=False, target_weight=0.0,
                    signal_date=signal_date, fill_date=fill_date,
                )
            events.append(
                TargetEvent(
                    strategy=self.name, ticker=ticker, signal_date=signal_date, fill_date=fill_date,
                    target_weight=0.0, sizing_price=market.close(ticker, signal_date),
                    reason="risk_off",
                )
            )
        return events
