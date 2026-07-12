"""Synchronized multi-sleeve portfolio engine for the lab.

The production `--strategy portfolio` mode runs each sleeve as a fully
independent `run_backtest` call and sums the equity curves afterwards. That
is correct for a static allocation, but portfolio-LEVEL enhancements
(volatility targeting, drawdown throttles, dynamic sleeve allocation,
cross-sleeve concentration caps) need to observe combined portfolio state
*while the walk is in progress* and to place real, cost-bearing orders in
response. This engine therefore walks every sleeve through ONE shared
day loop.

Equivalence guarantee: sleeves never share state unless an enhancement is
enabled, and each sleeve's per-day sequence (execute fills -> mark equity ->
strategy.on_day -> queue events) is byte-for-byte the same code path as
src/engine.py's run_backtest -- fills go through the SAME `_execute_events` /
`_validate_event` functions, not a reimplementation. With every enhancement
disabled the outputs are therefore identical to the production portfolio
mode; tests/test_lab_equivalence.py asserts this field by field.

No lookahead: strategies still only ever see a MarketDataView bounded at the
current day; enhancement overlays (added by later commits) may only read
state derived from data through the current signal date, and their orders
fill the next trading day like every other order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .. import data
from ..engine import BacktestResult, _execute_events, _validate_event
from ..market_view import MarketDataView
from ..metrics import compute_all_metrics, spy_standalone_metrics
from ..portfolio import (
    PortfolioResult,
    SleeveAllocation,
    _build_combined_result,
    _combine_sleeve_equities,
    allocate_capital,
    validate_portfolio_weights,
)
from ..strategies.base import TargetEvent
from ..tournament import STRATEGY_REGISTRY, StrategySpec
from .dataprep import PreparedSleeve, prepare_sleeve
from .lab_config import LabConfig


class LabEngineError(Exception):
    pass


@dataclass
class _SleeveState:
    """Mutable per-sleeve walk state -- the same variables run_backtest keeps
    as locals, held per sleeve so the day loop can interleave sleeves."""

    prepared: PreparedSleeve
    enriched: dict = field(default_factory=dict)
    calendar_walk: pd.DatetimeIndex | None = None
    walk_set: set = field(default_factory=set)
    cash: float = 0.0
    shares: dict = field(default_factory=dict)
    avg_cost: dict = field(default_factory=dict)
    avg_cost_incl_fees: dict = field(default_factory=dict)
    lot_entry_date: dict = field(default_factory=dict)
    pending_by_fill_date: dict = field(default_factory=dict)
    seen_keys: set = field(default_factory=set)
    target_events_log: list = field(default_factory=list)
    transactions: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    positions: list = field(default_factory=list)
    equity_curve: dict = field(default_factory=dict)
    equity_today: float = 0.0
    # Last raw (pre-overlay) target weight per ticker, as emitted by the
    # strategy. Overlays use this to re-scale held positions without asking
    # the strategy to re-decide.
    raw_targets: dict = field(default_factory=dict)

    def close_lookup(self, ticker: str, d: pd.Timestamp) -> float:
        return float(self.enriched[ticker].loc[d, "Close"])

    def queue_event(self, e: TargetEvent, sizing_equity: float) -> None:
        e.requested_notional = e.target_weight * sizing_equity
        _validate_event(e, self.calendar_walk, self.seen_keys, self.enriched)
        self.target_events_log.append(e)
        self.pending_by_fill_date.setdefault(e.fill_date, []).append(e)

    def mark_equity(self, d: pd.Timestamp) -> float:
        strategy = self.prepared.strategy
        self.equity_today = self.cash + sum(
            self.shares.get(t, 0.0) * self.close_lookup(t, d)
            for t in strategy.universe
            if self.shares.get(t, 0.0) != 0.0
        )
        return self.equity_today


@dataclass
class LabRunResult:
    """A lab portfolio run: the standard PortfolioResult (identical shape to
    the production mode) plus the lab context that produced it."""

    portfolio: PortfolioResult
    prepared: list[PreparedSleeve]
    lab_config: LabConfig
    trace: object | None = None  # DecisionTrace when tracing is enabled (see trace.py)


def run_lab_portfolio(
    pairs: list[tuple[str, float]],
    total_capital: float,
    start,
    end,
    cost_bps: float,
    fractional_shares: bool,
    refresh_cache: bool,
    lab_config: LabConfig | None = None,
    registry: dict[str, StrategySpec] | None = None,
    mr_universe: list[str] | None = None,
    mr_universe_info: dict | None = None,
    recorder=None,
    param_overrides_by_strategy: dict[str, dict] | None = None,
    warmup_overrides_by_strategy: dict[str, int] | None = None,
) -> LabRunResult:
    """Run the weighted portfolio through the synchronized day loop.

    With `lab_config` at defaults (or None) this reproduces the production
    `portfolio.run_portfolio` outputs exactly. `recorder` (see trace.py) is
    observational and behavior-neutral either way.
    """
    if registry is None:
        registry = STRATEGY_REGISTRY
    if lab_config is None:
        lab_config = LabConfig()
    lab_config.validate()
    validate_portfolio_weights(pairs, registry)
    param_overrides_by_strategy = param_overrides_by_strategy or {}
    warmup_overrides_by_strategy = warmup_overrides_by_strategy or {}

    overlays = _build_overlays(lab_config, pairs)

    # --- Prepare each positive-weight sleeve (identical data path to the
    # production mode; see dataprep.py) ---
    allocations = allocate_capital(pairs, total_capital)
    states: list[_SleeveState] = []
    skipped_zero_weight: list[str] = []
    for (name, weight), (_name, allocated) in zip(pairs, allocations):
        if allocated <= 0:
            skipped_zero_weight.append(name)
            continue
        spec = registry[name]
        prepared = prepare_sleeve(
            spec, weight, allocated, start, end, refresh_cache,
            universe=mr_universe if spec.uses_stock_universe else None,
            param_overrides=param_overrides_by_strategy.get(name),
            warmup_override=warmup_overrides_by_strategy.get(name),
        )
        states.append(_SleeveState(prepared=prepared, cash=float(allocated)))

    if not states:
        raise LabEngineError("No sleeves with positive capital to run (all weights were zero?).")

    # --- Per-sleeve setup: reset, prepare, warmup initial events (mirrors
    # run_backtest's pre-walk block exactly) ---
    for st in states:
        prepared = st.prepared
        strategy = prepared.strategy
        calendar_walk = prepared.full_calendar[
            (prepared.full_calendar >= prepared.walk_start)
            & (prepared.full_calendar <= prepared.walk_end)
        ].sort_values()
        if len(calendar_walk) == 0:
            raise ValueError(
                f"No trading days in [{prepared.walk_start.date()}, {prepared.walk_end.date()}]"
            )
        st.calendar_walk = calendar_walk
        st.walk_set = set(calendar_walk)

        strategy.reset()
        if recorder is not None:
            strategy.recorder = recorder
        st.enriched = strategy.prepare(prepared.price_data, calendar_walk, prepared.walk_start)

        before_start = prepared.full_calendar[prepared.full_calendar < calendar_walk[0]]
        if len(before_start) > 0:
            warmup_view = MarketDataView(st.enriched, as_of=before_start[-1], calendar=prepared.full_calendar)
            if recorder is not None:
                recorder.set_context(sleeve=strategy.name, portfolio_equity=None)
            for e in strategy.initial_events(warmup_view, st.cash):
                e.fill_date = calendar_walk[0]
                st.queue_event(e, st.cash)
                st.raw_targets[e.ticker] = e.target_weight

    # --- Master day loop over the union of sleeve walk calendars ---
    master_calendar = states[0].calendar_walk
    for st in states[1:]:
        master_calendar = master_calendar.union(st.calendar_walk)
    master_calendar = master_calendar.sort_values()

    for d in master_calendar:
        active = [st for st in states if d in st.walk_set]

        # Phase 0 (enhancements only): scheduled cross-sleeve cash transfers
        # execute before fills; sells from shrinking sleeves were scheduled
        # alongside, so cash cannot go structurally negative end-of-day.
        if overlays is not None:
            overlays.apply_transfers(d, active, recorder)

        # Phase 1: execute today's fills and mark sleeve equity -- the exact
        # per-sleeve sequence run_backtest uses.
        for st in active:
            events_today = st.pending_by_fill_date.pop(d, [])
            todays_transactions, st.cash = _execute_events(
                events_today, d, st.close_lookup, st.cash, st.shares, st.avg_cost,
                st.avg_cost_incl_fees, st.lot_entry_date, cost_bps, fractional_shares,
                st.trades, st.calendar_walk,
            )
            equity = st.mark_equity(d)
            for tx in todays_transactions:
                tx.actual_weight_after = (
                    (tx.position_shares_after * tx.fill_price) / equity if equity > 0 else 0.0
                )
            st.transactions.extend(todays_transactions)
            if recorder is not None and todays_transactions:
                recorder.record_fills(st.prepared.name, todays_transactions)

            strategy = st.prepared.strategy
            for ticker in strategy.universe:
                sh = st.shares.get(ticker, 0.0)
                px = st.close_lookup(ticker, d) if d in st.enriched[ticker].index else float("nan")
                mv = sh * px if pd.notna(px) else 0.0
                st.positions.append(
                    {
                        "date": d, "strategy": strategy.name, "ticker": ticker, "shares": sh,
                        "adjusted_close": px, "market_value": mv,
                        "portfolio_weight": (mv / equity) if equity > 0 else 0.0,
                        "cash": st.cash, "sleeve_equity": equity,
                    }
                )
            st.equity_curve[d] = equity

        portfolio_equity_today = sum(st.equity_today for st in active)
        if recorder is not None:
            recorder.record_portfolio_day(d, active, portfolio_equity_today)

        # Phase 2: strategy decisions for today (signal date d, fill d+1).
        for st in active:
            strategy = st.prepared.strategy
            view = MarketDataView(st.enriched, as_of=d, calendar=st.calendar_walk)
            if recorder is not None:
                recorder.set_context(sleeve=strategy.name, portfolio_equity=portfolio_equity_today)
            sizing_equity = st.equity_today
            for e in strategy.on_day(d, view, sizing_equity):
                raw_weight = e.target_weight
                if overlays is not None:
                    e = overlays.transform_event(d, st, e, states, recorder)
                    if e is None:
                        continue
                st.queue_event(e, sizing_equity)
                if raw_weight == 0.0:
                    st.raw_targets.pop(e.ticker, None)
                else:
                    st.raw_targets[e.ticker] = raw_weight

        # Phase 3 (enhancements only): portfolio-level overlays observe state
        # through day d and may schedule adjustment/reallocation orders that
        # fill at d+1 -- the same signal->fill lag every strategy order has.
        if overlays is not None:
            overlays.on_day_end(d, active, states, recorder)

    # --- Assemble per-sleeve BacktestResults (same shape as run_backtest) ---
    sleeves: list[SleeveAllocation] = []
    for st in states:
        prepared = st.prepared
        strategy = prepared.strategy
        equity_series = pd.Series(st.equity_curve).sort_index()
        result = BacktestResult(
            strategy_name=strategy.name,
            capital=prepared.allocated_capital,
            start=st.calendar_walk[0],
            end=st.calendar_walk[-1],
            equity_curve=equity_series,
            target_events=st.target_events_log,
            transactions=st.transactions,
            trades=st.trades,
            positions=st.positions,
            dropped_tickers=list(prepared.pre_drops) + list(strategy.dropped_tickers),
            universe=list(strategy.universe),
            cost_bps=cost_bps,
            fractional_shares=fractional_shares,
        )
        metrics = compute_all_metrics(result, benchmark_close=prepared.spy_df["Close"])
        sleeves.append(
            SleeveAllocation(
                strategy=prepared.name, weight=prepared.weight,
                allocated_capital=prepared.allocated_capital, result=result, metrics=metrics,
            )
        )

    # --- Combine into a PortfolioResult (same helpers as the production mode) ---
    combined_equity, common = _combine_sleeve_equities(sleeves)
    common_start, common_end = common[0], common[-1]
    combined_result = _build_combined_result(sleeves, combined_equity, common, total_capital)

    spy_df = data.get_benchmark_data(common_start, common_end, force_refresh=refresh_cache)
    spy_close = spy_df["Close"]
    spy_close = spy_close[(spy_close.index >= common_start) & (spy_close.index <= common_end)]

    metrics = compute_all_metrics(combined_result, benchmark_close=spy_close)
    spy_metrics = spy_standalone_metrics(spy_close)

    portfolio_final = float(combined_equity.iloc[-1])
    for sleeve in sleeves:
        sleeve.final_value = float(sleeve.result.equity_curve.reindex(common).iloc[-1])
        sleeve.ending_weight = (sleeve.final_value / portfolio_final) if portfolio_final else float("nan")
        sleeve.pnl_contribution = sleeve.final_value - sleeve.allocated_capital
        sleeve.cost_contribution = sum(
            tx.transaction_cost
            for tx in sleeve.result.transactions
            if common_start <= tx.fill_date <= common_end
        )

    pf = PortfolioResult(
        weights=list(pairs),
        total_capital=total_capital,
        sleeves=sleeves,
        combined_result=combined_result,
        metrics=metrics,
        spy_metrics=spy_metrics,
        common_start=common_start,
        common_end=common_end,
        spy_range=f"{common_start.date()} to {common_end.date()}",
        skipped_zero_weight=skipped_zero_weight,
    )
    trace = recorder.finalize(pf) if recorder is not None else None
    return LabRunResult(
        portfolio=pf, prepared=[st.prepared for st in states], lab_config=lab_config, trace=trace,
    )


def _build_overlays(lab_config: LabConfig, pairs: list[tuple[str, float]]):
    """Overlay pipeline factory. Returns None when no behavior-changing
    enhancement is enabled -- the day loop then contains zero enhancement
    branches, which is what the baseline equivalence tests pin down."""
    if not lab_config.any_behavior_change():
        return None
    from .overlays import OverlayPipeline  # local import: overlays are optional machinery

    return OverlayPipeline(lab_config, pairs)
