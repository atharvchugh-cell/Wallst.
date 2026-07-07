"""The backtest engine: a per-sleeve day-by-day loop that executes
TargetEvents against a single strategy's own cash/shares/lots, with
average-cost P&L accounting and cash-constrained buy scaling.

Each strategy sleeve is run fully independently -- combined ("both") mode
never shares a cash pool between strategies; see `combine_results`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .market_view import MarketDataView
from .strategies.base import Strategy, TargetEvent

MIN_TRADE_NOTIONAL = 0.01  # trades smaller than this (e.g. dust from float drift) are no-ops
EPSILON_SHARES = 1e-9


@dataclass
class Transaction:
    strategy: str
    ticker: str
    signal_date: pd.Timestamp
    fill_date: pd.Timestamp
    action: str  # "buy" | "sell"
    requested_target_weight: float
    requested_notional: float
    sizing_price: float
    shares_traded: float
    fill_price: float
    executed_notional: float
    transaction_cost: float
    cash_after: float
    position_shares_after: float
    avg_cost_basis_after: float
    reason: str | None
    actual_weight_after: float | None = None  # filled in after the day's equity is known


@dataclass
class Trade:
    strategy: str
    ticker: str
    event_type: str  # "partial_sell" | "full_exit"
    date: pd.Timestamp
    shares_sold: float
    sale_price: float
    avg_cost_basis: float
    realized_pnl: float  # gross, ignores buy/sell transaction costs
    realized_return_pct: float  # gross
    reason: str | None
    holding_days: int  # trading days held (fill day counts as 0), matches config.MAX_HOLDING_DAYS units
    holding_calendar_days: int = 0
    realized_pnl_net: float | None = None  # net of this lot's buy+sell transaction costs
    realized_return_pct_net: float | None = None


@dataclass
class BacktestResult:
    strategy_name: str
    capital: float
    start: pd.Timestamp
    end: pd.Timestamp
    equity_curve: pd.Series
    target_events: list[TargetEvent]
    transactions: list[Transaction]
    trades: list[Trade]
    positions: list[dict]
    dropped_tickers: list[tuple[str, str]]
    universe: list[str]
    cost_bps: float
    fractional_shares: bool


class EventValidationError(Exception):
    pass


def _validate_event(e: TargetEvent, calendar_walk: pd.DatetimeIndex, seen_keys: set, price_data: dict) -> None:
    if e.fill_date is None:
        raise EventValidationError(f"{e.ticker}: event has no fill_date")
    if e.fill_date <= e.signal_date:
        raise EventValidationError(
            f"{e.ticker}: fill_date {e.fill_date.date()} is not after signal_date {e.signal_date.date()}"
        )
    if e.fill_date not in calendar_walk:
        raise EventValidationError(f"{e.ticker}: fill_date {e.fill_date.date()} is not in the canonical calendar")
    key = (e.strategy, e.ticker, e.fill_date)
    if key in seen_keys:
        raise EventValidationError(f"Duplicate target event for {key}")
    seen_keys.add(key)
    if e.ticker not in price_data or e.fill_date not in price_data[e.ticker].index:
        raise EventValidationError(f"{e.ticker}: no price data at scheduled fill_date {e.fill_date.date()}")


def run_backtest(
    strategy: Strategy,
    price_data: dict[str, pd.DataFrame],
    full_calendar: pd.DatetimeIndex,
    start,
    end,
    capital: float,
    cost_bps: float = 5.0,
    fractional_shares: bool = True,
) -> BacktestResult:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    calendar_walk = full_calendar[(full_calendar >= start) & (full_calendar <= end)].sort_values()
    if len(calendar_walk) == 0:
        raise ValueError(f"No trading days in [{start.date()}, {end.date()}]")

    strategy.reset()
    enriched = strategy.prepare(price_data, calendar_walk, start)

    def close_lookup(ticker: str, d: pd.Timestamp) -> float:
        return float(enriched[ticker].loc[d, "Close"])

    cash = float(capital)
    shares: dict[str, float] = {}
    avg_cost: dict[str, float] = {}
    avg_cost_incl_fees: dict[str, float] = {}  # basis including this lot's buy-side transaction cost
    lot_entry_date: dict[str, pd.Timestamp] = {}

    target_events_log: list[TargetEvent] = []
    transactions: list[Transaction] = []
    trades: list[Trade] = []
    positions: list[dict] = []
    equity_curve: dict[pd.Timestamp, float] = {}
    pending_by_fill_date: dict[pd.Timestamp, list[TargetEvent]] = {}
    seen_keys: set = set()

    def queue_event(e: TargetEvent, sizing_equity: float) -> None:
        e.requested_notional = e.target_weight * sizing_equity
        _validate_event(e, calendar_walk, seen_keys, enriched)
        target_events_log.append(e)
        pending_by_fill_date.setdefault(e.fill_date, []).append(e)

    # --- Warmup / initial events (pre-start decisions, e.g. sector rotation's
    # last-month-end-before-start rebalance) ---
    before_start = full_calendar[full_calendar < calendar_walk[0]]
    if len(before_start) > 0:
        warmup_as_of = before_start[-1]
        warmup_view = MarketDataView(enriched, as_of=warmup_as_of, calendar=full_calendar)
        for e in strategy.initial_events(warmup_view, cash):
            e.fill_date = calendar_walk[0]
            queue_event(e, cash)

    # --- Main day-by-day walk ---
    for d in calendar_walk:
        events_today = pending_by_fill_date.pop(d, [])
        todays_transactions, cash = _execute_events(
            events_today, d, close_lookup, cash, shares, avg_cost, avg_cost_incl_fees, lot_entry_date,
            cost_bps, fractional_shares, trades, calendar_walk,
        )

        sleeve_equity_today = cash + sum(
            shares.get(t, 0.0) * close_lookup(t, d) for t in strategy.universe if shares.get(t, 0.0) != 0.0
        )
        for tx in todays_transactions:
            tx.actual_weight_after = (
                (tx.position_shares_after * tx.fill_price) / sleeve_equity_today
                if sleeve_equity_today > 0
                else 0.0
            )
        transactions.extend(todays_transactions)

        for ticker in strategy.universe:
            sh = shares.get(ticker, 0.0)
            px = close_lookup(ticker, d) if d in enriched[ticker].index else float("nan")
            mv = sh * px if pd.notna(px) else 0.0
            positions.append(
                {
                    "date": d, "strategy": strategy.name, "ticker": ticker, "shares": sh,
                    "adjusted_close": px, "market_value": mv,
                    "portfolio_weight": (mv / sleeve_equity_today) if sleeve_equity_today > 0 else 0.0,
                    "cash": cash, "sleeve_equity": sleeve_equity_today,
                }
            )

        equity_curve[d] = sleeve_equity_today

        view = MarketDataView(enriched, as_of=d, calendar=calendar_walk)
        for e in strategy.on_day(d, view, sleeve_equity_today):
            queue_event(e, sleeve_equity_today)

    equity_series = pd.Series(equity_curve).sort_index()
    return BacktestResult(
        strategy_name=strategy.name,
        capital=capital,
        start=start,
        end=end,
        equity_curve=equity_series,
        target_events=target_events_log,
        transactions=transactions,
        trades=trades,
        positions=positions,
        dropped_tickers=list(strategy.dropped_tickers),
        universe=list(strategy.universe),
        cost_bps=cost_bps,
        fractional_shares=fractional_shares,
    )


def _execute_events(
    events_today: list[TargetEvent],
    d: pd.Timestamp,
    close_lookup,
    cash: float,
    shares: dict[str, float],
    avg_cost: dict[str, float],
    avg_cost_incl_fees: dict[str, float],
    lot_entry_date: dict[str, pd.Timestamp],
    cost_bps: float,
    fractional_shares: bool,
    trades: list[Trade],
    calendar_walk: pd.DatetimeIndex,
) -> tuple[list[Transaction], float]:
    if not events_today:
        return [], cash

    proposed = []
    for e in events_today:
        fill_price = close_lookup(e.ticker, d)
        target_shares = e.requested_notional / fill_price
        current_shares = shares.get(e.ticker, 0.0)
        delta = target_shares - current_shares
        proposed.append((e, fill_price, delta))

    todays_transactions: list[Transaction] = []

    # 1. Sells first -- always fully executed, they only free up cash.
    for e, fill_price, delta in proposed:
        if delta >= -EPSILON_SHARES:
            continue
        qty = -delta
        if qty * fill_price < MIN_TRADE_NOTIONAL:
            continue
        tx = _apply_sell(
            e, fill_price, qty, shares, avg_cost, avg_cost_incl_fees, lot_entry_date,
            cost_bps, d, trades, calendar_walk,
        )
        cash += tx.executed_notional - tx.transaction_cost
        tx.cash_after = cash
        todays_transactions.append(tx)

    # 2. Buys, scaled down proportionally if they'd exceed available cash.
    buys = [(e, fill_price, delta) for e, fill_price, delta in proposed if delta > EPSILON_SHARES]
    total_buy_notional = sum(delta * fill_price for _, fill_price, delta in buys)
    total_buy_cost = total_buy_notional * (cost_bps / 10000.0)
    scale = 1.0
    if total_buy_notional + total_buy_cost > cash and (total_buy_notional + total_buy_cost) > 0:
        scale = max(0.0, cash / (total_buy_notional + total_buy_cost))

    for e, fill_price, delta in buys:
        qty = delta * scale
        if not fractional_shares:
            qty = float(int(qty))
        if qty * fill_price < MIN_TRADE_NOTIONAL:
            continue
        tx = _apply_buy(e, fill_price, qty, shares, avg_cost, avg_cost_incl_fees, lot_entry_date, cost_bps, d)
        cash -= tx.executed_notional + tx.transaction_cost
        tx.cash_after = cash
        todays_transactions.append(tx)

    return todays_transactions, cash


def _apply_sell(
    e, fill_price, qty, shares, avg_cost, avg_cost_incl_fees, lot_entry_date, cost_bps, d, trades, calendar_walk,
) -> Transaction:
    ticker = e.ticker
    current_shares = shares.get(ticker, 0.0)
    basis = avg_cost.get(ticker, fill_price)
    basis_incl_fees = avg_cost_incl_fees.get(ticker, fill_price)
    cost = qty * fill_price * (cost_bps / 10000.0)
    new_shares = current_shares - qty
    if abs(new_shares) < EPSILON_SHARES:
        new_shares = 0.0

    realized_pnl = qty * (fill_price - basis)  # gross, ignores all transaction costs
    realized_return_pct = (fill_price / basis - 1.0) if basis else 0.0

    net_sale_price = fill_price - (cost / qty if qty else 0.0)  # sell-side fee reduces effective proceeds
    realized_pnl_net = qty * (net_sale_price - basis_incl_fees)  # basis already includes buy-side fee
    realized_return_pct_net = (net_sale_price / basis_incl_fees - 1.0) if basis_incl_fees else 0.0

    entry_date = lot_entry_date.get(ticker, d)
    holding_calendar_days = (d - entry_date).days
    entry_idx = int(calendar_walk.searchsorted(entry_date))
    exit_idx = int(calendar_walk.searchsorted(d))
    holding_trading_days = max(0, exit_idx - entry_idx)
    event_type = "full_exit" if new_shares == 0.0 else "partial_sell"
    trades.append(
        Trade(
            strategy=e.strategy, ticker=ticker, event_type=event_type, date=d,
            shares_sold=qty, sale_price=fill_price, avg_cost_basis=basis,
            realized_pnl=realized_pnl, realized_return_pct=realized_return_pct,
            reason=e.reason, holding_days=holding_trading_days, holding_calendar_days=holding_calendar_days,
            realized_pnl_net=realized_pnl_net, realized_return_pct_net=realized_return_pct_net,
        )
    )
    shares[ticker] = new_shares
    if new_shares == 0.0:
        avg_cost.pop(ticker, None)
        avg_cost_incl_fees.pop(ticker, None)
        lot_entry_date.pop(ticker, None)
    return Transaction(
        strategy=e.strategy, ticker=ticker, signal_date=e.signal_date, fill_date=e.fill_date,
        action="sell", requested_target_weight=e.target_weight, requested_notional=e.requested_notional,
        sizing_price=e.sizing_price, shares_traded=qty, fill_price=fill_price,
        executed_notional=qty * fill_price, transaction_cost=cost, cash_after=0.0,
        position_shares_after=new_shares, avg_cost_basis_after=avg_cost.get(ticker, 0.0), reason=e.reason,
    )


def _apply_buy(e, fill_price, qty, shares, avg_cost, avg_cost_incl_fees, lot_entry_date, cost_bps, d) -> Transaction:
    ticker = e.ticker
    current_shares = shares.get(ticker, 0.0)
    cost = qty * fill_price * (cost_bps / 10000.0)
    current_basis = avg_cost.get(ticker, 0.0)
    current_basis_incl_fees = avg_cost_incl_fees.get(ticker, 0.0)
    new_shares = current_shares + qty
    fill_price_incl_fee = fill_price + (cost / qty if qty else 0.0)
    new_basis = (current_shares * current_basis + qty * fill_price) / new_shares if new_shares else 0.0
    new_basis_incl_fees = (
        (current_shares * current_basis_incl_fees + qty * fill_price_incl_fee) / new_shares if new_shares else 0.0
    )
    if current_shares == 0.0:
        lot_entry_date[ticker] = d
    shares[ticker] = new_shares
    avg_cost[ticker] = new_basis
    avg_cost_incl_fees[ticker] = new_basis_incl_fees
    return Transaction(
        strategy=e.strategy, ticker=ticker, signal_date=e.signal_date, fill_date=e.fill_date,
        action="buy", requested_target_weight=e.target_weight, requested_notional=e.requested_notional,
        sizing_price=e.sizing_price, shares_traded=qty, fill_price=fill_price,
        executed_notional=qty * fill_price, transaction_cost=cost, cash_after=0.0,
        position_shares_after=new_shares, avg_cost_basis_after=new_basis, reason=e.reason,
    )


def combine_results(result_a: BacktestResult, result_b: BacktestResult) -> BacktestResult:
    """Combine two independently-run sleeves. Uses the INTERSECTION of their
    valid-equity date ranges (not an outer join, which would risk implying
    one sleeve was invested before it actually started). No cash is shared
    between sleeves at any point -- this only sums the two already-independent
    equity curves and concatenates their (already strategy-tagged) records."""
    common_index = result_a.equity_curve.index.intersection(result_b.equity_curve.index).sort_values()
    if len(common_index) == 0:
        raise ValueError("No overlapping dates between the two sleeves; cannot combine.")
    combined_equity = result_a.equity_curve.reindex(common_index) + result_b.equity_curve.reindex(common_index)

    return BacktestResult(
        strategy_name="both",
        capital=result_a.capital + result_b.capital,
        start=common_index[0],
        end=common_index[-1],
        equity_curve=combined_equity,
        target_events=result_a.target_events + result_b.target_events,
        transactions=result_a.transactions + result_b.transactions,
        trades=result_a.trades + result_b.trades,
        positions=result_a.positions + result_b.positions,
        dropped_tickers=result_a.dropped_tickers + result_b.dropped_tickers,
        universe=result_a.universe + result_b.universe,
        cost_bps=result_a.cost_bps,
        fractional_shares=result_a.fractional_shares,
    )
