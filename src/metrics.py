"""Performance metrics: returns, risk, cost/turnover auditability, and
benchmark comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .engine import BacktestResult, Trade, Transaction


def total_return(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def sharpe_ratio(equity: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sharpe on daily returns, risk_free_rate default 0.0.
    Returns NaN (not 0) when the return std is 0 -- Sharpe is undefined
    there, not "no risk-adjusted return"."""
    if len(equity) < 3:
        return float("nan")
    returns = equity.pct_change().dropna()
    excess = returns - risk_free_rate / periods_per_year
    std = excess.std(ddof=0)
    if std == 0 or pd.isna(std):
        return float("nan")
    return float((excess.mean() / std) * np.sqrt(periods_per_year))


def win_rate(trades: list[Trade]) -> float:
    """Fraction of round-trip (full_exit) trades with positive realized P&L.
    Partial sells are excluded -- they're not a completed round trip."""
    full_exits = [t for t in trades if t.event_type == "full_exit"]
    if not full_exits:
        return 0.0
    wins = sum(1 for t in full_exits if t.realized_pnl > 0)
    return wins / len(full_exits)


def total_transaction_costs(transactions: list[Transaction]) -> float:
    return float(sum(tx.transaction_cost for tx in transactions))


def total_turnover(transactions: list[Transaction]) -> float:
    return float(sum(tx.executed_notional for tx in transactions))


def cost_drag_pct(transactions: list[Transaction], capital: float) -> float:
    if capital <= 0:
        return 0.0
    return total_transaction_costs(transactions) / capital


def is_short_period(
    start: pd.Timestamp, end: pd.Timestamp, threshold_days: int = config.SHORT_PERIOD_WARNING_CALENDAR_DAYS
) -> bool:
    return (pd.Timestamp(end) - pd.Timestamp(start)).days < threshold_days


def exposure_stats(positions: list[dict], universe: list[str]) -> dict:
    """days_with_any_position_pct: fraction of days with ANY non-cash holding.
    average_capital_invested_pct: average fraction of equity actually invested
    (a strategy holding one 20% slot most days looks very different on these
    two numbers). Aggregates per (date, strategy) first, then across
    strategies for combined-mode results, so a combined run's denominator is
    the sum of both sleeves' equity rather than accidentally reusing one
    sleeve's equity as if it were the whole portfolio's.
    """
    if not positions:
        return {"days_with_any_position_pct": 0.0, "average_capital_invested_pct": 0.0}
    df = pd.DataFrame(positions)
    per_sleeve_day = (
        df.groupby(["date", "strategy"])
        .agg(
            any_position=("shares", lambda s: bool((s.abs() > 1e-9).any())),
            total_mv=("market_value", "sum"),
            sleeve_equity=("sleeve_equity", "first"),
        )
        .reset_index()
    )
    per_day = per_sleeve_day.groupby("date").agg(
        any_position=("any_position", "any"),
        total_mv=("total_mv", "sum"),
        total_equity=("sleeve_equity", "sum"),
    )
    per_day["invested_pct"] = per_day.apply(
        lambda r: (r["total_mv"] / r["total_equity"]) if r["total_equity"] else 0.0, axis=1
    )
    return {
        "days_with_any_position_pct": float(per_day["any_position"].mean()),
        "average_capital_invested_pct": float(per_day["invested_pct"].mean()),
    }


def benchmark_metrics(strategy_equity: pd.Series, benchmark_close: pd.Series) -> dict:
    """Compares strategy equity to a buy-and-hold of `benchmark_close` (an
    adjusted-price total-return proxy, e.g. SPY) over the strategy's own
    effective date range -- caller is responsible for passing a
    `benchmark_close` already restricted to that range."""
    common = strategy_equity.index.intersection(benchmark_close.index)
    if len(common) < 2:
        return {}
    bench = benchmark_close.reindex(common)
    strat = strategy_equity.reindex(common)
    bench_equity = bench / bench.iloc[0] * strat.iloc[0]

    strat_returns = strat.pct_change().dropna()
    bench_returns = bench_equity.pct_change().dropna()
    common_returns_idx = strat_returns.index.intersection(bench_returns.index)
    correlation = (
        float(strat_returns.reindex(common_returns_idx).corr(bench_returns.reindex(common_returns_idx)))
        if len(common_returns_idx) > 1
        else float("nan")
    )

    return {
        "benchmark_total_return": total_return(bench_equity),
        "benchmark_cagr": cagr(bench_equity),
        "benchmark_max_drawdown": max_drawdown(bench_equity),
        "benchmark_sharpe": sharpe_ratio(bench_equity),
        "excess_return": total_return(strat) - total_return(bench_equity),
        "correlation_to_benchmark": correlation,
    }


def compute_all_metrics(result: BacktestResult, benchmark_close: pd.Series | None = None) -> dict:
    equity = result.equity_curve
    metrics = {
        "total_return": total_return(equity),
        "cagr": cagr(equity),
        "max_drawdown": max_drawdown(equity),
        "sharpe_ratio": sharpe_ratio(equity),
        "win_rate": win_rate(result.trades),
        "num_trades": len([t for t in result.trades if t.event_type == "full_exit"]),
        "num_partial_sells": len([t for t in result.trades if t.event_type == "partial_sell"]),
        "num_transactions": len(result.transactions),
        "num_target_events": len(result.target_events),
        "total_transaction_costs": total_transaction_costs(result.transactions),
        "total_turnover": total_turnover(result.transactions),
        "cost_drag_pct": cost_drag_pct(result.transactions, result.capital),
        "short_period_warning": is_short_period(result.start, result.end),
        "final_equity": float(equity.iloc[-1]) if len(equity) else result.capital,
    }
    metrics.update(exposure_stats(result.positions, result.universe))
    if benchmark_close is not None:
        metrics.update(benchmark_metrics(equity, benchmark_close))
    return metrics
