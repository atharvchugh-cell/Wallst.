"""Robustness testing: run mean_reversion and sector_rotation once per
historical window, then compare fixed capital-allocation mixes between them
(plus a SPY benchmark row) across those windows.

Diagnostics only -- every allocation mix runs the existing strategies with
their existing (config.py) default parameters, completely unmodified. This
module does not tune or change RSI thresholds, SMA periods, holding-day
limits, stop rules, sector top-K, universes, or transaction-cost
assumptions, and it does not change the behavior of `mean_reversion`,
`sector_rotation`, `both`, or `compare` mode.

## How allocation mixes are computed (blended, not re-run)

For a given window, this module runs mean_reversion ONCE and sector_rotation
ONCE (each at the full requested `--capital`, exactly like `--strategy
compare` already does). Each allocation mix's numbers are then computed by
capital-weighting those two already-computed RAW equity curves --
`w_sr * sector_equity_curve + w_mr * mean_reversion_equity_curve`, summed
over the dates both sleeves share -- NOT by re-running the backtest engine
once per allocation mix, and NOT by normalizing either curve to a
1.0-based return index first. Normalizing first would erase each sleeve's
first-day fill/transaction-cost drag (the engine's recorded equity already
reflects costs paid on or before that date), so this module deliberately
blends the raw dollar curves instead -- the same convention
`engine.combine_results` uses for `--strategy both`.

This is equivalent to actually re-running each sleeve at its allocated
capital split PROVIDED both sleeves were run at the same `capital` this
module is blending for -- true for every caller in this codebase
(`run_robustness` always runs both sleeves at the same `args.capital`
before blending) -- because every dollar decision the engine makes is
sized as `target_weight * sleeve_equity` (a pure FRACTION of that sleeve's
own equity -- see `engine.py`/`strategies/*.py`) and `cost_bps` is a fixed
rate, so both scale linearly with capital. Two caveats where the
equivalence is not exact:

- Fixed-dollar dust thresholds in `engine.py` (`MIN_TRADE_NOTIONAL = 0.01`,
  `EPSILON_SHARES = 1e-9`), which never bind at any position size a
  $1k+ backtest actually produces.
- Whole-share rounding in `--no-fractional-shares` mode: a sleeve actually
  re-run at a smaller allocated capital could round share counts
  differently than the blend implies.

Blending is used instead of re-running because it turns 5 windows x 6
allocations x 2 sleeves (60 backtests) into 5 windows x 2 sleeves (10
backtests) plus arithmetic -- a ~6x reduction in yfinance fetches and
engine runs.
"""

from __future__ import annotations

import pandas as pd

from .engine import BacktestResult
from .metrics import (
    benchmark_metrics,
    best_worst_month,
    cagr,
    calmar_ratio,
    max_drawdown,
    max_drawdown_duration_days,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)

# (label, sector_rotation_weight, mean_reversion_weight)
ALLOCATION_MIXES: list[tuple[str, float, float]] = [
    ("100% sector / 0% mean-reversion", 1.00, 0.00),
    ("75% sector / 25% mean-reversion", 0.75, 0.25),
    ("50% sector / 50% mean-reversion", 0.50, 0.50),
    ("25% sector / 75% mean-reversion", 0.25, 0.75),
    ("0% sector / 100% mean-reversion", 0.00, 1.00),
]

# (label, start, end) -- calendar-year windows, inclusive. These are FIXED
# defaults (not derived from --start/--end), matching the specific windows
# requested for this robustness check.
DEFAULT_ROBUSTNESS_WINDOWS: list[tuple[str, str, str]] = [
    ("2019-2021", "2019-01-01", "2021-12-31"),
    ("2020-2022", "2020-01-01", "2022-12-31"),
    ("2021-2023", "2021-01-01", "2023-12-31"),
    ("2022-2024", "2022-01-01", "2024-12-31"),
    ("2019-2024", "2019-01-01", "2024-12-31"),
]


def blend_equity_curve(
    sr_result: BacktestResult, mr_result: BacktestResult, w_sr: float, w_mr: float
) -> tuple[pd.Series, pd.Timestamp, pd.Timestamp]:
    """Capital-weighted blend of two independently-run sleeves' RAW equity
    curves -- `w_sr * sr_equity + w_mr * mr_equity` over the dates both
    sleeves share. Deliberately does NOT normalize either curve to a
    1.0-based return index first: both sleeves were run at the same full
    capital (see module docstring), so their raw equity values are already
    in matching dollar units, and summing them directly -- rather than
    rescaling by each curve's own first value -- preserves each sleeve's
    actual first-day fill/transaction-cost economics instead of resetting
    every allocation mix back to a clean, cost-free `capital` starting
    point. Returns (blended_equity, window_start, window_end) over the
    INTERSECTION of the two sleeves' valid ranges (same convention
    `engine.combine_results` uses)."""
    common = sr_result.equity_curve.index.intersection(mr_result.equity_curve.index).sort_values()
    if len(common) == 0:
        raise ValueError("No overlapping dates between sector_rotation and mean_reversion sleeves.")
    window_start, window_end = common[0], common[-1]

    sr_equity = sr_result.equity_curve.reindex(common)
    mr_equity = mr_result.equity_curve.reindex(common)
    blended = w_sr * sr_equity + w_mr * mr_equity
    return blended, window_start, window_end


def _weighted_count(metrics: dict, key: str, weight: float):
    """A 0%-weighted sleeve contributes no trades to a blend (it never
    actually receives any capital in that mix); a nonzero-weighted sleeve
    contributes its full trade count unchanged -- trade COUNT/timing is
    insensitive to capital scale, only dollar sizes are (see module
    docstring)."""
    val = metrics.get(key)
    if val is None:
        return None
    return int(val) if weight > 0 else 0


def blend_metrics(
    sr_result: BacktestResult,
    sr_metrics: dict,
    mr_result: BacktestResult,
    mr_metrics: dict,
    w_sr: float,
    w_mr: float,
    capital: float,
    benchmark_close: pd.Series | None = None,
) -> dict:
    """Metrics for a hypothetical `w_sr`/`w_mr` capital split, built by
    blending the two already-computed FULL-capital sleeve results (see
    module docstring). `capital` must equal the capital both `sr_result`
    and `mr_result` were actually run at (every caller in this codebase
    satisfies this). Reuses metrics.py's existing equity-curve-based
    metric functions on the blended curve rather than duplicating any of
    that math."""
    equity, window_start, window_end = blend_equity_curve(sr_result, mr_result, w_sr, w_mr)

    metrics: dict = {
        "total_return": total_return(equity, start_capital=capital),
        "cagr": cagr(equity, start_capital=capital),
        "max_drawdown": max_drawdown(equity),
        "max_drawdown_duration_days": max_drawdown_duration_days(equity),
        "sharpe_ratio": sharpe_ratio(equity),
        "sortino_ratio": sortino_ratio(equity),
        "calmar_ratio": calmar_ratio(equity, start_capital=capital),
        "total_turnover": (
            w_sr * sr_metrics.get("total_turnover", 0.0) + w_mr * mr_metrics.get("total_turnover", 0.0)
        ),
        "total_transaction_costs": (
            w_sr * sr_metrics.get("total_transaction_costs", 0.0)
            + w_mr * mr_metrics.get("total_transaction_costs", 0.0)
        ),
        "num_transactions": (
            (_weighted_count(sr_metrics, "num_transactions", w_sr) or 0)
            + (_weighted_count(mr_metrics, "num_transactions", w_mr) or 0)
        ),
        "num_trades": (
            (_weighted_count(sr_metrics, "num_trades", w_sr) or 0)
            + (_weighted_count(mr_metrics, "num_trades", w_mr) or 0)
        ),
        "average_capital_invested_pct": (
            w_sr * sr_metrics.get("average_capital_invested_pct", 0.0)
            + w_mr * mr_metrics.get("average_capital_invested_pct", 0.0)
        ),
    }
    metrics["cost_drag_pct"] = metrics["total_transaction_costs"] / capital if capital else 0.0
    metrics.update(best_worst_month(equity))
    if benchmark_close is not None:
        metrics.update(benchmark_metrics(equity, benchmark_close))
    metrics["window_start"] = window_start
    metrics["window_end"] = window_end
    metrics["equity_curve"] = equity
    return metrics


def rank_allocations(window_metrics: dict[str, dict], key: str) -> dict[str, int]:
    """Rank every allocation label within ONE window by `key`, descending
    (rank 1 = best). Works unmodified for max_drawdown too, since drawdown
    values are negative -- a less-negative (closer to zero) drawdown is
    both the larger raw value and the better outcome, so descending sort
    already puts the best drawdown first."""
    items = [(label, m.get(key)) for label, m in window_metrics.items()]
    items = [(label, v) for label, v in items if v is not None and pd.notna(v)]
    items.sort(key=lambda pair: pair[1], reverse=True)
    return {label: i + 1 for i, (label, _) in enumerate(items)}


def average_ranks(all_window_metrics: dict[str, dict[str, dict]], key: str) -> dict[str, float]:
    """{allocation_label: average rank across all windows} for `key`,
    averaged only over windows where that allocation had a valid value."""
    per_window_ranks = [rank_allocations(wm, key) for wm in all_window_metrics.values()]
    labels = {label for r in per_window_ranks for label in r}
    result = {}
    for label in labels:
        ranks = [r[label] for r in per_window_ranks if label in r]
        result[label] = sum(ranks) / len(ranks) if ranks else float("nan")
    return result


def beats_spy_fraction(
    all_window_metrics: dict[str, dict[str, dict]], spy_metrics_by_window: dict[str, dict], key: str
) -> dict[str, float]:
    """Fraction of windows where each allocation's `key` beats SPY's `key`
    in that same window. "Beats" always means a HIGHER raw value -- for
    max_drawdown that correctly means a smaller-magnitude (less negative,
    better) drawdown, same reasoning as `rank_allocations`."""
    counts: dict[str, int] = {}
    totals: dict[str, int] = {}
    for window_label, alloc_dict in all_window_metrics.items():
        spy_metrics = spy_metrics_by_window.get(window_label, {})
        spy_val = spy_metrics.get(key)
        for label, m in alloc_dict.items():
            val = m.get(key)
            if val is None or spy_val is None or pd.isna(val) or pd.isna(spy_val):
                continue
            totals[label] = totals.get(label, 0) + 1
            if val > spy_val:
                counts[label] = counts.get(label, 0) + 1
    return {label: counts.get(label, 0) / totals[label] for label in totals}


def mean_reversion_tradeoff(all_window_metrics: dict[str, dict[str, dict]]) -> list[dict]:
    """For each window, for every allocation OTHER than 100% sector rotation,
    compares max-drawdown improvement and cost-drag increase relative to
    the 100%-sector baseline -- the core question of whether adding mean
    reversion buys enough drawdown protection to justify its cost drag.

    `worth_it` is a simple heuristic (drawdown improved AND the improvement
    in percentage points exceeds the cost-drag increase in percentage
    points), not a rigorous risk-adjusted verdict -- treat it as a
    conversation-starter for further analysis, not a final answer."""
    baseline_label = ALLOCATION_MIXES[0][0]
    rows = []
    for window_label, alloc_dict in all_window_metrics.items():
        baseline = alloc_dict.get(baseline_label)
        if baseline is None:
            continue
        for label, _w_sr, _w_mr in ALLOCATION_MIXES[1:]:
            m = alloc_dict.get(label)
            if m is None:
                continue
            dd_improvement = m["max_drawdown"] - baseline["max_drawdown"]  # positive = smaller (better) drawdown
            cost_drag_increase = m["cost_drag_pct"] - baseline["cost_drag_pct"]
            rows.append({
                "window": window_label,
                "allocation": label,
                "drawdown_improvement_pts": dd_improvement,
                "cost_drag_increase_pts": cost_drag_increase,
                "worth_it": bool(dd_improvement > 0 and dd_improvement > cost_drag_increase),
            })
    return rows
