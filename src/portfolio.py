"""Portfolio-combination backtest: allocate ONE account across several
tournament strategies by fixed weight, run each as a fully independent sleeve
with its proportional share of the starting capital, and combine their daily
equity curves into one total portfolio equity curve over the common date
intersection.

Static allocation (v1), by explicit design:
  - Capital is allocated ONCE at the start (weight * total capital per sleeve).
  - Sleeve weights then DRIFT with performance -- a sleeve that outperforms
    becomes a larger share of the portfolio over time.
  - Cash is NEVER transferred between sleeves during the backtest; there is no
    periodic rebalancing back to the target weights. Each sleeve is a fully
    independent engine run (its own cash/shares/lots), exactly like
    `--strategy both`'s two sleeves -- there is no shared cash pool and no
    capital is ever double-counted.
This is disclosed prominently in the report (see reporting.write_portfolio_report).

How this differs from the other multi-strategy modes:
  - `--strategy tournament`: every strategy runs at the FULL capital and they
    are compared side by side (alternatives for one account, NOT a combined
    portfolio).
  - `--strategy both` / `compare`: a fixed 50/50 split of exactly the two
    ORIGINAL strategies (mean_reversion + sector_rotation).
  - `--strategy portfolio` (this module): an arbitrary weighted split across
    any registered tournament strategies, summed into one portfolio curve.

Diagnostics only -- nothing here changes any strategy's thresholds,
transaction assumptions, or the behavior of any pre-existing CLI mode. Each
sleeve runs through tournament.run_tournament_sleeve, the same runner the
tournament uses (regression-tested to reproduce the incumbent standalone
runners' equity curves exactly).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import data
from .engine import BacktestResult
from .metrics import compute_all_metrics, spy_standalone_metrics
from .tournament import STRATEGY_REGISTRY, StrategySpec, run_tournament_sleeve

# The default portfolio requested for this feature: a $15k account allocated
# 60% momentum / 35% sector_rotation / 5% regime_switch. Kept as an ordered
# list (not a dict) so the report preserves the user's stated order.
DEFAULT_PORTFOLIO_WEIGHTS: list[tuple[str, float]] = [
    ("momentum", 0.60),
    ("sector_rotation", 0.35),
    ("regime_switch", 0.05),
]

# Weights must sum to 1.0 within this absolute tolerance (floating-point slack
# for values like 0.1+0.2). Anything further off is a user error, not rounding.
WEIGHT_SUM_TOLERANCE = 1e-6


class PortfolioError(Exception):
    pass


@dataclass
class SleeveAllocation:
    """One strategy sleeve in the portfolio: its target weight, the capital it
    was allocated at the start, its independent backtest result/metrics, and
    (filled in after the sleeves are combined) its final value, ending weight,
    and dollar contribution to the portfolio's total P&L -- all measured at the
    portfolio's common window end."""

    strategy: str
    weight: float
    allocated_capital: float
    result: BacktestResult
    metrics: dict
    final_value: float = 0.0        # sleeve equity at the common window end
    ending_weight: float = 0.0      # final_value / portfolio final equity
    pnl_contribution: float = 0.0   # final_value - allocated_capital (sums to portfolio P&L)
    cost_contribution: float = 0.0  # transaction costs paid within the common window


@dataclass
class PortfolioResult:
    weights: list[tuple[str, float]]
    total_capital: float
    sleeves: list[SleeveAllocation]
    combined_result: BacktestResult
    metrics: dict
    spy_metrics: dict
    common_start: pd.Timestamp
    common_end: pd.Timestamp
    spy_range: str
    skipped_zero_weight: list[str] = field(default_factory=list)


# --- Weight parsing & validation -----------------------------------------------

def parse_portfolio_weights(spec: str | None) -> list[tuple[str, float]]:
    """Parse a `--portfolio-weights` string like
    'momentum=0.60,sector_rotation=0.35,regime_switch=0.05' into an ordered
    list of (strategy, weight) pairs. `None` -> DEFAULT_PORTFOLIO_WEIGHTS.

    Returns a LIST (not a dict) deliberately: duplicate strategy names must
    survive parsing so validate_portfolio_weights can reject them rather than
    silently collapsing 'momentum=0.6,momentum=0.4' into one entry. Only the
    token SHAPE is checked here (well-formed 'name=number'); the semantic
    checks (existence, non-negativity, sum, duplicates) live in
    validate_portfolio_weights."""
    if spec is None:
        return list(DEFAULT_PORTFOLIO_WEIGHTS)

    pairs: list[tuple[str, float]] = []
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if "=" not in token:
            raise PortfolioError(
                f"Malformed --portfolio-weights entry {token!r}; expected 'strategy=weight' "
                f"(e.g. 'momentum=0.60,sector_rotation=0.35,regime_switch=0.05')."
            )
        name, _sep, weight_str = token.partition("=")
        name = name.strip()
        weight_str = weight_str.strip()
        if not name:
            raise PortfolioError(f"Empty strategy name in --portfolio-weights entry {token!r}.")
        try:
            weight = float(weight_str)
        except ValueError:
            raise PortfolioError(
                f"Non-numeric weight {weight_str!r} in --portfolio-weights entry {token!r}."
            )
        pairs.append((name, weight))

    if not pairs:
        raise PortfolioError("--portfolio-weights was empty; provide at least one 'strategy=weight'.")
    return pairs


def validate_portfolio_weights(
    pairs: list[tuple[str, float]], registry: dict[str, StrategySpec] | None = None
) -> None:
    """Enforce, with clear errors, that: every strategy name is registered; no
    strategy is listed twice; every weight is non-negative; and the weights sum
    to exactly 1.0 within WEIGHT_SUM_TOLERANCE. Raises PortfolioError on the
    first violation found."""
    if registry is None:
        registry = STRATEGY_REGISTRY

    names = [name for name, _ in pairs]

    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise PortfolioError(
            f"Duplicate strategy name(s) in --portfolio-weights: {duplicates}. "
            f"List each strategy at most once."
        )

    unknown = [name for name in names if name not in registry]
    if unknown:
        raise PortfolioError(
            f"Unknown strategy name(s) in --portfolio-weights: {unknown}. "
            f"Registered strategies: {list(registry)}."
        )

    negative = [(name, w) for name, w in pairs if w < 0]
    if negative:
        raise PortfolioError(
            f"Weights must be non-negative; got negative weight(s): {negative}."
        )

    total = sum(w for _, w in pairs)
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise PortfolioError(
            f"Portfolio weights must sum to 1.0 (within {WEIGHT_SUM_TOLERANCE:g}); "
            f"got {total:.6f} from {pairs}. Adjust the weights so they total 100%."
        )


def allocate_capital(pairs: list[tuple[str, float]], total_capital: float) -> list[tuple[str, float]]:
    """Proportional split of `total_capital`: each sleeve gets weight * total.
    Returns (strategy, allocated_capital) pairs in the same order. The sum of
    allocations equals total_capital up to floating-point precision (the
    weights were validated to sum to 1.0)."""
    return [(name, total_capital * weight) for name, weight in pairs]


# --- Combination ----------------------------------------------------------------

def _combine_sleeve_equities(sleeves: list[SleeveAllocation]) -> tuple[pd.Series, pd.DatetimeIndex]:
    """Sum the sleeves' independently-computed daily equity curves over the
    INTERSECTION of their valid date ranges (never an outer join, which would
    imply a sleeve was invested before it actually started). No cash is shared
    or transferred -- this is a pure elementwise sum of already-independent
    curves."""
    common = sleeves[0].result.equity_curve.index
    for sleeve in sleeves[1:]:
        common = common.intersection(sleeve.result.equity_curve.index)
    common = common.sort_values()
    if len(common) == 0:
        raise PortfolioError(
            "No overlapping dates across all sleeves; cannot combine them into one "
            "portfolio curve. (Sleeves' effective ranges differ via warmup/inception "
            "clipping; widen --start/--end so every sleeve is active over a shared span.)"
        )
    combined: pd.Series | None = None
    for sleeve in sleeves:
        aligned = sleeve.result.equity_curve.reindex(common)
        combined = aligned if combined is None else combined + aligned
    return combined, common


def _build_combined_result(
    sleeves: list[SleeveAllocation],
    combined_equity: pd.Series,
    common: pd.DatetimeIndex,
    total_capital: float,
) -> BacktestResult:
    """Assemble a synthetic BacktestResult for the whole portfolio so the
    standard compute_all_metrics machinery can measure it. Records
    (transactions/trades/target_events/positions) are the strategy-tagged union
    of the sleeves', RESTRICTED to the common window -- anything a sleeve did
    before the portfolio's shared window started is not part of what this
    portfolio curve measures, so it must not inflate the portfolio's turnover,
    cost, or trade counts."""
    cw_start, cw_end = common[0], common[-1]

    def _in_window(ts: pd.Timestamp) -> bool:
        return cw_start <= ts <= cw_end

    transactions = [
        tx for s in sleeves for tx in s.result.transactions if _in_window(tx.fill_date)
    ]
    trades = [t for s in sleeves for t in s.result.trades if _in_window(t.date)]
    target_events = [
        e for s in sleeves for e in s.result.target_events if _in_window(e.fill_date)
    ]
    positions = [p for s in sleeves for p in s.result.positions if _in_window(p["date"])]
    dropped = [d for s in sleeves for d in s.result.dropped_tickers]
    universe = [u for s in sleeves for u in s.result.universe]

    return BacktestResult(
        strategy_name="portfolio",
        capital=total_capital,
        start=cw_start,
        end=cw_end,
        equity_curve=combined_equity,
        target_events=target_events,
        transactions=transactions,
        trades=trades,
        positions=positions,
        dropped_tickers=dropped,
        universe=universe,
        cost_bps=sleeves[0].result.cost_bps,
        fractional_shares=sleeves[0].result.fractional_shares,
    )


def run_portfolio(
    pairs: list[tuple[str, float]],
    total_capital: float,
    start,
    end,
    cost_bps: float,
    fractional_shares: bool,
    refresh_cache: bool,
    output_dir: str,
    registry: dict[str, StrategySpec] | None = None,
    mr_universe: list[str] | None = None,
    mr_universe_info: dict | None = None,
    write_sleeve_artifacts: bool = True,
    param_overrides_by_strategy: dict[str, dict] | None = None,
    warmup_overrides_by_strategy: dict[str, int] | None = None,
) -> PortfolioResult:
    """Validate the weights, allocate `total_capital` proportionally, run each
    sleeve independently at its allocated capital, then combine the sleeves'
    daily equity curves into one portfolio curve over their common date
    intersection and compute portfolio-level metrics plus each sleeve's
    contribution.

    A zero-weight sleeve is skipped (it would receive $0 and contribute
    nothing); it is recorded in `skipped_zero_weight` and reported, not
    silently dropped. Stock-plan sleeves receive the pre-resolved
    `mr_universe`/`mr_universe_info`; sector-plan sleeves ignore them (they
    always use the 11 fixed sector ETFs).

    `param_overrides_by_strategy`/`warmup_overrides_by_strategy` (per strategy
    name) are used by walk-forward's optimize mode to run a sleeve with a
    variant's frozen parameters/warmup; when None (the default), every sleeve
    runs with its shipped defaults -- i.e. no parameter selection happens
    here, which is exactly the fixed-parameter baseline."""
    if registry is None:
        registry = STRATEGY_REGISTRY
    validate_portfolio_weights(pairs, registry)
    param_overrides_by_strategy = param_overrides_by_strategy or {}
    warmup_overrides_by_strategy = warmup_overrides_by_strategy or {}

    allocations = allocate_capital(pairs, total_capital)
    sleeves: list[SleeveAllocation] = []
    skipped_zero_weight: list[str] = []
    for (name, weight), (_name, allocated) in zip(pairs, allocations):
        if allocated <= 0:
            skipped_zero_weight.append(name)
            continue
        spec = registry[name]
        run = run_tournament_sleeve(
            spec, start, end, allocated, cost_bps, fractional_shares, refresh_cache, output_dir,
            universe=mr_universe if spec.uses_stock_universe else None,
            universe_info=mr_universe_info if spec.uses_stock_universe else None,
            param_overrides=param_overrides_by_strategy.get(name),
            warmup_override=warmup_overrides_by_strategy.get(name),
            write_artifacts=write_sleeve_artifacts,
        )
        sleeves.append(
            SleeveAllocation(
                strategy=name, weight=weight, allocated_capital=allocated,
                result=run.result, metrics=run.metrics,
            )
        )

    if not sleeves:
        raise PortfolioError("No sleeves with positive capital to run (all weights were zero?).")

    combined_equity, common = _combine_sleeve_equities(sleeves)
    common_start, common_end = common[0], common[-1]
    combined_result = _build_combined_result(sleeves, combined_equity, common, total_capital)

    spy_df = data.get_benchmark_data(common_start, common_end, force_refresh=refresh_cache)
    spy_close = spy_df["Close"]
    spy_close = spy_close[(spy_close.index >= common_start) & (spy_close.index <= common_end)]

    metrics = compute_all_metrics(combined_result, benchmark_close=spy_close)
    spy_metrics = spy_standalone_metrics(spy_close)

    # Per-sleeve contribution, all evaluated at the common window end so the
    # pieces reconcile exactly: sum(final_value) == portfolio final equity,
    # sum(ending_weight) == 1.0, sum(pnl_contribution) == portfolio total P&L.
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

    return PortfolioResult(
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
