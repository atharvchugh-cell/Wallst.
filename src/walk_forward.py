"""Walk-forward / out-of-sample validation of the portfolio strategy.

The question this answers: does the 60/35/5 (or any weighted) portfolio stay
credible when it is only ever *evaluated* on periods that were not used to
choose its parameters? The engine splits the requested span into a sequence of
(train, test) folds and evaluates the portfolio on each fold's TEST period,
carrying capital forward from one test period to the next, then stitches the
test-period equity curves into one continuous out-of-sample curve.

Two modes:
  - **fixed** (v1, default): the portfolio's shipped, fixed parameters are
    evaluated on each test period. No parameters are selected or tuned, so
    there is nothing to overfit -- this is the clean baseline. The training
    window is still reported (and its dates enforced to end before the test
    period), but in fixed mode it does not influence anything.
  - **optimize** (optional): for each fold, each sleeve's small, PREDEFINED
    sensitivity variants (from tournament.PARAM_SENSITIVITY_VARIANTS -- a
    handful per strategy, each with a written rationale, never a free sweep)
    are ranked on the TRAINING period alone, the best is FROZEN, and only then
    is the test period run with it. No test-period data can touch the
    selection.

Non-negotiable no-lookahead property: a fold's training window always ends
strictly before its test window begins (`train_end < test_start`), and each
fold is an independent run whose only dependency on other folds is the
carried-forward starting capital from EARLIER folds -- so future test data can
never change an earlier fold's selection or result.

Survivorship caveat (see docs/RED_TEAM.md A1): walk-forward validation fixes
in-sample *parameter* overfitting; it does NOT fix the stock universe being a
CURRENT-snapshot, survivorship-biased list. Both universes here are today's
survivors, not point-in-time constituents, in every fold -- so a clean
walk-forward result is still not evidence of a general, tradable edge.

Research / paper-trading only. Nothing here places live orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import data
from .metrics import cagr, max_drawdown, sharpe_ratio, total_return
from .portfolio import PortfolioError, run_portfolio, validate_portfolio_weights
from .tournament import PARAM_SENSITIVITY_VARIANTS, STRATEGY_REGISTRY, StrategySpec, run_tournament_sleeve

DEFAULT_TRAIN_YEARS = 3
DEFAULT_TEST_YEARS = 1
DEFAULT_STEP_YEARS = 1


class WalkForwardError(Exception):
    pass


@dataclass
class Fold:
    """One walk-forward fold: a training window that ends strictly before its
    test window, the capital carried into the test period, the portfolio's
    result on the TEST period, and (optimize mode only) the variant frozen for
    each sleeve from the training period."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    start_capital: float
    portfolio_result: object  # PortfolioResult for the TEST period
    selected_variants: dict[str, str]  # strategy -> variant label ("fixed" when not optimizing)

    @property
    def final_equity(self) -> float:
        return float(self.portfolio_result.combined_result.equity_curve.iloc[-1])

    @property
    def test_return(self) -> float:
        return self.portfolio_result.metrics.get("total_return")

    @property
    def spy_return(self) -> float:
        return self.portfolio_result.spy_metrics.get("total_return")

    @property
    def excess_return(self) -> float:
        return self.portfolio_result.metrics.get("excess_return")

    @property
    def max_drawdown(self) -> float:
        return self.portfolio_result.metrics.get("max_drawdown")

    @property
    def sharpe_ratio(self) -> float:
        return self.portfolio_result.metrics.get("sharpe_ratio")

    @property
    def transaction_costs(self) -> float:
        return self.portfolio_result.metrics.get("total_transaction_costs")

    @property
    def actual_test_range(self) -> str:
        idx = self.portfolio_result.combined_result.equity_curve.index
        return f"{idx[0].date()} to {idx[-1].date()}"


@dataclass
class WalkForwardResult:
    weights: list[tuple[str, float]]
    total_capital: float
    optimize: bool
    window_mode: str  # "expanding" | "rolling"
    train_years: int
    test_years: int
    step_years: int
    folds: list[Fold]
    stitched_equity: pd.Series
    aggregate: dict
    skipped_final_partial: tuple[str, str] | None = None


# --- Fold generation ------------------------------------------------------------

def generate_folds(
    start,
    end,
    train_years: int = DEFAULT_TRAIN_YEARS,
    test_years: int = DEFAULT_TEST_YEARS,
    step_years: int = DEFAULT_STEP_YEARS,
    expanding: bool = True,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Return a list of (train_start, train_end, test_start, test_end) folds.

    The first test period starts `train_years` after `start`; each subsequent
    fold's test period starts `step_years` later. `train_end` is always the day
    BEFORE `test_start`, so training never overlaps testing. Expanding mode
    anchors every fold's training window at `start` (growing training set);
    rolling mode uses a fixed `train_years`-wide window ending at `train_end`.
    A trailing fold whose test window would run past `end` is dropped (only
    complete test years are evaluated)."""
    if train_years <= 0 or test_years <= 0 or step_years <= 0:
        raise WalkForwardError("train/test/step years must all be positive integers.")
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    folds = []
    i = 0
    while True:
        test_start = start + pd.DateOffset(years=train_years + i * step_years)
        test_end = test_start + pd.DateOffset(years=test_years) - pd.DateOffset(days=1)
        if test_end > end:
            break
        train_end = test_start - pd.DateOffset(days=1)
        train_start = start if expanding else (test_start - pd.DateOffset(years=train_years))
        folds.append((train_start, train_end, test_start, test_end))
        i += 1
    return folds


# --- Parameter selection (optimize mode only) -----------------------------------

def _select_variant_for_sleeve(
    spec: StrategySpec,
    train_start,
    train_end,
    capital: float,
    cost_bps: float,
    fractional_shares: bool,
    refresh_cache: bool,
    output_dir: str,
    universe: list[str] | None,
    universe_info: dict | None,
) -> tuple[str, dict, int | None]:
    """Rank a sleeve's shipped defaults ('baseline') against its PREDEFINED
    sensitivity variants ON THE TRAINING WINDOW ONLY, and return the winner's
    (label, param_overrides, warmup_override). Ranking key: training-period
    Sharpe ratio (NaN treated as worst), tie-broken by training total return.
    Deliberately a small fixed menu, never a free parameter sweep. A variant
    that errors on the training window is skipped, not allowed to win."""
    candidates: list[tuple[str, dict | None, int | None]] = [("baseline", None, None)]
    for variant_label, overrides, _rationale, warmup in PARAM_SENSITIVITY_VARIANTS.get(spec.name, []):
        candidates.append((variant_label, overrides, warmup))

    best = None  # (sharpe_key, return_key, label, overrides, warmup)
    for label, overrides, warmup in candidates:
        try:
            run = run_tournament_sleeve(
                spec, train_start, train_end, capital, cost_bps, fractional_shares, refresh_cache,
                output_dir, universe=universe if spec.uses_stock_universe else None,
                universe_info=universe_info if spec.uses_stock_universe else None,
                param_overrides=overrides, warmup_override=warmup, write_artifacts=False,
            )
        except Exception:
            continue
        sharpe = run.metrics.get("sharpe_ratio")
        ret = run.metrics.get("total_return")
        sharpe_key = sharpe if (sharpe is not None and sharpe == sharpe) else float("-inf")
        ret_key = ret if (ret is not None and ret == ret) else float("-inf")
        key = (sharpe_key, ret_key)
        if best is None or key > best[0]:
            best = (key, label, overrides or {}, warmup)

    if best is None:
        # Every candidate failed on the training window -- fall back to shipped
        # defaults rather than inventing a selection.
        return "baseline", {}, None
    return best[1], best[2], best[3]


def _select_variants_for_fold(
    pairs, train_start, train_end, capital, cost_bps, fractional_shares, refresh_cache,
    output_dir, registry, mr_universe, mr_universe_info,
) -> tuple[dict[str, str], dict[str, dict], dict[str, int]]:
    selected_labels: dict[str, str] = {}
    param_overrides: dict[str, dict] = {}
    warmup_overrides: dict[str, int] = {}
    for name, weight in pairs:
        if weight <= 0:
            continue
        spec = registry[name]
        label, overrides, warmup = _select_variant_for_sleeve(
            spec, train_start, train_end, capital, cost_bps, fractional_shares, refresh_cache,
            output_dir, mr_universe, mr_universe_info,
        )
        selected_labels[name] = label
        if overrides:
            param_overrides[name] = overrides
        if warmup is not None:
            warmup_overrides[name] = warmup
    return selected_labels, param_overrides, warmup_overrides


# --- Walk-forward driver --------------------------------------------------------

def run_walk_forward(
    pairs: list[tuple[str, float]],
    total_capital: float,
    start,
    end,
    cost_bps: float,
    fractional_shares: bool,
    refresh_cache: bool,
    output_dir: str,
    train_years: int = DEFAULT_TRAIN_YEARS,
    test_years: int = DEFAULT_TEST_YEARS,
    step_years: int = DEFAULT_STEP_YEARS,
    expanding: bool = True,
    optimize: bool = False,
    registry: dict[str, StrategySpec] | None = None,
    mr_universe: list[str] | None = None,
    mr_universe_info: dict | None = None,
) -> WalkForwardResult:
    """Evaluate the weighted portfolio out-of-sample across walk-forward folds.

    For each fold, the portfolio is run on the fold's TEST period with the
    capital carried forward from the prior fold's ending equity (folds compound
    -- the account is continuous across the stitched out-of-sample span). In
    optimize mode the per-sleeve variant is chosen on the training window
    first and frozen before the test run. The test-period equity curves are
    concatenated into one stitched out-of-sample curve, on which the aggregate
    return/CAGR/max-drawdown are measured."""
    if registry is None:
        registry = STRATEGY_REGISTRY
    validate_portfolio_weights(pairs, registry)

    raw_folds = generate_folds(start, end, train_years, test_years, step_years, expanding)
    if not raw_folds:
        raise WalkForwardError(
            f"No complete walk-forward folds fit in [{pd.Timestamp(start).date()}, "
            f"{pd.Timestamp(end).date()}] with train={train_years}y test={test_years}y "
            f"step={step_years}y. Widen --start/--end or shrink the windows."
        )

    folds: list[Fold] = []
    running_capital = float(total_capital)
    for i, (train_start, train_end, test_start, test_end) in enumerate(raw_folds):
        selected_labels: dict[str, str] = {name: "fixed" for name, w in pairs if w > 0}
        param_overrides: dict[str, dict] = {}
        warmup_overrides: dict[str, int] = {}
        if optimize:
            selected_labels, param_overrides, warmup_overrides = _select_variants_for_fold(
                pairs, train_start, train_end, running_capital, cost_bps, fractional_shares,
                refresh_cache, output_dir, registry, mr_universe, mr_universe_info,
            )

        pf = run_portfolio(
            pairs, running_capital, test_start, test_end, cost_bps, fractional_shares,
            refresh_cache, output_dir, registry=registry, mr_universe=mr_universe,
            mr_universe_info=mr_universe_info, write_sleeve_artifacts=False,
            param_overrides_by_strategy=param_overrides or None,
            warmup_overrides_by_strategy=warmup_overrides or None,
        )
        fold = Fold(
            index=i, train_start=pd.Timestamp(train_start), train_end=pd.Timestamp(train_end),
            test_start=pd.Timestamp(test_start), test_end=pd.Timestamp(test_end),
            start_capital=running_capital, portfolio_result=pf, selected_variants=selected_labels,
        )
        folds.append(fold)
        running_capital = fold.final_equity  # carry forward to the next fold

    stitched_equity = _stitch_equity(folds)
    aggregate = _aggregate_metrics(folds, stitched_equity, total_capital)

    return WalkForwardResult(
        weights=list(pairs), total_capital=total_capital, optimize=optimize,
        window_mode="expanding" if expanding else "rolling",
        train_years=train_years, test_years=test_years, step_years=step_years,
        folds=folds, stitched_equity=stitched_equity, aggregate=aggregate,
    )


def _stitch_equity(folds: list[Fold]) -> pd.Series:
    """Concatenate the folds' TEST-period portfolio equity curves into one
    continuous out-of-sample series. Because each fold was run with the prior
    fold's ending equity as its starting capital, the pieces already line up in
    dollar terms; the test windows are non-overlapping by construction, so the
    concatenation contains ONLY test-period dates and never any training
    date."""
    pieces = [f.portfolio_result.combined_result.equity_curve for f in folds]
    stitched = pd.concat(pieces).sort_index()
    # Defensive: a shared boundary date between adjacent folds should not occur
    # (test windows are disjoint) but if it ever did, keep the first.
    stitched = stitched[~stitched.index.duplicated(keep="first")]
    return stitched


def _aggregate_metrics(folds: list[Fold], stitched_equity: pd.Series, total_capital: float) -> dict:
    num_folds = len(folds)
    beats_spy = sum(
        1 for f in folds
        if f.test_return is not None and f.spy_return is not None and f.test_return > f.spy_return
    )
    profitable = sum(1 for f in folds if f.test_return is not None and f.test_return > 0)
    return {
        "num_folds": num_folds,
        "stitched_total_return": total_return(stitched_equity, start_capital=total_capital),
        "stitched_cagr": cagr(stitched_equity, start_capital=total_capital),
        "stitched_max_drawdown": max_drawdown(stitched_equity),
        "stitched_sharpe_ratio": sharpe_ratio(stitched_equity),
        "final_equity": float(stitched_equity.iloc[-1]) if len(stitched_equity) else total_capital,
        "pct_folds_beating_spy": (beats_spy / num_folds) if num_folds else 0.0,
        "pct_folds_profitable": (profitable / num_folds) if num_folds else 0.0,
        "total_transaction_costs": float(sum((f.transaction_costs or 0.0) for f in folds)),
        "oos_start": str(stitched_equity.index[0].date()) if len(stitched_equity) else None,
        "oos_end": str(stitched_equity.index[-1].date()) if len(stitched_equity) else None,
    }
