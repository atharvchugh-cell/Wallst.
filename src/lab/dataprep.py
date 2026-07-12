"""Sleeve data preparation for the lab engine.

Mirrors tournament._run_stock_plan / _run_sector_plan's data handling STEP FOR
STEP (same functions, same arguments, same order) but stops before running the
backtest -- the lab's synchronized multi-sleeve engine needs the prepared
inputs, not a finished run. Deliberately does NOT refactor tournament.py to
share code: the unmerged paper-trading branch already restructures that
module, and an equivalence test (tests/test_lab_equivalence.py) pins this
mirror to the production path far more strongly than shared plumbing would --
if the two ever diverge on the baseline configuration, the test fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .. import config, data
from ..strategies.base import Strategy
from ..tournament import BROAD_FETCH_START, StrategySpec, TournamentError


@dataclass
class PreparedSleeve:
    """Everything the lab engine needs to walk one sleeve day by day."""

    name: str
    weight: float
    allocated_capital: float
    strategy: Strategy                    # constructed, universe already trimmed; NOT yet prepare()d
    price_data: dict[str, pd.DataFrame]   # cleaned (gap-free) frames incl. signal tickers
    full_calendar: pd.DatetimeIndex       # canonical calendar over the fetch range
    walk_start: pd.Timestamp
    walk_end: pd.Timestamp
    spy_df: pd.DataFrame                  # benchmark frame over this sleeve's fetch range
    warmup_days: int
    pre_drops: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def prepare_sleeve(
    spec: StrategySpec,
    weight: float,
    allocated_capital: float,
    start,
    end,
    refresh_cache: bool,
    universe: list[str] | None = None,
    param_overrides: dict | None = None,
    warmup_override: int | None = None,
) -> PreparedSleeve:
    kwargs = dict(param_overrides or {})
    if spec.uses_stock_universe:
        strategy = spec.factory(universe=universe, **kwargs)
    else:
        strategy = spec.factory(**kwargs)

    if spec.data_plan == "stock":
        return _prepare_stock_plan(
            spec, strategy, weight, allocated_capital, start, end, refresh_cache, warmup_override
        )
    if spec.data_plan == "sector":
        return _prepare_sector_plan(spec, strategy, weight, allocated_capital, start, end, refresh_cache)
    raise TournamentError(f"Unknown data plan {spec.data_plan!r} for strategy {spec.name!r}")


def _prepare_stock_plan(
    spec, strategy, weight, allocated_capital, start, end, refresh_cache, warmup_override
) -> PreparedSleeve:
    warnings: list[str] = []
    warmup_days = warmup_override if warmup_override is not None else spec.warmup_calendar_days
    original_universe_size = len(strategy.universe)

    fetch_tickers = list(strategy.universe) + [
        t for t in strategy.signal_tickers if t != config.BENCHMARK_TICKER
    ]
    price_data, fetch_dropped = data.get_price_data(
        fetch_tickers, start, end,
        warmup_calendar_days=warmup_days,
        hard_fail_on_missing=False, force_refresh=refresh_cache,
    )
    fetch_start = pd.Timestamp(start) - pd.Timedelta(days=warmup_days)
    spy_df = data.get_benchmark_data(fetch_start, end, force_refresh=refresh_cache)
    if config.BENCHMARK_TICKER in strategy.signal_tickers:
        price_data[config.BENCHMARK_TICKER] = spy_df.copy()

    full_calendar = data.build_canonical_calendar(spy_df, fetch_start, end)
    full_calendar, excluded_today = data.exclude_unfinalized_today(full_calendar)
    if excluded_today:
        warnings.append("Excluded today's bar (not finalized at fetch time).")

    present_universe = [t for t in strategy.universe if t in price_data]
    _hist_kept, history_excluded = data.filter_by_sufficient_history(
        price_data, present_universe, start, warmup_days
    )
    for t, _reason in history_excluded:
        price_data.pop(t, None)

    calendar_in_range = full_calendar[
        (full_calendar >= pd.Timestamp(start)) & (full_calendar <= pd.Timestamp(end))
    ]
    gap_dropped: list[tuple[str, str]] = []
    clean_price_data = {}
    for ticker, df in price_data.items():
        gaps = data.find_gaps(df, calendar_in_range)
        if len(gaps) > 0:
            if ticker in strategy.signal_tickers:
                raise data.FetchError(
                    f"Signal ticker {ticker} (required by {strategy.name}) has "
                    f"{len(gaps)} gap day(s) in the active range (e.g. {gaps[0].date()}); "
                    f"refusing to run with a degraded regime signal."
                )
            gap_dropped.append((ticker, f"{len(gaps)} gap day(s) in active range, e.g. {gaps[0].date()}"))
            continue
        clean_price_data[ticker] = df

    strategy.universe = [t for t in strategy.universe if t in clean_price_data]
    pre_drops = list(fetch_dropped) + history_excluded + gap_dropped

    min_required = max(1, int(original_universe_size * config.MIN_MEAN_REVERSION_UNIVERSE_FRACTION))
    if len(strategy.universe) < min_required:
        raise data.FetchError(
            f"Only {len(strategy.universe)}/{original_universe_size} {strategy.name} tickers have "
            f"usable data (minimum required: {min_required}, "
            f"{config.MIN_MEAN_REVERSION_UNIVERSE_FRACTION:.0%} of configured universe). Refusing "
            f"to run on a degraded universe. Dropped: {pre_drops}"
        )

    effective_end = full_calendar[-1] if len(full_calendar) else pd.Timestamp(end)
    return PreparedSleeve(
        name=spec.name, weight=weight, allocated_capital=allocated_capital,
        strategy=strategy, price_data=clean_price_data, full_calendar=full_calendar,
        walk_start=pd.Timestamp(start), walk_end=pd.Timestamp(effective_end),
        spy_df=spy_df, warmup_days=warmup_days, pre_drops=pre_drops, warnings=warnings,
    )


def _prepare_sector_plan(spec, strategy, weight, allocated_capital, start, end, refresh_cache) -> PreparedSleeve:
    warnings: list[str] = []
    fetch_tickers = list(strategy.universe) + [
        t for t in strategy.signal_tickers if t != config.BENCHMARK_TICKER
    ]
    price_data, _ = data.get_price_data(
        fetch_tickers, BROAD_FETCH_START, end,
        warmup_calendar_days=0, hard_fail_on_missing=True, force_refresh=refresh_cache,
    )
    spy_df = data.get_benchmark_data(BROAD_FETCH_START, end, force_refresh=refresh_cache)
    if config.BENCHMARK_TICKER in strategy.signal_tickers:
        price_data[config.BENCHMARK_TICKER] = spy_df.copy()

    universe_frames = {t: price_data[t] for t in strategy.universe}
    effective_start, _first_dates = data.compute_sector_effective_start(
        universe_frames, start, getattr(strategy, "lookback_months", config.SECTOR_LOOKBACK_MONTHS)
    )
    if effective_start > pd.Timestamp(start):
        warnings.append(
            f"Requested start {pd.Timestamp(start).date()} predates full sector history; "
            f"clipped to effective start {effective_start.date()}."
        )

    full_calendar = data.build_canonical_calendar(spy_df, BROAD_FETCH_START, end)
    full_calendar, excluded_today = data.exclude_unfinalized_today(full_calendar)
    if excluded_today:
        warnings.append("Excluded today's bar (not finalized at fetch time).")

    calendar_in_range = full_calendar[
        (full_calendar >= effective_start) & (full_calendar <= pd.Timestamp(end))
    ]
    for ticker, df in price_data.items():
        gaps = data.find_gaps(df, calendar_in_range)
        if len(gaps) > 0:
            raise data.FetchError(
                f"{strategy.name} requires a complete history for {ticker}, but found "
                f"{len(gaps)} gap day(s) in its active range (e.g. {gaps[0].date()})."
            )

    effective_end = full_calendar[-1] if len(full_calendar) else pd.Timestamp(end)
    return PreparedSleeve(
        name=spec.name, weight=weight, allocated_capital=allocated_capital,
        strategy=strategy, price_data=price_data, full_calendar=full_calendar,
        walk_start=pd.Timestamp(effective_start), walk_end=pd.Timestamp(effective_end),
        spy_df=spy_df, warmup_days=0, pre_drops=[], warnings=warnings,
    )
