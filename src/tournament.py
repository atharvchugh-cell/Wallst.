"""Strategy tournament: run N registered strategies under identical
conditions (same capital, same cost assumption, same requested windows, same
canonical-calendar construction, same benchmark) and report them side by
side, across multiple market-regime windows, with cost- and parameter-
sensitivity probes.

Diagnostics only -- nothing here changes any existing strategy's thresholds,
transaction assumptions, or the behavior of any pre-existing CLI mode. The
two incumbent strategies (`mean_reversion`, `sector_rotation`) are run in
tournament mode through the SAME sleeve-runner functions their standalone
modes use (cli.py passes them in), so their tournament rows are identical to
their standalone results; the generic runner in this module exists for the
newer strategies and for sensitivity re-runs, and is regression-tested to
reproduce the incumbent runners' equity curves exactly.

Anti-overfitting stance (see docs/TOURNAMENT_DESIGN.md):
- Parameter-sensitivity variants are FEW, fixed in code with a written
  rationale each, and NEVER auto-selected -- the report shows dispersion so
  fragility is visible, it does not pick a winner from a sweep.
- The robustness score's formula is disclosed next to every use, and its
  raw components are always printed alongside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from . import config, data
from .engine import BacktestResult, run_backtest
from .metrics import compute_all_metrics, spy_standalone_metrics
from .reporting import write_run_artifacts
from .strategies.base import Strategy
from .strategies.mean_reversion import MeanReversionStrategy
from .strategies.mean_reversion_filtered import FilteredMeanReversionStrategy
from .strategies.momentum import MomentumStrategy
from .strategies.regime_switch import RegimeSwitchStrategy
from .strategies.sector_rotation import SectorRotationStrategy

# Same broad-fetch convention cli.py's sector-rotation runner uses: fetch
# sector-plan histories from well before any plausible window so effective-
# start clipping (ETF inception + lookback) is computed from full history.
BROAD_FETCH_START = "1990-01-01"


@dataclass
class StrategySpec:
    """Registry entry: how to build a strategy and how to feed it data.

    data_plan:
      - "stock": per-window fetch with `warmup_calendar_days` of pre-start
        history; gapped/missing universe tickers are DROPPED with a warning
        (subject to the min-universe-fraction guard); the stock universe
        (--universe default/us_50b/--universe-csv) applies.
      - "sector": broad fetch from BROAD_FETCH_START; ANY missing/gapped
        ticker hard-fails; effective start is clipped to the latest ETF
        inception + lookback months; the stock universe does NOT apply.
    """

    name: str
    factory: Callable[..., Strategy]
    data_plan: str  # "stock" | "sector"
    warmup_calendar_days: int = config.MEAN_REVERSION_WARMUP_CALENDAR_DAYS
    uses_stock_universe: bool = False
    summary: str = ""


STRATEGY_REGISTRY: dict[str, StrategySpec] = {
    "mean_reversion": StrategySpec(
        name="mean_reversion",
        factory=MeanReversionStrategy,
        data_plan="stock",
        warmup_calendar_days=config.MEAN_REVERSION_WARMUP_CALENDAR_DAYS,
        uses_stock_universe=True,
        summary="Baseline RSI-oversold mean reversion (existing strategy, unchanged).",
    ),
    "mean_reversion_filtered": StrategySpec(
        name="mean_reversion_filtered",
        factory=FilteredMeanReversionStrategy,
        data_plan="stock",
        # Needs SPY's 200-day regime SMA valid by the first walk day, which
        # requires more pre-start history than the baseline's indicators.
        warmup_calendar_days=config.REGIME_WARMUP_CALENDAR_DAYS,
        uses_stock_universe=True,
        summary="Baseline mean reversion + SPY>SMA200 regime gate + falling-knife guard "
                "(entry filters only; every inherited threshold unchanged).",
    ),
    "momentum": StrategySpec(
        name="momentum",
        factory=MomentumStrategy,
        data_plan="stock",
        warmup_calendar_days=config.MOMENTUM_WARMUP_CALENDAR_DAYS,
        uses_stock_universe=True,
        summary="Monthly top-5 six-month momentum with 200-day-SMA + absolute-momentum "
                "eligibility filters and a cash fallback.",
    ),
    "sector_rotation": StrategySpec(
        name="sector_rotation",
        factory=SectorRotationStrategy,
        data_plan="sector",
        uses_stock_universe=False,
        summary="Monthly top-K sector-ETF momentum (existing strategy, unchanged).",
    ),
    "regime_switch": StrategySpec(
        name="regime_switch",
        factory=RegimeSwitchStrategy,
        data_plan="sector",
        uses_stock_universe=False,
        summary="Sector rotation while SPY > 200-day SMA; 100% cash otherwise.",
    ),
}


# --- Named regime windows -----------------------------------------------------
# Labels describe consensus market character and were chosen before looking
# at any tournament result (see docs/TOURNAMENT_DESIGN.md section 5). ETF
# history (XLC inception 2018 + 3-month lookback) bounds how early a sector-
# plan window can start.
REGIME_WINDOWS: list[tuple[str, str, str]] = [
    ("2019_bull", "2019-01-01", "2019-12-31"),
    ("2020_covid", "2020-01-01", "2020-12-31"),
    ("2021_bull", "2021-01-01", "2021-12-31"),
    ("2022_bear", "2022-01-01", "2022-12-31"),
    ("2023_2024_bull", "2023-01-01", "2024-12-31"),
    ("2019_2024_full", "2019-01-01", "2024-12-31"),
]


def parse_tournament_windows(spec: str | None, start, end) -> list[tuple[str, str, str]]:
    """None -> one window spanning the run's --start/--end. "regimes" -> the
    named REGIME_WINDOWS presets. Otherwise "start:end,start:end,..." with
    auto-generated year-range labels (same syntax --robustness-windows uses)."""
    if not spec:
        return [("full", str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date()))]
    if spec.strip().lower() == "regimes":
        return list(REGIME_WINDOWS)
    windows = []
    for chunk in spec.split(","):
        start_str, end_str = chunk.strip().split(":")
        start_str, end_str = start_str.strip(), end_str.strip()
        label = f"{pd.Timestamp(start_str).year}-{pd.Timestamp(end_str).year}"
        windows.append((label, start_str, end_str))
    return windows


# --- Parameter-sensitivity variants -------------------------------------------
# (variant_label, constructor overrides, rationale, warmup_calendar_days
# override or None). Deliberately FEW per strategy, fixed here with a written
# rationale, and never auto-selected -- the sensitivity report shows how much
# results move when a knob is nudged; it does not pick the best variant.
PARAM_SENSITIVITY_VARIANTS: dict[str, list[tuple[str, dict, str, int | None]]] = {
    "mean_reversion": [
        ("rsi_entry_30", {"rsi_entry": 30.0},
         "canonical RSI oversold threshold; the shipped 35 was tuned mid-development", None),
        ("rsi_entry_25", {"rsi_entry": 25.0}, "stricter oversold requirement", None),
        ("max_holding_20", {"max_holding_days": 20}, "original pre-tuning holding period", None),
    ],
    "mean_reversion_filtered": [
        ("knife_10pct", {"knife_return_threshold": -0.10}, "stricter falling-knife guard", None),
        ("knife_20pct", {"knife_return_threshold": -0.20}, "looser falling-knife guard", None),
        ("regime_sma_150", {"regime_sma_period": 150}, "faster regime filter", None),
        ("regime_sma_250", {"regime_sma_period": 250}, "slower regime filter", None),
    ],
    "momentum": [
        ("lookback_3m", {"lookback_trading_days": 63}, "shorter canonical momentum horizon", None),
        # A 252-day lookback stacked on the 200-day SMA needs ~650 calendar
        # days of pre-start history -- more than the default 500-day warmup.
        ("lookback_12m", {"lookback_trading_days": 252}, "longer canonical momentum horizon", 700),
        ("top_k_3", {"top_k": 3}, "more concentrated portfolio", None),
        ("top_k_7", {"top_k": 7}, "more diversified portfolio", None),
    ],
    "sector_rotation": [
        ("top_k_3", {"top_k": 3}, "original pre-tuning value; the shipped 2 was tuned", None),
        ("lookback_6m", {"lookback_months": 6}, "canonical alternative momentum horizon", None),
    ],
    "regime_switch": [
        ("regime_sma_150", {"regime_sma_period": 150}, "faster regime filter", None),
        ("regime_sma_250", {"regime_sma_period": 250}, "slower regime filter", None),
    ],
}


class TournamentError(Exception):
    pass


@dataclass
class SleeveRun:
    """One strategy run in one window: everything the report needs."""

    strategy: str
    window_label: str
    result: BacktestResult
    metrics: dict
    run_config: dict
    run_dir: object | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class PreparedSleeveData:
    """The fully validated inputs a sleeve's engine walk needs: clean price
    frames, the canonical calendar, the (possibly clipped) effective start, the
    benchmark frame, and the audit bookkeeping (dropped tickers, warnings,
    cache summary). Produced once by prepare_stock_plan/prepare_sector_plan and
    consumed by BOTH the tournament sleeve runners and the paper-trading driver,
    so paper can never diverge from backtest in how data is fetched, gap-checked,
    or history-filtered."""

    clean_price_data: dict[str, pd.DataFrame]
    full_calendar: pd.DatetimeIndex
    effective_start: pd.Timestamp
    spy_df: pd.DataFrame
    pre_drops: list[tuple[str, str]]
    warnings: list[str]
    cache_summary: str
    history_excluded: list[tuple[str, str]] = field(default_factory=list)
    first_dates: dict = field(default_factory=dict)


def prepare_stock_plan(
    spec: StrategySpec,
    strategy: Strategy,
    start,
    end,
    refresh_cache: bool,
    warmup_override: int | None = None,
) -> PreparedSleeveData:
    """Fetch + validate a "stock" data plan: per-window warmup fetch, per-window
    listing-history filter, in-range gap drops (signal-ticker gaps hard-fail),
    and the minimum-usable-universe guard. Mutates strategy.universe down to the
    surviving tickers. This is the exact logic the stock sleeve runner used
    inline; extracted verbatim so paper reuses it."""
    warmup_days = warmup_override if warmup_override is not None else spec.warmup_calendar_days
    original_universe_size = len(strategy.universe)
    warnings: list[str] = []

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

    return PreparedSleeveData(
        clean_price_data=clean_price_data,
        full_calendar=full_calendar,
        effective_start=pd.Timestamp(start),
        spy_df=spy_df,
        pre_drops=pre_drops,
        warnings=warnings,
        cache_summary=f"{len(clean_price_data)} tickers used, {len(pre_drops)} dropped",
        history_excluded=history_excluded,
    )


def prepare_sector_plan(
    strategy: Strategy, start, end, refresh_cache: bool,
) -> PreparedSleeveData:
    """Fetch + validate a "sector" data plan: broad fetch from BROAD_FETCH_START,
    hard-fail on any missing/gapped universe ticker, effective start clipped to
    latest ETF inception + lookback. Extracted verbatim from the sector sleeve
    runner so paper reuses it."""
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
    effective_start, first_dates = data.compute_sector_effective_start(
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

    return PreparedSleeveData(
        clean_price_data=price_data,
        full_calendar=full_calendar,
        effective_start=effective_start,
        spy_df=spy_df,
        pre_drops=[],
        warnings=warnings,
        cache_summary=(
            f"{len(price_data)} tickers used, first available dates: "
            + ", ".join(f"{t}={d.date()}" for t, d in first_dates.items())
        ),
        first_dates=first_dates,
    )


# --- Generic sleeve runner ------------------------------------------------------

def run_tournament_sleeve(
    spec: StrategySpec,
    start,
    end,
    capital: float,
    cost_bps: float,
    fractional_shares: bool,
    refresh_cache: bool,
    output_dir: str,
    universe: list[str] | None = None,
    universe_info: dict | None = None,
    param_overrides: dict | None = None,
    warmup_override: int | None = None,
    write_artifacts: bool = True,
) -> SleeveRun:
    """Run one strategy through the same engine/costs/calendar conventions
    the existing sleeve runners use. Mirrors cli.run_mean_reversion_sleeve
    ("stock" plan) / cli.run_sector_rotation_sleeve ("sector" plan) exactly
    -- a regression test asserts the equity curves match those runners --
    so sensitivity re-runs and new-strategy rows are directly comparable to
    the incumbents' standalone results. `write_artifacts=False` is used for
    sensitivity probes (cost/parameter variants), which would otherwise
    write dozens of near-duplicate artifact directories per tournament."""
    kwargs = dict(param_overrides or {})
    if spec.uses_stock_universe:
        strategy = spec.factory(universe=universe, **kwargs)
    else:
        strategy = spec.factory(**kwargs)

    warnings: list[str] = []
    if spec.data_plan == "stock":
        run = _run_stock_plan(
            spec, strategy, start, end, capital, cost_bps, fractional_shares, refresh_cache,
            warmup_override, warnings,
        )
    elif spec.data_plan == "sector":
        run = _run_sector_plan(
            spec, strategy, start, end, capital, cost_bps, fractional_shares, refresh_cache, warnings,
        )
    else:
        raise TournamentError(f"Unknown data plan {spec.data_plan!r} for strategy {spec.name!r}")

    result, metrics, run_config = run
    if universe_info is not None and spec.uses_stock_universe:
        run_config["universe_info"] = universe_info
    run_dir = write_run_artifacts(result, metrics, run_config, output_dir=output_dir) if write_artifacts else None
    return SleeveRun(
        strategy=spec.name, window_label="", result=result, metrics=metrics,
        run_config=run_config, run_dir=run_dir, warnings=warnings,
    )


def _run_stock_plan(
    spec, strategy, start, end, capital, cost_bps, fractional_shares, refresh_cache,
    warmup_override, warnings,
):
    prepared = prepare_stock_plan(spec, strategy, start, end, refresh_cache, warmup_override)
    warnings.extend(prepared.warnings)

    effective_end = prepared.full_calendar[-1] if len(prepared.full_calendar) else pd.Timestamp(end)
    result = run_backtest(
        strategy, prepared.clean_price_data, prepared.full_calendar, start, effective_end,
        capital, cost_bps, fractional_shares,
    )
    result.dropped_tickers = prepared.pre_drops + list(result.dropped_tickers)

    metrics = compute_all_metrics(result, benchmark_close=prepared.spy_df["Close"])
    run_config = _base_run_config(strategy, result, start, end, capital, cost_bps, fractional_shares, warnings)
    run_config["cache_summary"] = prepared.cache_summary
    run_config["num_history_excluded"] = len(prepared.history_excluded)
    run_config["num_universe_window_excluded"] = len(prepared.pre_drops)
    run_config["universe_window_excluded"] = [
        {"ticker": t, "reason": r} for t, r in prepared.pre_drops
    ]
    return result, metrics, run_config


def _run_sector_plan(spec, strategy, start, end, capital, cost_bps, fractional_shares, refresh_cache, warnings):
    prepared = prepare_sector_plan(strategy, start, end, refresh_cache)
    warnings.extend(prepared.warnings)

    effective_end = prepared.full_calendar[-1] if len(prepared.full_calendar) else pd.Timestamp(end)
    result = run_backtest(
        strategy, prepared.clean_price_data, prepared.full_calendar, prepared.effective_start,
        effective_end, capital, cost_bps, fractional_shares,
    )
    metrics = compute_all_metrics(result, benchmark_close=prepared.spy_df["Close"])
    run_config = _base_run_config(strategy, result, start, end, capital, cost_bps, fractional_shares, warnings)
    run_config["cache_summary"] = prepared.cache_summary
    return result, metrics, run_config


def _base_run_config(strategy, result, start, end, capital, cost_bps, fractional_shares, warnings) -> dict:
    import yfinance as yf

    return {
        "strategy": strategy.name,
        "requested_start": str(pd.Timestamp(start).date()),
        "requested_end": str(pd.Timestamp(end).date()),
        "effective_start": str(result.start.date()),
        "effective_end": str(result.end.date()),
        "capital": capital,
        "cost_bps": cost_bps,
        "fractional_shares": fractional_shares,
        "universe": list(strategy.universe),
        "yfinance_version": yf.__version__,
        "warnings": list(warnings),
    }


# --- Cross-window robustness scoring --------------------------------------------

def robustness_components(
    all_window_metrics: dict[str, dict[str, dict]],
    spy_metrics_by_window: dict[str, dict],
) -> dict[str, dict]:
    """Per-strategy robustness components across windows, plus the composite
    `robustness_score = mean(pct_windows_beats_spy_return,
    pct_windows_positive_return)`.

    A strategy that FAILS or is missing in a window it was expected to run
    must NOT come out looking stronger than one that ran everywhere -- a
    strategy that only survived its one easy window should not score 1.0
    while a full-coverage peer that beat SPY in 4 of 6 windows scores 0.67.
    So the fraction denominators are the number of windows the strategy was
    EXPECTED to run (every window that genuinely ran for at least one
    strategy), NOT just the windows it happened to survive. A missing/failed
    window therefore counts as neither a beat nor a positive -- i.e. it
    counts against the score, exactly as a lost window would.

    Windows in which NO strategy produced a result at all (a tournament-level
    data/infrastructure failure, not a strategy-specific one) are excluded
    from the expected set for everyone, so they penalize no one.

    Reported per strategy: `num_windows_ran` (coverage numerator),
    `num_windows_expected` (denominator), `num_missing_windows`, and
    `full_coverage` -- so incomplete coverage is always visible next to the
    score rather than silently baked into a flattering fraction. `num_windows`
    is kept as an alias of `num_windows_ran` for backward compatibility."""
    strategies = sorted({s for wm in all_window_metrics.values() for s in wm})

    def _window_ran(per_strategy: dict) -> bool:
        return any(m is not None and m.get("total_return") is not None for m in per_strategy.values())

    ranked_windows = [w for w, per in all_window_metrics.items() if _window_ran(per)]
    num_expected = len(ranked_windows)
    out: dict[str, dict] = {}
    if num_expected == 0:
        return out

    for strat in strategies:
        total_returns: list[float] = []
        beats_spy = 0
        positive = 0
        ran = 0
        worst_dd = None
        for window_label in ranked_windows:
            m = all_window_metrics[window_label].get(strat)
            if m is None or m.get("total_return") is None:
                # Expected (a peer ran this window) but this strategy did not:
                # a missing/failed window. It contributes to the denominator
                # via num_expected below but adds nothing to beats/positive,
                # so it drags the score down instead of being quietly dropped.
                continue
            ran += 1
            tr = m["total_return"]
            total_returns.append(tr)
            if tr > 0:
                positive += 1
            spy_tr = spy_metrics_by_window.get(window_label, {}).get("total_return")
            if spy_tr is not None and tr > spy_tr:
                beats_spy += 1
            dd = m.get("max_drawdown")
            if dd is not None and (worst_dd is None or dd < worst_dd):
                worst_dd = dd
        if ran == 0:
            # Never ran in any expected window -- still emit a visibly-worst
            # row (score 0, all windows missing) rather than vanishing from
            # the table, so a strategy that failed everywhere can't hide.
            out[strat] = {
                "num_windows": 0, "num_windows_ran": 0, "num_windows_expected": num_expected,
                "num_missing_windows": num_expected, "full_coverage": False,
                "pct_windows_beats_spy_return": 0.0, "pct_windows_positive_return": 0.0,
                "worst_window_max_drawdown": None, "return_dispersion": 0.0,
                "robustness_score": 0.0,
            }
            continue
        beats_frac = beats_spy / num_expected
        pos_frac = positive / num_expected
        dispersion = float(pd.Series(total_returns).std(ddof=0)) if len(total_returns) > 1 else 0.0
        out[strat] = {
            "num_windows": ran,  # backward-compat alias of num_windows_ran
            "num_windows_ran": ran,
            "num_windows_expected": num_expected,
            "num_missing_windows": num_expected - ran,
            "full_coverage": ran == num_expected,
            "pct_windows_beats_spy_return": beats_frac,
            "pct_windows_positive_return": pos_frac,
            "worst_window_max_drawdown": worst_dd,
            "return_dispersion": dispersion,
            "robustness_score": (beats_frac + pos_frac) / 2.0,
        }
    return out


# --- SPY row helper --------------------------------------------------------------

def spy_row_for_window(
    results_by_strategy: dict[str, BacktestResult], refresh_cache: bool = False
) -> tuple[dict, str] | None:
    """SPY buy-and-hold metrics over the INTERSECTION of every strategy's
    effective range in this window -- one common yardstick per window,
    labeled with the range it actually covers (strategies' own effective
    ranges can differ via warmup/inception clipping; their per-strategy
    benchmark fields still use their own ranges)."""
    if not results_by_strategy:
        return None
    starts = [r.start for r in results_by_strategy.values()]
    ends = [r.end for r in results_by_strategy.values()]
    common_start, common_end = max(starts), min(ends)
    if common_start >= common_end:
        return None
    spy_df = data.get_benchmark_data(common_start, common_end, force_refresh=refresh_cache)
    spy_close = spy_df["Close"]
    spy_close = spy_close[(spy_close.index >= common_start) & (spy_close.index <= common_end)]
    if len(spy_close) < 2:
        return None
    return spy_standalone_metrics(spy_close), f"{common_start.date()} to {common_end.date()}"
