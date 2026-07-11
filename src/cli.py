"""CLI orchestration: fetch data, run one or both strategy sleeves, combine
if needed, and write the full audit artifact set."""

from __future__ import annotations

import argparse
import sys
import warnings as warnings_module
from pathlib import Path

import pandas as pd
import yfinance as yf

import shutil

from . import config, data
from . import paper as paper_module
from . import paper_db as paper_db_module
from . import portfolio as portfolio_module
from . import tournament as tournament_module
from . import universe as universe_module
from . import walk_forward as walk_forward_module
from .engine import combine_results, run_backtest
from .metrics import compute_all_metrics, spy_standalone_metrics
from .reporting import (
    write_run_artifacts, write_comparison_report, compute_sleeve_contribution, write_robustness_report,
    write_tournament_report, write_portfolio_report, write_walk_forward_report, write_paper_artifacts,
)
from .robustness import ALLOCATION_MIXES, DEFAULT_ROBUSTNESS_WINDOWS, blend_metrics
from .strategies.mean_reversion import MeanReversionStrategy
from .strategies.sector_rotation import SectorRotationStrategy

SECTOR_BROAD_FETCH_START = "1990-01-01"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest mean-reversion and/or sector-rotation strategies. "
        "Research/educational tool only -- does not place live trades."
    )
    parser.add_argument(
        "--strategy",
        choices=[
            "mean_reversion", "sector_rotation", "both", "compare", "robustness",
            "tournament", "portfolio", "walk_forward", "paper",
        ],
        required=True,
    )
    today = pd.Timestamp.now().normalize()
    default_start = (today - pd.DateOffset(years=config.DEFAULT_START_YEARS_BACK)).date().isoformat()
    parser.add_argument("--start", default=default_start, help="YYYY-MM-DD")
    parser.add_argument("--end", default=today.date().isoformat(), help="YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=config.DEFAULT_CAPITAL)
    parser.add_argument("--cost-bps", type=float, default=config.DEFAULT_COST_BPS)
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--no-fractional-shares", action="store_true")
    parser.add_argument(
        "--compare-years", default=None,
        help="Comma-separated calendar years for --strategy compare's annual-returns table, "
        "e.g. 2022,2023,2024 (default: every calendar year spanned by --start/--end). "
        "Ignored for other --strategy values.",
    )
    parser.add_argument(
        "--robustness-windows", default=None,
        help="Comma-separated start:end windows for --strategy robustness, e.g. "
        "'2019-01-01:2021-12-31,2020-01-01:2022-12-31' (default: the 5 standard windows "
        "2019-2021/2020-2022/2021-2023/2022-2024/2019-2024). Ignored for other --strategy values.",
    )
    parser.add_argument(
        "--tournament-strategies", default=None,
        help="Comma-separated strategy names for --strategy tournament (default: all "
        f"registered: {','.join(tournament_module.STRATEGY_REGISTRY)}). Ignored for "
        "other --strategy values.",
    )
    parser.add_argument(
        "--tournament-windows", default=None,
        help="Windows for --strategy tournament: omit for one window spanning --start/--end; "
        "'regimes' for the named bull/bear/choppy presets (see src/tournament.py); or "
        "explicit 'start:end,start:end,...'. Ignored for other --strategy values.",
    )
    parser.add_argument(
        "--tournament-cost-bps-list", default=None,
        help="Comma-separated per-trade cost levels in bps (e.g. '0,5,10,20') for --strategy "
        "tournament's cost-sensitivity section, each a full re-run of every strategy over "
        "--start/--end. Omit to skip cost sensitivity. Ignored for other --strategy values.",
    )
    parser.add_argument(
        "--tournament-param-sensitivity", action="store_true",
        help="Run --strategy tournament's small, disclosed parameter-sensitivity variants "
        "(see src/tournament.py PARAM_SENSITIVITY_VARIANTS -- a few variants per strategy, "
        "each with a written rationale, NEVER auto-selected). Off by default; adds one "
        "re-run per variant over --start/--end.",
    )
    parser.add_argument(
        "--portfolio-weights", default=None,
        help="Allocation for --strategy portfolio and --strategy walk_forward: "
        "'strategy=weight,...' e.g. 'momentum=0.60,sector_rotation=0.35,regime_switch=0.05'. "
        "Weights must be non-negative and sum to 1.0. Each sleeve gets weight x --capital and "
        "runs as a fully independent, static (non-rebalanced) sleeve. Omit to use the default "
        "60/35/5 momentum/sector_rotation/regime_switch mix. Ignored for other --strategy values.",
    )
    parser.add_argument(
        "--walk-forward-train-years", type=int, default=walk_forward_module.DEFAULT_TRAIN_YEARS,
        help="--strategy walk_forward: length of each fold's training window in years (default 3). "
        "In the default fixed-parameter mode the training window is reported and its dates are "
        "enforced to end before the test period, but it does not influence results. Ignored for "
        "other --strategy values.",
    )
    parser.add_argument(
        "--walk-forward-test-years", type=int, default=walk_forward_module.DEFAULT_TEST_YEARS,
        help="--strategy walk_forward: length of each fold's out-of-sample test window in years "
        "(default 1). Ignored for other --strategy values.",
    )
    parser.add_argument(
        "--walk-forward-step-years", type=int, default=walk_forward_module.DEFAULT_STEP_YEARS,
        help="--strategy walk_forward: how many years to move forward between folds (default 1). "
        "Ignored for other --strategy values.",
    )
    parser.add_argument(
        "--walk-forward-window", choices=["expanding", "rolling"], default="expanding",
        help="--strategy walk_forward: 'expanding' (default) anchors every fold's training window "
        "at --start (growing training set); 'rolling' uses a fixed train-years-wide window that "
        "slides forward. Ignored for other --strategy values.",
    )
    parser.add_argument(
        "--walk-forward-optimize", action="store_true",
        help="--strategy walk_forward: enable the OPTIONAL optimize mode -- for each fold, rank each "
        "sleeve's PREDEFINED sensitivity variants (from tournament.PARAM_SENSITIVITY_VARIANTS, never "
        "a free sweep) on the TRAINING window only, freeze the best per sleeve, then evaluate the test "
        "period with it. Off by default: v1 evaluates the shipped fixed parameters (no selection, no "
        "new overfitting). Ignored for other --strategy values.",
    )
    # --- Forward paper-trading (--strategy paper) --------------------------------
    # Research/paper-trading only: never connects to a broker, never sends a real
    # order. Exactly one action flag is chosen per invocation.
    paper_group = parser.add_argument_group("paper trading (--strategy paper)")
    paper_group.add_argument(
        "--paper-state-dir", default=paper_module.DEFAULT_PAPER_STATE_DIR,
        help="Directory holding the persistent paper account (default 'paper_state').",
    )
    paper_group.add_argument(
        "--paper-init", action="store_true",
        help="Initialize a new paper account: split --capital across --portfolio-weights "
        "(default 60/35/5), freeze the universe snapshot, and write an all-cash starting ledger. "
        "Inception is --paper-start (default today).",
    )
    paper_group.add_argument(
        "--paper-start", default=None,
        help="Paper account inception date (YYYY-MM-DD) for --paper-init. Default: today. "
        "Pass a past date to enable deterministic historical multi-day simulation via --paper-date.",
    )
    paper_group.add_argument(
        "--paper-run", action="store_true",
        help="Process the next unprocessed finalized market session (advance one day).",
    )
    paper_group.add_argument(
        "--paper-date", default=None,
        help="Process forward through this finalized session (YYYY-MM-DD), one session at a time, "
        "enforcing the information boundary that existed on each date. Rejects future/unfinalized dates.",
    )
    paper_group.add_argument("--paper-status", action="store_true", help="Print the current account status.")
    paper_group.add_argument("--paper-orders", action="store_true", help="List pending (and stale) orders.")
    paper_group.add_argument("--paper-trades", action="store_true", help="List completed paper trades (fills).")
    paper_group.add_argument(
        "--paper-export", action="store_true",
        help="Export a timestamped copy of all ledger artifacts to --output-dir.",
    )
    paper_group.add_argument(
        "--paper-reconcile", action="store_true",
        help="Re-verify the persisted state's accounting invariants (no replay).",
    )
    paper_group.add_argument(
        "--paper-reset", action="store_true",
        help="Reset the account. Requires --confirm-paper-reset; backs up state first, never deletes silently.",
    )
    paper_group.add_argument(
        "--confirm-paper-reset", action="store_true",
        help="Explicit confirmation required by --paper-reset.",
    )
    parser.add_argument(
        "--universe",
        choices=["default", "us_50b"],
        default="default",
        help="Mean-reversion universe to use. 'default' (default) is the existing hardcoded "
        "~25-stock universe -- old results stay reproducible. 'us_50b' is a current-snapshot "
        "universe of US-listed common stock with market cap >= $50B, built in bulk via Yahoo "
        "Finance's market-cap screener (see src/universe.py; still survivorship-biased and NOT "
        "a point-in-time historical constituent list). Ignored by sector_rotation (always the "
        "11 fixed sector ETFs).",
    )
    parser.add_argument(
        "--universe-csv", default=None,
        help="Path to a CSV with a 'ticker' column (optionally also name/market_cap/exchange/"
        "snapshot_date, the same schema data_cache/universe_us_50b.csv uses) to use as the "
        "mean-reversion universe. Overrides --universe when given.",
    )
    parser.add_argument(
        "--refresh-universe", action="store_true",
        help="Rebuild the 'us_50b' universe from live data instead of using the cached "
        "data_cache/universe_us_50b.csv snapshot, if present. Ignored unless --universe us_50b "
        "is also set.",
    )
    parser.add_argument(
        "--max-universe-candidates", type=int, default=None,
        help="Debug/safety cap on how many us_50b screener results to consider. NOT applied by "
        "default -- omit this for a real run, which considers every qualifying result. Mainly "
        "useful for fast, bounded dev/test runs. Ignored unless --universe us_50b --refresh-universe.",
    )
    parser.add_argument(
        "--universe-allow-screener-only", action="store_true",
        help="Debug escape hatch: if the Nasdaq Trader symbol directories can't be fetched during "
        "a us_50b --refresh-universe build, proceed with screener-only results instead of hard-"
        "failing. NOT the default -- Nasdaq Trader is normally a REQUIRED eligibility gate on "
        "screener results, since the screener alone can return non-US-listed/non-tradable symbols.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if end < start:
        raise ValueError(f"--end ({end.date()}) is before --start ({start.date()})")
    if args.capital <= 0:
        raise ValueError(f"--capital must be > 0, got {args.capital}")
    if args.cost_bps < 0:
        raise ValueError(f"--cost-bps must be >= 0, got {args.cost_bps}")


def run_mean_reversion_sleeve(
    start, end, capital, cost_bps, fractional_shares, refresh_cache, output_dir,
    universe=None, universe_info=None,
):
    warnings: list[str] = []
    strat = MeanReversionStrategy(universe=universe)
    original_universe_size = len(strat.universe)

    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        price_data, fetch_dropped = data.get_price_data(
            strat.universe, start, end,
            warmup_calendar_days=config.MEAN_REVERSION_WARMUP_CALENDAR_DAYS,
            hard_fail_on_missing=False, force_refresh=refresh_cache,
        )

        fetch_start = pd.Timestamp(start) - pd.Timedelta(days=config.MEAN_REVERSION_WARMUP_CALENDAR_DAYS)
        spy_df = data.get_benchmark_data(fetch_start, end, force_refresh=refresh_cache)
        warnings.extend(f"Data fetch: {w.message}" for w in caught)

    full_calendar = data.build_canonical_calendar(spy_df, fetch_start, end)
    full_calendar, excluded_today = data.exclude_unfinalized_today(full_calendar)
    if excluded_today:
        warnings.append("Excluded today's bar (not finalized at fetch time); effective end shifted back one trading day.")

    # Per-window listing-history filter: exclude any universe ticker that
    # lacks enough history to warm up mean reversion's indicators for THIS
    # window (e.g. a current-$50B name via --universe us_50b that IPO'd after
    # the window start). A NO-OP for the default hardcoded universe, whose
    # members all predate any window -- so default-universe behavior is
    # unchanged -- and only bites survivorship-biased snapshot universes.
    present_universe = [t for t in strat.universe if t in price_data]
    _hist_kept, history_excluded = data.filter_by_sufficient_history(
        price_data, present_universe, start, config.MEAN_REVERSION_WARMUP_CALENDAR_DAYS
    )
    for t, _reason in history_excluded:
        price_data.pop(t, None)

    calendar_in_range = full_calendar[(full_calendar >= pd.Timestamp(start)) & (full_calendar <= pd.Timestamp(end))]
    gap_dropped: list[tuple[str, str]] = []
    clean_price_data = {}
    for ticker, df in price_data.items():
        gaps = data.find_gaps(df, calendar_in_range)
        if len(gaps) > 0:
            gap_dropped.append((ticker, f"{len(gaps)} gap day(s) in active range, e.g. {gaps[0].date()}"))
            continue
        clean_price_data[ticker] = df

    strat.universe = [t for t in strat.universe if t in clean_price_data]
    pre_drops = list(fetch_dropped) + history_excluded + gap_dropped

    min_required = max(1, int(original_universe_size * config.MIN_MEAN_REVERSION_UNIVERSE_FRACTION))
    if len(strat.universe) < min_required:
        raise data.FetchError(
            f"Only {len(strat.universe)}/{original_universe_size} mean-reversion tickers have usable "
            f"data (minimum required: {min_required}, "
            f"{config.MIN_MEAN_REVERSION_UNIVERSE_FRACTION:.0%} of configured universe). Refusing to run "
            f"on a degraded universe rather than silently produce a thin/no-trade backtest. "
            f"Dropped: {pre_drops}"
        )

    effective_end = full_calendar[-1] if len(full_calendar) else pd.Timestamp(end)
    result = run_backtest(
        strat, clean_price_data, full_calendar, start, effective_end, capital, cost_bps, fractional_shares
    )
    result.dropped_tickers = pre_drops + list(result.dropped_tickers)

    metrics = compute_all_metrics(result, benchmark_close=spy_df["Close"])
    run_config = {
        "strategy": "mean_reversion",
        "requested_start": str(pd.Timestamp(start).date()),
        "requested_end": str(pd.Timestamp(end).date()),
        "effective_start": str(result.start.date()),
        "effective_end": str(result.end.date()),
        "capital": capital,
        "cost_bps": cost_bps,
        "fractional_shares": fractional_shares,
        "universe": list(strat.universe),
        "yfinance_version": yf.__version__,
        "warnings": warnings,
        "cache_summary": f"{len(clean_price_data)} tickers used, {len(pre_drops)} dropped",
    }
    if universe_info is not None:
        run_config["universe_info"] = universe_info
    run_dir = write_run_artifacts(result, metrics, run_config, output_dir=output_dir)
    return result, metrics, run_config, run_dir


def run_sector_rotation_sleeve(start, end, capital, cost_bps, fractional_shares, refresh_cache, output_dir):
    warnings: list[str] = []
    strat = SectorRotationStrategy()

    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        price_data, _ = data.get_price_data(
            strat.universe, SECTOR_BROAD_FETCH_START, end,
            warmup_calendar_days=0, hard_fail_on_missing=True, force_refresh=refresh_cache,
        )
        spy_df = data.get_benchmark_data(SECTOR_BROAD_FETCH_START, end, force_refresh=refresh_cache)
        warnings.extend(f"Data fetch: {w.message}" for w in caught)

    effective_start, first_dates = data.compute_sector_effective_start(
        price_data, start, config.SECTOR_LOOKBACK_MONTHS
    )
    if effective_start > pd.Timestamp(start):
        warnings.append(
            f"Requested start {pd.Timestamp(start).date()} predates full sector history; "
            f"clipped to effective start {effective_start.date()} "
            f"(latest ETF inception + {config.SECTOR_LOOKBACK_MONTHS} months)."
        )

    full_calendar = data.build_canonical_calendar(spy_df, SECTOR_BROAD_FETCH_START, end)
    full_calendar, excluded_today = data.exclude_unfinalized_today(full_calendar)
    if excluded_today:
        warnings.append("Excluded today's bar (not finalized at fetch time); effective end shifted back one trading day.")

    calendar_in_range = full_calendar[(full_calendar >= effective_start) & (full_calendar <= pd.Timestamp(end))]
    for ticker, df in price_data.items():
        gaps = data.find_gaps(df, calendar_in_range)
        if len(gaps) > 0:
            raise data.FetchError(
                f"Sector rotation requires a complete history for {ticker}, but found "
                f"{len(gaps)} gap day(s) in its active range (e.g. {gaps[0].date()}). "
                f"This strategy hard-fails rather than silently continuing with incomplete data."
            )

    effective_end = full_calendar[-1] if len(full_calendar) else pd.Timestamp(end)
    result = run_backtest(
        strat, price_data, full_calendar, effective_start, effective_end, capital, cost_bps, fractional_shares
    )

    metrics = compute_all_metrics(result, benchmark_close=spy_df["Close"])
    run_config = {
        "strategy": "sector_rotation",
        "requested_start": str(pd.Timestamp(start).date()),
        "requested_end": str(pd.Timestamp(end).date()),
        "effective_start": str(result.start.date()),
        "effective_end": str(result.end.date()),
        "capital": capital,
        "cost_bps": cost_bps,
        "fractional_shares": fractional_shares,
        "universe": list(strat.universe),
        "yfinance_version": yf.__version__,
        "warnings": warnings,
        "cache_summary": f"{len(price_data)} ETFs used, first available dates: "
        + ", ".join(f"{t}={d.date()}" for t, d in first_dates.items()),
    }
    run_dir = write_run_artifacts(result, metrics, run_config, output_dir=output_dir)
    return result, metrics, run_config, run_dir


def _parse_robustness_windows(spec: str | None) -> list[tuple[str, str, str]]:
    if not spec:
        return list(DEFAULT_ROBUSTNESS_WINDOWS)
    windows = []
    for chunk in spec.split(","):
        start_str, end_str = chunk.strip().split(":")
        start_str, end_str = start_str.strip(), end_str.strip()
        label = f"{pd.Timestamp(start_str).year}-{pd.Timestamp(end_str).year}"
        windows.append((label, start_str, end_str))
    return windows


def resolve_mean_reversion_universe(
    args: argparse.Namespace, backtest_start=None, backtest_end=None,
) -> universe_module.UniverseResolution:
    """Resolve the mean-reversion universe ONCE per run (not once per
    window/sleeve-call) -- a `us_50b` universe is a current snapshot, not a
    per-window computation, so every sleeve call in this run (mean_reversion,
    both, compare, or every window of robustness) shares the exact same
    ticker list. Sector rotation never calls this -- it always uses the 11
    fixed sector ETFs regardless of --universe. Passes `progress=print` so a
    live `--refresh-universe` build prints its candidate/screener-page/
    qualifying-count progress to the console rather than running silently.
    `backtest_start`/`backtest_end` are the actual price-data window a
    `us_50b` universe's final tickers get validated (or, if loaded from a
    cache whose recorded validated window doesn't cover this one,
    re-validated) against -- see src/universe.py. They default to
    `args.start`/`args.end`, but run_robustness passes the UNION of all its
    window start/ends instead, since this function is called ONCE and its
    result shared across every window in that run, not re-resolved per
    window."""
    return universe_module.resolve_mean_reversion_universe(
        mode=args.universe, csv_path=args.universe_csv, refresh=args.refresh_universe,
        max_candidates=args.max_universe_candidates, progress=print,
        allow_screener_only=args.universe_allow_screener_only,
        backtest_start=backtest_start if backtest_start is not None else args.start,
        backtest_end=backtest_end if backtest_end is not None else args.end,
    )


def run_robustness(args: argparse.Namespace) -> int:
    """Diagnostics only: runs mean_reversion and sector_rotation ONCE per
    window (reusing run_mean_reversion_sleeve/run_sector_rotation_sleeve
    unmodified -- same strategy logic, same config.py defaults, same
    per-sleeve artifact writing as every other --strategy mode), then
    blends those two already-computed results into each of the 5 fixed
    allocation mixes plus a SPY row via `src/robustness.py` -- see that
    module's docstring for why blending is used instead of re-running the
    engine once per allocation."""
    fractional_shares = not args.no_fractional_shares
    windows = _parse_robustness_windows(args.robustness_windows)

    try:
        # Validate the shared universe's price data against the UNION of
        # every robustness window (earliest start to latest end), not just
        # args.start/args.end -- the universe is resolved once and reused
        # across all windows below, so it must be checked against the full
        # span any of them actually needs.
        union_start = min(pd.Timestamp(w_start) for _label, w_start, _w_end in windows)
        union_end = max(pd.Timestamp(w_end) for _label, _w_start, w_end in windows)
        mr_universe = resolve_mean_reversion_universe(args, backtest_start=union_start, backtest_end=union_end)
    except (universe_module.UniverseError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    all_window_metrics: dict[str, dict[str, dict]] = {}
    window_ranges: dict[str, str] = {}

    for window_label, w_start, w_end in windows:
        print(f"\n=== Robustness window {window_label}: {w_start} to {w_end} ===")
        try:
            mr_result, mr_metrics, _mr_config, _mr_dir = run_mean_reversion_sleeve(
                w_start, w_end, args.capital, args.cost_bps, fractional_shares,
                args.refresh_cache, args.output_dir,
                universe=mr_universe.tickers, universe_info=mr_universe.info,
            )
            sr_result, sr_metrics, _sr_config, _sr_dir = run_sector_rotation_sleeve(
                w_start, w_end, args.capital, args.cost_bps, fractional_shares,
                args.refresh_cache, args.output_dir,
            )
        except (data.FetchError, ValueError) as e:
            print(f"Error in window {window_label}: {e}", file=sys.stderr)
            continue

        common_index = sr_result.equity_curve.index.intersection(mr_result.equity_curve.index)
        if len(common_index) == 0:
            print(f"Error: no overlapping dates for window {window_label}; skipping.", file=sys.stderr)
            continue

        spy_df = data.get_benchmark_data(common_index[0], common_index[-1], force_refresh=args.refresh_cache)
        spy_close = spy_df["Close"]

        alloc_metrics: dict[str, dict] = {}
        for label, w_sr, w_mr in ALLOCATION_MIXES:
            alloc_metrics[label] = blend_metrics(
                sr_result, sr_metrics, mr_result, mr_metrics, w_sr, w_mr, args.capital, benchmark_close=spy_close,
            )
        alloc_metrics["SPY"] = spy_standalone_metrics(spy_close)

        all_window_metrics[window_label] = alloc_metrics
        window_ranges[window_label] = f"{common_index[0].date()} to {common_index[-1].date()}"
        print(f"[{window_label}] done")

    if not all_window_metrics:
        print("Error: robustness run produced no usable windows.", file=sys.stderr)
        return 1

    run_config = {
        "requested_start": str(pd.Timestamp(args.start).date()),
        "requested_end": str(pd.Timestamp(args.end).date()),
        "capital": args.capital,
        "cost_bps": args.cost_bps,
        "fractional_shares": fractional_shares,
        "windows": [label for label, _s, _e in windows],
    }
    run_dir = write_robustness_report(all_window_metrics, window_ranges, run_config, output_dir=args.output_dir)
    print(f"\n[robustness] artifacts written to: {run_dir}")
    return 0


def run_tournament(args: argparse.Namespace) -> int:
    """Run every requested strategy under identical conditions (same capital,
    cost assumption, windows, calendar construction, benchmark) and write one
    side-by-side tournament report. Each strategy runs at the FULL --capital
    -- the tournament compares alternatives for the same account, it does not
    split capital into simultaneous sleeves (that's --strategy both/compare).

    The two incumbent strategies run through the SAME sleeve runners their
    standalone modes use, so their tournament rows are identical to their
    standalone results; newer strategies and all sensitivity re-runs go
    through tournament.run_tournament_sleeve, which is regression-tested to
    reproduce the incumbent runners' equity curves exactly."""
    fractional_shares = not args.no_fractional_shares
    windows = tournament_module.parse_tournament_windows(args.tournament_windows, args.start, args.end)

    if args.tournament_strategies:
        names = [n.strip() for n in args.tournament_strategies.split(",") if n.strip()]
        unknown = [n for n in names if n not in tournament_module.STRATEGY_REGISTRY]
        if unknown:
            print(
                f"Error: unknown tournament strategies {unknown}; registered: "
                f"{list(tournament_module.STRATEGY_REGISTRY)}", file=sys.stderr,
            )
            return 1
    else:
        names = list(tournament_module.STRATEGY_REGISTRY)

    cost_bps_levels: list[float] = []
    if args.tournament_cost_bps_list:
        try:
            cost_bps_levels = [float(x.strip()) for x in args.tournament_cost_bps_list.split(",")]
        except ValueError:
            print(f"Error: could not parse --tournament-cost-bps-list {args.tournament_cost_bps_list!r}", file=sys.stderr)
            return 1
        if any(b < 0 for b in cost_bps_levels):
            print("Error: --tournament-cost-bps-list values must be >= 0", file=sys.stderr)
            return 1

    needs_stock_universe = any(
        tournament_module.STRATEGY_REGISTRY[n].uses_stock_universe for n in names
    )
    mr_universe = None
    if needs_stock_universe:
        try:
            union_start = min(pd.Timestamp(w_start) for _l, w_start, _e in windows)
            union_end = max(pd.Timestamp(w_end) for _l, _s, w_end in windows)
            mr_universe = resolve_mean_reversion_universe(args, backtest_start=union_start, backtest_end=union_end)
        except (universe_module.UniverseError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    def _run_one(name: str, w_start, w_end, cost_bps: float, overrides=None, warmup=None, artifacts=True):
        spec = tournament_module.STRATEGY_REGISTRY[name]
        if overrides is None and artifacts:
            # Baseline table rows for the incumbents go through their own
            # standalone runners -- guaranteed identical to standalone modes.
            if name == "mean_reversion":
                result, metrics, _cfg, _dir = run_mean_reversion_sleeve(
                    w_start, w_end, args.capital, cost_bps, fractional_shares,
                    args.refresh_cache, args.output_dir,
                    universe=mr_universe.tickers if mr_universe else None,
                    universe_info=mr_universe.info if mr_universe else None,
                )
                return result, metrics
            if name == "sector_rotation":
                result, metrics, _cfg, _dir = run_sector_rotation_sleeve(
                    w_start, w_end, args.capital, cost_bps, fractional_shares,
                    args.refresh_cache, args.output_dir,
                )
                return result, metrics
        run = tournament_module.run_tournament_sleeve(
            spec, w_start, w_end, args.capital, cost_bps, fractional_shares,
            args.refresh_cache, args.output_dir,
            universe=mr_universe.tickers if (mr_universe and spec.uses_stock_universe) else None,
            universe_info=mr_universe.info if (mr_universe and spec.uses_stock_universe) else None,
            param_overrides=overrides, warmup_override=warmup, write_artifacts=artifacts,
        )
        return run.result, run.metrics

    metrics_by_window: dict[str, dict[str, dict]] = {}
    window_ranges: dict[str, str] = {}
    failures: list[tuple[str, str, str]] = []
    # {window_label: {strategy: [(ticker, reason), ...]}} -- tickers dropped
    # for THIS strategy/window (insufficient listing history, no data for the
    # window, or in-range gaps). Surfaced in the tournament report so the
    # us_50b current-universe/listing-history exclusions are explicit, never
    # silent.
    universe_exclusions: dict[str, dict[str, list]] = {}

    for window_label, w_start, w_end in windows:
        print(f"\n=== Tournament window {window_label}: {w_start} to {w_end} ===")
        results_by_strategy: dict = {}
        per_strategy_metrics: dict[str, dict] = {}
        for name in names:
            try:
                result, metrics = _run_one(name, w_start, w_end, args.cost_bps)
            except (data.FetchError, ValueError, tournament_module.TournamentError) as e:
                print(f"  [{window_label}] {name} FAILED: {e}", file=sys.stderr)
                failures.append((window_label, name, str(e)))
                continue
            results_by_strategy[name] = result
            per_strategy_metrics[name] = metrics
            dropped = list(getattr(result, "dropped_tickers", []) or [])
            if dropped:
                universe_exclusions.setdefault(window_label, {})[name] = dropped
            print(
                f"  [{window_label}] {name}: total_return={metrics.get('total_return'):.2%}  "
                f"maxDD={metrics.get('max_drawdown'):.2%}  sharpe={metrics.get('sharpe_ratio'):.2f}"
                f"  ({len(result.universe)} tickers, {len(dropped)} excluded for this window)"
            )
        if not per_strategy_metrics:
            continue
        spy_row = tournament_module.spy_row_for_window(results_by_strategy, refresh_cache=args.refresh_cache)
        if spy_row is not None:
            spy_metrics, spy_range = spy_row
            per_strategy_metrics["SPY"] = spy_metrics
            window_ranges[window_label] = f"{w_start} to {w_end} (SPY row over common range {spy_range})"
        else:
            window_ranges[window_label] = f"{w_start} to {w_end}"
        metrics_by_window[window_label] = per_strategy_metrics

    if not metrics_by_window:
        print("Error: tournament produced no usable strategy runs.", file=sys.stderr)
        return 1

    robustness = None
    if len(metrics_by_window) > 1:
        spy_by_window = {w: m.get("SPY", {}) for w, m in metrics_by_window.items()}
        strat_only = {
            w: {s: m for s, m in wm.items() if s != "SPY"} for w, wm in metrics_by_window.items()
        }
        robustness = tournament_module.robustness_components(strat_only, spy_by_window)

    cost_sensitivity = None
    if cost_bps_levels:
        print("\n=== Cost sensitivity (full --start/--end window) ===")
        cost_sensitivity = {}
        for name in names:
            by_cost: dict[float, dict] = {}
            for bps in cost_bps_levels:
                try:
                    _result, metrics = _run_one(name, args.start, args.end, bps, artifacts=False)
                except (data.FetchError, ValueError, tournament_module.TournamentError) as e:
                    failures.append((f"cost_{bps}bps", name, str(e)))
                    continue
                by_cost[bps] = metrics
            if by_cost:
                cost_sensitivity[name] = by_cost

    param_sensitivity = None
    param_rationale: dict[str, dict[str, str]] = {}
    if args.tournament_param_sensitivity:
        print("\n=== Parameter sensitivity (full --start/--end window) ===")
        param_sensitivity = {}
        for name in names:
            variants = tournament_module.PARAM_SENSITIVITY_VARIANTS.get(name, [])
            if not variants:
                continue
            by_variant: dict[str, dict] = {}
            rationales: dict[str, str] = {}
            try:
                _result, baseline_metrics = _run_one(name, args.start, args.end, args.cost_bps, artifacts=False)
                by_variant["baseline"] = baseline_metrics
                rationales["baseline"] = "shipped defaults, re-run through the same generic runner"
            except (data.FetchError, ValueError, tournament_module.TournamentError) as e:
                failures.append(("param_baseline", name, str(e)))
            for variant_label, overrides, rationale, warmup in variants:
                try:
                    _result, metrics = _run_one(
                        name, args.start, args.end, args.cost_bps,
                        overrides=overrides, warmup=warmup, artifacts=False,
                    )
                except (data.FetchError, ValueError, tournament_module.TournamentError) as e:
                    failures.append((f"param_{variant_label}", name, str(e)))
                    continue
                by_variant[variant_label] = metrics
                rationales[variant_label] = rationale
            if by_variant:
                param_sensitivity[name] = by_variant
                param_rationale[name] = rationales

    describe_by_strategy = {}
    for name in names:
        spec = tournament_module.STRATEGY_REGISTRY[name]
        if spec.uses_stock_universe and mr_universe is not None:
            describe_by_strategy[name] = spec.factory(universe=mr_universe.tickers).describe()
        else:
            describe_by_strategy[name] = spec.factory().describe()

    run_config = {
        "requested_start": str(pd.Timestamp(args.start).date()),
        "requested_end": str(pd.Timestamp(args.end).date()),
        "capital": args.capital,
        "cost_bps": args.cost_bps,
        "fractional_shares": fractional_shares,
        "strategies": names,
        "windows": [label for label, _s, _e in windows],
        "universe_mode": args.universe,
        "capital_note": "each strategy runs at the FULL --capital (alternatives for one "
                        "account, not simultaneous sleeves)",
    }
    run_dir = write_tournament_report(
        metrics_by_window, window_ranges, describe_by_strategy, run_config,
        robustness=robustness, cost_sensitivity=cost_sensitivity,
        param_sensitivity=param_sensitivity, param_rationale=param_rationale,
        failures=failures or None, universe_exclusions=universe_exclusions or None,
        output_dir=args.output_dir,
    )
    print(f"\n[tournament] artifacts written to: {run_dir}")
    if robustness:
        print("[tournament] robustness scores (mean of beats-SPY-fraction and positive-return-fraction):")
        for strat, comp in sorted(robustness.items(), key=lambda kv: -kv[1]["robustness_score"]):
            worst_dd = comp.get("worst_window_max_drawdown")
            worst_dd_str = f"{worst_dd:.2%}" if worst_dd is not None else "n/a"
            coverage = f"{comp.get('num_windows_ran', comp['num_windows'])}/{comp.get('num_windows_expected', comp['num_windows'])}"
            flag = "" if comp.get("full_coverage", True) else f"  [INCOMPLETE: missing {comp.get('num_missing_windows', 0)} window(s), score penalized]"
            print(
                f"  {strat}: score={comp['robustness_score']:.2f} "
                f"(beats SPY {comp['pct_windows_beats_spy_return']:.0%} / positive "
                f"{comp['pct_windows_positive_return']:.0%} of {coverage} expected windows, "
                f"worst window maxDD {worst_dd_str}){flag}"
            )
    return 0


def run_portfolio(args: argparse.Namespace) -> int:
    """Backtest ONE account allocated across weighted strategy sleeves (default
    60% momentum / 35% sector_rotation / 5% regime_switch). Each sleeve gets
    weight x --capital and runs as a fully independent, static (non-rebalanced)
    sleeve; the portfolio curve is the sum of the sleeves' independent equity
    curves over their common date intersection. Distinct from --strategy
    tournament (every strategy at FULL capital, compared) and --strategy
    both/compare (fixed 50/50 of the two original strategies)."""
    fractional_shares = not args.no_fractional_shares

    try:
        pairs = portfolio_module.parse_portfolio_weights(args.portfolio_weights)
        portfolio_module.validate_portfolio_weights(pairs, tournament_module.STRATEGY_REGISTRY)
    except portfolio_module.PortfolioError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Resolve the shared stock universe ONCE if any selected sleeve needs it
    # (momentum / mean_reversion*). Sector-plan sleeves ignore --universe.
    needs_stock_universe = any(
        tournament_module.STRATEGY_REGISTRY[name].uses_stock_universe for name, _ in pairs
    )
    mr_universe = None
    if needs_stock_universe:
        try:
            mr_universe = resolve_mean_reversion_universe(args)
        except (universe_module.UniverseError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    try:
        pf = portfolio_module.run_portfolio(
            pairs, args.capital, args.start, args.end, args.cost_bps, fractional_shares,
            args.refresh_cache, args.output_dir, tournament_module.STRATEGY_REGISTRY,
            mr_universe=mr_universe.tickers if mr_universe else None,
            mr_universe_info=mr_universe.info if mr_universe else None,
        )
    except (data.FetchError, ValueError, portfolio_module.PortfolioError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    describe_by_strategy = {}
    for name, _weight in pairs:
        spec = tournament_module.STRATEGY_REGISTRY[name]
        if spec.uses_stock_universe and mr_universe is not None:
            describe_by_strategy[name] = spec.factory(universe=mr_universe.tickers).describe()
        else:
            describe_by_strategy[name] = spec.factory().describe()

    run_config = {
        "requested_start": str(pd.Timestamp(args.start).date()),
        "requested_end": str(pd.Timestamp(args.end).date()),
        "capital": args.capital,
        "cost_bps": args.cost_bps,
        "fractional_shares": fractional_shares,
        "universe_mode": args.universe,
        "weights": {name: w for name, w in pairs},
        "allocation_note": "static allocation: capital split once at the start, sleeve weights "
                           "drift with performance, no cash transferred between sleeves",
    }
    run_dir = write_portfolio_report(pf, run_config, describe_by_strategy, output_dir=args.output_dir)

    print("\n[portfolio] allocation and result:")
    for sleeve in pf.sleeves:
        print(
            f"  {sleeve.strategy}: start ${sleeve.allocated_capital:,.2f} ({sleeve.weight:.0%}) "
            f"-> final ${sleeve.final_value:,.2f} (end wt {sleeve.ending_weight:.0%}, "
            f"P&L ${sleeve.pnl_contribution:,.2f})"
        )
    print(
        f"  PORTFOLIO: ${pf.total_capital:,.2f} -> ${pf.metrics.get('final_equity', 0):,.2f}  "
        f"total_return={pf.metrics.get('total_return'):.2%}  cagr={pf.metrics.get('cagr'):.2%}  "
        f"maxDD={pf.metrics.get('max_drawdown'):.2%}  sharpe={pf.metrics.get('sharpe_ratio'):.2f}  "
        f"excess vs SPY={pf.metrics.get('excess_return'):.2%}"
    )
    print(f"[portfolio] artifacts written to: {run_dir}")
    return 0


def run_walk_forward(args: argparse.Namespace) -> int:
    """Walk-forward / out-of-sample validation of the weighted portfolio: split
    --start/--end into (train, test) folds, evaluate the portfolio on each
    fold's TEST period with capital carried forward, and stitch the test-period
    curves into one out-of-sample equity curve. v1 (default) evaluates the
    shipped fixed parameters -- no selection, no new overfitting. With
    --walk-forward-optimize, each sleeve's predefined variants are ranked on the
    training window and frozen before the test period."""
    fractional_shares = not args.no_fractional_shares

    try:
        pairs = portfolio_module.parse_portfolio_weights(args.portfolio_weights)
        portfolio_module.validate_portfolio_weights(pairs, tournament_module.STRATEGY_REGISTRY)
    except portfolio_module.PortfolioError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    needs_stock_universe = any(
        tournament_module.STRATEGY_REGISTRY[name].uses_stock_universe for name, _ in pairs
    )
    mr_universe = None
    if needs_stock_universe:
        try:
            mr_universe = resolve_mean_reversion_universe(args)
        except (universe_module.UniverseError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    try:
        wf = walk_forward_module.run_walk_forward(
            pairs, args.capital, args.start, args.end, args.cost_bps, fractional_shares,
            args.refresh_cache, args.output_dir,
            train_years=args.walk_forward_train_years, test_years=args.walk_forward_test_years,
            step_years=args.walk_forward_step_years, expanding=(args.walk_forward_window == "expanding"),
            optimize=args.walk_forward_optimize, registry=tournament_module.STRATEGY_REGISTRY,
            mr_universe=mr_universe.tickers if mr_universe else None,
            mr_universe_info=mr_universe.info if mr_universe else None,
        )
    except (data.FetchError, ValueError, walk_forward_module.WalkForwardError,
            portfolio_module.PortfolioError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    run_config = {
        "requested_start": str(pd.Timestamp(args.start).date()),
        "requested_end": str(pd.Timestamp(args.end).date()),
        "capital": args.capital,
        "cost_bps": args.cost_bps,
        "fractional_shares": fractional_shares,
        "universe_mode": args.universe,
    }
    run_dir = write_walk_forward_report(wf, run_config, output_dir=args.output_dir)

    agg = wf.aggregate
    mode = "optimize" if wf.optimize else "fixed"
    print(f"\n[walk_forward] mode={mode}, {wf.window_mode} windows "
          f"(train={wf.train_years}y test={wf.test_years}y step={wf.step_years}y), "
          f"{agg['num_folds']} folds")
    for fold in wf.folds:
        sharpe = fold.sharpe_ratio
        sharpe_str = f"{sharpe:.2f}" if pd.notna(sharpe) else "n/a"
        print(f"  fold {fold.index}: test {fold.test_start.date()}..{fold.test_end.date()}  "
              f"port={fold.test_return:.2%}  spy={fold.spy_return:.2%}  "
              f"excess={fold.excess_return:.2%}  maxDD={fold.max_drawdown:.2%}  sharpe={sharpe_str}")
    print(f"  STITCHED OOS: total_return={agg['stitched_total_return']:.2%}  "
          f"cagr={agg['stitched_cagr']:.2%}  maxDD={agg['stitched_max_drawdown']:.2%}  "
          f"beat SPY {agg['pct_folds_beating_spy']:.0%} of folds, "
          f"profitable {agg['pct_folds_profitable']:.0%} of folds")
    print(f"[walk_forward] artifacts written to: {run_dir}")
    return 0


def _print_paper_banner() -> None:
    print("=" * 60)
    for n in paper_module.PAPER_NOTICES:
        print(f"** {n}")
    print("=" * 60)


def _print_paper_run_summary(s: dict) -> None:
    print(
        f"[paper] processed {s['paper_date']} (data cutoff {s['data_cutoff_date']}): "
        f"{s['num_new_signals']} new signals, {s['num_pending_created']} pending orders created, "
        f"{s['num_fills']} fills, {s['num_stale']} stale."
    )
    print(
        f"        equity ${s['total_equity']:,.2f}  daily {s['daily_return']:+.2%}  "
        f"cumulative {s['cumulative_return']:+.2%}  costs ${s['transaction_costs_run']:,.2f}  "
        f"reconciliation {'OK' if s['reconciliation_ok'] else 'FAILED'}"
    )


def _print_paper_file(state_dir, filename: str, title: str) -> None:
    df = pd.read_csv(Path(state_dir) / filename)
    print(f"--- {title} ({len(df)}) ---")
    if len(df):
        print(df.to_string(index=False))
    else:
        print("(none)")


def _paper_export(args, cfg: dict, st: dict, state_dir) -> Path:
    ts = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
    export_dir = Path(args.output_dir) / f"{ts}_paper_export"
    write_paper_artifacts(cfg, st, export_dir)
    # Include the authoritative SQLite ledger (paper_db.DB_FILENAME) alongside
    # the derived paper_config.json/paper_state.json views and CSVs.
    export_names = [paper_module.CONFIG_FILENAME, paper_module.STATE_FILENAME, paper_db_module.DB_FILENAME]
    for fname in export_names:
        src = Path(state_dir) / fname
        if src.exists():
            shutil.copy2(src, export_dir / fname)
    return export_dir


def run_paper(args: argparse.Namespace) -> int:
    """Forward paper-trading driver: a persistent, reloadable simulation of the
    fixed 60/35/5 portfolio that processes finalized sessions one at a time using
    only data available as of each date. NEVER connects to a broker or sends a
    real order. Exactly one action flag is chosen per invocation."""
    state_dir = args.paper_state_dir
    fractional_shares = not args.no_fractional_shares

    action_flags = [
        args.paper_init, args.paper_run, args.paper_date is not None, args.paper_status,
        args.paper_orders, args.paper_trades, args.paper_export, args.paper_reconcile,
        args.paper_reset,
    ]
    n_actions = sum(1 for a in action_flags if a)
    if n_actions == 0:
        print(
            "Error: --strategy paper requires exactly one action: --paper-init, --paper-run, "
            "--paper-date, --paper-status, --paper-orders, --paper-trades, --paper-export, "
            "--paper-reconcile, or --paper-reset.", file=sys.stderr,
        )
        return 1
    if n_actions > 1:
        print("Error: choose exactly one paper action per invocation.", file=sys.stderr)
        return 1

    try:
        if args.paper_reset:
            backup = paper_module.reset_account(state_dir, confirm=args.confirm_paper_reset)
            print(f"[paper] account reset. Prior state backed up to: {backup}")
            print("[paper] run --paper-init to start a new account.")
            return 0

        if args.paper_init:
            pairs = portfolio_module.parse_portfolio_weights(args.portfolio_weights)
            portfolio_module.validate_portfolio_weights(pairs, tournament_module.STRATEGY_REGISTRY)
            inception = args.paper_start or pd.Timestamp.now().normalize().date().isoformat()
            needs_stock = any(
                tournament_module.STRATEGY_REGISTRY[n].uses_stock_universe for n, _ in pairs
            )
            mr_universe = None
            if needs_stock:
                mr_universe = resolve_mean_reversion_universe(
                    args, backtest_start=inception, backtest_end=pd.Timestamp.now().normalize(),
                )
            paper_module.init_account(
                state_dir, args.capital, inception, pairs, args.cost_bps, fractional_shares,
                args.universe, mr_universe.tickers if mr_universe else None,
                mr_universe.info if mr_universe else None,
            )
            cfg, st = paper_module.load_account(state_dir)
            write_paper_artifacts(cfg, st, state_dir)
            print(f"[paper] initialized account in {state_dir}: ${args.capital:,.2f}, inception {inception}")
            for name, w in pairs:
                print(f"  {name}: {w:.0%}")
            _print_paper_banner()
            return 0

        if args.paper_run or args.paper_date is not None:
            result = paper_module.advance(
                state_dir, target_date=args.paper_date, refresh_cache=args.refresh_cache,
            )
            cfg, st = result["config"], result["state"]
            write_paper_artifacts(cfg, st, state_dir)
            if result["message"]:
                print(f"[paper] {result['message']}")
            for s in result["processed"]:
                _print_paper_run_summary(s)
            if result["processed"]:
                print(f"[paper] ledger updated in: {state_dir}")
            _print_paper_banner()
            return 0

        # Read-only commands: load + regenerate the derived views, then display.
        cfg, st = paper_module.load_account(state_dir)
        write_paper_artifacts(cfg, st, state_dir)
        if args.paper_status:
            print((Path(state_dir) / "paper_status.txt").read_text())
        elif args.paper_orders:
            _print_paper_file(state_dir, "paper_orders.csv", "Pending / stale orders")
            _print_paper_banner()
        elif args.paper_trades:
            _print_paper_file(state_dir, "paper_trades.csv", "Completed paper trades")
            _print_paper_banner()
        elif args.paper_reconcile:
            recon = paper_module.reconcile_saved_state(state_dir)
            print(f"[paper] reconciliation of persisted state: {'OK' if recon['ok'] else 'FAILED'}")
            for c in recon["checks"]:
                print(f"  [{'ok' if c['ok'] else 'XX'}] {c['check']}: {c['detail']}")
            return 0 if recon["ok"] else 1
        elif args.paper_export:
            export_dir = _paper_export(args, cfg, st, state_dir)
            print(f"[paper] exported all ledger artifacts to: {export_dir}")
        return 0
    except (paper_module.PaperError, data.FetchError, ValueError,
            universe_module.UniverseError, portfolio_module.PortfolioError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _print_summary(label: str, metrics: dict, run_dir) -> None:
    print(f"\n[{label}] total_return={metrics.get('total_return'):.2%}  "
          f"cagr={metrics.get('cagr'):.2%}  max_drawdown={metrics.get('max_drawdown'):.2%}  "
          f"sharpe={metrics.get('sharpe_ratio'):.2f}  trades={metrics.get('num_trades')}")
    print(f"[{label}] artifacts written to: {run_dir}")


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        validate_args(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.strategy == "robustness":
        return run_robustness(args)

    if args.strategy == "tournament":
        return run_tournament(args)

    if args.strategy == "portfolio":
        return run_portfolio(args)

    if args.strategy == "walk_forward":
        return run_walk_forward(args)

    if args.strategy == "paper":
        return run_paper(args)

    fractional_shares = not args.no_fractional_shares
    mr_result = sr_result = None

    try:
        if args.strategy in ("mean_reversion", "both", "compare"):
            mr_universe = resolve_mean_reversion_universe(args)
            capital = args.capital / 2.0 if args.strategy in ("both", "compare") else args.capital
            mr_result, mr_metrics, mr_config, mr_dir = run_mean_reversion_sleeve(
                args.start, args.end, capital, args.cost_bps, fractional_shares,
                args.refresh_cache, args.output_dir,
                universe=mr_universe.tickers, universe_info=mr_universe.info,
            )
            _print_summary("mean_reversion", mr_metrics, mr_dir)

        if args.strategy in ("sector_rotation", "both", "compare"):
            capital = args.capital / 2.0 if args.strategy in ("both", "compare") else args.capital
            sr_result, sr_metrics, sr_config, sr_dir = run_sector_rotation_sleeve(
                args.start, args.end, capital, args.cost_bps, fractional_shares,
                args.refresh_cache, args.output_dir,
            )
            _print_summary("sector_rotation", sr_metrics, sr_dir)

        if args.strategy in ("both", "compare"):
            combined = combine_results(mr_result, sr_result)
            spy_df = data.get_benchmark_data(combined.start, combined.end, force_refresh=args.refresh_cache)
            combined_metrics = compute_all_metrics(combined, benchmark_close=spy_df["Close"])
            combined_config = {
                "strategy": "both",
                "requested_start": str(pd.Timestamp(args.start).date()),
                "requested_end": str(pd.Timestamp(args.end).date()),
                "effective_start": str(combined.start.date()),
                "effective_end": str(combined.end.date()),
                "capital": args.capital,
                "cost_bps": args.cost_bps,
                "fractional_shares": fractional_shares,
                "universe": list(combined.universe),
                "yfinance_version": yf.__version__,
                "warnings": [
                    "Combined mode runs the two strategies as fully independent sleeves "
                    "(separate cash/shares/lots); this report sums their two independently "
                    "computed equity curves over the INTERSECTION of their valid date ranges."
                ],
                "cache_summary": "combined from two independently-run sleeves",
            }
            combined_dir = write_run_artifacts(combined, combined_metrics, combined_config, output_dir=args.output_dir)
            _print_summary("both (combined)", combined_metrics, combined_dir)

        if args.strategy == "compare":
            if args.compare_years:
                years = [int(y.strip()) for y in args.compare_years.split(",")]
            else:
                years = list(range(pd.Timestamp(args.start).year, pd.Timestamp(args.end).year + 1))

            spy_metrics = spy_standalone_metrics(spy_df["Close"])
            contribution = compute_sleeve_contribution(mr_result, sr_result, combined)

            metrics_by_label = {
                "mean_reversion": mr_metrics,
                "sector_rotation": sr_metrics,
                "both": combined_metrics,
                "SPY": spy_metrics,
            }
            equity_by_label = {
                "mean_reversion": mr_result.equity_curve,
                "sector_rotation": sr_result.equity_curve,
                "both": combined.equity_curve,
                "SPY": spy_df["Close"],
            }
            ranges_by_label = {
                "mean_reversion": f"{mr_result.start.date()} to {mr_result.end.date()}",
                "sector_rotation": f"{sr_result.start.date()} to {sr_result.end.date()}",
                "both": f"{combined.start.date()} to {combined.end.date()}",
                "SPY": f"{combined.start.date()} to {combined.end.date()} (combined/intersection window)",
            }
            compare_run_config = {
                "requested_start": str(pd.Timestamp(args.start).date()),
                "requested_end": str(pd.Timestamp(args.end).date()),
                "capital": args.capital,
                "cost_bps": args.cost_bps,
                "years": years,
            }
            comparison_dir = write_comparison_report(
                metrics_by_label, equity_by_label, ranges_by_label, years, compare_run_config,
                contribution=contribution, output_dir=args.output_dir,
            )
            print(f"\n[compare] artifacts written to: {comparison_dir}")

    except (data.FetchError, ValueError, universe_module.UniverseError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
