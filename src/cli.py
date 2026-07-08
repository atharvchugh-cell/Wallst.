"""CLI orchestration: fetch data, run one or both strategy sleeves, combine
if needed, and write the full audit artifact set."""

from __future__ import annotations

import argparse
import sys
import warnings as warnings_module

import pandas as pd
import yfinance as yf

from . import config, data
from .engine import combine_results, run_backtest
from .metrics import compute_all_metrics, spy_standalone_metrics
from .reporting import (
    write_run_artifacts, write_comparison_report, compute_sleeve_contribution, write_robustness_report,
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
        choices=["mean_reversion", "sector_rotation", "both", "compare", "robustness"],
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


def run_mean_reversion_sleeve(start, end, capital, cost_bps, fractional_shares, refresh_cache, output_dir):
    warnings: list[str] = []
    strat = MeanReversionStrategy()
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
    pre_drops = list(fetch_dropped) + gap_dropped

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

    all_window_metrics: dict[str, dict[str, dict]] = {}
    window_ranges: dict[str, str] = {}

    for window_label, w_start, w_end in windows:
        print(f"\n=== Robustness window {window_label}: {w_start} to {w_end} ===")
        try:
            mr_result, mr_metrics, _mr_config, _mr_dir = run_mean_reversion_sleeve(
                w_start, w_end, args.capital, args.cost_bps, fractional_shares,
                args.refresh_cache, args.output_dir,
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

    fractional_shares = not args.no_fractional_shares
    mr_result = sr_result = None

    try:
        if args.strategy in ("mean_reversion", "both", "compare"):
            capital = args.capital / 2.0 if args.strategy in ("both", "compare") else args.capital
            mr_result, mr_metrics, mr_config, mr_dir = run_mean_reversion_sleeve(
                args.start, args.end, capital, args.cost_bps, fractional_shares,
                args.refresh_cache, args.output_dir,
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

    except (data.FetchError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
