"""report.txt + CSV/JSON audit artifacts + equity-curve PNG, written to a
timestamped, parameterized run directory so repeated runs don't clobber
prior audit trails. CSVs are the primary audit artifact; the PNG is
secondary.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402
from .engine import BacktestResult  # noqa: E402
from .metrics import annual_returns, monthly_returns  # noqa: E402
from .robustness import (  # noqa: E402
    ALLOCATION_MIXES,
    average_ranks,
    beats_spy_fraction,
    mean_reversion_tradeoff,
    rank_allocations,
)

if TYPE_CHECKING:  # avoid a runtime import cycle (reporting <- portfolio/walk_forward <- tournament <- reporting)
    from .portfolio import PortfolioResult
    from .walk_forward import WalkForwardResult

EXECUTION_MODEL_LINE = (
    "Execution model: close-to-close, one-day signal-to-fill lag, "
    "no intraday stops, no bid/ask spread, no market impact. This is LESS "
    "BIASED than same-close fills, not a claim of realistic live execution."
)
ADJUSTED_PRICE_LINE = (
    "All prices/trades are auto_adjust=True adjusted-price units -- synthetic "
    "adjusted share counts and fill prices, not literal historical execution "
    "prices you could have actually transacted at."
)
STOP_LOSS_WORDING_LINE = (
    "The mean-reversion 'stop-loss' is a DELAYED EXIT RULE, not a real stop: "
    "it only reacts to a day's closing price and exits at the following "
    "day's close, so it will not protect against an intraday or overnight gap."
)
PRETAX_WARNING_LINE = (
    "All returns/metrics are PRE-TAX. Short-term trading gains (common in "
    "mean reversion) are typically taxed at ordinary income rates; after-tax "
    "results can be materially worse than shown here."
)
SURVIVORSHIP_WARNING = (
    "WARNING: the mean-reversion universe is a SURVIVORSHIP-BIASED research "
    "universe (today's large-cap survivors), not a point-in-time constituent "
    "list. Results here validate the mechanics of the strategy, not a "
    "general edge -- see README."
)
UNIVERSE_SNAPSHOT_WARNING = (
    "WARNING: this universe is a CURRENT SNAPSHOT (today's market caps / "
    "listing status), not a point-in-time historical constituent list. A "
    "ticker's presence here does not mean it met the market-cap threshold "
    "at every historical date tested -- market caps drift over time, and "
    "re-running --refresh-universe on a different day can change which "
    "tickers qualify. Like the default universe, this is still "
    "survivorship-biased research, not a point-in-time historical validation."
)

TARGET_EVENT_COLUMNS = [
    "strategy", "ticker", "signal_date", "fill_date", "target_weight",
    "requested_notional", "sizing_price", "reason",
]
TRANSACTION_COLUMNS = [
    "strategy", "ticker", "signal_date", "fill_date", "action", "requested_target_weight",
    "requested_notional", "sizing_price", "shares_traded", "fill_price", "executed_notional",
    "actual_weight_after", "transaction_cost", "cash_after", "position_shares_after",
    "avg_cost_basis_after", "reason",
]
TRADE_COLUMNS = [
    "strategy", "ticker", "event_type", "date", "shares_sold", "sale_price", "avg_cost_basis",
    "realized_pnl", "realized_return_pct", "realized_pnl_net", "realized_return_pct_net",
    "reason", "holding_days", "holding_calendar_days",
]
POSITION_COLUMNS = [
    "date", "strategy", "ticker", "shares", "adjusted_close", "market_value",
    "portfolio_weight", "cash", "sleeve_equity",
]


def make_run_dir(output_dir: str, strategy_name: str, start, end) -> Path:
    ts = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
    dirname = f"{ts}_{strategy_name}_{pd.Timestamp(start).date()}_to_{pd.Timestamp(end).date()}"
    path = Path(output_dir) / dirname
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(path: Path, rows: list, columns: list[str]) -> None:
    if rows and dataclasses.is_dataclass(rows[0]):
        df = pd.DataFrame([dataclasses.asdict(r) for r in rows])
    else:
        df = pd.DataFrame(rows)
    for c in columns:
        if c not in df.columns:
            df[c] = None
    df = df[columns] if len(df) else pd.DataFrame(columns=columns)
    df.to_csv(path, index=False)


def write_run_artifacts(
    result: BacktestResult, metrics: dict, run_config: dict, output_dir: str = config.OUTPUT_DIR
) -> Path:
    run_dir = make_run_dir(output_dir, result.strategy_name, result.start, result.end)

    _write_csv(run_dir / "target_events.csv", result.target_events, TARGET_EVENT_COLUMNS)
    _write_csv(run_dir / "transactions.csv", result.transactions, TRANSACTION_COLUMNS)
    _write_csv(run_dir / "trades.csv", result.trades, TRADE_COLUMNS)
    _write_csv(run_dir / "positions.csv", result.positions, POSITION_COLUMNS)

    result.equity_curve.rename("equity").to_csv(run_dir / "equity_curve.csv", header=True, index_label="date")

    metrics_out = dict(run_config)
    metrics_out["metrics"] = metrics
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)

    _write_equity_png(result, run_dir / "equity_curve.png")
    _write_report_txt(result, metrics, run_config, run_dir / "report.txt")

    return run_dir


def _write_equity_png(result: BacktestResult, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    result.equity_curve.plot(ax=ax)
    ax.set_title(f"{result.strategy_name} equity curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fmt_pct(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{x * 100:.2f}%"


def _fmt_num(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{x:,.2f}"


def _write_report_txt(result: BacktestResult, metrics: dict, run_config: dict, path: Path) -> None:
    lines = []
    lines.append(f"=== Backtest Report: {result.strategy_name} ===")
    lines.append("")
    lines.append("** RESEARCH / EDUCATIONAL USE ONLY. NOT FINANCIAL ADVICE. **")
    lines.append("Past performance does not indicate future results. No live trades were placed.")
    lines.append("")
    lines.append(f"Requested window: {run_config.get('requested_start')} to {run_config.get('requested_end')}")
    lines.append(f"Effective window used: {result.start.date()} to {result.end.date()}")
    if run_config.get("warnings"):
        lines.append("")
        lines.append("--- Warnings ---")
        for w in run_config["warnings"]:
            lines.append(f"  - {w}")
    if result.strategy_name == "mean_reversion":
        lines.append("")
        lines.append(SURVIVORSHIP_WARNING)

    lines.append("")
    lines.append(f"Starting capital: ${run_config.get('capital', result.capital):,.2f}")
    lines.append(f"Universe ({len(result.universe)}): {', '.join(result.universe)}")
    universe_info = run_config.get("universe_info")
    if universe_info:
        lines.append("")
        lines.append("--- Universe selection ---")
        lines.append(f"Universe mode: {universe_info.get('mode')}")
        lines.append(f"Tickers selected: {universe_info.get('num_selected', len(result.universe))}")
        if universe_info.get("num_candidates") is not None:
            lines.append(f"Screener results considered: {universe_info['num_candidates']}")
        if universe_info.get("num_dropped_lookup_failed") is not None:
            lines.append(
                f"Unparseable screener results: {universe_info['num_dropped_lookup_failed']}"
            )
        if universe_info.get("num_excluded_not_listed") is not None:
            lines.append(
                f"Excluded (not in Nasdaq Trader candidate set): {universe_info['num_excluded_not_listed']}"
            )
        if universe_info.get("num_excluded_identity_mismatch") is not None:
            lines.append(
                f"Excluded (screener/Nasdaq Trader name mismatch): "
                f"{universe_info['num_excluded_identity_mismatch']}"
            )
        if universe_info.get("num_excluded_non_common") is not None:
            lines.append(
                f"Excluded (non-common-stock by name): {universe_info['num_excluded_non_common']}"
            )
        if universe_info.get("num_excluded_no_price_data") is not None:
            lines.append(
                f"Excluded (no usable price data): {universe_info['num_excluded_no_price_data']}"
            )
        if universe_info.get("num_duplicate_companies_collapsed") is not None:
            lines.append(
                f"Duplicate-company tickers collapsed (e.g. multiple share classes): "
                f"{universe_info['num_duplicate_companies_collapsed']}"
            )
        if universe_info.get("min_market_cap") is not None and universe_info.get("max_market_cap") is not None:
            lines.append(
                f"Market cap range in universe: ${universe_info['min_market_cap']:,.0f} - "
                f"${universe_info['max_market_cap']:,.0f}"
            )
        if universe_info.get("cache_file"):
            lines.append(f"Universe cache file: {universe_info['cache_file']}")
        if universe_info.get("snapshot_date"):
            lines.append(f"Universe snapshot timestamp: {universe_info['snapshot_date']}")
        if universe_info.get("price_data_validated_start") and universe_info.get("price_data_validated_end"):
            lines.append(
                f"Price data validated for window: {universe_info['price_data_validated_start']} to "
                f"{universe_info['price_data_validated_end']} (warmup-adjusted; a later run requesting "
                f"a window outside this range re-validates before reusing this cache)"
            )
        if universe_info.get("mode") != "default":
            lines.append(UNIVERSE_SNAPSHOT_WARNING)
    if result.dropped_tickers:
        lines.append("Dropped tickers:")
        for t, reason in result.dropped_tickers:
            lines.append(f"  - {t}: {reason}")
    lines.append(f"Transaction cost assumption: {result.cost_bps} bps per trade")
    lines.append(f"Fractional shares: {'enabled' if result.fractional_shares else 'disabled'}")
    lines.append(f"Cache hits/misses: {run_config.get('cache_summary', 'n/a')}")
    lines.append("")
    lines.append(EXECUTION_MODEL_LINE)
    lines.append(ADJUSTED_PRICE_LINE)
    lines.append(PRETAX_WARNING_LINE)
    if result.strategy_name in ("mean_reversion", "both"):
        lines.append(STOP_LOSS_WORDING_LINE)

    lines.append("")
    lines.append("--- Performance ---")
    lines.append(f"Final equity: ${_fmt_num(metrics.get('final_equity'))}")
    lines.append(f"Total return: {_fmt_pct(metrics.get('total_return'))}")
    lines.append(f"CAGR: {_fmt_pct(metrics.get('cagr'))}")
    if metrics.get("short_period_warning"):
        lines.append("  (WARNING: effective period < 90 days -- CAGR/Sharpe are unstable over short windows)")
    lines.append(f"Max drawdown: {_fmt_pct(metrics.get('max_drawdown'))}")
    lines.append(f"Max drawdown duration: {metrics.get('max_drawdown_duration_days')} calendar days")
    sharpe = metrics.get("sharpe_ratio")
    lines.append(f"Sharpe ratio (rf=0): {sharpe:.2f}" if pd.notna(sharpe) else "Sharpe ratio (rf=0): n/a")
    sortino = metrics.get("sortino_ratio")
    lines.append(f"Sortino ratio (rf=0): {sortino:.2f}" if pd.notna(sortino) else "Sortino ratio (rf=0): n/a")
    calmar = metrics.get("calmar_ratio")
    lines.append(f"Calmar ratio (CAGR / |max DD|): {calmar:.2f}" if pd.notna(calmar) else "Calmar ratio: n/a")
    best_m, worst_m = metrics.get("best_month"), metrics.get("worst_month")
    lines.append(
        f"Best month: {_fmt_pct(best_m)}  |  Worst month: {_fmt_pct(worst_m)}"
    )

    if result.strategy_name == "sector_rotation":
        lines.append(
            "(Win rate de-emphasized for sector rotation -- long holds + partial "
            "rebalances make it a weak signal here.)"
        )
    lines.append(
        f"Win rate, net of costs (full-exit trades): {_fmt_pct(metrics.get('win_rate'))}  "
        f"(gross, pre-costs: {_fmt_pct(metrics.get('win_rate_gross'))})"
    )
    lines.append(
        f"Round-trip trades: {metrics.get('num_trades')}  |  "
        f"Partial rebalance sells: {metrics.get('num_partial_sells')}"
    )
    lines.append(
        f"Transactions: {metrics.get('num_transactions')}  |  "
        f"Target events: {metrics.get('num_target_events')}"
    )

    lines.append("")
    lines.append("--- Cost & turnover ---")
    lines.append(f"Total transaction costs paid: ${_fmt_num(metrics.get('total_transaction_costs'))}")
    lines.append(f"Total turnover: ${_fmt_num(metrics.get('total_turnover'))}")
    lines.append(f"Cost drag (% of starting capital): {_fmt_pct(metrics.get('cost_drag_pct'))}")

    lines.append("")
    lines.append("--- Exposure ---")
    lines.append(f"Days with any position: {_fmt_pct(metrics.get('days_with_any_position_pct'))}")
    lines.append(f"Average capital invested: {_fmt_pct(metrics.get('average_capital_invested_pct'))}")

    if "benchmark_total_return" in metrics:
        lines.append("")
        lines.append("--- Benchmark (adjusted-price SPY total-return proxy) ---")
        lines.append(f"SPY total return: {_fmt_pct(metrics.get('benchmark_total_return'))}")
        lines.append(f"SPY CAGR: {_fmt_pct(metrics.get('benchmark_cagr'))}")
        lines.append(f"SPY max drawdown: {_fmt_pct(metrics.get('benchmark_max_drawdown'))}")
        lines.append(f"Excess return vs. SPY: {_fmt_pct(metrics.get('excess_return'))}")
        corr = metrics.get("correlation_to_benchmark")
        lines.append(f"Correlation to SPY: {corr:.2f}" if pd.notna(corr) else "Correlation to SPY: n/a")

    lines.append("")
    lines.append("--- Sample Transactions (first 10) ---")
    lines.append(
        f"{'ticker':<8}{'signal_date':<14}{'fill_date':<14}{'req_wt':<10}"
        f"{'fill_price':<12}{'exec_notional':<14}{'actual_wt':<10}{'reason':<16}"
    )
    for tx in result.transactions[:10]:
        lines.append(
            f"{tx.ticker:<8}{str(tx.signal_date.date()):<14}{str(tx.fill_date.date()):<14}"
            f"{tx.requested_target_weight:<10.3f}{tx.fill_price:<12.2f}{tx.executed_notional:<14.2f}"
            f"{(tx.actual_weight_after or 0):<10.3f}{(tx.reason or ''):<16}"
        )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# --- Comparison report: mean_reversion vs sector_rotation vs both vs SPY ---
# Diagnostics only -- this section does not run backtests itself or touch
# strategy parameters; it only formats/compares results the caller already
# computed (see cli.py's `compare` mode).

def _fmt_int(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{int(x)}"


def _fmt_ratio(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{x:.2f}"


# (metrics dict key, human-readable label, formatter) -- shared by the CSV
# column order and the human-readable table.
COMPARISON_METRIC_FIELDS = [
    ("total_return", "Total return", _fmt_pct),
    ("cagr", "CAGR", _fmt_pct),
    ("max_drawdown", "Max drawdown", _fmt_pct),
    ("max_drawdown_duration_days", "Max drawdown duration (days)", _fmt_int),
    ("sharpe_ratio", "Sharpe", _fmt_ratio),
    ("sortino_ratio", "Sortino", _fmt_ratio),
    ("calmar_ratio", "Calmar", _fmt_ratio),
    ("best_month", "Best month", _fmt_pct),
    ("worst_month", "Worst month", _fmt_pct),
    ("total_turnover", "Turnover ($)", _fmt_num),
    ("total_transaction_costs", "Transaction costs ($)", _fmt_num),
    ("cost_drag_pct", "Cost drag (% of capital)", _fmt_pct),
    ("num_transactions", "Transactions", _fmt_int),
    ("num_trades", "Round-trip trades", _fmt_int),
    ("average_capital_invested_pct", "Avg capital invested", _fmt_pct),
    ("correlation_to_benchmark", "Correlation to SPY", _fmt_ratio),
    ("excess_return", "Excess return vs SPY", _fmt_pct),
]


def compute_sleeve_contribution(mr_result: BacktestResult, sr_result: BacktestResult, combined: BacktestResult) -> dict:
    """How much of the combined result's dollar gain and transaction costs
    came from each sleeve, measured over the COMBINED window (the
    intersection of both sleeves' valid ranges) -- not each sleeve's own
    full history, since anything before the combined window isn't part of
    what the combined report is claiming to measure. Computed directly from
    each sleeve's OWN equity curve/transactions (not `combined.transactions`,
    which is a simple concatenation and may include pre-window rows)."""
    window_start, window_end = combined.start, combined.end

    def _costs_in_window(result: BacktestResult) -> float:
        return sum(
            tx.transaction_cost for tx in result.transactions if window_start <= tx.fill_date <= window_end
        )

    mr_start_equity = float(mr_result.equity_curve.loc[window_start])
    mr_end_equity = float(mr_result.equity_curve.loc[window_end])
    sr_start_equity = float(sr_result.equity_curve.loc[window_start])
    sr_end_equity = float(sr_result.equity_curve.loc[window_end])
    mr_dollar_gain = mr_end_equity - mr_start_equity
    sr_dollar_gain = sr_end_equity - sr_start_equity
    total_dollar_gain = mr_dollar_gain + sr_dollar_gain

    mr_costs = _costs_in_window(mr_result)
    sr_costs = _costs_in_window(sr_result)
    total_costs = mr_costs + sr_costs

    return {
        "window_start": window_start,
        "window_end": window_end,
        "mr_start_equity": mr_start_equity,
        "mr_final_equity": mr_end_equity,
        "sr_start_equity": sr_start_equity,
        "sr_final_equity": sr_end_equity,
        "mr_dollar_gain": mr_dollar_gain,
        "sr_dollar_gain": sr_dollar_gain,
        "mr_return_contribution_pct": (mr_dollar_gain / total_dollar_gain) if total_dollar_gain else float("nan"),
        "sr_return_contribution_pct": (sr_dollar_gain / total_dollar_gain) if total_dollar_gain else float("nan"),
        "mr_transaction_costs": mr_costs,
        "sr_transaction_costs": sr_costs,
        "mr_cost_contribution_pct": (mr_costs / total_costs) if total_costs else float("nan"),
        "sr_cost_contribution_pct": (sr_costs / total_costs) if total_costs else float("nan"),
    }


def _write_comparison_csv(path: Path, metrics_by_label: dict[str, dict], labels: list[str]) -> None:
    rows = []
    for key, label_name, _fmt in COMPARISON_METRIC_FIELDS:
        row = {"metric": label_name}
        for lbl in labels:
            row[lbl] = metrics_by_label[lbl].get(key)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_annual_returns_csv(
    path: Path, equity_by_label: dict[str, pd.Series], labels: list[str], years: list[int]
) -> None:
    rows = []
    for y in years:
        row = {"year": y}
        for lbl in labels:
            row[lbl] = annual_returns(equity_by_label[lbl], years=[y]).get(y)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_monthly_returns_csv(path: Path, equity_by_label: dict[str, pd.Series], labels: list[str]) -> None:
    series_by_label = {lbl: monthly_returns(equity_by_label[lbl]) for lbl in labels}
    df = pd.DataFrame(series_by_label)
    df.index.name = "month"
    df.to_csv(path)


def _write_comparison_txt(
    path: Path,
    metrics_by_label: dict[str, dict],
    equity_by_label: dict[str, pd.Series],
    labels: list[str],
    years: list[int],
    ranges_by_label: dict[str, str],
    contribution: dict | None,
    run_config: dict,
) -> None:
    lines = []
    lines.append("=== Strategy Comparison Report ===")
    lines.append("")
    lines.append("** RESEARCH / EDUCATIONAL USE ONLY. NOT FINANCIAL ADVICE. **")
    lines.append(
        "Diagnostics only -- this report does not change or tune any strategy "
        "parameter; it exists to help decide WHERE underperformance (if any) "
        "is coming from before touching RSI/SMA/holding-day/top-K/allocation."
    )
    lines.append("")
    lines.append(f"Requested window: {run_config.get('requested_start')} to {run_config.get('requested_end')}")
    lines.append("Effective ranges used (may differ per sleeve -- warmup/clipping/intersection):")
    for lbl in labels:
        lines.append(f"  {lbl}: {ranges_by_label.get(lbl, 'n/a')}")

    col_w = 18
    name_w = 30
    lines.append("")
    lines.append("--- Comparison table ---")
    lines.append(f"{'Metric':<{name_w}}" + "".join(f"{lbl:>{col_w}}" for lbl in labels))
    for key, label_name, fmt in COMPARISON_METRIC_FIELDS:
        cells = "".join(f"{fmt(metrics_by_label[lbl].get(key)):>{col_w}}" for lbl in labels)
        lines.append(f"{label_name:<{name_w}}{cells}")

    lines.append("")
    lines.append(f"--- Calendar-year returns ({', '.join(str(y) for y in years)}) ---")
    lines.append(f"{'Year':<{name_w}}" + "".join(f"{lbl:>{col_w}}" for lbl in labels))
    for y in years:
        cells = "".join(
            f"{_fmt_pct(annual_returns(equity_by_label[lbl], years=[y]).get(y)):>{col_w}}" for lbl in labels
        )
        lines.append(f"{y:<{name_w}}{cells}")
    lines.append(
        "(A year with no data for a given row -- e.g. before that sleeve's "
        "effective start -- shows n/a, not 0%.)"
    )

    if contribution:
        lines.append("")
        lines.append("--- Strategy contribution (combined sleeve, over the combined window) ---")
        lines.append(f"Combined window: {contribution['window_start'].date()} to {contribution['window_end'].date()}")
        lines.append(
            f"Mean-reversion equity: ${_fmt_num(contribution['mr_start_equity'])} -> "
            f"${_fmt_num(contribution['mr_final_equity'])}  "
            f"(${_fmt_num(contribution['mr_dollar_gain'])})"
        )
        lines.append(
            f"Sector-rotation equity: ${_fmt_num(contribution['sr_start_equity'])} -> "
            f"${_fmt_num(contribution['sr_final_equity'])}  "
            f"(${_fmt_num(contribution['sr_dollar_gain'])})"
        )
        lines.append(
            f"Contribution to combined $ gain: mean_reversion "
            f"{_fmt_pct(contribution['mr_return_contribution_pct'])}  |  "
            f"sector_rotation {_fmt_pct(contribution['sr_return_contribution_pct'])}"
        )
        lines.append(
            f"Transaction costs: mean_reversion ${_fmt_num(contribution['mr_transaction_costs'])} "
            f"({_fmt_pct(contribution['mr_cost_contribution_pct'])} of combined costs)  |  "
            f"sector_rotation ${_fmt_num(contribution['sr_transaction_costs'])} "
            f"({_fmt_pct(contribution['sr_cost_contribution_pct'])} of combined costs)"
        )
        lines.append(
            "(If contribution % is strongly negative for one sleeve while the "
            "other is strongly positive, that sleeve is a net drag over this "
            "window -- see README before concluding it should be cut, since a "
            "single window is not a validated result.)"
        )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_comparison_report(
    metrics_by_label: dict[str, dict],
    equity_by_label: dict[str, pd.Series],
    ranges_by_label: dict[str, str],
    years: list[int],
    run_config: dict,
    contribution: dict | None = None,
    output_dir: str = config.OUTPUT_DIR,
) -> Path:
    """Writes a 4-way comparison (mean_reversion, sector_rotation, both, SPY)
    of already-computed results. `metrics_by_label`/`equity_by_label` must
    share the same keys (the row labels, in display order)."""
    labels = list(metrics_by_label.keys())
    ts = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
    dirname = f"{ts}_comparison_{run_config.get('requested_start')}_to_{run_config.get('requested_end')}"
    run_dir = Path(output_dir) / dirname
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_comparison_csv(run_dir / "comparison.csv", metrics_by_label, labels)
    _write_annual_returns_csv(run_dir / "annual_returns.csv", equity_by_label, labels, years)
    _write_monthly_returns_csv(run_dir / "monthly_returns.csv", equity_by_label, labels)
    _write_comparison_txt(
        run_dir / "comparison.txt", metrics_by_label, equity_by_label, labels, years,
        ranges_by_label, contribution, run_config,
    )

    metrics_out = dict(run_config)
    metrics_out["metrics_by_label"] = metrics_by_label
    metrics_out["contribution"] = contribution
    with open(run_dir / "comparison.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)

    return run_dir


# --- Tournament report: N strategies x windows vs. SPY --------------------------
# Diagnostics only -- formats results the caller (cli.py's tournament mode /
# src/tournament.py) already computed; no strategy parameter is touched here.

# Superset of COMPARISON_METRIC_FIELDS (kept separate so the existing
# comparison/robustness artifacts' contents don't change): adds win rate,
# best/worst year, and time-in-market.
TOURNAMENT_METRIC_FIELDS = [
    ("total_return", "Total return", _fmt_pct),
    ("cagr", "CAGR", _fmt_pct),
    ("max_drawdown", "Max drawdown", _fmt_pct),
    ("max_drawdown_duration_days", "Max DD duration (days)", _fmt_int),
    ("sharpe_ratio", "Sharpe", _fmt_ratio),
    ("sortino_ratio", "Sortino", _fmt_ratio),
    ("calmar_ratio", "Calmar", _fmt_ratio),
    ("win_rate", "Win rate (net, full exits)", _fmt_pct),
    ("num_trades", "Round-trip trades", _fmt_int),
    ("num_transactions", "Transactions", _fmt_int),
    ("total_turnover", "Turnover ($)", _fmt_num),
    ("total_transaction_costs", "Transaction costs ($)", _fmt_num),
    ("cost_drag_pct", "Cost drag (% of capital)", _fmt_pct),
    ("days_with_any_position_pct", "Time in market (any position)", _fmt_pct),
    ("average_capital_invested_pct", "Avg capital invested", _fmt_pct),
    ("best_month", "Best month", _fmt_pct),
    ("worst_month", "Worst month", _fmt_pct),
    ("best_year", "Best year", _fmt_pct),
    ("worst_year", "Worst year", _fmt_pct),
    ("excess_return", "Excess return vs SPY", _fmt_pct),
    ("correlation_to_benchmark", "Correlation to SPY", _fmt_ratio),
]

TOURNAMENT_PREAMBLE = (
    "Strategies are ranked for ROBUSTNESS and risk-adjusted performance, not "
    "raw return. Every strategy in this table ran under the same capital, the "
    "same transaction-cost assumption, the same requested windows, the same "
    "canonical-calendar construction, and the same benchmark. A strategy "
    "looking too good is grounds for suspecting a bug or a bias, not for "
    "celebrating -- see docs/TOURNAMENT_DESIGN.md and docs/RED_TEAM.md."
)
ROBUSTNESS_SCORE_FORMULA_LINE = (
    "robustness_score = mean(fraction of windows beating SPY's total return, "
    "fraction of windows with a positive total return). Deliberately simple; "
    "its raw components are printed alongside so the composite never has to "
    "be trusted blindly."
)
ROBUSTNESS_COVERAGE_LINE = (
    "Denominators are the number of windows the strategy was EXPECTED to run "
    "(every window that ran for at least one strategy), NOT just the windows it "
    "survived -- so a missing/failed window counts against the score exactly "
    "like a lost one, and a strategy that only cleared its easy windows cannot "
    "outrank a full-coverage peer. The 'ran/exp' column shows coverage; any "
    "strategy with missing windows is flagged below the table."
)


def _write_tournament_summary_csv(path: Path, metrics_by_window: dict[str, dict[str, dict]]) -> None:
    rows = []
    for window_label, per_strategy in metrics_by_window.items():
        for strategy_label, m in per_strategy.items():
            row = {"window": window_label, "strategy": strategy_label}
            for key, _name, _fmt in TOURNAMENT_METRIC_FIELDS:
                row[key] = m.get(key)
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_tournament_table(lines: list, per_strategy: dict[str, dict], name_w: int = 30, col_w: int = 26) -> None:
    labels = list(per_strategy.keys())
    lines.append(f"{'Metric':<{name_w}}" + "".join(f"{lbl:>{col_w}}" for lbl in labels))
    for key, label_name, fmt in TOURNAMENT_METRIC_FIELDS:
        cells = "".join(f"{fmt(per_strategy[lbl].get(key)):>{col_w}}" for lbl in labels)
        lines.append(f"{label_name:<{name_w}}{cells}")


def write_tournament_report(
    metrics_by_window: dict[str, dict[str, dict]],
    window_ranges: dict[str, str],
    describe_by_strategy: dict[str, dict],
    run_config: dict,
    robustness: dict[str, dict] | None = None,
    cost_sensitivity: dict[str, dict[float, dict]] | None = None,
    param_sensitivity: dict[str, dict[str, dict]] | None = None,
    param_rationale: dict[str, dict[str, str]] | None = None,
    failures: list[tuple[str, str, str]] | None = None,
    universe_exclusions: dict[str, dict[str, list]] | None = None,
    output_dir: str = config.OUTPUT_DIR,
) -> Path:
    """Full tournament artifact set. `metrics_by_window` is
    `{window_label: {strategy_or_"SPY": metrics}}`; `robustness` is the
    per-strategy cross-window components dict (None for single-window runs);
    `cost_sensitivity` is `{strategy: {cost_bps: metrics}}`;
    `param_sensitivity` is `{strategy: {variant_label_or_"baseline": metrics}}`
    with `param_rationale[strategy][variant_label]` explaining each variant;
    `failures` lists (window, strategy, error) runs that could not complete;
    `universe_exclusions` is `{window: {strategy: [(ticker, reason), ...]}}`
    of tickers dropped per strategy/window (insufficient listing history for
    that window, no data, or in-range gaps) -- surfaced so a survivorship-
    biased snapshot universe's per-window exclusions are explicit."""
    ts = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
    dirname = f"{ts}_tournament_{run_config.get('requested_start', 'na')}_to_{run_config.get('requested_end', 'na')}"
    run_dir = Path(output_dir) / dirname
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_tournament_summary_csv(run_dir / "tournament_summary.csv", metrics_by_window)

    lines: list[str] = []
    lines.append("=== Strategy Tournament Report ===")
    lines.append("")
    lines.append("** RESEARCH / EDUCATIONAL USE ONLY. NOT FINANCIAL ADVICE. **")
    lines.append("Past performance does not indicate future results. No live trades were placed.")
    lines.append("")
    lines.append(TOURNAMENT_PREAMBLE)
    lines.append("")
    lines.append(EXECUTION_MODEL_LINE)
    lines.append(ADJUSTED_PRICE_LINE)
    lines.append(PRETAX_WARNING_LINE)
    lines.append(SURVIVORSHIP_WARNING)
    lines.append("")
    lines.append(f"Requested window: {run_config.get('requested_start')} to {run_config.get('requested_end')}")
    lines.append(f"Capital: ${run_config.get('capital', 0):,.2f}  |  Cost assumption: {run_config.get('cost_bps')} bps")
    lines.append(f"Strategies: {', '.join(run_config.get('strategies', []))}")
    lines.append(f"Windows tested: {', '.join(window_ranges.keys())}")
    for w, rng in window_ranges.items():
        lines.append(f"  {w}: {rng}")

    if failures:
        lines.append("")
        lines.append("--- Runs that could NOT complete (excluded from tables below) ---")
        for window_label, strategy_label, err in failures:
            lines.append(f"  [{window_label}] {strategy_label}: {err}")

    if universe_exclusions:
        lines.append("")
        lines.append("--- Per-strategy/window universe exclusions (listing-history filter) ---")
        lines.append(
            "Tickers dropped for a specific strategy/window because they lacked enough "
            "listing history to warm up that strategy's indicators for that window (a "
            "current-snapshot universe like us_50b contains names that did not exist for "
            "older windows). These are EXCLUDED, not run with missing history -- so the "
            "same rules run only on names actually tradable in each window. The full "
            "per-ticker reason list is in each strategy/window's own report.txt."
        )
        for window_label in metrics_by_window:
            per_strategy = universe_exclusions.get(window_label)
            if not per_strategy:
                continue
            lines.append(f"  Window {window_label}:")
            for strategy_label, dropped in per_strategy.items():
                if not dropped:
                    continue
                sample = ", ".join(t for t, _r in dropped[:8])
                more = f", +{len(dropped) - 8} more" if len(dropped) > 8 else ""
                lines.append(f"    {strategy_label}: {len(dropped)} excluded ({sample}{more})")

    for window_label, per_strategy in metrics_by_window.items():
        lines.append("")
        lines.append(f"--- Window {window_label} ({window_ranges.get(window_label, 'n/a')}) ---")
        _write_tournament_table(lines, per_strategy)

    if robustness:
        lines.append("")
        lines.append("--- Cross-window robustness (per strategy) ---")
        lines.append(ROBUSTNESS_SCORE_FORMULA_LINE)
        lines.append(ROBUSTNESS_COVERAGE_LINE)
        name_w, col_w = 26, 16
        header = (
            f"{'Strategy':<{name_w}}{'ran/exp':>{col_w}}{'beats SPY %':>{col_w}}"
            f"{'positive %':>{col_w}}{'worst DD':>{col_w}}{'dispersion':>{col_w}}{'SCORE':>{col_w}}"
        )
        lines.append(header)
        incomplete = []
        for strat, comp in sorted(robustness.items(), key=lambda kv: -kv[1]["robustness_score"]):
            ran = comp.get("num_windows_ran", comp.get("num_windows"))
            exp = comp.get("num_windows_expected", ran)
            coverage = f"{ran}/{exp}"
            flag = "" if comp.get("full_coverage", ran == exp) else "  <- INCOMPLETE"
            lines.append(
                f"{strat:<{name_w}}{coverage:>{col_w}}"
                f"{_fmt_pct(comp['pct_windows_beats_spy_return']):>{col_w}}"
                f"{_fmt_pct(comp['pct_windows_positive_return']):>{col_w}}"
                f"{_fmt_pct(comp['worst_window_max_drawdown']):>{col_w}}"
                f"{_fmt_pct(comp['return_dispersion']):>{col_w}}"
                f"{_fmt_ratio(comp['robustness_score']):>{col_w}}{flag}"
            )
            if comp.get("num_missing_windows", 0) > 0:
                incomplete.append((strat, comp["num_missing_windows"], exp))
        if incomplete:
            lines.append("")
            for strat, missing, exp in incomplete:
                lines.append(
                    f"  ! {strat} is missing {missing} of {exp} expected windows -- its score is "
                    f"penalized for them (they count as neither positive nor beating SPY). Missing "
                    f"windows usually mean the strategy FAILED there (e.g. too few tickers with "
                    f"enough history); do not read its score as validated over the full period."
                )

    if cost_sensitivity:
        lines.append("")
        lines.append("--- Cost sensitivity (total return / excess vs SPY at each cost level) ---")
        lines.append(
            "A strategy whose excess return vs SPY flips sign as costs rise has no "
            "margin for real-world frictions (spreads, slippage) beyond the modeled "
            "per-trade cost -- treat it as NOT beating SPY."
        )
        cost_rows = []
        for strat, by_cost in cost_sensitivity.items():
            for bps, m in sorted(by_cost.items()):
                cost_rows.append({
                    "strategy": strat, "cost_bps": bps,
                    "total_return": m.get("total_return"),
                    "excess_return_vs_spy": m.get("excess_return"),
                    "cost_drag_pct": m.get("cost_drag_pct"),
                })
                lines.append(
                    f"  {strat:<26}{bps:>6.1f} bps   total {_fmt_pct(m.get('total_return')):>10}   "
                    f"excess vs SPY {_fmt_pct(m.get('excess_return')):>10}   "
                    f"cost drag {_fmt_pct(m.get('cost_drag_pct')):>8}"
                )
            base_excess = by_cost.get(min(by_cost), {}).get("excess_return")
            worst_excess = by_cost.get(max(by_cost), {}).get("excess_return")
            if base_excess is not None and worst_excess is not None and base_excess > 0 and worst_excess <= 0:
                lines.append(f"  ^ WARNING: {strat}'s edge vs SPY disappears at higher costs (sign flip).")
        pd.DataFrame(cost_rows).to_csv(run_dir / "cost_sensitivity.csv", index=False)

    if param_sensitivity:
        lines.append("")
        lines.append("--- Parameter sensitivity (small disclosed variants; NEVER auto-selected) ---")
        lines.append(
            "Each variant nudges ONE disclosed knob (rationale shown). Large spread "
            "between variants = fragile parameter = weaker evidence. No variant's "
            "result is ever promoted to a default by this report."
        )
        param_rows = []
        for strat, by_variant in param_sensitivity.items():
            lines.append(f"  {strat}:")
            for variant_label, m in by_variant.items():
                rationale = (param_rationale or {}).get(strat, {}).get(variant_label, "")
                param_rows.append({
                    "strategy": strat, "variant": variant_label,
                    "total_return": m.get("total_return"),
                    "sharpe_ratio": m.get("sharpe_ratio"),
                    "max_drawdown": m.get("max_drawdown"),
                    "excess_return_vs_spy": m.get("excess_return"),
                    "rationale": rationale,
                })
                lines.append(
                    f"    {variant_label:<18} total {_fmt_pct(m.get('total_return')):>10}   "
                    f"Sharpe {_fmt_ratio(m.get('sharpe_ratio')):>6}   "
                    f"maxDD {_fmt_pct(m.get('max_drawdown')):>9}   "
                    f"excess {_fmt_pct(m.get('excess_return')):>10}"
                    + (f"   ({rationale})" if rationale else "")
                )
            returns = [m.get("total_return") for m in by_variant.values() if m.get("total_return") is not None]
            excesses = [m.get("excess_return") for m in by_variant.values() if m.get("excess_return") is not None]
            if len(returns) > 1:
                spread = max(returns) - min(returns)
                lines.append(f"    -> total-return spread across variants: {_fmt_pct(spread)}")
            if excesses and min(excesses) <= 0 <= max(excesses):
                lines.append(
                    "    -> WARNING: beat-SPY conclusion FLIPS across these variants -- "
                    "treat this strategy's edge as parameter-fragile."
                )
        pd.DataFrame(param_rows).to_csv(run_dir / "param_sensitivity.csv", index=False)

    lines.append("")
    lines.append("--- Strategy assumptions & parameters (self-declared via describe()) ---")
    for strat, info in describe_by_strategy.items():
        lines.append(f"  {strat} (family: {info.get('family')}, universe size: {info.get('universe_size')})")
        for k, v in (info.get("params") or {}).items():
            lines.append(f"    param {k} = {v}")
        for a in info.get("assumptions") or []:
            lines.append(f"    ! {a}")

    with open(run_dir / "tournament_report.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    json_payload = {
        "run_config": run_config,
        "windows": window_ranges,
        "results": metrics_by_window,
        "robustness": robustness,
        "cost_sensitivity": (
            {s: {str(b): m for b, m in bc.items()} for s, bc in cost_sensitivity.items()}
            if cost_sensitivity else None
        ),
        "param_sensitivity": param_sensitivity,
        "describe_by_strategy": describe_by_strategy,
        "failures": failures,
    }
    with open(run_dir / "tournament.json", "w") as f:
        json.dump(json_payload, f, indent=2, default=str)

    return run_dir


# --- Robustness report: allocation mixes x historical windows vs. SPY ---
# Diagnostics only -- see src/robustness.py's module docstring for why
# allocation mixes are capital-weighted blends rather than separately
# re-run backtests.

def _write_robustness_summary_csv(path: Path, all_window_metrics: dict[str, dict[str, dict]]) -> None:
    rows = []
    for window_label, alloc_dict in all_window_metrics.items():
        for label, m in alloc_dict.items():
            row = {"window": window_label, "allocation": label}
            for key, _name, _fmt in COMPARISON_METRIC_FIELDS:
                row[key] = m.get(key)
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _compute_robustness_rankings(
    alloc_only_by_window: dict[str, dict[str, dict]], spy_metrics_by_window: dict[str, dict]
) -> dict:
    rank_fields = [
        ("total_return", "avg_rank_total_return"),
        ("sharpe_ratio", "avg_rank_sharpe"),
        ("max_drawdown", "avg_rank_max_drawdown"),
        ("calmar_ratio", "avg_rank_calmar"),
    ]
    avg_ranks_by_field = {out_key: average_ranks(alloc_only_by_window, key) for key, out_key in rank_fields}
    return {
        **avg_ranks_by_field,
        "pct_windows_beats_spy_return": beats_spy_fraction(alloc_only_by_window, spy_metrics_by_window, "total_return"),
        "pct_windows_lower_drawdown_than_spy": beats_spy_fraction(
            alloc_only_by_window, spy_metrics_by_window, "max_drawdown"
        ),
        "best_by_window": {
            key: {w: rank_allocations(wm, key) for w, wm in alloc_only_by_window.items()}
            for key, _out_key in rank_fields
        },
    }


def _write_robustness_rankings_csv(path: Path, rankings: dict) -> None:
    rows = []
    for label, _w_sr, _w_mr in ALLOCATION_MIXES:
        rows.append({
            "allocation": label,
            "avg_rank_total_return": rankings["avg_rank_total_return"].get(label),
            "avg_rank_sharpe": rankings["avg_rank_sharpe"].get(label),
            "avg_rank_max_drawdown": rankings["avg_rank_max_drawdown"].get(label),
            "avg_rank_calmar": rankings["avg_rank_calmar"].get(label),
            "pct_windows_beats_spy_return": rankings["pct_windows_beats_spy_return"].get(label),
            "pct_windows_lower_drawdown_than_spy": rankings["pct_windows_lower_drawdown_than_spy"].get(label),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_robustness_heatmap_csv(path: Path, all_window_metrics: dict[str, dict[str, dict]]) -> None:
    """Wide matrix (allocation x window) of TOTAL RETURN -- the headline
    metric most useful to visualize as a heatmap. Other metrics are
    available in the long-format robustness_summary.csv if a different
    heatmap is needed."""
    window_labels = list(all_window_metrics.keys())
    row_labels: list[str] = []
    for wm in all_window_metrics.values():
        for label in wm:
            if label not in row_labels:
                row_labels.append(label)
    rows = []
    for label in row_labels:
        row = {"allocation": label}
        for w in window_labels:
            row[w] = all_window_metrics[w].get(label, {}).get("total_return")
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_robustness_summary_txt(
    path: Path,
    all_window_metrics: dict[str, dict[str, dict]],
    window_ranges: dict[str, str],
    rankings: dict,
    tradeoff_rows: list[dict],
    run_config: dict,
) -> None:
    lines = []
    lines.append("=== Robustness Testing Report ===")
    lines.append("")
    lines.append("** RESEARCH / EDUCATIONAL USE ONLY. NOT FINANCIAL ADVICE. **")
    lines.append(
        "Diagnostics only -- no RSI/SMA/holding-day/top-K/stop-loss/universe/"
        "transaction-cost parameter was tuned to produce this report."
    )
    lines.append(
        "Allocation mixes are capital-weighted blends of independently-run "
        "mean_reversion and sector_rotation sleeves (each run ONCE per window "
        "at full capital), not separately re-run backtests per allocation. The "
        "blend preserves each sleeve's raw equity curve (including first-day "
        "fill/transaction-cost drag) and scales it by allocation weight -- see "
        "src/robustness.py's module docstring for the exact formula and its "
        "edge cases."
    )
    lines.append("")
    lines.append(f"Windows tested: {', '.join(window_ranges.keys())}")
    for w, rng in window_ranges.items():
        lines.append(f"  {w}: {rng}")

    col_w = 15
    name_w = 30
    for window_label, alloc_dict in all_window_metrics.items():
        labels = list(alloc_dict.keys())
        lines.append("")
        lines.append(f"--- Window {window_label} ({window_ranges.get(window_label, 'n/a')}) ---")
        lines.append(f"{'Metric':<{name_w}}" + "".join(f"{lbl:>{col_w}}" for lbl in labels))
        for key, label_name, fmt in COMPARISON_METRIC_FIELDS:
            cells = "".join(f"{fmt(alloc_dict[lbl].get(key)):>{col_w}}" for lbl in labels)
            lines.append(f"{label_name:<{name_w}}{cells}")

    lines.append("")
    lines.append("--- Best allocation per window ---")
    for metric_key, metric_label in [
        ("total_return", "Total return"), ("sharpe_ratio", "Sharpe"),
        ("max_drawdown", "Max drawdown"), ("calmar_ratio", "Calmar"),
    ]:
        lines.append(f"By {metric_label}:")
        for w, ranks in rankings["best_by_window"][metric_key].items():
            best_label = next((lbl for lbl, r in ranks.items() if r == 1), "n/a")
            lines.append(f"  {w}: {best_label}")

    lines.append("")
    lines.append("--- Average rank across windows (1 = best, excludes SPY) ---")
    lines.append(
        f"{'Allocation':<{name_w}}{'Return':>{col_w}}{'Sharpe':>{col_w}}{'MaxDD':>{col_w}}{'Calmar':>{col_w}}"
    )
    for label, _w_sr, _w_mr in ALLOCATION_MIXES:
        lines.append(
            f"{label:<{name_w}}"
            f"{_fmt_ratio(rankings['avg_rank_total_return'].get(label)):>{col_w}}"
            f"{_fmt_ratio(rankings['avg_rank_sharpe'].get(label)):>{col_w}}"
            f"{_fmt_ratio(rankings['avg_rank_max_drawdown'].get(label)):>{col_w}}"
            f"{_fmt_ratio(rankings['avg_rank_calmar'].get(label)):>{col_w}}"
        )

    lines.append("")
    lines.append("--- How often each allocation beats SPY, across tested windows ---")
    lines.append(f"{'Allocation':<{name_w}}{'Beats SPY return':>{col_w + 6}}{'Lower DD than SPY':>{col_w + 6}}")
    for label, _w_sr, _w_mr in ALLOCATION_MIXES:
        lines.append(
            f"{label:<{name_w}}"
            f"{_fmt_pct(rankings['pct_windows_beats_spy_return'].get(label)):>{col_w + 6}}"
            f"{_fmt_pct(rankings['pct_windows_lower_drawdown_than_spy'].get(label)):>{col_w + 6}}"
        )

    lines.append("")
    lines.append("--- Does mean reversion's drawdown protection justify its cost drag? ---")
    lines.append(
        "(vs. the 100%-sector-rotation baseline in the same window; a simple "
        "heuristic -- drawdown improved AND by more percentage points than the "
        "cost-drag increase -- not a rigorous risk-adjusted verdict.)"
    )
    for row in tradeoff_rows:
        verdict = "YES" if row["worth_it"] else "no"
        lines.append(
            f"  {row['window']:<12}{row['allocation']:<32}"
            f"drawdown {_fmt_pct(row['drawdown_improvement_pts'])} better, "
            f"cost drag {_fmt_pct(row['cost_drag_increase_pts'])} higher -> {verdict}"
        )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_robustness_report(
    all_window_metrics: dict[str, dict[str, dict]],
    window_ranges: dict[str, str],
    run_config: dict,
    output_dir: str = config.OUTPUT_DIR,
) -> Path:
    """Writes the full robustness-testing artifact set: `all_window_metrics`
    is `{window_label: {allocation_label_or_"SPY": metrics_dict}}` for every
    tested window, already computed by the caller (see cli.py's `robustness`
    mode and src/robustness.py's blending functions)."""
    spy_metrics_by_window = {w: m.get("SPY", {}) for w, m in all_window_metrics.items()}
    alloc_only_by_window = {
        w: {label: m for label, m in wm.items() if label != "SPY"} for w, wm in all_window_metrics.items()
    }
    rankings = _compute_robustness_rankings(alloc_only_by_window, spy_metrics_by_window)
    tradeoff_rows = mean_reversion_tradeoff(alloc_only_by_window)

    ts = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
    dirname = f"{ts}_robustness_{run_config.get('requested_start', 'na')}_to_{run_config.get('requested_end', 'na')}"
    run_dir = Path(output_dir) / dirname
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_robustness_summary_csv(run_dir / "robustness_summary.csv", all_window_metrics)
    _write_robustness_rankings_csv(run_dir / "robustness_rankings.csv", rankings)
    _write_robustness_heatmap_csv(run_dir / "robustness_heatmap_data.csv", all_window_metrics)
    _write_robustness_summary_txt(
        run_dir / "robustness_summary.txt", all_window_metrics, window_ranges, rankings, tradeoff_rows, run_config
    )

    # "equity_curve" (a raw pd.Series, put there for the blending math's own
    # use) doesn't belong in a JSON summary -- everything a reader needs is
    # already in the scalar metric fields.
    json_safe_results = {
        w: {label: {k: v for k, v in m.items() if k != "equity_curve"} for label, m in alloc_dict.items()}
        for w, alloc_dict in all_window_metrics.items()
    }
    with open(run_dir / "robustness_summary.json", "w") as f:
        json.dump(
            {
                "run_config": run_config,
                "windows": window_ranges,
                "results": json_safe_results,
                "rankings": rankings,
                "mean_reversion_tradeoff": tradeoff_rows,
            },
            f, indent=2, default=str,
        )

    return run_dir


# --- Portfolio report: one account allocated across weighted sleeves ------------
# Diagnostics only -- formats a PortfolioResult the caller (cli.py's portfolio
# mode / src/portfolio.py) already computed; no strategy parameter is touched.

STATIC_ALLOCATION_DISCLOSURE = (
    "STATIC ALLOCATION (v1): capital was allocated ONCE at the start (weight x "
    "starting capital per sleeve). Sleeve weights then DRIFT with performance "
    "and NO cash is ever transferred between sleeves -- there is no periodic "
    "rebalancing back to the target weights. Each sleeve is a fully independent "
    "engine run (its own cash/shares/lots); the portfolio curve is the sum of "
    "the sleeves' independent equity curves over their common date "
    "intersection. No cash is shared and no capital is double-counted."
)


def _write_portfolio_summary_csv(path: Path, metrics_by_label: dict[str, dict], labels: list[str]) -> None:
    rows = []
    for lbl in labels:
        row = {"strategy": lbl}
        for key, _name, _fmt in TOURNAMENT_METRIC_FIELDS:
            row[key] = metrics_by_label[lbl].get(key)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_portfolio_equity_csv(path: Path, pf: "PortfolioResult") -> None:
    """One row per common-window date: the total portfolio equity plus each
    sleeve's equity aligned to that same common index (so the columns visibly
    sum to the portfolio column, proving the combination is a plain sum)."""
    common = pf.combined_result.equity_curve.index
    frame = {"portfolio": pf.combined_result.equity_curve}
    for sleeve in pf.sleeves:
        frame[sleeve.strategy] = sleeve.result.equity_curve.reindex(common)
    df = pd.DataFrame(frame, index=common)
    df.index.name = "date"
    df.to_csv(path)


def _write_portfolio_sleeves_csv(path: Path, pf: "PortfolioResult") -> None:
    rows = []
    for sleeve in pf.sleeves:
        rows.append({
            "strategy": sleeve.strategy,
            "target_weight": sleeve.weight,
            "allocated_capital": sleeve.allocated_capital,
            "final_value": sleeve.final_value,
            "ending_weight": sleeve.ending_weight,
            "pnl_contribution": sleeve.pnl_contribution,
            "cost_contribution": sleeve.cost_contribution,
            "total_return": sleeve.metrics.get("total_return"),
            "cagr": sleeve.metrics.get("cagr"),
            "max_drawdown": sleeve.metrics.get("max_drawdown"),
            "sharpe_ratio": sleeve.metrics.get("sharpe_ratio"),
            "total_turnover": sleeve.metrics.get("total_turnover"),
            "total_transaction_costs": sleeve.metrics.get("total_transaction_costs"),
        })
    # A reconciliation total row: allocated capital and final value sum to the
    # portfolio's, ending weights sum to 1.0, P&L contributions sum to the
    # portfolio's total dollar gain -- all visible in one place.
    rows.append({
        "strategy": "PORTFOLIO_TOTAL",
        "target_weight": sum(w for _n, w in pf.weights),
        "allocated_capital": sum(s.allocated_capital for s in pf.sleeves),
        "final_value": sum(s.final_value for s in pf.sleeves),
        "ending_weight": sum(s.ending_weight for s in pf.sleeves),
        "pnl_contribution": sum(s.pnl_contribution for s in pf.sleeves),
        "cost_contribution": sum(s.cost_contribution for s in pf.sleeves),
        "total_return": pf.metrics.get("total_return"),
        "cagr": pf.metrics.get("cagr"),
        "max_drawdown": pf.metrics.get("max_drawdown"),
        "sharpe_ratio": pf.metrics.get("sharpe_ratio"),
        "total_turnover": pf.metrics.get("total_turnover"),
        "total_transaction_costs": pf.metrics.get("total_transaction_costs"),
    })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_portfolio_txt(
    path: Path, pf: "PortfolioResult", run_config: dict, describe_by_strategy: dict[str, dict]
) -> None:
    m = pf.metrics
    lines = []
    lines.append("=== Portfolio Combination Report ===")
    lines.append("")
    lines.append("** RESEARCH / EDUCATIONAL USE ONLY. NOT FINANCIAL ADVICE. **")
    lines.append("Past performance does not indicate future results. No live trades were placed.")
    lines.append("")
    lines.append(STATIC_ALLOCATION_DISCLOSURE)
    lines.append("")
    lines.append(f"Requested window: {run_config.get('requested_start')} to {run_config.get('requested_end')}")
    lines.append(
        f"Effective (common) window used: {pf.common_start.date()} to {pf.common_end.date()} "
        f"(intersection of all sleeves' valid ranges)"
    )
    lines.append(f"Starting capital: ${pf.total_capital:,.2f}")
    lines.append(f"Transaction cost assumption: {run_config.get('cost_bps')} bps per trade")
    lines.append(
        f"Fractional shares: {'enabled' if run_config.get('fractional_shares') else 'disabled'}"
    )
    if pf.skipped_zero_weight:
        lines.append(
            f"Skipped zero-weight sleeves (allocated $0, contribute nothing): "
            f"{', '.join(pf.skipped_zero_weight)}"
        )

    lines.append("")
    lines.append("--- Allocation (start -> end) ---")
    header = (
        f"{'Strategy':<24}{'Target wt':>12}{'Allocated $':>16}"
        f"{'Final $':>16}{'End wt':>12}{'P&L $':>16}"
    )
    lines.append(header)
    for sleeve in pf.sleeves:
        lines.append(
            f"{sleeve.strategy:<24}{_fmt_pct(sleeve.weight):>12}"
            f"{_fmt_num(sleeve.allocated_capital):>16}{_fmt_num(sleeve.final_value):>16}"
            f"{_fmt_pct(sleeve.ending_weight):>12}{_fmt_num(sleeve.pnl_contribution):>16}"
        )
    lines.append(
        f"{'PORTFOLIO':<24}{_fmt_pct(sum(w for _n, w in pf.weights)):>12}"
        f"{_fmt_num(pf.total_capital):>16}{_fmt_num(m.get('final_equity')):>16}"
        f"{_fmt_pct(sum(s.ending_weight for s in pf.sleeves)):>12}"
        f"{_fmt_num(m.get('final_equity', 0) - pf.total_capital):>16}"
    )
    lines.append(
        "(Target weights are the START allocation; end weights are what those "
        "sleeves drifted to. They differ precisely because this is a static, "
        "non-rebalanced allocation.)"
    )

    lines.append("")
    lines.append(EXECUTION_MODEL_LINE)
    lines.append(ADJUSTED_PRICE_LINE)
    lines.append(PRETAX_WARNING_LINE)
    # Any stock-universe sleeve (momentum, mean_reversion*) inherits the
    # survivorship-biased research universe -- keep that warning explicit.
    if any(
        d.get("universe_size", 0) and d.get("family") not in ("sector_rotation", "regime_switch")
        for d in describe_by_strategy.values()
    ):
        lines.append("")
        lines.append(SURVIVORSHIP_WARNING)

    lines.append("")
    lines.append("--- Portfolio performance ---")
    lines.append(f"Final equity: ${_fmt_num(m.get('final_equity'))}")
    lines.append(f"Total return: {_fmt_pct(m.get('total_return'))}")
    lines.append(f"CAGR: {_fmt_pct(m.get('cagr'))}")
    if m.get("short_period_warning"):
        lines.append("  (WARNING: effective period < 90 days -- CAGR/Sharpe are unstable over short windows)")
    lines.append(f"Max drawdown: {_fmt_pct(m.get('max_drawdown'))}")
    lines.append(f"Max drawdown duration: {m.get('max_drawdown_duration_days')} calendar days")
    for key, label in [
        ("sharpe_ratio", "Sharpe ratio (rf=0)"),
        ("sortino_ratio", "Sortino ratio (rf=0)"),
        ("calmar_ratio", "Calmar ratio (CAGR / |max DD|)"),
    ]:
        val = m.get(key)
        lines.append(f"{label}: {val:.2f}" if pd.notna(val) else f"{label}: n/a")
    lines.append(
        f"Best month: {_fmt_pct(m.get('best_month'))}  |  Worst month: {_fmt_pct(m.get('worst_month'))}"
    )
    lines.append(
        f"Best year: {_fmt_pct(m.get('best_year'))}  |  Worst year: {_fmt_pct(m.get('worst_year'))}"
    )

    lines.append("")
    lines.append("--- Calendar-year returns (portfolio) ---")
    yearly = annual_returns(pf.combined_result.equity_curve)
    for y in sorted(yearly):
        lines.append(f"  {y}: {_fmt_pct(yearly[y])}")

    lines.append("")
    lines.append("--- Cost & turnover (combined, over the common window) ---")
    lines.append(f"Total transaction costs paid: ${_fmt_num(m.get('total_transaction_costs'))}")
    lines.append(f"Total turnover: ${_fmt_num(m.get('total_turnover'))}")
    lines.append(f"Cost drag (% of starting capital): {_fmt_pct(m.get('cost_drag_pct'))}")
    lines.append(f"Round-trip trades: {m.get('num_trades')}  |  Transactions: {m.get('num_transactions')}")

    lines.append("")
    lines.append("--- Exposure ---")
    lines.append(f"Days with any position: {_fmt_pct(m.get('days_with_any_position_pct'))}")
    lines.append(f"Average capital invested: {_fmt_pct(m.get('average_capital_invested_pct'))}")

    lines.append("")
    lines.append("--- Benchmark (adjusted-price SPY total-return proxy, common window) ---")
    lines.append(f"SPY total return: {_fmt_pct(pf.spy_metrics.get('total_return'))}")
    lines.append(f"SPY CAGR: {_fmt_pct(pf.spy_metrics.get('cagr'))}")
    lines.append(f"SPY max drawdown: {_fmt_pct(pf.spy_metrics.get('max_drawdown'))}")
    lines.append(f"Excess return vs. SPY: {_fmt_pct(m.get('excess_return'))}")
    corr = m.get("correlation_to_benchmark")
    lines.append(f"Correlation to SPY: {corr:.2f}" if pd.notna(corr) else "Correlation to SPY: n/a")

    lines.append("")
    lines.append("--- Per-sleeve breakdown (each sleeve's own standalone metrics) ---")
    metrics_by_label = {s.strategy: s.metrics for s in pf.sleeves}
    _write_tournament_table(lines, metrics_by_label, name_w=30, col_w=22)
    lines.append("")
    lines.append(
        "Contribution to portfolio P&L (sums to the portfolio's total $ gain), "
        "and each sleeve's share of combined transaction costs:"
    )
    total_pnl = sum(s.pnl_contribution for s in pf.sleeves)
    total_costs = sum(s.cost_contribution for s in pf.sleeves)
    for sleeve in pf.sleeves:
        pnl_share = (sleeve.pnl_contribution / total_pnl) if total_pnl else float("nan")
        cost_share = (sleeve.cost_contribution / total_costs) if total_costs else float("nan")
        lines.append(
            f"  {sleeve.strategy}: P&L ${_fmt_num(sleeve.pnl_contribution)} "
            f"({_fmt_pct(pnl_share)} of portfolio P&L)  |  "
            f"costs ${_fmt_num(sleeve.cost_contribution)} ({_fmt_pct(cost_share)} of combined costs)"
        )

    # Each sleeve's disclosed assumptions -- nothing hidden behind the blend.
    lines.append("")
    lines.append("--- Sleeve assumptions (from each strategy's describe()) ---")
    for name, desc in describe_by_strategy.items():
        assumptions = desc.get("assumptions") or []
        if assumptions:
            lines.append(f"  {name}:")
            for a in assumptions:
                lines.append(f"    - {a}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_portfolio_report(
    pf: "PortfolioResult",
    run_config: dict,
    describe_by_strategy: dict[str, dict],
    output_dir: str = config.OUTPUT_DIR,
) -> Path:
    """Write the portfolio-combination artifact set (portfolio_report.txt,
    portfolio_summary.csv, portfolio_equity.csv, portfolio_sleeves.csv,
    portfolio.json) for an already-computed PortfolioResult. Diagnostics only:
    formats results, runs no backtest, touches no strategy parameter."""
    ts = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
    dirname = f"{ts}_portfolio_{run_config.get('requested_start')}_to_{run_config.get('requested_end')}"
    run_dir = Path(output_dir) / dirname
    run_dir.mkdir(parents=True, exist_ok=True)

    # Summary table rows: each sleeve, the portfolio, and SPY.
    metrics_by_label: dict[str, dict] = {s.strategy: s.metrics for s in pf.sleeves}
    metrics_by_label["portfolio"] = pf.metrics
    metrics_by_label["SPY"] = pf.spy_metrics
    labels = [s.strategy for s in pf.sleeves] + ["portfolio", "SPY"]

    _write_portfolio_summary_csv(run_dir / "portfolio_summary.csv", metrics_by_label, labels)
    _write_portfolio_equity_csv(run_dir / "portfolio_equity.csv", pf)
    _write_portfolio_sleeves_csv(run_dir / "portfolio_sleeves.csv", pf)
    _write_portfolio_txt(run_dir / "portfolio_report.txt", pf, run_config, describe_by_strategy)
    _write_equity_png(pf.combined_result, run_dir / "portfolio_equity.png")

    json_out = dict(run_config)
    json_out["weights"] = {name: w for name, w in pf.weights}
    json_out["total_capital"] = pf.total_capital
    json_out["common_window"] = {"start": str(pf.common_start.date()), "end": str(pf.common_end.date())}
    json_out["skipped_zero_weight"] = pf.skipped_zero_weight
    json_out["portfolio_metrics"] = pf.metrics
    json_out["spy_metrics"] = pf.spy_metrics
    json_out["sleeves"] = [
        {
            "strategy": s.strategy, "target_weight": s.weight, "allocated_capital": s.allocated_capital,
            "final_value": s.final_value, "ending_weight": s.ending_weight,
            "pnl_contribution": s.pnl_contribution, "cost_contribution": s.cost_contribution,
            "metrics": s.metrics,
        }
        for s in pf.sleeves
    ]
    json_out["describe"] = describe_by_strategy
    with open(run_dir / "portfolio.json", "w") as f:
        json.dump(json_out, f, indent=2, default=str)

    return run_dir


# --- Walk-forward report: out-of-sample validation across (train, test) folds ---
# Diagnostics only -- formats a WalkForwardResult the caller (cli.py's
# walk_forward mode / src/walk_forward.py) already computed.

WALK_FORWARD_SURVIVORSHIP_DISCLOSURE = (
    "WALK-FORWARD SCOPE: this validates that the portfolio's PARAMETERS are not "
    "overfit -- it evaluates only on test periods that did not influence the "
    "parameters used. It does NOT fix survivorship bias: the stock universe is a "
    "CURRENT snapshot of today's survivors (not a point-in-time constituent "
    "list) in EVERY fold, so even a clean out-of-sample result here is not "
    "evidence of a general, tradable edge. See docs/RED_TEAM.md A1 and the "
    "README's Known Limitations."
)


def _selected_variants_str(selected: dict[str, str]) -> str:
    return "; ".join(f"{k}={v}" for k, v in selected.items())


def _write_walk_forward_folds_csv(path: Path, wf: "WalkForwardResult") -> None:
    rows = []
    for f in wf.folds:
        rows.append({
            "fold": f.index,
            "train_start": f.train_start.date(),
            "train_end": f.train_end.date(),
            "test_start": f.test_start.date(),
            "test_end": f.test_end.date(),
            "actual_test_range": f.actual_test_range,
            "start_capital": f.start_capital,
            "final_equity": f.final_equity,
            "portfolio_test_return": f.test_return,
            "spy_test_return": f.spy_return,
            "excess_return": f.excess_return,
            "max_drawdown": f.max_drawdown,
            "sharpe_ratio": f.sharpe_ratio,
            "transaction_costs": f.transaction_costs,
            "selected_variants": _selected_variants_str(f.selected_variants),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_walk_forward_equity_csv(path: Path, wf: "WalkForwardResult") -> None:
    """The stitched out-of-sample equity curve. A `fold` column tags each date
    with the fold whose TEST period it belongs to, making it self-evident that
    the stitched curve contains only test-period dates."""
    rows = []
    for f in wf.folds:
        eq = f.portfolio_result.combined_result.equity_curve
        for d, v in eq.items():
            rows.append({"date": d.date(), "equity": float(v), "fold": f.index})
    df = pd.DataFrame(rows).drop_duplicates(subset="date", keep="first")
    df.to_csv(path, index=False)


def _write_walk_forward_txt(path: Path, wf: "WalkForwardResult") -> None:
    agg = wf.aggregate
    lines = []
    lines.append("=== Walk-Forward / Out-of-Sample Validation Report ===")
    lines.append("")
    lines.append("** RESEARCH / EDUCATIONAL USE ONLY. NOT FINANCIAL ADVICE. **")
    lines.append("Past performance does not indicate future results. No live trades were placed.")
    lines.append("")
    lines.append(WALK_FORWARD_SURVIVORSHIP_DISCLOSURE)
    lines.append("")
    mode = "OPTIMIZE (per-sleeve variant ranked on training, frozen for test)" if wf.optimize \
        else "FIXED parameters (shipped defaults; NO parameter selection -- clean baseline)"
    lines.append(f"Mode: {mode}")
    lines.append(
        f"Windows: {wf.window_mode}, train={wf.train_years}y test={wf.test_years}y "
        f"step={wf.step_years}y"
    )
    weight_str = ", ".join(f"{n}={w:.0%}" for n, w in wf.weights)
    lines.append(f"Portfolio weights: {weight_str}")
    lines.append(f"Starting capital: ${wf.total_capital:,.2f}")
    lines.append(
        "Fold boundaries: each fold's training window ends the day BEFORE its test window begins "
        "(train_end < test_start), so no test data can influence the parameters used. Capital "
        "compounds across folds -- each test period starts with the prior fold's ending equity, and "
        "re-establishes the portfolio from cash (fold boundaries incur re-entry costs; conservative)."
    )

    lines.append("")
    lines.append("--- Per-fold results (evaluated on the TEST period only) ---")
    header = (
        f"{'Fold':<5}{'Train':<26}{'Test':<26}{'Port ret':>10}{'SPY ret':>10}"
        f"{'Excess':>10}{'MaxDD':>10}{'Sharpe':>8}{'Costs $':>12}"
    )
    lines.append(header)
    for f in wf.folds:
        train = f"{f.train_start.date()}..{f.train_end.date()}"
        test = f"{f.test_start.date()}..{f.test_end.date()}"
        sharpe = f.sharpe_ratio
        sharpe_str = f"{sharpe:.2f}" if pd.notna(sharpe) else "n/a"
        lines.append(
            f"{f.index:<5}{train:<26}{test:<26}{_fmt_pct(f.test_return):>10}"
            f"{_fmt_pct(f.spy_return):>10}{_fmt_pct(f.excess_return):>10}"
            f"{_fmt_pct(f.max_drawdown):>10}{sharpe_str:>8}{_fmt_num(f.transaction_costs):>12}"
        )
    if wf.optimize:
        lines.append("")
        lines.append("Selected variant per fold (frozen from the training window):")
        for f in wf.folds:
            lines.append(f"  fold {f.index} ({f.test_start.date()}..{f.test_end.date()}): "
                         f"{_selected_variants_str(f.selected_variants)}")

    lines.append("")
    lines.append("--- Aggregate stitched out-of-sample result ---")
    lines.append(f"Out-of-sample span: {agg.get('oos_start')} to {agg.get('oos_end')} "
                 f"({agg.get('num_folds')} folds)")
    lines.append(f"Stitched final equity: ${_fmt_num(agg.get('final_equity'))}")
    lines.append(f"Stitched total return: {_fmt_pct(agg.get('stitched_total_return'))}")
    lines.append(f"Stitched CAGR: {_fmt_pct(agg.get('stitched_cagr'))}")
    lines.append(f"Stitched max drawdown: {_fmt_pct(agg.get('stitched_max_drawdown'))}")
    stitched_sharpe = agg.get("stitched_sharpe_ratio")
    lines.append(f"Stitched Sharpe (rf=0): {stitched_sharpe:.2f}" if pd.notna(stitched_sharpe)
                 else "Stitched Sharpe (rf=0): n/a")
    lines.append(f"Total transaction costs (all folds): ${_fmt_num(agg.get('total_transaction_costs'))}")
    lines.append(f"Test folds beating SPY: {_fmt_pct(agg.get('pct_folds_beating_spy'))} "
                 f"of {agg.get('num_folds')} folds")
    lines.append(f"Profitable test folds: {_fmt_pct(agg.get('pct_folds_profitable'))} "
                 f"of {agg.get('num_folds')} folds")

    lines.append("")
    lines.append(EXECUTION_MODEL_LINE)
    lines.append(ADJUSTED_PRICE_LINE)
    lines.append(PRETAX_WARNING_LINE)
    lines.append("")
    lines.append(
        "Read the aggregate as the honest number: it is the return an investor would have earned "
        "taking each period's decision using only prior data. A big in-sample backtest that decays "
        "to a weak stitched out-of-sample result is the classic overfitting signature."
    )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_walk_forward_report(
    wf: "WalkForwardResult",
    run_config: dict,
    output_dir: str = config.OUTPUT_DIR,
) -> Path:
    """Write the walk-forward artifact set (walk_forward_report.txt,
    walk_forward_folds.csv, walk_forward_equity.csv, walk_forward.json) for an
    already-computed WalkForwardResult. Diagnostics only."""
    ts = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
    dirname = f"{ts}_walk_forward_{run_config.get('requested_start')}_to_{run_config.get('requested_end')}"
    run_dir = Path(output_dir) / dirname
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_walk_forward_folds_csv(run_dir / "walk_forward_folds.csv", wf)
    _write_walk_forward_equity_csv(run_dir / "walk_forward_equity.csv", wf)
    _write_walk_forward_txt(run_dir / "walk_forward_report.txt", wf)
    if len(wf.stitched_equity):
        _plot_stitched_equity(wf, run_dir / "walk_forward_equity.png")

    json_out = dict(run_config)
    json_out["mode"] = "optimize" if wf.optimize else "fixed"
    json_out["window_mode"] = wf.window_mode
    json_out["train_years"] = wf.train_years
    json_out["test_years"] = wf.test_years
    json_out["step_years"] = wf.step_years
    json_out["weights"] = {n: w for n, w in wf.weights}
    json_out["total_capital"] = wf.total_capital
    json_out["aggregate"] = wf.aggregate
    json_out["folds"] = [
        {
            "fold": f.index,
            "train_start": str(f.train_start.date()), "train_end": str(f.train_end.date()),
            "test_start": str(f.test_start.date()), "test_end": str(f.test_end.date()),
            "start_capital": f.start_capital, "final_equity": f.final_equity,
            "portfolio_test_return": f.test_return, "spy_test_return": f.spy_return,
            "excess_return": f.excess_return, "max_drawdown": f.max_drawdown,
            "sharpe_ratio": f.sharpe_ratio, "transaction_costs": f.transaction_costs,
            "selected_variants": f.selected_variants,
        }
        for f in wf.folds
    ]
    with open(run_dir / "walk_forward.json", "w") as f:
        json.dump(json_out, f, indent=2, default=str)

    return run_dir


def _plot_stitched_equity(wf: "WalkForwardResult", path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    wf.stitched_equity.plot(ax=ax)
    # Shade fold boundaries so the stitched-from-test-periods structure is visible.
    for f in wf.folds:
        ax.axvline(f.test_start, color="grey", alpha=0.25, linestyle="--")
    ax.set_title("Stitched out-of-sample equity curve (test periods only)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --- Paper trading artifacts ----------------------------------------------------
# Prominent notices printed on every paper report (requirement 20). These are a
# hard, non-negotiable banner: this system never sends real orders.
PAPER_NOTICE_LINES = [
    "PAPER TRADING ONLY",
    "NO REAL ORDERS WERE SENT",
    "NOT FINANCIAL ADVICE",
    "NO BROKERAGE CONNECTION EXISTS",
    "RESULTS MAY DIFFER FROM LIVE EXECUTION",
]

PAPER_ORDER_COLUMNS = [
    "id", "sleeve", "ticker", "signal_date", "scheduled_fill_date", "intended_side",
    "target_weight", "requested_notional", "sizing_price", "reason", "status",
]
PAPER_TRADE_COLUMNS = [
    "id", "sleeve", "ticker", "signal_date", "fill_date", "side", "shares",
    "fill_price", "notional", "transaction_cost", "reason",
]
PAPER_POSITION_COLUMNS = ["sleeve", "ticker", "shares", "avg_cost", "last_price", "market_value"]
PAPER_SLEEVE_COLUMNS = [
    "strategy", "weight", "allocated_capital", "cash", "equity", "ending_weight",
    "transaction_costs", "turnover",
]
PAPER_RUN_LOG_COLUMNS = [
    "run_index", "timestamp", "paper_date", "data_cutoff_date", "next_session",
    "num_new_signals", "num_pending_created", "num_fills", "num_stale",
    "total_equity", "cumulative_return", "daily_return", "transaction_costs_run",
    "reconciliation_ok",
]


def _paper_banner() -> list[str]:
    line = "*" * 64
    out = [line]
    for n in PAPER_NOTICE_LINES:
        out.append(f"** {n}")
    out.append(line)
    return out


def _write_paper_orders_csv(path: Path, state: dict) -> None:
    rows = list(state.get("pending_orders", [])) + list(state.get("rejected_stale_orders", []))
    _write_csv(path, rows, PAPER_ORDER_COLUMNS)


def _write_paper_trades_csv(path: Path, state: dict) -> None:
    _write_csv(path, list(state.get("completed_trades", [])), PAPER_TRADE_COLUMNS)


def _write_paper_positions_csv(path: Path, state: dict) -> None:
    rows = []
    for name, sleeve in state.get("sleeves", {}).items():
        for p in sleeve.get("positions", []):
            rows.append({"sleeve": name, **p})
    _write_csv(path, rows, PAPER_POSITION_COLUMNS)


def _write_paper_sleeves_csv(path: Path, state: dict) -> None:
    sleeves = state.get("sleeves", {})
    rows = [
        {k: s.get(k) for k in PAPER_SLEEVE_COLUMNS}
        for s in sleeves.values()
    ]
    # Reconciliation total row: allocated sums to the account, ending weights to 1.
    if sleeves:
        rows.append({
            "strategy": "PAPER_TOTAL",
            "weight": sum(s.get("weight", 0.0) for s in sleeves.values()),
            "allocated_capital": sum(s.get("allocated_capital", 0.0) for s in sleeves.values()),
            "cash": sum(s.get("cash", 0.0) for s in sleeves.values()),
            "equity": sum(s.get("equity", 0.0) for s in sleeves.values()),
            "ending_weight": sum(s.get("ending_weight", 0.0) for s in sleeves.values()),
            "transaction_costs": sum(s.get("transaction_costs", 0.0) for s in sleeves.values()),
            "turnover": sum(s.get("turnover", 0.0) for s in sleeves.values()),
        })
    _write_csv(path, rows, PAPER_SLEEVE_COLUMNS)


def _write_paper_equity_csv(path: Path, state: dict) -> None:
    port_hist = state.get("portfolio_equity_history", [])
    sleeve_hist = state.get("sleeve_equity_history", {})
    if not port_hist:
        pd.DataFrame(columns=["date", "portfolio"]).to_csv(path, index=False)
        return
    df = pd.DataFrame(port_hist).set_index("date").rename(columns={"equity": "portfolio"})
    for name, hist in sleeve_hist.items():
        if hist:
            s = pd.DataFrame(hist).set_index("date")["equity"].rename(name)
            df = df.join(s, how="left")
    df.to_csv(path, index_label="date")


def _write_paper_run_log_csv(path: Path, state: dict) -> None:
    _write_csv(path, list(state.get("run_log", [])), PAPER_RUN_LOG_COLUMNS)


def _write_paper_status_txt(path: Path, config_data: dict, state: dict) -> None:
    lines: list[str] = []
    lines.extend(_paper_banner())
    lines.append("")
    lines.append("=== Paper Trading Status ===")
    lines.append("Research/educational simulation only. No brokerage connection exists.")
    lines.append("")
    lines.append(f"Account created: {config_data.get('created_at')}")
    lines.append(f"Inception date: {config_data.get('inception_date')}")
    lines.append(f"Starting capital: ${_fmt_num(state.get('starting_capital'))}")
    lines.append(f"Portfolio weights: " + ", ".join(
        f"{n}={w:.0%}" for n, w in config_data.get("weights", [])
    ))
    lines.append(f"Universe mode: {config_data.get('universe_mode')} "
                 f"({len(config_data.get('universe_tickers', []))} tickers, frozen at init)")
    lines.append(f"Cost assumption: {config_data.get('cost_bps')} bps/trade   "
                 f"Fractional shares: {'on' if config_data.get('fractional_shares') else 'off'}")
    lines.append(f"State schema version: {state.get('schema_version')}")
    lines.append(f"Git commit at init: {config_data.get('git_commit_sha')}")
    lines.append("")
    lines.append("--- Account snapshot ---")
    lines.append(f"Last processed market date: {state.get('last_processed_date') or '(none yet)'}")
    lines.append(f"Data cutoff date: {state.get('data_cutoff_date') or '(none yet)'}")
    lines.append(f"Sessions processed (runs): {state.get('num_runs', 0)}")
    lines.append(f"Total portfolio equity: ${_fmt_num(state.get('total_equity'))}")
    lines.append(f"Cumulative return: {_fmt_pct(state.get('cumulative_return'))}")
    lines.append(f"Cumulative transaction costs: ${_fmt_num(state.get('transaction_costs_total'))}")
    lines.append(f"Cumulative turnover (traded notional): ${_fmt_num(state.get('turnover_total'))}")

    recon = state.get("reconciliation", {})
    lines.append("")
    lines.append(f"Reconciliation: {'OK' if recon.get('ok') else 'FAILED'}")
    for c in recon.get("checks", []):
        flag = "ok " if c.get("ok") else "XX "
        lines.append(f"  [{flag}] {c.get('check')}: {c.get('detail')}")

    lines.append("")
    lines.append("--- Sleeves (independent; no shared cash, no rebalancing) ---")
    for name, s in state.get("sleeves", {}).items():
        lines.append(
            f"  {name}: start ${_fmt_num(s.get('allocated_capital'))} ({s.get('weight', 0):.0%}) "
            f"-> equity ${_fmt_num(s.get('equity'))} (cash ${_fmt_num(s.get('cash'))}, "
            f"end wt {s.get('ending_weight', float('nan')):.0%}, {len(s.get('positions', []))} positions)"
        )

    positions = [(name, p) for name, s in state.get("sleeves", {}).items() for p in s.get("positions", [])]
    lines.append("")
    lines.append(f"--- Current positions ({len(positions)}) ---")
    for name, p in positions[:40]:
        lines.append(
            f"  {name} {p['ticker']}: {p['shares']:.4f} sh @ avg ${_fmt_num(p['avg_cost'])} "
            f"(last ${_fmt_num(p['last_price'])}, mkt ${_fmt_num(p['market_value'])})"
        )
    if len(positions) > 40:
        lines.append(f"  ... and {len(positions) - 40} more (see paper_positions.csv)")

    pending = state.get("pending_orders", [])
    lines.append("")
    lines.append(f"--- Pending orders ({len(pending)}) -- fill on next processed session ---")
    for o in pending[:40]:
        lines.append(
            f"  {o['sleeve']} {o['ticker']}: {o['intended_side']} target wt {o['target_weight']:.2%} "
            f"(~${_fmt_num(o['requested_notional'])}), signal {o['signal_date']} -> fill {o['scheduled_fill_date']}"
        )
    if len(pending) > 40:
        lines.append(f"  ... and {len(pending) - 40} more (see paper_orders.csv)")

    stale = state.get("rejected_stale_orders", [])
    if stale:
        lines.append("")
        lines.append(f"--- Rejected / stale orders ({len(stale)}) -- reported, NOT filled ---")
        for o in stale[:20]:
            lines.append(f"  {o['sleeve']} {o['ticker']}: {o['reason']} (scheduled {o['scheduled_fill_date']})")

    log = state.get("run_log", [])
    if log:
        last = log[-1]
        lines.append("")
        lines.append("--- Most recent run ---")
        lines.append(f"  Paper date processed: {last.get('paper_date')}  (data cutoff {last.get('data_cutoff_date')})")
        lines.append(f"  New signals: {last.get('num_new_signals')}   Pending created: {last.get('num_pending_created')}"
                     f"   Fills: {last.get('num_fills')}   Stale: {last.get('num_stale')}")
        lines.append(f"  Daily return: {_fmt_pct(last.get('daily_return'))}   "
                     f"Cumulative: {_fmt_pct(last.get('cumulative_return'))}")
        lines.append(f"  Run transaction costs: ${_fmt_num(last.get('transaction_costs_run'))}")
        lines.append(f"  Reconciliation: {'OK' if last.get('reconciliation_ok') else 'FAILED'}")

    lines.append("")
    lines.append("Execution model: close-to-close fills with a one-trading-day signal-to-fill "
                 "lag (a signal on date T fills at the next session's close). No intraday "
                 "assumptions, no same-close execution, no bid/ask spread, no market impact.")
    lines.append("")
    lines.extend(_paper_banner())

    path.write_text("\n".join(lines) + "\n")


def write_paper_artifacts(config_data: dict, state: dict, dest_dir) -> Path:
    """Write the derived paper artifacts (status txt + CSVs) into ``dest_dir``.
    The authoritative paper_state.json / paper_config.json are written by
    src.paper on each advance; this renders the human/CSV VIEWS of that state and
    is safe to regenerate any time (it is a pure function of the passed dicts)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    _write_paper_status_txt(dest_dir / "paper_status.txt", config_data, state)
    _write_paper_orders_csv(dest_dir / "paper_orders.csv", state)
    _write_paper_trades_csv(dest_dir / "paper_trades.csv", state)
    _write_paper_positions_csv(dest_dir / "paper_positions.csv", state)
    _write_paper_equity_csv(dest_dir / "paper_equity.csv", state)
    _write_paper_sleeves_csv(dest_dir / "paper_sleeves.csv", state)
    _write_paper_run_log_csv(dest_dir / "paper_run_log.csv", state)
    return dest_dir
