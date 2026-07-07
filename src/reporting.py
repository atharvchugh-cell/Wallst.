"""report.txt + CSV/JSON audit artifacts + equity-curve PNG, written to a
timestamped, parameterized run directory so repeated runs don't clobber
prior audit trails. CSVs are the primary audit artifact; the PNG is
secondary.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402
from .engine import BacktestResult  # noqa: E402
from .metrics import annual_returns, monthly_returns  # noqa: E402

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
