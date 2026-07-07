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

EXECUTION_MODEL_LINE = (
    "Execution model: close-to-close, one-day signal-to-fill lag, "
    "no intraday stops, no bid/ask spread, no market impact."
)
ADJUSTED_PRICE_LINE = (
    "All prices/trades are auto_adjust=True adjusted-price units, not literal "
    "historical execution prices."
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
    "realized_pnl", "realized_return_pct", "reason", "holding_days",
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

    lines.append("")
    lines.append("--- Performance ---")
    lines.append(f"Final equity: ${_fmt_num(metrics.get('final_equity'))}")
    lines.append(f"Total return: {_fmt_pct(metrics.get('total_return'))}")
    lines.append(f"CAGR: {_fmt_pct(metrics.get('cagr'))}")
    if metrics.get("short_period_warning"):
        lines.append("  (WARNING: effective period < 90 days -- CAGR/Sharpe are unstable over short windows)")
    lines.append(f"Max drawdown: {_fmt_pct(metrics.get('max_drawdown'))}")
    sharpe = metrics.get("sharpe_ratio")
    lines.append(f"Sharpe ratio (rf=0): {sharpe:.2f}" if pd.notna(sharpe) else "Sharpe ratio (rf=0): n/a")

    if result.strategy_name == "sector_rotation":
        lines.append(
            "(Win rate de-emphasized for sector rotation -- long holds + partial "
            "rebalances make it a weak signal here.)"
        )
    lines.append(f"Win rate (full-exit trades): {_fmt_pct(metrics.get('win_rate'))}")
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
