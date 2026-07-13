"""Level-3 decision-audit queries over a written lab run directory.

Reads the deterministic trace artifacts (decision_trace.csv, order_trace.csv,
portfolio_trace.csv) and answers the questions the mission asks:

  --explain-date YYYY-MM-DD   every decision that date, grouped by sleeve
  --explain-ticker TICKER     every decision involving one ticker over time
  --show-reason REASON_CODE   every decision with a given reason code

All output is rendered from the traced numeric fields -- nothing is
recomputed or invented. Queries are read-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .reasons import REASON_CODES


class ExplainError(Exception):
    pass


def _find_latest_lab_run(output_dir: str) -> Path | None:
    root = Path(output_dir)
    if not root.exists():
        return None
    candidates = [
        p for p in root.glob("*_portfolio_*")
        if (p / "decision_trace.csv").exists() and (p / "run_manifest.json").exists()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def _load_trace(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "decision_trace.csv"
    if not path.exists():
        raise ExplainError(
            f"No decision_trace.csv in {run_dir}. Run a traced lab experiment first "
            f"(e.g. --strategy strategy_lab --experiment baseline)."
        )
    df = pd.read_csv(path, parse_dates=["decision_date", "signal_date", "fill_date"])
    return df


def _fmt_row(row: pd.Series) -> str:
    parts = [f"  {row['sleeve']}/{row['ticker']}: {row['reason_code']}"]
    if pd.notna(row.get("human_reason")) and row.get("human_reason"):
        parts.append(f"    {row['human_reason']}")
    return "\n".join(parts)


def explain_date(df: pd.DataFrame, date: str) -> str:
    target = pd.Timestamp(date)
    day = df[df["decision_date"] == target]
    if day.empty:
        return (
            f"No decisions recorded on {target.date()} (not a rebalance/signal date for any "
            f"sleeve, or outside the run window)."
        )
    lines = [f"=== Decisions on {target.date()} ==="]
    for sleeve, g in day.groupby("sleeve"):
        selected = g[g["selected"] == True]  # noqa: E712
        lines.append(f"\n[{sleeve}] {len(g)} decisions, {len(selected)} selected")
        for _i, row in g.sort_values(["selected", "rank"], ascending=[False, True]).iterrows():
            lines.append(_fmt_row(row))
    return "\n".join(lines)


def explain_ticker(df: pd.DataFrame, ticker: str) -> str:
    tk = df[df["ticker"].astype(str).str.upper() == ticker.upper()]
    if tk.empty:
        return f"No decisions recorded for ticker {ticker!r} in this run."
    lines = [f"=== Decision history for {ticker.upper()} ==="]
    counts = tk["reason_code"].value_counts()
    lines.append("Reason-code counts: " + ", ".join(f"{c}={n}" for c, n in counts.items()))
    lines.append("")
    for _i, row in tk.sort_values("decision_date").iterrows():
        lines.append(f"{row['decision_date'].date()} [{row['sleeve']}] {row['reason_code']}")
        if pd.notna(row.get("human_reason")) and row.get("human_reason"):
            lines.append(f"    {row['human_reason']}")
    return "\n".join(lines)


def show_reason(df: pd.DataFrame, code: str) -> str:
    code = code.upper()
    if code not in REASON_CODES:
        raise ExplainError(
            f"Unknown reason code {code!r}. Known codes: {', '.join(sorted(REASON_CODES))}"
        )
    rows = df[df["reason_code"] == code]
    if rows.empty:
        return f"Reason code {code} did not fire in this run."
    lines = [
        f"=== {code}: {REASON_CODES[code].template} ===",
        f"{len(rows)} occurrences across {rows['ticker'].nunique()} tickers, "
        f"{rows['decision_date'].nunique()} dates.",
        "",
    ]
    by_ticker = rows.groupby("ticker").size().sort_values(ascending=False)
    lines.append("By ticker: " + ", ".join(f"{t}={n}" for t, n in by_ticker.head(20).items()))
    lines.append("")
    lines.append("Most recent occurrences:")
    for _i, row in rows.sort_values("decision_date", ascending=False).head(15).iterrows():
        lines.append(f"  {row['decision_date'].date()} [{row['sleeve']}] {row['ticker']}: "
                     f"{row.get('human_reason', '')}")
    return "\n".join(lines)


def run_explain_query(args) -> int:
    run_dir = Path(args.lab_run_dir) if args.lab_run_dir else _find_latest_lab_run(args.output_dir)
    if run_dir is None or not run_dir.exists():
        print(
            "Error: no lab run directory found. Pass --lab-run-dir, or run "
            "--strategy strategy_lab --experiment baseline first.",
            file=sys.stderr,
        )
        return 1
    try:
        df = _load_trace(run_dir)
        if args.explain_date:
            print(explain_date(df, args.explain_date))
        elif args.explain_ticker:
            print(explain_ticker(df, args.explain_ticker))
        elif args.show_reason:
            print(show_reason(df, args.show_reason))
        print(f"\n[strategy_lab] queried {run_dir}")
        return 0
    except ExplainError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
