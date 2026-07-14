"""Performance and risk attribution for a lab portfolio run.

Everything here is computed from an already-produced PortfolioResult (and its
per-sleeve BacktestResults) plus the SPY benchmark -- no strategy logic and no
re-running. Two reconciliation identities are enforced by tests:

  1. Per-ticker total P&L sums EXACTLY to each sleeve's total P&L. A ticker's
     P&L = its net trading cash flow (sells - buys, each net of costs) + the
     mark-to-market value of any position still open at the sleeve's last day.
     Cash held idle contributes 0, which is correct.
  2. Per-year dollar P&L (from the equity curve) sums EXACTLY to total P&L.

Some measures are DISCLOSED APPROXIMATIONS (risk-contribution weights, and the
"exclude best/worst trades/days" recomputations, which subtract dollar P&L
without re-simulating compounding). Each such function documents the caveat.
Megacap-tech classification is a CURRENT static list -- it introduces no future
information (sector membership is effectively static and known in advance) but
is survivorship-flavored like the universe; it is disclosed, never ranked
against unbiased runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine import BacktestResult
from ..portfolio import PortfolioResult

# Current static megacap-tech classification of the default research universe.
# Used only for the "P&L excluding megacap tech" diagnostic; disclosed as a
# current (not point-in-time) label that adds no future information.
MEGACAP_TECH = {
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "NVDA", "NFLX",
    "TSLA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "CSCO",
}

TRADING_DAYS = 252


# --- Sleeve-level attribution --------------------------------------------------------

def _aligned_returns(pf: PortfolioResult) -> tuple[pd.DataFrame, pd.Series]:
    """Daily returns of each sleeve and the portfolio over the common window."""
    common = pf.combined_result.equity_curve.index
    cols = {}
    for s in pf.sleeves:
        cols[s.strategy] = s.result.equity_curve.reindex(common)
    sleeve_equity = pd.DataFrame(cols)
    sleeve_ret = sleeve_equity.pct_change()
    port_ret = pf.combined_result.equity_curve.pct_change()
    return sleeve_ret, port_ret


def _capture_ratios(sleeve_ret: pd.Series, bench_ret: pd.Series) -> tuple[float, float]:
    common = sleeve_ret.index.intersection(bench_ret.index)
    s = sleeve_ret.reindex(common).dropna()
    b = bench_ret.reindex(common).dropna()
    common = s.index.intersection(b.index)
    s, b = s.reindex(common), b.reindex(common)
    up = b > 0
    down = b < 0
    upside = (s[up].mean() / b[up].mean()) if up.any() and b[up].mean() != 0 else float("nan")
    downside = (s[down].mean() / b[down].mean()) if down.any() and b[down].mean() != 0 else float("nan")
    return float(upside), float(downside)


def _trade_stats(result: BacktestResult) -> dict:
    exits = [t for t in result.trades if t.event_type == "full_exit"]
    if not exits:
        return {
            "hit_rate": float("nan"), "avg_gain": float("nan"), "avg_loss": float("nan"),
            "profit_factor": float("nan"), "payoff_ratio": float("nan"),
            "avg_holding_days": float("nan"), "num_round_trips": 0,
        }
    pnls = [t.realized_pnl_net if t.realized_pnl_net is not None else t.realized_pnl for t in exits]
    gains = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win = sum(gains)
    gross_loss = -sum(losses)
    return {
        "hit_rate": len(gains) / len(exits),
        "avg_gain": float(np.mean(gains)) if gains else float("nan"),
        "avg_loss": float(np.mean(losses)) if losses else float("nan"),
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else float("nan"),
        "payoff_ratio": (np.mean(gains) / -np.mean(losses)) if gains and losses else float("nan"),
        "avg_holding_days": float(np.mean([t.holding_days for t in exits])),
        "num_round_trips": len(exits),
    }


def sleeve_attribution(pf: PortfolioResult, benchmark_close: pd.Series) -> list[dict]:
    """Per-sleeve capital, P&L, risk contribution, capture, correlations, and
    trade statistics. Risk contribution uses the standard time-average-weight
    approximation r_p ~= sum_i w_i r_i (drift ignored); the values sum to ~1.0
    and are labeled as an approximation."""
    sleeve_ret, port_ret = _aligned_returns(pf)
    common = pf.combined_result.equity_curve.index
    bench_ret = benchmark_close.reindex(common).pct_change()
    port_var = port_ret.var(ddof=0)

    # Portfolio peak->trough window for drawdown contribution (exact in dollars).
    port_equity = pf.combined_result.equity_curve
    running_max = port_equity.cummax()
    dd = port_equity / running_max - 1.0
    trough_date = dd.idxmin()
    peak_date = port_equity.loc[:trough_date].idxmax()
    port_dd_dollars = float(port_equity.loc[trough_date] - port_equity.loc[peak_date])

    total_pnl = sum(s.pnl_contribution for s in pf.sleeves)
    rows = []
    for s in pf.sleeves:
        eq = s.result.equity_curve.reindex(common)
        w_avg = float((eq / port_equity).mean())
        r_i = sleeve_ret[s.strategy]
        cov_ip = float(np.cov(r_i.dropna().reindex(port_ret.dropna().index).dropna(),
                              port_ret.reindex(r_i.dropna().index).dropna(), ddof=0)[0, 1]) \
            if r_i.dropna().size > 2 else float("nan")
        risk_contrib = (w_avg * cov_ip / port_var) if port_var and not np.isnan(cov_ip) else float("nan")

        dd_contrib = (
            float(eq.loc[trough_date] - eq.loc[peak_date]) / port_dd_dollars
            if port_dd_dollars != 0 else float("nan")
        )
        upside, downside = _capture_ratios(r_i, bench_ret)
        corr_spy = float(r_i.corr(bench_ret))
        other_corrs = {
            o.strategy: float(r_i.corr(sleeve_ret[o.strategy]))
            for o in pf.sleeves if o.strategy != s.strategy
        }
        invested = _sleeve_invested_pct(s.result)
        row = {
            "sleeve": s.strategy,
            "weight": s.weight,
            "allocated_capital": s.allocated_capital,
            "beginning_value": s.allocated_capital,
            "ending_value": s.final_value,
            "dollar_pnl": s.pnl_contribution,
            "pct_of_total_pnl": (s.pnl_contribution / total_pnl) if total_pnl else float("nan"),
            "risk_contribution_approx": risk_contrib,
            "drawdown_contribution": dd_contrib,
            "turnover": s.metrics.get("total_turnover"),
            "costs": s.cost_contribution,
            "avg_invested_pct": invested["avg_invested_pct"],
            "avg_cash_pct": 1.0 - invested["avg_invested_pct"],
            "correlation_to_spy": corr_spy,
            "upside_capture": upside,
            "downside_capture": downside,
            "sharpe": s.metrics.get("sharpe_ratio"),
            "max_drawdown": s.metrics.get("max_drawdown"),
        }
        for o_name, c in other_corrs.items():
            row[f"corr_to_{o_name}"] = c
        row.update(_trade_stats(s.result))
        rows.append(row)
    return rows


def _sleeve_invested_pct(result: BacktestResult) -> dict:
    if not result.positions:
        return {"avg_invested_pct": 0.0}
    df = pd.DataFrame(result.positions)
    per_day = df.groupby("date").agg(mv=("market_value", "sum"), eq=("sleeve_equity", "first"))
    per_day["inv"] = per_day.apply(lambda r: (r["mv"] / r["eq"]) if r["eq"] else 0.0, axis=1)
    return {"avg_invested_pct": float(per_day["inv"].mean())}


# --- Ticker-level attribution --------------------------------------------------------

def _ticker_pnl_for_sleeve(result: BacktestResult) -> pd.DataFrame:
    """Exact per-ticker P&L decomposition that sums to the sleeve's total P&L.
    ticker P&L = net trading cash flow + mark-to-market of any still-open
    position at the sleeve's last day."""
    cashflow: dict[str, float] = {}
    costs: dict[str, float] = {}
    turnover: dict[str, float] = {}
    for tx in result.transactions:
        signed = (
            (tx.executed_notional - tx.transaction_cost) if tx.action == "sell"
            else -(tx.executed_notional + tx.transaction_cost)
        )
        cashflow[tx.ticker] = cashflow.get(tx.ticker, 0.0) + signed
        costs[tx.ticker] = costs.get(tx.ticker, 0.0) + tx.transaction_cost
        turnover[tx.ticker] = turnover.get(tx.ticker, 0.0) + tx.executed_notional

    # Ending market value per ticker at the sleeve's last recorded day.
    ending_value: dict[str, float] = {}
    if result.positions:
        pos = pd.DataFrame(result.positions)
        last_day = pos["date"].max()
        last = pos[pos["date"] == last_day]
        for _i, r in last.iterrows():
            if abs(r["shares"]) > 1e-9:
                ending_value[r["ticker"]] = ending_value.get(r["ticker"], 0.0) + r["market_value"]

    # Realized/round-trip stats per ticker.
    realized: dict[str, float] = {}
    round_trips: dict[str, int] = {}
    wins: dict[str, int] = {}
    hold_days: dict[str, list] = {}
    for t in result.trades:
        pnl = t.realized_pnl_net if t.realized_pnl_net is not None else t.realized_pnl
        realized[t.ticker] = realized.get(t.ticker, 0.0) + pnl
        if t.event_type == "full_exit":
            round_trips[t.ticker] = round_trips.get(t.ticker, 0) + 1
            if pnl > 0:
                wins[t.ticker] = wins.get(t.ticker, 0) + 1
            hold_days.setdefault(t.ticker, []).append(t.holding_days)

    tickers = set(cashflow) | set(ending_value) | set(realized)
    rows = []
    for tk in sorted(tickers):
        total = cashflow.get(tk, 0.0) + ending_value.get(tk, 0.0)
        rt = round_trips.get(tk, 0)
        rows.append({
            "sleeve": result.strategy_name,
            "ticker": tk,
            "total_pnl": total,
            "realized_pnl": realized.get(tk, 0.0),
            "ending_value": ending_value.get(tk, 0.0),
            "costs": costs.get(tk, 0.0),
            "turnover": turnover.get(tk, 0.0),
            "num_round_trips": rt,
            "win_rate": (wins.get(tk, 0) / rt) if rt else float("nan"),
            "avg_holding_days": float(np.mean(hold_days[tk])) if hold_days.get(tk) else float("nan"),
            "is_megacap_tech": tk in MEGACAP_TECH,
        })
    return pd.DataFrame(rows)


def ticker_attribution(pf: PortfolioResult) -> pd.DataFrame:
    frames = [_ticker_pnl_for_sleeve(s.result) for s in pf.sleeves]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return df
    return df.sort_values("total_pnl", ascending=False).reset_index(drop=True)


# --- Year-level attribution ----------------------------------------------------------

def pnl_by_year(pf: PortfolioResult) -> pd.DataFrame:
    """Dollar P&L per calendar year from the combined equity curve. Sums
    exactly to total P&L: each year's P&L is (year-end equity - prior
    year-end equity), with the first (partial) year anchored at starting
    capital."""
    equity = pf.combined_result.equity_curve.sort_index()
    capital = pf.total_capital
    rows = []
    prior = capital
    for year in sorted(set(equity.index.year)):
        year_equity = equity[equity.index.year == year]
        end_val = float(year_equity.iloc[-1])
        rows.append({"year": year, "start_value": prior, "end_value": end_val,
                     "dollar_pnl": end_val - prior, "return_pct": (end_val / prior - 1.0) if prior else float("nan")})
        prior = end_val
    return pd.DataFrame(rows)


# --- Concentration & robustness-to-outliers -----------------------------------------

def concentration_analysis(pf: PortfolioResult) -> dict:
    """How much of total P&L comes from the top contributors, and how the
    result changes when the biggest contributors / best-and-worst
    trades and days are removed.

    The leave-out figures are DISCLOSED APPROXIMATIONS: they subtract dollar
    P&L from the total rather than re-simulating the strategy without those
    events (which would change every later position). They answer 'how
    concentrated is the realized P&L', not 'what would have happened'."""
    ticker_df = ticker_attribution(pf)
    total_pnl = float(sum(s.pnl_contribution for s in pf.sleeves))

    out: dict = {"total_pnl": total_pnl}
    ranked = ticker_df.sort_values("total_pnl", ascending=False)
    for n in (1, 3, 5, 10):
        top_n = ranked.head(n)["total_pnl"].sum()
        out[f"top_{n}_pnl"] = float(top_n)
        out[f"top_{n}_pct_of_pnl"] = float(top_n / total_pnl) if total_pnl else float("nan")

    # Leave-one-out: total P&L with each single top contributor removed.
    loo = []
    for _i, r in ranked.head(10).iterrows():
        loo.append({
            "excluded_ticker": r["ticker"],
            "excluded_pnl": float(r["total_pnl"]),
            "pnl_without": float(total_pnl - r["total_pnl"]),
            "pct_of_total": float(r["total_pnl"] / total_pnl) if total_pnl else float("nan"),
        })
    out["leave_one_out_top10"] = loo

    # Best/worst round-trip trades (net) across sleeves.
    all_exits = []
    for s in pf.sleeves:
        for t in s.result.trades:
            if t.event_type == "full_exit":
                pnl = t.realized_pnl_net if t.realized_pnl_net is not None else t.realized_pnl
                all_exits.append(pnl)
    all_exits.sort()
    realized_total = float(sum(all_exits))
    out["realized_pnl_total"] = realized_total
    if len(all_exits) >= 5:
        out["realized_without_5_best_trades"] = float(realized_total - sum(all_exits[-5:]))
        out["realized_without_5_worst_trades"] = float(realized_total - sum(all_exits[:5]))

    # Best/worst days: recompute compounded return dropping those daily returns.
    port_ret = pf.combined_result.equity_curve.pct_change().dropna()
    full_growth = float((1.0 + port_ret).prod())
    ranked_days = port_ret.sort_values()
    if len(port_ret) >= 5:
        without_best = port_ret.drop(ranked_days.index[-5:])
        without_worst = port_ret.drop(ranked_days.index[:5])
        out["total_growth_factor"] = full_growth
        out["growth_without_5_best_days"] = float((1.0 + without_best).prod())
        out["growth_without_5_worst_days"] = float((1.0 + without_worst).prod())

    # P&L excluding megacap tech (current static classification, disclosed).
    tech_pnl = float(ticker_df[ticker_df["is_megacap_tech"]]["total_pnl"].sum())
    out["megacap_tech_pnl"] = tech_pnl
    out["pnl_excluding_megacap_tech"] = float(total_pnl - tech_pnl)
    out["megacap_tech_pct_of_pnl"] = float(tech_pnl / total_pnl) if total_pnl else float("nan")
    return out


# --- Exposure / sleeve overlap -------------------------------------------------------

def exposure_overlap(pf: PortfolioResult) -> pd.DataFrame:
    """Per-day count of tickers held by more than one sleeve simultaneously,
    and the aggregate weight in those overlapping names. Answers 'how often do
    independent sleeves double up on the same name'."""
    common = pf.combined_result.equity_curve.index
    port_equity = pf.combined_result.equity_curve
    # Build per (date, ticker) market value summed across sleeves, plus how
    # many sleeves held it.
    frames = []
    for s in pf.sleeves:
        pos = pd.DataFrame(s.result.positions)
        if pos.empty:
            continue
        pos = pos[pos["shares"].abs() > 1e-9]
        pos = pos[pos["date"].isin(common)]
        frames.append(pos[["date", "ticker", "market_value"]])
    if not frames:
        return pd.DataFrame(columns=["date", "num_overlap_tickers", "overlap_weight", "max_single_name_weight"])
    allpos = pd.concat(frames, ignore_index=True)
    grouped = allpos.groupby(["date", "ticker"]).agg(mv=("market_value", "sum"), n=("market_value", "size"))
    rows = []
    for date, g in grouped.groupby(level=0):
        eq = float(port_equity.loc[date]) if date in port_equity.index else float("nan")
        overlaps = g[g["n"] > 1]
        overlap_weight = float(overlaps["mv"].sum() / eq) if eq else float("nan")
        max_name = float(g["mv"].max() / eq) if eq and len(g) else 0.0
        rows.append({
            "date": date,
            "num_overlap_tickers": int(len(overlaps)),
            "overlap_weight": overlap_weight,
            "max_single_name_weight": max_name,
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
