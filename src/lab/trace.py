"""Decision trace: observational records of every decision, order, and
portfolio day in a lab run.

Behavior-neutrality contract (tested): the recorder only ever READS values
the strategies and engine already computed. It never mutates strategy or
engine state, never calls MarketDataView with a different date, and a run
with the recorder attached is byte-identical to one without.

Grounding contract (tested): every human-readable explanation is rendered
from the traced numeric fields via the central reason-code templates
(src/lab/reasons.py) -- explanations restate what the code actually used,
they are never generated after the fact from anything else. Unavailable
values stay None/null; nothing is imputed.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .reasons import human_reason, validate_code

TRACE_SCHEMA_VERSION = 1

# Column order for decision_trace.csv -- stable, documented in
# docs/STRATEGY_INTELLIGENCE.md. Fields that don't apply to a strategy stay
# null rather than being invented.
DECISION_COLUMNS = [
    "decision_id", "decision_date", "sleeve", "ticker", "tradable",
    "selected", "eligible", "reason_code", "reason_detail",
    "close", "lookback_return", "lookback_days", "trend_sma", "trend_sma_period",
    "rank_score", "rank", "num_ranked", "top_k", "rsi", "regime_state",
    "prior_weight", "target_weight", "weight_change",
    "signal_date", "fill_date", "sleeve_equity", "portfolio_equity",
    "human_reason", "lab_config_hash",
]

ORDER_COLUMNS = [
    "sleeve", "ticker", "signal_date", "fill_date", "action", "reason_code",
    "requested_target_weight", "requested_notional", "shares_traded", "fill_price",
    "executed_notional", "transaction_cost", "cash_after", "position_shares_after",
    "engine_reason",
]

PORTFOLIO_DAY_BASE_COLUMNS = [
    "date", "portfolio_equity", "daily_return", "benchmark_daily_return",
    "excess_daily_return", "cash_total", "cash_pct", "gross_exposure_pct",
    "num_positions", "max_single_name_weight", "max_single_name_ticker",
    "effective_num_positions", "overlap_num_tickers", "overlap_tickers",
    "turnover_today", "costs_today", "rolling_vol_20d_annualized",
    "drawdown", "drawdown_days", "exposure_multiplier", "active_risk_controls",
]


@dataclass
class DecisionRecord:
    decision_id: str
    decision_date: pd.Timestamp
    sleeve: str
    ticker: str
    tradable: bool
    selected: bool | None
    eligible: bool | None
    reason_code: str
    reason_detail: str | None = None
    close: float | None = None
    lookback_return: float | None = None
    lookback_days: int | None = None
    trend_sma: float | None = None
    trend_sma_period: int | None = None
    rank_score: float | None = None
    rank: int | None = None
    num_ranked: int | None = None
    top_k: int | None = None
    rsi: float | None = None
    regime_state: str | None = None
    prior_weight: float | None = None
    target_weight: float | None = None
    weight_change: float | None = None
    signal_date: pd.Timestamp | None = None
    fill_date: pd.Timestamp | None = None
    sleeve_equity: float | None = None
    portfolio_equity: float | None = None
    human_reason: str = ""
    lab_config_hash: str = ""
    # Free-form extras for reason-template rendering (kept out of the CSV
    # columns; serialized only into the JSONL rows).
    extra: dict = field(default_factory=dict)


@dataclass
class OrderTraceRecord:
    sleeve: str
    ticker: str
    signal_date: pd.Timestamp
    fill_date: pd.Timestamp
    action: str
    reason_code: str
    requested_target_weight: float
    requested_notional: float | None
    shares_traded: float
    fill_price: float
    executed_notional: float
    transaction_cost: float
    cash_after: float
    position_shares_after: float
    engine_reason: str | None


@dataclass
class PortfolioDayRecord:
    date: pd.Timestamp
    portfolio_equity: float
    cash_total: float
    cash_pct: float
    gross_exposure_pct: float
    num_positions: int
    max_single_name_weight: float
    max_single_name_ticker: str | None
    effective_num_positions: float
    overlap_num_tickers: int
    overlap_tickers: str
    turnover_today: float
    costs_today: float
    exposure_multiplier: float | None = None
    active_risk_controls: str = ""
    sleeve_values: dict = field(default_factory=dict)  # sleeve -> equity
    sleeve_cash: dict = field(default_factory=dict)    # sleeve -> cash


@dataclass
class DecisionTrace:
    lab_config_hash: str
    decisions: list = field(default_factory=list)
    orders: list = field(default_factory=list)
    portfolio_days: list = field(default_factory=list)
    common_start: pd.Timestamp | None = None
    common_end: pd.Timestamp | None = None
    sleeve_names: list = field(default_factory=list)


class DecisionRecorder:
    """Collects trace records during a lab run. Purely observational."""

    def __init__(self, lab_config_hash: str = ""):
        self.lab_config_hash = lab_config_hash
        self.decisions: list[DecisionRecord] = []
        self.orders: list[OrderTraceRecord] = []
        self.portfolio_days: list[PortfolioDayRecord] = []
        self._sleeve: str | None = None
        self._portfolio_equity: float | None = None
        self._seq: dict = {}
        self._pending_turnover: float = 0.0
        self._pending_costs: float = 0.0

    # --- Context (set by the engine around strategy calls) -------------------

    def set_context(self, sleeve: str | None, portfolio_equity: float | None) -> None:
        self._sleeve = sleeve
        self._portfolio_equity = portfolio_equity

    # --- Strategy-side hook ---------------------------------------------------

    def record_decision(
        self,
        *,
        decision_date,
        ticker: str,
        reason_code: str,
        tradable: bool = True,
        selected: bool | None = None,
        eligible: bool | None = None,
        sleeve: str | None = None,
        **fields,
    ) -> None:
        """Called by strategies at each decision branch. `fields` may contain
        any DECISION_COLUMNS field plus extra template values; unknown extras
        land in `extra` (JSONL only) so templates can render them."""
        validate_code(reason_code)
        sleeve = sleeve or self._sleeve or "unknown"
        decision_date = pd.Timestamp(decision_date)
        key = (sleeve, decision_date, ticker)
        seq = self._seq.get(key, 0)
        self._seq[key] = seq + 1
        decision_id = f"{sleeve}:{decision_date.date()}:{ticker}:{seq}"

        known = {f for f in DecisionRecord.__dataclass_fields__ if f not in (
            "decision_id", "decision_date", "sleeve", "ticker", "tradable", "selected",
            "eligible", "reason_code", "human_reason", "lab_config_hash", "extra",
        )}
        record_kwargs = {k: v for k, v in fields.items() if k in known}
        extra = {k: v for k, v in fields.items() if k not in known}

        prior = record_kwargs.get("prior_weight")
        target = record_kwargs.get("target_weight")
        if prior is not None and target is not None and record_kwargs.get("weight_change") is None:
            record_kwargs["weight_change"] = target - prior

        rec = DecisionRecord(
            decision_id=decision_id,
            decision_date=decision_date,
            sleeve=sleeve,
            ticker=ticker,
            tradable=tradable,
            selected=selected,
            eligible=eligible,
            reason_code=reason_code,
            portfolio_equity=self._portfolio_equity,
            lab_config_hash=self.lab_config_hash,
            extra=extra,
            **record_kwargs,
        )
        template_fields = {**{k: v for k, v in asdict(rec).items() if k != "extra"}, **extra}
        rec.human_reason = human_reason(reason_code, template_fields)
        self.decisions.append(rec)

    # --- Engine-side hooks ------------------------------------------------------

    def record_fills(self, sleeve: str, transactions: list) -> None:
        for tx in transactions:
            self.orders.append(
                OrderTraceRecord(
                    sleeve=sleeve, ticker=tx.ticker, signal_date=tx.signal_date,
                    fill_date=tx.fill_date, action=tx.action, reason_code="ORDER_FILLED",
                    requested_target_weight=tx.requested_target_weight,
                    requested_notional=tx.requested_notional, shares_traded=tx.shares_traded,
                    fill_price=tx.fill_price, executed_notional=tx.executed_notional,
                    transaction_cost=tx.transaction_cost, cash_after=tx.cash_after,
                    position_shares_after=tx.position_shares_after, engine_reason=tx.reason,
                )
            )
            self._pending_turnover += tx.executed_notional
            self._pending_costs += tx.transaction_cost

    def record_overlay_order(self, sleeve: str, event, reason_code: str, detail: str | None = None) -> None:
        """Overlay-scheduled orders (exposure scaling, reallocation) at
        schedule time -- fills appear via record_fills like any other order."""
        validate_code(reason_code)
        self.record_decision(
            decision_date=event.signal_date, ticker=event.ticker, reason_code=reason_code,
            sleeve=sleeve, selected=None, eligible=None, reason_detail=detail,
            target_weight=event.target_weight, signal_date=event.signal_date,
            fill_date=event.fill_date,
        )

    def record_portfolio_day(self, d: pd.Timestamp, active_states: list, portfolio_equity: float) -> None:
        cash_total = sum(st.cash for st in active_states)
        sleeve_values = {st.prepared.name: st.equity_today for st in active_states}
        sleeve_cash = {st.prepared.name: st.cash for st in active_states}

        # Aggregate per-ticker market value across sleeves (a ticker can be
        # held by several sleeves at once -- that's exactly what the overlap
        # diagnostics measure).
        ticker_mv: dict[str, float] = {}
        ticker_sleeves: dict[str, int] = {}
        for st in active_states:
            for t, sh in st.shares.items():
                if abs(sh) <= 1e-9:
                    continue
                try:
                    px = st.close_lookup(t, d)
                except KeyError:
                    continue
                mv = sh * px
                ticker_mv[t] = ticker_mv.get(t, 0.0) + mv
                ticker_sleeves[t] = ticker_sleeves.get(t, 0) + 1

        gross = sum(abs(v) for v in ticker_mv.values())
        weights = {t: (v / portfolio_equity) for t, v in ticker_mv.items()} if portfolio_equity > 0 else {}
        max_ticker, max_w = None, 0.0
        for t, w in weights.items():
            if abs(w) > max_w:
                max_ticker, max_w = t, abs(w)
        sq = sum(w * w for w in weights.values())
        effective_n = (sum(abs(w) for w in weights.values()) ** 2 / sq) if sq > 0 else 0.0
        overlap = sorted(t for t, n in ticker_sleeves.items() if n > 1)

        self.portfolio_days.append(
            PortfolioDayRecord(
                date=pd.Timestamp(d),
                portfolio_equity=portfolio_equity,
                cash_total=cash_total,
                cash_pct=(cash_total / portfolio_equity) if portfolio_equity > 0 else 0.0,
                gross_exposure_pct=(gross / portfolio_equity) if portfolio_equity > 0 else 0.0,
                num_positions=len(ticker_mv),
                max_single_name_weight=max_w,
                max_single_name_ticker=max_ticker,
                effective_num_positions=effective_n,
                overlap_num_tickers=len(overlap),
                overlap_tickers=";".join(overlap),
                turnover_today=self._pending_turnover,
                costs_today=self._pending_costs,
                sleeve_values=sleeve_values,
                sleeve_cash=sleeve_cash,
            )
        )
        self._pending_turnover = 0.0
        self._pending_costs = 0.0

    def annotate_day(self, exposure_multiplier: float | None = None,
                     active_risk_controls: str | None = None) -> None:
        """Overlays call this after record_portfolio_day to stamp the day's
        overlay state onto the just-recorded row."""
        if not self.portfolio_days:
            return
        rec = self.portfolio_days[-1]
        if exposure_multiplier is not None:
            rec.exposure_multiplier = exposure_multiplier
        if active_risk_controls is not None:
            rec.active_risk_controls = active_risk_controls

    def finalize(self, pf) -> DecisionTrace:
        return DecisionTrace(
            lab_config_hash=self.lab_config_hash,
            decisions=self.decisions,
            orders=self.orders,
            portfolio_days=self.portfolio_days,
            common_start=pf.common_start,
            common_end=pf.common_end,
            sleeve_names=[s.strategy for s in pf.sleeves],
        )


# --- Artifact writing ---------------------------------------------------------------

def _atomic_to_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".csv.tmp")
    os.close(fd)
    try:
        df.to_csv(tmp, **kwargs)
        shutil.move(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def decisions_frame(trace: DecisionTrace) -> pd.DataFrame:
    rows = []
    for rec in trace.decisions:
        d = asdict(rec)
        d.pop("extra", None)
        rows.append(d)
    df = pd.DataFrame(rows)
    for c in DECISION_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[DECISION_COLUMNS] if len(df) else pd.DataFrame(columns=DECISION_COLUMNS)
    return df.sort_values(
        ["decision_date", "sleeve", "ticker", "decision_id"], kind="mergesort"
    ).reset_index(drop=True) if len(df) else df


def orders_frame(trace: DecisionTrace) -> pd.DataFrame:
    df = pd.DataFrame([asdict(o) for o in trace.orders])
    for c in ORDER_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[ORDER_COLUMNS] if len(df) else pd.DataFrame(columns=ORDER_COLUMNS)
    return df.sort_values(
        ["fill_date", "sleeve", "ticker"], kind="mergesort"
    ).reset_index(drop=True) if len(df) else df


def portfolio_frame(trace: DecisionTrace, benchmark_close: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    for rec in trace.portfolio_days:
        d = asdict(rec)
        sleeve_values = d.pop("sleeve_values")
        sleeve_cash = d.pop("sleeve_cash")
        for name in trace.sleeve_names:
            d[f"value_{name}"] = sleeve_values.get(name)
            d[f"cash_{name}"] = sleeve_cash.get(name)
            eq = d["portfolio_equity"]
            d[f"weight_{name}"] = (sleeve_values.get(name, 0.0) / eq) if eq else None
        rows.append(d)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=PORTFOLIO_DAY_BASE_COLUMNS)
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)

    equity = df.set_index("date")["portfolio_equity"]
    df["daily_return"] = equity.pct_change().values
    running_max = equity.cummax()
    df["drawdown"] = (equity / running_max - 1.0).values
    # Days spent below the running peak (0 at a fresh high) -- causal.
    dd_days, count = [], 0
    for under in (equity < running_max).values:
        count = count + 1 if under else 0
        dd_days.append(count)
    df["drawdown_days"] = dd_days
    df["rolling_vol_20d_annualized"] = (
        equity.pct_change().rolling(20, min_periods=20).std(ddof=0) * (252 ** 0.5)
    ).values

    if benchmark_close is not None:
        bench = benchmark_close.reindex(equity.index)
        df["benchmark_daily_return"] = bench.pct_change().values
        df["excess_daily_return"] = df["daily_return"] - df["benchmark_daily_return"]
    else:
        df["benchmark_daily_return"] = None
        df["excess_daily_return"] = None

    sleeve_cols = [c for c in df.columns if c.startswith(("value_", "cash_", "weight_"))]
    return df[PORTFOLIO_DAY_BASE_COLUMNS + sorted(sleeve_cols)]


def reason_code_summary(trace: DecisionTrace) -> pd.DataFrame:
    df = decisions_frame(trace)
    if df.empty:
        return pd.DataFrame(columns=["reason_code", "sleeve", "count", "first_date", "last_date"])
    grouped = (
        df.groupby(["reason_code", "sleeve"])
        .agg(count=("decision_id", "size"),
             first_date=("decision_date", "min"),
             last_date=("decision_date", "max"))
        .reset_index()
        .sort_values(["count", "reason_code", "sleeve"], ascending=[False, True, True],
                     kind="mergesort")
        .reset_index(drop=True)
    )
    return grouped


def selection_funnel(trace: DecisionTrace) -> pd.DataFrame:
    """Per (sleeve, decision date): universe considered -> had data -> passed
    filters (eligible) -> selected. Quantifies which stage rejects the most
    candidates."""
    df = decisions_frame(trace)
    if df.empty:
        return pd.DataFrame(columns=[
            "sleeve", "decision_date", "considered", "had_data", "eligible", "selected",
        ])
    data_missing_codes = {"MISSING_SIGNAL_DATA", "INSUFFICIENT_HISTORY", "NOT_IN_UNIVERSE"}
    rows = []
    for (sleeve, date), g in df.groupby(["sleeve", "decision_date"]):
        considered = len(g)
        had_data = int((~g["reason_code"].isin(data_missing_codes)).sum())
        eligible = int((g["eligible"] == True).sum())  # noqa: E712 -- nullable column
        selected = int((g["selected"] == True).sum())  # noqa: E712
        rows.append({
            "sleeve": sleeve, "decision_date": date, "considered": considered,
            "had_data": had_data, "eligible": eligible, "selected": selected,
        })
    return pd.DataFrame(rows).sort_values(["decision_date", "sleeve"], kind="mergesort").reset_index(drop=True)


def write_trace_artifacts(trace: DecisionTrace, run_dir, benchmark_close: pd.Series | None = None) -> None:
    """Write the deterministic trace artifact set into `run_dir`. If
    `benchmark_close` is not passed, SPY is loaded from the cache over the
    trace's common window for the benchmark columns."""
    run_dir = Path(run_dir)
    if benchmark_close is None and trace.common_start is not None:
        from .. import data as data_module

        spy_df = data_module.get_benchmark_data(trace.common_start, trace.common_end)
        benchmark_close = spy_df["Close"]

    dec = decisions_frame(trace)
    _atomic_to_csv(dec, run_dir / "decision_trace.csv", index=False)

    # JSONL variant carries the extras that don't fit fixed CSV columns.
    jsonl_path = run_dir / "decision_trace.jsonl"
    fd, tmp = tempfile.mkstemp(dir=run_dir, suffix=".jsonl.tmp")
    os.close(fd)
    try:
        with open(tmp, "w") as f:
            for rec in trace.decisions:
                f.write(json.dumps(asdict(rec), default=str, sort_keys=True) + "\n")
        shutil.move(tmp, jsonl_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    _atomic_to_csv(orders_frame(trace), run_dir / "order_trace.csv", index=False)
    _atomic_to_csv(portfolio_frame(trace, benchmark_close), run_dir / "portfolio_trace.csv", index=False)
    _atomic_to_csv(reason_code_summary(trace), run_dir / "reason_code_summary.csv", index=False)
    _atomic_to_csv(selection_funnel(trace), run_dir / "selection_funnel.csv", index=False)

    summary = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "lab_config_hash": trace.lab_config_hash,
        "num_decisions": len(trace.decisions),
        "num_orders": len(trace.orders),
        "num_portfolio_days": len(trace.portfolio_days),
        "sleeves": trace.sleeve_names,
        "window": {
            "start": str(trace.common_start.date()) if trace.common_start is not None else None,
            "end": str(trace.common_end.date()) if trace.common_end is not None else None,
        },
        "reason_code_counts": (
            reason_code_summary(trace).groupby("reason_code")["count"].sum().to_dict()
            if trace.decisions else {}
        ),
    }
    from .manifest import atomic_write_json

    atomic_write_json(run_dir / "decision_summary.json", summary)
