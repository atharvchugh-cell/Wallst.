"""Forward paper-trading ledger for the fixed 60/35/5 portfolio.

This is a *simulation* layer, not a broker. It NEVER connects to Robinhood or
any brokerage, never transmits an order, and holds no credentials. It processes
finalized daily market sessions one at a time, generates simulated orders using
only information available as of each processed date, and maintains a
persistent, reloadable ledger under ``paper_state/``.

Design -- deterministic replay, so paper can never diverge from backtest
=======================================================================
The authoritative content of the ledger at any processed date ``T`` (settled
fills, positions, cash, equity history) is DERIVED by replaying the ordinary
backtest engine from the account's inception up to ``T``, per sleeve, using the
SAME strategy/market-view/engine machinery the backtest and portfolio modes use
(via ``tournament.prepare_stock_plan`` / ``prepare_sector_plan`` +
``engine.run_backtest``). Because that replay only ever reads data on/before
``T`` (the ``MarketDataView`` refuses later reads, and the sleeve fetch is
sliced to ``<= T``), there is no lookahead, and re-processing a date is
idempotent by construction -- the same inputs always yield the same ledger.

Two engine features (added backward-compatibly) make single-run paper semantics
exact:
  - ``signal_calendar``: the strategy sees one session BEYOND ``T`` for calendar
    navigation only, so a day-``T`` signal gets a real next-session fill date and
    a true month-end on ``T`` is detected. Those next-session orders are returned
    as ``result.pending_orders`` -- created today, filled on the next processed
    session (the existing one-trading-day signal-to-fill lag; no same-close
    execution).
  - ``defer_unfillable``: an order whose scheduled fill price is unavailable is
    reported as a stale/rejected order, never silently filled.

Sleeves are fully independent (no shared cash, no cross-sleeve transfers, no
rebalancing) -- exactly like ``--strategy portfolio``. The account is static:
capital is split once at init (60/35/5 of the starting capital) and weights then
drift with performance.

Persistence is split across two files in the state dir:
  - ``paper_config.json``  -- immutable account identity (created-at timestamp,
    starting capital, weights, cost/fractional settings, frozen universe
    snapshot + version, strategy params, git SHA, schema version, inception).
  - ``paper_state.json``   -- the evolving derived ledger snapshot (last
    processed date, per-sleeve cash/positions/equity, pending orders, completed
    trades, stale orders, equity histories, costs/turnover, run log,
    reconciliation).
Both are written atomically (temp file + fsync + rename); a file lock prevents
concurrent runs from corrupting state; schema is validated on load and the run
fails closed on corrupt/inconsistent state without overwriting it.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config, data
from . import tournament as tournament_module
from .engine import run_backtest
from .portfolio import (
    DEFAULT_PORTFOLIO_WEIGHTS,
    allocate_capital,
    parse_portfolio_weights,
    validate_portfolio_weights,
)

STATE_SCHEMA_VERSION = 1
DEFAULT_PAPER_STATE_DIR = "paper_state"
CONFIG_FILENAME = "paper_config.json"
STATE_FILENAME = "paper_state.json"
LOCK_FILENAME = ".paper_lock"
BACKUP_DIRNAME = "backups"

# Extra warmup buffer (calendar days) fetched before the account inception so a
# sleeve's indicators are warm on the very first processed session. Uses the
# largest strategy warmup (momentum/regime = 500) so any registered sleeve is
# covered.
WARMUP_BUFFER_DAYS = max(
    config.MEAN_REVERSION_WARMUP_CALENDAR_DAYS,
    config.MOMENTUM_WARMUP_CALENDAR_DAYS,
    config.REGIME_WARMUP_CALENDAR_DAYS,
)

RECONCILE_ABS_TOL = 0.01     # $0.01
RECONCILE_REL_TOL = 1e-6

# Shown at the top of every report/status output (requirement 20).
PAPER_NOTICES = [
    "PAPER TRADING ONLY",
    "NO REAL ORDERS WERE SENT",
    "NOT FINANCIAL ADVICE",
    "NO BROKERAGE CONNECTION EXISTS",
    "RESULTS MAY DIFFER FROM LIVE EXECUTION",
]


class PaperError(Exception):
    """Base class for paper-trading errors."""


class PaperStateError(PaperError):
    """Persistent state is missing, corrupt, schema-mismatched, or internally
    inconsistent. The run fails closed rather than overwriting it."""


class PaperLockError(PaperError):
    """Another paper run holds the state lock."""


# --- Small persistence utilities -----------------------------------------------

def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def _atomic_write_json(path: Path, obj: dict) -> None:
    """Write JSON via a temp file in the same directory, fsync, then atomic
    rename -- a crash mid-write leaves the previous file intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic on POSIX
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


@contextmanager
def _file_lock(state_dir: Path):
    """Exclusive, non-blocking advisory lock on the state directory. A second
    concurrent paper run fails fast rather than racing on the same state."""
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / LOCK_FILENAME
    f = open(lock_path, "w")
    try:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            raise PaperLockError(
                f"Another paper run holds the lock on {state_dir} "
                f"({lock_path}). Wait for it to finish, or remove the lock file "
                f"if you are certain no other run is active."
            )
        yield
    finally:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()


# --- Paths ---------------------------------------------------------------------

def config_path(state_dir) -> Path:
    return Path(state_dir) / CONFIG_FILENAME


def state_path(state_dir) -> Path:
    return Path(state_dir) / STATE_FILENAME


def account_exists(state_dir) -> bool:
    return config_path(state_dir).exists() and state_path(state_dir).exists()


# --- Schema validation (fail closed) -------------------------------------------

_REQUIRED_CONFIG_KEYS = {
    "schema_version", "created_at", "starting_capital", "weights", "cost_bps",
    "fractional_shares", "inception_date", "universe_mode", "universe_tickers",
    "strategy_params", "git_commit_sha",
}
_REQUIRED_STATE_KEYS = {
    "schema_version", "last_processed_date", "total_equity", "starting_capital",
    "sleeves", "pending_orders", "completed_trades", "rejected_stale_orders",
    "realized_trades", "portfolio_equity_history", "sleeve_equity_history",
    "run_log", "reconciliation",
}


def _validate_config(cfg: dict) -> None:
    if not isinstance(cfg, dict):
        raise PaperStateError("paper_config.json is not a JSON object.")
    missing = _REQUIRED_CONFIG_KEYS - set(cfg)
    if missing:
        raise PaperStateError(f"paper_config.json is missing required keys: {sorted(missing)}.")
    if cfg["schema_version"] != STATE_SCHEMA_VERSION:
        raise PaperStateError(
            f"paper_config.json schema_version {cfg['schema_version']} != "
            f"expected {STATE_SCHEMA_VERSION}. Refusing to run against an incompatible account."
        )


def _validate_state(st: dict) -> None:
    if not isinstance(st, dict):
        raise PaperStateError("paper_state.json is not a JSON object.")
    missing = _REQUIRED_STATE_KEYS - set(st)
    if missing:
        raise PaperStateError(f"paper_state.json is missing required keys: {sorted(missing)}.")
    if st["schema_version"] != STATE_SCHEMA_VERSION:
        raise PaperStateError(
            f"paper_state.json schema_version {st['schema_version']} != "
            f"expected {STATE_SCHEMA_VERSION}. Refusing to run against an incompatible account."
        )


def load_account(state_dir) -> tuple[dict, dict]:
    """Load and validate (config, state). Raises PaperStateError -- failing
    closed -- if either file is missing, unparseable, schema-mismatched, or
    missing required keys. Never overwrites on failure."""
    cpath, spath = config_path(state_dir), state_path(state_dir)
    if not cpath.exists() or not spath.exists():
        raise PaperStateError(
            f"No paper account found in {state_dir} (expected {CONFIG_FILENAME} and "
            f"{STATE_FILENAME}). Run --paper-init first."
        )
    try:
        cfg = _read_json(cpath)
    except (json.JSONDecodeError, OSError) as e:
        raise PaperStateError(f"paper_config.json is corrupt/unreadable ({e}). Not overwriting.") from e
    try:
        st = _read_json(spath)
    except (json.JSONDecodeError, OSError) as e:
        raise PaperStateError(f"paper_state.json is corrupt/unreadable ({e}). Not overwriting.") from e
    _validate_config(cfg)
    _validate_state(st)
    if st["starting_capital"] != cfg["starting_capital"]:
        raise PaperStateError(
            "paper_state.json and paper_config.json disagree on starting_capital "
            f"({st['starting_capital']} vs {cfg['starting_capital']}); inconsistent account."
        )
    return cfg, st


# --- Calendar / session helpers ------------------------------------------------

def _project_next_weekday(d: pd.Timestamp) -> pd.Timestamp:
    """Next Mon-Fri after ``d`` (a projected next session used only to LABEL a
    frontier pending order's fill date when the real next session is not yet a
    finalized bar; the actual fill always lands on the next real session that a
    later run reaches)."""
    nxt = d + pd.Timedelta(days=1)
    while nxt.dayofweek >= 5:  # Sat=5, Sun=6
        nxt += pd.Timedelta(days=1)
    return nxt


def finalized_calendar(inception, refresh_cache: bool = False) -> pd.DatetimeIndex:
    """The finalized SPY trading sessions from a bit before inception through
    the latest finalized bar (today's not-yet-finalized bar excluded). Only the
    DATES are used (calendar navigation is public information); no price beyond a
    processed date is ever read by a sleeve replay."""
    fetch_start = pd.Timestamp(inception) - pd.Timedelta(days=WARMUP_BUFFER_DAYS)
    today = pd.Timestamp.now().normalize()
    spy = data.get_benchmark_data(fetch_start, today, force_refresh=refresh_cache)
    cal = data.build_canonical_calendar(spy, fetch_start, today)
    cal, _ = data.exclude_unfinalized_today(cal)
    return cal


def _sessions_on_or_after(cal: pd.DatetimeIndex, d: pd.Timestamp) -> pd.DatetimeIndex:
    return cal[cal >= pd.Timestamp(d)]


def _next_session(cal: pd.DatetimeIndex, after: pd.Timestamp | None, inception) -> pd.Timestamp | None:
    """Next finalized session to process: the first session >= inception if
    nothing processed yet, else the first session strictly after ``after``.
    Returns None if the account is already at the latest finalized session."""
    if after is None:
        candidates = cal[cal >= pd.Timestamp(inception)]
    else:
        candidates = cal[cal > pd.Timestamp(after)]
    return candidates[0] if len(candidates) else None


# --- Account initialization ----------------------------------------------------

def init_account(
    state_dir,
    starting_capital: float,
    inception,
    weights: list[tuple[str, float]],
    cost_bps: float,
    fractional_shares: bool,
    universe_mode: str,
    universe_tickers: list[str] | None,
    universe_info: dict | None,
    registry: dict | None = None,
    overwrite: bool = False,
) -> dict:
    """Create a fresh paper account: validate weights, freeze the universe
    snapshot + strategy params, and write paper_config.json + an all-cash
    initial paper_state.json (each sleeve holding its allocated slice, no
    positions, nothing processed yet). Refuses to clobber an existing account
    unless ``overwrite`` (used by --paper-reset, which backs up first)."""
    registry = registry or tournament_module.STRATEGY_REGISTRY
    state_dir = Path(state_dir)
    if account_exists(state_dir) and not overwrite:
        raise PaperError(
            f"A paper account already exists in {state_dir}. Use --paper-reset "
            f"--confirm-paper-reset to reset it (a backup is made first)."
        )
    if starting_capital <= 0:
        raise PaperError(f"starting capital must be > 0, got {starting_capital}")
    validate_portfolio_weights(weights, registry)

    allocations = allocate_capital(weights, starting_capital)
    strategy_params: dict[str, dict] = {}
    for name, _w in weights:
        spec = registry[name]
        if spec.uses_stock_universe:
            strat = spec.factory(universe=universe_tickers)
        else:
            strat = spec.factory()
        strategy_params[name] = strat.describe().get("params", {})

    created_at = datetime.now(timezone.utc).isoformat()
    inception_ts = pd.Timestamp(inception).normalize()

    cfg = {
        "schema_version": STATE_SCHEMA_VERSION,
        "created_at": created_at,
        "inception_date": str(inception_ts.date()),
        "starting_capital": float(starting_capital),
        "weights": [[name, float(w)] for name, w in weights],
        "cost_bps": float(cost_bps),
        "fractional_shares": bool(fractional_shares),
        "universe_mode": universe_mode,
        "universe_tickers": list(universe_tickers) if universe_tickers else [],
        "universe_info": universe_info or {},
        "universe_snapshot_version": (universe_info or {}).get("snapshot_date"),
        "strategy_params": strategy_params,
        "git_commit_sha": _git_sha(),
    }

    sleeves = {}
    sleeve_equity_history: dict[str, list] = {}
    for (name, w), (_n, alloc) in zip(weights, allocations):
        sleeves[name] = {
            "strategy": name,
            "weight": float(w),
            "allocated_capital": float(alloc),
            "cash": float(alloc),
            "equity": float(alloc),
            "ending_weight": float(w),
            "positions": [],
            "transaction_costs": 0.0,
            "turnover": 0.0,
        }
        sleeve_equity_history[name] = []

    st = {
        "schema_version": STATE_SCHEMA_VERSION,
        "created_at": created_at,
        "inception_date": str(inception_ts.date()),
        "last_processed_date": None,
        "data_cutoff_date": None,
        "num_runs": 0,
        "starting_capital": float(starting_capital),
        "total_equity": float(starting_capital),
        "cumulative_return": 0.0,
        "transaction_costs_total": 0.0,
        "turnover_total": 0.0,
        "sleeves": sleeves,
        "pending_orders": [],
        "completed_trades": [],
        "rejected_stale_orders": [],
        "realized_trades": [],
        "portfolio_equity_history": [],
        "sleeve_equity_history": sleeve_equity_history,
        "run_log": [],
        "reconciliation": {"ok": True, "checks": [], "note": "no sessions processed yet"},
        "git_commit_sha": cfg["git_commit_sha"],
    }

    with _file_lock(state_dir):
        _atomic_write_json(config_path(state_dir), cfg)
        _atomic_write_json(state_path(state_dir), st)
    return {"config": cfg, "state": st}


# --- Per-sleeve replay to a processed date -------------------------------------

@dataclass
class SleeveSnapshot:
    strategy: str
    weight: float
    allocated_capital: float
    cash: float
    equity: float
    positions: list[dict]          # ticker/shares/avg_cost/last_price/market_value
    transactions: list             # engine Transaction, fill_date <= S
    trades: list                   # engine Trade (realized)
    pending: list                  # engine TargetEvent, fill_date == next session
    stale: list                    # engine TargetEvent, unfillable
    equity_curve: pd.Series
    transaction_costs: float
    turnover: float


def _build_sleeve_strategy(spec, universe_tickers):
    if spec.uses_stock_universe:
        return spec.factory(universe=universe_tickers)
    return spec.factory()


def _replay_sleeve_to_date(
    spec, universe_tickers, inception, session, next_session,
    allocated, cost_bps, fractional_shares, refresh_cache,
) -> SleeveSnapshot:
    """Deterministically replay one sleeve from ``inception`` through the
    processed ``session`` and read off its settled state, plus the pending
    (next-session) and stale orders. Reuses the tournament data-prep + the
    backtest engine verbatim, so paper cannot diverge from backtest."""
    strategy = _build_sleeve_strategy(spec, universe_tickers)
    if spec.data_plan == "stock":
        prepared = tournament_module.prepare_stock_plan(spec, strategy, inception, session, refresh_cache)
    elif spec.data_plan == "sector":
        prepared = tournament_module.prepare_sector_plan(strategy, inception, session, refresh_cache)
    else:
        raise PaperError(f"Unknown data plan {spec.data_plan!r} for {spec.name!r}")

    full_cal = prepared.full_calendar
    in_window = full_cal[full_cal <= pd.Timestamp(session)]
    if len(in_window) == 0:
        raise PaperError(
            f"{spec.name}: no trading sessions in [{pd.Timestamp(inception).date()}, "
            f"{pd.Timestamp(session).date()}] to process."
        )
    exec_end = in_window[-1]
    # Extend the strategy's calendar by exactly one session (the real next
    # session, or a projected weekday at the live frontier) so a day-`exec_end`
    # signal gets a valid future fill date and a true month-end is detected.
    sig_cal = in_window
    if next_session is not None:
        sig_cal = in_window.append(pd.DatetimeIndex([pd.Timestamp(next_session)])).unique().sort_values()

    result = run_backtest(
        strategy, prepared.clean_price_data, full_cal, prepared.effective_start, exec_end,
        allocated, cost_bps, fractional_shares,
        signal_calendar=sig_cal, defer_unfillable=True,
    )
    return _snapshot_from_result(spec.name, allocated, result, exec_end)


def _snapshot_from_result(name, allocated, result, session) -> SleeveSnapshot:
    session = pd.Timestamp(session)
    # Cash and per-ticker shares at `session` come from the day's position rows.
    rows_at_session = [p for p in result.positions if pd.Timestamp(p["date"]) == session]
    cash = rows_at_session[0]["cash"] if rows_at_session else float(allocated)
    equity = float(result.equity_curve.loc[session]) if session in result.equity_curve.index else float(allocated)

    # avg_cost per still-held ticker: the last transaction on/before `session`.
    last_tx_by_ticker: dict[str, object] = {}
    for tx in result.transactions:
        if pd.Timestamp(tx.fill_date) <= session:
            last_tx_by_ticker[tx.ticker] = tx

    positions = []
    for row in rows_at_session:
        shares = float(row["shares"])
        if abs(shares) < 1e-9:
            continue
        tx = last_tx_by_ticker.get(row["ticker"])
        avg_cost = float(tx.avg_cost_basis_after) if tx is not None else float("nan")
        positions.append({
            "ticker": row["ticker"],
            "shares": shares,
            "avg_cost": avg_cost,
            "last_price": float(row["adjusted_close"]),
            "market_value": float(row["market_value"]),
        })

    txns = [tx for tx in result.transactions if pd.Timestamp(tx.fill_date) <= session]
    costs = sum(tx.transaction_cost for tx in txns)
    turnover = sum(tx.executed_notional for tx in txns)
    return SleeveSnapshot(
        strategy=name,
        weight=0.0,  # filled in by caller
        allocated_capital=float(allocated),
        cash=float(cash),
        equity=equity,
        positions=positions,
        transactions=txns,
        trades=list(result.trades),
        pending=list(result.pending_orders),
        stale=list(result.stale_orders),
        equity_curve=result.equity_curve,
        transaction_costs=float(costs),
        turnover=float(turnover),
    )


# --- Reconciliation ------------------------------------------------------------

def _approx(a: float, b: float) -> bool:
    return abs(a - b) <= RECONCILE_ABS_TOL + RECONCILE_REL_TOL * max(abs(a), abs(b))


def _reconcile(snapshots: list[SleeveSnapshot], starting_capital: float, portfolio_equity: float) -> dict:
    """Verify the ledger's internal invariants (requirement 17). Returns a
    structured result; the caller fails closed if ``ok`` is False."""
    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    # 1. Starting sleeve capital sums to the account capital.
    alloc_sum = sum(s.allocated_capital for s in snapshots)
    record("allocations_sum_to_capital", _approx(alloc_sum, starting_capital),
           f"sum(allocated)={alloc_sum:.2f} vs starting_capital={starting_capital:.2f}")

    # 2. Total portfolio equity equals the sum of sleeve equity.
    sleeve_eq_sum = sum(s.equity for s in snapshots)
    record("portfolio_equity_is_sum_of_sleeves", _approx(sleeve_eq_sum, portfolio_equity),
           f"sum(sleeve equity)={sleeve_eq_sum:.2f} vs portfolio_equity={portfolio_equity:.2f}")

    for s in snapshots:
        # 3. Sleeve equity == cash + market value of positions.
        mv = sum(p["market_value"] for p in s.positions)
        record(f"{s.strategy}:equity=cash+positions", _approx(s.cash + mv, s.equity),
               f"cash={s.cash:.2f}+mv={mv:.2f} vs equity={s.equity:.2f}")

        # 4. Completed fills reconcile with cash and share changes.
        cash = s.allocated_capital
        shares: dict[str, float] = {}
        for tx in sorted(s.transactions, key=lambda t: (pd.Timestamp(t.fill_date), t.ticker)):
            if tx.action == "buy":
                cash -= tx.executed_notional + tx.transaction_cost
                shares[tx.ticker] = shares.get(tx.ticker, 0.0) + tx.shares_traded
            else:
                cash += tx.executed_notional - tx.transaction_cost
                shares[tx.ticker] = shares.get(tx.ticker, 0.0) - tx.shares_traded
        record(f"{s.strategy}:fills_reconcile_cash", _approx(cash, s.cash),
               f"cash from fills={cash:.2f} vs snapshot cash={s.cash:.2f}")
        pos_shares = {p["ticker"]: p["shares"] for p in s.positions}
        shares_ok = True
        for ticker, qty in shares.items():
            if abs(qty) < 1e-9:
                continue
            if not _approx(qty, pos_shares.get(ticker, 0.0)):
                shares_ok = False
                break
        record(f"{s.strategy}:fills_reconcile_shares", shares_ok,
               "share counts from fills match snapshot positions")

    return {"ok": all(c["ok"] for c in checks), "checks": checks}


# --- Ledger assembly -----------------------------------------------------------

def _order_from_event(sleeve: str, e, status: str, current_mv: float = 0.0) -> dict:
    fill = pd.Timestamp(e.fill_date)
    sig = pd.Timestamp(e.signal_date)
    requested = float(e.requested_notional) if e.requested_notional is not None else 0.0
    if e.target_weight <= 0:
        side = "exit"
    elif requested >= current_mv:
        side = "buy"
    else:
        side = "sell"  # trim an existing position down toward the target
    return {
        "id": f"ord:{sleeve}:{e.ticker}:{sig.date()}:{fill.date()}",
        "sleeve": sleeve,
        "ticker": e.ticker,
        "signal_date": str(sig.date()),
        "scheduled_fill_date": str(fill.date()),
        "intended_side": side,
        "target_weight": float(e.target_weight),
        "requested_notional": float(e.requested_notional) if e.requested_notional is not None else None,
        "sizing_price": float(e.sizing_price),
        "reason": e.reason,
        "status": status,
    }


def _fill_from_tx(sleeve: str, tx) -> dict:
    fill = pd.Timestamp(tx.fill_date)
    sig = pd.Timestamp(tx.signal_date)
    return {
        "id": f"fill:{sleeve}:{tx.ticker}:{fill.date()}:{tx.action}",
        "sleeve": sleeve,
        "ticker": tx.ticker,
        "signal_date": str(sig.date()),
        "fill_date": str(fill.date()),
        "side": tx.action,
        "shares": float(tx.shares_traded),
        "fill_price": float(tx.fill_price),
        "notional": float(tx.executed_notional),
        "transaction_cost": float(tx.transaction_cost),
        "reason": tx.reason,
    }


def _realized_from_trade(sleeve: str, tr) -> dict:
    d = pd.Timestamp(tr.date)
    return {
        "id": f"trade:{sleeve}:{tr.ticker}:{d.date()}:{tr.event_type}",
        "sleeve": sleeve,
        "ticker": tr.ticker,
        "event_type": tr.event_type,
        "date": str(d.date()),
        "shares_sold": float(tr.shares_sold),
        "sale_price": float(tr.sale_price),
        "avg_cost_basis": float(tr.avg_cost_basis),
        "realized_pnl": float(tr.realized_pnl),
        "realized_pnl_net": None if tr.realized_pnl_net is None else float(tr.realized_pnl_net),
        "reason": tr.reason,
        "holding_days": int(tr.holding_days),
    }


def _combine_equity_histories(snapshots: list[SleeveSnapshot]) -> tuple[pd.Series, dict[str, pd.Series]]:
    """Portfolio equity per date = sum of sleeve equity curves over their common
    date intersection (never an outer join). Returns (portfolio_curve,
    {sleeve: reindexed_curve})."""
    common = snapshots[0].equity_curve.index
    for s in snapshots[1:]:
        common = common.intersection(s.equity_curve.index)
    common = common.sort_values()
    sleeve_curves = {s.strategy: s.equity_curve.reindex(common) for s in snapshots}
    portfolio = None
    for curve in sleeve_curves.values():
        portfolio = curve.copy() if portfolio is None else portfolio + curve
    return portfolio, sleeve_curves


def _process_session(cfg: dict, session: pd.Timestamp, next_session, refresh_cache: bool) -> dict:
    """Replay every sleeve to ``session`` and assemble the full ledger snapshot
    (not yet persisted). Returns the new state dict plus a per-session run
    summary under key ``_summary``."""
    registry = tournament_module.STRATEGY_REGISTRY
    weights = [(name, float(w)) for name, w in cfg["weights"]]
    allocations = allocate_capital(weights, cfg["starting_capital"])
    universe_tickers = cfg["universe_tickers"] or None

    snapshots: list[SleeveSnapshot] = []
    for (name, w), (_n, alloc) in zip(weights, allocations):
        spec = registry[name]
        snap = _replay_sleeve_to_date(
            spec, universe_tickers, cfg["inception_date"], session, next_session,
            alloc, cfg["cost_bps"], cfg["fractional_shares"], refresh_cache,
        )
        snap.weight = w
        snapshots.append(snap)

    portfolio_curve, sleeve_curves = _combine_equity_histories(snapshots)
    portfolio_equity = float(portfolio_curve.iloc[-1])

    reconciliation = _reconcile(snapshots, cfg["starting_capital"], portfolio_equity)

    # Assemble ledger collections.
    sleeves_out: dict[str, dict] = {}
    pending_orders: list[dict] = []
    completed_trades: list[dict] = []
    stale_orders: list[dict] = []
    realized_trades: list[dict] = []
    for snap in snapshots:
        sleeves_out[snap.strategy] = {
            "strategy": snap.strategy,
            "weight": snap.weight,
            "allocated_capital": snap.allocated_capital,
            "cash": snap.cash,
            "equity": snap.equity,
            "ending_weight": (snap.equity / portfolio_equity) if portfolio_equity else float("nan"),
            "positions": snap.positions,
            "transaction_costs": snap.transaction_costs,
            "turnover": snap.turnover,
        }
        mv_by_ticker = {p["ticker"]: p["market_value"] for p in snap.positions}
        pending_orders.extend(
            _order_from_event(snap.strategy, e, "pending", mv_by_ticker.get(e.ticker, 0.0))
            for e in snap.pending
        )
        stale_orders.extend(
            _order_from_event(snap.strategy, e, "stale", mv_by_ticker.get(e.ticker, 0.0))
            for e in snap.stale
        )
        completed_trades.extend(_fill_from_tx(snap.strategy, tx) for tx in snap.transactions)
        realized_trades.extend(_realized_from_trade(snap.strategy, tr) for tr in snap.trades)

    portfolio_equity_history = [
        {"date": str(pd.Timestamp(d).date()), "equity": float(v)}
        for d, v in portfolio_curve.items()
    ]
    sleeve_equity_history = {
        name: [{"date": str(pd.Timestamp(d).date()), "equity": float(v)} for d, v in curve.items()]
        for name, curve in sleeve_curves.items()
    }

    # This-session activity for the run summary.
    fills_this_session = [f for f in completed_trades if f["fill_date"] == str(session.date())]
    costs_total = sum(s.transaction_costs for s in snapshots)
    turnover_total = sum(s.turnover for s in snapshots)
    costs_run = sum(f["transaction_cost"] for f in fills_this_session)

    starting_capital = cfg["starting_capital"]
    cumulative_return = portfolio_equity / starting_capital - 1.0 if starting_capital else 0.0
    daily_return = 0.0
    if len(portfolio_curve) >= 2:
        prev = float(portfolio_curve.iloc[-2])
        daily_return = portfolio_equity / prev - 1.0 if prev else 0.0

    summary = {
        "paper_date": str(session.date()),
        "data_cutoff_date": str(session.date()),
        "next_session": str(pd.Timestamp(next_session).date()) if next_session is not None else None,
        "num_new_signals": len(pending_orders),
        "num_pending_created": len(pending_orders),
        "num_fills": len(fills_this_session),
        "num_stale": len(stale_orders),
        "total_equity": portfolio_equity,
        "cumulative_return": cumulative_return,
        "daily_return": daily_return,
        "transaction_costs_run": costs_run,
        "reconciliation_ok": reconciliation["ok"],
    }

    new_state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "created_at": cfg["created_at"],
        "inception_date": cfg["inception_date"],
        "last_processed_date": str(session.date()),
        "data_cutoff_date": str(session.date()),
        "starting_capital": starting_capital,
        "total_equity": portfolio_equity,
        "cumulative_return": cumulative_return,
        "transaction_costs_total": costs_total,
        "turnover_total": turnover_total,
        "sleeves": sleeves_out,
        "pending_orders": pending_orders,
        "completed_trades": completed_trades,
        "rejected_stale_orders": stale_orders,
        "realized_trades": realized_trades,
        "portfolio_equity_history": portfolio_equity_history,
        "sleeve_equity_history": sleeve_equity_history,
        "reconciliation": reconciliation,
        "git_commit_sha": cfg["git_commit_sha"],
        "_summary": summary,
    }
    return new_state


# --- Advancing the account -----------------------------------------------------

def advance(
    state_dir,
    target_date=None,
    refresh_cache: bool = False,
) -> dict:
    """Process finalized sessions forward. With ``target_date`` None, process
    exactly the next unprocessed session (--paper-run). With a ``target_date``,
    process every session up to and including the last finalized session on or
    before it (--paper-date). Idempotent: a date already processed is a no-op.

    Each processed session is persisted atomically before the next is processed,
    so state survives an interruption mid-loop and reloads cleanly between runs.
    Reconciliation must pass for a session to be committed; otherwise the run
    fails closed and the prior state is left intact.
    """
    state_dir = Path(state_dir)
    with _file_lock(state_dir):
        cfg, st = load_account(state_dir)
        inception = pd.Timestamp(cfg["inception_date"])
        cal = finalized_calendar(inception, refresh_cache=refresh_cache)
        if len(cal) == 0:
            raise PaperError("No finalized trading sessions available to process.")
        latest_finalized = cal[-1]

        last_processed = (
            pd.Timestamp(st["last_processed_date"]) if st["last_processed_date"] else None
        )

        # Resolve the target session (last finalized session <= target_date).
        if target_date is not None:
            target_ts = pd.Timestamp(target_date).normalize()
            if target_ts > latest_finalized:
                raise PaperError(
                    f"Requested paper date {target_ts.date()} is after the latest finalized "
                    f"market session ({latest_finalized.date()}). Refusing to process an "
                    f"unfinalized/future date."
                )
            eligible = cal[cal <= target_ts]
            if len(eligible) == 0 or eligible[-1] < inception:
                raise PaperError(
                    f"Requested paper date {target_ts.date()} is before the account inception "
                    f"({inception.date()}); nothing to process."
                )
            target_session = eligible[-1]
            if last_processed is not None and target_session <= last_processed:
                return {
                    "processed": [],
                    "message": f"Already processed through {last_processed.date()} "
                               f"(requested {target_ts.date()} resolves to {target_session.date()}).",
                    "config": cfg, "state": st,
                }
        else:
            target_session = None  # single next session

        processed_summaries: list[dict] = []
        current_last = last_processed
        while True:
            nxt = _next_session(cal, current_last, inception)
            if nxt is None:
                break
            if target_session is not None and nxt > target_session:
                break
            # The session AFTER `nxt` (real if finalized, else projected weekday)
            # is where `nxt`'s signals will fill.
            pos = cal.searchsorted(nxt, side="right")
            following = cal[pos] if pos < len(cal) else _project_next_weekday(nxt)

            new_state = _process_session(cfg, nxt, following, refresh_cache)
            if not new_state["reconciliation"]["ok"]:
                raise PaperStateError(
                    f"Reconciliation FAILED processing {nxt.date()}; refusing to commit. "
                    f"Prior state left intact. Checks: {new_state['reconciliation']['checks']}"
                )

            summary = new_state.pop("_summary")
            # Append this run to the append-only run log and persist atomically.
            run_index = st["num_runs"] + 1
            run_row = {"run_index": run_index, "timestamp": datetime.now(timezone.utc).isoformat(), **summary}
            new_state["num_runs"] = run_index
            new_state["run_log"] = list(st["run_log"]) + [run_row]
            _atomic_write_json(state_path(state_dir), new_state)

            st = new_state
            processed_summaries.append(summary)
            current_last = nxt
            if target_session is None:
                break  # --paper-run advances exactly one session

        if not processed_summaries:
            msg = (
                f"Already up to date; latest finalized session {latest_finalized.date()} "
                f"is already processed." if last_processed is not None
                else "No sessions to process."
            )
            return {"processed": [], "message": msg, "config": cfg, "state": st}

        return {"processed": processed_summaries, "message": None, "config": cfg, "state": st}


# --- Reset ---------------------------------------------------------------------

def reset_account(state_dir, confirm: bool) -> Path:
    """Reset the account, but only with explicit confirmation. Backs up the
    current state dir first (never deletes state silently), then removes the
    live config/state so a fresh --paper-init can proceed."""
    state_dir = Path(state_dir)
    if not confirm:
        raise PaperError(
            "Refusing to reset the paper account without --confirm-paper-reset. "
            "This guard prevents accidental loss of the ledger."
        )
    if not account_exists(state_dir):
        raise PaperError(f"No paper account found in {state_dir} to reset.")

    ts = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
    backup_dir = state_dir / BACKUP_DIRNAME / f"reset_{ts}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    with _file_lock(state_dir):
        for fname in (CONFIG_FILENAME, STATE_FILENAME):
            src = state_dir / fname
            if src.exists():
                shutil.copy2(src, backup_dir / fname)
        # Remove the live files (backed up above) so re-init is clean.
        for fname in (CONFIG_FILENAME, STATE_FILENAME):
            src = state_dir / fname
            if src.exists():
                src.unlink()
    return backup_dir


# --- Reconcile-on-demand -------------------------------------------------------

def reconcile_saved_state(state_dir) -> dict:
    """Re-verify the invariants against the PERSISTED snapshot (no replay): the
    same identities checked at write time, recomputed from what is on disk, so a
    hand-edited or bit-rotted state file is caught."""
    cfg, st = load_account(state_dir)
    checks: list[dict] = []

    def record(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    sleeves = st["sleeves"]
    alloc_sum = sum(s["allocated_capital"] for s in sleeves.values())
    record("allocations_sum_to_capital", _approx(alloc_sum, st["starting_capital"]),
           f"sum(allocated)={alloc_sum:.2f} vs starting_capital={st['starting_capital']:.2f}")

    sleeve_eq_sum = sum(s["equity"] for s in sleeves.values())
    record("portfolio_equity_is_sum_of_sleeves", _approx(sleeve_eq_sum, st["total_equity"]),
           f"sum(sleeve equity)={sleeve_eq_sum:.2f} vs total_equity={st['total_equity']:.2f}")

    end_wt_sum = sum(s["ending_weight"] for s in sleeves.values()) if sleeves else 0.0
    if st["last_processed_date"] is not None:
        record("ending_weights_sum_to_one", _approx(end_wt_sum, 1.0),
               f"sum(ending_weight)={end_wt_sum:.4f}")

    for name, s in sleeves.items():
        mv = sum(p["market_value"] for p in s["positions"])
        record(f"{name}:equity=cash+positions", _approx(s["cash"] + mv, s["equity"]),
               f"cash={s['cash']:.2f}+mv={mv:.2f} vs equity={s['equity']:.2f}")

    return {"ok": all(c["ok"] for c in checks), "checks": checks,
            "config": cfg, "state": st}
