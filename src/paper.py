"""Forward paper-trading ledger for the fixed 60/35/5 portfolio.

This is a *simulation* layer, not a broker. It NEVER connects to Robinhood or
any brokerage, never transmits an order, and holds no credentials.

Architecture -- a true incremental forward ledger, not a full-history replay
====================================================================
The authoritative store is a per-account SQLite database
(``paper_state/paper_ledger.db``, see ``src.paper_db`` for the schema):
append-only ``signals``/``orders``/``fills``/``realized_trades``/
``equity_history``/``sleeve_equity_history``/``run_log`` tables, plus the
materialized *current* ``sleeves``/``positions`` tables (always independently
reconstructible from the fills ledger -- see ``paper_db.reconcile``).

Each ``advance()`` call processes ONE finalized session at a time, per sleeve,
strictly incrementally:
  1. load + pre-validate the existing ledger (fails closed on any semantic
     inconsistency, BEFORE touching anything new);
  2. fill (or mark stale) this sleeve's already-persisted PENDING orders whose
     scheduled_fill_date is this session, using the finalized closing price --
     this is the only place fills ever happen, and it only ever consumes
     orders created on a PRIOR session (the existing one-trading-day
     signal-to-fill lag; no same-close execution);
  3. update positions/cash incrementally from those fills (average-cost
     accounting, reusing ``engine._execute_events`` -- the exact same
     buy/sell/cash-scaling code the backtest engine uses -- applied to the
     persisted state instead of a fresh in-memory walk);
  4. mark every current holding to market and record one equity row per
     sleeve, then one portfolio-level equity row;
  5. generate this session's NEW signals by replaying the strategy from
     inception through this session (reusing the exact same
     ``tournament.prepare_stock_plan``/``prepare_sector_plan`` +
     ``engine.run_backtest`` machinery the backtest/portfolio modes use) --
     but that replay's ACCOUNTING (cash, positions, equity, historical fills)
     is discarded entirely; only ``result.pending_orders`` whose
     ``signal_date`` is this session are ever persisted, as brand-new signal +
     order rows. Every already-persisted signal/order/fill/cost/equity row is
     untouched;
  6. independently reconcile the just-updated ledger (``paper_db.reconcile``);
  7. commit the whole session (every sleeve, every table write) in ONE SQLite
     transaction if reconciliation passes, else roll back everything and raise
     -- prior state is left byte-for-byte intact.

Because market data revisions, later strategy-code edits, or config changes
can never rewrite an already-committed session's rows (there is no "replay
from scratch and overwrite" step), the ledger can never retroactively change
what already happened -- unlike a full-history-replay design. A frozen
config fingerprint (weights, strategy params, cost model, universe, and
strategy source hashes -- see ``compute_config_fingerprint``) is checked
before every ``advance()``; if the runtime strategy/config has drifted from
what the account was initialized with, advancing is refused (fail closed)
while every read-only command (status/orders/trades/export/reconcile) keeps
working against the untouched ledger.

Scheduled fill dates use ``src.nyse_calendar`` (a fixed holiday-aware NYSE
session calendar), not a weekend-only projection, so a signal generated right
before a market holiday still gets the correct next-session fill date.

Sleeves are fully independent (no shared cash, no cross-sleeve transfers, no
rebalancing) -- exactly like ``--strategy portfolio``. The account is static:
capital is split once at init (60/35/5 of the starting capital) and weights
then drift with performance.

``paper_config.json``/``paper_state.json`` and the CSV artifacts are
non-authoritative EXPORTED VIEWS of the SQLite ledger, regenerated on every
operation for convenience/inspection -- the database is the source of truth.
"""

from __future__ import annotations

import fcntl
import hashlib
import inspect
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config, data, nyse_calendar
from . import paper_db
from . import tournament as tournament_module
from . import engine as engine_module
from .engine import run_backtest
from .portfolio import (
    DEFAULT_PORTFOLIO_WEIGHTS,
    allocate_capital,
    parse_portfolio_weights,
    validate_portfolio_weights,
)
from .strategies.base import TargetEvent

DEFAULT_PAPER_STATE_DIR = "paper_state"
# Non-authoritative exported views of the SQLite ledger (see module docstring).
CONFIG_FILENAME = "paper_config.json"
STATE_FILENAME = "paper_state.json"
BACKUP_DIRNAME = "backups"

# Bump when the session-processing algorithm changes materially (e.g. a fix
# that would change what signals/fills a session produces) -- part of the
# config fingerprint, so an account initialized under an older algorithm
# refuses to silently advance under the new one.
SIGNAL_EXECUTION_VERSION = 1

# Extra warmup buffer (calendar days) fetched before the account inception so
# a sleeve's indicators are warm on the very first processed session. Uses the
# largest strategy warmup (momentum/regime = 500) so any registered sleeve is
# covered.
WARMUP_BUFFER_DAYS = max(
    config.MEAN_REVERSION_WARMUP_CALENDAR_DAYS,
    config.MOMENTUM_WARMUP_CALENDAR_DAYS,
    config.REGIME_WARMUP_CALENDAR_DAYS,
)

PRICE_SEMANTICS = "adjusted_close" if config.YFINANCE_AUTO_ADJUST else "raw_close"
DATA_SOURCE = "yfinance"

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
    """Persistent ledger is missing, corrupt, schema-mismatched, or
    internally inconsistent. The run fails closed rather than overwriting or
    repairing it."""


class PaperFingerprintError(PaperStateError):
    """The runtime strategy/config fingerprint no longer matches the
    account's frozen fingerprint from init. Advancing is refused; read-only
    commands remain available."""


class PaperLockError(PaperError):
    """Another paper run holds the state lock."""


# --- Small filesystem utilities --------------------------------------------------

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
    rename -- used only for the derived paper_config.json/paper_state.json
    VIEWS (the SQLite ledger is authoritative; see paper_db.py)."""
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


@contextmanager
def _file_lock(state_dir: Path):
    """Exclusive, non-blocking advisory lock on the state directory. A second
    concurrent paper run fails fast rather than racing on the same ledger."""
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / ".paper_lock"
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


# --- Paths -------------------------------------------------------------------------

def config_path(state_dir) -> Path:
    return Path(state_dir) / CONFIG_FILENAME


def state_path(state_dir) -> Path:
    return Path(state_dir) / STATE_FILENAME


def account_exists(state_dir) -> bool:
    """True only when a FULLY-CREATED account database is present. A ledger DB
    whose `sleeves` table is empty is a crashed/partial --paper-init (the file
    and schema exist, but the meta+sleeves commit never landed); treating that
    as "exists" would refuse a clean retry of --paper-init, so it reports
    False. A file that exists but isn't a readable account DB (corrupt/foreign)
    reports True, so we never silently clobber it."""
    path = paper_db.db_path(state_dir)
    if not path.exists():
        return False
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            has_sleeves = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='sleeves'"
            ).fetchone()[0]
            if not has_sleeves:
                return False
            return conn.execute("SELECT COUNT(*) FROM sleeves").fetchone()[0] > 0
        finally:
            conn.close()
    except sqlite3.Error:
        return True  # a file is there but unreadable -- do not clobber it


# --- Config fingerprint (freeze & enforce strategy identity) ----------------------

def _strategy_source_hash(name: str, registry) -> str:
    spec = registry[name]
    try:
        src = inspect.getsource(spec.factory)
    except (OSError, TypeError):
        src = repr(spec.factory)
    return hashlib.sha256(src.encode()).hexdigest()


def compute_config_fingerprint(
    weights: list[tuple[str, float]],
    strategy_params: dict,
    cost_bps: float,
    fractional_shares: bool,
    universe_mode: str,
    universe_tickers: list[str] | None,
    registry=None,
) -> str:
    """A deterministic SHA-256 fingerprint of everything that must stay fixed
    for this account's ledger to remain valid: the fixed weights, strategy
    parameters, cost/fractional-share assumptions, the frozen universe, the
    session-processing algorithm version, and each in-use strategy's own
    source code (so an edit to e.g. MomentumStrategy's on_day logic is
    detected even though its declared `params` dict didn't change)."""
    registry = registry or tournament_module.STRATEGY_REGISTRY
    payload = {
        "weights": sorted([[n, round(float(w), 10)] for n, w in weights]),
        "strategy_params": strategy_params,
        "cost_bps": float(cost_bps),
        "fractional_shares": bool(fractional_shares),
        "universe_mode": universe_mode,
        "universe_tickers_hash": hashlib.sha256(
            ",".join(sorted(universe_tickers or [])).encode()
        ).hexdigest(),
        "signal_execution_version": SIGNAL_EXECUTION_VERSION,
        "strategy_source_hashes": {
            name: _strategy_source_hash(name, registry) for name, _w in weights
        },
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _check_fingerprint(cfg: dict) -> None:
    weights = [(n, float(w)) for n, w in cfg["weights"]]
    runtime_fp = compute_config_fingerprint(
        weights, cfg["strategy_params"], cfg["cost_bps"], cfg["fractional_shares"],
        cfg["universe_mode"], cfg["universe_tickers"],
    )
    if runtime_fp != cfg["config_fingerprint"]:
        raise PaperFingerprintError(
            "Runtime strategy/config fingerprint does not match this account's frozen "
            f"fingerprint from init (expected {cfg['config_fingerprint'][:12]}..., got "
            f"{runtime_fp[:12]}...). The strategy code, weights, cost model, universe, or "
            "signal/execution version has changed since --paper-init. Refusing to advance "
            "and silently recompute history under changed code. Read-only commands "
            "(--paper-status/--paper-orders/--paper-trades/--paper-export/--paper-reconcile) "
            "remain available. An explicit migration (freezing a new fingerprint after "
            "deliberate review) must be implemented before this account can advance again."
        )


# --- Calendar / session helpers ----------------------------------------------------

def finalized_calendar(inception, refresh_cache: bool = False) -> pd.DatetimeIndex:
    """The finalized SPY trading sessions from a bit before inception through
    the latest finalized bar (today's not-yet-finalized bar excluded)."""
    fetch_start = pd.Timestamp(inception) - pd.Timedelta(days=WARMUP_BUFFER_DAYS)
    today = pd.Timestamp.now().normalize()
    spy = data.get_benchmark_data(fetch_start, today, force_refresh=refresh_cache)
    cal = data.build_canonical_calendar(spy, fetch_start, today)
    cal, _ = data.exclude_unfinalized_today(cal)
    return cal


def _next_session(cal: pd.DatetimeIndex, after: pd.Timestamp | None, inception) -> pd.Timestamp | None:
    """Next finalized session to process: the first session >= inception if
    nothing processed yet, else the first session strictly after `after`.
    Returns None if the account is already at the latest finalized session."""
    if after is None:
        candidates = cal[cal >= pd.Timestamp(inception)]
    else:
        candidates = cal[cal > pd.Timestamp(after)]
    return candidates[0] if len(candidates) else None


# --- Account initialization ---------------------------------------------------------

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
    snapshot + strategy params + a config fingerprint, and create an all-cash
    ledger (each sleeve holding its allocated slice, no positions, nothing
    processed yet). Refuses to clobber an existing account unless
    `overwrite` (used after --paper-reset, which backs up first)."""
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
        strat = spec.factory(universe=universe_tickers) if spec.uses_stock_universe else spec.factory()
        strategy_params[name] = strat.describe().get("params", {})

    created_at = datetime.now(timezone.utc).isoformat()
    inception_ts = pd.Timestamp(inception).normalize()
    universe_tickers_list = list(universe_tickers) if universe_tickers else []

    fingerprint = compute_config_fingerprint(
        weights, strategy_params, cost_bps, fractional_shares, universe_mode,
        universe_tickers_list, registry,
    )

    meta = {
        "schema_version": paper_db.SCHEMA_VERSION,
        "created_at": created_at,
        "inception_date": str(inception_ts.date()),
        "starting_capital": float(starting_capital),
        "weights": [[name, float(w)] for name, w in weights],
        "cost_bps": float(cost_bps),
        "fractional_shares": bool(fractional_shares),
        "universe_mode": universe_mode,
        "universe_tickers": universe_tickers_list,
        "universe_info": universe_info or {},
        "strategy_params": strategy_params,
        "git_commit_sha": _git_sha(),
        "config_fingerprint": fingerprint,
        "signal_execution_version": SIGNAL_EXECUTION_VERSION,
    }

    with _file_lock(state_dir):
        conn = paper_db.connect(state_dir, create=True)
        try:
            paper_db.write_meta(conn, meta)
            conn.execute("BEGIN IMMEDIATE")
            try:
                for (name, w), (_n, alloc) in zip(weights, allocations):
                    paper_db.insert_sleeve(conn, name, float(w), float(alloc), float(alloc))
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            cfg = build_config_view(conn)
            st = build_state_view(conn, cfg)
            _write_json_views(state_dir, cfg, st)
        finally:
            conn.close()
    return {"config": cfg, "state": st}


# --- Loading / views -----------------------------------------------------------------

_REQUIRED_META_KEYS = {
    "schema_version", "created_at", "inception_date", "starting_capital", "weights",
    "cost_bps", "fractional_shares", "universe_mode", "universe_tickers", "strategy_params",
    "git_commit_sha", "config_fingerprint", "signal_execution_version",
}


def build_config_view(conn: sqlite3.Connection) -> dict:
    """The account's immutable identity, read from `meta`. Validates required
    keys and the schema version -- fails closed (PaperStateError) rather than
    returning a partial view."""
    meta = paper_db.read_meta(conn)
    missing = _REQUIRED_META_KEYS - set(meta)
    if missing:
        raise PaperStateError(f"Paper ledger `meta` table is missing required keys: {sorted(missing)}.")
    if meta["schema_version"] != paper_db.SCHEMA_VERSION:
        raise PaperStateError(
            f"paper ledger schema_version {meta['schema_version']} != expected "
            f"{paper_db.SCHEMA_VERSION}. Refusing to run against an incompatible account."
        )
    return {
        "schema_version": meta["schema_version"],
        "created_at": meta["created_at"],
        "inception_date": meta["inception_date"],
        "starting_capital": meta["starting_capital"],
        "weights": meta["weights"],
        "cost_bps": meta["cost_bps"],
        "fractional_shares": meta["fractional_shares"],
        "universe_mode": meta["universe_mode"],
        "universe_tickers": meta["universe_tickers"],
        "universe_info": meta.get("universe_info", {}),
        "strategy_params": meta["strategy_params"],
        "git_commit_sha": meta["git_commit_sha"],
        "config_fingerprint": meta["config_fingerprint"],
        "signal_execution_version": meta["signal_execution_version"],
    }


def _order_view(row: sqlite3.Row) -> dict:
    return {
        "id": f"ORD-{row['id']}",
        "sleeve": row["sleeve"],
        "ticker": row["ticker"],
        "signal_date": row["sig_signal_date"],
        "scheduled_fill_date": row["scheduled_fill_date"],
        "intended_side": row["side"],
        "target_weight": row["sig_target_weight"],
        "requested_notional": row["requested_notional"],
        "sizing_price": row["sig_sizing_price"],
        "reason": row["sig_reason"],
        "status": row["status"],
    }


def _fill_view(row: sqlite3.Row) -> dict:
    return {
        "id": f"FILL-{row['id']}",
        "sleeve": row["sleeve"],
        "ticker": row["ticker"],
        "signal_date": row["sig_signal_date"],
        "fill_date": row["fill_date"],
        "side": row["side"],
        "shares": row["shares"],
        "fill_price": row["fill_price"],
        "notional": row["notional"],
        "transaction_cost": row["transaction_cost"],
        "reason": row["reason"],
    }


def _realized_view(row: sqlite3.Row) -> dict:
    return {
        "id": f"TRADE-{row['id']}",
        "sleeve": row["sleeve"],
        "ticker": row["ticker"],
        "event_type": row["event_type"],
        "date": row["trade_date"],
        "shares_sold": row["shares_sold"],
        "sale_price": row["sale_price"],
        "avg_cost_basis": row["avg_cost_basis"],
        "realized_pnl": row["realized_pnl"],
        "realized_pnl_net": row["realized_pnl_net"],
        "reason": row["reason"],
        "holding_days": row["holding_days"],
    }


def build_state_view(conn: sqlite3.Connection, cfg: dict | None = None,
                      reconciliation: dict | None = None) -> dict:
    """The evolving ledger, rendered into the same dict shape the paper
    reporting/CLI layer consumes -- a pure, read-only projection of the
    SQLite tables (never used to drive `advance()`'s own logic).

    `reconciliation` lets a caller that already ran `paper_db.reconcile`
    (e.g. `reconcile_saved_state`) pass it in so this view doesn't repeat the
    full per-sleeve fills-ledger replay a second time."""
    cfg = cfg or build_config_view(conn)
    sleeves_rows = paper_db.fetch_sleeves(conn)
    last_processed = paper_db.fetch_last_processed_session(conn)
    latest_eq = paper_db.fetch_latest_equity_row(conn)
    total_equity = latest_eq["total_equity"] if latest_eq else cfg["starting_capital"]
    cumulative_return = latest_eq["cumulative_return"] if latest_eq else 0.0

    sleeves_out: dict[str, dict] = {}
    for s in sleeves_rows:
        name = s["name"]
        positions_rows = paper_db.fetch_positions(conn, name)
        positions = [
            {
                "ticker": p["ticker"], "shares": p["shares"], "avg_cost": p["avg_cost"],
                "last_price": p["last_price"],
                "market_value": (p["shares"] or 0.0) * (p["last_price"] or 0.0),
            }
            for p in positions_rows
        ]
        latest_sleeve_eq = conn.execute(
            "SELECT * FROM sleeve_equity_history WHERE sleeve = ? ORDER BY session_date DESC LIMIT 1", (name,)
        ).fetchone()
        equity = latest_sleeve_eq["equity"] if latest_sleeve_eq else s["allocated_capital"]
        costs_row = conn.execute(
            "SELECT COALESCE(SUM(transaction_cost), 0) c, COALESCE(SUM(notional), 0) t "
            "FROM fills WHERE sleeve = ?", (name,)
        ).fetchone()
        sleeves_out[name] = {
            "strategy": name,
            "weight": s["weight"],
            "allocated_capital": s["allocated_capital"],
            "cash": s["cash"],
            "equity": equity,
            "ending_weight": (equity / total_equity) if total_equity else float("nan"),
            "positions": positions,
            "transaction_costs": costs_row["c"],
            "turnover": costs_row["t"],
        }

    pending_orders = [_order_view(r) for r in paper_db.fetch_all_pending_orders(conn)]
    stale_orders = [_order_view(r) for r in paper_db.fetch_all_stale_orders(conn)]
    fills = [_fill_view(r) for r in paper_db.fetch_fills_view(conn)]
    realized = [_realized_view(r) for r in paper_db.fetch_all_realized_trades(conn)]

    portfolio_equity_history = [
        {"date": r["session_date"], "equity": r["total_equity"]} for r in paper_db.fetch_equity_history(conn)
    ]
    sleeve_equity_history = {
        s["name"]: [{"date": r["session_date"], "equity": r["equity"]}
                     for r in paper_db.fetch_sleeve_equity_history(conn, s["name"])]
        for s in sleeves_rows
    }

    run_log = paper_db.fetch_run_log(conn)
    if reconciliation is None:
        reconciliation = paper_db.reconcile(conn, cfg)

    costs_total_row = conn.execute(
        "SELECT COALESCE(SUM(transaction_cost), 0) c, COALESCE(SUM(notional), 0) t FROM fills"
    ).fetchone()

    return {
        "schema_version": cfg["schema_version"],
        "created_at": cfg["created_at"],
        "inception_date": cfg["inception_date"],
        "last_processed_date": last_processed,
        "data_cutoff_date": last_processed,
        "num_runs": paper_db.fetch_num_runs(conn),
        "starting_capital": cfg["starting_capital"],
        "total_equity": total_equity,
        "cumulative_return": cumulative_return,
        "transaction_costs_total": costs_total_row["c"],
        "turnover_total": costs_total_row["t"],
        "sleeves": sleeves_out,
        "pending_orders": pending_orders,
        "completed_trades": fills,
        "rejected_stale_orders": stale_orders,
        "realized_trades": realized,
        "portfolio_equity_history": portfolio_equity_history,
        "sleeve_equity_history": sleeve_equity_history,
        "run_log": run_log,
        "reconciliation": reconciliation,
        "git_commit_sha": cfg["git_commit_sha"],
        "config_fingerprint": cfg["config_fingerprint"],
    }


def _write_json_views(state_dir, cfg: dict, st: dict) -> None:
    _atomic_write_json(config_path(state_dir), cfg)
    _atomic_write_json(state_path(state_dir), st)


def load_account(state_dir) -> tuple[dict, dict]:
    """Load and validate (config, state). Raises PaperStateError -- failing
    closed -- if the ledger database is missing, unreadable, schema-
    mismatched, or missing required keys. Never overwrites on failure."""
    state_dir = Path(state_dir)
    if not account_exists(state_dir):
        raise PaperStateError(
            f"No paper account found in {state_dir} (expected {paper_db.DB_FILENAME}). "
            f"Run --paper-init first."
        )
    try:
        conn = paper_db.connect(state_dir)
    except (sqlite3.Error, paper_db.LedgerError) as e:
        raise PaperStateError(f"Paper ledger database is corrupt/unreadable ({e}). Not overwriting.") from e
    try:
        cfg = build_config_view(conn)
        st = build_state_view(conn, cfg)
    except sqlite3.Error as e:
        raise PaperStateError(f"Paper ledger database is corrupt/inconsistent ({e}). Not overwriting.") from e
    finally:
        conn.close()
    return cfg, st


# --- Per-session incremental processing ----------------------------------------------

def _process_session(
    conn: sqlite3.Connection, cfg: dict, session: pd.Timestamp, next_session: pd.Timestamp,
    full_calendar: pd.DatetimeIndex, refresh_cache: bool,
) -> dict:
    """Process exactly one finalized session, incrementally, for every
    sleeve. Must be called inside an already-open write transaction; raises
    (never partially commits) on any problem. Returns the run summary dict
    (requirement 11's full field list)."""
    registry = tournament_module.STRATEGY_REGISTRY
    weights = [(name, float(w)) for name, w in cfg["weights"]]
    universe_tickers = cfg["universe_tickers"] or None
    cost_bps = cfg["cost_bps"]
    fractional_shares = cfg["fractional_shares"]
    inception = pd.Timestamp(cfg["inception_date"])

    sleeve_summaries: list[dict] = []
    warnings: list[str] = []
    total_new_signals = total_pending_created = total_fills = total_stale = 0
    total_costs_run = 0.0
    sleeve_equities: dict[str, float] = {}
    # True only for the very first session this account ever processes. A
    # strategy's `initial_events()` (e.g. sector rotation's/momentum's last-
    # month-end-before-inception warmup rebalance) decides using data dated
    # BEFORE inception and is assigned fill_date == the walk's first day by
    # the engine -- i.e. it fills on THIS session, not a later one. Since the
    # signal-generation replay's accounting is normally entirely discarded
    # (only brand-new `pending_orders` are harvested), that day-one fill would
    # otherwise be silently lost. On session 1 only, the replay window is
    # exactly one day (effective_start == session), so EVERY transaction the
    # replay executes necessarily fills TODAY -- safe to harvest directly.
    is_first_session = paper_db.fetch_last_processed_session(conn) is None

    for name, _weight in weights:
        spec = registry[name]
        sleeve_row = paper_db.fetch_sleeve(conn, name)
        if sleeve_row is None:
            raise PaperStateError(f"Sleeve {name!r} from account config has no `sleeves` row; inconsistent ledger.")

        strategy = spec.factory(universe=universe_tickers) if spec.uses_stock_universe else spec.factory()
        if spec.data_plan == "stock":
            prepared = tournament_module.prepare_stock_plan(spec, strategy, inception, session, refresh_cache)
        elif spec.data_plan == "sector":
            prepared = tournament_module.prepare_sector_plan(strategy, inception, session, refresh_cache)
        else:
            raise PaperError(f"Unknown data plan {spec.data_plan!r} for {name!r}")
        data_retrieved_at = datetime.now(timezone.utc).isoformat()

        def close_lookup(ticker, d, _prepared=prepared):
            df = _prepared.clean_price_data.get(ticker)
            if df is None or d not in df.index:
                return float("nan")
            v = df.loc[d, "Close"]
            return float(v) if pd.notna(v) else float("nan")

        # --- Load current sleeve state (positions/cash) into engine-shaped dicts ---
        cash = float(sleeve_row["cash"])
        shares: dict[str, float] = {}
        avg_cost: dict[str, float] = {}
        avg_cost_incl_fees: dict[str, float] = {}
        lot_entry_date: dict[str, pd.Timestamp] = {}
        prior_last_price: dict[str, float] = {}
        prior_last_price_date: dict[str, str] = {}
        for p in paper_db.fetch_positions(conn, name):
            shares[p["ticker"]] = p["shares"]
            avg_cost[p["ticker"]] = p["avg_cost"]
            avg_cost_incl_fees[p["ticker"]] = p["avg_cost_incl_fees"]
            lot_entry_date[p["ticker"]] = pd.Timestamp(p["entry_date"])
            prior_last_price[p["ticker"]] = p["last_price"]
            prior_last_price_date[p["ticker"]] = p["last_price_date"]

        # --- Step 1: resolve previously-persisted pending orders scheduled for TODAY ---
        pending_rows = paper_db.fetch_pending_orders_for_sleeve_session(conn, name, session)
        events_today: list[TargetEvent] = []
        order_by_ticker: dict[str, sqlite3.Row] = {}
        stale_this_session = 0
        for row in pending_rows:
            ticker = row["ticker"]
            price = close_lookup(ticker, session)
            if pd.isna(price):
                paper_db.transition_order_status(
                    conn, row["id"], session, "pending", "stale",
                    reason=f"no finalized price for {ticker} at scheduled fill date {session.date()}",
                )
                stale_this_session += 1
                continue
            e = TargetEvent(
                strategy=name, ticker=ticker, signal_date=pd.Timestamp(row["sig_signal_date"]),
                fill_date=session, target_weight=(row["sig_target_weight"] or 0.0),
                sizing_price=row["sig_sizing_price"] or price, reason=row["sig_reason"],
            )
            e.requested_notional = row["requested_notional"]
            events_today.append(e)
            order_by_ticker[ticker] = row

        trades_out: list = []
        todays_transactions, cash = engine_module._execute_events(
            events_today, session, close_lookup, cash, shares, avg_cost, avg_cost_incl_fees,
            lot_entry_date, cost_bps, fractional_shares, trades_out, full_calendar,
        )

        fills_by_ticker = {tx.ticker: tx for tx in todays_transactions}
        trades_by_ticker: dict[str, list] = {}
        for tr in trades_out:
            trades_by_ticker.setdefault(tr.ticker, []).append(tr)

        session_costs = 0.0
        session_fills = 0
        for ticker, order_row in order_by_ticker.items():
            tx = fills_by_ticker.get(ticker)
            if tx is None:
                # Below the minimum trade notional: a no-op fill (order still
                # resolves rather than being left pending forever).
                fill_price = close_lookup(ticker, session)
                fill_id = paper_db.insert_fill(
                    conn, order_id=order_row["id"], sleeve=name, ticker=ticker, fill_date=session,
                    side=("sell" if order_row["side"] in ("exit", "sell") else "buy"), shares=0.0,
                    fill_price=fill_price, price_semantics=PRICE_SEMANTICS, data_source=DATA_SOURCE,
                    data_retrieved_at=data_retrieved_at, notional=0.0, transaction_cost=0.0,
                    cash_after=cash, shares_after=shares.get(ticker, 0.0),
                    avg_cost_after=avg_cost.get(ticker, 0.0),
                    reason="dust: requested change below the minimum trade notional",
                )
                paper_db.transition_order_status(conn, order_row["id"], session, "pending", "filled",
                                                  reason="dust no-op fill")
                session_fills += 1
                continue

            fill_id = paper_db.insert_fill(
                conn, order_id=order_row["id"], sleeve=name, ticker=ticker, fill_date=session,
                side=tx.action, shares=tx.shares_traded, fill_price=tx.fill_price,
                price_semantics=PRICE_SEMANTICS, data_source=DATA_SOURCE, data_retrieved_at=data_retrieved_at,
                notional=tx.executed_notional, transaction_cost=tx.transaction_cost, cash_after=tx.cash_after,
                shares_after=tx.position_shares_after, avg_cost_after=tx.avg_cost_basis_after, reason=tx.reason,
            )
            paper_db.transition_order_status(conn, order_row["id"], session, "pending", "filled", reason=None)
            session_fills += 1
            session_costs += tx.transaction_cost
            for tr in trades_by_ticker.get(ticker, []):
                paper_db.insert_realized_trade(
                    conn, fill_id=fill_id, sleeve=name, ticker=ticker, event_type=tr.event_type,
                    trade_date=tr.date, shares_sold=tr.shares_sold, sale_price=tr.sale_price,
                    avg_cost_basis=tr.avg_cost_basis, realized_pnl=tr.realized_pnl,
                    realized_return_pct=tr.realized_return_pct, realized_pnl_net=tr.realized_pnl_net,
                    realized_return_pct_net=tr.realized_return_pct_net, holding_days=tr.holding_days,
                    holding_calendar_days=tr.holding_calendar_days, reason=tr.reason,
                )
        total_fills += session_fills
        total_stale += stale_this_session

        # --- Step 1b: replay the strategy from inception through TODAY to get this
        # session's decisions. Only `result.pending_orders` (signal_date == session)
        # is ever treated as authoritative -- everything else about this replay's
        # accounting (equity_curve/positions/trades) is discarded, EXCEPT on the
        # account's very first session (see `is_first_session` above), where
        # `result.transactions` captures a legitimate pre-inception warmup rebalance
        # that the engine assigns to fill TODAY.
        in_window = prepared.full_calendar[prepared.full_calendar <= session]
        if len(in_window) == 0:
            raise PaperError(
                f"{name}: no trading sessions in [{inception.date()}, {session.date()}] to process."
            )
        exec_end = in_window[-1]
        # A sleeve whose effective start postdates this session is not yet
        # active -- e.g. a sector-plan sleeve whose ETF inception + lookback
        # (prepare_sector_plan clips effective_start forward) lands after the
        # account's early sessions. It can generate no signals and holds only
        # its allocated cash so far. Skip the signal-generation replay rather
        # than calling run_backtest with an inverted [effective_start, exec_end]
        # window, which would raise ValueError and (being caught + rolled back)
        # leave the account permanently unable to complete this session.
        if pd.Timestamp(prepared.effective_start) > exec_end:
            result = None
        else:
            sig_cal = in_window.append(pd.DatetimeIndex([pd.Timestamp(next_session)])).unique().sort_values()
            result = run_backtest(
                strategy, prepared.clean_price_data, prepared.full_calendar, prepared.effective_start, exec_end,
                sleeve_row["allocated_capital"], cost_bps, fractional_shares,
                signal_calendar=sig_cal, defer_unfillable=True,
            )

        if is_first_session and result is not None:
            # Harvest the strategy's pre-inception warmup rebalance (its
            # initial_events(), which the engine assigns to fill on the walk's
            # first day == this session). A warmup event whose fill price is
            # unavailable is recorded stale -- never silently filled -- exactly
            # like the regular pending-order path above.
            warmup_events: list[TargetEvent] = []
            warmup_stale: list[TargetEvent] = []
            for tx in result.transactions:
                e = TargetEvent(
                    strategy=name, ticker=tx.ticker, signal_date=tx.signal_date, fill_date=session,
                    target_weight=tx.requested_target_weight, sizing_price=tx.sizing_price, reason=tx.reason,
                )
                e.requested_notional = tx.requested_notional
                (warmup_stale if pd.isna(close_lookup(tx.ticker, session)) else warmup_events).append(e)
            for e in warmup_stale:
                sig_id = paper_db.insert_signal(
                    conn, name, e.ticker, e.signal_date, session, e.target_weight, e.requested_notional,
                    e.sizing_price, sleeve_row["allocated_capital"], e.reason, session,
                )
                order_id = paper_db.insert_order(conn, sig_id, name, e.ticker, "buy", session, e.requested_notional)
                paper_db.transition_order_status(
                    conn, order_id, session, "pending", "stale",
                    reason=f"no finalized price for {e.ticker} at warmup fill date {session.date()}",
                )
                stale_this_session += 1
            total_stale += len(warmup_stale)
            warmup_trades_out: list = []
            warmup_transactions, cash = engine_module._execute_events(
                warmup_events, session, close_lookup, cash, shares, avg_cost, avg_cost_incl_fees,
                lot_entry_date, cost_bps, fractional_shares, warmup_trades_out, full_calendar,
            )
            warmup_fills_by_ticker = {tx.ticker: tx for tx in warmup_transactions}
            for e in warmup_events:
                tx = warmup_fills_by_ticker.get(e.ticker)
                sig_id = paper_db.insert_signal(
                    conn, name, e.ticker, e.signal_date, session, e.target_weight, e.requested_notional,
                    e.sizing_price, sleeve_row["allocated_capital"], e.reason, session,
                )
                order_id = paper_db.insert_order(conn, sig_id, name, e.ticker, "buy" if tx and tx.action == "buy"
                                                  else "sell", session, e.requested_notional)
                if tx is None:
                    fill_price = close_lookup(e.ticker, session)
                    paper_db.insert_fill(
                        conn, order_id=order_id, sleeve=name, ticker=e.ticker, fill_date=session, side="buy",
                        shares=0.0, fill_price=fill_price, price_semantics=PRICE_SEMANTICS,
                        data_source=DATA_SOURCE, data_retrieved_at=data_retrieved_at, notional=0.0,
                        transaction_cost=0.0, cash_after=cash, shares_after=shares.get(e.ticker, 0.0),
                        avg_cost_after=avg_cost.get(e.ticker, 0.0),
                        reason="dust: requested change below the minimum trade notional",
                    )
                else:
                    paper_db.insert_fill(
                        conn, order_id=order_id, sleeve=name, ticker=e.ticker, fill_date=session, side=tx.action,
                        shares=tx.shares_traded, fill_price=tx.fill_price, price_semantics=PRICE_SEMANTICS,
                        data_source=DATA_SOURCE, data_retrieved_at=data_retrieved_at,
                        notional=tx.executed_notional, transaction_cost=tx.transaction_cost,
                        cash_after=tx.cash_after, shares_after=tx.position_shares_after,
                        avg_cost_after=tx.avg_cost_basis_after, reason=tx.reason,
                    )
                    session_costs += tx.transaction_cost
                paper_db.transition_order_status(conn, order_id, session, "pending", "filled",
                                                  reason="initial (pre-inception warmup) rebalance")
                session_fills += 1
            total_fills += len(warmup_events)

        # --- Step 2: persist positions & sleeve cash post-fill (mark ALL holdings) ---
        paper_db.update_sleeve_cash(conn, name, cash)
        touched_tickers = set(shares) | set(prior_last_price)
        mv_by_ticker: dict[str, float] = {}
        for ticker in touched_tickers:
            qty = shares.get(ticker, 0.0)
            if abs(qty) < engine_module.EPSILON_SHARES:
                if ticker in prior_last_price:
                    paper_db.delete_position(conn, name, ticker)
                continue
            price = close_lookup(ticker, session)
            # `last_price_date` records the session the mark's price actually
            # came from; when today's finalized price is missing and we carry a
            # prior mark forward, it stays the ORIGINAL date so the provenance
            # column honestly reports staleness rather than looking fresh.
            price_date = session
            if pd.isna(price):
                price = prior_last_price.get(ticker)
                price_date = prior_last_price_date.get(ticker) or session
                if price is None:
                    raise PaperStateError(
                        f"{name}: held position {ticker} has no price at all as of "
                        f"{session.date()} to mark to market."
                    )
                warnings.append(
                    f"{name}:{ticker}: no finalized price at {session.date()}; carried forward "
                    f"last known price {price} (as of {price_date}) to mark to market."
                )
            paper_db.upsert_position(
                conn, name, ticker, qty, avg_cost.get(ticker, 0.0), avg_cost_incl_fees.get(ticker, 0.0),
                lot_entry_date.get(ticker, session), price, price_date,
            )
            mv_by_ticker[ticker] = qty * price

        sleeve_equity = cash + sum(mv_by_ticker.values())
        sleeve_equities[name] = sleeve_equity
        paper_db.insert_sleeve_equity_row(conn, session, name, cash, sum(mv_by_ticker.values()), sleeve_equity)

        # --- Step 3: generate THIS session's new signals from the replay computed in
        # Step 1b. Only new signals (signal_date == session) from result.pending_orders
        # are ever persisted; result.equity_curve/.positions/.trades (and, apart from
        # the is_first_session harvest above, result.transactions) are discarded --
        # all accounting above came exclusively from the incrementally-maintained ledger. ---
        new_signals_count = pending_created_count = 0
        for e in (result.pending_orders if result is not None else []):
            if pd.Timestamp(e.signal_date) != session:
                continue
            sig_id = paper_db.insert_signal(
                conn, name, e.ticker, session, session, e.target_weight, e.requested_notional,
                e.sizing_price, sleeve_equity, e.reason, e.fill_date,
            )
            new_signals_count += 1
            current_mv = mv_by_ticker.get(e.ticker, 0.0)
            requested = e.requested_notional or 0.0
            side = "exit" if e.target_weight <= 0 else ("buy" if requested >= current_mv else "sell")
            paper_db.insert_order(conn, sig_id, name, e.ticker, side, e.fill_date, e.requested_notional)
            pending_created_count += 1

        total_new_signals += new_signals_count
        total_pending_created += pending_created_count
        total_costs_run += session_costs

        sleeve_summaries.append({
            "sleeve": name, "cash": cash, "equity": sleeve_equity,
            "num_new_signals": new_signals_count, "num_pending_created": pending_created_count,
            "num_fills": session_fills, "num_stale": stale_this_session,
            "transaction_costs_run": session_costs,
        })

    portfolio_equity = sum(sleeve_equities.values())
    prev_eq_row = paper_db.fetch_latest_equity_row(conn)
    starting_capital = cfg["starting_capital"]
    cumulative_return = portfolio_equity / starting_capital - 1.0 if starting_capital else 0.0
    daily_return = 0.0
    if prev_eq_row is not None and prev_eq_row["total_equity"]:
        daily_return = portfolio_equity / prev_eq_row["total_equity"] - 1.0

    paper_db.insert_equity_row(conn, session, portfolio_equity, cumulative_return, daily_return)
    paper_db.insert_processed_session(conn, session, session)

    return {
        "paper_date": str(session.date()),
        "data_cutoff_date": str(session.date()),
        "next_session": str(pd.Timestamp(next_session).date()) if next_session is not None else None,
        "num_new_signals": total_new_signals,
        "num_pending_created": total_pending_created,
        "num_fills": total_fills,
        "num_stale": total_stale,
        "total_equity": portfolio_equity,
        "cumulative_return": cumulative_return,
        "daily_return": daily_return,
        "transaction_costs_run": total_costs_run,
        "sleeves": sleeve_summaries,
        "warnings": warnings,
    }


# --- Advancing the account ------------------------------------------------------------

def advance(state_dir, target_date=None, refresh_cache: bool = False) -> dict:
    """Process finalized sessions forward, ONE AT A TIME, strictly
    incrementally (see module docstring). With `target_date` None, process
    exactly the next unprocessed session (--paper-run). With a `target_date`,
    process every session up to and including the last finalized session on
    or before it (--paper-date). Idempotent: a date already processed is a
    no-op -- and, unlike a full-replay design, re-processing never happens
    even internally, since only already-persisted pending orders and brand-
    new same-session signals are ever touched.

    Refuses to advance (fail closed) if: the runtime strategy/config
    fingerprint has drifted from the account's frozen fingerprint, or the
    persisted ledger already fails reconciliation before any new session is
    processed. Each session is a single SQLite transaction: it commits in
    full only if independent reconciliation passes, otherwise the whole
    session (every sleeve, every table write) is rolled back and prior state
    is left byte-for-byte intact. A recoverable snapshot of the database is
    taken before each session's transaction begins."""
    state_dir = Path(state_dir)
    with _file_lock(state_dir):
        cfg, _ = load_account(state_dir)
        _check_fingerprint(cfg)

        inception = pd.Timestamp(cfg["inception_date"])
        cal = finalized_calendar(inception, refresh_cache=refresh_cache)
        if len(cal) == 0:
            raise PaperError("No finalized trading sessions available to process.")
        latest_finalized = cal[-1]

        conn = paper_db.connect(state_dir)
        try:
            pre_check = paper_db.reconcile(conn, cfg)
            if not pre_check["ok"]:
                raise PaperStateError(
                    "Persisted ledger fails reconciliation BEFORE processing any new session; "
                    "refusing to advance on top of an inconsistent ledger. Run --paper-reconcile "
                    f"for details. Checks: {pre_check['checks']}"
                )

            last_processed_str = paper_db.fetch_last_processed_session(conn)
            last_processed = pd.Timestamp(last_processed_str) if last_processed_str else None

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
                    st = build_state_view(conn, cfg)
                    _write_json_views(state_dir, cfg, st)
                    return {
                        "processed": [],
                        "message": f"Already processed through {last_processed.date()} "
                                   f"(requested {target_ts.date()} resolves to {target_session.date()}).",
                        "config": cfg, "state": st,
                    }
            else:
                target_session = None

            processed_summaries: list[dict] = []
            current_last = last_processed
            while True:
                nxt = _next_session(cal, current_last, inception)
                if nxt is None:
                    break
                if target_session is not None and nxt > target_session:
                    break
                pos = cal.searchsorted(nxt, side="right")
                following = cal[pos] if pos < len(cal) else nyse_calendar.next_nyse_session(nxt)

                paper_db.snapshot_before_session(conn, state_dir, str(nxt.date()))

                conn.execute("BEGIN IMMEDIATE")
                try:
                    summary = _process_session(conn, cfg, nxt, following, cal, refresh_cache)
                    reconciliation = paper_db.reconcile(conn, cfg)
                    if not reconciliation["ok"]:
                        raise PaperStateError(
                            f"Reconciliation FAILED processing {nxt.date()}; refusing to commit. "
                            f"Prior state left intact. Checks: {reconciliation['checks']}"
                        )
                    summary["reconciliation_ok"] = True
                    run_index = paper_db.fetch_num_runs(conn) + 1
                    paper_db.insert_run_log(conn, run_index, nxt, summary)
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise

                processed_summaries.append(summary)
                current_last = nxt
                if target_session is None:
                    break

            st = build_state_view(conn, cfg)
            _write_json_views(state_dir, cfg, st)

            if not processed_summaries:
                msg = (
                    f"Already up to date; latest finalized session {latest_finalized.date()} "
                    f"is already processed." if last_processed is not None
                    else "No sessions to process."
                )
                return {"processed": [], "message": msg, "config": cfg, "state": st}

            return {"processed": processed_summaries, "message": None, "config": cfg, "state": st}
        finally:
            conn.close()


# --- Reset -------------------------------------------------------------------------

def reset_account(state_dir, confirm: bool) -> Path:
    """Reset the account, but only with explicit confirmation. Backs up the
    live database (and the derived JSON views) first -- never deletes state
    silently -- then removes the live files so a fresh --paper-init can
    proceed."""
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
        db = paper_db.db_path(state_dir)
        if db.exists():
            shutil.copy2(db, backup_dir / paper_db.DB_FILENAME)
        for fname in (CONFIG_FILENAME, STATE_FILENAME):
            src = state_dir / fname
            if src.exists():
                shutil.copy2(src, backup_dir / fname)
        # Remove the live files (backed up above) so re-init is clean.
        if db.exists():
            db.unlink()
        for fname in (CONFIG_FILENAME, STATE_FILENAME):
            src = state_dir / fname
            if src.exists():
                src.unlink()
    return backup_dir


# --- Reconcile-on-demand -------------------------------------------------------------

def reconcile_saved_state(state_dir) -> dict:
    """Independently re-verify the persisted ledger's invariants (see
    `paper_db.reconcile`) -- rebuilt from the immutable signals/orders/fills/
    realized_trades tables, never from the materialized sleeves/positions
    tables, so a hand-tampered or drifted database is caught."""
    conn = paper_db.connect(state_dir)
    try:
        cfg = build_config_view(conn)
        # Reconcile ONCE and reuse the result for the returned view, rather than
        # loading the account (which reconciles) and reconciling again.
        result = paper_db.reconcile(conn, cfg)
        st = build_state_view(conn, cfg, reconciliation=result)
    finally:
        conn.close()
    return {"ok": result["ok"], "checks": result["checks"], "config": cfg, "state": st}
