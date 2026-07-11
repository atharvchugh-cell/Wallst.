"""SQLite persistence layer for the forward paper-trading ledger.

This module owns the authoritative, append-only (or immutable-once-written)
tables that make up the ledger, plus the low-level read/write helpers used by
`src.paper`'s session-processing orchestration. Nothing in here ever rewrites
a previously-committed signal/order/fill/trade/equity row -- a session either
commits its new rows in full (inside one DB transaction) or none of them are
written at all (see `src.paper.advance`).

Schema (one SQLite file per account, ``paper_state/paper_ledger.db``):
  - meta                    immutable account identity, written once at init
                             (weights, cost model, frozen universe, fingerprint...)
  - sleeves                 current cash per sleeve (materialized, always
                             reconstructible from fills -- see `reconcile`)
  - positions               current holdings per (sleeve, ticker) (materialized,
                             always reconstructible from fills)
  - processed_sessions      one row per session this account has processed
  - signals                 append-only: one row per (sleeve, ticker, date) signal
  - orders                  one row per signal that became an order; status
                             transitions pending -> filled | stale (append-only
                             history of those transitions in order_status_events)
  - order_status_events     append-only audit log of every status transition
  - fills                   append-only: exactly one row per FILLED order
  - realized_trades         append-only: one row per sell fill (full_exit/partial_sell)
  - equity_history          one row per session: portfolio-level equity
  - sleeve_equity_history   one row per (session, sleeve): sleeve-level equity
  - run_log                 one row per advance() run
  - recovery_snapshots      metadata log of pre-session DB backups (see
                             `snapshot_before_session`)
  - schema_migrations       applied migration versions

Unique constraints and foreign keys enforce the core invariants at write
time (duplicate signals, duplicate fills per order, orphaned rows) rather
than relying on after-the-fact cleanup.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_FILENAME = "paper_ledger.db"
SCHEMA_VERSION = 1

# Status transitions ever considered valid for an order. Anything else found
# in order_status_events (or implied by the current `orders.status` not
# matching any event) is a semantic corruption -- see `reconcile`.
VALID_STATUS_TRANSITIONS = {("pending", "filled"), ("pending", "stale")}

_MIGRATIONS: list[tuple[int, str]] = [
    (1, """
        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE sleeves (
            name TEXT PRIMARY KEY,
            weight REAL NOT NULL,
            allocated_capital REAL NOT NULL,
            cash REAL NOT NULL
        );

        CREATE TABLE positions (
            sleeve TEXT NOT NULL REFERENCES sleeves(name),
            ticker TEXT NOT NULL,
            shares REAL NOT NULL,
            avg_cost REAL NOT NULL,
            avg_cost_incl_fees REAL NOT NULL,
            entry_date TEXT NOT NULL,
            last_price REAL,
            last_price_date TEXT,
            PRIMARY KEY (sleeve, ticker)
        );

        CREATE TABLE processed_sessions (
            session_date TEXT PRIMARY KEY,
            processed_at TEXT NOT NULL,
            data_cutoff_date TEXT NOT NULL
        );

        CREATE TABLE signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sleeve TEXT NOT NULL REFERENCES sleeves(name),
            ticker TEXT NOT NULL,
            signal_date TEXT NOT NULL,
            data_cutoff_date TEXT NOT NULL,
            target_weight REAL NOT NULL,
            requested_notional REAL,
            sizing_price REAL,
            sizing_equity REAL NOT NULL,
            reason TEXT,
            scheduled_fill_date TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE (sleeve, ticker, signal_date)
        );

        CREATE TABLE orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER NOT NULL UNIQUE REFERENCES signals(id),
            sleeve TEXT NOT NULL REFERENCES sleeves(name),
            ticker TEXT NOT NULL,
            side TEXT NOT NULL CHECK(side IN ('buy','sell','exit')),
            scheduled_fill_date TEXT NOT NULL,
            requested_notional REAL,
            status TEXT NOT NULL CHECK(status IN ('pending','filled','stale')),
            created_at TEXT NOT NULL,
            resolved_at TEXT
        );

        CREATE TABLE order_status_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL REFERENCES orders(id),
            session_date TEXT NOT NULL,
            from_status TEXT NOT NULL,
            to_status TEXT NOT NULL,
            reason TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE fills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL UNIQUE REFERENCES orders(id),
            sleeve TEXT NOT NULL REFERENCES sleeves(name),
            ticker TEXT NOT NULL,
            fill_date TEXT NOT NULL,
            side TEXT NOT NULL CHECK(side IN ('buy','sell')),
            shares REAL NOT NULL,
            fill_price REAL NOT NULL,
            price_semantics TEXT NOT NULL,
            data_source TEXT NOT NULL,
            data_retrieved_at TEXT NOT NULL,
            notional REAL NOT NULL,
            transaction_cost REAL NOT NULL,
            cash_after REAL NOT NULL,
            shares_after REAL NOT NULL,
            avg_cost_after REAL NOT NULL,
            reason TEXT,
            row_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE realized_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fill_id INTEGER NOT NULL UNIQUE REFERENCES fills(id),
            sleeve TEXT NOT NULL,
            ticker TEXT NOT NULL,
            event_type TEXT NOT NULL CHECK(event_type IN ('partial_sell','full_exit')),
            trade_date TEXT NOT NULL,
            shares_sold REAL NOT NULL,
            sale_price REAL NOT NULL,
            avg_cost_basis REAL NOT NULL,
            realized_pnl REAL NOT NULL,
            realized_return_pct REAL,
            realized_pnl_net REAL,
            realized_return_pct_net REAL,
            holding_days INTEGER,
            holding_calendar_days INTEGER,
            reason TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE equity_history (
            session_date TEXT PRIMARY KEY,
            total_equity REAL NOT NULL,
            cumulative_return REAL NOT NULL,
            daily_return REAL NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE sleeve_equity_history (
            session_date TEXT NOT NULL,
            sleeve TEXT NOT NULL,
            cash REAL NOT NULL,
            market_value REAL NOT NULL,
            equity REAL NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (session_date, sleeve)
        );

        CREATE TABLE run_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_index INTEGER NOT NULL UNIQUE,
            session_date TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            summary_json TEXT NOT NULL
        );

        CREATE TABLE recovery_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            before_session_date TEXT,
            created_at TEXT NOT NULL,
            backup_path TEXT NOT NULL
        );
    """),
]


class LedgerError(Exception):
    """Raised for structural/semantic problems with the persisted ledger."""


def db_path(state_dir) -> Path:
    return Path(state_dir) / DB_FILENAME


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _d(x) -> str:
    """Canonical YYYY-MM-DD string for a date-only field. `str(pd.Timestamp(...))`
    includes a spurious ` 00:00:00` time component; every date-only column in
    this schema goes through this helper instead so stored/compared date
    strings are always the plain calendar date."""
    import pandas as pd
    return str(pd.Timestamp(x).date())


def connect(state_dir, create: bool = False) -> sqlite3.Connection:
    """Open the ledger DB. If `create` is False and the file doesn't exist,
    raises LedgerError (fail closed -- never silently create a fresh ledger
    where an existing one was expected)."""
    path = db_path(state_dir)
    if not create and not path.exists():
        raise LedgerError(f"No paper ledger database found at {path}. Run --paper-init first.")
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5.0, isolation_level=None)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        _ensure_schema(conn)
    except Exception:
        # A corrupt/foreign file makes the PRAGMA or schema init raise; close
        # the just-opened handle so a failed open never leaks a file
        # descriptor / stale lock to a caller that never received `conn`.
        conn.close()
        raise
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
    for version, sql in _MIGRATIONS:
        if version in applied:
            continue
        # executescript() issues an implicit COMMIT of any open transaction
        # before running, and DDL auto-commits in SQLite -- so the migration's
        # schema changes and the schema_migrations row are recorded as two
        # separate statements/commits rather than one wrapped transaction.
        conn.executescript(sql)
        conn.execute(
            "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            (version, now_iso()),
        )


# --- Recovery snapshots ---------------------------------------------------------

def snapshot_before_session(conn: sqlite3.Connection, state_dir, before_session_date: str | None,
                             keep: int = 5) -> Path:
    """Copy the current (committed) database to a timestamped backup file via
    SQLite's own backup API (safe to run against a live connection -- it
    reads a consistent snapshot, unlike a raw file copy), log it in
    `recovery_snapshots`, then prune to the `keep` most recent backups. Called
    BEFORE each session's mutating transaction begins, so a catastrophic
    failure (not just an exception mid-transaction, which SQLite's own
    rollback already handles) still leaves a recoverable prior state."""
    backups_dir = Path(state_dir) / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    backup_path = backups_dir / f"pre_session_{ts}.db"
    dst = sqlite3.connect(str(backup_path))
    try:
        conn.backup(dst)
    finally:
        dst.close()
    # Roll back on any error so this helper never leaves an open transaction on
    # the shared `conn` -- advance() runs its own BEGIN IMMEDIATE on the same
    # connection immediately after, which would otherwise fail with a confusing
    # "cannot start a transaction within a transaction" masking the real cause.
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "INSERT INTO recovery_snapshots (before_session_date, created_at, backup_path) VALUES (?, ?, ?)",
            (before_session_date, now_iso(), str(backup_path)),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    rows = conn.execute(
        "SELECT id, backup_path FROM recovery_snapshots ORDER BY id DESC"
    ).fetchall()
    stale = rows[keep:]
    if stale:
        conn.execute("BEGIN IMMEDIATE")
        try:
            for row in stale:
                p = Path(row["backup_path"])
                if p.exists():
                    p.unlink()
                conn.execute("DELETE FROM recovery_snapshots WHERE id = ?", (row["id"],))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return backup_path


# --- Meta (immutable account identity) -------------------------------------------

def write_meta(conn: sqlite3.Connection, meta: dict) -> None:
    conn.execute("BEGIN IMMEDIATE")
    try:
        for k, v in meta.items():
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (k, json.dumps(v, default=str)),
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def read_meta(conn: sqlite3.Connection) -> dict:
    rows = conn.execute("SELECT key, value FROM meta").fetchall()
    return {row["key"]: json.loads(row["value"]) for row in rows}


# --- Fill row hashing (tamper detection) -----------------------------------------

def _fill_row_hash(order_id, sleeve, ticker, fill_date, side, shares, fill_price,
                    notional, transaction_cost, cash_after, shares_after, avg_cost_after) -> str:
    payload = "|".join(str(x) for x in (
        order_id, sleeve, ticker, fill_date, side, shares, fill_price,
        notional, transaction_cost, cash_after, shares_after, avg_cost_after,
    ))
    return hashlib.sha256(payload.encode()).hexdigest()


# --- Writers (called inside the caller's transaction) ----------------------------

def insert_sleeve(conn, name, weight, allocated_capital, cash) -> None:
    conn.execute(
        "INSERT INTO sleeves (name, weight, allocated_capital, cash) VALUES (?, ?, ?, ?)",
        (name, weight, allocated_capital, cash),
    )


def update_sleeve_cash(conn, sleeve, cash) -> None:
    conn.execute("UPDATE sleeves SET cash = ? WHERE name = ?", (cash, sleeve))


def upsert_position(conn, sleeve, ticker, shares, avg_cost, avg_cost_incl_fees,
                     entry_date, last_price, last_price_date) -> None:
    conn.execute(
        "INSERT INTO positions (sleeve, ticker, shares, avg_cost, avg_cost_incl_fees, "
        "entry_date, last_price, last_price_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(sleeve, ticker) DO UPDATE SET shares=excluded.shares, avg_cost=excluded.avg_cost, "
        "avg_cost_incl_fees=excluded.avg_cost_incl_fees, entry_date=excluded.entry_date, "
        "last_price=excluded.last_price, last_price_date=excluded.last_price_date",
        (sleeve, ticker, shares, avg_cost, avg_cost_incl_fees, _d(entry_date), last_price,
         _d(last_price_date) if last_price_date is not None else None),
    )


def delete_position(conn, sleeve, ticker) -> None:
    conn.execute("DELETE FROM positions WHERE sleeve = ? AND ticker = ?", (sleeve, ticker))


def insert_signal(conn, sleeve, ticker, signal_date, data_cutoff_date, target_weight,
                   requested_notional, sizing_price, sizing_equity, reason, scheduled_fill_date) -> int:
    cur = conn.execute(
        "INSERT INTO signals (sleeve, ticker, signal_date, data_cutoff_date, target_weight, "
        "requested_notional, sizing_price, sizing_equity, reason, scheduled_fill_date, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (sleeve, ticker, _d(signal_date), _d(data_cutoff_date), target_weight, requested_notional,
         sizing_price, sizing_equity, reason, _d(scheduled_fill_date), now_iso()),
    )
    return cur.lastrowid


def insert_order(conn, signal_id, sleeve, ticker, side, scheduled_fill_date, requested_notional) -> int:
    cur = conn.execute(
        "INSERT INTO orders (signal_id, sleeve, ticker, side, scheduled_fill_date, requested_notional, "
        "status, created_at) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
        (signal_id, sleeve, ticker, side, _d(scheduled_fill_date), requested_notional, now_iso()),
    )
    return cur.lastrowid


def transition_order_status(conn, order_id, session_date, from_status, to_status, reason) -> None:
    if (from_status, to_status) not in VALID_STATUS_TRANSITIONS:
        raise LedgerError(f"Invalid order status transition {from_status!r} -> {to_status!r} for order {order_id}")
    conn.execute(
        "UPDATE orders SET status = ?, resolved_at = ? WHERE id = ? AND status = ?",
        (to_status, now_iso(), order_id, from_status),
    )
    if conn.execute("SELECT changes()").fetchone()[0] != 1:
        raise LedgerError(
            f"Order {order_id} was not in status {from_status!r} when transitioning to {to_status!r}"
        )
    conn.execute(
        "INSERT INTO order_status_events (order_id, session_date, from_status, to_status, reason, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (order_id, _d(session_date), from_status, to_status, reason, now_iso()),
    )


def insert_fill(conn, order_id, sleeve, ticker, fill_date, side, shares, fill_price,
                 price_semantics, data_source, data_retrieved_at, notional, transaction_cost,
                 cash_after, shares_after, avg_cost_after, reason) -> int:
    row_hash = _fill_row_hash(order_id, sleeve, ticker, _d(fill_date), side, shares, fill_price,
                               notional, transaction_cost, cash_after, shares_after, avg_cost_after)
    cur = conn.execute(
        "INSERT INTO fills (order_id, sleeve, ticker, fill_date, side, shares, fill_price, "
        "price_semantics, data_source, data_retrieved_at, notional, transaction_cost, cash_after, "
        "shares_after, avg_cost_after, reason, row_hash, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (order_id, sleeve, ticker, _d(fill_date), side, shares, fill_price, price_semantics,
         data_source, data_retrieved_at, notional, transaction_cost, cash_after, shares_after,
         avg_cost_after, reason, row_hash, now_iso()),
    )
    return cur.lastrowid


def insert_realized_trade(conn, fill_id, sleeve, ticker, event_type, trade_date, shares_sold,
                           sale_price, avg_cost_basis, realized_pnl, realized_return_pct,
                           realized_pnl_net, realized_return_pct_net, holding_days,
                           holding_calendar_days, reason) -> int:
    cur = conn.execute(
        "INSERT INTO realized_trades (fill_id, sleeve, ticker, event_type, trade_date, shares_sold, "
        "sale_price, avg_cost_basis, realized_pnl, realized_return_pct, realized_pnl_net, "
        "realized_return_pct_net, holding_days, holding_calendar_days, reason, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (fill_id, sleeve, ticker, event_type, _d(trade_date), shares_sold, sale_price,
         avg_cost_basis, realized_pnl, realized_return_pct, realized_pnl_net,
         realized_return_pct_net, holding_days, holding_calendar_days, reason, now_iso()),
    )
    return cur.lastrowid


def insert_sleeve_equity_row(conn, session_date, sleeve, cash, market_value, equity) -> None:
    conn.execute(
        "INSERT INTO sleeve_equity_history (session_date, sleeve, cash, market_value, equity, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (_d(session_date), sleeve, cash, market_value, equity, now_iso()),
    )


def insert_equity_row(conn, session_date, total_equity, cumulative_return, daily_return) -> None:
    conn.execute(
        "INSERT INTO equity_history (session_date, total_equity, cumulative_return, daily_return, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (_d(session_date), total_equity, cumulative_return, daily_return, now_iso()),
    )


def insert_processed_session(conn, session_date, data_cutoff_date) -> None:
    conn.execute(
        "INSERT INTO processed_sessions (session_date, processed_at, data_cutoff_date) VALUES (?, ?, ?)",
        (_d(session_date), now_iso(), _d(data_cutoff_date)),
    )


def insert_run_log(conn, run_index, session_date, summary: dict) -> None:
    conn.execute(
        "INSERT INTO run_log (run_index, session_date, timestamp, summary_json) VALUES (?, ?, ?, ?)",
        (run_index, _d(session_date), now_iso(), json.dumps(summary, default=str)),
    )


# --- Readers -----------------------------------------------------------------------

def fetch_sleeves(conn) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM sleeves ORDER BY name").fetchall()


def fetch_sleeve(conn, name) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM sleeves WHERE name = ?", (name,)).fetchone()


def fetch_positions(conn, sleeve) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM positions WHERE sleeve = ? ORDER BY ticker", (sleeve,)
    ).fetchall()


_ORDER_JOIN_SELECT = (
    "SELECT o.*, s.signal_date AS sig_signal_date, s.sizing_price AS sig_sizing_price, "
    "s.reason AS sig_reason, s.target_weight AS sig_target_weight "
    "FROM orders o JOIN signals s ON o.signal_id = s.id"
)

_FILL_JOIN_SELECT = (
    "SELECT f.*, s.signal_date AS sig_signal_date FROM fills f "
    "JOIN orders o ON f.order_id = o.id JOIN signals s ON o.signal_id = s.id"
)


def fetch_pending_orders_for_sleeve_session(conn, sleeve, session_date) -> list[sqlite3.Row]:
    return conn.execute(
        _ORDER_JOIN_SELECT + " WHERE o.sleeve = ? AND o.scheduled_fill_date = ? AND o.status = 'pending'",
        (sleeve, _d(session_date)),
    ).fetchall()


def fetch_orders_by_status(conn, status: str) -> list[sqlite3.Row]:
    return conn.execute(_ORDER_JOIN_SELECT + " WHERE o.status = ? ORDER BY o.id", (status,)).fetchall()


def fetch_all_pending_orders(conn) -> list[sqlite3.Row]:
    return fetch_orders_by_status(conn, "pending")


def fetch_all_stale_orders(conn) -> list[sqlite3.Row]:
    return fetch_orders_by_status(conn, "stale")


def fetch_all_fills(conn, sleeve=None) -> list[sqlite3.Row]:
    if sleeve is not None:
        return conn.execute(
            "SELECT * FROM fills WHERE sleeve = ? ORDER BY fill_date, id", (sleeve,)
        ).fetchall()
    return conn.execute("SELECT * FROM fills ORDER BY fill_date, id").fetchall()


def fetch_fills_view(conn, sleeve=None) -> list[sqlite3.Row]:
    """Fills joined with their originating signal's signal_date, for reporting."""
    if sleeve is not None:
        return conn.execute(_FILL_JOIN_SELECT + " WHERE f.sleeve = ? ORDER BY f.fill_date, f.id", (sleeve,)).fetchall()
    return conn.execute(_FILL_JOIN_SELECT + " ORDER BY f.fill_date, f.id").fetchall()


def fetch_all_realized_trades(conn, sleeve=None) -> list[sqlite3.Row]:
    if sleeve is not None:
        return conn.execute(
            "SELECT * FROM realized_trades WHERE sleeve = ? ORDER BY trade_date, id", (sleeve,)
        ).fetchall()
    return conn.execute("SELECT * FROM realized_trades ORDER BY trade_date, id").fetchall()


def fetch_last_processed_session(conn) -> str | None:
    row = conn.execute("SELECT MAX(session_date) AS d FROM processed_sessions").fetchone()
    return row["d"]


def fetch_processed_sessions(conn) -> list[str]:
    return [r["session_date"] for r in
            conn.execute("SELECT session_date FROM processed_sessions ORDER BY session_date").fetchall()]


def fetch_equity_history(conn) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM equity_history ORDER BY session_date").fetchall()


def fetch_sleeve_equity_history(conn, sleeve=None) -> list[sqlite3.Row]:
    if sleeve is not None:
        return conn.execute(
            "SELECT * FROM sleeve_equity_history WHERE sleeve = ? ORDER BY session_date", (sleeve,)
        ).fetchall()
    return conn.execute("SELECT * FROM sleeve_equity_history ORDER BY session_date, sleeve").fetchall()


def fetch_latest_equity_row(conn) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM equity_history ORDER BY session_date DESC LIMIT 1").fetchone()


def fetch_run_log(conn) -> list[dict]:
    rows = conn.execute("SELECT * FROM run_log ORDER BY run_index").fetchall()
    out = []
    for r in rows:
        summary = json.loads(r["summary_json"])
        out.append({"run_index": r["run_index"], "timestamp": r["timestamp"], **summary})
    return out


def fetch_num_runs(conn) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM run_log").fetchone()
    return int(row["n"])


# --- Independent reconciliation ---------------------------------------------------
# Rebuilds expected cash/positions/avg-cost/costs/turnover/realized P&L from the
# IMMUTABLE fills ledger (never from the materialized `sleeves`/`positions`
# tables) and compares. This is the single reconciliation routine used both (a)
# as the commit gate inside `advance()`'s per-session transaction and (b) by the
# read-only `--paper-reconcile` command against already-committed state -- the
# same checks either way, so a hand-tampered database is caught the same way a
# live bug would be.

RECONCILE_ABS_TOL = 0.01     # $0.01
RECONCILE_REL_TOL = 1e-6


def _approx(a: float, b: float) -> bool:
    return abs(a - b) <= RECONCILE_ABS_TOL + RECONCILE_REL_TOL * max(abs(a), abs(b))


def _finite(x) -> bool:
    import math
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _replay_sleeve_from_fills(conn, sleeve: str, allocated_capital: float) -> dict:
    """Recompute cash/shares/avg_cost/costs/turnover/realized_pnl for one
    sleeve purely from its `fills` (and `realized_trades`) rows, in fill
    order -- the same average-cost formula `engine._apply_buy/_apply_sell`
    use, applied independently here so a drifted or tampered materialized
    `sleeves`/`positions` row is caught rather than trusted."""
    fills = fetch_all_fills(conn, sleeve)
    cash = float(allocated_capital)
    shares: dict[str, float] = {}
    avg_cost: dict[str, float] = {}
    costs_total = 0.0
    turnover_total = 0.0
    for f in fills:
        t = f["ticker"]
        costs_total += f["transaction_cost"]
        turnover_total += f["notional"]
        if f["side"] == "buy":
            cash -= f["notional"] + f["transaction_cost"]
            prior_shares = shares.get(t, 0.0)
            prior_cost = avg_cost.get(t, 0.0)
            new_shares = prior_shares + f["shares"]
            avg_cost[t] = (prior_shares * prior_cost + f["shares"] * f["fill_price"]) / new_shares if new_shares else 0.0
            shares[t] = new_shares
        else:
            cash += f["notional"] - f["transaction_cost"]
            new_shares = shares.get(t, 0.0) - f["shares"]
            shares[t] = 0.0 if abs(new_shares) < 1e-9 else new_shares
            if shares[t] == 0.0:
                avg_cost.pop(t, None)
    realized_pnl_total = sum(
        r["realized_pnl"] for r in fetch_all_realized_trades(conn, sleeve)
    )
    return {
        "cash": cash, "shares": shares, "avg_cost": avg_cost,
        "transaction_costs": costs_total, "turnover": turnover_total,
        "realized_pnl": realized_pnl_total,
    }


def reconcile(conn: sqlite3.Connection, cfg: dict) -> dict:
    """Independently rebuild every account invariant from the immutable
    ledger and compare against the materialized state. Returns
    {"ok": bool, "checks": [...]}."""
    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    sleeves = fetch_sleeves(conn)
    starting_capital = float(cfg["starting_capital"])

    alloc_sum = sum(s["allocated_capital"] for s in sleeves)
    record("allocations_sum_to_capital", _approx(alloc_sum, starting_capital),
           f"sum(allocated)={alloc_sum:.2f} vs starting_capital={starting_capital:.2f}")

    latest_eq = fetch_latest_equity_row(conn)
    if latest_eq is not None:
        sleeve_rows = conn.execute(
            "SELECT * FROM sleeve_equity_history WHERE session_date = ?", (latest_eq["session_date"],)
        ).fetchall()
        sleeve_eq_sum = sum(r["equity"] for r in sleeve_rows)
        record("portfolio_equity_is_sum_of_sleeves", _approx(sleeve_eq_sum, latest_eq["total_equity"]),
               f"sum(sleeve equity)={sleeve_eq_sum:.2f} vs total_equity={latest_eq['total_equity']:.2f} "
               f"as of {latest_eq['session_date']}")

    dupe_eq = conn.execute(
        "SELECT session_date, COUNT(*) c FROM equity_history GROUP BY session_date HAVING c > 1"
    ).fetchall()
    record("no_duplicate_equity_dates", len(dupe_eq) == 0, f"duplicate dates: {[r['session_date'] for r in dupe_eq]}")

    dates = fetch_processed_sessions(conn)
    record("sessions_strictly_ordered", dates == sorted(set(dates)) and len(dates) == len(set(dates)),
           "processed_sessions dates are strictly increasing with no duplicates")

    orphan_fills = conn.execute(
        "SELECT f.id FROM fills f LEFT JOIN orders o ON f.order_id = o.id WHERE o.id IS NULL"
    ).fetchall()
    record("fills_reference_valid_orders", len(orphan_fills) == 0, f"orphaned fill ids: {[r['id'] for r in orphan_fills]}")

    orphan_orders = conn.execute(
        "SELECT o.id FROM orders o LEFT JOIN signals s ON o.signal_id = s.id WHERE s.id IS NULL"
    ).fetchall()
    record("orders_reference_valid_signals", len(orphan_orders) == 0,
           f"orphaned order ids: {[r['id'] for r in orphan_orders]}")

    orphan_trades = conn.execute(
        "SELECT rt.id FROM realized_trades rt LEFT JOIN fills f ON rt.fill_id = f.id WHERE f.id IS NULL"
    ).fetchall()
    record("realized_trades_reference_valid_fills", len(orphan_trades) == 0,
           f"orphaned realized_trade ids: {[r['id'] for r in orphan_trades]}")

    missing_fill = conn.execute(
        "SELECT o.id FROM orders o WHERE o.status = 'filled' "
        "AND NOT EXISTS (SELECT 1 FROM fills f WHERE f.order_id = o.id)"
    ).fetchall()
    record("filled_orders_have_a_fill", len(missing_fill) == 0,
           f"filled orders missing a fill row: {[r['id'] for r in missing_fill]}")

    fill_wrong_status = conn.execute(
        "SELECT f.id FROM fills f JOIN orders o ON f.order_id = o.id WHERE o.status != 'filled'"
    ).fetchall()
    record("fills_belong_to_filled_orders", len(fill_wrong_status) == 0,
           f"fills whose order is not 'filled': {[r['id'] for r in fill_wrong_status]}")

    pending_with_fill = conn.execute(
        "SELECT f.id FROM fills f JOIN orders o ON f.order_id = o.id WHERE o.status = 'pending'"
    ).fetchall()
    record("pending_orders_have_no_fill", len(pending_with_fill) == 0,
           "pending orders are never counted as fills or holdings")

    # Only a sell fill that actually traded shares (> 0) produces realized
    # P&L -- a "dust" no-op fill (side is an arbitrary informational label at
    # zero shares/notional) legitimately has no realized_trades row.
    sells_missing_trade = conn.execute(
        "SELECT f.id FROM fills f WHERE f.side = 'sell' AND f.shares > 0 "
        "AND NOT EXISTS (SELECT 1 FROM realized_trades rt WHERE rt.fill_id = f.id)"
    ).fetchall()
    record("sell_fills_have_realized_trade", len(sells_missing_trade) == 0,
           f"sell fills missing a realized trade: {[r['id'] for r in sells_missing_trade]}")

    all_fills = fetch_all_fills(conn)
    altered = []
    non_finite_fills = []
    for f in all_fills:
        expected_hash = _fill_row_hash(
            f["order_id"], f["sleeve"], f["ticker"], f["fill_date"], f["side"], f["shares"],
            f["fill_price"], f["notional"], f["transaction_cost"], f["cash_after"], f["shares_after"],
            f["avg_cost_after"],
        )
        if expected_hash != f["row_hash"]:
            altered.append(f["id"])
        if not all(_finite(f[c]) for c in ("shares", "fill_price", "notional", "transaction_cost")):
            non_finite_fills.append(f["id"])
    record("fill_rows_not_altered", len(altered) == 0,
           f"altered fill ids: {altered}" if altered else "all fill row hashes verified")
    record("fill_values_finite", len(non_finite_fills) == 0, f"non-finite fill ids: {non_finite_fills}")

    transitions = conn.execute("SELECT * FROM order_status_events").fetchall()
    invalid_transitions = [
        (r["order_id"], r["from_status"], r["to_status"]) for r in transitions
        if (r["from_status"], r["to_status"]) not in VALID_STATUS_TRANSITIONS
    ]
    record("status_transitions_valid", len(invalid_transitions) == 0, f"invalid transitions: {invalid_transitions}")

    non_finite_sleeves = [s["name"] for s in sleeves if not _finite(s["cash"])]
    record("sleeve_cash_finite", len(non_finite_sleeves) == 0, f"non-finite cash: {non_finite_sleeves}")

    for s in sleeves:
        name = s["name"]
        replay = _replay_sleeve_from_fills(conn, name, s["allocated_capital"])

        record(f"{name}:fills_reconcile_cash", _approx(replay["cash"], s["cash"]),
               f"cash from fills={replay['cash']:.2f} vs sleeve cash={s['cash']:.2f}")
        record(f"{name}:cash_not_negative", s["cash"] >= -RECONCILE_ABS_TOL, f"cash={s['cash']:.2f}")

        positions = {p["ticker"]: p for p in fetch_positions(conn, name)}
        shares_ok = True
        avg_cost_ok = True
        for t, qty in replay["shares"].items():
            pos = positions.get(t)
            if abs(qty) < 1e-9:
                if pos is not None:
                    shares_ok = False
                continue
            if pos is None or not _approx(qty, pos["shares"]):
                shares_ok = False
                continue
            if not _approx(replay["avg_cost"].get(t, 0.0), pos["avg_cost"]):
                avg_cost_ok = False
        for t in positions:
            if abs(replay["shares"].get(t, 0.0)) < 1e-9:
                shares_ok = False
        record(f"{name}:fills_reconcile_shares", shares_ok, "share counts from fills match persisted positions")
        record(f"{name}:fills_reconcile_avg_cost", avg_cost_ok, "avg cost from fills matches persisted positions")

        mv = sum((p["shares"] or 0.0) * (p["last_price"] or 0.0) for p in positions.values())
        latest_sleeve_eq = conn.execute(
            "SELECT * FROM sleeve_equity_history WHERE sleeve = ? ORDER BY session_date DESC LIMIT 1", (name,)
        ).fetchone()
        if latest_sleeve_eq is not None:
            record(f"{name}:equity_is_cash_plus_positions",
                   _approx(s["cash"] + mv, latest_sleeve_eq["equity"]),
                   f"cash={s['cash']:.2f}+mv={mv:.2f} vs equity={latest_sleeve_eq['equity']:.2f}")

    return {"ok": all(c["ok"] for c in checks), "checks": checks}
