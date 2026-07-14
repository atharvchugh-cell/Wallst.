"""SQLite-backed execution ledger.

The broker remains authoritative for what actually happened.  This ledger is
the durable local record used to make submissions idempotent, recover after a
crash, derive expected positions from fills, and explain every control action.
"""

from __future__ import annotations

import json
import hmac
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback is process-local only.
    fcntl = None

from .models import (
    ACTIVE_ORDER_STATUSES,
    AccountSnapshot,
    BrokerOrder,
    Fill,
    IntentStatus,
    OrderRequest,
    OrderStatus,
    Position,
    Side,
    TERMINAL_INTENT_STATUSES,
    TargetPositionIntent,
    ZERO,
    as_decimal,
    json_safe,
    utc_now,
)


SCHEMA_VERSION = 3


class LedgerError(RuntimeError):
    pass


class LedgerConflict(LedgerError):
    pass


def _ledger_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate stored JSON key: {key}")
        result[key] = value
    return result


class Ledger:
    def __init__(self, path: str | Path = ":memory:", *, clock=utc_now) -> None:
        self.path = str(Path(path).expanduser()) if str(path) != ":memory:" else ":memory:"
        self.clock = clock
        self._execution_rlock = threading.RLock()
        self._batch_rlock = threading.RLock()
        if self.path != ":memory:":
            Path(self.path).resolve().parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=30.0)
        if self.path != ":memory:":
            os.chmod(self.path, 0o600)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA busy_timeout = 30000")
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute("PRAGMA synchronous = FULL")
        self._create_schema()
        integrity = self.conn.execute("PRAGMA quick_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            self.conn.close()
            raise LedgerError("SQLite quick_check failed; execution ledger is not safe to use")
        if self.conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
            self.conn.close()
            raise LedgerError("SQLite foreign_key_check failed; execution ledger is not safe to use")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Cursor]:
        cur = self.conn.cursor()
        try:
            cur.execute("BEGIN IMMEDIATE")
            yield cur
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cur.close()

    @contextmanager
    def execution_guard(self) -> Iterator[None]:
        """Serialize broker submissions against arm/disarm/kill state changes."""
        with self._execution_rlock:
            if self.path == ":memory:":
                yield
                return
            if fcntl is None:
                raise LedgerError("File-backed execution fencing requires fcntl support")
            lock_path = f"{self.path}.execution.lock"
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            os.chmod(lock_path, 0o600)
            with os.fdopen(fd, "a+") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def batch_execution_guard(self) -> Iterator[None]:
        """Allow only one Phase-3 batch orchestrator per ledger at a time."""
        with self._batch_rlock:
            if self.path == ":memory:":
                yield
                return
            if fcntl is None:
                raise LedgerError("File-backed batch fencing requires fcntl support")
            lock_path = f"{self.path}.batch.lock"
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            os.chmod(lock_path, 0o600)
            with os.fdopen(fd, "a+") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS control_state (
                account_id TEXT PRIMARY KEY,
                armed INTEGER NOT NULL DEFAULT 0,
                kill_switch INTEGER NOT NULL DEFAULT 0,
                reason TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS intents (
                intent_id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                account_id TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                target_quantity TEXT NOT NULL,
                signal_at TEXT NOT NULL,
                target_version TEXT NOT NULL,
                reference_price TEXT NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL,
                status_detail TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                intent_id TEXT NOT NULL UNIQUE REFERENCES intents(intent_id),
                account_id TEXT NOT NULL,
                client_order_id TEXT NOT NULL UNIQUE,
                broker_order_id TEXT UNIQUE,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity TEXT NOT NULL,
                filled_quantity TEXT NOT NULL DEFAULT '0',
                reference_price TEXT NOT NULL,
                order_type TEXT NOT NULL,
                time_in_force TEXT NOT NULL,
                limit_price TEXT,
                status TEXT NOT NULL,
                rejection_reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS fills (
                fill_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL REFERENCES orders(order_id),
                broker_order_id TEXT NOT NULL,
                client_order_id TEXT NOT NULL,
                account_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity TEXT NOT NULL,
                price TEXT NOT NULL,
                commission TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                recorded_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS position_state (
                account_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                quantity TEXT NOT NULL,
                avg_price TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (account_id, symbol)
            );

            CREATE TABLE IF NOT EXISTS account_state (
                account_id TEXT PRIMARY KEY,
                expected_cash TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS reconciliation_runs (
                run_id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                clean INTEGER NOT NULL,
                issue_count INTEGER NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS equity_guardrails (
                account_id TEXT PRIMARY KEY,
                trading_date TEXT NOT NULL,
                day_start_equity TEXT NOT NULL,
                high_water_equity TEXT NOT NULL,
                last_equity TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS execution_batches (
                batch_id TEXT PRIMARY KEY,
                plan_hash TEXT NOT NULL UNIQUE,
                source_hash TEXT NOT NULL,
                account_id TEXT NOT NULL,
                deployment_id TEXT NOT NULL,
                trading_date TEXT NOT NULL,
                signal_at TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                status TEXT NOT NULL,
                approved_at TEXT,
                approved_by TEXT,
                approval_reason TEXT,
                last_error TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_intents_status ON intents(status);
            CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
            CREATE INDEX IF NOT EXISTS idx_fills_occurred ON fills(occurred_at);
            CREATE INDEX IF NOT EXISTS idx_execution_batches_account
              ON execution_batches(account_id, created_at);
            CREATE UNIQUE INDEX IF NOT EXISTS uq_active_order_per_symbol
              ON orders(account_id, symbol)
              WHERE status IN ('pending_submit', 'submitted', 'partially_filled');

            CREATE TRIGGER IF NOT EXISTS audit_events_no_update
              BEFORE UPDATE ON audit_events
              BEGIN
                SELECT RAISE(ABORT, 'audit events are append-only');
              END;

            CREATE TRIGGER IF NOT EXISTS audit_events_no_delete
              BEFORE DELETE ON audit_events
              BEGIN
                SELECT RAISE(ABORT, 'audit events are append-only');
              END;
            """
        )
        batch_columns = {
            row["name"] for row in self.conn.execute("PRAGMA table_info(execution_batches)")
        }
        if "source_hash" not in batch_columns:
            self.conn.execute(
                "ALTER TABLE execution_batches ADD COLUMN source_hash TEXT NOT NULL DEFAULT ''"
            )
            rows = self.conn.execute(
                "SELECT batch_id, plan_json FROM execution_batches"
            ).fetchall()
            for row in rows:
                try:
                    payload = json.loads(row["plan_json"])
                    source_hash = str(payload["source_hash"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise LedgerError(
                        "Existing execution batch cannot be migrated to source-hash protection"
                    ) from exc
                self.conn.execute(
                    "UPDATE execution_batches SET source_hash = ? WHERE batch_id = ?",
                    (source_hash, row["batch_id"]),
                )
        try:
            self.conn.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS uq_execution_batch_source
                   ON execution_batches(account_id, source_hash)"""
            )
        except sqlite3.IntegrityError as exc:
            raise LedgerError(
                "Ledger contains multiple previews for one target source/version"
            ) from exc
        existing = self.conn.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        if existing is not None and int(existing["value"]) not in {1, 2, SCHEMA_VERSION}:
            raise LedgerError(
                f"Unsupported ledger schema {existing['value']}; expected {SCHEMA_VERSION}"
            )
        if existing is None:
            self.conn.execute(
                "INSERT INTO metadata(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        elif int(existing["value"]) in {1, 2}:
            # v2 added expected-cash reconciliation. v3 adds Phase-3 equity
            # guardrails and immutable reviewed execution batches. Migration
            # never invents an equity baseline or an operator approval.
            self.conn.execute(
                "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
                (str(SCHEMA_VERSION),),
            )
        self.conn.commit()

    @staticmethod
    def _dict(row: sqlite3.Row | None) -> dict | None:
        return dict(row) if row is not None else None

    def _audit(
        self,
        cur: sqlite3.Cursor,
        event_type: str,
        entity_type: str,
        entity_id: str,
        payload: dict | None = None,
    ) -> None:
        cur.execute(
            """INSERT INTO audit_events(
                   occurred_at, event_type, entity_type, entity_id, payload_json
               ) VALUES (?, ?, ?, ?, ?)""",
            (
                self.clock().isoformat(),
                event_type,
                entity_type,
                entity_id,
                json.dumps(json_safe(payload or {}), sort_keys=True, separators=(",", ":")),
            ),
        )

    def record_audit(
        self,
        event_type: str,
        entity_type: str,
        entity_id: str,
        payload: dict | None = None,
    ) -> None:
        with self._tx() as cur:
            self._audit(cur, event_type, entity_type, entity_id, payload)

    # --- Safety state ---------------------------------------------------------

    def get_control_state(self, account_id: str) -> dict:
        row = self.conn.execute(
            "SELECT * FROM control_state WHERE account_id = ?", (account_id,)
        ).fetchone()
        if row is None:
            return {
                "account_id": account_id,
                "armed": False,
                "kill_switch": False,
                "reason": "not initialized",
                "updated_at": None,
            }
        result = dict(row)
        result["armed"] = bool(result["armed"])
        result["kill_switch"] = bool(result["kill_switch"])
        return result

    def set_control_state(
        self,
        account_id: str,
        *,
        armed: bool,
        kill_switch: bool,
        reason: str,
    ) -> None:
        account_id = account_id.strip()
        reason = reason.strip()
        if not account_id or not reason:
            raise ValueError("Control changes require an account ID and operator reason")
        now = self.clock().isoformat()
        with self.execution_guard():
            with self._tx() as cur:
                cur.execute(
                    """INSERT INTO control_state(account_id, armed, kill_switch, reason, updated_at)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(account_id) DO UPDATE SET
                         armed=excluded.armed,
                         kill_switch=excluded.kill_switch,
                         reason=excluded.reason,
                         updated_at=excluded.updated_at""",
                    (account_id, int(armed), int(kill_switch), reason, now),
                )
                self._audit(
                    cur,
                    "control_state_changed",
                    "account",
                    account_id,
                    {"armed": armed, "kill_switch": kill_switch, "reason": reason},
                )

    def bound_account_id(self) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM metadata WHERE key = 'bound_account_id'"
        ).fetchone()
        if row is not None:
            return str(row["value"])
        inferred = self.conn.execute(
            "SELECT DISTINCT account_id FROM account_state ORDER BY account_id"
        ).fetchall()
        if len(inferred) > 1:
            raise LedgerConflict("Execution ledger contains more than one account baseline")
        return str(inferred[0]["account_id"]) if inferred else None

    def assert_account_binding(self, account_id: str) -> None:
        bound = self.bound_account_id()
        if bound is not None and bound != account_id:
            raise LedgerConflict(
                f"Ledger is bound to account {bound}, not authenticated account {account_id}"
            )

    def known_account_ids(self) -> tuple[str, ...]:
        rows = self.conn.execute(
            """SELECT account_id FROM control_state
               UNION SELECT account_id FROM account_state
               UNION SELECT account_id FROM intents
               UNION SELECT account_id FROM orders
               ORDER BY account_id"""
        ).fetchall()
        return tuple(str(row["account_id"]) for row in rows)

    # --- Position baseline and expected state --------------------------------

    def positions_bootstrapped(self, account_id: str) -> bool:
        key = f"positions_bootstrapped:{account_id}"
        return self.conn.execute(
            "SELECT 1 FROM metadata WHERE key = ?", (key,)
        ).fetchone() is not None

    def bootstrap_positions(
        self,
        account: AccountSnapshot,
        positions: list[Position],
    ) -> None:
        """Record the one-time opening position and cash baseline.

        A later call is rejected even if the opening account was empty.  This
        prevents reconciliation breaks from being accidentally erased by
        treating current broker state as a fresh baseline.
        """
        account_id = account.account_id
        key = f"positions_bootstrapped:{account_id}"
        with self._tx() as cur:
            bound = cur.execute(
                "SELECT value FROM metadata WHERE key = 'bound_account_id'"
            ).fetchone()
            if bound is None:
                inferred = cur.execute(
                    "SELECT DISTINCT account_id FROM account_state ORDER BY account_id"
                ).fetchall()
                if inferred and any(row["account_id"] != account_id for row in inferred):
                    raise LedgerConflict("Execution ledger already contains another account baseline")
            if bound is not None and bound["value"] != account_id:
                raise LedgerConflict(
                    f"Ledger is bound to account {bound['value']}, not {account_id}"
                )
            if cur.execute("SELECT 1 FROM metadata WHERE key = ?", (key,)).fetchone():
                raise LedgerConflict(f"Position baseline already exists for {account_id}")
            if cur.execute(
                "SELECT 1 FROM fills WHERE account_id = ? LIMIT 1", (account_id,)
            ).fetchone():
                raise LedgerConflict("Cannot bootstrap positions after fills exist")
            if cur.execute(
                "SELECT 1 FROM account_state WHERE account_id = ?", (account_id,)
            ).fetchone():
                raise LedgerConflict(f"Cash baseline already exists for {account_id}")
            now = self.clock().isoformat()
            for p in positions:
                if p.quantity < ZERO:
                    raise LedgerConflict("Short opening positions are unsupported in phase one")
                if p.quantity != p.quantity.to_integral_value():
                    raise LedgerConflict("Fractional opening positions are unsupported")
                if p.quantity == ZERO:
                    continue
                cur.execute(
                    """INSERT INTO position_state(
                           account_id, symbol, quantity, avg_price, updated_at
                       ) VALUES (?, ?, ?, ?, ?)""",
                    (account_id, p.symbol, str(p.quantity), str(p.avg_price), now),
                )
            cur.execute(
                """INSERT INTO account_state(account_id, expected_cash, updated_at)
                   VALUES (?, ?, ?)""",
                (account_id, str(account.cash), now),
            )
            cur.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", (key, now))
            cur.execute(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES('bound_account_id', ?)",
                (account_id,),
            )
            self._audit(
                cur,
                "position_baseline_created",
                "account",
                account_id,
                {"expected_cash": account.cash, "positions": [
                    {"symbol": p.symbol, "quantity": p.quantity, "avg_price": p.avg_price}
                    for p in positions if p.quantity != ZERO
                ]},
            )

    def expected_cash(self, account_id: str) -> Decimal | None:
        row = self.conn.execute(
            "SELECT expected_cash FROM account_state WHERE account_id = ?", (account_id,)
        ).fetchone()
        return Decimal(row["expected_cash"]) if row is not None else None

    def list_positions(self, account_id: str) -> list[Position]:
        rows = self.conn.execute(
            "SELECT * FROM position_state WHERE account_id = ? ORDER BY symbol", (account_id,)
        ).fetchall()
        return [
            Position(r["symbol"], Decimal(r["quantity"]), Decimal(r["avg_price"]), Decimal(r["avg_price"]))
            for r in rows
            if Decimal(r["quantity"]) != ZERO
        ]

    # --- Intents -------------------------------------------------------------

    def create_intent(self, intent: TargetPositionIntent) -> tuple[dict, bool]:
        key = intent.idempotency_key
        intent_id = f"int-{key[:24]}"
        now = self.clock().isoformat()
        with self._tx() as cur:
            existing = cur.execute(
                "SELECT * FROM intents WHERE idempotency_key = ?", (key,)
            ).fetchone()
            if existing is not None:
                compared = {
                    "account_id": intent.account_id,
                    "strategy_id": intent.strategy_id,
                    "symbol": intent.symbol,
                    "target_quantity": str(intent.target_quantity),
                    "signal_at": intent.signal_at.isoformat(),
                    "target_version": intent.target_version,
                    "reference_price": str(intent.reference_price),
                    "reason": intent.reason,
                }
                if any(existing[name] != value for name, value in compared.items()):
                    raise LedgerConflict(
                        "Idempotency key was reused with different target content; increment target_version"
                    )
                return dict(existing), False
            cur.execute(
                """INSERT INTO intents(
                       intent_id, idempotency_key, account_id, strategy_id, symbol,
                       target_quantity, signal_at, target_version, reference_price,
                       reason, status, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    intent_id, key, intent.account_id, intent.strategy_id, intent.symbol,
                    str(intent.target_quantity), intent.signal_at.isoformat(), intent.target_version,
                    str(intent.reference_price), intent.reason, IntentStatus.CREATED.value, now, now,
                ),
            )
            self._audit(
                cur, "intent_created", "intent", intent_id,
                {
                    "idempotency_key": key,
                    "symbol": intent.symbol,
                    "target_quantity": intent.target_quantity,
                    "target_version": intent.target_version,
                },
            )
            row = cur.execute("SELECT * FROM intents WHERE intent_id = ?", (intent_id,)).fetchone()
            return dict(row), True

    def get_intent(self, intent_id: str) -> dict | None:
        return self._dict(self.conn.execute(
            "SELECT * FROM intents WHERE intent_id = ?", (intent_id,)
        ).fetchone())

    def list_intents(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM intents ORDER BY created_at, intent_id"
        ).fetchall()]

    def find_intent(
        self,
        *,
        account_id: str,
        strategy_id: str,
        symbol: str,
        signal_at: str,
        target_version: str,
    ) -> dict | None:
        return self._dict(self.conn.execute(
            """SELECT * FROM intents WHERE account_id = ? AND strategy_id = ?
                 AND symbol = ? AND signal_at = ? AND target_version = ?""",
            (account_id, strategy_id, symbol, signal_at, target_version),
        ).fetchone())

    def set_intent_status(
        self,
        intent_id: str,
        status: IntentStatus,
        detail: str = "",
        *,
        payload: dict | None = None,
    ) -> None:
        with self._tx() as cur:
            current = cur.execute(
                "SELECT status FROM intents WHERE intent_id = ?", (intent_id,)
            ).fetchone()
            if current is None:
                raise LedgerError(f"Unknown intent: {intent_id}")
            current_status = IntentStatus(current["status"])
            if current_status in TERMINAL_INTENT_STATUSES and current_status != status:
                raise LedgerConflict(
                    f"Terminal intent {intent_id} cannot change from {current_status.value} to {status.value}"
                )
            result = cur.execute(
                "UPDATE intents SET status = ?, status_detail = ?, updated_at = ? WHERE intent_id = ?",
                (status.value, detail, self.clock().isoformat(), intent_id),
            )
            if result.rowcount != 1:
                raise LedgerError(f"Unknown intent: {intent_id}")
            self._audit(
                cur, "intent_status_changed", "intent", intent_id,
                {"status": status, "detail": detail, **(payload or {})},
            )

    # --- Orders and fills ----------------------------------------------------

    def plan_order(self, request: OrderRequest) -> dict:
        order, _created = self.plan_order_with_created(request)
        return order

    def plan_order_with_created(self, request: OrderRequest) -> tuple[dict, bool]:
        """Plan once and return whether this transaction inserted the order.

        The boolean is a submission capability: concurrent callers that find
        the already-planned order may synchronize it but must never POST it.
        """
        order_id = f"ord-{request.client_order_id[-24:]}"
        now = self.clock().isoformat()
        with self._tx() as cur:
            existing = cur.execute(
                "SELECT * FROM orders WHERE intent_id = ?", (request.intent_id,)
            ).fetchone()
            if existing is not None:
                if existing["client_order_id"] != request.client_order_id:
                    raise LedgerConflict("Intent already has a different order")
                return dict(existing), False
            active = cur.execute(
                """SELECT order_id FROM orders
                   WHERE account_id = ? AND symbol = ?
                     AND status IN ('pending_submit', 'submitted', 'partially_filled')
                   LIMIT 1""",
                (request.account_id, request.symbol),
            ).fetchone()
            if active is not None:
                raise LedgerConflict(
                    f"Active order {active['order_id']} already exists for {request.symbol}"
                )
            cur.execute(
                """INSERT INTO orders(
                       order_id, intent_id, account_id, client_order_id, symbol, side,
                       quantity, reference_price, order_type, time_in_force, limit_price,
                       status, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    order_id, request.intent_id, request.account_id, request.client_order_id,
                    request.symbol, request.side.value, str(request.quantity),
                    str(request.reference_price), request.order_type.value,
                    request.time_in_force.value,
                    str(request.limit_price) if request.limit_price is not None else None,
                    OrderStatus.PENDING_SUBMIT.value, now, now,
                ),
            )
            self._audit(
                cur, "order_planned", "order", order_id,
                {
                    "intent_id": request.intent_id,
                    "client_order_id": request.client_order_id,
                    "symbol": request.symbol,
                    "side": request.side,
                    "quantity": request.quantity,
                },
            )
            return dict(cur.execute(
                "SELECT * FROM orders WHERE order_id = ?", (order_id,)
            ).fetchone()), True

    def get_order(self, order_id: str) -> dict | None:
        return self._dict(self.conn.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone())

    def get_order_for_intent(self, intent_id: str) -> dict | None:
        return self._dict(self.conn.execute(
            "SELECT * FROM orders WHERE intent_id = ?", (intent_id,)
        ).fetchone())

    def list_orders(self, *, active_only: bool = False) -> list[dict]:
        if active_only:
            values = tuple(s.value for s in ACTIVE_ORDER_STATUSES)
            placeholders = ",".join("?" for _ in values)
            rows = self.conn.execute(
                f"SELECT * FROM orders WHERE status IN ({placeholders}) ORDER BY created_at", values
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM orders ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    def acknowledge_order(self, order_id: str, broker_order: BrokerOrder) -> None:
        with self._tx() as cur:
            local = cur.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
            if local is None:
                raise LedgerError(f"Unknown order: {order_id}")
            mismatches = []
            for field, actual in (
                ("client_order_id", broker_order.client_order_id),
                ("account_id", broker_order.account_id),
                ("symbol", broker_order.symbol),
                ("side", broker_order.side.value),
                ("quantity", str(broker_order.quantity)),
                ("order_type", broker_order.order_type.value),
                ("time_in_force", broker_order.time_in_force.value),
            ):
                if local[field] != actual:
                    mismatches.append(field)
            if mismatches:
                raise LedgerConflict(f"Broker acknowledgement mismatched: {', '.join(mismatches)}")
            if local["broker_order_id"] and local["broker_order_id"] != broker_order.broker_order_id:
                raise LedgerConflict("Broker acknowledgement changed broker_order_id")
            local_limit = Decimal(local["limit_price"]) if local["limit_price"] else None
            if local_limit != broker_order.limit_price:
                raise LedgerConflict("Broker acknowledgement mismatched: limit_price")
            local_status = OrderStatus(local["status"])
            terminal_orders = {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED}
            if local_status in terminal_orders and local_status != broker_order.status:
                raise LedgerConflict(
                    f"Terminal order {order_id} cannot change from {local_status.value} "
                    f"to {broker_order.status.value}"
                )
            if Decimal(local["filled_quantity"]) > broker_order.filled_quantity:
                raise LedgerConflict("Broker acknowledgement reduced filled quantity")
            if (
                local["broker_order_id"]
                and broker_order.updated_at < datetime.fromisoformat(local["updated_at"])
            ):
                raise LedgerConflict("Broker acknowledgement moved updated_at backwards")
            if (
                local_status == OrderStatus.PARTIALLY_FILLED
                and broker_order.status == OrderStatus.SUBMITTED
            ):
                raise LedgerConflict("Broker acknowledgement regressed partial-fill state")
            cur.execute(
                """UPDATE orders SET broker_order_id = ?, filled_quantity = ?, status = ?,
                     rejection_reason = ?, updated_at = ? WHERE order_id = ?""",
                (
                    broker_order.broker_order_id, str(broker_order.filled_quantity),
                    broker_order.status.value, broker_order.rejection_reason,
                    broker_order.updated_at.isoformat(), order_id,
                ),
            )
            self._audit(
                cur, "broker_order_synchronized", "order", order_id,
                {
                    "broker_order_id": broker_order.broker_order_id,
                    "status": broker_order.status,
                    "filled_quantity": broker_order.filled_quantity,
                },
            )

    def abandon_missing_order(self, order_id: str, reason: str) -> dict:
        """Close a local order only after an operator confirms broker absence.

        This is deliberately distinct from a broker cancellation: it records
        that the broker could not find the client ID and makes the exceptional
        local resolution visible in the audit trail.
        """
        if not reason.strip():
            raise ValueError("An operator reason is required")
        with self._tx() as cur:
            order = cur.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
            if order is None:
                raise LedgerError(f"Unknown order: {order_id}")
            if OrderStatus(order["status"]) not in ACTIVE_ORDER_STATUSES:
                raise LedgerConflict("Only an active local order can be abandoned")
            cur.execute(
                """UPDATE orders SET status = ?, rejection_reason = ?, updated_at = ?
                   WHERE order_id = ?""",
                (OrderStatus.CANCELED.value, reason, self.clock().isoformat(), order_id),
            )
            self._audit(
                cur,
                "missing_order_abandoned_by_operator",
                "order",
                order_id,
                {"reason": reason, "client_order_id": order["client_order_id"]},
            )
            return dict(cur.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone())

    def record_fill(self, order_id: str, fill: Fill) -> bool:
        """Append one fill and atomically apply it to expected positions.

        Returns ``False`` for a previously recorded broker fill, making replay
        of broker event history safe.
        """
        with self._tx() as cur:
            order = cur.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
            if order is None:
                raise LedgerError(f"Unknown order: {order_id}")
            replay = cur.execute(
                "SELECT * FROM fills WHERE fill_id = ?", (fill.fill_id,)
            ).fetchone()
            if replay is not None:
                compared = {
                    "order_id": order_id,
                    "broker_order_id": fill.broker_order_id,
                    "client_order_id": fill.client_order_id,
                    "account_id": fill.account_id,
                    "symbol": fill.symbol,
                    "side": fill.side.value,
                    "quantity": str(fill.quantity),
                    "price": str(fill.price),
                    "commission": str(fill.commission),
                    "occurred_at": fill.occurred_at.isoformat(),
                }
                mismatches = [
                    field for field, expected in compared.items()
                    if replay[field] != expected
                ]
                if mismatches:
                    raise LedgerConflict(
                        f"Fill ID {fill.fill_id} was replayed with changed fields: "
                        f"{', '.join(mismatches)}"
                    )
                return False
            for expected, actual, label in (
                (order["client_order_id"], fill.client_order_id, "client_order_id"),
                (order["account_id"], fill.account_id, "account_id"),
                (order["symbol"], fill.symbol, "symbol"),
                (order["side"], fill.side.value, "side"),
            ):
                if expected != actual:
                    raise LedgerConflict(f"Fill {label} does not match its order")
            if order["broker_order_id"] and order["broker_order_id"] != fill.broker_order_id:
                raise LedgerConflict("Fill broker_order_id does not match its order")
            prior_fills = cur.execute(
                "SELECT quantity FROM fills WHERE order_id = ?", (order_id,)
            ).fetchall()
            cumulative = sum((Decimal(r["quantity"]) for r in prior_fills), ZERO) + fill.quantity
            if cumulative > Decimal(order["quantity"]):
                raise LedgerConflict("Cumulative fill quantity exceeds the ordered quantity")
            if order["broker_order_id"] and cumulative > Decimal(order["filled_quantity"]):
                raise LedgerConflict(
                    "Cumulative fill activity exceeds broker-reported filled quantity"
                )

            now = self.clock().isoformat()
            cur.execute(
                """INSERT INTO fills(
                       fill_id, order_id, broker_order_id, client_order_id, account_id,
                       symbol, side, quantity, price, commission, occurred_at, recorded_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fill.fill_id, order_id, fill.broker_order_id, fill.client_order_id,
                    fill.account_id, fill.symbol, fill.side.value, str(fill.quantity),
                    str(fill.price), str(fill.commission), fill.occurred_at.isoformat(), now,
                ),
            )
            position = cur.execute(
                "SELECT * FROM position_state WHERE account_id = ? AND symbol = ?",
                (fill.account_id, fill.symbol),
            ).fetchone()
            old_qty = Decimal(position["quantity"]) if position else ZERO
            old_avg = Decimal(position["avg_price"]) if position else ZERO
            if fill.side == Side.BUY:
                new_qty = old_qty + fill.quantity
                new_avg = ((old_qty * old_avg) + (fill.quantity * fill.price)) / new_qty
            else:
                new_qty = old_qty - fill.quantity
                if new_qty < ZERO:
                    raise LedgerConflict("Recorded sell fill would create a short position")
                new_avg = old_avg if new_qty > ZERO else ZERO
            if new_qty == ZERO:
                cur.execute(
                    "DELETE FROM position_state WHERE account_id = ? AND symbol = ?",
                    (fill.account_id, fill.symbol),
                )
            else:
                cur.execute(
                    """INSERT INTO position_state(account_id, symbol, quantity, avg_price, updated_at)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(account_id, symbol) DO UPDATE SET
                         quantity=excluded.quantity,
                         avg_price=excluded.avg_price,
                         updated_at=excluded.updated_at""",
                    (fill.account_id, fill.symbol, str(new_qty), str(new_avg), now),
                )
            account_state = cur.execute(
                "SELECT expected_cash FROM account_state WHERE account_id = ?",
                (fill.account_id,),
            ).fetchone()
            if account_state is None:
                raise LedgerConflict("Cash baseline is missing for fill accounting")
            cash_before = Decimal(account_state["expected_cash"])
            cash_delta = fill.quantity * fill.price
            cash_after = (
                cash_before - cash_delta - fill.commission
                if fill.side == Side.BUY
                else cash_before + cash_delta - fill.commission
            )
            cur.execute(
                """UPDATE account_state SET expected_cash = ?, updated_at = ?
                   WHERE account_id = ?""",
                (str(cash_after), now, fill.account_id),
            )
            self._audit(
                cur, "fill_recorded", "fill", fill.fill_id,
                {
                    "order_id": order_id,
                    "symbol": fill.symbol,
                    "side": fill.side,
                    "quantity": fill.quantity,
                    "price": fill.price,
                    "position_quantity_after": new_qty,
                    "expected_cash_after": cash_after,
                },
            )
            return True

    def list_fills(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM fills ORDER BY occurred_at, fill_id"
        ).fetchall()]

    def filled_quantity_for_order(self, order_id: str) -> Decimal:
        rows = self.conn.execute(
            "SELECT quantity FROM fills WHERE order_id = ?", (order_id,)
        ).fetchall()
        return sum((Decimal(r["quantity"]) for r in rows), ZERO)

    def daily_turnover(self, account_id: str, utc_date: str) -> Decimal:
        rows = self.conn.execute(
            """SELECT quantity, price FROM fills
               WHERE account_id = ? AND substr(occurred_at, 1, 10) = ?""",
            (account_id, utc_date),
        ).fetchall()
        return sum((Decimal(r["quantity"]) * Decimal(r["price"]) for r in rows), ZERO)

    # --- Phase-3 durable equity guardrails ----------------------------------

    def get_equity_guardrails(self, account_id: str) -> dict | None:
        return self._dict(self.conn.execute(
            "SELECT * FROM equity_guardrails WHERE account_id = ?", (account_id,)
        ).fetchone())

    def observe_equity(
        self,
        account: AccountSnapshot,
        trading_date: str,
        *,
        allow_new_session: bool,
    ) -> dict:
        """Persist an immutable daily baseline and monotonic all-time high-water.

        A new session uses the broker's documented previous-close equity, not
        whatever intraday equity happened to be observed first. New-session
        creation always requires the caller's explicit operator confirmation.
        """
        try:
            parsed_date = date.fromisoformat(trading_date)
        except (TypeError, ValueError) as exc:
            raise LedgerConflict("trading_date must be an ISO calendar date") from exc
        if account.equity <= ZERO:
            raise LedgerConflict("Positive current equity is required")
        with self._tx() as cur:
            row = cur.execute(
                "SELECT * FROM equity_guardrails WHERE account_id = ?",
                (account.account_id,),
            ).fetchone()
            now = self.clock().isoformat()
            if row is None:
                if not allow_new_session:
                    raise LedgerConflict(
                        "Equity guardrails are not initialized for this trading session"
                    )
                if account.last_equity is None or account.last_equity <= ZERO:
                    raise LedgerConflict("Broker previous-close last_equity is required")
                day_start = account.last_equity
                high_water = max(day_start, account.equity)
                cur.execute(
                    """INSERT INTO equity_guardrails(
                           account_id, trading_date, day_start_equity,
                           high_water_equity, last_equity, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        account.account_id, parsed_date.isoformat(), str(day_start),
                        str(high_water), str(account.equity), now,
                    ),
                )
                self._audit(
                    cur, "equity_session_initialized", "account", account.account_id,
                    {
                        "trading_date": parsed_date.isoformat(),
                        "day_start_equity": day_start,
                        "high_water_equity": high_water,
                    },
                )
            else:
                current_date = date.fromisoformat(row["trading_date"])
                if parsed_date < current_date:
                    raise LedgerConflict("Equity trading_date cannot move backwards")
                if parsed_date > current_date:
                    if not allow_new_session:
                        raise LedgerConflict(
                            "A new equity session requires explicit operator confirmation"
                        )
                    active = cur.execute(
                        """SELECT batch_id FROM execution_batches
                           WHERE account_id = ? AND status IN ('approved', 'executing', 'submitted')
                           LIMIT 1""",
                        (account.account_id,),
                    ).fetchone()
                    if active is not None:
                        raise LedgerConflict(
                            f"Cannot roll equity session while batch {active['batch_id']} is active"
                        )
                    if account.last_equity is None or account.last_equity <= ZERO:
                        raise LedgerConflict("Broker previous-close last_equity is required")
                    high_water = max(Decimal(row["high_water_equity"]), account.equity)
                    cur.execute(
                        """UPDATE equity_guardrails SET trading_date = ?,
                             day_start_equity = ?, high_water_equity = ?,
                             last_equity = ?, updated_at = ? WHERE account_id = ?""",
                        (
                            parsed_date.isoformat(), str(account.last_equity), str(high_water),
                            str(account.equity), now, account.account_id,
                        ),
                    )
                    self._audit(
                        cur, "equity_session_rolled", "account", account.account_id,
                        {
                            "trading_date": parsed_date.isoformat(),
                            "day_start_equity": account.last_equity,
                            "high_water_equity": high_water,
                        },
                    )
                else:
                    old_high = Decimal(row["high_water_equity"])
                    high_water = max(old_high, account.equity)
                    cur.execute(
                        """UPDATE equity_guardrails SET high_water_equity = ?,
                             last_equity = ?, updated_at = ? WHERE account_id = ?""",
                        (str(high_water), str(account.equity), now, account.account_id),
                    )
                    if high_water > old_high:
                        self._audit(
                            cur, "equity_high_water_advanced", "account", account.account_id,
                            {"high_water_equity": high_water, "trading_date": trading_date},
                        )
            result = cur.execute(
                "SELECT * FROM equity_guardrails WHERE account_id = ?",
                (account.account_id,),
            ).fetchone()
            return dict(result)

    # --- Phase-3 immutable preview and approval batches ---------------------

    def create_execution_batch(self, plan: object) -> tuple[dict, bool]:
        from .deployment import ExecutionPlan

        if not isinstance(plan, ExecutionPlan):
            raise TypeError("plan must be an ExecutionPlan")
        payload = plan.to_payload()
        # Re-parse before persistence so malformed in-memory plans cannot
        # bypass the same hash verification used after restart.
        ExecutionPlan.from_payload(payload)
        plan_json = json.dumps(
            json_safe(payload), sort_keys=True, separators=(",", ":")
        )
        now = self.clock().isoformat()
        with self._tx() as cur:
            existing = cur.execute(
                """SELECT * FROM execution_batches
                   WHERE batch_id = ? OR plan_hash = ?
                      OR (account_id = ? AND source_hash = ?)""",
                (plan.batch_id, plan.plan_hash, plan.account_id, plan.source_hash),
            ).fetchone()
            if existing is not None:
                if (
                    existing["batch_id"] != plan.batch_id
                    or existing["plan_hash"] != plan.plan_hash
                    or existing["plan_json"] != plan_json
                ):
                    raise LedgerConflict(
                        "Target source/version was already previewed with different market state; "
                        "publish a fresh target_version before creating another batch"
                    )
                if existing["status"] in {"voided", "failed", "complete"}:
                    raise LedgerConflict(
                        "Target source/version belongs to a terminal batch; "
                        "publish a fresh target_version before creating another batch"
                    )
                return dict(existing), False
            cur.execute(
                """INSERT INTO execution_batches(
                       batch_id, plan_hash, source_hash, account_id, deployment_id, trading_date,
                       signal_at, plan_json, status, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'previewed', ?, ?)""",
                (
                    plan.batch_id, plan.plan_hash, plan.source_hash,
                    plan.account_id, plan.deployment_id,
                    plan.trading_date, plan.signal_at.isoformat(), plan_json, now, now,
                ),
            )
            self._audit(
                cur, "execution_batch_previewed", "execution_batch", plan.batch_id,
                {
                    "plan_hash": plan.plan_hash,
                    "deployment_id": plan.deployment_id,
                    "trading_date": plan.trading_date,
                    "order_count": sum(
                        1 for item in plan.items if item.delta_quantity != ZERO
                    ),
                },
            )
            return dict(cur.execute(
                "SELECT * FROM execution_batches WHERE batch_id = ?", (plan.batch_id,)
            ).fetchone()), True

    def get_execution_batch(self, batch_id: str) -> dict | None:
        return self._dict(self.conn.execute(
            "SELECT * FROM execution_batches WHERE batch_id = ?", (batch_id,)
        ).fetchone())

    def list_execution_batches(self, account_id: str | None = None) -> list[dict]:
        if account_id is None:
            rows = self.conn.execute(
                "SELECT * FROM execution_batches ORDER BY created_at, batch_id"
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT * FROM execution_batches WHERE account_id = ?
                   ORDER BY created_at, batch_id""",
                (account_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def load_execution_plan(self, batch_id: str):
        from .deployment import DeploymentError, ExecutionPlan

        row = self.get_execution_batch(batch_id)
        if row is None:
            raise LedgerError(f"Unknown execution batch: {batch_id}")
        try:
            payload = json.loads(row["plan_json"], object_pairs_hook=_ledger_no_duplicates)
            plan = ExecutionPlan.from_payload(payload)
        except (DeploymentError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise LedgerConflict("Stored execution plan failed integrity validation") from exc
        metadata_matches = (
            plan.batch_id == row["batch_id"]
            and plan.plan_hash == row["plan_hash"]
            and plan.source_hash == row["source_hash"]
            and plan.account_id == row["account_id"]
            and plan.deployment_id == row["deployment_id"]
            and plan.trading_date == row["trading_date"]
            and plan.signal_at.isoformat() == row["signal_at"]
        )
        if not metadata_matches:
            raise LedgerConflict("Stored execution batch metadata does not match its plan")
        return plan

    def approve_execution_batch(
        self,
        batch_id: str,
        expected_plan_hash: str,
        *,
        approved_by: str,
        reason: str,
    ) -> dict:
        approved_by = approved_by.strip()
        reason = reason.strip()
        if not approved_by or len(approved_by) > 100:
            raise ValueError("approved_by is required and may not exceed 100 characters")
        if not reason or len(reason) > 500:
            raise ValueError("approval reason is required and may not exceed 500 characters")
        if len(expected_plan_hash) != 64:
            raise LedgerConflict("Approval requires the full 64-character plan hash")
        now = self.clock().isoformat()
        with self._tx() as cur:
            row = cur.execute(
                "SELECT * FROM execution_batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if row is None:
                raise LedgerError(f"Unknown execution batch: {batch_id}")
            if not hmac.compare_digest(row["plan_hash"], expected_plan_hash):
                raise LedgerConflict("Approval plan hash does not match the reviewed batch")
            # Approval is a state transition on the reviewed artifact, not
            # merely on its denormalized hash column. Re-parse and cross-check
            # the complete stored plan before persisting submit authority.
            self.load_execution_plan(batch_id)
            if not self.positions_bootstrapped(row["account_id"]):
                raise LedgerConflict("Position baseline is required before approval")
            control = self.get_control_state(row["account_id"])
            if control["armed"] or control["kill_switch"]:
                raise LedgerConflict("Batch approval requires a disarmed account with kill clear")
            if row["status"] == "approved":
                if row["approved_by"] != approved_by or row["approval_reason"] != reason:
                    raise LedgerConflict("Approved batch cannot be changed")
                return dict(row)
            if row["status"] != "previewed":
                raise LedgerConflict(f"Cannot approve batch in status {row['status']}")
            other = cur.execute(
                """SELECT batch_id FROM execution_batches
                   WHERE account_id = ? AND batch_id != ?
                     AND status IN ('approved', 'executing', 'submitted') LIMIT 1""",
                (row["account_id"], batch_id),
            ).fetchone()
            if other is not None:
                raise LedgerConflict(f"Another execution batch is active: {other['batch_id']}")
            cur.execute(
                """UPDATE execution_batches SET status = 'approved', approved_at = ?,
                     approved_by = ?, approval_reason = ?, updated_at = ?
                   WHERE batch_id = ?""",
                (now, approved_by, reason, now, batch_id),
            )
            self._audit(
                cur, "execution_batch_approved", "execution_batch", batch_id,
                {"plan_hash": expected_plan_hash, "approved_by": approved_by, "reason": reason},
            )
            return dict(cur.execute(
                "SELECT * FROM execution_batches WHERE batch_id = ?", (batch_id,)
            ).fetchone())

    def begin_execution_batch(self, batch_id: str) -> dict:
        now = self.clock().isoformat()
        with self._tx() as cur:
            row = cur.execute(
                "SELECT * FROM execution_batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if row is None:
                raise LedgerError(f"Unknown execution batch: {batch_id}")
            if row["status"] not in {"approved", "executing", "submitted"}:
                raise LedgerConflict(f"Cannot execute batch in status {row['status']}")
            resumed = row["status"] != "approved"
            cur.execute(
                """UPDATE execution_batches SET status = 'executing', last_error = '',
                     updated_at = ? WHERE batch_id = ?""",
                (now, batch_id),
            )
            self._audit(
                cur, "execution_batch_started", "execution_batch", batch_id,
                {"resumed": resumed, "prior_status": row["status"]},
            )
            return dict(cur.execute(
                "SELECT * FROM execution_batches WHERE batch_id = ?", (batch_id,)
            ).fetchone())

    def set_execution_batch_status(self, batch_id: str, status: str) -> dict:
        if status not in {"submitted", "complete", "failed"}:
            raise ValueError("Invalid execution batch terminal/progress status")
        now = self.clock().isoformat()
        with self._tx() as cur:
            row = cur.execute(
                "SELECT * FROM execution_batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if row is None:
                raise LedgerError(f"Unknown execution batch: {batch_id}")
            if row["status"] == "complete":
                if status != "complete":
                    raise LedgerConflict("Completed execution batch is immutable")
                return dict(row)
            if row["status"] != "executing":
                raise LedgerConflict(f"Cannot set result from batch status {row['status']}")
            cur.execute(
                """UPDATE execution_batches SET status = ?, last_error = '', updated_at = ?
                   WHERE batch_id = ?""",
                (status, now, batch_id),
            )
            self._audit(
                cur, "execution_batch_status_changed", "execution_batch", batch_id,
                {"status": status},
            )
            return dict(cur.execute(
                "SELECT * FROM execution_batches WHERE batch_id = ?", (batch_id,)
            ).fetchone())

    def record_execution_batch_error(self, batch_id: str, error: str) -> None:
        safe_error = error.strip()[:500] or "unspecified execution failure"
        with self._tx() as cur:
            row = cur.execute(
                "SELECT status FROM execution_batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if row is None:
                raise LedgerError(f"Unknown execution batch: {batch_id}")
            if row["status"] == "complete":
                raise LedgerConflict("Completed execution batch is immutable")
            cur.execute(
                "UPDATE execution_batches SET last_error = ?, updated_at = ? WHERE batch_id = ?",
                (safe_error, self.clock().isoformat(), batch_id),
            )
            self._audit(
                cur, "execution_batch_interrupted", "execution_batch", batch_id,
                {"error": safe_error, "status": row["status"]},
            )

    def void_execution_batch(self, batch_id: str, *, operator: str, reason: str) -> dict:
        operator = operator.strip()
        reason = reason.strip()
        if not operator or not reason:
            raise ValueError("Voiding a batch requires an operator and reason")
        with self._tx() as cur:
            row = cur.execute(
                "SELECT * FROM execution_batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if row is None:
                raise LedgerError(f"Unknown execution batch: {batch_id}")
            if row["status"] == "voided":
                return dict(row)
            if row["status"] not in {"previewed", "approved"}:
                raise LedgerConflict(
                    "Only a never-started previewed or approved batch can be voided"
                )
            control = self.get_control_state(row["account_id"])
            if control["armed"]:
                raise LedgerConflict("Disarm before voiding an execution batch")
            cur.execute(
                """UPDATE execution_batches SET status = 'voided', last_error = ?,
                     updated_at = ? WHERE batch_id = ?""",
                (reason[:500], self.clock().isoformat(), batch_id),
            )
            self._audit(
                cur, "execution_batch_voided", "execution_batch", batch_id,
                {"operator": operator[:100], "reason": reason[:500]},
            )
            return dict(cur.execute(
                "SELECT * FROM execution_batches WHERE batch_id = ?", (batch_id,)
            ).fetchone())

    # --- Reconciliation audit ------------------------------------------------

    def record_reconciliation(
        self,
        run_id: str,
        account_id: str,
        started_at: str,
        completed_at: str,
        clean: bool,
        issues: list[dict],
    ) -> None:
        with self._tx() as cur:
            cur.execute(
                """INSERT INTO reconciliation_runs(
                       run_id, account_id, started_at, completed_at, clean,
                       issue_count, payload_json
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, account_id, started_at, completed_at, int(clean), len(issues),
                    json.dumps(json_safe({"issues": issues}), sort_keys=True),
                ),
            )
            self._audit(
                cur, "reconciliation_completed", "reconciliation", run_id,
                {"account_id": account_id, "clean": clean, "issue_count": len(issues)},
            )

    def list_reconciliations(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM reconciliation_runs ORDER BY started_at"
        ).fetchall()]

    def latest_reconciliation(self, account_id: str) -> dict | None:
        return self._dict(self.conn.execute(
            """SELECT * FROM reconciliation_runs WHERE account_id = ?
               ORDER BY rowid DESC LIMIT 1""",
            (account_id,),
        ).fetchone())

    def reconciliation_follows_last_control_change(
        self, account_id: str, run_id: str
    ) -> bool:
        """Compare append-only audit sequence, avoiding timestamp tie ambiguity."""
        rec = self.conn.execute(
            """SELECT sequence FROM audit_events
               WHERE event_type = 'reconciliation_completed' AND entity_id = ?
               ORDER BY sequence DESC LIMIT 1""",
            (run_id,),
        ).fetchone()
        control = self.conn.execute(
            """SELECT sequence FROM audit_events
               WHERE event_type = 'control_state_changed' AND entity_id = ?
               ORDER BY sequence DESC LIMIT 1""",
            (account_id,),
        ).fetchone()
        if rec is None:
            return False
        return control is None or rec["sequence"] > control["sequence"]

    def list_audit_events(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM audit_events ORDER BY sequence"
        ).fetchall()]

    def snapshot(self, account_id: str) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "control": self.get_control_state(account_id),
            "positions_bootstrapped": self.positions_bootstrapped(account_id),
            "positions": [
                {"symbol": p.symbol, "quantity": str(p.quantity), "avg_price": str(p.avg_price)}
                for p in self.list_positions(account_id)
            ],
            "expected_cash": (
                str(self.expected_cash(account_id))
                if self.expected_cash(account_id) is not None else None
            ),
            "equity_guardrails": self.get_equity_guardrails(account_id),
            "execution_batches": self.list_execution_batches(account_id),
            "intents": self.list_intents(),
            "orders": self.list_orders(),
            "fills": self.list_fills(),
            "reconciliations": self.list_reconciliations(),
        }
