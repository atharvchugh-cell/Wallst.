"""Daily and cumulative Phase-4 paper-soak evidence."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import Any

from .ledger import Ledger
from .phase4_store import Phase4Store


class PaperSoakReporter:
    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger
        self.store = Phase4Store(ledger)

    def report(self, trading_date: str | None = None) -> dict[str, Any]:
        day = trading_date or self.ledger.clock().date().isoformat()
        daily = self._metrics(day)
        cumulative = self._metrics(None)
        return {
            "report_schema_version": 1,
            "paper_only": True,
            "paper_fill_limitation": (
                "Alpaca paper fills do not establish live queue position, market impact, "
                "latency, availability, or achievable real-money execution."
            ),
            "daily": {"trading_date": day, **daily},
            "cumulative": cumulative,
        }

    def _metrics(self, day: str | None) -> dict[str, Any]:
        date_filter = " AND substr(created_at,1,10)=?" if day else ""
        params = (day,) if day else ()
        schedule = self.ledger.conn.execute(
            f"SELECT status, COUNT(*) n FROM scheduler_runs WHERE 1=1{date_filter} GROUP BY status",
            params,
        ).fetchall()
        snapshots = self.ledger.conn.execute(
            f"SELECT COUNT(*) FROM target_snapshots WHERE 1=1{date_filter}", params
        ).fetchone()[0]
        batches = self.ledger.conn.execute(
            f"SELECT status, COUNT(*) n FROM execution_batches WHERE 1=1{date_filter} GROUP BY status",
            params,
        ).fetchall()
        orders = self.ledger.conn.execute(
            f"SELECT * FROM orders WHERE 1=1{date_filter}", params
        ).fetchall()
        fill_filter = " WHERE substr(recorded_at,1,10)=?" if day else ""
        fills = self.ledger.conn.execute(
            f"SELECT * FROM fills{fill_filter}", params
        ).fetchall()
        rec_filter = " WHERE substr(completed_at,1,10)=?" if day else ""
        reconciliations = self.ledger.conn.execute(
            f"SELECT * FROM reconciliation_runs{rec_filter}", params
        ).fetchall()
        slippage_weighted = Decimal("0")
        slippage_qty = Decimal("0")
        order_by_id = {row["broker_order_id"]: row for row in orders if row["broker_order_id"]}
        # Cumulative fill reports need all orders for the reference lookup.
        if not day:
            order_by_id = {
                row["broker_order_id"]: row for row in self.ledger.conn.execute("SELECT * FROM orders")
                if row["broker_order_id"]
            }
        for fill in fills:
            order = order_by_id.get(fill["broker_order_id"])
            if order is None:
                continue
            price = Decimal(fill["price"])
            reference = Decimal(order["reference_price"])
            qty = Decimal(fill["quantity"])
            bps = (
                (price / reference - Decimal("1")) * Decimal("10000")
                if fill["side"] == "buy"
                else (reference / price - Decimal("1")) * Decimal("10000")
            )
            slippage_weighted += bps * qty
            slippage_qty += qty
        observation_rows = self.store.soak_observations()
        if day:
            observation_rows = [row for row in observation_rows if row["trading_date"] == day]
        observations: dict[str, list[Decimal]] = defaultdict(list)
        for row in observation_rows:
            try:
                observations[row["metric"]].append(Decimal(row["value"]))
            except Exception:
                pass
        stream = self.store.stream_state()
        unresolved = self.store.list_alerts(unresolved_only=True)
        if day:
            unresolved = [row for row in unresolved if row["first_seen_at"].startswith(day)]
        quick = self.ledger.conn.execute("PRAGMA quick_check").fetchone()
        return {
            "scheduled_decisions": sum(row["n"] for row in schedule),
            "scheduler_statuses": {row["status"]: row["n"] for row in schedule},
            "published_snapshots": snapshots,
            "approved_plans": sum(
                row["n"] for row in batches
                if row["status"] in {"approved", "executing", "submitted", "complete", "failed"}
            ),
            "plan_statuses": {row["status"]: row["n"] for row in batches},
            "submitted_orders": sum(1 for row in orders if row["broker_order_id"]),
            "fills": len(fills),
            "rejects": sum(1 for row in orders if row["status"] == "rejected"),
            "cancellations": sum(1 for row in orders if row["status"] == "canceled"),
            "partial_fills": sum(1 for row in orders if Decimal(row["filled_quantity"]) not in {
                Decimal("0"), Decimal(row["quantity"])
            }),
            "reconciliations": len(reconciliations),
            "clean_reconciliations": sum(1 for row in reconciliations if row["clean"]),
            "slippage_vs_reference_quote_bps": (
                str(slippage_weighted / slippage_qty) if slippage_qty else None
            ),
            "slippage_vs_next_close_bps": self._mean(observations["next_close_slippage_bps"]),
            "target_vs_actual_weight_error_bps": self._mean(
                observations["target_weight_error_bps"]
            ),
            "process_uptime_seconds": self._sum(observations["process_uptime_seconds"]),
            "unresolved_alerts": len(unresolved),
            "stream_disconnects": stream["disconnect_count"] if stream else 0,
            "stream_recoveries": stream["recovery_count"] if stream else 0,
            "recovery_events": len([
                row for row in self.ledger.list_audit_events()
                if row["event_type"] in {"reconciliation_completed", "alert_escalated"}
                and (day is None or row["occurred_at"].startswith(day))
            ]),
            "database_integrity": bool(quick and quick[0] == "ok"),
            "duplicate_prevention": {
                "unique_client_order_ids": self._is_unique("orders", "client_order_id"),
                "unique_snapshot_sessions": self._is_unique(
                    "target_snapshots", "decision_session"
                ),
                "unique_stream_event_ids": self._is_unique("stream_events", "event_id"),
            },
        }

    @staticmethod
    def _mean(values: list[Decimal]) -> str | None:
        return str(sum(values, Decimal("0")) / len(values)) if values else None

    @staticmethod
    def _sum(values: list[Decimal]) -> str | None:
        return str(sum(values, Decimal("0"))) if values else None

    def _is_unique(self, table: str, column: str) -> bool:
        total, distinct = self.ledger.conn.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT {column}) FROM {table}"
        ).fetchone()
        return total == distinct
