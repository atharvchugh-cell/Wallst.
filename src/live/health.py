"""Operational health snapshot for supervised Phase-4 paper trading."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .ledger import Ledger
from .alerts import AlertManager
from .models import ensure_aware
from .phase4_models import Phase4Policy, SnapshotSigner
from .phase4_store import (
    RECONCILIATION_READY_SECONDS,
    STREAM_LEASE_SECONDS,
    Phase4Store,
)


class HealthReporter:
    def __init__(
        self,
        ledger: Ledger,
        policy: Phase4Policy,
        *,
        signer: SnapshotSigner | None = None,
        broker=None,
        scheduler=None,
        publisher=None,
        alerts: AlertManager | None = None,
        clock=None,
    ) -> None:
        self.ledger = ledger
        self.store = Phase4Store(ledger)
        self.policy = policy
        self.signer = signer
        self.broker = broker
        self.scheduler = scheduler
        self.publisher = publisher
        self.alerts = alerts
        self.clock = clock or ledger.clock

    def report(self, account_id: str) -> dict[str, Any]:
        now = ensure_aware(self.clock(), "health time")
        quick = self.ledger.conn.execute("PRAGMA quick_check").fetchone()
        foreign = self.ledger.conn.execute("PRAGMA foreign_key_check").fetchone()
        integrity = quick is not None and quick[0] == "ok" and foreign is None
        latest_snapshot = self.store.latest_snapshot()
        snapshot_valid = False
        snapshot_error = None
        if latest_snapshot is not None:
            try:
                snapshot = self.store.load_snapshot(latest_snapshot["snapshot_id"])
                snapshot.verify(
                    self.policy, self.signer, now=now
                )
                if self.publisher is not None:
                    self.publisher.to_execution_targets(snapshot)
                snapshot_valid = True
            except Exception as exc:
                snapshot_error = type(exc).__name__
        broker_connected = None
        broker_error = None
        if self.broker is not None:
            try:
                account = self.broker.get_account()
                broker_connected = account.account_id == account_id
            except Exception as exc:
                broker_connected = False
                broker_error = type(exc).__name__
        active_orders = self.ledger.list_orders(active_only=True)
        overdue = []
        for order in active_orders:
            age = (now - ensure_aware(datetime.fromisoformat(order["updated_at"]), "order time")).total_seconds()
            if age > self.policy.max_open_order_age_seconds:
                overdue.append(order["order_id"])
        latest_reconciliation = self.ledger.latest_reconciliation(account_id)
        reconciliation_fresh = False
        if latest_reconciliation is not None:
            try:
                reconciliation_age = (
                    now - ensure_aware(
                        datetime.fromisoformat(latest_reconciliation["completed_at"]),
                        "reconciliation time",
                    )
                ).total_seconds()
                reconciliation_fresh = -2 <= reconciliation_age <= RECONCILIATION_READY_SECONDS
            except (TypeError, ValueError):
                reconciliation_fresh = False
        stream = self.store.stream_state()
        stream_lease_fresh = False
        if stream is not None:
            try:
                stream_age = (
                    now - ensure_aware(datetime.fromisoformat(stream["updated_at"]), "stream time")
                ).total_seconds()
                stream_lease_fresh = -2 <= stream_age <= STREAM_LEASE_SECONDS
            except (TypeError, ValueError):
                stream_lease_fresh = False
        control = self.ledger.get_control_state(account_id)
        unresolved = self.store.list_alerts(unresolved_only=True)
        next_action = None
        if self.scheduler is not None:
            try:
                next_action = self.scheduler.next_expected_action(now=now)
            except Exception as exc:
                next_action = {"action": "calendar_error", "error": type(exc).__name__}
        blockers = []
        if not integrity:
            blockers.append("database_integrity")
        if control["kill_switch"]:
            blockers.append("kill_switch")
        if (
            latest_reconciliation is None
            or not latest_reconciliation["clean"]
            or not reconciliation_fresh
        ):
            blockers.append("reconciliation")
        if not snapshot_valid:
            blockers.append("valid_snapshot")
        if overdue:
            blockers.append("overdue_orders")
        if any(row["severity"] == "critical" for row in unresolved):
            blockers.append("critical_alerts")
        if (
            stream is None or not stream["connected"] or stream["recovering"]
            or not stream_lease_fresh
        ):
            blockers.append("stream_recovery")
        if self.broker is not None and not broker_connected:
            blockers.append("broker_connectivity")
        report = {
            "as_of": now.isoformat(),
            "operating_mode": self.policy.mode.value,
            "armed": control["armed"],
            "kill_switch": control["kill_switch"],
            "broker_connectivity": broker_connected,
            "broker_error_type": broker_error,
            "stream_state": stream,
            "stream_lease_fresh": stream_lease_fresh,
            "latest_reconciliation": latest_reconciliation,
            "reconciliation_fresh": reconciliation_fresh,
            "latest_valid_snapshot": latest_snapshot if snapshot_valid else None,
            "snapshot_error_type": snapshot_error,
            "unresolved_orders": active_orders,
            "overdue_order_ids": overdue,
            "unresolved_alerts": unresolved,
            "latest_backup": self.store.latest_backup(),
            "database_integrity": integrity,
            "next_expected_scheduled_action": next_action,
            "submission_ready": not blockers,
            "submission_blockers": blockers,
        }
        if self.alerts is not None:
            self._emit_health_alerts(report)
            escalated = self.store.escalate_alerts(
                older_than_seconds=self.policy.critical_alert_escalation_seconds
            )
            for alert in escalated:
                self.alerts.deliver(alert)
            report["escalated_alert_ids"] = [row["alert_id"] for row in escalated]
            report["unresolved_alerts"] = self.store.list_alerts(unresolved_only=True)
            if (
                any(row["severity"] == "critical" for row in report["unresolved_alerts"])
                and "critical_alerts" not in report["submission_blockers"]
            ):
                report["submission_blockers"].append("critical_alerts")
                report["submission_ready"] = False
        return report

    def _emit_health_alerts(self, report: dict[str, Any]) -> None:
        if not report["database_integrity"]:
            self.alerts.emit(
                "critical", "database_integrity_failure", "SQLite integrity check failed",
                dedupe_key="database-integrity",
            )
        if report["kill_switch"]:
            self.alerts.emit(
                "critical", "kill_switch", "Persistent kill switch is engaged",
                dedupe_key="kill-switch-engaged",
            )
        if report["latest_valid_snapshot"] is None:
            self.alerts.emit(
                "high", "snapshot_invalid_or_expired",
                "No currently valid signed target snapshot",
                dedupe_key="latest-snapshot-invalid",
            )
        for order in report["unresolved_orders"]:
            if order["order_id"] not in report["overdue_order_ids"]:
                continue
            category = (
                "partial_fill_overdue" if order["status"] == "partially_filled"
                else "overdue_open_order"
            )
            self.alerts.emit(
                "high", category, "Paper order exceeded maximum open-order age",
                entity_id=order["order_id"], dedupe_key=f"overdue:{order['order_id']}",
            )
        stream = report["stream_state"]
        if (
            stream is None or not stream["connected"] or stream["recovering"]
            or not report["stream_lease_fresh"]
        ):
            self.alerts.emit(
                "critical", "broker_disconnection",
                "Order stream is disconnected, stale, or awaiting REST recovery",
                entity_id="alpaca-paper-trade-updates", dedupe_key="health-stream-blocked",
            )
        reconciliation = report["latest_reconciliation"]
        if (
            reconciliation is None
            or not reconciliation["clean"]
            or not report["reconciliation_fresh"]
        ):
            self.alerts.emit(
                "critical", "reconciliation_mismatch",
                "No recent clean paper reconciliation",
                dedupe_key="health-reconciliation-blocked",
            )
