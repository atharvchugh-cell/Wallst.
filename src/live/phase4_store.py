"""Durable Phase-4 records stored inside the authoritative execution ledger."""

from __future__ import annotations

import hashlib
import json
import uuid
from contextlib import nullcontext
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from .ledger import Ledger, LedgerConflict, LedgerError, _ledger_no_duplicates
from .models import ZERO, as_decimal, ensure_aware, json_safe
from .phase4_models import PublishedTargetSnapshot, canonical_bytes


ALERT_SEVERITIES = {"info", "warning", "high", "critical"}
SCHEDULER_STATUSES = {"due", "published", "skipped", "delayed", "failed"}
PHASE4_STREAM_NAME = "alpaca-paper-trade-updates"
STREAM_LEASE_SECONDS = 45
CONTINUOUS_RECONCILE_SECONDS = 60
RECONCILIATION_READY_SECONDS = 120
SOAK_METRICS = {
    "next_close_slippage_bps", "target_weight_error_bps", "process_uptime_seconds",
}


class Phase4Store:
    """Transactional access to Phase-4 tables in a :class:`Ledger`."""

    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger

    def publish_snapshot(self, snapshot: PublishedTargetSnapshot) -> tuple[dict, bool]:
        # Reparse the exact envelope before persistence. This catches malformed
        # hand-constructed dataclass instances and makes restart validation the
        # same as first-write validation.
        parsed = PublishedTargetSnapshot.from_payload(snapshot.to_payload())
        content = parsed.content
        required = {
            "creation_timestamp", "decision_session", "expected_execution_session",
            "account_id_fingerprint", "operation_mode", "expiration_time",
        }
        missing = required - set(content)
        if missing:
            raise LedgerConflict(f"Snapshot content is missing required metadata: {sorted(missing)}")
        snapshot_json = canonical_bytes(parsed.to_payload()).decode("utf-8")
        # Publication is the irreversible boundary between the legacy Phase-3
        # approval model and signed Phase-4 authority. A ledger can never mix
        # both profiles, even if a caller later uses the older CLI.
        self.ledger.bind_execution_profile("phase4")
        now = self.ledger.clock().isoformat()
        with self.ledger._tx() as cur:
            row = cur.execute(
                "SELECT * FROM target_snapshots WHERE snapshot_id=? OR content_hash=?",
                (parsed.snapshot_id, parsed.content_hash),
            ).fetchone()
            if row is not None:
                if row["snapshot_json"] != snapshot_json:
                    raise LedgerConflict("Snapshot identifier was reused with different content")
                return dict(row), False
            duplicate = cur.execute(
                "SELECT snapshot_id FROM target_snapshots WHERE decision_session=?",
                (str(content["decision_session"]),),
            ).fetchone()
            if duplicate is not None:
                raise LedgerConflict(
                    f"Decision session already has immutable snapshot {duplicate['snapshot_id']}"
                )
            cur.execute(
                """INSERT INTO target_snapshots(
                       snapshot_id, content_hash, decision_session, expected_execution_session,
                       account_fingerprint, mode, signed, expires_at, snapshot_json, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    parsed.snapshot_id, parsed.content_hash, str(content["decision_session"]),
                    str(content["expected_execution_session"]),
                    str(content["account_id_fingerprint"]), str(content["operation_mode"]),
                    int(parsed.signature is not None), str(content["expiration_time"]),
                    snapshot_json, now,
                ),
            )
            self.ledger._audit(
                cur, "target_snapshot_published", "target_snapshot", parsed.snapshot_id,
                {
                    "content_hash": parsed.content_hash,
                    "decision_session": content["decision_session"],
                    "expected_execution_session": content["expected_execution_session"],
                    "signed": parsed.signature is not None,
                    "mode": content["operation_mode"],
                },
            )
            return dict(cur.execute(
                "SELECT * FROM target_snapshots WHERE snapshot_id=?", (parsed.snapshot_id,)
            ).fetchone()), True

    def load_snapshot(self, snapshot_id: str) -> PublishedTargetSnapshot:
        row = self.ledger.conn.execute(
            "SELECT * FROM target_snapshots WHERE snapshot_id=?", (snapshot_id,)
        ).fetchone()
        if row is None:
            raise LedgerError(f"Unknown target snapshot: {snapshot_id}")
        try:
            payload = json.loads(row["snapshot_json"], object_pairs_hook=_ledger_no_duplicates)
            snapshot = PublishedTargetSnapshot.from_payload(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LedgerConflict("Stored target snapshot failed integrity validation") from exc
        content = snapshot.content
        if not (
            snapshot.snapshot_id == row["snapshot_id"]
            and snapshot.content_hash == row["content_hash"]
            and str(content["decision_session"]) == row["decision_session"]
            and str(content["expected_execution_session"]) == row["expected_execution_session"]
            and str(content["account_id_fingerprint"]) == row["account_fingerprint"]
            and str(content["operation_mode"]) == row["mode"]
            and int(snapshot.signature is not None) == row["signed"]
            and str(content["expiration_time"]) == row["expires_at"]
        ):
            raise LedgerConflict("Stored target snapshot metadata does not match its envelope")
        return snapshot

    def latest_snapshot(self) -> dict | None:
        row = self.ledger.conn.execute(
            "SELECT * FROM target_snapshots ORDER BY created_at DESC, snapshot_id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row is not None else None

    def list_snapshots(self) -> list[dict]:
        return [dict(row) for row in self.ledger.conn.execute(
            "SELECT * FROM target_snapshots ORDER BY created_at, snapshot_id"
        ).fetchall()]

    def link_execution_plan(
        self, snapshot_id: str, batch_id: str, operation_mode: str,
        *, paper_submission_allowed: bool,
    ) -> tuple[dict, bool]:
        now = self.ledger.clock().isoformat()
        with self.ledger._tx() as cur:
            existing = cur.execute(
                "SELECT * FROM phase4_plan_links WHERE batch_id=?", (batch_id,)
            ).fetchone()
            if existing is not None:
                if not (
                    existing["snapshot_id"] == snapshot_id
                    and existing["operation_mode"] == operation_mode
                    and existing["paper_submission_allowed"] == int(paper_submission_allowed)
                ):
                    raise LedgerConflict("Execution plan is already linked to another policy")
                return dict(existing), False
            other = cur.execute(
                "SELECT batch_id FROM phase4_plan_links WHERE snapshot_id=?", (snapshot_id,)
            ).fetchone()
            if other is not None:
                raise LedgerConflict(
                    f"Target snapshot already produced execution plan {other['batch_id']}"
                )
            cur.execute(
                """INSERT INTO phase4_plan_links(
                       batch_id, snapshot_id, operation_mode, paper_submission_allowed, created_at
                   ) VALUES (?, ?, ?, ?, ?)""",
                (batch_id, snapshot_id, operation_mode, int(paper_submission_allowed), now),
            )
            self.ledger._audit(
                cur, "phase4_plan_linked", "execution_batch", batch_id,
                {"snapshot_id": snapshot_id, "operation_mode": operation_mode,
                 "paper_submission_allowed": paper_submission_allowed},
            )
            return dict(cur.execute(
                "SELECT * FROM phase4_plan_links WHERE batch_id=?", (batch_id,)
            ).fetchone()), True

    def execution_plan_link(self, batch_id: str) -> dict | None:
        row = self.ledger.conn.execute(
            "SELECT * FROM phase4_plan_links WHERE batch_id=?", (batch_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def execution_plan_link_for_snapshot(self, snapshot_id: str) -> dict | None:
        row = self.ledger.conn.execute(
            "SELECT * FROM phase4_plan_links WHERE snapshot_id=?", (snapshot_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def claim_schedule(
        self,
        decision_session: str,
        expected_execution_session: str | None,
        *,
        status: str = "due",
        detail: str = "",
    ) -> tuple[dict, bool]:
        if status not in SCHEDULER_STATUSES:
            raise ValueError("Invalid scheduler status")
        run_id = f"schedule-{decision_session}-{uuid.uuid4().hex[:8]}"
        now = self.ledger.clock().isoformat()
        with self.ledger._tx() as cur:
            existing = cur.execute(
                "SELECT * FROM scheduler_runs WHERE decision_session=?", (decision_session,)
            ).fetchone()
            if existing is not None:
                if existing["expected_execution_session"] != expected_execution_session:
                    raise LedgerConflict(
                        "Authenticated next execution session changed after schedule claim"
                    )
                return dict(existing), False
            cur.execute(
                """INSERT INTO scheduler_runs(
                       run_id, decision_session, expected_execution_session, status, detail,
                       created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, decision_session, expected_execution_session, status,
                    detail.strip()[:1000], now, now,
                ),
            )
            self.ledger._audit(
                cur, "scheduler_run_recorded", "scheduler_run", run_id,
                {
                    "decision_session": decision_session,
                    "expected_execution_session": expected_execution_session,
                    "status": status,
                    "detail": detail.strip()[:1000],
                },
            )
            return dict(cur.execute(
                "SELECT * FROM scheduler_runs WHERE run_id=?", (run_id,)
            ).fetchone()), True

    def update_schedule(
        self, run_id: str, status: str, *, detail: str = "", snapshot_id: str | None = None
    ) -> dict:
        if status not in SCHEDULER_STATUSES:
            raise ValueError("Invalid scheduler status")
        normalized_detail = detail.strip()[:1000]
        now = self.ledger.clock().isoformat()
        with self.ledger._tx() as cur:
            row = cur.execute("SELECT * FROM scheduler_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise LedgerError(f"Unknown scheduler run: {run_id}")
            if row["status"] in {"published", "skipped"}:
                effective_snapshot = snapshot_id if snapshot_id is not None else row["snapshot_id"]
                if (
                    status == row["status"]
                    and normalized_detail == row["detail"]
                    and effective_snapshot == row["snapshot_id"]
                ):
                    return dict(row)
                raise LedgerConflict(
                    f"A {row['status']} scheduler run cannot be rewritten"
                )
            cur.execute(
                """UPDATE scheduler_runs SET status=?, detail=?, snapshot_id=COALESCE(?,snapshot_id),
                     updated_at=? WHERE run_id=?""",
                (status, normalized_detail, snapshot_id, now, run_id),
            )
            self.ledger._audit(
                cur, "scheduler_run_updated", "scheduler_run", run_id,
                {"status": status, "detail": normalized_detail, "snapshot_id": snapshot_id},
            )
            return dict(cur.execute(
                "SELECT * FROM scheduler_runs WHERE run_id=?", (run_id,)
            ).fetchone())

    def list_schedule_runs(self) -> list[dict]:
        return [dict(row) for row in self.ledger.conn.execute(
            "SELECT * FROM scheduler_runs ORDER BY decision_session, created_at"
        ).fetchall()]

    def emit_alert(
        self,
        severity: str,
        category: str,
        message: str,
        *,
        entity_id: str = "",
        dedupe_key: str | None = None,
    ) -> tuple[dict, bool]:
        alert, created, _upgraded = self.emit_alert_with_transition(
            severity, category, message,
            entity_id=entity_id, dedupe_key=dedupe_key,
        )
        return alert, created

    def emit_alert_with_transition(
        self,
        severity: str,
        category: str,
        message: str,
        *,
        entity_id: str = "",
        dedupe_key: str | None = None,
    ) -> tuple[dict, bool, bool]:
        """Emit and atomically report whether an existing incident upgraded."""
        severity = severity.strip().lower()
        if severity not in ALERT_SEVERITIES:
            raise ValueError("Invalid alert severity")
        category = category.strip()[:100]
        message = message.strip()[:2000]
        entity_id = entity_id.strip()[:200]
        if not category or not message:
            raise ValueError("Alert category and message are required")
        key = dedupe_key or hashlib.sha256(
            f"{category}\0{entity_id}\0{message}".encode("utf-8")
        ).hexdigest()
        now = self.ledger.clock().isoformat()
        # A critical alert revokes Phase-4 submit authority. Serialize that
        # durable state transition against the final broker-submission fence.
        guard = self.ledger.execution_guard() if severity == "critical" else nullcontext()
        with guard, self.ledger._tx() as cur:
            row = cur.execute(
                "SELECT * FROM alerts WHERE dedupe_key=? AND resolved_at IS NULL", (key,)
            ).fetchone()
            if row is not None:
                previous_severity = row["severity"]
                cur.execute(
                    """UPDATE alerts SET last_seen_at=?, occurrence_count=occurrence_count+1,
                         severity=CASE
                           WHEN severity='critical' OR ?='critical' THEN 'critical'
                           WHEN severity='high' OR ?='high' THEN 'high'
                           WHEN severity='warning' OR ?='warning' THEN 'warning'
                           ELSE 'info' END
                       WHERE alert_id=?""",
                    (now, severity, severity, severity, row["alert_id"]),
                )
                alert = dict(cur.execute(
                    "SELECT * FROM alerts WHERE alert_id=?", (row["alert_id"],)
                ).fetchone())
                ranks = {"info": 0, "warning": 1, "high": 2, "critical": 3}
                return alert, False, ranks[alert["severity"]] > ranks[previous_severity]
            alert_id = f"alert-{uuid.uuid4().hex[:24]}"
            cur.execute(
                """INSERT INTO alerts(
                       alert_id, dedupe_key, severity, category, message, entity_id,
                       first_seen_at, last_seen_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (alert_id, key, severity, category, message, entity_id, now, now),
            )
            self.ledger._audit(
                cur, "alert_created", "alert", alert_id,
                {"severity": severity, "category": category, "entity_id": entity_id},
            )
            return dict(cur.execute(
                "SELECT * FROM alerts WHERE alert_id=?", (alert_id,)
            ).fetchone()), True, False

    def acknowledge_alert(self, alert_id: str, *, operator: str, note: str) -> dict:
        operator, note = operator.strip()[:100], note.strip()[:500]
        if not operator or not note:
            raise ValueError("Alert acknowledgement requires operator and note")
        now = self.ledger.clock().isoformat()
        with self.ledger._tx() as cur:
            row = cur.execute("SELECT * FROM alerts WHERE alert_id=?", (alert_id,)).fetchone()
            if row is None:
                raise LedgerError(f"Unknown alert: {alert_id}")
            if row["acknowledged_at"] is None:
                cur.execute(
                    """UPDATE alerts SET acknowledged_at=?, acknowledged_by=?,
                         acknowledgement_note=? WHERE alert_id=?""",
                    (now, operator, note, alert_id),
                )
                self.ledger._audit(
                    cur, "alert_acknowledged", "alert", alert_id,
                    {"operator": operator, "note": note},
                )
            elif row["acknowledged_by"] != operator or row["acknowledgement_note"] != note:
                raise LedgerConflict("Alert acknowledgement is immutable")
            return dict(cur.execute(
                "SELECT * FROM alerts WHERE alert_id=?", (alert_id,)
            ).fetchone())

    def resolve_alert(self, alert_id: str, *, operator: str, note: str) -> dict:
        operator, note = operator.strip()[:100], note.strip()[:500]
        if not operator or not note:
            raise ValueError("Alert resolution requires operator and note")
        now = self.ledger.clock().isoformat()
        with self.ledger._tx() as cur:
            row = cur.execute("SELECT * FROM alerts WHERE alert_id=?", (alert_id,)).fetchone()
            if row is None:
                raise LedgerError(f"Unknown alert: {alert_id}")
            if row["resolved_at"] is None:
                cur.execute("UPDATE alerts SET resolved_at=? WHERE alert_id=?", (now, alert_id))
                self.ledger._audit(
                    cur, "alert_resolved", "alert", alert_id,
                    {"operator": operator, "note": note},
                )
            return dict(cur.execute(
                "SELECT * FROM alerts WHERE alert_id=?", (alert_id,)
            ).fetchone())

    def list_alerts(self, *, unresolved_only: bool = False) -> list[dict]:
        where = " WHERE resolved_at IS NULL" if unresolved_only else ""
        return [dict(row) for row in self.ledger.conn.execute(
            f"SELECT * FROM alerts{where} ORDER BY first_seen_at, alert_id"
        ).fetchall()]

    def escalate_alerts(self, *, older_than_seconds: int) -> list[dict]:
        now = ensure_aware(self.ledger.clock(), "alert escalation time")
        escalated: list[dict] = []
        with self.ledger._tx() as cur:
            rows = cur.execute(
                """SELECT * FROM alerts WHERE resolved_at IS NULL AND severity='critical'
                   ORDER BY first_seen_at"""
            ).fetchall()
            for row in rows:
                first = ensure_aware(datetime.fromisoformat(row["first_seen_at"]), "alert time")
                basis = first
                if row["last_escalated_at"]:
                    basis = ensure_aware(
                        datetime.fromisoformat(row["last_escalated_at"]),
                        "alert escalation time",
                    )
                if (now - basis).total_seconds() < older_than_seconds:
                    continue
                cur.execute(
                    """UPDATE alerts SET escalation_count=escalation_count+1,
                         last_escalated_at=? WHERE alert_id=?""",
                    (now.isoformat(), row["alert_id"]),
                )
                self.ledger._audit(
                    cur, "alert_escalated", "alert", row["alert_id"],
                    {"age_seconds": (now - first).total_seconds()},
                )
                escalated.append(dict(cur.execute(
                    "SELECT * FROM alerts WHERE alert_id=?", (row["alert_id"],)
                ).fetchone()))
        return escalated

    def record_stream_event(
        self,
        *,
        event_id: str,
        sequence: int | None,
        client_order_id: str,
        event_type: str,
        broker_updated_at: str,
        payload: dict[str, Any],
        disposition: str,
        recovery_stream_name: str | None = None,
    ) -> bool:
        encoded = canonical_bytes(payload).decode("utf-8")
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        now = self.ledger.clock().isoformat()
        payload_conflict = False
        created = False
        guard = (
            self.ledger.execution_guard()
            if recovery_stream_name is not None else nullcontext()
        )
        with guard, self.ledger._tx() as cur:
            existing = cur.execute(
                "SELECT * FROM stream_events WHERE event_id=?", (event_id,)
            ).fetchone()
            if existing is not None:
                if existing["payload_hash"] != digest:
                    payload_conflict = True
                needs_recovery = (
                    payload_conflict
                    or existing["disposition"] not in {"applied", "recovered"}
                )
            else:
                cur.execute(
                    """INSERT INTO stream_events(
                           event_id, stream_sequence, client_order_id, event_type,
                           broker_updated_at, payload_hash, payload_json, received_at, disposition
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        event_id, sequence, client_order_id, event_type, broker_updated_at,
                        digest, encoded, now, disposition,
                    ),
                )
                self.ledger._audit(
                    cur, "stream_event_recorded", "broker_order", client_order_id,
                    {"event_id": event_id, "sequence": sequence, "event_type": event_type,
                     "disposition": disposition},
                )
                created = True
                needs_recovery = True

            if recovery_stream_name is not None and needs_recovery:
                # Event ingestion and submit-authority revocation are one
                # transaction under the same cross-process execution fence as
                # the final OMS authorizer. A crash can therefore leave either
                # the old healthy state with no event or a durable recovering
                # state with the event, never a healthy pending-event gap.
                cur.execute(
                    """INSERT INTO stream_state(
                           stream_name, connected, recovering, last_sequence,
                           last_event_at, last_recovery_at, disconnect_count,
                           recovery_count, updated_at
                       ) VALUES (?, 0, 1, NULL, NULL, NULL, 0, 0, ?)
                       ON CONFLICT(stream_name) DO UPDATE SET
                         recovering=1,
                         updated_at=excluded.updated_at""",
                    (recovery_stream_name, now),
                )

        if payload_conflict:
            raise LedgerConflict("Stream event ID was replayed with different content")
        return created

    def stream_event(self, event_id: str) -> dict | None:
        row = self.ledger.conn.execute(
            "SELECT * FROM stream_events WHERE event_id=?", (event_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def pending_stream_events(self) -> list[dict]:
        return [dict(row) for row in self.ledger.conn.execute(
            "SELECT * FROM stream_events WHERE disposition='pending' ORDER BY received_at, event_id"
        ).fetchall()]

    def mark_stream_event_disposition(self, event_id: str, disposition: str) -> dict:
        disposition = disposition.strip()[:100]
        if not disposition or disposition == "pending":
            raise ValueError("Final stream-event disposition is required")
        with self.ledger._tx() as cur:
            row = cur.execute(
                "SELECT * FROM stream_events WHERE event_id=?", (event_id,)
            ).fetchone()
            if row is None:
                raise LedgerError(f"Unknown stream event: {event_id}")
            if row["disposition"] == "pending":
                cur.execute(
                    "UPDATE stream_events SET disposition=? WHERE event_id=?",
                    (disposition, event_id),
                )
                self.ledger._audit(
                    cur, "stream_event_applied", "broker_order", row["client_order_id"],
                    {"event_id": event_id, "disposition": disposition},
                )
            elif row["disposition"] != disposition:
                raise LedgerConflict("Stream event final disposition is immutable")
            return dict(cur.execute(
                "SELECT * FROM stream_events WHERE event_id=?", (event_id,)
            ).fetchone())

    def set_stream_state(
        self,
        stream_name: str,
        *,
        connected: bool,
        recovering: bool,
        last_sequence: int | None = None,
        event_at: str | None = None,
        recovery_completed: bool = False,
        disconnected: bool = False,
    ) -> dict:
        now = self.ledger.clock().isoformat()
        # Stream health is a submit-authority input. Its transition and the
        # final OMS authorization check must have one linearizable order.
        with self.ledger.execution_guard(), self.ledger._tx() as cur:
            cur.execute(
                """INSERT INTO stream_state(
                       stream_name, connected, recovering, last_sequence, last_event_at,
                       last_recovery_at, disconnect_count, recovery_count, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(stream_name) DO UPDATE SET
                     connected=excluded.connected,
                     recovering=excluded.recovering,
                     last_sequence=COALESCE(excluded.last_sequence,stream_state.last_sequence),
                     last_event_at=COALESCE(excluded.last_event_at,stream_state.last_event_at),
                     last_recovery_at=CASE WHEN ? THEN excluded.updated_at ELSE stream_state.last_recovery_at END,
                     disconnect_count=stream_state.disconnect_count + ?,
                     recovery_count=stream_state.recovery_count + ?,
                     updated_at=excluded.updated_at""",
                (
                    stream_name, int(connected), int(recovering), last_sequence, event_at,
                    now if recovery_completed else None, int(disconnected),
                    int(recovery_completed), now, int(recovery_completed),
                    int(disconnected), int(recovery_completed),
                ),
            )
            if disconnected:
                self.ledger._audit(
                    cur,
                    "stream_disconnected",
                    "stream",
                    stream_name,
                    {"connected": connected, "recovering": recovering},
                )
            if recovery_completed:
                self.ledger._audit(
                    cur,
                    "stream_recovery_completed",
                    "stream",
                    stream_name,
                    {"connected": connected, "recovering": recovering},
                )
            return dict(cur.execute(
                "SELECT * FROM stream_state WHERE stream_name=?", (stream_name,)
            ).fetchone())

    def stream_state(self, stream_name: str = PHASE4_STREAM_NAME) -> dict | None:
        row = self.ledger.conn.execute(
            "SELECT * FROM stream_state WHERE stream_name=?", (stream_name,)
        ).fetchone()
        return dict(row) if row is not None else None

    def record_backup(
        self, backup_id: str, ledger_hash: str, manifest_hash: str, backup_path: str
    ) -> dict:
        now = self.ledger.clock().isoformat()
        with self.ledger._tx() as cur:
            cur.execute(
                """INSERT INTO backups(
                       backup_id, ledger_hash, manifest_hash, backup_path, verified, created_at
                   ) VALUES (?, ?, ?, ?, 1, ?)""",
                (backup_id, ledger_hash, manifest_hash, backup_path, now),
            )
            self.ledger._audit(
                cur, "backup_verified", "backup", backup_id,
                {"ledger_hash": ledger_hash, "manifest_hash": manifest_hash},
            )
            return dict(cur.execute(
                "SELECT * FROM backups WHERE backup_id=?", (backup_id,)
            ).fetchone())

    def latest_backup(self) -> dict | None:
        row = self.ledger.conn.execute(
            "SELECT * FROM backups ORDER BY created_at DESC, backup_id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row is not None else None

    def record_soak_observation(
        self, trading_date: str, metric: str, value: Decimal | int | str, detail: dict | None = None
    ) -> dict:
        try:
            normalized_date = date.fromisoformat(trading_date).isoformat()
        except (TypeError, ValueError) as exc:
            raise ValueError("Soak observation trading_date must be ISO format") from exc
        normalized_metric = metric.strip()
        if normalized_metric not in SOAK_METRICS:
            raise ValueError("Unsupported soak observation metric")
        normalized_value = as_decimal(value)
        if normalized_metric in {"target_weight_error_bps", "process_uptime_seconds"}:
            if normalized_value < ZERO:
                raise ValueError(f"{normalized_metric} cannot be negative")
        if detail is not None and not isinstance(detail, dict):
            raise ValueError("Soak observation detail must be an object")
        observation_id = f"soak-{uuid.uuid4().hex[:24]}"
        now = self.ledger.clock().isoformat()
        with self.ledger._tx() as cur:
            cur.execute(
                """INSERT INTO soak_observations(
                       observation_id, trading_date, metric, value, detail_json, recorded_at
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    observation_id, normalized_date, normalized_metric, str(normalized_value),
                    canonical_bytes(detail or {}).decode("utf-8"), now,
                ),
            )
            return dict(cur.execute(
                "SELECT * FROM soak_observations WHERE observation_id=?", (observation_id,)
            ).fetchone())

    def soak_observations(self) -> list[dict]:
        return [dict(row) for row in self.ledger.conn.execute(
            "SELECT * FROM soak_observations ORDER BY trading_date, recorded_at"
        ).fetchall()]
