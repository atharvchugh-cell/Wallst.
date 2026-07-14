"""Restart-safe, exchange-calendar-driven monthly Phase-4 scheduler."""

from __future__ import annotations

import calendar as month_calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Protocol

from .alerts import AlertManager
from .market_data import MarketCalendarDay
from .models import ensure_aware
from .phase4_models import Phase4Error
from .phase4_store import Phase4Store


class ExchangeCalendarSource(Protocol):
    def get_market_calendar(self, start: date, end: date) -> tuple[MarketCalendarDay, ...]: ...


@dataclass(frozen=True)
class ScheduledDecision:
    decision_day: MarketCalendarDay
    execution_day: MarketCalendarDay
    catch_up_required: bool
    delay_seconds: float


class SupervisedMonthlyScheduler:
    def __init__(
        self,
        source: ExchangeCalendarSource,
        store: Phase4Store,
        *,
        alerts: AlertManager | None = None,
        clock=None,
    ) -> None:
        self.source = source
        self.store = store
        self.alerts = alerts
        self.clock = clock or store.ledger.clock

    def _range(self, start: date, end: date) -> tuple[MarketCalendarDay, ...]:
        rows: list[MarketCalendarDay] = []
        cursor = start
        while cursor <= end:
            chunk_end = min(end, cursor + timedelta(days=7))
            rows.extend(self.source.get_market_calendar(cursor, chunk_end))
            cursor = chunk_end + timedelta(days=1)
        dates = [row.trading_date for row in rows]
        if dates != sorted(dates) or len(dates) != len(set(dates)):
            raise Phase4Error("Exchange calendar returned duplicate or unsorted sessions")
        return tuple(rows)

    def for_month(self, year: int, month: int, *, now: datetime | None = None) -> ScheduledDecision:
        current = ensure_aware(now or self.clock(), "scheduler time")
        last_day = month_calendar.monthrange(year, month)[1]
        start = date(year, month, 1)
        end = date(year, month, last_day)
        sessions = self._range(start, end)
        if not sessions:
            raise Phase4Error(f"Exchange calendar has no sessions for {year:04d}-{month:02d}")
        decision = sessions[-1]
        next_rows = self._range(end + timedelta(days=1), end + timedelta(days=14))
        if not next_rows:
            raise Phase4Error("Exchange calendar cannot identify the next regular session")
        execution = next_rows[0]
        if current <= decision.close_at:
            raise Phase4Error("Monthly decision is not due before the official close")
        delay = (current - decision.close_at).total_seconds()
        return ScheduledDecision(
            decision_day=decision,
            execution_day=execution,
            catch_up_required=current >= execution.open_at,
            delay_seconds=delay,
        )

    def latest_due(self, *, now: datetime | None = None) -> ScheduledDecision:
        current = ensure_aware(now or self.clock(), "scheduler time")
        candidates = [(current.year, current.month)]
        previous = (date(current.year, current.month, 1) - timedelta(days=1))
        candidates.append((previous.year, previous.month))
        for year, month in candidates:
            try:
                return self.for_month(year, month, now=current)
            except Phase4Error as exc:
                if "not due" not in str(exc):
                    raise
        raise Phase4Error("No completed monthly decision session is due")

    def claim_due(
        self, *, confirm_manual_catch_up: bool = False, now: datetime | None = None
    ) -> tuple[ScheduledDecision, dict, bool]:
        current = ensure_aware(now or self.clock(), "scheduler claim time")
        due = self.latest_due(now=current)
        decision_date = due.decision_day.trading_date.isoformat()
        execution_date = due.execution_day.trading_date.isoformat()
        status = "due"
        detail = "official close complete; publication claimed"
        execution_window_missed = current >= due.execution_day.close_at
        if execution_window_missed:
            status = "failed"
            detail = (
                "expected next-session execution window has closed; late publication is forbidden"
            )
        elif due.catch_up_required and not confirm_manual_catch_up:
            status = "delayed"
            detail = "missed scheduled publication; explicit manual catch-up confirmation required"
        row, created = self.store.claim_schedule(
            decision_date, execution_date, status=status, detail=detail
        )
        if row["status"] == "published":
            raise Phase4Error(f"Decision session already published as {row['snapshot_id']}")
        if execution_window_missed:
            if row["status"] != "failed":
                row = self.store.update_schedule(row["run_id"], "failed", detail=detail)
            if self.alerts:
                self.alerts.emit(
                    "critical", "scheduler_missed_run", detail,
                    entity_id=decision_date, dedupe_key=f"scheduler-missed:{decision_date}",
                )
            raise Phase4Error(detail)
        if due.catch_up_required and not confirm_manual_catch_up:
            if self.alerts:
                self.alerts.emit(
                    "critical", "scheduler_missed_run", detail,
                    entity_id=decision_date, dedupe_key=f"scheduler-missed:{decision_date}",
                )
            raise Phase4Error(detail)
        if row["status"] == "delayed":
            row = self.store.update_schedule(
                row["run_id"], "due", detail="operator-confirmed manual catch-up"
            )
        return due, row, created

    def mark_published(self, run_id: str, snapshot_id: str) -> dict:
        return self.store.update_schedule(
            run_id, "published", detail="immutable target snapshot published",
            snapshot_id=snapshot_id,
        )

    def mark_failed(self, run_id: str, exc: Exception) -> dict:
        detail = f"{type(exc).__name__}: {str(exc)[:800]}"
        if self.alerts:
            self.alerts.emit(
                "critical", "scheduler_publication_failure", detail,
                entity_id=run_id, dedupe_key=f"scheduler-failure:{run_id}",
            )
        return self.store.update_schedule(run_id, "failed", detail=detail)

    def mark_skipped(self, run_id: str, *, operator: str, reason: str) -> dict:
        operator = operator.strip()
        reason = reason.strip()
        if not operator or not reason:
            raise ValueError("Skipping a scheduled publication requires operator and reason")
        return self.store.update_schedule(
            run_id, "skipped", detail=f"{operator[:100]}: {reason[:800]}"
        )

    def next_expected_action(self, *, now: datetime | None = None) -> dict:
        current = ensure_aware(now or self.clock(), "scheduler time")
        year, month = current.year, current.month
        last_day = month_calendar.monthrange(year, month)[1]
        sessions = self._range(date(year, month, 1), date(year, month, last_day))
        if not sessions:
            return {"action": "calendar_error", "at": None}
        decision = sessions[-1]
        return {
            "action": "publish_monthly_targets_after_official_close",
            "decision_session": decision.trading_date.isoformat(),
            "at": decision.close_at.isoformat(),
        }
