"""Restart-safe, exchange-calendar-driven monthly Phase-4 scheduler."""

from __future__ import annotations

import calendar as month_calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Protocol

from .alerts import AlertManager
from .market_data import NEW_YORK, MarketCalendarDay
from .models import ensure_aware
from .phase4_models import Phase4Error
from .phase4_store import Phase4Store


MAX_RESTART_CATCH_UP_MONTHS = 24


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
        exchange_day = current.astimezone(NEW_YORK).date()
        runs = self.store.list_schedule_runs()
        latest_recorded: date | None = None
        latest_recorded_row: dict | None = None
        outstanding = next((
            row for row in runs if row["status"] not in {"published", "skipped"}
        ), None)
        if outstanding is not None:
            try:
                pending = date.fromisoformat(outstanding["decision_session"])
            except (TypeError, ValueError) as exc:
                raise Phase4Error("Stored scheduler decision session is invalid") from exc
            return self.for_month(pending.year, pending.month, now=current)

        if runs:
            try:
                dated_runs = sorted(
                    (
                        date.fromisoformat(row["decision_session"]),
                        row,
                    )
                    for row in runs
                )
            except (TypeError, ValueError) as exc:
                raise Phase4Error("Stored scheduler decision session is invalid") from exc
            latest_recorded, latest_recorded_row = dated_runs[-1]
            candidate = self._following_month(
                latest_recorded.year, latest_recorded.month
            )
            # A later terminal row must not hide a hole between two recorded
            # months. Walk adjacent durable months and select the earliest gap
            # after the ledger's defensible scheduler inception.
            for (left, _left_row), (right, _right_row) in zip(
                dated_runs, dated_runs[1:]
            ):
                distance = (right.year - left.year) * 12 + right.month - left.month
                if distance > 1:
                    if distance - 1 > MAX_RESTART_CATCH_UP_MONTHS:
                        raise Phase4Error(
                            "Scheduler history gap exceeds the bounded monthly catch-up scan; "
                            "resolve the missing history explicitly"
                        )
                    candidate = self._following_month(left.year, left.month)
                    break
            distance_to_current = (
                (exchange_day.year - candidate[0]) * 12
                + exchange_day.month
                - candidate[1]
            )
            if distance_to_current >= MAX_RESTART_CATCH_UP_MONTHS:
                raise Phase4Error(
                    "Scheduler downtime exceeds the bounded monthly catch-up scan; "
                    "resolve the missing history explicitly"
                )
            candidates = []
            if candidate <= (exchange_day.year, exchange_day.month):
                candidates.append(candidate)
        else:
            # With no durable scheduling history there is no defensible system
            # inception date. Preserve the fresh-ledger behavior: consider only
            # this exchange-local month and the immediately preceding month.
            candidates = [(exchange_day.year, exchange_day.month)]
            previous = date(exchange_day.year, exchange_day.month, 1) - timedelta(days=1)
            candidates.append((previous.year, previous.month))
        for year, month in candidates:
            try:
                return self.for_month(year, month, now=current)
            except Phase4Error as exc:
                if "not due" not in str(exc):
                    raise
        if (
            latest_recorded is not None
            and latest_recorded_row is not None
            and latest_recorded_row["status"] == "published"
        ):
            # Preserve duplicate-publication detection when every month after
            # the latest terminal run is still in the future. claim_due() will
            # re-read that durable row and report "already published".
            try:
                return self.for_month(
                    latest_recorded.year, latest_recorded.month, now=current
                )
            except Phase4Error as exc:
                if "not due" not in str(exc):
                    raise
        raise Phase4Error("No completed monthly decision session is due")

    @staticmethod
    def _following_month(year: int, month: int) -> tuple[int, int]:
        return (year + 1, 1) if month == 12 else (year, month + 1)

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
        if row["status"] == "skipped":
            raise Phase4Error(
                f"Decision session was terminally skipped as {row['run_id']}"
            )
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
        # A durable due/delayed/failed run is an operator action until it is
        # explicitly published or skipped. Do not hide it behind a later
        # month's calendar date after a restart.
        outstanding = next((
            row for row in self.store.list_schedule_runs()
            if row["status"] not in {"published", "skipped"}
        ), None)
        if outstanding is not None:
            try:
                decision_date = date.fromisoformat(outstanding["decision_session"])
                execution_date = date.fromisoformat(
                    outstanding["expected_execution_session"]
                )
                decision_rows = self._range(decision_date, decision_date)
                execution_rows = self._range(execution_date, execution_date)
                if len(decision_rows) != 1 or len(execution_rows) != 1:
                    raise Phase4Error("Stored schedule sessions are absent from the exchange calendar")
                due = ScheduledDecision(
                    decision_day=decision_rows[0],
                    execution_day=execution_rows[0],
                    catch_up_required=current >= execution_rows[0].open_at,
                    delay_seconds=(current - decision_rows[0].close_at).total_seconds(),
                )
            except (TypeError, ValueError, Phase4Error):
                return {
                    "action": "calendar_error",
                    "at": None,
                    "decision_session": outstanding["decision_session"],
                    "scheduler_status": outstanding["status"],
                }
            return self._due_action(due, current, outstanding["status"])

        due = None
        try:
            due = self.latest_due(now=current)
        except Phase4Error as exc:
            if "No completed monthly decision session is due" not in str(exc):
                raise
        if due is not None:
            decision_session = due.decision_day.trading_date.isoformat()
            run = next((
                row for row in self.store.list_schedule_runs()
                if row["decision_session"] == decision_session
            ), None)
            if run is None:
                return self._due_action(due, current, None)

        exchange_day = current.astimezone(NEW_YORK).date()
        year, month = exchange_day.year, exchange_day.month
        if (
            due is not None
            and (due.decision_day.trading_date.year, due.decision_day.trading_date.month)
            == (year, month)
        ):
            year, month = self._following_month(year, month)
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

    @staticmethod
    def _due_action(
        due: ScheduledDecision, current: datetime, scheduler_status: str | None
    ) -> dict:
        if current >= due.execution_day.close_at:
            action = "investigate_missed_monthly_run"
        elif scheduler_status == "failed":
            action = "retry_failed_monthly_publication"
        elif current >= due.execution_day.open_at or scheduler_status == "delayed":
            action = "confirm_manual_catch_up_and_publish"
        else:
            action = "publish_monthly_targets_after_official_close"
        return {
            "action": action,
            "decision_session": due.decision_day.trading_date.isoformat(),
            "expected_execution_session": due.execution_day.trading_date.isoformat(),
            "scheduler_status": scheduler_status or "unclaimed",
            "at": due.decision_day.close_at.isoformat(),
        }
