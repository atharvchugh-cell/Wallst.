"""Focused regressions for Phase-4 publication and operational failure paths."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, time, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.live import phase4_cli
from src.live.ledger import Ledger, LedgerConflict
from src.live.market_data import MarketCalendarDay, NEW_YORK
from src.live.phase4_models import OperationMode, Phase4Error
from src.live.phase4_store import Phase4Store
from src.live.publisher import StrategyTargetPublisher
from src.live.scheduler import SupervisedMonthlyScheduler
from tests.test_live_phase4 import (
    DECISION,
    PUBLISH_TIME,
    deployment,
    market_bundle,
    policy,
    signer,
)


class WorkflowFailure(RuntimeError):
    pass


class BackupFailure(RuntimeError):
    pass


def _args(tmp_path, **overrides):
    values = {
        "db": str(tmp_path / "ledger.sqlite3"),
        "deployment": str(tmp_path / "deployment.json"),
        "policy": str(tmp_path / "policy.json"),
        "cache_dir": str(tmp_path / "cache"),
        "snapshot_dir": str(tmp_path / "snapshots"),
        "snapshot_id": "snapshot-test",
        "batch_id": "batch-test",
        "plan_hash": "a" * 64,
        "operator": "red-team",
        "reason": "injected failure",
        "mode": None,
        "confirm_manual_catch_up": False,
        "confirm_new_equity_session": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _common_cli_fakes(monkeypatch):
    deployment_value = SimpleNamespace(account_id="PAPER")
    policy_value = SimpleNamespace(
        mode=OperationMode.PAPER_MANUAL,
        system_id="red-team-system",
    )
    monkeypatch.setattr(
        phase4_cli,
        "_load",
        lambda *_args, **_kwargs: (deployment_value, policy_value, None),
    )
    monkeypatch.setattr(phase4_cli, "_alerts", lambda _store: SimpleNamespace())
    monkeypatch.setattr(
        phase4_cli,
        "_paper_stack",
        lambda: (SimpleNamespace(), SimpleNamespace()),
    )
    monkeypatch.setattr(
        phase4_cli,
        "_publisher",
        lambda *_args, **_kwargs: SimpleNamespace(
            to_execution_targets=lambda _snapshot: ()
        ),
    )
    return deployment_value, policy_value


def _mutation_probe(monkeypatch, calls, *, backup_error=False, expected_status=None):
    def backup(ledger, _policy, _args, _alerts):
        events = ledger.list_audit_events()
        if expected_status is None:
            assert any(row["event_type"] == "redteam_mutation" for row in events)
        else:
            row = ledger.conn.execute(
                "SELECT status FROM scheduler_runs ORDER BY created_at LIMIT 1"
            ).fetchone()
            assert row is not None and row["status"] == expected_status
        calls.append(expected_status or "workflow")
        if backup_error:
            raise BackupFailure("backup also exploded")
        return {"backup_id": "backup-test"}

    monkeypatch.setattr(phase4_cli, "_automatic_backup", backup)


def test_publisher_rejects_carried_ohlc_even_when_final_volume_changes(tmp_path):
    bundle = market_bundle()
    symbol = sorted(bundle.frames)[0]
    frame = bundle.frames[symbol].copy()
    frame.loc[frame.index[-1], ["Open", "High", "Low", "Close"]] = frame.loc[
        frame.index[-2], ["Open", "High", "Low", "Close"]
    ].to_numpy()
    frame.loc[frame.index[-1], "Volume"] = frame.loc[frame.index[-2], "Volume"] + 17
    bundle.frames[symbol] = frame
    publisher = StrategyTargetPublisher(
        deployment(), policy(), repo_root=Path.cwd(), signer=signer(tmp_path / "key"),
        clock=lambda: PUBLISH_TIME,
    )
    decision_day = MarketCalendarDay(
        DECISION,
        datetime.combine(DECISION, time(9, 30), tzinfo=NEW_YORK),
        datetime.combine(DECISION, time(16, 0), tzinfo=NEW_YORK),
    )

    with pytest.raises(Phase4Error, match="carried-forward final bar"):
        publisher._validate_market_data(bundle, decision_day, PUBLISH_TIME)


class _MutatingSupervisor:
    def __init__(self, ledger, *_args, **_kwargs):
        self.ledger = ledger

    def _fail(self, workflow):
        self.ledger.record_audit(
            "redteam_mutation", "phase4_workflow", workflow, {"committed": True}
        )
        raise WorkflowFailure(f"{workflow} exploded")

    def prepare_plan(self, *_args, **_kwargs):
        self._fail("prepare-plan")

    def run_paper(self, *_args, **_kwargs):
        self._fail("run-paper")


def test_prepare_failure_attempts_backup_and_preserves_original_if_backup_fails(
    monkeypatch, tmp_path
):
    _common_cli_fakes(monkeypatch)
    monkeypatch.setattr(phase4_cli, "Phase4Supervisor", _MutatingSupervisor)
    calls = []
    _mutation_probe(monkeypatch, calls, backup_error=True)
    args = _args(tmp_path)

    with pytest.raises(WorkflowFailure, match="prepare-plan exploded") as caught:
        phase4_cli._prepare(args)

    assert calls == ["workflow"]
    assert isinstance(caught.value.__cause__, BackupFailure)
    with Ledger(args.db) as ledger:
        assert any(
            row["event_type"] == "automatic_backup_failed_after_workflow_error"
            for row in ledger.list_audit_events()
        )


def test_run_paper_failure_after_mutation_attempts_backup(monkeypatch, tmp_path):
    _common_cli_fakes(monkeypatch)
    monkeypatch.setattr(phase4_cli, "Phase4Supervisor", _MutatingSupervisor)
    calls = []
    _mutation_probe(monkeypatch, calls)

    with pytest.raises(WorkflowFailure, match="run-paper exploded"):
        phase4_cli._run_paper(_args(tmp_path))

    assert calls == ["workflow"]


def test_reconciliation_failure_after_mutation_attempts_backup(monkeypatch, tmp_path):
    _common_cli_fakes(monkeypatch)

    class MutatingReconciler:
        def __init__(self, ledger, _broker):
            self.ledger = ledger

        def reconcile(self):
            self.ledger.record_audit(
                "redteam_mutation", "phase4_workflow", "reconcile", {"committed": True}
            )
            raise WorkflowFailure("reconcile exploded")

    monkeypatch.setattr(phase4_cli, "Reconciler", MutatingReconciler)
    calls = []
    _mutation_probe(monkeypatch, calls)

    with pytest.raises(WorkflowFailure, match="reconcile exploded"):
        phase4_cli._reconcile(_args(tmp_path))

    assert calls == ["workflow"]


def test_approval_failure_after_mutation_attempts_backup(monkeypatch, tmp_path):
    deployment_value, _policy_value = _common_cli_fakes(monkeypatch)
    monkeypatch.setattr(phase4_cli, "account_fingerprint", lambda *_args: "fingerprint")

    class Snapshot:
        content = {"account_id_fingerprint": "fingerprint"}

        def verify(self, *_args, **_kwargs):
            return None

    class Store:
        def __init__(self, ledger):
            self.ledger = ledger

        def execution_plan_link(self, _batch_id):
            return {
                "paper_submission_allowed": 1,
                "operation_mode": OperationMode.PAPER_MANUAL.value,
                "snapshot_id": "snapshot-test",
            }

        def load_snapshot(self, _snapshot_id):
            return Snapshot()

    monkeypatch.setattr(phase4_cli, "Phase4Store", Store)

    def fail_approval(ledger, *_args, **_kwargs):
        ledger.record_audit(
            "redteam_mutation", "phase4_workflow", "approve", {"committed": True}
        )
        raise WorkflowFailure("approve exploded")

    monkeypatch.setattr(Ledger, "approve_phase4_execution_batch", fail_approval)
    calls = []
    _mutation_probe(monkeypatch, calls)

    with pytest.raises(WorkflowFailure, match="approve exploded"):
        phase4_cli._approve(_args(tmp_path))

    assert deployment_value.account_id == "PAPER"
    assert calls == ["workflow"]


def test_publish_claim_failure_is_backed_up(monkeypatch, tmp_path):
    _common_cli_fakes(monkeypatch)

    class ClaimThenFail:
        def __init__(self, _source, store, **_kwargs):
            self.store = store

        def claim_due(self, **_kwargs):
            self.store.claim_schedule(
                "2026-06-30", "2026-07-01", status="delayed", detail="operator needed"
            )
            raise WorkflowFailure("claim exploded")

    monkeypatch.setattr(phase4_cli, "SupervisedMonthlyScheduler", ClaimThenFail)
    calls = []
    _mutation_probe(monkeypatch, calls, expected_status="delayed")

    with pytest.raises(WorkflowFailure, match="claim exploded"):
        phase4_cli._publish(_args(tmp_path))

    assert calls == ["delayed"]


def test_publish_marks_failed_before_failure_backup(monkeypatch, tmp_path):
    _common_cli_fakes(monkeypatch)

    class DueThenFailScheduler:
        def __init__(self, _source, store, **_kwargs):
            self.store = store

        def claim_due(self, **_kwargs):
            row, created = self.store.claim_schedule(
                "2026-06-30", "2026-07-01", status="due"
            )
            return SimpleNamespace(), row, created

        def mark_failed(self, run_id, exc):
            return self.store.update_schedule(
                run_id, "failed", detail=f"{type(exc).__name__}: {exc}"
            )

    class AccountFailureBroker:
        def get_account(self):
            raise WorkflowFailure("account fetch exploded")

    monkeypatch.setattr(
        phase4_cli, "_paper_stack", lambda: (AccountFailureBroker(), SimpleNamespace())
    )
    monkeypatch.setattr(
        phase4_cli, "SupervisedMonthlyScheduler", DueThenFailScheduler
    )
    calls = []
    _mutation_probe(monkeypatch, calls, expected_status="failed")

    with pytest.raises(WorkflowFailure, match="account fetch exploded"):
        phase4_cli._publish(_args(tmp_path))

    assert calls == ["failed"]


def test_publish_finalizes_persisted_snapshot_before_failure_backup(monkeypatch, tmp_path):
    _common_cli_fakes(monkeypatch)
    snapshot = SimpleNamespace(snapshot_id="snapshot-persisted", content_hash="b" * 64)

    class Scheduler:
        def __init__(self, _source, store, **_kwargs):
            self.store = store

        def claim_due(self, **_kwargs):
            row, created = self.store.claim_schedule(
                "2026-06-30", "2026-07-01", status="due"
            )
            due = SimpleNamespace(
                decision_day=SimpleNamespace(trading_date=date(2026, 6, 30)),
                execution_day=SimpleNamespace(trading_date=date(2026, 7, 1)),
            )
            return due, row, created

        def mark_failed(self, run_id, exc):
            return self.store.update_schedule(run_id, "failed", detail=str(exc))

        def mark_published(self, run_id, snapshot_id):
            return self.store.update_schedule(
                run_id,
                "published",
                detail="immutable target snapshot published",
                snapshot_id=snapshot_id,
            )

    class Broker:
        def get_account(self):
            return SimpleNamespace(account_id="PAPER")

        def get_positions(self):
            return []

    class Publisher:
        def required_data_symbols(self):
            return ()

        def publish(self, **_kwargs):
            return snapshot

    class PersistThenFail:
        def __init__(self, ledger, *_args, **_kwargs):
            self.ledger = ledger

        def persist_snapshot(self, _snapshot, **_kwargs):
            now = datetime(2026, 6, 30, 20, 5, tzinfo=timezone.utc).isoformat()
            with self.ledger._tx() as cur:
                cur.execute(
                    """INSERT INTO target_snapshots(
                           snapshot_id, content_hash, decision_session,
                           expected_execution_session, account_fingerprint, mode,
                           signed, expires_at, snapshot_json, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        snapshot.snapshot_id,
                        snapshot.content_hash,
                        "2026-06-30",
                        "2026-07-01",
                        "fingerprint",
                        OperationMode.PAPER_MANUAL.value,
                        1,
                        "2026-07-02T00:00:00+00:00",
                        "{}",
                        now,
                    ),
                )
            raise WorkflowFailure("snapshot export exploded")

    monkeypatch.setattr(phase4_cli, "SupervisedMonthlyScheduler", Scheduler)
    monkeypatch.setattr(phase4_cli, "Phase4Supervisor", PersistThenFail)
    monkeypatch.setattr(phase4_cli, "_publisher", lambda *_args, **_kwargs: Publisher())
    monkeypatch.setattr(phase4_cli, "_paper_stack", lambda: (Broker(), SimpleNamespace()))
    monkeypatch.setattr(phase4_cli, "collect_assets", lambda *_args: {})
    monkeypatch.setattr(
        phase4_cli,
        "ResearchHistoricalDataSource",
        lambda **_kwargs: SimpleNamespace(load=lambda *_args: SimpleNamespace()),
    )
    monkeypatch.setattr(phase4_cli, "_sha256_file", lambda _path: "c" * 64)
    monkeypatch.setattr(Ledger, "assert_account_binding", lambda *_args: None)
    monkeypatch.setattr(Ledger, "positions_bootstrapped", lambda *_args: True)
    calls = []
    _mutation_probe(monkeypatch, calls, expected_status="published")

    with pytest.raises(WorkflowFailure, match="snapshot export exploded"):
        phase4_cli._publish(_args(tmp_path))

    assert calls == ["published"]


def test_skip_schedule_is_offline_backed_up_terminal_and_immutable(monkeypatch, tmp_path):
    _common_cli_fakes(monkeypatch)
    args = _args(tmp_path, run_id="")
    with Ledger(args.db) as ledger:
        row, _created = Phase4Store(ledger).claim_schedule(
            "2026-06-30", "2026-07-01", status="failed", detail="window missed"
        )
        args.run_id = row["run_id"]

    calls = []
    _mutation_probe(monkeypatch, calls, expected_status="skipped")
    result = phase4_cli._skip_schedule(args)
    assert result["network_used"] is False
    assert result["scheduler_run"]["status"] == "skipped"
    assert calls == ["skipped"]

    with Ledger(args.db) as ledger:
        store = Phase4Store(ledger)
        replay = SupervisedMonthlyScheduler(None, store).mark_skipped(
            args.run_id, operator=args.operator, reason=args.reason
        )
        assert replay["detail"] == f"{args.operator}: {args.reason}"
        with pytest.raises(LedgerConflict, match="skipped scheduler run"):
            SupervisedMonthlyScheduler(None, store).mark_skipped(
                args.run_id, operator=args.operator, reason="rewritten reason"
            )


def test_skipped_month_is_not_reclaimed_and_next_missing_month_advances(tmp_path):
    class Calendar:
        def get_market_calendar(self, start, end):
            rows = []
            cursor = start
            while cursor <= end:
                if cursor.weekday() < 5:
                    rows.append(
                        MarketCalendarDay(
                            cursor,
                            datetime.combine(cursor, time(9, 30), tzinfo=NEW_YORK),
                            datetime.combine(cursor, time(16, 0), tzinfo=NEW_YORK),
                        )
                    )
                cursor = date.fromordinal(cursor.toordinal() + 1)
            return tuple(rows)

    now = datetime(2026, 7, 31, 21, 0, tzinfo=timezone.utc)
    with Ledger(tmp_path / "scheduler.sqlite3", clock=lambda: now) as ledger:
        store = Phase4Store(ledger)
        june, _created = store.claim_schedule(
            "2026-06-30", "2026-07-01", status="failed", detail="window missed"
        )
        scheduler = SupervisedMonthlyScheduler(Calendar(), store, clock=lambda: now)
        scheduler.mark_skipped(june["run_id"], operator="ops", reason="investigated")

        due = scheduler.latest_due(now=now)
        assert due.decision_day.trading_date == date(2026, 7, 31)
        _due, july, created = scheduler.claim_due(
            confirm_manual_catch_up=True, now=now
        )
        assert created and july["decision_session"] == "2026-07-31"
        assert store.list_schedule_runs()[0]["status"] == "skipped"


def test_alert_resolution_safety_check_and_write_share_one_execution_guard(
    monkeypatch, tmp_path
):
    events = []
    original_guard = Ledger.execution_guard

    @contextmanager
    def tracked_guard(ledger):
        events.append("enter")
        with original_guard(ledger):
            yield
        events.append("exit")

    def safe(ledger, _store, _alert_id):
        assert getattr(ledger._execution_guard_local, "depth", 0) == 1
        events.append("safe")

    def resolve(store, alert_id, **_kwargs):
        assert getattr(store.ledger._execution_guard_local, "depth", 0) == 1
        events.append("resolve")
        return {"alert_id": alert_id, "resolved_at": "now"}

    monkeypatch.setattr(Ledger, "execution_guard", tracked_guard)
    monkeypatch.setattr(phase4_cli, "_assert_alert_resolution_safe", safe)
    monkeypatch.setattr(Phase4Store, "resolve_alert", resolve)
    monkeypatch.setattr(phase4_cli, "_print", lambda _payload: None)

    result = phase4_cli.main(
        [
            "alerts",
            "--db",
            str(tmp_path / "alerts.sqlite3"),
            "--resolve",
            "alert-test",
            "--operator",
            "ops",
            "--note",
            "reviewed",
            "--confirm-alert-change",
        ]
    )

    assert result == 0
    assert events == ["enter", "safe", "resolve", "exit"]
