"""Value-object and durable-ledger invariants for the phase-one foundation."""

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import os
import sqlite3
import stat
import threading

import pytest

from src.live.fake_broker import FakeBroker
from src.live.ledger import Ledger, LedgerConflict
from src.live.models import OrderRequest, Side, TargetPositionIntent
from src.live.oms import OrderManagementSystem
from src.live.reconcile import Reconciler
from src.live.risk import PreTradeRiskEngine


NOW = datetime(2026, 7, 14, 14, 0, tzinfo=timezone.utc)


class FixedClock:
    def __init__(self, now=NOW):
        self.now = now

    def __call__(self):
        return self.now


def make_intent(target="10", version="v1", *, reference="100"):
    return TargetPositionIntent(
        account_id="TEST",
        strategy_id="aggregated-portfolio",
        symbol="aapl",
        target_quantity=Decimal(target),
        signal_at=NOW,
        target_version=version,
        reference_price=Decimal(reference),
        reason="test target",
    )


def test_target_intent_is_normalized_immutable_and_deterministic():
    first = make_intent()
    second = make_intent()
    assert first.symbol == "AAPL"
    assert first.idempotency_key == second.idempotency_key
    with pytest.raises(Exception):
        first.symbol = "MSFT"


def test_target_intent_rejects_naive_time_negative_target_and_bad_price():
    with pytest.raises(ValueError, match="timezone-aware"):
        TargetPositionIntent("A", "S", "SPY", Decimal("1"), datetime(2026, 1, 1), "v1", Decimal("1"))
    with pytest.raises(ValueError, match="cannot be negative"):
        TargetPositionIntent("A", "S", "SPY", Decimal("-1"), NOW, "v1", Decimal("1"))
    with pytest.raises(ValueError, match="positive"):
        TargetPositionIntent("A", "S", "SPY", Decimal("1"), NOW, "v1", Decimal("0"))
    with pytest.raises(ValueError, match="finite"):
        TargetPositionIntent("A", "S", "SPY", Decimal("NaN"), NOW, "v1", Decimal("1"))


def test_same_idempotency_key_with_changed_payload_is_rejected():
    ledger = Ledger(":memory:", clock=FixedClock())
    ledger.create_intent(make_intent("10"))
    with pytest.raises(LedgerConflict, match="increment target_version"):
        ledger.create_intent(make_intent("11"))


def test_ledger_enforces_only_one_active_order_per_account_symbol():
    ledger = Ledger(":memory:", clock=FixedClock())
    first, _ = ledger.create_intent(make_intent("10", "v1"))
    second, _ = ledger.create_intent(make_intent("8", "v2"))
    ledger.plan_order(OrderRequest(
        "TEST", "wslab-first", first["intent_id"], "AAPL", Side.BUY,
        Decimal("10"), Decimal("100"),
    ))
    with pytest.raises(LedgerConflict, match="already exists"):
        ledger.plan_order(OrderRequest(
            "TEST", "wslab-second", second["intent_id"], "AAPL", Side.BUY,
            Decimal("8"), Decimal("100"),
        ))


def test_only_the_transaction_that_inserts_an_order_gets_submission_authority():
    ledger = Ledger(":memory:", clock=FixedClock())
    target, _ = ledger.create_intent(make_intent())
    request = OrderRequest(
        "TEST", "wslab-only-submitter", target["intent_id"], "AAPL", Side.BUY,
        Decimal("10"), Decimal("100"),
    )
    first, first_created = ledger.plan_order_with_created(request)
    second, second_created = ledger.plan_order_with_created(request)
    assert first["order_id"] == second["order_id"]
    assert first_created is True
    assert second_created is False


def test_competing_ledger_connections_issue_exactly_one_submission_capability(tmp_path):
    path = tmp_path / "execution.sqlite3"
    with Ledger(path, clock=FixedClock()) as setup:
        target, _ = setup.create_intent(make_intent())
    request = OrderRequest(
        "TEST", "wslab-concurrent-submitter", target["intent_id"], "AAPL", Side.BUY,
        Decimal("10"), Decimal("100"),
    )
    barrier = threading.Barrier(2)
    created_flags = []
    errors = []
    mutex = threading.Lock()

    def plan_from_connection():
        try:
            with Ledger(path, clock=FixedClock()) as ledger:
                barrier.wait(timeout=2)
                _order, created = ledger.plan_order_with_created(request)
            with mutex:
                created_flags.append(created)
        except Exception as exc:  # surfaced by the assertion below
            with mutex:
                errors.append(exc)

    threads = [threading.Thread(target=plan_from_connection) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert errors == []
    assert sorted(created_flags) == [False, True]


def test_execution_fence_serializes_control_change_across_connections(tmp_path):
    path = tmp_path / "execution.sqlite3"
    with Ledger(path):
        pass
    acquired = threading.Event()
    release = threading.Event()
    changed = threading.Event()

    def hold_submission_fence():
        with Ledger(path) as ledger:
            with ledger.execution_guard():
                acquired.set()
                assert release.wait(timeout=2)

    def change_control():
        assert acquired.wait(timeout=2)
        with Ledger(path) as ledger:
            ledger.set_control_state(
                "TEST", armed=False, kill_switch=True, reason="concurrent kill"
            )
        changed.set()

    holder = threading.Thread(target=hold_submission_fence)
    controller = threading.Thread(target=change_control)
    holder.start()
    controller.start()
    assert acquired.wait(timeout=2)
    assert changed.wait(timeout=0.1) is False
    release.set()
    holder.join(timeout=3)
    controller.join(timeout=3)
    assert changed.is_set()


def test_position_baseline_is_explicit_and_cannot_be_rebased():
    clock = FixedClock()
    broker = FakeBroker(account_id="TEST", cash="10000", clock=clock)
    broker.set_quote("AAPL", "100", spread_bps="0", as_of=NOW)
    broker.seed_position("AAPL", "3", "90")
    ledger = Ledger(":memory:", clock=clock)
    reconciler = Reconciler(ledger, broker, clock=clock)
    reconciler.bootstrap_positions()
    assert ledger.positions_bootstrapped("TEST")
    assert ledger.list_positions("TEST")[0].quantity == Decimal("3")
    with pytest.raises(LedgerConflict, match="already exists"):
        reconciler.bootstrap_positions()


def test_fractional_opening_position_is_rejected_by_whole_share_boundary():
    clock = FixedClock()
    broker = FakeBroker(account_id="TEST", cash="10000", clock=clock)
    broker.seed_position("AAPL", "1.5", "100")
    ledger = Ledger(":memory:", clock=clock)
    with pytest.raises(LedgerConflict, match="Fractional"):
        Reconciler(ledger, broker, clock=clock).bootstrap_positions()


def test_file_ledger_permissions_are_owner_only(tmp_path):
    path = tmp_path / "execution.sqlite3"
    with Ledger(path):
        pass
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_file_ledger_survives_reopen_with_controls_fills_and_positions(tmp_path):
    clock = FixedClock()
    path = tmp_path / "execution.sqlite3"
    broker = FakeBroker(account_id="TEST", cash="10000", clock=clock)
    quote = broker.set_quote("AAPL", "100", spread_bps="0", as_of=NOW)
    ledger = Ledger(path, clock=clock)
    reconciler = Reconciler(ledger, broker, clock=clock)
    reconciler.bootstrap_positions()
    assert reconciler.reconcile().clean
    oms = OrderManagementSystem(ledger, broker, PreTradeRiskEngine(clock=clock), clock=clock)
    oms.arm("test")
    oms.process_intent(
        make_intent(), quote=quote, market_open=True,
        day_start_equity=Decimal("10000"), high_water_equity=Decimal("10000"),
    )
    ledger.close()

    reopened = Ledger(path, clock=clock)
    assert reopened.get_control_state("TEST")["armed"] is True
    assert reopened.list_positions("TEST")[0].quantity == Decimal("10")
    assert reopened.expected_cash("TEST") == Decimal("9000")
    assert len(reopened.list_fills()) == 1
    assert reopened.list_intents()[0]["status"] == "filled"
    assert any(e["event_type"] == "fill_recorded" for e in reopened.list_audit_events())


def test_replaying_a_broker_fill_does_not_move_position_twice():
    clock = FixedClock()
    broker = FakeBroker(account_id="TEST", cash="10000", clock=clock)
    quote = broker.set_quote("AAPL", "100", spread_bps="0", as_of=NOW)
    ledger = Ledger(":memory:", clock=clock)
    reconciler = Reconciler(ledger, broker, clock=clock)
    reconciler.bootstrap_positions()
    reconciler.reconcile()
    oms = OrderManagementSystem(ledger, broker, PreTradeRiskEngine(clock=clock), clock=clock)
    oms.arm("test")
    result = oms.process_intent(
        make_intent(), quote=quote, market_open=True,
        day_start_equity=Decimal("10000"), high_water_equity=Decimal("10000"),
    )
    fill = broker.get_fills()[0]
    assert ledger.record_fill(result.order_id, fill) is False
    with pytest.raises(LedgerConflict, match="changed fields: price"):
        ledger.record_fill(result.order_id, replace(fill, price=Decimal("101")))
    assert ledger.list_positions("TEST")[0].quantity == Decimal("10")
    with pytest.raises(LedgerConflict, match="exceeds the ordered quantity"):
        ledger.record_fill(result.order_id, replace(fill, fill_id="bad-overfill", quantity=Decimal("1")))


def test_audit_events_are_append_only_at_database_layer():
    ledger = Ledger(":memory:", clock=FixedClock())
    ledger.record_audit("test_event", "test", "one", {"safe": True})
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        ledger.conn.execute("UPDATE audit_events SET entity_id = 'tampered'")
    ledger.conn.rollback()
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        ledger.conn.execute("DELETE FROM audit_events")
