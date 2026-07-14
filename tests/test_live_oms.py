"""OMS lifecycle, idempotency, restart, partial-fill, and kill-switch tests."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import threading

import pytest

from src.live.broker import BrokerError
from src.live.fake_broker import FakeBroker
from src.live.ledger import Ledger, LedgerConflict
from src.live.models import IntentStatus, OrderRequest, Side, TargetPositionIntent
from src.live.oms import (
    ExecutionBlocked,
    KillSwitchError,
    OrderManagementSystem,
    ReconciliationRequired,
)
from src.live.reconcile import Reconciler
from src.live.risk import PreTradeRiskEngine, RiskLimits


class ManualClock:
    def __init__(self):
        self.now = datetime(2026, 7, 14, 14, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


def intent(clock, target="10", version="v1", reference="100"):
    return TargetPositionIntent(
        "TEST", "aggregated-portfolio", "AAPL", Decimal(target), clock(), version,
        Decimal(reference), "test",
    )


def armed_stack(tmp_path=None, *, auto_fill=True, broker_class=FakeBroker):
    clock = ManualClock()
    broker = broker_class(account_id="TEST", cash="10000", auto_fill=auto_fill, clock=clock)
    quote = broker.set_quote("AAPL", "100", spread_bps="0", as_of=clock())
    path = tmp_path / "ledger.sqlite3" if tmp_path else ":memory:"
    ledger = Ledger(path, clock=clock)
    reconciler = Reconciler(ledger, broker, clock=clock)
    reconciler.bootstrap_positions()
    assert reconciler.reconcile().clean
    oms = OrderManagementSystem(ledger, broker, PreTradeRiskEngine(clock=clock), clock=clock)
    oms.arm("test arm")
    return clock, broker, quote, ledger, reconciler, oms


def submit(oms, target_intent, quote):
    return oms.process_intent(
        target_intent,
        quote=quote,
        market_open=True,
        day_start_equity=Decimal("10000"),
        high_water_equity=Decimal("10000"),
    )


def test_successful_target_is_filled_and_audited():
    clock, broker, quote, ledger, _reconciler, oms = armed_stack()
    result = submit(oms, intent(clock), quote)
    assert result.intent_status == IntentStatus.FILLED
    assert result.client_order_id.startswith("wslab-")
    assert broker.submission_count == 1
    assert ledger.list_positions("TEST")[0].quantity == Decimal("10")
    assert len(ledger.list_fills()) == 1


def test_duplicate_intent_returns_existing_result_without_second_order():
    clock, broker, quote, _ledger, _reconciler, oms = armed_stack()
    target = intent(clock)
    first = submit(oms, target, quote)
    second = submit(oms, target, quote)
    assert first.broker_order_id == second.broker_order_id
    assert second.duplicate_intent is True
    assert broker.submission_count == 1


def test_target_equal_to_broker_position_is_noop():
    clock = ManualClock()
    broker = FakeBroker(account_id="TEST", cash="9000", clock=clock)
    quote = broker.set_quote("AAPL", "100", spread_bps="0", as_of=clock())
    broker.seed_position("AAPL", "10", "100")
    ledger = Ledger(":memory:", clock=clock)
    rec = Reconciler(ledger, broker, clock=clock)
    rec.bootstrap_positions()
    rec.reconcile()
    oms = OrderManagementSystem(ledger, broker, PreTradeRiskEngine(clock=clock), clock=clock)
    oms.arm("test")
    result = submit(oms, intent(clock), quote)
    assert result.intent_status == IntentStatus.NOOP
    assert broker.submission_count == 0


def test_new_target_version_reduces_an_existing_position():
    clock, broker, quote, ledger, reconciler, oms = armed_stack()
    submit(oms, intent(clock, "10", "v1"), quote)
    assert reconciler.reconcile().clean
    reduced = submit(oms, intent(clock, "4", "v2"), quote)
    assert reduced.intent_status == IntentStatus.FILLED
    assert ledger.list_positions("TEST")[0].quantity == Decimal("4")
    assert broker.get_positions()[0].quantity == Decimal("4")


def test_lost_acknowledgement_recovers_by_same_client_id_without_duplicate():
    clock, broker, quote, ledger, _reconciler, oms = armed_stack()
    broker.raise_after_submit_once = True
    target = intent(clock)
    with pytest.raises(BrokerError, match="acknowledgement loss"):
        submit(oms, target, quote)
    assert broker.submission_count == 1
    assert ledger.list_positions("TEST") == []

    recovered = submit(oms, target, quote)
    assert recovered.intent_status == IntentStatus.FILLED
    assert recovered.duplicate_intent is True
    assert broker.submission_count == 1
    assert ledger.list_positions("TEST")[0].quantity == Decimal("10")


def test_lost_ack_recovery_never_scans_lifetime_fill_history():
    class BoundedFillBroker(FakeBroker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fill_queries = []

        def get_fills(self, since=None):
            if since is None:
                raise BrokerError("lifetime fill scans are forbidden")
            self.fill_queries.append(since)
            return super().get_fills(since)

    clock, broker, quote, ledger, _reconciler, oms = armed_stack(
        broker_class=BoundedFillBroker
    )
    broker.raise_after_submit_once = True
    target = intent(clock)
    with pytest.raises(BrokerError, match="acknowledgement loss"):
        submit(oms, target, quote)

    recovered = submit(oms, target, quote)
    assert recovered.intent_status == IntentStatus.FILLED
    assert broker.submission_count == 1
    assert ledger.list_positions("TEST")[0].quantity == Decimal("10")
    assert broker.fill_queries
    assert all(since is not None for since in broker.fill_queries)


def test_recovery_never_resubmits_a_missing_order_after_disarm_and_allows_audited_abandon():
    class NeverAcceptBroker(FakeBroker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.submit_attempts = 0

        def submit_order(self, request):
            self.submit_attempts += 1
            raise BrokerError("simulated broker outage before acceptance")

    clock, broker, quote, ledger, _reconciler, oms = armed_stack(broker_class=NeverAcceptBroker)
    with pytest.raises(BrokerError, match="outage"):
        submit(oms, intent(clock), quote)
    order = ledger.list_orders()[0]
    oms.disarm("outage review")
    recovered = oms.recover_pending()
    assert recovered[0].intent_status == IntentStatus.ORDER_PENDING
    assert broker.submit_attempts == 1
    abandoned = oms.abandon_missing_order(order["order_id"], "broker confirmed no client order")
    assert abandoned.intent_status == IntentStatus.CANCELED


def test_disarm_after_planning_fences_broker_submission():
    clock, broker, _quote, ledger, _reconciler, oms = armed_stack(auto_fill=False)
    target_row, _ = ledger.create_intent(intent(clock))
    request = OrderRequest(
        "TEST", "wslab-fenced-order", target_row["intent_id"], "AAPL", Side.BUY,
        Decimal("10"), Decimal("100"),
    )
    order, created = ledger.plan_order_with_created(request)
    assert created is True
    ledger.set_intent_status(
        target_row["intent_id"], IntentStatus.ORDER_PENDING, "test planned order"
    )
    oms.disarm("operator won the control race")
    with pytest.raises(ExecutionBlocked, match="Control state changed"):
        oms._submit_or_synchronize(order, allow_submit=True)
    assert broker.submission_count == 0
    assert "broker_submit_fenced" in {
        event["event_type"] for event in ledger.list_audit_events()
    }


def test_kill_switch_keeps_kill_engaged_when_a_broker_cancellation_fails():
    class CancelFailureBroker(FakeBroker):
        def cancel_order(self, broker_order_id):
            raise BrokerError("simulated cancellation outage")

    clock, broker, quote, ledger, _reconciler, oms = armed_stack(
        auto_fill=False, broker_class=CancelFailureBroker
    )
    submit(oms, intent(clock), quote)
    with pytest.raises(KillSwitchError, match="remains engaged"):
        oms.engage_kill_switch("operator emergency")
    assert ledger.get_control_state("TEST")["kill_switch"] is True
    assert len(broker.get_open_orders()) == 1


def test_kill_switch_records_fill_when_fill_wins_cancel_race():
    class FillBeforeCancelBroker(FakeBroker):
        def cancel_order(self, broker_order_id):
            self.fill_order(broker_order_id)
            return super().cancel_order(broker_order_id)

    clock, broker, quote, ledger, _reconciler, oms = armed_stack(
        auto_fill=False, broker_class=FillBeforeCancelBroker
    )
    result = submit(oms, intent(clock), quote)
    oms.engage_kill_switch("operator emergency")
    assert ledger.get_intent(result.intent_id)["status"] == "filled"
    assert ledger.list_positions("TEST")[0].quantity == Decimal("10")


def test_kill_switch_verifies_orders_are_really_gone():
    class UnconfirmedCancelBroker(FakeBroker):
        def cancel_order(self, broker_order_id):
            return self._orders_by_broker[broker_order_id]

    clock, broker, quote, ledger, _reconciler, oms = armed_stack(
        auto_fill=False, broker_class=UnconfirmedCancelBroker
    )
    submit(oms, intent(clock), quote)
    with pytest.raises(KillSwitchError, match="remains engaged"):
        oms.engage_kill_switch("operator emergency")
    assert ledger.get_control_state("TEST")["kill_switch"] is True
    assert len(broker.get_open_orders()) == 1


def test_operator_can_cancel_one_tracked_order_with_audited_reason():
    clock, broker, quote, ledger, _reconciler, oms = armed_stack(auto_fill=False)
    submitted = submit(oms, intent(clock), quote)
    canceled = oms.cancel_tracked_order(submitted.order_id, "operator review")
    assert canceled.intent_status == IntentStatus.CANCELED
    assert broker.get_open_orders() == []
    events = ledger.list_audit_events()
    assert "operator_cancel_requested" in {event["event_type"] for event in events}


def test_restart_recovery_imports_ambiguous_fill_from_file_ledger(tmp_path):
    clock, broker, quote, ledger, _reconciler, oms = armed_stack(tmp_path)
    broker.raise_after_submit_once = True
    with pytest.raises(BrokerError):
        submit(oms, intent(clock), quote)
    ledger.close()

    reopened = Ledger(tmp_path / "ledger.sqlite3", clock=clock)
    restarted = OrderManagementSystem(
        reopened, broker, PreTradeRiskEngine(clock=clock), clock=clock
    )
    results = restarted.recover_pending()
    assert len(results) == 1
    assert results[0].intent_status == IntentStatus.FILLED
    assert broker.submission_count == 1
    assert reopened.list_positions("TEST")[0].quantity == Decimal("10")


def test_partial_fills_are_applied_once_and_finish_on_recovery():
    clock, broker, quote, ledger, _reconciler, oms = armed_stack(auto_fill=False)
    result = submit(oms, intent(clock), quote)
    assert result.intent_status == IntentStatus.ORDER_SUBMITTED
    order = broker.get_order_by_client_id(result.client_order_id)
    broker.fill_order(order.broker_order_id, "4")
    partial = oms.recover_pending()[0]
    assert partial.intent_status == IntentStatus.ORDER_SUBMITTED
    assert ledger.list_positions("TEST")[0].quantity == Decimal("4")
    broker.fill_order(order.broker_order_id, "6")
    complete = oms.recover_pending()[0]
    assert complete.intent_status == IntentStatus.FILLED
    assert ledger.list_positions("TEST")[0].quantity == Decimal("10")
    assert len(ledger.list_fills()) == 2


def test_second_target_for_symbol_is_blocked_while_first_order_is_open():
    clock, broker, quote, _ledger, _reconciler, oms = armed_stack(auto_fill=False)
    submit(oms, intent(clock, "10", "v1"), quote)
    with pytest.raises(ExecutionBlocked, match="active order already exists"):
        submit(oms, intent(clock, "8", "v2"), quote)
    assert broker.submission_count == 1


def test_disarmed_intent_is_persistently_risk_rejected():
    clock, broker, quote, _ledger, _reconciler, oms = armed_stack()
    oms.disarm("operator stop")
    result = submit(oms, intent(clock), quote)
    assert result.intent_status == IntentStatus.RISK_REJECTED
    assert "NOT_ARMED" in {v.code for v in result.risk_violations}
    assert broker.submission_count == 0


def test_pretrade_cash_mismatch_disarms_before_order_construction():
    clock, broker, quote, ledger, _reconciler, oms = armed_stack()
    broker._cash = Decimal("9999")  # external debit after arm
    with pytest.raises(ReconciliationRequired, match="account state differ"):
        submit(oms, intent(clock), quote)
    assert ledger.get_control_state("TEST")["armed"] is False
    assert broker.submission_count == 0


def test_pretrade_external_open_order_disarms_before_order_construction():
    clock, broker, quote, ledger, _reconciler, oms = armed_stack(auto_fill=False)
    broker.submit_order(OrderRequest(
        "TEST", "manual-order", "external", "AAPL", Side.BUY, Decimal("1"), Decimal("100")
    ))
    with pytest.raises(ReconciliationRequired, match="account state differ"):
        submit(oms, intent(clock), quote)
    assert ledger.get_control_state("TEST")["armed"] is False


def test_broker_transport_error_during_pretrade_disarms_execution():
    class PositionOutageBroker(FakeBroker):
        fail_positions = False

        def get_positions(self):
            if self.fail_positions:
                raise BrokerError("simulated positions outage")
            return super().get_positions()

    clock, broker, quote, ledger, _reconciler, oms = armed_stack(
        broker_class=PositionOutageBroker
    )
    broker.fail_positions = True
    with pytest.raises(BrokerError, match="positions outage"):
        submit(oms, intent(clock), quote)
    assert ledger.get_control_state("TEST")["armed"] is False
    assert broker.submission_count == 0


def test_stale_quote_is_rejected_before_submission():
    clock, broker, quote, _ledger, _reconciler, oms = armed_stack()
    clock.advance(61)
    result = submit(oms, intent(clock), quote)
    assert result.intent_status == IntentStatus.RISK_REJECTED
    assert "STALE_QUOTE" in {v.code for v in result.risk_violations}
    assert broker.submission_count == 0


def test_arming_session_expires_and_disarms_before_risk():
    clock, broker, quote, ledger, _reconciler, oms = armed_stack()
    clock.advance(901)
    result = submit(oms, intent(clock), quote)
    assert result.intent_status == IntentStatus.RISK_REJECTED
    assert "NOT_ARMED" in {v.code for v in result.risk_violations}
    assert ledger.get_control_state("TEST")["armed"] is False
    assert broker.submission_count == 0


def test_daily_loss_and_drawdown_violation_engages_persistent_kill_switch():
    clock, broker, quote, ledger, _reconciler, oms = armed_stack()
    result = oms.process_intent(
        intent(clock),
        quote=quote,
        market_open=True,
        day_start_equity=Decimal("11000"),
        high_water_equity=Decimal("12000"),
    )
    assert result.intent_status == IntentStatus.RISK_REJECTED
    assert {"DAILY_LOSS", "DRAWDOWN"} <= {v.code for v in result.risk_violations}
    control = ledger.get_control_state("TEST")
    assert control["armed"] is False
    assert control["kill_switch"] is True
    assert broker.submission_count == 0


def test_kill_switch_cancels_open_order_and_requires_reset_and_reconcile():
    clock, broker, quote, ledger, reconciler, oms = armed_stack(auto_fill=False)
    result = submit(oms, intent(clock), quote)
    oms.engage_kill_switch("operator emergency")
    assert broker.get_open_orders() == []
    assert ledger.get_control_state("TEST")["kill_switch"] is True
    assert ledger.get_intent(result.intent_id)["status"] == "canceled"
    with pytest.raises(ExecutionBlocked, match="Reset"):
        oms.arm("too soon")
    oms.reset_kill_switch("reviewed")
    with pytest.raises(ReconciliationRequired, match="newer"):
        oms.arm("still too soon")
    assert reconciler.reconcile().clean
    oms.arm("post-emergency clean reconciliation")
    assert ledger.get_control_state("TEST")["armed"] is True


def test_arming_requires_baseline_and_clean_reconciliation():
    clock = ManualClock()
    broker = FakeBroker(account_id="TEST", cash="10000", clock=clock)
    broker.set_quote("AAPL", "100", spread_bps="0", as_of=clock())
    ledger = Ledger(":memory:", clock=clock)
    oms = OrderManagementSystem(ledger, broker, PreTradeRiskEngine(clock=clock), clock=clock)
    with pytest.raises(ReconciliationRequired, match="bootstrapped"):
        oms.arm("unsafe")
    Reconciler(ledger, broker, clock=clock).bootstrap_positions()
    with pytest.raises(ReconciliationRequired, match="clean reconciliation"):
        oms.arm("still unsafe")


def test_active_buy_orders_reserve_gross_exposure_before_they_fill():
    clock = ManualClock()
    broker = FakeBroker("TEST", "100000", auto_fill=False, clock=clock)
    quotes = {
        symbol: broker.set_quote(symbol, "100", spread_bps="0", as_of=clock())
        for symbol in ("AAPL", "MSFT")
    }
    ledger = Ledger(":memory:", clock=clock)
    reconciler = Reconciler(ledger, broker, clock=clock)
    reconciler.bootstrap_positions()
    assert reconciler.reconcile().clean
    limits = RiskLimits(
        max_order_notional=Decimal("20000"),
        max_gross_exposure_pct=Decimal("0.15"),
        max_symbol_exposure_pct=Decimal("0.15"),
        max_daily_turnover_pct=Decimal("1"),
    )
    oms = OrderManagementSystem(
        ledger, broker, PreTradeRiskEngine(limits, clock=clock), clock=clock
    )
    oms.arm("reservation test")

    def target(symbol):
        return TargetPositionIntent(
            "TEST", "aggregated-portfolio", symbol, Decimal("100"),
            clock(), "v1", Decimal("100"), "reservation test",
        )

    first = submit(oms, target("AAPL"), quotes["AAPL"])
    second = submit(oms, target("MSFT"), quotes["MSFT"])
    assert first.intent_status == IntentStatus.ORDER_SUBMITTED
    assert second.intent_status == IntentStatus.RISK_REJECTED
    assert "GROSS_EXPOSURE" in {v.code for v in second.risk_violations}
    assert broker.submission_count == 1


def test_active_orders_reserve_buying_power_cash_and_turnover():
    clock = ManualClock()
    broker = FakeBroker("TEST", "100000", auto_fill=False, clock=clock)
    quotes = {
        symbol: broker.set_quote(symbol, "100", spread_bps="0", as_of=clock())
        for symbol in ("AAPL", "MSFT")
    }
    ledger = Ledger(":memory:", clock=clock)
    reconciler = Reconciler(ledger, broker, clock=clock)
    reconciler.bootstrap_positions()
    assert reconciler.reconcile().clean
    limits = RiskLimits(
        max_order_notional=Decimal("70000"),
        max_gross_exposure_pct=Decimal("2"),
        max_symbol_exposure_pct=Decimal("1"),
        max_daily_turnover_pct=Decimal("1"),
    )
    oms = OrderManagementSystem(
        ledger, broker, PreTradeRiskEngine(limits, clock=clock), clock=clock
    )
    oms.arm("reservation test")
    for symbol in ("AAPL", "MSFT"):
        target = TargetPositionIntent(
            "TEST", "aggregated-portfolio", symbol, Decimal("600"),
            clock(), "v1", Decimal("100"), "reservation test",
        )
        result = submit(oms, target, quotes[symbol])
        if symbol == "AAPL":
            assert result.intent_status == IntentStatus.ORDER_SUBMITTED
        else:
            assert result.intent_status == IntentStatus.RISK_REJECTED
            assert {"BUYING_POWER", "CASH_BUFFER", "DAILY_TURNOVER"} <= {
                violation.code for violation in result.risk_violations
            }
    assert broker.submission_count == 1


def test_account_execution_fence_serializes_snapshot_risk_and_submit(tmp_path):
    clock = ManualClock()
    broker = FakeBroker("TEST", "100000", auto_fill=True, clock=clock)
    quotes = {
        symbol: broker.set_quote(symbol, "100", spread_bps="0", as_of=clock())
        for symbol in ("AAPL", "MSFT")
    }
    path = tmp_path / "serialized-risk.sqlite3"
    limits = RiskLimits(
        max_order_notional=Decimal("20000"),
        max_gross_exposure_pct=Decimal("0.15"),
        max_symbol_exposure_pct=Decimal("0.15"),
        max_daily_turnover_pct=Decimal("1"),
    )
    with Ledger(path, clock=clock) as ledger:
        reconciler = Reconciler(ledger, broker, clock=clock)
        reconciler.bootstrap_positions()
        assert reconciler.reconcile().clean
        OrderManagementSystem(
            ledger, broker, PreTradeRiskEngine(limits, clock=clock), clock=clock
        ).arm("concurrency test")

    first_in_risk = threading.Event()
    release_first = threading.Event()
    second_in_risk = threading.Event()
    results = {}
    errors = []

    class PausingRisk(PreTradeRiskEngine):
        def evaluate(self, request, **kwargs):
            decision = super().evaluate(request, **kwargs)
            if request.symbol == "AAPL":
                first_in_risk.set()
                assert release_first.wait(timeout=3)
            else:
                second_in_risk.set()
            return decision

    def run(symbol):
        try:
            with Ledger(path, clock=clock) as ledger:
                oms = OrderManagementSystem(
                    ledger, broker, PausingRisk(limits, clock=clock), clock=clock
                )
                target = TargetPositionIntent(
                    "TEST", "aggregated-portfolio", symbol, Decimal("100"),
                    clock(), "v1", Decimal("100"), "concurrency test",
                )
                results[symbol] = submit(oms, target, quotes[symbol])
        except Exception as exc:  # surfaced below
            errors.append(exc)

    first = threading.Thread(target=run, args=("AAPL",))
    second = threading.Thread(target=run, args=("MSFT",))
    first.start()
    assert first_in_risk.wait(timeout=3)
    second.start()
    assert second_in_risk.wait(timeout=0.2) is False
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert errors == []
    assert results["AAPL"].intent_status == IntentStatus.FILLED
    assert results["MSFT"].intent_status == IntentStatus.RISK_REJECTED
    assert broker.submission_count == 1


def test_broker_acknowledgement_integrity_conflict_disarms():
    class MismatchedAcknowledgementBroker(FakeBroker):
        def submit_order(self, request):
            return replace(super().submit_order(request), symbol="MSFT")

    clock, _broker, quote, ledger, _reconciler, oms = armed_stack(
        auto_fill=False, broker_class=MismatchedAcknowledgementBroker
    )
    with pytest.raises(LedgerConflict, match="mismatched: symbol"):
        submit(oms, intent(clock), quote)
    assert ledger.get_control_state("TEST")["armed"] is False


def test_disarm_always_targets_the_ledger_bound_account():
    clock, _broker, _quote, ledger, _reconciler, _oms = armed_stack()
    wrong_broker = FakeBroker("OTHER-ACCOUNT", "10000", clock=clock)
    wrong_oms = OrderManagementSystem(
        ledger, wrong_broker, PreTradeRiskEngine(clock=clock), clock=clock
    )
    wrong_oms.disarm("credentials changed; local emergency stop")
    assert ledger.get_control_state("TEST")["armed"] is False
    assert ledger.get_control_state("OTHER-ACCOUNT")["updated_at"] is None


def test_cancel_missing_active_order_disarms_before_resolution():
    clock, broker, quote, ledger, _reconciler, oms = armed_stack(auto_fill=False)
    result = submit(oms, intent(clock), quote)
    broker._orders_by_client.clear()
    broker._orders_by_broker.clear()
    with pytest.raises(ReconciliationRequired, match="cannot find"):
        oms.cancel_tracked_order(result.order_id, "operator cancel")
    assert ledger.get_control_state("TEST")["armed"] is False


def test_submission_authorizer_runs_inside_reentrant_execution_fence():
    clock = ManualClock()
    broker = FakeBroker("TEST", "10000", auto_fill=False, clock=clock)
    quote = broker.set_quote("AAPL", "100", spread_bps="0", as_of=clock())
    ledger = Ledger(":memory:", clock=clock)
    reconciler = Reconciler(ledger, broker, clock=clock)
    reconciler.bootstrap_positions()
    assert reconciler.reconcile().clean
    calls = []

    def authorize():
        calls.append(getattr(ledger._execution_guard_local, "depth", 0))

    oms = OrderManagementSystem(
        ledger, broker, PreTradeRiskEngine(clock=clock), clock=clock,
        submission_authorizer=authorize,
    )
    oms.arm("authorizer test")
    result = submit(oms, intent(clock), quote)
    assert result.intent_status == IntentStatus.ORDER_SUBMITTED
    assert len(calls) == 1 and calls[0] >= 1


def test_submission_authorizer_failure_disarms_without_posting():
    clock = ManualClock()
    broker = FakeBroker("TEST", "10000", auto_fill=False, clock=clock)
    quote = broker.set_quote("AAPL", "100", spread_bps="0", as_of=clock())
    ledger = Ledger(":memory:", clock=clock)
    reconciler = Reconciler(ledger, broker, clock=clock)
    reconciler.bootstrap_positions()
    assert reconciler.reconcile().clean

    def reject_submission():
        raise ExecutionBlocked("final policy changed")

    oms = OrderManagementSystem(
        ledger, broker, PreTradeRiskEngine(clock=clock), clock=clock,
        submission_authorizer=reject_submission,
    )
    oms.arm("authorizer test")
    with pytest.raises(ExecutionBlocked, match="policy changed"):
        submit(oms, intent(clock), quote)
    assert broker.submission_count == 0
    assert ledger.get_control_state("TEST")["armed"] is False
