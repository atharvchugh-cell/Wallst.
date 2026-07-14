"""Risk-limit and strict broker/ledger reconciliation tests."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import threading

import pytest

from src.live.broker import BrokerError
from src.live.fake_broker import FakeBroker
from src.live.ledger import Ledger, LedgerConflict
from src.live.models import (
    AccountSnapshot,
    Fill,
    OrderRequest,
    OrderStatus,
    OrderType,
    Position,
    Quote,
    Side,
    TargetPositionIntent,
)
from src.live.oms import OrderManagementSystem
from src.live.reconcile import Reconciler
from src.live.risk import PreTradeRiskEngine, RiskLimits


NOW = datetime(2026, 7, 14, 14, 0, tzinfo=timezone.utc)


class FixedClock:
    def __call__(self):
        return NOW


def request(side=Side.BUY, quantity="10", reference="100"):
    return OrderRequest(
        account_id="TEST",
        client_order_id="wslab-risk-test",
        intent_id="intent-risk-test",
        symbol="AAPL",
        side=side,
        quantity=Decimal(quantity),
        reference_price=Decimal(reference),
    )


def evaluate(req, *, limits=None, quote=None, positions=None, account=None, **overrides):
    quote = quote or Quote("AAPL", Decimal("100"), Decimal("100"), Decimal("100"), NOW)
    account = account or AccountSnapshot(
        "TEST", Decimal("10000"), Decimal("10000"), Decimal("10000"), NOW
    )
    values = {
        "quote": quote,
        "account": account,
        "positions": positions or [],
        "open_order_count": 0,
        "daily_turnover": Decimal("0"),
        "day_start_equity": Decimal("10000"),
        "high_water_equity": Decimal("10000"),
        "armed": True,
        "kill_switch": False,
        "market_open": True,
        "signal_at": NOW,
        "now": NOW,
    }
    values.update(overrides)
    return PreTradeRiskEngine(limits or RiskLimits(), clock=FixedClock()).evaluate(req, **values)


def codes(decision):
    return {v.code for v in decision.violations}


def test_risk_checks_price_collar_market_state_and_fractional_quantity():
    moved = Quote("AAPL", Decimal("102"), Decimal("102"), Decimal("102"), NOW)
    decision = evaluate(
        request(quantity="1.5"), quote=moved, market_open=False,
    )
    assert {"PRICE_COLLAR", "MARKET_CLOSED", "FRACTIONAL_DISABLED"} <= codes(decision)


def test_limit_order_risk_uses_limit_price_as_worst_case_buy_notional():
    limit_request = OrderRequest(
        account_id="TEST",
        client_order_id="wslab-limit-risk",
        intent_id="intent-limit-risk",
        symbol="AAPL",
        side=Side.BUY,
        quantity=Decimal("10"),
        reference_price=Decimal("100"),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("120"),
    )
    decision = evaluate(
        limit_request,
        limits=RiskLimits(max_order_notional=Decimal("1100")),
    )
    assert decision.order_notional == Decimal("1200")
    assert {"ORDER_NOTIONAL", "PRICE_COLLAR"} <= codes(decision)


def test_risk_blocks_short_sale_and_buying_power_breach():
    long = Position("AAPL", Decimal("2"), Decimal("100"), Decimal("100"))
    sell = evaluate(request(side=Side.SELL, quantity="3"), positions=[long])
    assert "SHORT_SALE" in codes(sell)
    poor = AccountSnapshot("TEST", Decimal("50"), Decimal("10000"), Decimal("50"), NOW)
    buy = evaluate(request(quantity="1"), account=poor)
    assert {"BUYING_POWER", "CASH_BUFFER"} <= codes(buy)


def test_risk_blocks_stale_account_snapshot_and_existing_short():
    stale = AccountSnapshot(
        "TEST", Decimal("10000"), Decimal("10000"), Decimal("10000"),
        NOW - timedelta(seconds=61),
    )
    short = Position("MSFT", Decimal("-1"), Decimal("100"), Decimal("100"))
    decision = evaluate(request(), account=stale, positions=[short])
    assert {"STALE_ACCOUNT_SNAPSHOT", "EXISTING_SHORT"} <= codes(decision)


def test_risk_blocks_nonready_broker_account_flags():
    blocked = AccountSnapshot(
        "TEST",
        Decimal("10000"),
        Decimal("10000"),
        Decimal("10000"),
        NOW,
        status="ACCOUNT_UPDATED",
        trading_blocked=True,
    )
    decision = evaluate(request(), account=blocked)
    assert {"ACCOUNT_NOT_ACTIVE", "BROKER_TRADING_BLOCKED"} <= codes(decision)


def test_risk_blocks_stale_and_future_signals():
    stale = evaluate(request(), signal_at=NOW - timedelta(days=2))
    future = evaluate(request(), signal_at=NOW + timedelta(seconds=1))
    assert "STALE_SIGNAL" in codes(stale)
    assert "FUTURE_SIGNAL" in codes(future)


def test_risk_blocks_notional_exposure_turnover_daily_loss_and_drawdown():
    limits = RiskLimits(
        max_order_notional=Decimal("500"),
        max_gross_exposure_pct=Decimal("0.05"),
        max_symbol_exposure_pct=Decimal("0.04"),
        max_daily_turnover_pct=Decimal("0.05"),
        max_daily_loss_pct=Decimal("0.02"),
        max_drawdown_pct=Decimal("0.10"),
    )
    account = AccountSnapshot("TEST", Decimal("9000"), Decimal("9000"), Decimal("9000"), NOW)
    decision = evaluate(
        request(quantity="10"),
        limits=limits,
        account=account,
        daily_turnover=Decimal("400"),
        day_start_equity=Decimal("10000"),
        high_water_equity=Decimal("11000"),
    )
    assert {
        "ORDER_NOTIONAL", "GROSS_EXPOSURE", "SYMBOL_EXPOSURE", "DAILY_TURNOVER",
        "DAILY_LOSS", "DRAWDOWN",
    } <= codes(decision)


def test_reconciliation_detects_position_mismatch_and_disarms():
    clock = FixedClock()
    broker = FakeBroker(account_id="TEST", cash="10000", clock=clock)
    broker.set_quote("AAPL", "100", spread_bps="0", as_of=NOW)
    ledger = Ledger(":memory:", clock=clock)
    reconciler = Reconciler(ledger, broker, clock=clock)
    reconciler.bootstrap_positions()
    assert reconciler.reconcile().clean
    oms = OrderManagementSystem(ledger, broker, PreTradeRiskEngine(clock=clock), clock=clock)
    oms.arm("test")
    broker.seed_position("AAPL", "2", "100")  # simulates manual/external activity
    report = reconciler.reconcile()
    assert report.clean is False
    assert "POSITION_MISMATCH" in {i.code for i in report.issues}
    assert ledger.get_control_state("TEST")["armed"] is False


def test_reconciliation_detects_cash_mismatch_and_disarms():
    clock = FixedClock()
    broker = FakeBroker(account_id="TEST", cash="10000", clock=clock)
    broker.set_quote("AAPL", "100", spread_bps="0", as_of=NOW)
    ledger = Ledger(":memory:", clock=clock)
    reconciler = Reconciler(ledger, broker, clock=clock)
    reconciler.bootstrap_positions()
    assert reconciler.reconcile().clean
    oms = OrderManagementSystem(ledger, broker, PreTradeRiskEngine(clock=clock), clock=clock)
    oms.arm("test")
    broker._cash = Decimal("9999")  # external cash debit, e.g. unrecorded fee
    report = reconciler.reconcile()
    assert "CASH_MISMATCH" in {i.code for i in report.issues}
    assert ledger.get_control_state("TEST")["armed"] is False


def test_dirty_reconciliation_disarms_before_crash_during_alert_persistence(
    monkeypatch, tmp_path
):
    from src.live.phase4_store import Phase4Store

    clock = FixedClock()
    broker = FakeBroker(account_id="TEST", cash="10000", clock=clock)
    broker.set_quote("AAPL", "100", spread_bps="0", as_of=NOW)
    with Ledger(tmp_path / "atomic-dirty-reconciliation.sqlite3", clock=clock) as ledger:
        reconciler = Reconciler(ledger, broker, clock=clock)
        reconciler.bootstrap_positions()
        assert reconciler.reconcile().clean
        oms = OrderManagementSystem(
            ledger, broker, PreTradeRiskEngine(clock=clock), clock=clock
        )
        oms.arm("test crash boundary")
        broker.seed_position("AAPL", "2", "100")

        def simulate_process_crash(*_args, **_kwargs):
            # BaseException bypasses Reconciler's ordinary exception cleanup,
            # modeling process death at the alert boundary.
            raise SystemExit("simulated crash during alert persistence")

        monkeypatch.setattr(Phase4Store, "emit_alert", simulate_process_crash)
        with pytest.raises(SystemExit, match="simulated crash"):
            reconciler.reconcile()

        latest = ledger.latest_reconciliation("TEST")
        assert latest is not None and not latest["clean"]
        assert ledger.get_control_state("TEST")["armed"] is False


def test_stale_clean_reconciliation_cannot_commit_after_newer_dirty_snapshot(tmp_path):
    clock = FixedClock()
    path = tmp_path / "serialized-reconciliation.sqlite3"
    clean_broker = FakeBroker(account_id="TEST", cash="10000", clock=clock)
    dirty_broker = FakeBroker(account_id="TEST", cash="10000", clock=clock)
    dirty_broker.seed_position("AAPL", "1", "100")
    with Ledger(path, clock=clock) as ledger:
        Reconciler(ledger, clean_broker, clock=clock).bootstrap_positions()

    clean_ready = threading.Event()
    release_clean = threading.Event()
    dirty_finished = threading.Event()
    errors = []

    class PausingCleanReconciler(Reconciler):
        def _record_reconciliation_state(self, **kwargs):
            clean_ready.set()
            assert release_clean.wait(timeout=3)
            return super()._record_reconciliation_state(**kwargs)

    def run_clean():
        try:
            with Ledger(path, clock=clock) as ledger:
                PausingCleanReconciler(ledger, clean_broker, clock=clock).reconcile()
        except Exception as exc:  # surfaced below
            errors.append(exc)

    def run_dirty():
        try:
            with Ledger(path, clock=clock) as ledger:
                report = Reconciler(ledger, dirty_broker, clock=clock).reconcile()
                assert not report.clean
        except Exception as exc:  # surfaced below
            errors.append(exc)
        finally:
            dirty_finished.set()

    clean_thread = threading.Thread(target=run_clean)
    dirty_thread = threading.Thread(target=run_dirty)
    clean_thread.start()
    assert clean_ready.wait(timeout=3)
    dirty_thread.start()
    assert not dirty_finished.wait(timeout=0.2)
    release_clean.set()
    clean_thread.join(timeout=5)
    dirty_thread.join(timeout=5)

    assert errors == []
    with Ledger(path, clock=clock) as ledger:
        latest = ledger.latest_reconciliation("TEST")
        assert latest is not None and not latest["clean"]
        assert ledger.get_control_state("TEST")["armed"] is False


def test_reconciliation_detects_external_open_order():
    clock = FixedClock()
    broker = FakeBroker(account_id="TEST", cash="10000", auto_fill=False, clock=clock)
    broker.set_quote("AAPL", "100", spread_bps="0", as_of=NOW)
    ledger = Ledger(":memory:", clock=clock)
    reconciler = Reconciler(ledger, broker, clock=clock)
    reconciler.bootstrap_positions()
    broker.submit_order(OrderRequest(
        account_id="TEST", client_order_id="manual-order-1", intent_id="external",
        symbol="AAPL", side=Side.BUY, quantity=Decimal("1"), reference_price=Decimal("100"),
    ))
    report = reconciler.reconcile()
    assert "EXTERNAL_OPEN_ORDER" in {i.code for i in report.issues}


def test_reconciliation_watermark_catches_terminal_order_created_during_prior_run():
    class MutableClock:
        def __init__(self, value):
            self.value = value

        def __call__(self):
            return self.value

        def advance(self, seconds):
            self.value += timedelta(seconds=seconds)

    class SubmissionTimeRacingBroker(FakeBroker):
        inject_terminal_order = False
        injected = False

        def get_recent_orders(self, since=None):
            # Match Alpaca's documented exclusive submission-time `after`
            # filter rather than FakeBroker's update-time convenience filter.
            orders = list(self._orders_by_client.values())
            if since is not None:
                orders = [order for order in orders if order.submitted_at > since]
            return sorted(orders, key=lambda order: (order.submitted_at, order.broker_order_id))

        def get_positions(self):
            positions = super().get_positions()
            if self.inject_terminal_order and not self.injected:
                self.injected = True
                self.clock.advance(1)
                order = self.submit_order(OrderRequest(
                    account_id="TEST", client_order_id="external-terminal-race",
                    intent_id="external", symbol="AAPL", side=Side.BUY,
                    quantity=Decimal("1"), reference_price=Decimal("100"),
                ))
                self.cancel_order(order.broker_order_id)
                self.clock.advance(1)
            return positions

    clock = MutableClock(NOW)
    broker = SubmissionTimeRacingBroker(
        account_id="TEST", cash="10000", auto_fill=False, clock=clock
    )
    broker.set_quote("AAPL", "100", spread_bps="0", as_of=NOW)
    ledger = Ledger(":memory:", clock=clock)
    reconciler = Reconciler(ledger, broker, clock=clock)
    reconciler.bootstrap_positions()

    # The order appears only after the first run's recent/open-order queries and
    # is already terminal before that run records its completion watermark.
    broker.inject_terminal_order = True
    first = reconciler.reconcile()
    assert first.clean
    assert broker.get_open_orders() == []

    second = reconciler.reconcile()
    assert "EXTERNAL_RECENT_ORDER" in {issue.code for issue in second.issues}


def test_reconciliation_flags_non_namespaced_untracked_fill():
    clock = FixedClock()
    broker = FakeBroker(account_id="TEST", cash="10000", clock=clock)
    ledger = Ledger(":memory:", clock=clock)
    reconciler = Reconciler(ledger, broker, clock=clock)
    reconciler.bootstrap_positions()
    broker._fills.append(Fill(
        fill_id="external-fill-1",
        broker_order_id="external-order-1",
        client_order_id="manual-client-1",
        account_id="TEST",
        symbol="AAPL",
        side=Side.BUY,
        quantity=Decimal("1"),
        price=Decimal("100"),
        commission=Decimal("0"),
        occurred_at=NOW,
    ))

    report = reconciler.reconcile()
    assert "EXTERNAL_FILL" in {issue.code for issue in report.issues}


def test_reconciliation_fetches_one_overlap_bounded_fill_snapshot():
    class BoundedFillBroker(FakeBroker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fill_queries = []

        def get_fills(self, since=None):
            if since is None:
                raise BrokerError("lifetime fill scans are forbidden")
            self.fill_queries.append(since)
            return super().get_fills(since)

    clock = FixedClock()
    broker = BoundedFillBroker(account_id="TEST", cash="10000", clock=clock)
    ledger = Ledger(":memory:", clock=clock)
    reconciler = Reconciler(ledger, broker, clock=clock)
    reconciler.bootstrap_positions()

    assert reconciler.reconcile().clean
    assert len(broker.fill_queries) == 1
    assert broker.fill_queries[0] < NOW


def test_reconciliation_imports_fill_after_lost_ack_and_ends_clean():
    clock = FixedClock()
    broker = FakeBroker(account_id="TEST", cash="10000", clock=clock)
    quote = broker.set_quote("AAPL", "100", spread_bps="0", as_of=NOW)
    ledger = Ledger(":memory:", clock=clock)
    reconciler = Reconciler(ledger, broker, clock=clock)
    reconciler.bootstrap_positions()
    reconciler.reconcile()
    oms = OrderManagementSystem(ledger, broker, PreTradeRiskEngine(clock=clock), clock=clock)
    oms.arm("test")
    broker.raise_after_submit_once = True
    target = TargetPositionIntent(
        "TEST", "aggregate", "AAPL", Decimal("10"), NOW, "v1", Decimal("100")
    )
    with pytest.raises(BrokerError):
        oms.process_intent(
            target, quote=quote, market_open=True,
            day_start_equity=Decimal("10000"), high_water_equity=Decimal("10000"),
        )
    report = reconciler.reconcile()
    assert report.clean
    assert ledger.list_positions("TEST")[0].quantity == Decimal("10")
    assert ledger.list_intents()[0]["status"] == "filled"


def test_reconciliation_flags_stale_broker_order_state_instead_of_regressing_ledger():
    clock = FixedClock()
    broker = FakeBroker(account_id="TEST", cash="10000", clock=clock)
    quote = broker.set_quote("AAPL", "100", spread_bps="0", as_of=NOW)
    ledger = Ledger(":memory:", clock=clock)
    reconciler = Reconciler(ledger, broker, clock=clock)
    reconciler.bootstrap_positions()
    reconciler.reconcile()
    oms = OrderManagementSystem(ledger, broker, PreTradeRiskEngine(clock=clock), clock=clock)
    oms.arm("test")
    target = TargetPositionIntent(
        "TEST", "aggregate", "AAPL", Decimal("10"), NOW, "v1", Decimal("100")
    )
    result = oms.process_intent(
        target, quote=quote, market_open=True,
        day_start_equity=Decimal("10000"), high_water_equity=Decimal("10000"),
    )
    broker_order = broker.get_order_by_client_id(result.client_order_id)
    stale = replace(broker_order, status=OrderStatus.SUBMITTED, filled_quantity=Decimal("0"))
    broker._orders_by_client[result.client_order_id] = stale
    broker._orders_by_broker[stale.broker_order_id] = stale
    report = reconciler.reconcile()
    assert "ORDER_STATE_CONFLICT" in {issue.code for issue in report.issues}
    assert ledger.get_intent(result.intent_id)["status"] == "filled"


def test_wrong_broker_account_is_rejected_and_bound_account_is_disarmed():
    clock = FixedClock()
    broker_a = FakeBroker(account_id="ACCOUNT-A", cash="10000", clock=clock)
    ledger = Ledger(":memory:", clock=clock)
    reconciler_a = Reconciler(ledger, broker_a, clock=clock)
    reconciler_a.bootstrap_positions()
    assert reconciler_a.reconcile().clean
    oms = OrderManagementSystem(
        ledger, broker_a, PreTradeRiskEngine(clock=clock), clock=clock
    )
    oms.arm("bind and arm account A")

    broker_b = FakeBroker(account_id="ACCOUNT-B", cash="10000", clock=clock)
    with pytest.raises(LedgerConflict, match="bound to account ACCOUNT-A"):
        Reconciler(ledger, broker_b, clock=clock).reconcile()
    assert ledger.get_control_state("ACCOUNT-A")["armed"] is False
    assert ledger.bound_account_id() == "ACCOUNT-A"


def test_reconciliation_transport_failure_invalidates_prior_clean_state():
    class FailingFillBroker(FakeBroker):
        fail_fills = False

        def get_fills(self, since=None):
            if self.fail_fills:
                raise BrokerError("simulated activity endpoint outage")
            return super().get_fills(since)

    clock = FixedClock()
    broker = FailingFillBroker(account_id="TEST", cash="10000", clock=clock)
    ledger = Ledger(":memory:", clock=clock)
    reconciler = Reconciler(ledger, broker, clock=clock)
    reconciler.bootstrap_positions()
    assert reconciler.reconcile().clean
    oms = OrderManagementSystem(
        ledger, broker, PreTradeRiskEngine(clock=clock), clock=clock
    )
    oms.arm("test")
    broker.fail_fills = True
    with pytest.raises(BrokerError, match="endpoint outage"):
        reconciler.reconcile()
    assert ledger.get_control_state("TEST")["armed"] is False
    assert "reconciliation_failed" in {
        event["event_type"] for event in ledger.list_audit_events()
    }


def test_reconciliation_rejects_legacy_fractional_position_even_when_quantities_match():
    clock = FixedClock()
    broker = FakeBroker(account_id="TEST", cash="10000", clock=clock)
    ledger = Ledger(":memory:", clock=clock)
    reconciler = Reconciler(ledger, broker, clock=clock)
    reconciler.bootstrap_positions()
    broker.seed_position("AAPL", "1.5", "100")
    ledger.conn.execute(
        """INSERT INTO position_state(account_id, symbol, quantity, avg_price, updated_at)
           VALUES (?, ?, ?, ?, ?)""",
        ("TEST", "AAPL", "1.5", "100", NOW.isoformat()),
    )
    ledger.conn.commit()
    report = reconciler.reconcile()
    assert "FRACTIONAL_POSITION_UNSUPPORTED" in {issue.code for issue in report.issues}
