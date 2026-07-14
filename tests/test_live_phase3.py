"""Phase-3 aggregation, approval, equity, and paper execution red-team tests."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import pytest

from src.live.deployment import (
    DeploymentConfig,
    DeploymentError,
    ExecutionPlan,
    SleeveTargetSnapshot,
    aggregate_sleeves,
    build_execution_plan,
    load_strict_json,
)
from src.live import paper_cli
from src.live.execution import PaperExecutionService
from src.live.fake_broker import FakeBroker
from src.live.ledger import Ledger, LedgerConflict
from src.live.market_data import (
    NEW_YORK,
    MarketCalendarDay,
    MarketDataError,
    MarketDataProvider,
)
from src.live.models import Quote, TargetPositionIntent
from src.live.oms import ExecutionBlocked, OrderManagementSystem
from src.live.reconcile import Reconciler
from src.live.risk import PreTradeRiskEngine


NOW = datetime(2026, 7, 14, 14, 0, tzinfo=timezone.utc)  # 10:00 ET
SIGNAL_AT = datetime(2026, 7, 13, 20, 5, tzinfo=timezone.utc)


class FixedClock:
    def __init__(self, value=NOW):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += timedelta(seconds=seconds)


class ClockResponse:
    def __init__(self, timestamp, is_open=True):
        self.timestamp = timestamp
        self.is_open = is_open
        self.next_close = timestamp.replace(hour=20, minute=0)
        self.next_open = (timestamp + timedelta(days=1)).replace(hour=13, minute=30)


class Phase3FakeBroker(FakeBroker):
    market_open = True
    clock_offset = timedelta(0)

    def get_market_clock(self):
        return ClockResponse(self.clock() + self.clock_offset, self.market_open)

    def get_market_calendar(self, start, end):
        closed = getattr(self, "calendar_closed_dates", set())
        early_closes = getattr(self, "calendar_early_closes", {})
        days = []
        current = start
        while current <= end:
            if current.weekday() < 5 and current.isoformat() not in closed:
                close_value = early_closes.get(current.isoformat(), time(16, 0))
                days.append(MarketCalendarDay(
                    current,
                    datetime.combine(current, time(9, 30), tzinfo=NEW_YORK),
                    datetime.combine(current, close_value, tzinfo=NEW_YORK),
                ))
            current += timedelta(days=1)
        return tuple(days)

    def drain_request_ids(self):
        return ()


class FakeMarketData(MarketDataProvider):
    def __init__(self, broker):
        self.broker = broker

    def get_quotes(self, symbols):
        return {symbol: self.broker.get_quote(symbol) for symbol in symbols}

    def drain_request_ids(self):
        return ()


def risk_payload(**overrides):
    payload = {
        "max_order_notional": "10000",
        "max_gross_exposure_pct": "0.80",
        "max_symbol_exposure_pct": "0.50",
        "max_daily_turnover_pct": "0.80",
        "max_daily_loss_pct": "0.02",
        "max_drawdown_pct": "0.10",
        "max_open_orders": 5,
        "min_cash_buffer": "100",
        "quote_max_age_seconds": 60,
        "account_max_age_seconds": 60,
        "max_signal_age_seconds": 86400,
        "price_collar_bps": "100",
        "allow_fractional_shares": False,
        "require_market_open": True,
    }
    payload.update(overrides)
    return payload


def deployment_payload(**overrides):
    payload = {
        "schema_version": 1,
        "deployment_id": "paper-v1",
        "account_id": "PAPER",
        "allocation_policy": "rebalance_to_deployment_weights",
        "execution_policy": "next_session_regular_hours_market",
        "managed_symbols": ["AAPL", "SPY"],
        "sleeve_weights": {"momentum": "0.60", "sector_rotation": "0.40"},
        "risk_limits": risk_payload(),
        "max_batch_orders": 5,
    }
    payload.update(overrides)
    return payload


def targets_payload(**overrides):
    payload = {
        "schema_version": 1,
        "signal_at": SIGNAL_AT.isoformat(),
        "target_version": "signals-v1",
        "sleeves": {
            "momentum": {"AAPL": "0.50"},
            "sector_rotation": {"SPY": "0.50"},
        },
    }
    payload.update(overrides)
    return payload


def artifacts():
    deployment = DeploymentConfig.from_payload(deployment_payload())
    targets = SleeveTargetSnapshot.from_payload(targets_payload(), deployment)
    return deployment, targets


def broker_stack(tmp_path, *, auto_fill=True):
    clock = FixedClock()
    broker = Phase3FakeBroker(
        account_id="PAPER", cash="10000", auto_fill=auto_fill, clock=clock
    )
    broker.set_quote("AAPL", "100", spread_bps="0", as_of=clock())
    broker.set_quote("SPY", "100", spread_bps="0", as_of=clock())
    db = tmp_path / "paper.sqlite3"
    with Ledger(db, clock=clock) as ledger:
        reconciler = Reconciler(ledger, broker, clock=clock)
        reconciler.bootstrap_positions()
        assert reconciler.reconcile().clean
    return clock, broker, FakeMarketData(broker), db


def preview_approve(tmp_path, *, auto_fill=True):
    clock, broker, data, db = broker_stack(tmp_path, auto_fill=auto_fill)
    deployment, targets = artifacts()
    with Ledger(db, clock=clock) as ledger:
        service = PaperExecutionService(ledger, broker, data, clock=clock)
        plan, created = service.preview(
            deployment, targets, confirm_new_equity_session=True
        )
        assert created
        ledger.approve_execution_batch(
            plan.batch_id, plan.plan_hash,
            approved_by="test-operator", reason="reviewed exact quantities",
        )
    return clock, broker, data, db, plan


def test_strict_deployment_aggregates_sleeves_to_account_targets():
    deployment, targets = artifacts()
    combined = aggregate_sleeves(deployment, targets)
    assert combined == {"AAPL": Decimal("0.3000"), "SPY": Decimal("0.2000")}


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda p: p.update(managed_symbols=["SPY", "AAPL"]), "sorted"),
        (lambda p: p.update(allocation_policy="implicit-drift"), "allocation_policy"),
        (lambda p: p.update(execution_policy="same-close"), "execution_policy"),
        (lambda p: p.update(sleeve_weights={"momentum": "0.8", "other": "0.3"}), "100%"),
        (lambda p: p.update(risk_limits=risk_payload(allow_fractional_shares=True)), "fractional"),
        (lambda p: p.update(risk_limits=risk_payload(require_market_open=False)), "market_open"),
        (lambda p: p.update(risk_limits=risk_payload(quote_max_age_seconds=61)), "60 seconds"),
        (lambda p: p.update(max_batch_orders=6), "max_open_orders"),
    ],
)
def test_deployment_rejects_unsafe_policy(mutator, match):
    payload = deployment_payload()
    mutator(payload)
    with pytest.raises(DeploymentError, match=match):
        DeploymentConfig.from_payload(payload)


def test_deployment_caps_daily_signal_staleness_window():
    payload = deployment_payload(
        risk_limits=risk_payload(max_signal_age_seconds=7 * 86400 + 1)
    )
    with pytest.raises(DeploymentError, match="7 calendar days"):
        DeploymentConfig.from_payload(payload)


def test_targets_require_every_sleeve_and_managed_symbols_only():
    deployment, _targets = artifacts()
    missing = targets_payload(sleeves={"momentum": {"AAPL": "0.5"}})
    with pytest.raises(DeploymentError, match="exactly match"):
        SleeveTargetSnapshot.from_payload(missing, deployment)
    outside = targets_payload(sleeves={
        "momentum": {"MSFT": "0.5"}, "sector_rotation": {"SPY": "0.5"}
    })
    with pytest.raises(DeploymentError, match="outside managed"):
        SleeveTargetSnapshot.from_payload(outside, deployment)


def test_strict_json_rejects_duplicate_keys_and_symlinks(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(DeploymentError, match="Duplicate JSON key"):
        load_strict_json(duplicate)
    link = tmp_path / "link.json"
    link.symlink_to(duplicate)
    with pytest.raises(DeploymentError, match="symbolic"):
        load_strict_json(link)


def test_plan_sizing_is_whole_share_deterministic_and_hash_verified():
    deployment, targets = artifacts()
    broker = Phase3FakeBroker("PAPER", "10000", clock=lambda: NOW)
    quotes = {
        "AAPL": broker.set_quote("AAPL", "101", spread_bps="0", as_of=NOW),
        "SPY": broker.set_quote("SPY", "99", spread_bps="0", as_of=NOW),
    }
    plan = build_execution_plan(
        deployment, targets, account=broker.get_account(), positions=[], quotes=quotes,
        trading_date="2026-07-14",
    )
    by_symbol = {item.symbol: item for item in plan.items}
    assert by_symbol["AAPL"].target_quantity == Decimal("29")
    assert by_symbol["SPY"].target_quantity == Decimal("20")
    assert ExecutionPlan.from_payload(plan.to_payload()) == plan
    tampered = plan.to_payload()
    tampered["items"][0]["target_quantity"] = "999"
    with pytest.raises(DeploymentError, match="delta|hash"):
        ExecutionPlan.from_payload(tampered)


def test_plan_refuses_unmanaged_positions_and_excessive_batch_size():
    deployment, targets = artifacts()
    broker = Phase3FakeBroker("PAPER", "10000", clock=lambda: NOW)
    quotes = {
        symbol: broker.set_quote(symbol, "100", spread_bps="0", as_of=NOW)
        for symbol in deployment.managed_symbols
    }
    broker.set_quote("MSFT", "100", spread_bps="0", as_of=NOW)
    broker.seed_position("MSFT", "1", "100")
    with pytest.raises(DeploymentError, match="unmanaged"):
        build_execution_plan(
            deployment, targets, account=broker.get_account(),
            positions=broker.get_positions(), quotes=quotes, trading_date="2026-07-14",
        )
    one_order = DeploymentConfig.from_payload(deployment_payload(max_batch_orders=1))
    with pytest.raises(DeploymentError, match="exceeding max_batch_orders"):
        build_execution_plan(
            one_order, targets, account=broker.get_account(), positions=[], quotes=quotes,
            trading_date="2026-07-14",
        )


def test_plan_preflights_complete_batch_daily_turnover_before_approval():
    deployment = DeploymentConfig.from_payload(deployment_payload(
        risk_limits=risk_payload(max_daily_turnover_pct="0.49")
    ))
    targets = SleeveTargetSnapshot.from_payload(targets_payload(), deployment)
    broker = Phase3FakeBroker("PAPER", "10000", clock=lambda: NOW)
    quotes = {
        symbol: broker.set_quote(symbol, "100", spread_bps="0", as_of=NOW)
        for symbol in deployment.managed_symbols
    }
    with pytest.raises(DeploymentError, match="complete batch.*daily_turnover"):
        build_execution_plan(
            deployment, targets, account=broker.get_account(), positions=[], quotes=quotes,
            trading_date="2026-07-14",
        )


def test_equity_guardrails_use_previous_close_and_never_reset_silently():
    clock = FixedClock()
    broker = Phase3FakeBroker("PAPER", "10000", clock=clock)
    account = broker.get_account()
    with Ledger(":memory:", clock=clock) as ledger:
        with pytest.raises(LedgerConflict, match="not initialized"):
            ledger.observe_equity(account, "2026-07-14", allow_new_session=False)
        first = ledger.observe_equity(account, "2026-07-14", allow_new_session=True)
        assert first["day_start_equity"] == "10000"
        broker._cash = Decimal("10100")
        raised = ledger.observe_equity(
            broker.get_account(), "2026-07-14", allow_new_session=False
        )
        assert raised["high_water_equity"] == "10100"
        broker._cash = Decimal("9900")
        lower = ledger.observe_equity(
            broker.get_account(), "2026-07-14", allow_new_session=False
        )
        assert lower["day_start_equity"] == "10000"
        assert lower["high_water_equity"] == "10100"
        with pytest.raises(LedgerConflict, match="explicit operator"):
            ledger.observe_equity(
                broker.get_account(), "2026-07-15", allow_new_session=False
            )


def test_batch_approval_requires_exact_full_hash_and_is_immutable(tmp_path):
    clock, broker, data, db = broker_stack(tmp_path)
    deployment, targets = artifacts()
    with Ledger(db, clock=clock) as ledger:
        plan, _created = PaperExecutionService(
            ledger, broker, data, clock=clock
        ).preview(deployment, targets, confirm_new_equity_session=True)
        with pytest.raises(LedgerConflict, match="full 64"):
            ledger.approve_execution_batch(
                plan.batch_id, plan.plan_hash[:12], approved_by="op", reason="review"
            )
        with pytest.raises(LedgerConflict, match="does not match"):
            ledger.approve_execution_batch(
                plan.batch_id, "0" * 64, approved_by="op", reason="review"
            )
        ledger.approve_execution_batch(
            plan.batch_id, plan.plan_hash, approved_by="op", reason="review"
        )
        with pytest.raises(LedgerConflict, match="cannot be changed"):
            ledger.approve_execution_batch(
                plan.batch_id, plan.plan_hash, approved_by="other", reason="changed"
            )


def test_same_target_source_cannot_be_repriced_without_fresh_version(tmp_path):
    clock, broker, data, db = broker_stack(tmp_path)
    deployment, targets = artifacts()
    with Ledger(db, clock=clock) as ledger:
        service = PaperExecutionService(ledger, broker, data, clock=clock)
        service.preview(deployment, targets, confirm_new_equity_session=True)
        broker.set_quote("AAPL", "101", spread_bps="0", as_of=clock())
        with pytest.raises(LedgerConflict, match="fresh target_version"):
            service.preview(deployment, targets, confirm_new_equity_session=False)


def test_stored_plan_sql_tampering_is_detected(tmp_path):
    _clock, _broker, _data, db, plan = preview_approve(tmp_path)
    conn = sqlite3.connect(db)
    payload = json.loads(conn.execute(
        "SELECT plan_json FROM execution_batches WHERE batch_id = ?", (plan.batch_id,)
    ).fetchone()[0])
    payload["items"][0]["target_quantity"] = "999"
    conn.execute(
        "UPDATE execution_batches SET plan_json = ? WHERE batch_id = ?",
        (json.dumps(payload, sort_keys=True, separators=(",", ":")), plan.batch_id),
    )
    conn.commit()
    conn.close()
    with Ledger(db) as ledger:
        with pytest.raises(LedgerConflict, match="integrity"):
            ledger.load_execution_plan(plan.batch_id)


@pytest.mark.parametrize(
    "column,value",
    [
        ("source_hash", "0" * 64),
        ("account_id", "OTHER"),
        ("deployment_id", "other-deployment"),
        ("trading_date", "2026-07-15"),
        ("signal_at", "2026-07-12T20:05:00+00:00"),
    ],
)
def test_stored_plan_metadata_tampering_blocks_load_and_approval(tmp_path, column, value):
    clock, broker, data, db = broker_stack(tmp_path)
    deployment, targets = artifacts()
    with Ledger(db, clock=clock) as ledger:
        plan, _created = PaperExecutionService(
            ledger, broker, data, clock=clock
        ).preview(deployment, targets, confirm_new_equity_session=True)
    conn = sqlite3.connect(db)
    conn.execute(
        f"UPDATE execution_batches SET {column} = ? WHERE batch_id = ?",
        (value, plan.batch_id),
    )
    conn.commit()
    conn.close()
    with Ledger(db, clock=clock) as ledger:
        with pytest.raises(LedgerConflict, match="metadata"):
            ledger.load_execution_plan(plan.batch_id)
        with pytest.raises(LedgerConflict, match="metadata"):
            ledger.approve_execution_batch(
                plan.batch_id, plan.plan_hash, approved_by="op", reason="review"
            )


def test_preview_never_submits_and_execute_requires_approval(tmp_path):
    clock, broker, data, db = broker_stack(tmp_path)
    deployment, targets = artifacts()
    with Ledger(db, clock=clock) as ledger:
        service = PaperExecutionService(ledger, broker, data, clock=clock)
        plan, _created = service.preview(
            deployment, targets, confirm_new_equity_session=True
        )
        assert broker.submission_count == 0
        with pytest.raises(ExecutionBlocked, match="approved"):
            service.execute(plan.batch_id, operator="op", reason="not approved")
    assert broker.submission_count == 0


def test_approved_batch_executes_once_and_persists_complete(tmp_path):
    clock, broker, data, db, plan = preview_approve(tmp_path)
    with Ledger(db, clock=clock) as ledger:
        service = PaperExecutionService(ledger, broker, data, clock=clock)
        result = service.execute(plan.batch_id, operator="op", reason="paper test")
        assert result.status == "complete"
        assert broker.submission_count == 2
        assert ledger.get_control_state("PAPER")["armed"] is False
        assert ledger.get_execution_batch(plan.batch_id)["status"] == "complete"
        replay = service.execute(plan.batch_id, operator="op", reason="replay")
        assert replay.status == "complete"
        assert replay.reconciliation_clean is False
        assert broker.submission_count == 2

        settled = service.settle(
            plan.batch_id, operator="op", reason="verify terminal broker state"
        )
        assert settled.status == "complete"
        assert settled.reconciliation_clean is True
        audits = ledger.list_audit_events()
        assert any(
            event["event_type"] == "execution_batch_settlement_requested"
            for event in audits
        )


def test_submitted_batch_can_settle_after_fills_without_new_submission(tmp_path):
    clock, broker, data, db, plan = preview_approve(tmp_path, auto_fill=False)
    with Ledger(db, clock=clock) as ledger:
        submitted = PaperExecutionService(ledger, broker, data, clock=clock).execute(
            plan.batch_id, operator="op", reason="submit async paper orders"
        )
        assert submitted.status == "submitted"
        assert broker.submission_count == 1
    for order in list(broker.get_open_orders()):
        broker.fill_order(order.broker_order_id)
    with Ledger(db, clock=clock) as ledger:
        observed = PaperExecutionService(ledger, broker, data, clock=clock).settle(
            plan.batch_id, operator="op", reason="observe terminal fills"
        )
        assert observed.status == "submitted"
        assert broker.submission_count == 1
        resumed = PaperExecutionService(ledger, broker, data, clock=clock).execute(
            plan.batch_id, operator="op", reason="submit next serialized item"
        )
        assert resumed.status == "submitted"
        assert broker.submission_count == 2
    for order in list(broker.get_open_orders()):
        broker.fill_order(order.broker_order_id)
    with Ledger(db, clock=clock) as ledger:
        settled = PaperExecutionService(ledger, broker, data, clock=clock).settle(
            plan.batch_id, operator="op", reason="observe all terminal fills"
        )
        assert settled.status == "complete"
        assert broker.submission_count == 2
        assert ledger.get_control_state("PAPER")["armed"] is False


def test_never_started_batch_can_be_voided_but_source_version_stays_consumed(tmp_path):
    clock, broker, data, db = broker_stack(tmp_path)
    deployment, targets = artifacts()
    with Ledger(db, clock=clock) as ledger:
        plan, _created = PaperExecutionService(
            ledger, broker, data, clock=clock
        ).preview(deployment, targets, confirm_new_equity_session=True)
        voided = ledger.void_execution_batch(
            plan.batch_id, operator="op", reason="targets withdrawn"
        )
        assert voided["status"] == "voided"
        with pytest.raises(LedgerConflict, match="fresh target_version"):
            PaperExecutionService(ledger, broker, data, clock=clock).preview(
                deployment, targets, confirm_new_equity_session=False
            )


def test_acknowledgement_loss_resumes_without_duplicate_submission(tmp_path):
    clock, broker, data, db, plan = preview_approve(tmp_path)
    broker.raise_after_submit_once = True
    with Ledger(db, clock=clock) as ledger:
        service = PaperExecutionService(ledger, broker, data, clock=clock)
        with pytest.raises(Exception, match="acknowledgement loss"):
            service.execute(plan.batch_id, operator="op", reason="fault injection")
        assert broker.submission_count == 1
        assert ledger.get_execution_batch(plan.batch_id)["status"] == "executing"
        assert ledger.get_control_state("PAPER")["armed"] is False
    with Ledger(db, clock=clock) as ledger:
        resumed = PaperExecutionService(ledger, broker, data, clock=clock).execute(
            plan.batch_id, operator="op", reason="restart recovery"
        )
        assert resumed.status == "complete"
        assert broker.submission_count == 2


def test_execute_rechecks_price_collar_and_fails_without_submission(tmp_path):
    clock, broker, data, db, plan = preview_approve(tmp_path)
    broker.set_quote("AAPL", "110", spread_bps="0", as_of=clock())
    with Ledger(db, clock=clock) as ledger:
        result = PaperExecutionService(ledger, broker, data, clock=clock).execute(
            plan.batch_id, operator="op", reason="moved market"
        )
        assert result.status == "failed"
        assert broker.submission_count == 0
        assert any(
            violation.code == "PRICE_COLLAR"
            for item in result.results for violation in item.risk_violations
        )


def test_market_clock_closed_stale_or_wrong_day_blocks_before_submission(tmp_path):
    clock, broker, data, db, plan = preview_approve(tmp_path)
    broker.market_open = False
    with Ledger(db, clock=clock) as ledger:
        with pytest.raises(MarketDataError, match="closed"):
            PaperExecutionService(ledger, broker, data, clock=clock).execute(
                plan.batch_id, operator="op", reason="closed"
            )
    assert broker.submission_count == 0
    broker.market_open = True
    broker.clock_offset = timedelta(minutes=-2)
    with Ledger(db, clock=clock) as ledger:
        with pytest.raises(MarketDataError, match="stale"):
            PaperExecutionService(ledger, broker, data, clock=clock).execute(
                plan.batch_id, operator="op", reason="stale clock"
            )
    assert broker.submission_count == 0


def test_stale_quote_blocks_preview_and_future_clock_flag_is_not_trusted(tmp_path):
    clock, broker, data, db = broker_stack(tmp_path)
    deployment, targets = artifacts()
    broker.set_quote("AAPL", "100", spread_bps="0", as_of=NOW - timedelta(minutes=5))
    with Ledger(db, clock=clock) as ledger:
        with pytest.raises(MarketDataError, match="stale"):
            PaperExecutionService(ledger, broker, data, clock=clock).preview(
                deployment, targets, confirm_new_equity_session=True
            )
    broker.set_quote("AAPL", "100", spread_bps="0", as_of=NOW)
    broker.clock_offset = timedelta(seconds=10)
    with Ledger(db, clock=clock) as ledger:
        with pytest.raises(MarketDataError, match="future"):
            PaperExecutionService(ledger, broker, data, clock=clock).preview(
                deployment, targets, confirm_new_equity_session=True
            )


def test_preview_rechecks_quote_freshness_after_account_reconciliation(tmp_path):
    clock = FixedClock()

    class AgingBroker(Phase3FakeBroker):
        age_on_next_positions = False

        def get_positions(self):
            if self.age_on_next_positions:
                self.age_on_next_positions = False
                clock.advance(61)
            return super().get_positions()

    class AgingData(FakeMarketData):
        def get_quotes(self, symbols):
            result = super().get_quotes(symbols)
            self.broker.age_on_next_positions = True
            return result

    broker = AgingBroker("PAPER", "10000", clock=clock)
    broker.set_quote("AAPL", "100", spread_bps="0", as_of=clock())
    broker.set_quote("SPY", "100", spread_bps="0", as_of=clock())
    deployment, targets = artifacts()
    db = tmp_path / "aging-preview.sqlite3"
    with Ledger(db, clock=clock) as ledger:
        reconciler = Reconciler(ledger, broker, clock=clock)
        reconciler.bootstrap_positions()
        assert reconciler.reconcile().clean
        with pytest.raises(MarketDataError, match="stale"):
            PaperExecutionService(
                ledger, broker, AgingData(broker), clock=clock
            ).preview(deployment, targets, confirm_new_equity_session=True)
    assert broker.submission_count == 0


def test_daily_close_signal_cannot_execute_on_its_own_calendar_date(tmp_path):
    clock, broker, data, db = broker_stack(tmp_path)
    deployment, _targets = artifacts()
    same_day = SleeveTargetSnapshot.from_payload(
        targets_payload(signal_at=NOW.isoformat()), deployment
    )
    with Ledger(db, clock=clock) as ledger:
        with pytest.raises(DeploymentError, match="must precede"):
            PaperExecutionService(ledger, broker, data, clock=clock).preview(
                deployment, same_day, confirm_new_equity_session=True
            )


def test_daily_close_signal_must_be_timestamped_after_close(tmp_path):
    clock, broker, data, db = broker_stack(tmp_path)
    deployment, _targets = artifacts()
    intraday = SleeveTargetSnapshot.from_payload(
        targets_payload(signal_at="2026-07-13T10:00:00-04:00"), deployment
    )
    with Ledger(db, clock=clock) as ledger:
        with pytest.raises(MarketDataError, match="official session close"):
            PaperExecutionService(ledger, broker, data, clock=clock).preview(
                deployment, intraday, confirm_new_equity_session=True
            )


def test_signal_must_be_from_real_immediately_prior_session(tmp_path):
    clock, broker, data, db = broker_stack(tmp_path)
    deployment, targets = artifacts()
    broker.calendar_closed_dates = {"2026-07-13"}
    with Ledger(db, clock=clock) as ledger:
        with pytest.raises(MarketDataError, match="immediately preceding"):
            PaperExecutionService(ledger, broker, data, clock=clock).preview(
                deployment, targets, confirm_new_equity_session=True
            )


def test_early_close_signal_uses_exchange_calendar_not_hardcoded_1600(tmp_path):
    clock, broker, data, db = broker_stack(tmp_path)
    deployment, _targets = artifacts()
    broker.calendar_early_closes = {"2026-07-13": time(13, 0)}
    early_close_targets = SleeveTargetSnapshot.from_payload(
        targets_payload(signal_at="2026-07-13T13:05:00-04:00"), deployment
    )
    with Ledger(db, clock=clock) as ledger:
        plan, created = PaperExecutionService(
            ledger, broker, data, clock=clock
        ).preview(
            deployment, early_close_targets, confirm_new_equity_session=True
        )
    assert created and plan.signal_at == early_close_targets.signal_at


def test_network_latency_does_not_make_receipt_timestamp_look_future_dated(tmp_path):
    clock = FixedClock()

    class LatencyBroker(Phase3FakeBroker):
        first_account = True
        first_market_clock = True

        def get_account(self):
            if self.first_account:
                self.first_account = False
                clock.advance(3)
            return super().get_account()

        def get_market_clock(self):
            if self.first_market_clock:
                self.first_market_clock = False
                clock.advance(3)
            return super().get_market_clock()

    broker = LatencyBroker("PAPER", "10000", clock=clock)
    broker.set_quote("AAPL", "100", spread_bps="0", as_of=NOW)
    broker.set_quote("SPY", "100", spread_bps="0", as_of=NOW)
    data = FakeMarketData(broker)
    deployment, targets = artifacts()
    db = tmp_path / "latency.sqlite3"
    with Ledger(db, clock=clock) as ledger:
        reconciler = Reconciler(ledger, broker, clock=clock)
        reconciler.bootstrap_positions()
        assert reconciler.reconcile().clean
        broker.first_account = True
        plan, created = PaperExecutionService(
            ledger, broker, data, clock=clock
        ).preview(deployment, targets, confirm_new_equity_session=True)
        assert created and plan.trading_date == "2026-07-14"


def test_preview_refuses_even_a_known_reconciled_open_order(tmp_path):
    clock, broker, data, db = broker_stack(tmp_path, auto_fill=False)
    deployment, targets = artifacts()
    with Ledger(db, clock=clock) as ledger:
        reconciler = Reconciler(ledger, broker, clock=clock)
        assert reconciler.reconcile().clean
        oms = OrderManagementSystem(
            ledger, broker, PreTradeRiskEngine(clock=clock), clock=clock
        )
        oms.arm("create known active order")
        oms.process_intent(
            TargetPositionIntent(
                "PAPER", "old-batch", "AAPL", Decimal("1"), NOW,
                "old-v1", Decimal("100"), "known old order",
            ),
            quote=broker.get_quote("AAPL"), market_open=True,
            day_start_equity=Decimal("10000"), high_water_equity=Decimal("10000"),
            trading_date="2026-07-14",
        )
        oms.disarm("old order remains open")
        assert reconciler.reconcile().clean
        with pytest.raises(ExecutionBlocked, match="zero.*open orders"):
            PaperExecutionService(ledger, broker, data, clock=clock).preview(
                deployment, targets, confirm_new_equity_session=True
            )


def test_concurrent_execute_calls_are_serialized_and_submit_each_order_once(tmp_path):
    clock, broker, data, db, plan = preview_approve(tmp_path)
    barrier = threading.Barrier(2)
    results = []
    errors = []
    mutex = threading.Lock()

    def execute_from_connection():
        try:
            with Ledger(db, clock=clock) as ledger:
                barrier.wait(timeout=2)
                result = PaperExecutionService(
                    ledger, broker, data, clock=clock
                ).execute(plan.batch_id, operator="thread", reason="race test")
            with mutex:
                results.append(result.status)
        except Exception as exc:
            with mutex:
                errors.append(exc)

    threads = [threading.Thread(target=execute_from_connection) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert errors == []
    assert results == ["complete", "complete"]
    assert broker.submission_count == 2


def test_v2_ledger_migrates_without_inventing_equity_or_approval(tmp_path):
    path = tmp_path / "v2.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO metadata VALUES('schema_version', '2')")
    conn.commit()
    conn.close()
    with Ledger(path) as ledger:
        assert ledger.conn.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()[0] == "3"
        assert ledger.get_equity_guardrails("PAPER") is None
        assert ledger.list_execution_batches("PAPER") == []


def test_phase3_cli_requires_separate_network_approval_and_submit_confirmations(
    tmp_path,
):
    with pytest.raises(SystemExit) as preview_error:
        paper_cli.main([
            "preview", "--db", str(tmp_path / "x.sqlite3"),
            "--deployment", "d.json", "--targets", "t.json",
        ])
    assert preview_error.value.code == 2
    with pytest.raises(SystemExit) as approve_error:
        paper_cli.main([
            "approve", "--db", str(tmp_path / "x.sqlite3"),
            "--batch-id", "batch-x", "--plan-hash", "0" * 64,
            "--operator", "op", "--reason", "reviewed",
        ])
    assert approve_error.value.code == 2
    with pytest.raises(SystemExit) as execute_error:
        paper_cli.main([
            "execute", "--db", str(tmp_path / "x.sqlite3"),
            "--batch-id", "batch-x", "--operator", "op", "--reason", "run",
            "--confirm-paper-network",
    ])
    assert execute_error.value.code == 2
    with pytest.raises(SystemExit) as settle_error:
        paper_cli.main([
            "settle-batch", "--db", str(tmp_path / "x.sqlite3"),
            "--batch-id", "batch-x", "--operator", "op", "--reason", "check",
        ])
    assert settle_error.value.code == 2
    with pytest.raises(SystemExit) as void_error:
        paper_cli.main([
            "void-batch", "--db", str(tmp_path / "x.sqlite3"),
            "--batch-id", "batch-x", "--operator", "op", "--reason", "withdraw",
        ])
    assert void_error.value.code == 2


def test_phase3_cli_end_to_end_uses_preview_hash_and_paper_fake_only(
    tmp_path, monkeypatch, capsys
):
    clock, broker, data, db = broker_stack(tmp_path)
    deployment_path = tmp_path / "deployment.json"
    targets_path = tmp_path / "targets.json"
    deployment_path.write_text(json.dumps(deployment_payload()), encoding="utf-8")
    targets_path.write_text(json.dumps(targets_payload()), encoding="utf-8")

    class FixedService(PaperExecutionService):
        def __init__(self, ledger, selected_broker, selected_data):
            super().__init__(ledger, selected_broker, selected_data, clock=clock)

    class FixedLedger(Ledger):
        def __init__(self, path):
            super().__init__(path, clock=clock)

    monkeypatch.setattr(paper_cli, "_broker", lambda: broker)
    monkeypatch.setattr(paper_cli, "_market_data", lambda: data)
    monkeypatch.setattr(paper_cli, "PaperExecutionService", FixedService)
    monkeypatch.setattr(paper_cli, "Ledger", FixedLedger)
    rc = paper_cli.main([
        "preview", "--db", str(db),
        "--deployment", str(deployment_path), "--targets", str(targets_path),
        "--confirm-paper-network", "--confirm-new-equity-session",
    ])
    assert rc == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["submits_orders"] is False
    assert broker.submission_count == 0
    batch_id = preview["plan"]["batch_id"]
    plan_hash = preview["plan"]["plan_hash"]

    def forbidden():
        raise AssertionError("offline approval must not construct network adapters")

    monkeypatch.setattr(paper_cli, "_broker", forbidden)
    monkeypatch.setattr(paper_cli, "_market_data", forbidden)
    assert paper_cli.main([
        "approve", "--db", str(db), "--batch-id", batch_id,
        "--plan-hash", plan_hash, "--operator", "cli-operator",
        "--reason", "reviewed printed plan", "--confirm-approve-paper-batch",
    ]) == 0
    approved = json.loads(capsys.readouterr().out)
    assert approved["status"] == "approved"
    assert broker.submission_count == 0

    monkeypatch.setattr(paper_cli, "_broker", lambda: broker)
    monkeypatch.setattr(paper_cli, "_market_data", lambda: data)
    assert paper_cli.main([
        "execute", "--db", str(db), "--batch-id", batch_id,
        "--operator", "cli-operator", "--reason", "approved paper run",
        "--confirm-paper-network", "--confirm-submit-paper-orders",
    ]) == 0
    executed = json.loads(capsys.readouterr().out)
    assert executed["execution"]["status"] == "complete"
    assert broker.submission_count == 2
    assert paper_cli.main([
        "settle-batch", "--db", str(db), "--batch-id", batch_id,
        "--operator", "cli-operator", "--reason", "verify terminal state",
        "--confirm-paper-network",
    ]) == 0
    settled = json.loads(capsys.readouterr().out)
    assert settled["submits_new_orders"] is False
    assert settled["settlement"]["status"] == "complete"
    assert broker.submission_count == 2
