"""Phase-4 publisher, supervision, recovery, operations, and adversarial tests."""

from __future__ import annotations

import json
import hashlib
import multiprocessing
import os
import sqlite3
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.live.alpaca_paper import AlpacaAsset, AlpacaPaperConfig
from src.live.alerts import AlertManager, WebhookConfig
from src.live.backups import BackupManager
from src.live.broker import BrokerError
from src.live.deployment import DeploymentConfig
from src.live.execution import PaperExecutionService
from src.live.fake_broker import FakeBroker
from src.live.health import HealthReporter
from src.live.ledger import Ledger, LedgerConflict, LedgerError
from src.live.market_data import NEW_YORK, MarketCalendarDay, MarketDataProvider
from src.live.models import (
    BrokerOrder,
    Fill,
    OrderRequest,
    OrderStatus,
    Position,
    Quote,
    Side,
    TargetPositionIntent,
    json_safe,
)
from src.live.oms import ExecutionBlocked
from src.live.phase4_models import (
    HMACFileSigner,
    OperationMode,
    Phase4Error,
    Phase4Policy,
    PublishedTargetSnapshot,
    account_fingerprint,
    canonical_bytes,
)
from src.live.phase4_store import Phase4Store
from src.live import phase4_cli
from src.live.publisher import HistoricalDataBundle, StrategyTargetPublisher
from src.live.reconcile import Reconciler
from src.live.scheduler import SupervisedMonthlyScheduler
from src.live.soak import PaperSoakReporter
from src.live.streaming import (
    AlpacaPaperTradeUpdateStream,
    OrderStreamEvent,
    OrderStreamSupervisor,
    STREAM_NAME,
)
from src.live.supervisor import Phase4Supervisor


DECISION = date(2026, 6, 30)
PUBLISH_TIME = datetime(2026, 6, 30, 20, 5, tzinfo=timezone.utc)
EXECUTION_TIME = datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc)


class FixedClock:
    def __init__(self, value):
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


class Phase4FakeBroker(FakeBroker):
    market_open = True
    closed_dates: set[str] = set()
    early_closes: dict[str, time] = {}

    def get_market_clock(self):
        return ClockResponse(self.clock(), self.market_open)

    def get_asset(self, symbol):
        return assets()[symbol]

    def get_market_calendar(self, start, end):
        rows = []
        cursor = start
        while cursor <= end:
            if cursor.weekday() < 5 and cursor.isoformat() not in self.closed_dates:
                close = self.early_closes.get(cursor.isoformat(), time(16, 0))
                rows.append(MarketCalendarDay(
                    cursor,
                    datetime.combine(cursor, time(9, 30), tzinfo=NEW_YORK),
                    datetime.combine(cursor, close, tzinfo=NEW_YORK),
                ))
            cursor += timedelta(days=1)
        return tuple(rows)


class FakeQuotes(MarketDataProvider):
    def __init__(self, broker):
        self.broker = broker

    def get_quotes(self, symbols):
        return {symbol: self.broker.get_quote(symbol) for symbol in symbols}


def deployment() -> DeploymentConfig:
    symbols = list(StrategyTargetPublisher.required_tradable_symbols())
    return DeploymentConfig.from_payload({
        "schema_version": 1,
        "deployment_id": "phase4-test",
        "account_id": "PAPER",
        "allocation_policy": "rebalance_to_deployment_weights",
        "execution_policy": "next_session_regular_hours_market",
        "managed_symbols": symbols,
        "sleeve_weights": {
            "momentum": "0.60", "sector_rotation": "0.35", "regime_switch": "0.05",
        },
        "risk_limits": {
            "max_order_notional": "25000",
            "max_gross_exposure_pct": "1.00",
            "max_symbol_exposure_pct": "0.25",
            "max_daily_turnover_pct": "2.00",
            "max_daily_loss_pct": "0.02",
            "max_drawdown_pct": "0.10",
            "max_open_orders": 20,
            "min_cash_buffer": "0",
            "quote_max_age_seconds": 30,
            "account_max_age_seconds": 30,
            "max_signal_age_seconds": 604800,
            "price_collar_bps": "200",
            "allow_fractional_shares": False,
            "require_market_open": True,
        },
        "max_batch_orders": 20,
    })


def policy(mode=OperationMode.OBSERVE, **overrides) -> Phase4Policy:
    values = dict(
        system_id="phase4-test",
        mode=mode,
        publisher_identity="pytest-publisher",
        require_signing=True,
        signing_key_id="pytest-key-v1",
        max_quote_spread_bps=Decimal("50"),
        max_price_deviation_bps=Decimal("500"),
        dirty_worktree_policy="allow_labelled",
        min_trade_notional=Decimal("1"),
    )
    values.update(overrides)
    return Phase4Policy(**values)


def signer(tmp_path) -> HMACFileSigner:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "signing.key"
    path.write_bytes(b"phase4-test-key-material-32-bytes-minimum-123456789")
    path.chmod(0o600)
    return HMACFileSigner(path, "pytest-key-v1")


def market_bundle(
    *, decision_date=DECISION, retrieved_at=PUBLISH_TIME, stale_symbol=None, carried=False
) -> HistoricalDataBundle:
    calendar = pd.bdate_range(end=pd.Timestamp(decision_date), periods=300)
    frames = {}
    symbols = StrategyTargetPublisher.required_data_symbols()
    stocks = list(config.MEAN_REVERSION_UNIVERSE)
    sectors = list(config.SECTOR_ETFS)
    for index, symbol in enumerate(symbols):
        if symbol in stocks:
            rank = stocks.index(symbol)
            growth = 0.05 + rank * 0.01
        elif symbol in sectors:
            rank = sectors.index(symbol)
            growth = 0.03 + rank * 0.015
        else:
            growth = 0.25
        close = np.linspace(100.0, 100.0 * (1.0 + growth), len(calendar))
        frame = pd.DataFrame({
            "Open": close,
            "High": close * 1.001,
            "Low": close * 0.999,
            "Close": close,
            "Volume": np.full(len(calendar), 1_000_000.0),
        }, index=calendar)
        if stale_symbol == symbol:
            frame = frame.iloc[:-1]
        if carried and symbol == symbols[0]:
            frame.iloc[-1, frame.columns.get_loc("Volume")] = 0
            frame.iloc[-1, :4] = frame.iloc[-2, :4]
        frames[symbol] = frame
    return HistoricalDataBundle(
        frames=frames,
        calendar=calendar,
        retrieved_at=retrieved_at,
        source_id="deterministic-test-data",
        input_file_hashes={"fixture": "a" * 64},
    )


def assets(*, changed_symbol=None, wrong_symbol=None) -> dict[str, AlpacaAsset]:
    result = {}
    for symbol in StrategyTargetPublisher.required_data_symbols():
        result[symbol] = AlpacaAsset(
            asset_id=f"asset-{symbol}-changed" if symbol == changed_symbol else f"asset-{symbol}",
            symbol="WRONG" if symbol == wrong_symbol else symbol,
            asset_class="us_equity",
            status="active",
            tradable=True,
            exchange="NASDAQ",
            fractionable=True,
            shortable=True,
            easy_to_borrow=True,
            marginable=True,
            name=f"Issuer {symbol}",
        )
    return result


def published(
    tmp_path,
    mode=OperationMode.OBSERVE,
    *,
    decision_date=DECISION,
    publish_time=PUBLISH_TIME,
    execution_date=date(2026, 7, 1),
    **publish_overrides,
):
    deployment_value = deployment()
    policy_value = policy(mode)
    signer_value = signer(tmp_path)
    publisher = StrategyTargetPublisher(
        deployment_value, policy_value, repo_root=Path.cwd(), signer=signer_value,
        clock=lambda: publish_time,
    )
    account = Phase4FakeBroker("PAPER", "100000", clock=lambda: publish_time).get_account()
    snapshot = publisher.publish(
        decision_day=MarketCalendarDay(
            decision_date,
            datetime.combine(decision_date, time(13, 30), tzinfo=timezone.utc),
            datetime.combine(decision_date, time(20, 0), tzinfo=timezone.utc),
        ),
        execution_day=MarketCalendarDay(
            execution_date,
            datetime.combine(execution_date, time(13, 30), tzinfo=timezone.utc),
            datetime.combine(execution_date, time(20, 0), tzinfo=timezone.utc),
        ),
        account=account,
        positions=[],
        assets=publish_overrides.pop("asset_rows", assets()),
        market_data=publish_overrides.pop(
            "bundle", market_bundle(decision_date=decision_date, retrieved_at=publish_time)
        ),
        **publish_overrides,
    )
    return deployment_value, policy_value, signer_value, publisher, snapshot


def execution_stack(tmp_path, mode):
    deployment_value, policy_value, signer_value, publisher, snapshot = published(tmp_path, mode)
    clock = FixedClock(EXECUTION_TIME)
    broker = Phase4FakeBroker("PAPER", "100000", auto_fill=True, clock=clock)
    references = {
        row["symbol"]: row["research_reference_price"]
        for row in snapshot.content["required_target_deltas"]
    }
    for symbol, price in references.items():
        broker.set_quote(symbol, price, spread_bps="2", as_of=clock())
    db = tmp_path / f"{mode.value}.sqlite3"
    with Ledger(db, clock=clock) as ledger:
        Reconciler(ledger, broker, clock=clock).bootstrap_positions()
        assert Reconciler(ledger, broker, clock=clock).reconcile().clean
        store = Phase4Store(ledger)
        store.publish_snapshot(snapshot)
    return deployment_value, policy_value, signer_value, publisher, snapshot, clock, broker, db


def _shadow_rebalance_worker(
    db_path: str,
    snapshot_path: str,
    deployment_path: str,
    policy_path: str,
    key_path: str,
    execution_time: str,
    result_path: str,
) -> None:
    """Cold-process shadow worker used to prove durable restart semantics."""
    clock = FixedClock(datetime.fromisoformat(execution_time))
    deployment_value = DeploymentConfig.from_file(deployment_path)
    policy_value = Phase4Policy.from_file(policy_path)
    signer_value = HMACFileSigner(key_path, policy_value.signing_key_id)
    snapshot = PublishedTargetSnapshot.from_file(snapshot_path)
    broker = Phase4FakeBroker("PAPER", "100000", auto_fill=False, clock=clock)
    for row in snapshot.content["required_target_deltas"]:
        broker.set_quote(
            row["symbol"], row["research_reference_price"], spread_bps="2", as_of=clock()
        )
    with Ledger(db_path, clock=clock) as ledger:
        reconciler = Reconciler(ledger, broker, clock=clock)
        if not ledger.positions_bootstrapped("PAPER"):
            reconciler.bootstrap_positions()
        if not reconciler.reconcile().clean:
            raise AssertionError("shadow worker reconciliation was not clean")
        store = Phase4Store(ledger)
        schedule, schedule_created = store.claim_schedule(
            snapshot.content["decision_session"],
            snapshot.content["expected_execution_session"],
        )
        _snapshot_row, snapshot_created = store.publish_snapshot(snapshot)
        publisher = StrategyTargetPublisher(
            deployment_value,
            policy_value,
            repo_root=Path.cwd(),
            signer=signer_value,
            clock=clock,
        )
        plan, plan_created = Phase4Supervisor(
            ledger,
            deployment_value,
            policy_value,
            publisher,
            broker,
            FakeQuotes(broker),
            signer=signer_value,
            clock=clock,
        ).prepare_plan(snapshot.snapshot_id, confirm_new_equity_session=True)
        store.update_schedule(
            schedule["run_id"], "published", snapshot_id=snapshot.snapshot_id
        )
        Path(result_path).write_text(json.dumps({
            "schedule_created": schedule_created,
            "snapshot_created": snapshot_created,
            "plan_created": plan_created,
            "batch_id": plan.batch_id,
            "status": ledger.get_execution_batch(plan.batch_id)["status"],
            "orders": len(ledger.list_orders()),
        }), encoding="utf-8")


def test_registered_publisher_exact_strategy_and_aggregation(tmp_path):
    deployment_value, policy_value, signer_value, publisher, snapshot = published(tmp_path)
    snapshot.verify(policy_value, signer_value, now=PUBLISH_TIME)
    content = snapshot.content
    assert set(content["sleeve_level_targets"]) == {
        "momentum", "sector_rotation", "regime_switch"
    }
    assert "mean_reversion" not in json.dumps(content)
    momentum = content["sleeve_level_targets"]["momentum"]
    assert len([weight for weight in momentum.values() if Decimal(weight) > 0]) == 5
    assert set(Decimal(value) for value in momentum.values()) <= {Decimal("0"), Decimal("0.2")}
    rotation = content["sleeve_level_targets"]["sector_rotation"]
    regime = content["sleeve_level_targets"]["regime_switch"]
    assert rotation == regime
    selected = [symbol for symbol, weight in rotation.items() if Decimal(weight) > 0]
    assert len(selected) == 2
    for symbol in selected:
        assert Decimal(content["aggregated_ticker_targets"][symbol]) == Decimal("0.20")
    assert content["complete_managed_universe"]["survivorship_biased"] is True
    assert set(deployment_value.managed_symbols) == set(
        content["complete_managed_universe"]["tradable_symbols"]
    )


def test_snapshot_signature_tamper_expiry_and_wrong_account_fail_closed(tmp_path):
    deployment_value, policy_value, signer_value, publisher, snapshot = published(tmp_path)
    payload = snapshot.to_payload()
    payload["content"]["aggregated_ticker_targets"]["AAPL"] = "0.99"
    with pytest.raises(Phase4Error, match="content hash"):
        PublishedTargetSnapshot.from_payload(payload)
    with pytest.raises(Phase4Error, match="expired"):
        snapshot.verify(
            policy_value, signer_value,
            now=PUBLISH_TIME + timedelta(seconds=policy_value.snapshot_ttl_seconds + 1),
        )
    changed = dict(snapshot.content)
    changed["account_id_fingerprint"] = account_fingerprint("OTHER", policy_value.system_id)
    wrong = PublishedTargetSnapshot.create(changed, signer=signer_value)
    with pytest.raises(Phase4Error, match="different paper account"):
        publisher.to_execution_targets(wrong)
    snapshot_path = tmp_path / "round-trip.json"
    snapshot_path.write_text(json.dumps(snapshot.to_payload()), encoding="utf-8")
    reloaded = PublishedTargetSnapshot.from_file(snapshot_path)
    assert reloaded.to_payload() == snapshot.to_payload()
    reloaded.verify(policy_value, signer_value, now=PUBLISH_TIME)
    extended = json.loads(json.dumps(snapshot.content))
    extended["expiration_time"] = (
        PUBLISH_TIME + timedelta(seconds=policy_value.snapshot_ttl_seconds + 1)
    ).isoformat()
    with pytest.raises(Phase4Error, match="expiration exceeds"):
        PublishedTargetSnapshot.create(extended, signer=signer_value).verify(
            policy_value, signer_value, now=PUBLISH_TIME
        )
    with pytest.raises(Phase4Error, match="7 calendar days"):
        replace(policy_value, snapshot_ttl_seconds=604801, max_snapshot_age_seconds=604801)


def test_invalid_signature_unsigned_and_key_permissions_are_rejected(tmp_path):
    deployment_value, policy_value, signer_value, publisher, snapshot = published(tmp_path)
    payload = snapshot.to_payload()
    payload["signature"]["value"] = "0" * 64
    parsed = PublishedTargetSnapshot.from_payload(payload)
    with pytest.raises(Phase4Error, match="signature verification"):
        parsed.verify(policy_value, signer_value, now=PUBLISH_TIME)
    unsigned = PublishedTargetSnapshot.create(snapshot.content)
    with pytest.raises(Phase4Error, match="Unsigned"):
        unsigned.verify(policy_value, None, now=PUBLISH_TIME)
    insecure = tmp_path / "insecure.key"
    insecure.write_bytes(b"x" * 32)
    insecure.chmod(0o644)
    with pytest.raises(Phase4Error, match="permissions"):
        HMACFileSigner(insecure, "pytest-key-v1")


def test_strategy_universe_policy_and_input_drift_are_rejected(tmp_path):
    deployment_value, policy_value, signer_value, publisher, snapshot = published(tmp_path)
    changed = json.loads(json.dumps(snapshot.content))
    changed["strategy_parameters"]["momentum"]["lookback_trading_days"] = 127
    with pytest.raises(Phase4Error, match="strategy parameters"):
        publisher.to_execution_targets(PublishedTargetSnapshot.create(changed, signer=signer_value))

    changed = json.loads(json.dumps(snapshot.content))
    changed["complete_managed_universe"]["tradable_symbols"] = changed[
        "complete_managed_universe"
    ]["tradable_symbols"][:-1]
    from src.live.phase4_models import content_hash
    changed["universe_snapshot_hash"] = content_hash(changed["complete_managed_universe"])
    with pytest.raises(Phase4Error, match="managed universe"):
        publisher.to_execution_targets(PublishedTargetSnapshot.create(changed, signer=signer_value))

    changed_policy = replace(policy_value, max_quote_spread_bps=Decimal("40"))
    changed_publisher = StrategyTargetPublisher(
        deployment_value, changed_policy, repo_root=Path.cwd(), signer=signer_value,
        clock=lambda: PUBLISH_TIME,
    )
    with pytest.raises(Phase4Error, match="policy has drifted"):
        changed_publisher.to_execution_targets(snapshot)

    input_path = tmp_path / "publisher-input.json"
    input_path.write_text('{"version":1}', encoding="utf-8")
    import hashlib
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    _d, _p, _s, input_publisher, input_snapshot = published(
        tmp_path / "input-snapshot",
        input_file_hashes={str(input_path.resolve()): digest},
    )
    input_path.write_text('{"version":2}', encoding="utf-8")
    with pytest.raises(Phase4Error, match="input changed"):
        input_publisher.to_execution_targets(input_snapshot)


def test_duplicate_publication_restart_and_immutability(tmp_path):
    _d, _p, _s, _publisher, snapshot = published(tmp_path)
    path = tmp_path / "ledger.sqlite3"
    with Ledger(path, clock=lambda: PUBLISH_TIME) as ledger:
        store = Phase4Store(ledger)
        row, created = store.publish_snapshot(snapshot)
        assert created
        same, created = store.publish_snapshot(snapshot)
        assert not created and same["snapshot_id"] == row["snapshot_id"]
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            ledger.conn.execute(
                "UPDATE target_snapshots SET mode='shadow' WHERE snapshot_id=?",
                (snapshot.snapshot_id,),
            )
    with Ledger(path, clock=lambda: PUBLISH_TIME) as ledger:
        assert Phase4Store(ledger).load_snapshot(snapshot.snapshot_id).content_hash == snapshot.content_hash


def test_duplicate_decision_changed_snapshot_refused(tmp_path):
    _d, _p, signer_value, _publisher, snapshot = published(tmp_path)
    changed = dict(snapshot.content)
    changed["publisher_identity"] = "another-publisher"
    other = PublishedTargetSnapshot.create(changed, signer=signer_value)
    with Ledger(":memory:", clock=lambda: PUBLISH_TIME) as ledger:
        store = Phase4Store(ledger)
        store.publish_snapshot(snapshot)
        with pytest.raises(LedgerConflict, match="Decision session"):
            store.publish_snapshot(other)


def test_stale_carried_data_universe_asset_and_corporate_action_drift_fail(tmp_path):
    deployment_value, policy_value, signer_value, publisher, snapshot = published(tmp_path)
    account = Phase4FakeBroker("PAPER", "100000", clock=lambda: PUBLISH_TIME).get_account()
    decision = MarketCalendarDay(
        DECISION, datetime(2026, 6, 30, 13, 30, tzinfo=timezone.utc),
        datetime(2026, 6, 30, 20, 0, tzinfo=timezone.utc),
    )
    execution = MarketCalendarDay(
        date(2026, 7, 1), datetime(2026, 7, 1, 13, 30, tzinfo=timezone.utc),
        datetime(2026, 7, 1, 20, 0, tzinfo=timezone.utc),
    )
    with pytest.raises(Phase4Error, match="stale"):
        publisher.publish(
            decision_day=decision, execution_day=execution, account=account, positions=[],
            assets=assets(), market_data=market_bundle(stale_symbol="AAPL"),
        )
    with pytest.raises(Phase4Error, match="carried-forward|non-finite"):
        publisher.publish(
            decision_day=decision, execution_day=execution, account=account, positions=[],
            assets=assets(), market_data=market_bundle(carried=True),
        )
    with pytest.raises(Phase4Error, match="ticker change"):
        publisher.publish(
            decision_day=decision, execution_day=execution, account=account, positions=[],
            assets=assets(wrong_symbol="AAPL"), market_data=market_bundle(),
        )
    with pytest.raises(Phase4Error, match="identity changed"):
        publisher.publish(
            decision_day=decision, execution_day=execution, account=account, positions=[],
            assets=assets(changed_symbol="AAPL"), market_data=market_bundle(),
            previous_snapshot=snapshot,
        )


def test_dirty_code_policy_and_early_close(monkeypatch, tmp_path):
    deployment_value = deployment()
    policy_value = policy(dirty_worktree_policy="reject")
    signer_value = signer(tmp_path)
    publisher = StrategyTargetPublisher(
        deployment_value, policy_value, repo_root=Path.cwd(), signer=signer_value,
        clock=lambda: datetime(2026, 6, 30, 17, 5, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        "src.live.publisher.git_state", lambda _root: ("a" * 40, True, (" M file.py",))
    )
    account = Phase4FakeBroker("PAPER", "100000", clock=publisher.clock).get_account()
    early = MarketCalendarDay(
        DECISION, datetime(2026, 6, 30, 13, 30, tzinfo=timezone.utc),
        datetime(2026, 6, 30, 17, 0, tzinfo=timezone.utc),
    )
    next_day = MarketCalendarDay(
        date(2026, 7, 1), datetime(2026, 7, 1, 13, 30, tzinfo=timezone.utc),
        datetime(2026, 7, 1, 20, 0, tzinfo=timezone.utc),
    )
    with pytest.raises(Phase4Error, match="Dirty worktree"):
        publisher.publish(
            decision_day=early, execution_day=next_day, account=account, positions=[],
            assets=assets(), market_data=market_bundle(),
        )
    allowed = replace(policy_value, dirty_worktree_policy="allow_labelled")
    publisher = StrategyTargetPublisher(
        deployment_value, allowed, repo_root=Path.cwd(), signer=signer_value,
        clock=lambda: datetime(2026, 6, 30, 17, 5, tzinfo=timezone.utc),
    )
    snapshot = publisher.publish(
        decision_day=early, execution_day=next_day, account=account, positions=[],
        assets=assets(), market_data=market_bundle(
            retrieved_at=datetime(2026, 6, 30, 17, 5, tzinfo=timezone.utc)
        ),
    )
    assert snapshot.content["decision_close_utc"] == early.close_at.isoformat()


def test_scheduler_early_close_unexpected_closure_miss_and_duplicate(tmp_path):
    clock = FixedClock(datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc))
    broker = Phase4FakeBroker("PAPER", clock=clock)
    broker.closed_dates = {"2026-06-30"}
    broker.early_closes = {"2026-06-29": time(13, 0)}
    with Ledger(tmp_path / "schedule.sqlite3", clock=clock) as ledger:
        scheduler = SupervisedMonthlyScheduler(broker, Phase4Store(ledger), clock=clock)
        due = scheduler.for_month(2026, 6)
        assert due.decision_day.trading_date == date(2026, 6, 29)
        assert due.decision_day.close_at.astimezone(NEW_YORK).time() == time(13, 0)
        first, created = Phase4Store(ledger).claim_schedule(
            "2026-06-29", "2026-07-01"
        )
        second, created_again = Phase4Store(ledger).claim_schedule(
            "2026-06-29", "2026-07-01"
        )
        assert created and not created_again and first["run_id"] == second["run_id"]
        with pytest.raises(Phase4Error, match="manual catch-up"):
            scheduler.claim_due(confirm_manual_catch_up=False)

    late_clock = FixedClock(datetime(2026, 7, 2, 21, 0, tzinfo=timezone.utc))
    late_broker = Phase4FakeBroker("PAPER", clock=late_clock)
    with Ledger(tmp_path / "missed.sqlite3", clock=late_clock) as ledger:
        scheduler = SupervisedMonthlyScheduler(
            late_broker, Phase4Store(ledger), clock=late_clock
        )
        with pytest.raises(Phase4Error, match="execution window has closed"):
            scheduler.claim_due(confirm_manual_catch_up=True)
        assert Phase4Store(ledger).list_schedule_runs()[0]["status"] == "failed"


def test_publish_backup_failure_preserves_published_schedule_and_restart_safety(
    monkeypatch, tmp_path
):
    deployment_value = deployment()
    policy_value = policy(OperationMode.OBSERVE)
    signer_value = signer(tmp_path / "key")
    publisher = StrategyTargetPublisher(
        deployment_value,
        policy_value,
        repo_root=Path.cwd(),
        signer=signer_value,
        clock=lambda: PUBLISH_TIME,
    )
    broker = Phase4FakeBroker("PAPER", "100000", clock=lambda: PUBLISH_TIME)
    broker.drain_request_ids = lambda: ()
    db = tmp_path / "publish-backup-failure.sqlite3"
    deployment_path = tmp_path / "deployment.json"
    policy_path = tmp_path / "policy.json"
    deployment_path.write_text(
        json.dumps(json_safe(deployment_value.to_payload())), encoding="utf-8"
    )
    policy_path.write_text(json.dumps(policy_value.to_payload()), encoding="utf-8")
    with Ledger(db, clock=lambda: PUBLISH_TIME) as ledger:
        Reconciler(ledger, broker, clock=lambda: PUBLISH_TIME).bootstrap_positions()

    class PublishLedger(Ledger):
        def __init__(self, path):
            super().__init__(path, clock=lambda: PUBLISH_TIME)

    class StaticHistory:
        def __init__(self, **_kwargs):
            pass

        def load(self, symbols, decision_session):
            assert symbols == publisher.required_data_symbols()
            assert decision_session == DECISION
            return market_bundle()

    backup_rows = []

    def backup_then_fail(ledger, _policy, args, _alerts):
        schedule = ledger.conn.execute("SELECT * FROM scheduler_runs").fetchone()
        assert schedule["status"] == "published"
        assert schedule["snapshot_id"]
        backup = BackupManager(ledger, tmp_path / "backups").create(
            (args.deployment, args.policy)
        )
        backup_rows.append(backup)
        raise Phase4Error("injected automatic backup failure")

    monkeypatch.setattr(phase4_cli, "Ledger", PublishLedger)
    monkeypatch.setattr(
        phase4_cli, "_load", lambda _args: (deployment_value, policy_value, signer_value)
    )
    monkeypatch.setattr(
        phase4_cli, "_paper_stack", lambda: (broker, FakeQuotes(broker))
    )
    monkeypatch.setattr(phase4_cli, "_publisher", lambda *_args, **_kwargs: publisher)
    monkeypatch.setattr(phase4_cli, "ResearchHistoricalDataSource", StaticHistory)
    monkeypatch.setattr(phase4_cli, "_automatic_backup", backup_then_fail)
    args = phase4_cli.argparse.Namespace(
        db=str(db),
        deployment=str(deployment_path),
        policy=str(policy_path),
        cache_dir=str(tmp_path / "cache"),
        snapshot_dir=str(tmp_path / "snapshots"),
        mode=OperationMode.OBSERVE.value,
        confirm_manual_catch_up=False,
    )

    with pytest.raises(Phase4Error, match="injected automatic backup failure"):
        phase4_cli._publish(args)

    with Ledger(db, clock=lambda: PUBLISH_TIME) as ledger:
        schedule = ledger.conn.execute("SELECT * FROM scheduler_runs").fetchone()
        assert schedule["status"] == "published"
        assert schedule["snapshot_id"]
        assert ledger.conn.execute("SELECT COUNT(*) FROM target_snapshots").fetchone()[0] == 1
    assert len(backup_rows) == 1
    backup_db = Path(backup_rows[0]["backup_path"]) / "execution-ledger.sqlite3"
    connection = sqlite3.connect(f"file:{backup_db}?mode=ro&immutable=1", uri=True)
    try:
        assert connection.execute("SELECT status FROM scheduler_runs").fetchone()[0] == "published"
    finally:
        connection.close()

    with pytest.raises(Phase4Error, match="already published"):
        phase4_cli._publish(args)
    with Ledger(db, clock=lambda: PUBLISH_TIME) as ledger:
        assert ledger.conn.execute("SELECT COUNT(*) FROM target_snapshots").fetchone()[0] == 1
        assert ledger.conn.execute("SELECT status FROM scheduler_runs").fetchone()[0] == "published"


def test_observe_and_shadow_can_never_submit(tmp_path):
    for mode in (OperationMode.OBSERVE, OperationMode.SHADOW):
        values = execution_stack(tmp_path / mode.value, mode)
        deployment_value, policy_value, signer_value, publisher, snapshot, clock, broker, db = values
        with Ledger(db, clock=clock) as ledger:
            supervisor = Phase4Supervisor(
                ledger, deployment_value, policy_value, publisher, broker, FakeQuotes(broker),
                signer=signer_value, clock=clock,
            )
            plan, _created = supervisor.prepare_plan(
                snapshot.snapshot_id, confirm_new_equity_session=True
            )
            if mode == OperationMode.OBSERVE:
                assert plan is None and broker.submission_count == 0
            else:
                assert ledger.get_execution_batch(plan.batch_id)["status"] == "voided"
                with pytest.raises(ExecutionBlocked, match="non-submitting"):
                    PaperExecutionService(
                        ledger, broker, FakeQuotes(broker), clock=clock
                    ).execute(plan.batch_id, operator="op", reason="bypass attempt")
                assert broker.submission_count == 0


def test_phase4_ledger_profile_blocks_legacy_preview_and_approval(tmp_path):
    values = execution_stack(tmp_path, OperationMode.PAPER_MANUAL)
    deployment_value, policy_value, signer_value, publisher, snapshot, clock, broker, db = values
    with Ledger(db, clock=clock) as ledger:
        assert ledger.execution_profile() == "phase4"
        with pytest.raises(ExecutionBlocked, match="Legacy Phase-3 preview"):
            PaperExecutionService(
                ledger, broker, FakeQuotes(broker), clock=clock
            ).preview(
                deployment_value,
                publisher.to_execution_targets(snapshot),
                confirm_new_equity_session=True,
            )
        supervisor = Phase4Supervisor(
            ledger, deployment_value, policy_value, publisher, broker, FakeQuotes(broker),
            signer=signer_value, clock=clock,
        )
        plan, _created = supervisor.prepare_plan(
            snapshot.snapshot_id, confirm_new_equity_session=True
        )
        with pytest.raises(LedgerConflict, match="Phase4Supervisor"):
            ledger.approve_execution_batch(
                plan.batch_id, plan.plan_hash, approved_by="legacy", reason="bypass"
            )
        supervisor.approve(plan.batch_id, plan.plan_hash, operator="op", reason="reviewed")
        with pytest.raises(sqlite3.IntegrityError, match="execution profile is immutable"):
            ledger.conn.execute(
                "UPDATE metadata SET value='phase3' WHERE key='execution_profile'"
            )
        ledger.conn.rollback()
        assert ledger.execution_profile() == "phase4"


def test_manual_paper_requires_approval_stream_recovery_and_all_submissions_use_oms(tmp_path):
    values = execution_stack(tmp_path, OperationMode.PAPER_MANUAL)
    deployment_value, policy_value, signer_value, publisher, snapshot, clock, broker, db = values
    with Ledger(db, clock=clock) as ledger:
        supervisor = Phase4Supervisor(
            ledger, deployment_value, policy_value, publisher, broker, FakeQuotes(broker),
            signer=signer_value, clock=clock,
        )
        plan, _created = supervisor.prepare_plan(
            snapshot.snapshot_id, confirm_new_equity_session=True
        )
        with pytest.raises(ExecutionBlocked, match="Phase4Supervisor"):
            # Direct Phase-3 path cannot enter any Phase-4 authorization path.
            PaperExecutionService(ledger, broker, FakeQuotes(broker), clock=clock).execute(
                plan.batch_id, operator="op", reason="approval bypass"
            )
        supervisor.approve(
            plan.batch_id, plan.plan_hash, operator="op", reason="reviewed paper plan"
        )
        with pytest.raises(ExecutionBlocked, match="Phase4Supervisor"):
            PaperExecutionService(ledger, broker, FakeQuotes(broker), clock=clock).execute(
                plan.batch_id, operator="op", reason="post-approval bypass"
            )
        with pytest.raises(Phase4Error, match="healthy recovered order stream"):
            supervisor.run_paper(plan.batch_id, operator="op", reason="submit paper")
        Phase4Store(ledger).set_stream_state(
            STREAM_NAME, connected=True, recovering=False, recovery_completed=True
        )
        clock.advance(46)
        with pytest.raises(Phase4Error, match="stream lease is stale"):
            supervisor.run_paper(plan.batch_id, operator="op", reason="submit paper")
        for row in snapshot.content["required_target_deltas"]:
            broker.set_quote(
                row["symbol"], row["research_reference_price"], spread_bps="2", as_of=clock()
            )
        Phase4Store(ledger).set_stream_state(
            STREAM_NAME, connected=True, recovering=False
        )
        result = supervisor.run_paper(plan.batch_id, operator="op", reason="submit paper")
        assert result.status == "complete"
        assert broker.submission_count > 0
        assert all(row["client_order_id"].startswith("wslab-") for row in ledger.list_orders())


def test_phase4_policy_is_rechecked_with_current_equity_before_execution(tmp_path, monkeypatch):
    values = execution_stack(tmp_path, OperationMode.PAPER_MANUAL)
    deployment_value, policy_value, signer_value, publisher, snapshot, clock, broker, db = values
    with Ledger(db, clock=clock) as ledger:
        supervisor = Phase4Supervisor(
            ledger, deployment_value, policy_value, publisher, broker, FakeQuotes(broker),
            signer=signer_value, clock=clock,
        )
        plan, _created = supervisor.prepare_plan(
            snapshot.snapshot_id, confirm_new_equity_session=True
        )
        quotes = FakeQuotes(broker).get_quotes(deployment_value.managed_symbols)
        with pytest.raises(Phase4Error, match="max_cash_deployment_pct"):
            supervisor._validate_phase4_plan(
                plan, Decimal("50000"),
                snapshot.content["expected_execution_session"], quotes,
            )

        supervisor.approve(
            plan.batch_id, plan.plan_hash, operator="op", reason="reviewed"
        )
        Phase4Store(ledger).set_stream_state(
            STREAM_NAME, connected=True, recovering=False, recovery_completed=True
        )

        def runtime_policy_failure(*_args, **_kwargs):
            raise Phase4Error("execution-time Phase-4 policy rejection")

        monkeypatch.setattr(supervisor, "_validate_phase4_plan", runtime_policy_failure)
        with pytest.raises(Phase4Error, match="execution-time Phase-4 policy"):
            supervisor.run_paper(plan.batch_id, operator="op", reason="runtime recheck")
        assert broker.submission_count == 0


def test_final_submit_fence_blocks_disconnect_after_outer_authorization(tmp_path, monkeypatch):
    values = execution_stack(tmp_path, OperationMode.PAPER_MANUAL)
    deployment_value, policy_value, signer_value, publisher, snapshot, clock, broker, db = values
    with Ledger(db, clock=clock) as ledger:
        supervisor = Phase4Supervisor(
            ledger, deployment_value, policy_value, publisher, broker, FakeQuotes(broker),
            signer=signer_value, clock=clock,
        )
        plan, _created = supervisor.prepare_plan(
            snapshot.snapshot_id, confirm_new_equity_session=True
        )
        supervisor.approve(plan.batch_id, plan.plan_hash, operator="op", reason="reviewed")
        stream = OrderStreamSupervisor(ledger, broker, clock=clock)
        stream.connected()
        assert stream.recover("initial recovery").clean
        original = supervisor._authorize_submit_fence
        injected = False

        def disconnect_at_final_fence(batch_id):
            nonlocal injected
            if not injected:
                injected = True
                stream.disconnected("disconnect raced with final submit authorization")
            original(batch_id)

        monkeypatch.setattr(supervisor, "_authorize_submit_fence", disconnect_at_final_fence)
        with pytest.raises(Phase4Error, match="control|stream|alert"):
            supervisor.run_paper(plan.batch_id, operator="op", reason="race regression")
        assert injected
        assert broker.submission_count == 0
        assert ledger.get_control_state("PAPER")["armed"] is False


def test_full_mocked_lifecycle_survives_partial_fill_disconnect_restart_and_backup(tmp_path):
    values = execution_stack(tmp_path, OperationMode.PAPER_MANUAL)
    deployment_value, policy_value, signer_value, publisher, snapshot, clock, broker, db = values
    broker.auto_fill = False
    with Ledger(db, clock=clock) as ledger:
        supervisor = Phase4Supervisor(
            ledger, deployment_value, policy_value, publisher, broker, FakeQuotes(broker),
            signer=signer_value, clock=clock,
        )
        plan, created = supervisor.prepare_plan(
            snapshot.snapshot_id, confirm_new_equity_session=True
        )
        assert created
        supervisor.approve(plan.batch_id, plan.plan_hash, operator="op", reason="reviewed")
        stream = OrderStreamSupervisor(ledger, broker, clock=clock)
        stream.connected()
        assert stream.recover("initial stream recovery").clean
        first_result = supervisor.run_paper(
            plan.batch_id, operator="op", reason="mocked lifecycle"
        )
        assert first_result.status == "submitted"
        order = broker.get_open_orders()[0]
        assert order.quantity > Decimal("1")
        partial_order = broker.fill_order(order.broker_order_id, order.quantity - Decimal("1"))
        partial_fill = broker.get_fills()[-1]
        assert stream.process(OrderStreamEvent(
            "lifecycle-partial", "partial_fill", partial_order, (partial_fill,), 1
        )) == "applied"
        stream.disconnected("simulated websocket interruption")

    # A new ledger connection represents a process restart. Broker truth stays
    # external, and REST recovery must converge before any later submission.
    with Ledger(db, clock=clock) as ledger:
        stream = OrderStreamSupervisor(ledger, broker, clock=clock)
        stream.connected()
        assert stream.recover("restart recovery after partial fill").clean
        store = Phase4Store(ledger)
        for alert in store.list_alerts(unresolved_only=True):
            if alert["severity"] == "critical":
                store.resolve_alert(
                    alert["alert_id"], operator="restart-operator",
                    note="REST recovery and clean reconciliation reviewed",
                )
        remaining_order = broker.fill_order(order.broker_order_id)
        remaining_fill = broker.get_fills()[-1]
        assert stream.process(OrderStreamEvent(
            "lifecycle-terminal-first", "fill", remaining_order, (remaining_fill,), 2
        )) == "applied"
        supervisor = Phase4Supervisor(
            ledger, deployment_value, policy_value, publisher, broker, FakeQuotes(broker),
            signer=signer_value, clock=clock,
        )
        sequence = 3
        while True:
            result = supervisor.run_paper(
                plan.batch_id, operator="op", reason="resume mocked lifecycle"
            )
            if result.status == "complete":
                break
            assert result.status == "submitted"
            open_orders = broker.get_open_orders()
            assert len(open_orders) == 1
            terminal = broker.fill_order(open_orders[0].broker_order_id)
            fill = broker.get_fills()[-1]
            assert stream.process(OrderStreamEvent(
                f"lifecycle-fill-{sequence}", "fill", terminal, (fill,), sequence
            )) == "applied"
            sequence += 1
        assert Reconciler(ledger, broker, clock=clock).reconcile().clean
        backup = BackupManager(ledger, tmp_path / "lifecycle-backups").create()
        BackupManager(ledger, tmp_path / "lifecycle-backups").verify(backup["backup_path"])
        soak = PaperSoakReporter(ledger).report(clock().date().isoformat())
        assert soak["cumulative"]["fills"] == len(broker.get_fills())
        assert soak["cumulative"]["clean_reconciliations"] > 0
        assert all(soak["cumulative"]["duplicate_prevention"].values())


def test_two_month_shadow_rebalances_are_restart_safe_across_cold_processes(tmp_path):
    first_values = published(tmp_path / "month-one", OperationMode.SHADOW)
    deployment_value, policy_value, _signer_one, _publisher_one, first = first_values
    second_decision = date(2026, 7, 31)
    second_publish = datetime(2026, 7, 31, 20, 5, tzinfo=timezone.utc)
    second_values = published(
        tmp_path / "month-two",
        OperationMode.SHADOW,
        decision_date=second_decision,
        publish_time=second_publish,
        execution_date=date(2026, 8, 3),
        previous_snapshot=first,
    )
    second = second_values[-1]
    deployment_path = tmp_path / "deployment.json"
    policy_path = tmp_path / "policy.json"
    deployment_path.write_text(
        json.dumps(json_safe(deployment_value.to_payload())), encoding="utf-8"
    )
    policy_path.write_text(json.dumps(policy_value.to_payload()), encoding="utf-8")
    key_path = tmp_path / "month-one" / "signing.key"
    snapshot_paths = []
    for index, snapshot in enumerate((first, second), start=1):
        path = tmp_path / f"snapshot-{index}.json"
        path.write_text(json.dumps(snapshot.to_payload()), encoding="utf-8")
        snapshot_paths.append(path)
    db = tmp_path / "multiprocess-shadow.sqlite3"
    jobs = (
        (snapshot_paths[0], EXECUTION_TIME, "month-one.json"),
        # Intentional duplicate process proves idempotent scheduler/snapshot/plan replay.
        (snapshot_paths[0], EXECUTION_TIME, "month-one-replay.json"),
        (
            snapshot_paths[1],
            datetime(2026, 8, 3, 14, 0, tzinfo=timezone.utc),
            "month-two.json",
        ),
    )
    context = multiprocessing.get_context("spawn")
    outputs = []
    for snapshot_path, when, output_name in jobs:
        output = tmp_path / output_name
        process = context.Process(target=_shadow_rebalance_worker, args=(
            str(db), str(snapshot_path), str(deployment_path), str(policy_path),
            str(key_path), when.isoformat(), str(output),
        ))
        process.start()
        process.join(30)
        assert process.exitcode == 0
        outputs.append(json.loads(output.read_text(encoding="utf-8")))
    assert outputs[0]["schedule_created"] and outputs[0]["snapshot_created"]
    assert not outputs[1]["schedule_created"] and not outputs[1]["snapshot_created"]
    assert not outputs[1]["plan_created"]
    with Ledger(db) as ledger:
        store = Phase4Store(ledger)
        assert len(store.list_schedule_runs()) == 2
        assert len(store.list_snapshots()) == 2
        links = ledger.conn.execute("SELECT * FROM phase4_plan_links").fetchall()
        assert len(links) == 2
        assert all(not row["paper_submission_allowed"] for row in links)
        assert len(ledger.list_orders()) == 0
        assert all(
            row["status"] == "voided"
            for row in ledger.conn.execute("SELECT status FROM execution_batches")
        )


def test_quote_spread_deviation_turnover_concentration_and_order_count_gates(tmp_path):
    values = execution_stack(tmp_path, OperationMode.PAPER_MANUAL)
    deployment_value, policy_value, signer_value, publisher, snapshot, clock, broker, db = values
    symbol = StrategyTargetPublisher.required_tradable_symbols()[0]
    ref = next(
        Decimal(str(row["research_reference_price"]))
        for row in snapshot.content["required_target_deltas"] if row["symbol"] == symbol
    )
    broker.set_quote(symbol, ref, spread_bps="100", as_of=clock())
    with Ledger(db, clock=clock) as ledger:
        supervisor = Phase4Supervisor(
            ledger, deployment_value, policy_value, publisher, broker, FakeQuotes(broker),
            signer=signer_value, clock=clock,
        )
        with pytest.raises(Phase4Error, match="spread"):
            supervisor.prepare_plan(snapshot.snapshot_id, confirm_new_equity_session=True)

    # A fresh ledger avoids the failed preview's equity/session side effects.
    db2 = tmp_path / "deviation.sqlite3"
    broker.set_quote(symbol, ref * Decimal("1.06"), spread_bps="2", as_of=clock())
    with Ledger(db2, clock=clock) as ledger:
        Reconciler(ledger, broker, clock=clock).bootstrap_positions()
        Reconciler(ledger, broker, clock=clock).reconcile()
        Phase4Store(ledger).publish_snapshot(snapshot)
        with pytest.raises(Phase4Error, match="deviation"):
            Phase4Supervisor(
                ledger, deployment_value, policy_value, publisher, broker, FakeQuotes(broker),
                signer=signer_value, clock=clock,
            ).prepare_plan(snapshot.snapshot_id, confirm_new_equity_session=True)


def _seed_stream_order(ledger, broker, clock):
    broker.set_quote("AAPL", "100", spread_bps="0", as_of=clock())
    Reconciler(ledger, broker, clock=clock).bootstrap_positions()
    intent = TargetPositionIntent(
        account_id="PAPER", strategy_id="test", symbol="AAPL",
        target_quantity=Decimal("10"), signal_at=clock(),
        target_version="stream-v1", reference_price=Decimal("100"), reason="stream test",
    )
    intent_row, _created = ledger.create_intent(intent)
    request = OrderRequest(
        account_id="PAPER", client_order_id="wslab-stream-test",
        intent_id=intent_row["intent_id"], symbol="AAPL", side=Side.BUY,
        quantity=Decimal("10"), reference_price=Decimal("100"),
    )
    row = ledger.plan_order(request)
    broker_order = broker.submit_order(request)
    ledger.acknowledge_order(row["order_id"], broker_order)
    return row, broker_order


def _alpaca_trade_update_payload(*, stream="trade_updates", event="new"):
    return {
        "stream": stream,
        "data": {
            "event": event,
            "event_id": "wire-event-1",
            "timestamp": "2026-07-01T14:00:00Z",
            "order": {
                "asset_class": "us_equity",
                "id": "wire-order-1",
                "client_order_id": "wslab-wire-order-1",
                "symbol": "AAPL",
                "side": "buy",
                "qty": "1",
                "filled_qty": "0",
                "status": "new",
                "submitted_at": "2026-07-01T14:00:00Z",
                "updated_at": "2026-07-01T14:00:00Z",
                "type": "market",
                "time_in_force": "day",
            },
        },
    }


def test_trade_update_parser_requires_exact_channel_and_supported_event_type():
    transport = AlpacaPaperTradeUpdateStream(
        AlpacaPaperConfig("paper-key", "paper-secret"), "PAPER"
    )
    parsed = transport._parse(json.dumps(_alpaca_trade_update_payload()))
    assert parsed.event_type == "new"
    assert parsed.broker_order.account_id == "PAPER"

    with pytest.raises(Phase4Error, match="Malformed"):
        transport._parse(json.dumps(_alpaca_trade_update_payload(stream="account_updates")))
    # Trade corrections and busts need reversal accounting; accepting them as
    # ordinary append-only fills would corrupt expected cash and positions.
    with pytest.raises(Phase4Error, match="Malformed"):
        transport._parse(json.dumps(_alpaca_trade_update_payload(event="trade_bust")))


def test_stream_multiple_partials_duplicate_out_of_order_fill_after_cancel_request(tmp_path):
    clock = FixedClock(EXECUTION_TIME)
    broker = Phase4FakeBroker("PAPER", "100000", auto_fill=False, clock=clock)
    with Ledger(tmp_path / "stream.sqlite3", clock=clock) as ledger:
        row, order = _seed_stream_order(ledger, broker, clock)
        stream = OrderStreamSupervisor(ledger, broker, clock=clock)
        stream.connected()
        # Simulate a cancel request still pending at the broker: the durable
        # order remains active, and a fill arriving in that window must apply.
        first_order = broker.fill_order(order.broker_order_id, "4")
        first_fill = broker.get_fills()[-1]
        event1 = OrderStreamEvent("event-1", "partial_fill", first_order, (first_fill,), 1)
        assert stream.process(event1) == "applied"
        assert stream.process(event1) == "duplicate"
        second_order = broker.fill_order(order.broker_order_id, "3")
        second_fill = broker.get_fills()[-1]
        event2 = OrderStreamEvent("event-2", "partial_fill", second_order, (second_fill,), 2)
        assert stream.process(event2) == "applied"
        assert ledger.filled_quantity_for_order(row["order_id"]) == Decimal("7")
        assert stream.process(
            OrderStreamEvent("event-stale", "partial_fill", first_order, (), 1)
        ) == "out_of_order"
        assert Phase4Store(ledger).stream_state()["recovering"] == 1


def test_stream_disconnect_rest_recovery_and_external_order_block(tmp_path):
    clock = FixedClock(EXECUTION_TIME)
    broker = Phase4FakeBroker("PAPER", "100000", auto_fill=False, clock=clock)
    with Ledger(tmp_path / "recover.sqlite3", clock=clock) as ledger:
        _row, order = _seed_stream_order(ledger, broker, clock)
        stream = OrderStreamSupervisor(ledger, broker, clock=clock)
        external = replace(order, client_order_id="external-client", broker_order_id="external-order")
        assert stream.process(OrderStreamEvent("external-event", "new", external, (), 1)) == "unknown_order"
        assert Phase4Store(ledger).stream_state()["recovering"] == 1
        # Remove the synthetic external event from broker truth (it was never
        # inserted there) and recover all local client IDs through REST.
        stream.connected()
        report = stream.recover()
        assert report.clean
        state = Phase4Store(ledger).stream_state()
        assert state["connected"] == 1 and state["recovering"] == 0
        stream.disconnected("network outage")
        assert Phase4Store(ledger).stream_state()["disconnect_count"] == 1


def test_stream_recovery_uses_bounded_recent_window_with_over_500_historical_orders(
    tmp_path,
):
    class PageLimitedHistoryBroker(Phase4FakeBroker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.recent_queries = []

        def get_recent_orders(self, since=None):
            self.recent_queries.append(since)
            orders = list(self._orders_by_client.values())
            if since is not None:
                orders = [order for order in orders if order.submitted_at > since]
            if len(orders) >= 500:
                raise BrokerError("simulated Alpaca all-status order page limit")
            return sorted(
                orders, key=lambda order: (order.submitted_at, order.broker_order_id)
            )

    clock = FixedClock(EXECUTION_TIME)
    broker = PageLimitedHistoryBroker(
        "PAPER", "100000", auto_fill=False, clock=clock
    )
    with Ledger(tmp_path / "bounded-recovery.sqlite3", clock=clock) as ledger:
        reconciler = Reconciler(ledger, broker, clock=clock)
        reconciler.bootstrap_positions()
        assert reconciler.reconcile().clean

        old = EXECUTION_TIME - timedelta(days=30)
        for index in range(501):
            order = BrokerOrder(
                broker_order_id=f"historical-order-{index}",
                client_order_id=f"historical-client-{index}",
                account_id="PAPER",
                symbol="AAPL",
                side=Side.BUY,
                quantity=Decimal("1"),
                filled_quantity=Decimal("0"),
                status=OrderStatus.CANCELED,
                submitted_at=old + timedelta(seconds=index),
                updated_at=old + timedelta(seconds=index),
            )
            broker._store_order(order)

        stream = OrderStreamSupervisor(ledger, broker, clock=clock)
        stream.connected()
        assert stream.recover("bounded historical-order recovery").clean
        assert broker.recent_queries
        assert all(value is not None for value in broker.recent_queries)


def test_stream_is_not_marked_ready_until_transport_callback_and_renews_lease(tmp_path):
    clock = FixedClock(EXECUTION_TIME)
    broker = Phase4FakeBroker("PAPER", "100000", auto_fill=False, clock=clock)
    with Ledger(tmp_path / "ready.sqlite3", clock=clock) as ledger:
        Reconciler(ledger, broker, clock=clock).bootstrap_positions()
        assert Reconciler(ledger, broker, clock=clock).reconcile().clean
        stream = OrderStreamSupervisor(ledger, broker, clock=clock)

        class ReadySource:
            supports_ready_callback = True

            def events(self, *, on_ready, on_heartbeat):
                assert Phase4Store(ledger).stream_state() is None
                on_ready()
                before = Phase4Store(ledger).stream_state()["updated_at"]
                clock.advance(10)
                on_heartbeat()
                assert Phase4Store(ledger).stream_state()["updated_at"] != before
                if False:
                    yield None

        with pytest.raises(Phase4Error, match="exhausted"):
            stream.supervise(lambda: ReadySource(), max_reconnects=0, sleep=lambda _delay: None)
        state = Phase4Store(ledger).stream_state()
        assert state["recovery_count"] == 1
        assert state["connected"] == 0 and state["recovering"] == 1


def test_alert_dedupe_ack_escalation_and_health(tmp_path):
    clock = FixedClock(EXECUTION_TIME)
    with Ledger(tmp_path / "health.sqlite3", clock=clock) as ledger:
        store = Phase4Store(ledger)
        alert, created = store.emit_alert(
            "critical", "database_integrity_failure", "test failure", dedupe_key="db"
        )
        repeated, created_again = store.emit_alert(
            "critical", "database_integrity_failure", "test failure", dedupe_key="db"
        )
        assert created and not created_again and repeated["occurrence_count"] == 2
        store.acknowledge_alert(alert["alert_id"], operator="op", note="investigating")
        clock.advance(901)
        assert store.escalate_alerts(older_than_seconds=900)[0]["escalation_count"] == 1
        report = HealthReporter(ledger, policy(), clock=clock).report("PAPER")
        assert report["operating_mode"] == "observe"
        assert "critical_alerts" in report["submission_blockers"]
        assert report["database_integrity"]


def test_alert_severity_upgrade_is_delivered_immediately(tmp_path):
    delivered = []

    class Sink:
        def send(self, alert):
            delivered.append((alert["alert_id"], alert["severity"]))

    with Ledger(tmp_path / "alert-upgrade.sqlite3", clock=lambda: EXECUTION_TIME) as ledger:
        manager = AlertManager(Phase4Store(ledger), (Sink(),))
        manager.emit("warning", "stream", "degraded", dedupe_key="same-incident")
        manager.emit("critical", "stream", "failed", dedupe_key="same-incident")
        manager.emit("critical", "stream", "still failed", dedupe_key="same-incident")
        assert [severity for _alert_id, severity in delivered] == ["warning", "critical"]


def test_health_never_claims_offline_or_non_submitting_submission_readiness(tmp_path):
    clock = FixedClock(EXECUTION_TIME)
    with Ledger(tmp_path / "health-readiness.sqlite3", clock=clock) as ledger:
        store = Phase4Store(ledger)
        observe = HealthReporter(
            ledger,
            policy(OperationMode.OBSERVE),
            alerts=AlertManager(store),
            clock=clock,
        ).report("PAPER")
        assert not observe["submission_ready"]
        assert "operating_mode_non_submitting" in observe["submission_blockers"]
        categories = {row["category"] for row in store.list_alerts(unresolved_only=True)}
        assert "broker_disconnection" not in categories
        assert "reconciliation_mismatch" not in categories

        offline_paper = HealthReporter(
            ledger, policy(OperationMode.PAPER_MANUAL), clock=clock
        ).report("PAPER")
        assert not offline_paper["submission_ready"]
        assert "broker_connectivity_unchecked" in offline_paper["submission_blockers"]


def test_health_rejects_connected_but_trading_blocked_account(tmp_path):
    class BlockedBroker(FakeBroker):
        def get_account(self):
            return replace(super().get_account(), trading_blocked=True)

    clock = FixedClock(EXECUTION_TIME)
    broker = BlockedBroker("PAPER", "100000", clock=clock)
    with Ledger(tmp_path / "health-blocked.sqlite3", clock=clock) as ledger:
        report = HealthReporter(
            ledger, policy(OperationMode.PAPER_MANUAL), broker=broker, clock=clock
        ).report("PAPER")
        assert report["broker_connectivity"] is False
        assert report["broker_error_type"] == "AccountNotReady"
        assert "broker_connectivity" in report["submission_blockers"]


def test_sqlite_backup_restore_hash_retention_and_failure(tmp_path):
    clock = FixedClock(EXECUTION_TIME)
    ledger_path = tmp_path / "ledger.sqlite3"
    config_path = tmp_path / "policy.json"
    config_path.write_text('{"mode":"observe"}', encoding="utf-8")
    with Ledger(ledger_path, clock=clock) as ledger:
        manager = BackupManager(ledger, tmp_path / "backups", retention=1)
        first = manager.create((config_path,))
        clock.advance(1)
        second = manager.create((config_path,))
        assert first["backup_id"] != second["backup_id"]
        assert len(list((tmp_path / "backups").glob("backup-*"))) == 1
        backup_dir = Path(second["backup_path"])
        manager.verify(backup_dir)
        with pytest.raises(LedgerError, match="destination ledger is open"):
            manager.restore(
                backup_dir, ledger_path,
                active_ledger_path=ledger_path, confirm_replace=True,
            )
        restored = manager.restore(backup_dir, tmp_path / "restored.sqlite3")
        with Ledger(restored) as reopened:
            assert reopened.conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        with pytest.raises(Phase4Error, match="confirmation"):
            manager.restore(backup_dir, restored)
        (backup_dir / "manifest.sha256").chmod(0o600)
        (backup_dir / "manifest.sha256").write_text("0" * 64, encoding="ascii")
        with pytest.raises(Phase4Error, match="manifest hash"):
            manager.verify(backup_dir)


def test_backup_refuses_credentials_and_live_style_env(monkeypatch, tmp_path):
    clock = FixedClock(EXECUTION_TIME)
    ledger_path = tmp_path / "ledger.sqlite3"
    secret = tmp_path / "bad.json"
    secret.write_text('{"api_secret":"never"}', encoding="utf-8")
    with Ledger(ledger_path, clock=clock) as ledger:
        with pytest.raises(Phase4Error, match="credential"):
            BackupManager(ledger, tmp_path / "backups").create((secret,))
        symlink = tmp_path / "policy-link.json"
        target = tmp_path / "policy.json"
        target.write_text('{"mode":"observe"}', encoding="utf-8")
        symlink.symlink_to(target)
        with pytest.raises(Phase4Error, match="symbolic link"):
            BackupManager(ledger, tmp_path / "backups-symlink").create((symlink,))
        shared = tmp_path / "shared"
        shared.mkdir(mode=0o755)
        with pytest.raises(Phase4Error, match="private directory"):
            BackupManager(ledger, shared).create()
        assert shared.stat().st_mode & 0o777 == 0o755
    monkeypatch.setenv("ALPACA_API_KEY", "live-style")
    monkeypatch.setenv("ALPACA_PAPER_API_KEY", "paper")
    monkeypatch.setenv("ALPACA_PAPER_API_SECRET", "paper-secret")
    with pytest.raises(ValueError, match="Live-style"):
        AlpacaPaperConfig.from_env()
    with pytest.raises(ValueError, match="hard-pinned"):
        AlpacaPaperConfig("x", "y", base_url="https://api.alpaca.markets")
    for url in (
        "https://10.0.0.1/hook", "https://169.254.169.254/hook",
        "https://[fd00::1]/hook", "https://service.local/hook",
    ):
        with pytest.raises(ValueError, match="public|globally"):
            WebhookConfig(url)


def test_backup_manifest_path_traversal_and_symlink_restore_are_rejected(tmp_path):
    clock = FixedClock(EXECUTION_TIME)
    ledger_path = tmp_path / "ledger.sqlite3"
    with Ledger(ledger_path, clock=clock) as ledger:
        manager = BackupManager(ledger, tmp_path / "backups")
        backup = manager.create()
        directory = Path(backup["backup_path"])
        destination = tmp_path / "existing.sqlite3"
        destination.write_bytes(b"do not replace")
        symlink = tmp_path / "restore-link.sqlite3"
        symlink.symlink_to(destination)
        with pytest.raises(Phase4Error, match="symbolic link"):
            manager.restore(directory, symlink, confirm_replace=True)
        manifest_path = directory / "manifest.json"
        hash_path = directory / "manifest.sha256"
        manifest_path.chmod(0o600)
        hash_path.chmod(0o600)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["ledger_file"] = "../ledger.sqlite3"
        encoded = canonical_bytes(manifest)
        manifest_path.write_bytes(encoded + b"\n")
        hash_path.write_text(hashlib.sha256(encoded).hexdigest() + "\n", encoding="ascii")
        with pytest.raises(Phase4Error, match="safety fields"):
            manager.verify(directory)


def test_restore_only_manager_does_not_require_or_open_active_ledger(tmp_path):
    source = tmp_path / "source.sqlite3"
    with Ledger(source, clock=lambda: EXECUTION_TIME) as ledger:
        backup = BackupManager(ledger, tmp_path / "backups").create()
    active_that_must_not_be_created = tmp_path / "missing-active.sqlite3"
    destination = tmp_path / "restored.sqlite3"
    restored = BackupManager(None, tmp_path / "backups").restore(
        backup["backup_path"], destination,
        active_ledger_path=active_that_must_not_be_created,
    )
    assert restored == destination.resolve()
    assert not active_that_must_not_be_created.exists()


def test_soak_report_daily_and_cumulative_not_live_equivalence(tmp_path):
    clock = FixedClock(EXECUTION_TIME)
    with Ledger(tmp_path / "soak.sqlite3", clock=clock) as ledger:
        store = Phase4Store(ledger)
        store.claim_schedule("2026-06-30", "2026-07-01", status="published")
        store.record_soak_observation(
            "2026-07-01", "next_close_slippage_bps", Decimal("12.5")
        )
        store.record_soak_observation(
            "2026-07-01", "target_weight_error_bps", Decimal("8")
        )
        report = PaperSoakReporter(ledger).report("2026-07-01")
        assert report["paper_only"] is True
        assert "do not establish live" in report["paper_fill_limitation"]
        assert report["daily"]["slippage_vs_next_close_bps"] == "12.5"
        assert report["cumulative"]["database_integrity"]
        assert all(report["cumulative"]["duplicate_prevention"].values())


def test_existing_strategy_interfaces_and_default_weights_unchanged():
    from src.portfolio import DEFAULT_PORTFOLIO_WEIGHTS
    from src.strategies.momentum import MomentumStrategy
    from src.strategies.regime_switch import RegimeSwitchStrategy
    from src.strategies.sector_rotation import SectorRotationStrategy

    assert DEFAULT_PORTFOLIO_WEIGHTS == [
        ("momentum", 0.60), ("sector_rotation", 0.35), ("regime_switch", 0.05)
    ]
    assert MomentumStrategy().describe()["params"]["lookback_trading_days"] == 126
    assert SectorRotationStrategy().describe()["params"]["lookback_months"] == 3
    assert RegimeSwitchStrategy().describe()["params"]["regime_sma_period"] == 200


def test_phase4_examples_and_cli_have_no_live_endpoint_or_implicit_confirmation(tmp_path):
    Phase4Policy.from_file("examples/live/phase4_policy.example.json")
    deployment_value = DeploymentConfig.from_file(
        "examples/live/phase4_deployment.example.json"
    )
    assert len(deployment_value.managed_symbols) == 36
    parser = phase4_cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "health", "--db", "x", "--deployment", "d", "--policy", "p",
            "--base-url", "https://api.alpaca.markets",
        ])
    with pytest.raises(SystemExit):
        phase4_cli.main([
            "run-paper", "--db", str(tmp_path / "x.sqlite3"),
            "--deployment", "d.json", "--policy", "p.json",
            "--batch-id", "batch-x", "--operator", "op", "--reason", "test",
        ])


def test_recovery_configuration_can_load_without_a_signing_key(monkeypatch):
    args = phase4_cli.argparse.Namespace(
        deployment="examples/live/phase4_deployment.example.json",
        policy="examples/live/phase4_policy.example.json",
    )
    monkeypatch.delenv("WSLAB_PHASE4_SIGNING_KEY_FILE", raising=False)
    with pytest.raises(Phase4Error, match="SIGNING_KEY_FILE"):
        phase4_cli._load(args)
    deployment_value, policy_value, signer_value = phase4_cli._load(
        args, require_signer=False, load_signer=False
    )
    assert deployment_value.account_id == "REPLACE_WITH_DEDICATED_PAPER_ACCOUNT_ID"
    assert policy_value.require_signing is True
    assert signer_value is None


def test_publisher_rejects_short_nonfinite_and_exactly_repeated_history(tmp_path):
    _deployment, _policy, _signer, publisher, _snapshot = published(tmp_path)
    account = Phase4FakeBroker("PAPER", "100000", clock=lambda: PUBLISH_TIME).get_account()
    decision = MarketCalendarDay(
        DECISION, datetime(2026, 6, 30, 13, 30, tzinfo=timezone.utc),
        datetime(2026, 6, 30, 20, 0, tzinfo=timezone.utc),
    )
    execution = MarketCalendarDay(
        date(2026, 7, 1), datetime(2026, 7, 1, 13, 30, tzinfo=timezone.utc),
        datetime(2026, 7, 1, 20, 0, tzinfo=timezone.utc),
    )
    original = market_bundle()
    short_calendar = original.calendar[-100:]
    short = replace(
        original,
        calendar=short_calendar,
        frames={symbol: frame.loc[short_calendar].copy() for symbol, frame in original.frames.items()},
    )
    with pytest.raises(Phase4Error, match="at least 200"):
        publisher.publish(
            decision_day=decision, execution_day=execution, account=account, positions=[],
            assets=assets(), market_data=short,
        )
    nonfinite = market_bundle()
    bad_symbol = StrategyTargetPublisher.required_data_symbols()[0]
    nonfinite.frames[bad_symbol].iloc[-20, nonfinite.frames[bad_symbol].columns.get_loc("Close")] = np.nan
    with pytest.raises(Phase4Error, match="non-finite"):
        publisher.publish(
            decision_day=decision, execution_day=execution, account=account, positions=[],
            assets=assets(), market_data=nonfinite,
        )
    repeated = market_bundle()
    repeated.frames[bad_symbol].iloc[-1] = repeated.frames[bad_symbol].iloc[-2]
    with pytest.raises(Phase4Error, match="carried-forward"):
        publisher.publish(
            decision_day=decision, execution_day=execution, account=account, positions=[],
            assets=assets(), market_data=repeated,
        )


@pytest.mark.parametrize("extra_session_kind", ["weekend", "intraday"])
def test_publisher_rejects_noncanonical_symbol_sessions(tmp_path, extra_session_kind):
    bundle = market_bundle()
    symbol = config.MEAN_REVERSION_UNIVERSE[0]
    frame = bundle.frames[symbol]
    source_session = bundle.calendar[-20]
    if extra_session_kind == "weekend":
        extra_session = next(
            value
            for value in pd.date_range(bundle.calendar[-30], bundle.calendar[-1], freq="D")
            if value.weekday() >= 5
        )
    else:
        extra_session = source_session + pd.Timedelta(hours=12)
    injected = frame.loc[[source_session]].copy()
    injected.index = pd.DatetimeIndex([extra_session])
    bundle.frames[symbol] = pd.concat([frame, injected]).sort_index()

    with pytest.raises(Phase4Error, match="session index"):
        published(tmp_path, bundle=bundle)


def test_dirty_labelled_snapshot_freezes_exact_dirty_file_content(monkeypatch, tmp_path):
    _deployment, _policy, _signer, publisher, snapshot = published(tmp_path)
    monkeypatch.setattr(
        "src.live.publisher.git_dirty_file_hashes",
        lambda _root: {"src/live/publisher.py": {"state": "file", "size": 1, "sha256": "0" * 64}},
    )
    with pytest.raises(Phase4Error, match="Dirty publisher file content changed"):
        publisher.to_execution_targets(snapshot)


def test_recent_terminal_external_order_is_reconciled_and_latched(tmp_path):
    clock = FixedClock(EXECUTION_TIME)
    broker = FakeBroker("PAPER", "100000", clock=clock)
    broker.set_quote("AAPL", "100", spread_bps="2", as_of=clock())
    broker.submit_order(OrderRequest(
        client_order_id="manual-terminal-order", account_id="PAPER", symbol="AAPL",
        side=Side.BUY, quantity=Decimal("1"), reference_price=Decimal("100"),
        intent_id="external-intent",
    ))
    with Ledger(tmp_path / "terminal.sqlite3", clock=clock) as ledger:
        reconciler = Reconciler(ledger, broker, clock=clock)
        reconciler.bootstrap_positions()
        report = reconciler.reconcile()
        assert not report.clean
        assert "EXTERNAL_RECENT_ORDER" in {issue.code for issue in report.issues}


def test_pending_stream_event_forces_recovery_before_replay(tmp_path):
    clock = FixedClock(EXECUTION_TIME)
    broker = Phase4FakeBroker("PAPER", "100000", auto_fill=False, clock=clock)
    with Ledger(tmp_path / "pending-event.sqlite3", clock=clock) as ledger:
        _row, order = _seed_stream_order(ledger, broker, clock)
        event = OrderStreamEvent("persisted-before-crash", "new", order, (), 1)
        store = Phase4Store(ledger)
        assert store.record_stream_event(
            event_id=event.event_id, sequence=event.sequence,
            client_order_id=order.client_order_id, event_type=event.event_type,
            broker_updated_at=order.updated_at.isoformat(), payload=event.payload(),
            disposition="pending",
        )
        stream = OrderStreamSupervisor(ledger, broker, clock=clock)
        stream.connected()
        with pytest.raises(Phase4Error, match="persisted but not fully applied"):
            stream.process(event)
        assert store.stream_state()["recovering"]
        assert stream.recover("recover incomplete durable event").clean
        assert store.stream_event(event.event_id)["disposition"] == "recovered"
        assert stream.process(event) == "duplicate"


def test_stream_event_persistence_atomically_revokes_submit_readiness_before_apply(
    tmp_path, monkeypatch
):
    clock = FixedClock(EXECUTION_TIME)
    broker = Phase4FakeBroker("PAPER", "100000", auto_fill=False, clock=clock)
    with Ledger(tmp_path / "stream-ingest-fence.sqlite3", clock=clock) as ledger:
        _row, order = _seed_stream_order(ledger, broker, clock)
        stream = OrderStreamSupervisor(ledger, broker, clock=clock)
        stream.connected()
        assert stream.recover("establish healthy stream state").clean
        store = Phase4Store(ledger)
        assert store.stream_state(STREAM_NAME)["recovering"] == 0

        event = OrderStreamEvent("crash-after-durable-ingest", "new", order, (), 1)

        def crash_before_order_application(*_args, **_kwargs):
            # The event insert must revoke submission authority before any
            # fallible application step begins. SystemExit models abrupt
            # process death, bypassing ordinary exception recovery.
            state = store.stream_state(STREAM_NAME)
            assert state["connected"] == 1
            assert state["recovering"] == 1
            raise SystemExit("simulated crash after durable stream ingest")

        monkeypatch.setattr(ledger, "acknowledge_order", crash_before_order_application)
        with pytest.raises(SystemExit, match="simulated crash"):
            stream.process(event)

        assert store.stream_event(event.event_id)["disposition"] == "pending"
        assert store.stream_state(STREAM_NAME)["recovering"] == 1


def test_ledger_paths_are_canonical_and_symlinks_fail_closed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    first = Ledger("canonical.sqlite3", clock=lambda: EXECUTION_TIME)
    second = Ledger(tmp_path / "canonical.sqlite3", clock=lambda: EXECUTION_TIME)
    try:
        assert first.path == second.path == str((tmp_path / "canonical.sqlite3").resolve())
    finally:
        second.close()
        first.close()
    link = tmp_path / "ledger-link.sqlite3"
    link.symlink_to(tmp_path / "canonical.sqlite3")
    with pytest.raises(Exception, match="symbolic link"):
        Ledger(link)


def test_alert_severity_upgrades_and_stale_reconciliation_is_not_ready(tmp_path):
    clock = FixedClock(EXECUTION_TIME)
    broker = FakeBroker("PAPER", "100000", clock=clock)
    with Ledger(tmp_path / "alert-health.sqlite3", clock=clock) as ledger:
        reconciler = Reconciler(ledger, broker, clock=clock)
        reconciler.bootstrap_positions()
        assert reconciler.reconcile().clean
        store = Phase4Store(ledger)
        store.emit_alert("info", "test", "first", dedupe_key="upgrade")
        warning, created = store.emit_alert(
            "warning", "test", "second", dedupe_key="upgrade"
        )
        assert not created and warning["severity"] == "warning"
        clock.advance(121)
        report = HealthReporter(ledger, policy(), clock=clock).report("PAPER")
        assert not report["reconciliation_fresh"]
        assert "reconciliation" in report["submission_blockers"]
