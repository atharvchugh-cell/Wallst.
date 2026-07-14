"""Phase-two paper CLI tests; every broker is in-process and no network is used."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.live import paper_cli
from src.live.fake_broker import FakeBroker
from src.live.ledger import Ledger
from src.live.models import TargetPositionIntent
from src.live.oms import OrderManagementSystem
from src.live.reconcile import Reconciler
from src.live.risk import PreTradeRiskEngine


NOW = datetime(2026, 7, 14, 14, 0, tzinfo=timezone.utc)


class OperationalPaperFake(FakeBroker):
    fail_account = False

    def get_account(self):
        if self.fail_account:
            from src.live.broker import BrokerError

            raise BrokerError("simulated account endpoint outage")
        return super().get_account()

    def drain_request_ids(self):
        return ()


def active_stack(tmp_path):
    clock = lambda: NOW
    broker = OperationalPaperFake(
        account_id="paper-account", cash="10000", auto_fill=False, clock=clock
    )
    quote = broker.set_quote("SPY", "100", spread_bps="0", as_of=NOW)
    db = tmp_path / "paper-ledger.sqlite3"
    with Ledger(db, clock=clock) as ledger:
        reconciler = Reconciler(ledger, broker, clock=clock)
        reconciler.bootstrap_positions()
        assert reconciler.reconcile().clean
        oms = OrderManagementSystem(
            ledger, broker, PreTradeRiskEngine(clock=clock), clock=clock
        )
        oms.arm("test setup")
        result = oms.process_intent(
            TargetPositionIntent(
                "paper-account", "aggregate", "SPY", Decimal("10"), NOW,
                "v1", Decimal("100"), "test paper CLI operations",
            ),
            quote=quote,
            market_open=True,
            day_start_equity=Decimal("10000"),
            high_water_equity=Decimal("10000"),
        )
    return broker, db, result


def test_recover_command_synchronizes_but_never_submits(tmp_path, monkeypatch, capsys):
    broker, db, _result = active_stack(tmp_path)
    assert broker.submission_count == 1
    monkeypatch.setattr(paper_cli, "_broker", lambda: broker)
    rc = paper_cli.main([
        "recover", "--db", str(db), "--confirm-paper-network",
    ])
    assert rc == 0
    assert broker.submission_count == 1
    assert '"clean": true' in capsys.readouterr().out


def test_kill_command_cancels_and_persists_kill_state(tmp_path, monkeypatch, capsys):
    broker, db, _result = active_stack(tmp_path)
    monkeypatch.setattr(paper_cli, "_broker", lambda: broker)
    rc = paper_cli.main([
        "kill",
        "--db", str(db),
        "--reason", "operator emergency test",
        "--confirm-paper-network",
        "--confirm-cancel-open-orders",
    ])
    assert rc == 0
    assert broker.get_open_orders() == []
    with Ledger(db) as ledger:
        assert ledger.get_control_state("paper-account")["kill_switch"] is True
    assert '"kill_switch": true' in capsys.readouterr().out


def test_networked_kill_persists_before_account_endpoint_failure(
    tmp_path, monkeypatch, capsys
):
    broker, db, _result = active_stack(tmp_path)
    broker.fail_account = True
    monkeypatch.setattr(paper_cli, "_broker", lambda: broker)
    rc = paper_cli.main([
        "kill",
        "--db", str(db),
        "--reason", "emergency during broker outage",
        "--confirm-paper-network",
        "--confirm-cancel-open-orders",
    ])
    assert rc == 1
    with Ledger(db) as ledger:
        control = ledger.get_control_state("paper-account")
        assert control["armed"] is False
        assert control["kill_switch"] is True
    assert "account endpoint outage" in capsys.readouterr().err


def test_offline_disarm_preserves_kill_and_never_constructs_broker(
    tmp_path, monkeypatch, capsys
):
    db = tmp_path / "paper-ledger.sqlite3"
    with Ledger(db) as ledger:
        ledger.set_control_state(
            "paper-account", armed=False, kill_switch=True, reason="setup"
        )

    def forbidden():
        raise AssertionError("offline disarm must not construct a broker")

    monkeypatch.setattr(paper_cli, "_broker", forbidden)
    rc = paper_cli.main([
        "disarm", "--db", str(db), "--account-id", "paper-account",
        "--reason", "operator remains stopped",
    ])
    assert rc == 0
    output = capsys.readouterr().out
    assert '"armed": false' in output
    assert '"kill_switch": true' in output


def test_offline_local_kill_persists_without_claiming_broker_cancellation(
    tmp_path, monkeypatch, capsys
):
    broker, db, _result = active_stack(tmp_path)

    def forbidden():
        raise AssertionError("local-kill must not construct a broker")

    monkeypatch.setattr(paper_cli, "_broker", forbidden)
    rc = paper_cli.main([
        "local-kill",
        "--db", str(db),
        "--account-id", "paper-account",
        "--reason", "network unavailable emergency",
        "--confirm-local-kill",
    ])
    assert rc == 0
    with Ledger(db) as ledger:
        control = ledger.get_control_state("paper-account")
        assert control["armed"] is False
        assert control["kill_switch"] is True
    assert len(broker.get_open_orders()) == 1
    output = capsys.readouterr().out
    assert '"broker_orders_canceled": false' in output
    assert "inspect and cancel broker orders separately" in output


def test_mutating_network_commands_require_both_confirmations(tmp_path):
    with pytest.raises(SystemExit) as exc:
        paper_cli.main([
            "kill", "--db", str(tmp_path / "ledger.sqlite3"),
            "--reason", "test", "--confirm-paper-network",
        ])
    assert exc.value.code == 2
