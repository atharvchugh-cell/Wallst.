"""Explicitly-confirmed, paper-only Alpaca operating commands.

Phase three adds an immutable preview -> exact-hash approval -> execute flow.
There is no generic arm or raw-order command and no live-money endpoint.
"""

from __future__ import annotations

import argparse
import json
import sys

from .alpaca_paper import AlpacaPaperBroker, AlpacaPaperConfig, PAPER_BASE_URL
from .alpaca_data import (
    DATA_BASE_URL,
    AlpacaDataConfig,
    AlpacaPaperMarketData,
)
from .broker import BrokerError
from .deployment import DeploymentConfig, DeploymentError, SleeveTargetSnapshot
from .execution import PaperExecutionService
from .ledger import Ledger, LedgerError
from .market_data import MarketDataError
from .models import IntentStatus, OMSResult, json_safe
from .oms import ExecutionBlocked, OrderManagementSystem
from .reconcile import Reconciler, ReconciliationReport
from .risk import PreTradeRiskEngine


NETWORK_COMMANDS = {
    "check",
    "bootstrap",
    "reconcile",
    "recover",
    "cancel",
    "kill",
    "abandon-missing",
    "reset-kill",
    "preview",
    "execute",
    "settle-batch",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper-only Alpaca operating utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="inspect paper readiness; never submits an order")
    _add_network_confirmation(check)

    bootstrap = sub.add_parser(
        "bootstrap", help="record the one-time position/cash baseline from paper broker"
    )
    _add_ledger(bootstrap)
    _add_network_confirmation(bootstrap)
    bootstrap.add_argument(
        "--confirm-initial-baseline",
        action="store_true",
        help="required acknowledgement that the irreversible local baseline is correct",
    )

    reconcile = sub.add_parser("reconcile", help="compare the durable ledger to paper broker")
    _add_ledger(reconcile)
    _add_network_confirmation(reconcile)

    recover = sub.add_parser(
        "recover", help="synchronize pending client IDs; never resubmits a missing order"
    )
    _add_ledger(recover)
    _add_network_confirmation(recover)

    cancel = sub.add_parser("cancel", help="cancel one active ledger-tracked paper order")
    _add_ledger(cancel)
    _add_network_confirmation(cancel)
    cancel.add_argument("--order-id", required=True, help="local durable order ID")
    cancel.add_argument("--reason", required=True, help="operator reason recorded in the audit log")
    cancel.add_argument("--confirm-cancel", action="store_true")

    kill = sub.add_parser(
        "kill", help="persistently engage the kill switch and cancel every paper open order"
    )
    _add_ledger(kill)
    _add_network_confirmation(kill)
    kill.add_argument("--reason", required=True, help="operator reason recorded in the audit log")
    kill.add_argument("--confirm-cancel-open-orders", action="store_true")

    abandon = sub.add_parser(
        "abandon-missing",
        help="resolve a local active order only after paper broker confirms its client ID is absent",
    )
    _add_ledger(abandon)
    _add_network_confirmation(abandon)
    abandon.add_argument("--order-id", required=True)
    abandon.add_argument("--reason", required=True)
    abandon.add_argument("--confirm-abandon-missing", action="store_true")

    reset = sub.add_parser(
        "reset-kill", help="clear the persistent kill switch into disarmed state, then reconcile"
    )
    _add_ledger(reset)
    _add_network_confirmation(reset)
    reset.add_argument("--reason", required=True)
    reset.add_argument("--confirm-reset-kill", action="store_true")

    status = sub.add_parser("status", help="inspect a ledger without broker access or credentials")
    _add_ledger(status)
    status.add_argument("--account-id", required=True)

    disarm = sub.add_parser("disarm", help="persistently disarm a ledger without broker access")
    _add_ledger(disarm)
    disarm.add_argument("--account-id", required=True)
    disarm.add_argument("--reason", required=True)

    local_kill = sub.add_parser(
        "local-kill",
        help="engage the persistent kill without broker access; does not cancel broker orders",
    )
    _add_ledger(local_kill)
    local_kill.add_argument("--account-id", required=True)
    local_kill.add_argument("--reason", required=True)
    local_kill.add_argument("--confirm-local-kill", action="store_true")

    preview = sub.add_parser(
        "preview",
        help="build and persist an immutable paper batch; never submits an order",
    )
    _add_ledger(preview)
    _add_network_confirmation(preview)
    preview.add_argument("--deployment", required=True, help="strict deployment JSON path")
    preview.add_argument("--targets", required=True, help="strict sleeve-target JSON path")
    preview.add_argument(
        "--confirm-new-equity-session",
        action="store_true",
        help="allow initialization/roll of day-start equity from broker last_equity",
    )

    approve = sub.add_parser(
        "approve", help="offline approval of one exact preview hash; never contacts Alpaca"
    )
    _add_ledger(approve)
    approve.add_argument("--batch-id", required=True)
    approve.add_argument("--plan-hash", required=True, help="full 64-character reviewed hash")
    approve.add_argument("--operator", required=True)
    approve.add_argument("--reason", required=True)
    approve.add_argument("--confirm-approve-paper-batch", action="store_true")

    execute = sub.add_parser(
        "execute", help="submit only an approved immutable batch to Alpaca paper"
    )
    _add_ledger(execute)
    _add_network_confirmation(execute)
    execute.add_argument("--batch-id", required=True)
    execute.add_argument("--operator", required=True)
    execute.add_argument("--reason", required=True)
    execute.add_argument("--confirm-submit-paper-orders", action="store_true")

    batch_status = sub.add_parser(
        "batch-status", help="inspect one persisted Phase-3 batch without network access"
    )
    _add_ledger(batch_status)
    batch_status.add_argument("--batch-id", required=True)

    settle = sub.add_parser(
        "settle-batch",
        help="synchronize an already-started batch without submitting a new order",
    )
    _add_ledger(settle)
    _add_network_confirmation(settle)
    settle.add_argument("--batch-id", required=True)
    settle.add_argument("--operator", required=True)
    settle.add_argument("--reason", required=True)

    void = sub.add_parser(
        "void-batch", help="offline void of a never-started preview/approval"
    )
    _add_ledger(void)
    void.add_argument("--batch-id", required=True)
    void.add_argument("--operator", required=True)
    void.add_argument("--reason", required=True)
    void.add_argument("--confirm-void-paper-batch", action="store_true")
    return parser


def _add_ledger(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", required=True, help="SQLite execution-ledger path")


def _add_network_confirmation(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--confirm-paper-network",
        action="store_true",
        help="required acknowledgement that this contacts Alpaca's paper endpoint",
    )


def _broker() -> AlpacaPaperBroker:
    return AlpacaPaperBroker(AlpacaPaperConfig.from_env())


def _market_data() -> AlpacaPaperMarketData:
    return AlpacaPaperMarketData(AlpacaDataConfig.from_paper_env())


def _account_payload(broker: AlpacaPaperBroker) -> dict:
    account = broker.get_account()
    blockers = []
    if account.status != "ACTIVE":
        blockers.append(f"account status is {account.status}")
    if account.currency != "USD":
        blockers.append(f"account currency is {account.currency}")
    if account.trading_blocked:
        blockers.append("trading_blocked")
    if account.account_blocked:
        blockers.append("account_blocked")
    if account.trade_suspended_by_user:
        blockers.append("trade_suspended_by_user")
    if account.cash < 0 or account.buying_power < 0 or account.equity <= 0:
        blockers.append("invalid cash, buying power, or equity")
    return {
        "account_id": account.account_id,
        "cash": account.cash,
        "equity": account.equity,
        "buying_power": account.buying_power,
        "last_equity": account.last_equity,
        "as_of": account.as_of,
        "status": account.status,
        "currency": account.currency,
        "trading_blocked": account.trading_blocked,
        "account_blocked": account.account_blocked,
        "trade_suspended_by_user": account.trade_suspended_by_user,
        "phase_two_ready": not blockers,
        "readiness_blockers": blockers,
    }


def _diagnostics_payload(broker: AlpacaPaperBroker) -> dict:
    account = _account_payload(broker)
    positions = broker.get_positions()
    orders = broker.get_open_orders()
    market_clock = broker.get_market_clock()
    initial_baseline_blockers = list(account["readiness_blockers"])
    if any(position.quantity < 0 for position in positions):
        initial_baseline_blockers.append("account contains a short position")
    if any(
        position.quantity != position.quantity.to_integral_value()
        for position in positions
    ):
        initial_baseline_blockers.append("account contains a fractional position")
    if orders:
        initial_baseline_blockers.append("account has open orders requiring operator review")
    return {
        "adapter": "AlpacaPaperBroker",
        "endpoint": PAPER_BASE_URL,
        "paper_submission_command_available": True,
        "submission_requires_preview_and_exact_hash_approval": True,
        "initial_baseline_ready": not initial_baseline_blockers,
        "initial_baseline_blockers": initial_baseline_blockers,
        "account": account,
        "market_clock": market_clock,
        "positions": positions,
        "open_orders": orders,
    }


def _report_payload(report: ReconciliationReport) -> dict:
    return {
        "run_id": report.run_id,
        "account_id": report.account_id,
        "clean": report.clean,
        "issues": report.issues,
        "started_at": report.started_at,
        "completed_at": report.completed_at,
    }


def _result_payload(result: OMSResult) -> dict:
    return {
        "intent_id": result.intent_id,
        "intent_status": result.intent_status,
        "order_id": result.order_id,
        "client_order_id": result.client_order_id,
        "broker_order_id": result.broker_order_id,
    }


def _record_request_ids(
    ledger: Ledger,
    broker: AlpacaPaperBroker,
    command: str,
    account_id: str,
) -> tuple[str, ...]:
    request_ids = broker.drain_request_ids()
    if request_ids:
        ledger.record_audit(
            "alpaca_paper_api_requests",
            "account",
            account_id or "unresolved-paper-account",
            {"command": command, "request_ids": request_ids},
        )
    return request_ids


def _run_network_ledger_command(args: argparse.Namespace) -> tuple[dict, bool]:
    broker = _broker()
    account_id = ""
    payload: dict = {"adapter": "AlpacaPaperBroker", "endpoint": PAPER_BASE_URL}
    clean = True
    with Ledger(args.db) as ledger:
        try:
            account = broker.get_account()
            account_id = account.account_id
            ledger.assert_account_binding(account_id)
            reconciler = Reconciler(ledger, broker)
            oms = OrderManagementSystem(ledger, broker, PreTradeRiskEngine())

            if args.command == "bootstrap":
                reconciler.bootstrap_positions()
                report = reconciler.reconcile()
                payload["reconciliation"] = _report_payload(report)
                clean = report.clean
            elif args.command == "reconcile":
                report = reconciler.reconcile()
                payload["reconciliation"] = _report_payload(report)
                clean = report.clean
            elif args.command == "recover":
                payload["recovered"] = [_result_payload(r) for r in oms.recover_pending()]
                report = reconciler.reconcile()
                payload["reconciliation"] = _report_payload(report)
                clean = report.clean
            elif args.command == "cancel":
                result = oms.cancel_tracked_order(args.order_id, args.reason)
                cancellation_confirmed = result.intent_status in {
                    IntentStatus.CANCELED, IntentStatus.FILLED
                }
                payload["cancellation"] = {
                    **_result_payload(result),
                    "terminal_broker_outcome_confirmed": cancellation_confirmed,
                }
                report = reconciler.reconcile()
                payload["reconciliation"] = _report_payload(report)
                clean = report.clean and cancellation_confirmed
            elif args.command == "kill":
                oms.engage_kill_switch(args.reason, cancel_open_orders=True)
                payload["control"] = ledger.get_control_state(account_id)
                report = reconciler.reconcile()
                payload["reconciliation"] = _report_payload(report)
                clean = report.clean
            elif args.command == "abandon-missing":
                payload["resolution"] = _result_payload(
                    oms.abandon_missing_order(args.order_id, args.reason)
                )
                report = reconciler.reconcile()
                payload["reconciliation"] = _report_payload(report)
                clean = report.clean
            elif args.command == "reset-kill":
                control = ledger.get_control_state(account_id)
                if not control["kill_switch"]:
                    raise ExecutionBlocked("Kill switch is not engaged")
                oms.reset_kill_switch(args.reason)
                report = reconciler.reconcile()
                payload["control"] = ledger.get_control_state(account_id)
                payload["reconciliation"] = _report_payload(report)
                clean = report.clean
            else:
                raise RuntimeError(f"Unsupported network ledger command: {args.command}")

            payload["account"] = _account_payload(broker)
        finally:
            payload["request_ids"] = _record_request_ids(
                ledger, broker, args.command, account_id
            )
    return payload, clean


def _validate_confirmations(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.command in NETWORK_COMMANDS and not args.confirm_paper_network:
        parser.error("this command requires --confirm-paper-network")
    if args.command == "bootstrap" and not args.confirm_initial_baseline:
        parser.error("bootstrap requires --confirm-initial-baseline")
    if args.command == "cancel" and not args.confirm_cancel:
        parser.error("cancel requires --confirm-cancel")
    if args.command == "kill" and not args.confirm_cancel_open_orders:
        parser.error("kill requires --confirm-cancel-open-orders")
    if args.command == "abandon-missing" and not args.confirm_abandon_missing:
        parser.error("abandon-missing requires --confirm-abandon-missing")
    if args.command == "reset-kill" and not args.confirm_reset_kill:
        parser.error("reset-kill requires --confirm-reset-kill")
    if args.command == "local-kill" and not args.confirm_local_kill:
        parser.error("local-kill requires --confirm-local-kill")
    if args.command == "approve" and not args.confirm_approve_paper_batch:
        parser.error("approve requires --confirm-approve-paper-batch")
    if args.command == "execute" and not args.confirm_submit_paper_orders:
        parser.error("execute requires --confirm-submit-paper-orders")
    if args.command == "void-batch" and not args.confirm_void_paper_batch:
        parser.error("void-batch requires --confirm-void-paper-batch")


def _drain_ids(source: object) -> tuple[str, ...]:
    drain = getattr(source, "drain_request_ids", None)
    return tuple(drain()) if drain is not None else ()


def _run_phase3_preview(args: argparse.Namespace) -> dict:
    deployment = DeploymentConfig.from_file(args.deployment)
    targets = SleeveTargetSnapshot.from_file(args.targets, deployment)
    broker = _broker()
    data = _market_data()
    with Ledger(args.db) as ledger:
        try:
            service = PaperExecutionService(ledger, broker, data)
            plan, created = service.preview(
                deployment, targets,
                confirm_new_equity_session=args.confirm_new_equity_session,
            )
            return {
                "adapter": "AlpacaPaperBroker",
                "trading_endpoint": PAPER_BASE_URL,
                "market_data_endpoint": DATA_BASE_URL,
                "market_data_feed": "iex",
                "submits_orders": False,
                "created": created,
                "plan": plan.to_payload(),
                "next_step": (
                    "Review every field, then approve offline using batch_id and the full plan_hash"
                ),
            }
        finally:
            request_ids = {
                "trading": _drain_ids(broker), "market_data": _drain_ids(data)
            }
            bound = ledger.bound_account_id() or deployment.account_id
            ledger.record_audit(
                "alpaca_phase3_api_requests", "account", bound,
                {"command": "preview", "request_ids": request_ids},
            )


def _run_phase3_execute(args: argparse.Namespace) -> tuple[dict, bool]:
    broker = _broker()
    data = _market_data()
    with Ledger(args.db) as ledger:
        try:
            service = PaperExecutionService(ledger, broker, data)
            result = service.execute(
                args.batch_id, operator=args.operator, reason=args.reason
            )
            batch = ledger.get_execution_batch(args.batch_id)
            payload = {
                "adapter": "AlpacaPaperBroker",
                "trading_endpoint": PAPER_BASE_URL,
                "market_data_endpoint": DATA_BASE_URL,
                "market_data_feed": "iex",
                "paper_orders_may_have_been_submitted": True,
                "execution": result.to_payload(),
                "batch": {
                    key: value for key, value in (batch or {}).items() if key != "plan_json"
                },
            }
            return payload, result.status in {"submitted", "complete"}
        finally:
            request_ids = {
                "trading": _drain_ids(broker), "market_data": _drain_ids(data)
            }
            bound = ledger.bound_account_id() or "unresolved-paper-account"
            ledger.record_audit(
                "alpaca_phase3_api_requests", "account", bound,
                {"command": "execute", "request_ids": request_ids},
            )


def _run_phase3_settle(args: argparse.Namespace) -> dict:
    broker = _broker()
    data = _market_data()
    with Ledger(args.db) as ledger:
        try:
            result = PaperExecutionService(ledger, broker, data).settle(
                args.batch_id, operator=args.operator, reason=args.reason
            )
            batch = ledger.get_execution_batch(args.batch_id)
            return {
                "adapter": "AlpacaPaperBroker",
                "trading_endpoint": PAPER_BASE_URL,
                "submits_new_orders": False,
                "settlement": result.to_payload(),
                "batch": {
                    key: value for key, value in (batch or {}).items() if key != "plan_json"
                },
            }
        finally:
            request_ids = {
                "trading": _drain_ids(broker), "market_data": _drain_ids(data)
            }
            bound = ledger.bound_account_id() or "unresolved-paper-account"
            ledger.record_audit(
                "alpaca_phase3_api_requests", "account", bound,
                {"command": "settle-batch", "request_ids": request_ids},
            )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_confirmations(args, parser)
    try:
        if args.command == "status":
            with Ledger(args.db) as ledger:
                ledger.assert_account_binding(args.account_id)
                payload = ledger.snapshot(args.account_id)
            print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
            return 0
        if args.command == "disarm":
            with Ledger(args.db) as ledger:
                ledger.assert_account_binding(args.account_id)
                control = ledger.get_control_state(args.account_id)
                ledger.set_control_state(
                    args.account_id,
                    armed=False,
                    kill_switch=control["kill_switch"],
                    reason=args.reason,
                )
                payload = ledger.get_control_state(args.account_id)
            print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
            return 0
        if args.command == "local-kill":
            with Ledger(args.db) as ledger:
                ledger.assert_account_binding(args.account_id)
                ledger.set_control_state(
                    args.account_id,
                    armed=False,
                    kill_switch=True,
                    reason=args.reason,
                )
                payload = {
                    "control": ledger.get_control_state(args.account_id),
                    "broker_orders_canceled": False,
                    "operator_action_required": (
                        "Network was not contacted; inspect and cancel broker orders separately"
                    ),
                }
            print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
            return 0
        if args.command == "batch-status":
            with Ledger(args.db) as ledger:
                batch = ledger.get_execution_batch(args.batch_id)
                if batch is None:
                    raise LedgerError(f"Unknown execution batch: {args.batch_id}")
                plan = ledger.load_execution_plan(args.batch_id)
                payload = {
                    "batch": {key: value for key, value in batch.items() if key != "plan_json"},
                    "plan": plan.to_payload(),
                }
            print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
            return 0
        if args.command == "approve":
            with Ledger(args.db) as ledger:
                batch = ledger.approve_execution_batch(
                    args.batch_id, args.plan_hash,
                    approved_by=args.operator, reason=args.reason,
                )
                payload = {key: value for key, value in batch.items() if key != "plan_json"}
            print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
            return 0
        if args.command == "void-batch":
            with Ledger(args.db) as ledger:
                batch = ledger.void_execution_batch(
                    args.batch_id, operator=args.operator, reason=args.reason
                )
                payload = {key: value for key, value in batch.items() if key != "plan_json"}
            print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
            return 0
        if args.command == "preview":
            payload = _run_phase3_preview(args)
            print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
            return 0
        if args.command == "execute":
            payload, clean = _run_phase3_execute(args)
            print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
            return 0 if clean else 1
        if args.command == "settle-batch":
            payload = _run_phase3_settle(args)
            print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
            return 0
        if args.command == "check":
            broker = _broker()
            payload = _diagnostics_payload(broker)
            payload["request_ids"] = broker.drain_request_ids()
            print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
            return 0 if (
                payload["account"]["phase_two_ready"]
                and payload["initial_baseline_ready"]
            ) else 1

        if args.command == "kill":
            # Persist the emergency stop before any network call. The bound
            # account identity came from the irreversible opening baseline.
            with Ledger(args.db) as ledger:
                bound = ledger.bound_account_id()
                if bound is None:
                    raise ExecutionBlocked("Bootstrap the ledger before using networked kill")
                ledger.set_control_state(
                    bound, armed=False, kill_switch=True, reason=args.reason
                )

        payload, clean = _run_network_ledger_command(args)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        return 0 if clean else 1
    except (
        BrokerError, DeploymentError, ExecutionBlocked, LedgerError,
        MarketDataError, ValueError,
    ) as exc:
        print(json.dumps({
            "adapter": "AlpacaPaperBroker" if args.command in NETWORK_COMMANDS else "offline-ledger",
            "command": args.command,
            "error": str(exc),
        }, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
