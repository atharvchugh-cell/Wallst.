"""Phase-4 supervised paper-trading CLI. There is no live endpoint support."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from .alerts import AlertManager, StructuredConsoleSink
from .alpaca_data import AlpacaDataConfig, AlpacaPaperMarketData
from .alpaca_paper import AlpacaPaperBroker, AlpacaPaperConfig, PAPER_BASE_URL
from .backups import BackupManager
from .deployment import DeploymentConfig
from .health import HealthReporter
from .ledger import Ledger
from .models import ensure_aware, json_safe
from .phase4_models import (
    HMACFileSigner,
    OperationMode,
    Phase4Error,
    Phase4Policy,
    PublishedTargetSnapshot,
    account_fingerprint,
)
from .phase4_store import STREAM_LEASE_SECONDS, Phase4Store
from .publisher import (
    ResearchHistoricalDataSource,
    StrategyTargetPublisher,
    _sha256_file,
    collect_assets,
)
from .reconcile import Reconciler
from .scheduler import SupervisedMonthlyScheduler
from .soak import PaperSoakReporter
from .streaming import AlpacaPaperTradeUpdateStream, OrderStreamSupervisor
from .supervisor import Phase4Supervisor


DESCRIPTION = (
    "Wall St Strategy Lab Phase 4: supervised Alpaca PAPER only. Observe/shadow never "
    "submit; paper modes require explicit controls. Fake brokers exist only in tests. "
    "Live trading is not implemented."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    sub = parser.add_subparsers(dest="command", required=True)

    publish = sub.add_parser(
        "publish", help="publish one signed authentic monthly target; read-only paper/Yahoo network"
    )
    _common(publish, db=True)
    publish.add_argument("--snapshot-dir", required=True)
    publish.add_argument("--mode", choices=[mode.value for mode in OperationMode])
    publish.add_argument("--confirm-paper-network", action="store_true")
    publish.add_argument("--confirm-publish", action="store_true")
    publish.add_argument("--confirm-manual-catch-up", action="store_true")

    inspect = sub.add_parser(
        "inspect-snapshot",
        help="offline structural/hash inspection; signature authenticity is not checked",
    )
    inspect_group = inspect.add_mutually_exclusive_group(required=True)
    inspect_group.add_argument("--snapshot")
    inspect_group.add_argument("--snapshot-id")
    inspect.add_argument("--db")

    prepare = sub.add_parser(
        "prepare-plan", help="freeze fresh paper quotes into Phase-3 OMS plan; never submits"
    )
    _common(prepare, db=True)
    prepare.add_argument("--snapshot-id", required=True)
    prepare.add_argument("--confirm-paper-network", action="store_true")
    prepare.add_argument("--confirm-prepare-paper-plan", action="store_true")
    prepare.add_argument("--confirm-new-equity-session", action="store_true")

    approve = sub.add_parser("approve", help="offline exact-hash approval for paper modes")
    _common(approve, db=True)
    approve.add_argument("--batch-id", required=True)
    approve.add_argument("--plan-hash", required=True)
    approve.add_argument("--operator", required=True)
    approve.add_argument("--reason", required=True)
    approve.add_argument("--confirm-approve-paper-plan", action="store_true")

    run = sub.add_parser("run-paper", help="submit an approved plan to Alpaca PAPER only")
    _common(run, db=True)
    run.add_argument("--batch-id", required=True)
    run.add_argument("--operator", required=True)
    run.add_argument("--reason", required=True)
    run.add_argument("--confirm-paper-network", action="store_true")
    run.add_argument("--confirm-submit-paper-orders", action="store_true")

    reconcile = sub.add_parser("reconcile", help="persist broker/ledger paper reconciliation")
    _common(reconcile, db=True)
    reconcile.add_argument("--confirm-paper-network", action="store_true")
    reconcile.add_argument("--confirm-record-reconciliation", action="store_true")

    stream = sub.add_parser("stream", help="supervise Alpaca paper trade_updates and REST recovery")
    _common(stream, db=True)
    stream.add_argument("--confirm-paper-network", action="store_true")
    stream.add_argument("--confirm-start-paper-stream", action="store_true")
    stream.add_argument("--max-reconnects", type=int, default=8)

    health = sub.add_parser("health", help="report readiness; network is optional")
    _common(health, db=True)
    health.add_argument("--with-paper-network", action="store_true")
    health.add_argument("--confirm-paper-network", action="store_true")
    health.add_argument("--record-health-alerts", action="store_true")
    health.add_argument("--confirm-health-alert-write", action="store_true")

    alerts = sub.add_parser("alerts", help="list, acknowledge, or resolve durable alerts")
    alerts.add_argument("--db", required=True)
    alerts.add_argument("--unresolved-only", action="store_true")
    alert_action = alerts.add_mutually_exclusive_group()
    alert_action.add_argument("--acknowledge")
    alert_action.add_argument("--resolve")
    alerts.add_argument("--operator")
    alerts.add_argument("--note")
    alerts.add_argument("--confirm-alert-change", action="store_true")

    backup = sub.add_parser("backup", help="create and verify a versioned SQLite-safe backup")
    _common(backup, db=True)
    backup.add_argument("--backup-dir", required=True)
    backup.add_argument("--confirm-create-backup", action="store_true")

    restore = sub.add_parser("restore", help="verify and restore a backup into an explicit path")
    restore.add_argument("--db", required=True, help="active ledger path used only for safety check")
    restore.add_argument("--backup-dir", required=True)
    restore.add_argument("--destination", required=True)
    restore.add_argument("--confirm-replace-ledger", action="store_true")

    soak = sub.add_parser("soak-report", help="offline daily and cumulative paper-soak evidence")
    soak.add_argument("--db", required=True)
    soak.add_argument("--trading-date")

    observe_soak = sub.add_parser(
        "soak-observe", help="record an operator-sourced soak metric with backup"
    )
    _common(observe_soak, db=True)
    observe_soak.add_argument("--trading-date", required=True)
    observe_soak.add_argument(
        "--metric",
        required=True,
        choices=[
            "next_close_slippage_bps", "target_weight_error_bps", "process_uptime_seconds",
        ],
    )
    observe_soak.add_argument("--value", required=True)
    observe_soak.add_argument("--operator", required=True)
    observe_soak.add_argument("--reason", required=True)
    observe_soak.add_argument("--confirm-record-soak-observation", action="store_true")
    return parser


def _common(parser: argparse.ArgumentParser, *, db: bool) -> None:
    if db:
        parser.add_argument("--db", required=True)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--cache-dir", default="data_cache")


def _require(value: bool, message: str, parser: argparse.ArgumentParser) -> None:
    if not value:
        parser.error(message)


def _load(args) -> tuple[DeploymentConfig, Phase4Policy, object | None]:
    deployment = DeploymentConfig.from_file(args.deployment)
    policy = Phase4Policy.from_file(args.policy)
    policy.validate_deployment(deployment)
    key_path = os.getenv("WSLAB_PHASE4_SIGNING_KEY_FILE", "").strip()
    signer = HMACFileSigner(key_path, policy.signing_key_id) if key_path else None
    if policy.require_signing and signer is None:
        raise Phase4Error(
            "Set WSLAB_PHASE4_SIGNING_KEY_FILE to an operator-owned mode-0600 key file"
        )
    return deployment, policy, signer


def _paper_stack():
    config = AlpacaPaperConfig.from_env()
    return AlpacaPaperBroker(config), AlpacaPaperMarketData(
        AlpacaDataConfig(config.api_key, config.api_secret)
    )


def _alerts(store: Phase4Store) -> AlertManager:
    sinks = [StructuredConsoleSink()]
    # A configured webhook is intentionally not auto-enabled by the CLI. An
    # operator can wire WebhookSink from a supervised service after separately
    # approving external alert delivery.
    return AlertManager(store, tuple(sinks))


def _publisher(deployment, policy, signer, *, clock=None):
    return StrategyTargetPublisher(
        deployment, policy, repo_root=Path.cwd(), signer=signer,
        **({"clock": clock} if clock else {}),
    )


def _automatic_backup(ledger, policy, args, alerts):
    root = Path(policy.automatic_backup_directory).expanduser()
    if not root.is_absolute():
        root = Path(args.db).expanduser().resolve().parent / root
    return BackupManager(
        ledger, root, retention=policy.backup_retention, alerts=alerts
    ).create((args.deployment, args.policy))


def _print(payload) -> None:
    print(json.dumps(json_safe(payload), indent=2, sort_keys=True))


def _publish(args) -> dict:
    deployment, policy, signer = _load(args)
    if args.mode and args.mode != policy.mode.value:
        raise Phase4Error("--mode must match the immutable policy file")
    broker, market_quotes = _paper_stack()
    with Ledger(args.db) as ledger:
        store = Phase4Store(ledger)
        alerts = _alerts(store)
        scheduler = SupervisedMonthlyScheduler(broker, store, alerts=alerts)
        due, run, _created = scheduler.claim_due(
            confirm_manual_catch_up=args.confirm_manual_catch_up
        )
        publisher = _publisher(deployment, policy, signer)
        try:
            account = broker.get_account()
            ledger.assert_account_binding(account.account_id)
            if not ledger.positions_bootstrapped(account.account_id):
                raise Phase4Error(
                    "Phase-1 position baseline must be bootstrapped before publication"
                )
            positions = broker.get_positions()
            assets = collect_assets(broker, publisher.required_data_symbols())
            historical = ResearchHistoricalDataSource(cache_dir=args.cache_dir).load(
                publisher.required_data_symbols(), due.decision_day.trading_date
            )
            previous_row = store.latest_snapshot()
            previous = store.load_snapshot(previous_row["snapshot_id"]) if previous_row else None
            inputs = {
                str(Path(args.deployment).expanduser().resolve()): _sha256_file(
                    Path(args.deployment).expanduser().resolve()
                ),
                str(Path(args.policy).expanduser().resolve()): _sha256_file(
                    Path(args.policy).expanduser().resolve()
                ),
            }
            snapshot = publisher.publish(
                decision_day=due.decision_day,
                execution_day=due.execution_day,
                account=account,
                positions=positions,
                assets=assets,
                market_data=historical,
                input_file_hashes=inputs,
                previous_snapshot=previous,
            )
            output = Path(args.snapshot_dir).expanduser() / f"{snapshot.snapshot_id}.json"
            supervisor = Phase4Supervisor(
                ledger, deployment, policy, publisher, broker, market_quotes,
                signer=signer, alerts=alerts,
            )
            row, created = supervisor.persist_snapshot(snapshot, output_path=output)
            ledger.record_audit(
                "phase4_paper_request_ids", "target_snapshot", snapshot.snapshot_id,
                {"request_ids": list(broker.drain_request_ids())},
            )
            backup = _automatic_backup(ledger, policy, args, alerts)
            scheduler.mark_published(run["run_id"], snapshot.snapshot_id)
            return {
                "paper_only": True, "network_orders_submitted": False,
                "snapshot": snapshot.to_payload(), "snapshot_path": str(output.resolve()),
                "created": created, "scheduler_run": run["run_id"],
                "automatic_backup": backup,
            }
        except Exception as exc:
            lowered = str(exc).lower()
            if any(token in lowered for token in ("stale", "missing finalized", "calendar incomplete")):
                alerts.emit(
                    "critical", "stale_data", f"Publication data rejected: {type(exc).__name__}",
                    entity_id=run["run_id"], dedupe_key=f"stale-data:{run['run_id']}",
                )
            scheduler.mark_failed(run["run_id"], exc)
            raise


def _prepare(args) -> dict:
    deployment, policy, signer = _load(args)
    broker, market_data = _paper_stack()
    with Ledger(args.db) as ledger:
        store = Phase4Store(ledger)
        alerts = _alerts(store)
        publisher = _publisher(deployment, policy, signer)
        supervisor = Phase4Supervisor(
            ledger, deployment, policy, publisher, broker, market_data,
            signer=signer, alerts=alerts,
        )
        plan, created = supervisor.prepare_plan(
            args.snapshot_id,
            confirm_new_equity_session=args.confirm_new_equity_session,
        )
        backup = _automatic_backup(ledger, policy, args, alerts)
        return {
            "paper_only": True,
            "network_orders_submitted": False,
            "mode": policy.mode.value,
            "plan": plan.to_payload() if plan else None,
            "created": created,
            "shadow_permanently_non_submitting": policy.mode == OperationMode.SHADOW,
            "automatic_backup": backup,
        }


def _approve(args) -> dict:
    deployment, policy, signer = _load(args)
    with Ledger(args.db) as ledger:
        store = Phase4Store(ledger)
        alerts = _alerts(store)
        link = store.execution_plan_link(args.batch_id)
        if link is None or not link["paper_submission_allowed"]:
            raise Phase4Error("Batch is not an executable Phase-4 paper plan")
        if link["operation_mode"] not in {
            OperationMode.PAPER_MANUAL.value, OperationMode.PAPER_SUPERVISED.value,
        }:
            raise Phase4Error("Only manual/supervised paper plans may be approved")
        snapshot = store.load_snapshot(link["snapshot_id"])
        snapshot.verify(policy, signer, now=ledger.clock())
        if snapshot.content["account_id_fingerprint"] != account_fingerprint(
            deployment.account_id, policy.system_id
        ):
            raise Phase4Error("Snapshot account fingerprint mismatch")
        _publisher(deployment, policy, signer).to_execution_targets(snapshot)
        row = ledger.approve_execution_batch(
            args.batch_id, args.plan_hash, approved_by=args.operator, reason=args.reason
        )
        return {
            "paper_only": True, "network_used": False, "batch": row,
            "automatic_backup": _automatic_backup(ledger, policy, args, alerts),
        }


def _run_paper(args) -> dict:
    deployment, policy, signer = _load(args)
    broker, market_data = _paper_stack()
    with Ledger(args.db) as ledger:
        store = Phase4Store(ledger)
        alerts = _alerts(store)
        supervisor = Phase4Supervisor(
            ledger, deployment, policy, _publisher(deployment, policy, signer),
            broker, market_data, signer=signer, alerts=alerts,
        )
        result = supervisor.run_paper(args.batch_id, operator=args.operator, reason=args.reason)
        ledger.record_audit(
            "phase4_paper_request_ids", "execution_batch", args.batch_id,
            {"trading_request_ids": list(broker.drain_request_ids()),
             "data_request_ids": list(market_data.drain_request_ids())},
        )
        payload = result.to_payload()
        payload["automatic_backup"] = _automatic_backup(ledger, policy, args, alerts)
        return payload


def _reconcile(args) -> dict:
    deployment, policy, signer = _load(args)
    broker, _market_data = _paper_stack()
    with Ledger(args.db) as ledger:
        store = Phase4Store(ledger)
        alerts = _alerts(store)
        report = Reconciler(ledger, broker).reconcile()
        for issue in report.issues:
            category = {
                "CASH_MISMATCH": "cash_mismatch",
                "POSITION_MISMATCH": "unexpected_position",
                "EXTERNAL_OPEN_ORDER": "externally_created_order",
                "DUPLICATE_BROKER_OPEN_ORDER": "duplicate_intent",
            }.get(issue.code, "reconciliation_mismatch")
            alerts.emit(
                "critical", category, issue.message,
                entity_id=issue.entity_id or report.run_id,
                dedupe_key=f"reconcile:{issue.code}:{issue.entity_id}",
            )
        return {
            "run_id": report.run_id, "clean": report.clean,
            "issues": [issue.__dict__ for issue in report.issues],
            "automatic_backup": _automatic_backup(ledger, policy, args, alerts),
        }


def _health(args) -> dict:
    deployment, policy, signer = _load(args)
    broker = None
    scheduler = None
    with Ledger(args.db) as ledger:
        alerts = _alerts(Phase4Store(ledger)) if args.record_health_alerts else None
        if args.with_paper_network:
            broker, _data = _paper_stack()
            scheduler = SupervisedMonthlyScheduler(broker, Phase4Store(ledger))
        report = HealthReporter(
            ledger, policy, signer=signer, broker=broker, scheduler=scheduler,
            publisher=_publisher(deployment, policy, signer), alerts=alerts,
        ).report(deployment.account_id)
        if args.record_health_alerts:
            report["automatic_backup"] = _automatic_backup(
                ledger, policy, args, alerts
            )
        return report


def _stream(args) -> None:
    deployment, policy, signer = _load(args)
    broker, _market_data = _paper_stack()
    with Ledger(args.db) as ledger:
        account = broker.get_account()
        if account.account_id != deployment.account_id:
            raise Phase4Error("Authenticated paper account mismatch")
        alerts = _alerts(Phase4Store(ledger))
        supervisor = OrderStreamSupervisor(ledger, broker, alerts=alerts)
        supervisor.supervise(
            lambda: AlpacaPaperTradeUpdateStream(broker.config, account.account_id),
            max_reconnects=args.max_reconnects,
        )


def _assert_alert_resolution_safe(ledger: Ledger, store: Phase4Store, alert_id: str) -> None:
    alert = next((row for row in store.list_alerts() if row["alert_id"] == alert_id), None)
    if alert is None:
        raise Phase4Error(f"Unknown alert: {alert_id}")
    if alert["severity"] != "critical" or alert["resolved_at"] is not None:
        return
    accounts = ledger.known_account_ids()
    if not accounts:
        raise Phase4Error("Critical-alert resolution requires a known paper account")
    for account_id in accounts:
        latest = ledger.latest_reconciliation(account_id)
        if latest is None or not latest["clean"]:
            raise Phase4Error(
                "Critical-alert resolution requires a later clean reconciliation"
            )
        if ensure_aware(
            datetime.fromisoformat(latest["completed_at"]), "reconciliation time"
        ) <= ensure_aware(
            datetime.fromisoformat(alert["last_seen_at"]), "alert time"
        ):
            raise Phase4Error(
                "Critical-alert resolution requires reconciliation after the last incident"
            )
    if alert["category"] in {"broker_disconnection", "stream_recovery_failed"}:
        stream = store.stream_state()
        if stream is None or not stream["connected"] or stream["recovering"]:
            raise Phase4Error("Stream incident cannot be resolved before REST recovery")
        age = (
            ensure_aware(ledger.clock(), "alert resolution time")
            - ensure_aware(datetime.fromisoformat(stream["updated_at"]), "stream lease time")
        ).total_seconds()
        if age < -2 or age > STREAM_LEASE_SECONDS:
            raise Phase4Error("Stream incident cannot be resolved with a stale lease")
    if alert["category"] == "kill_switch":
        for account_id in accounts:
            if ledger.get_control_state(account_id)["kill_switch"]:
                raise Phase4Error("Kill-switch alert cannot be resolved while kill is engaged")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "publish":
            _require(args.confirm_paper_network, "publish requires --confirm-paper-network", parser)
            _require(args.confirm_publish, "publish requires --confirm-publish", parser)
            _print(_publish(args))
        elif args.command == "inspect-snapshot":
            if args.snapshot:
                snapshot = PublishedTargetSnapshot.from_file(args.snapshot)
            else:
                _require(bool(args.db), "--snapshot-id requires --db", parser)
                with Ledger(args.db) as ledger:
                    snapshot = Phase4Store(ledger).load_snapshot(args.snapshot_id)
            _print({
                "snapshot": snapshot.to_payload(),
                "structural_hash_valid": True,
                "signature_authentication_performed": False,
            })
        elif args.command == "prepare-plan":
            _require(args.confirm_paper_network, "prepare-plan requires --confirm-paper-network", parser)
            _require(
                args.confirm_prepare_paper_plan,
                "prepare-plan requires --confirm-prepare-paper-plan", parser,
            )
            _print(_prepare(args))
        elif args.command == "approve":
            _require(
                args.confirm_approve_paper_plan,
                "approve requires --confirm-approve-paper-plan", parser,
            )
            _print(_approve(args))
        elif args.command == "run-paper":
            _require(args.confirm_paper_network, "run-paper requires --confirm-paper-network", parser)
            _require(
                args.confirm_submit_paper_orders,
                "run-paper requires --confirm-submit-paper-orders", parser,
            )
            _print(_run_paper(args))
        elif args.command == "reconcile":
            _require(args.confirm_paper_network, "reconcile requires --confirm-paper-network", parser)
            _require(
                args.confirm_record_reconciliation,
                "reconcile requires --confirm-record-reconciliation", parser,
            )
            _print(_reconcile(args))
        elif args.command == "stream":
            _require(args.confirm_paper_network, "stream requires --confirm-paper-network", parser)
            _require(
                args.confirm_start_paper_stream,
                "stream requires --confirm-start-paper-stream", parser,
            )
            _stream(args)
        elif args.command == "health":
            if args.with_paper_network:
                _require(
                    args.confirm_paper_network,
                    "--with-paper-network requires --confirm-paper-network", parser,
                )
            if args.record_health_alerts:
                _require(
                    args.confirm_health_alert_write,
                    "--record-health-alerts requires --confirm-health-alert-write", parser,
                )
            elif args.confirm_health_alert_write:
                parser.error("--confirm-health-alert-write requires --record-health-alerts")
            _print(_health(args))
        elif args.command == "alerts":
            with Ledger(args.db) as ledger:
                store = Phase4Store(ledger)
                if args.acknowledge or args.resolve:
                    _require(args.confirm_alert_change, "alert change requires confirmation", parser)
                    _require(bool(args.operator and args.note), "alert change requires operator/note", parser)
                    if args.resolve:
                        _assert_alert_resolution_safe(ledger, store, args.resolve)
                    row = (
                        store.acknowledge_alert(args.acknowledge, operator=args.operator, note=args.note)
                        if args.acknowledge else
                        store.resolve_alert(args.resolve, operator=args.operator, note=args.note)
                    )
                    _print(row)
                else:
                    _print(store.list_alerts(unresolved_only=args.unresolved_only))
        elif args.command == "backup":
            _require(args.confirm_create_backup, "backup requires --confirm-create-backup", parser)
            deployment, policy, signer = _load(args)
            with Ledger(args.db) as ledger:
                row = BackupManager(
                    ledger, args.backup_dir, retention=policy.backup_retention,
                    alerts=_alerts(Phase4Store(ledger)),
                ).create((args.deployment, args.policy))
                _print(row)
        elif args.command == "restore":
            # Recovery verification must not open, create, or migrate the
            # damaged/active ledger. --db is a path-only replacement fence.
            manager = BackupManager(None, Path(args.backup_dir).parent)
            restored = manager.restore(
                args.backup_dir, args.destination,
                active_ledger_path=args.db,
                confirm_replace=args.confirm_replace_ledger,
            )
            _print({"restored_to": str(restored), "verified": True})
        elif args.command == "soak-report":
            with Ledger(args.db) as ledger:
                _print(PaperSoakReporter(ledger).report(args.trading_date))
        elif args.command == "soak-observe":
            _require(
                args.confirm_record_soak_observation,
                "soak-observe requires --confirm-record-soak-observation", parser,
            )
            deployment, policy, signer = _load(args)
            if not args.operator.strip() or not args.reason.strip():
                raise Phase4Error("soak-observe requires non-empty operator and reason")
            with Ledger(args.db) as ledger:
                store = Phase4Store(ledger)
                row = store.record_soak_observation(
                    args.trading_date, args.metric, args.value,
                    {"operator": args.operator.strip()[:100], "reason": args.reason.strip()[:500]},
                )
                _print({
                    "observation": row,
                    "automatic_backup": _automatic_backup(
                        ledger, policy, args, _alerts(store)
                    ),
                })
        return 0
    except (Phase4Error, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
