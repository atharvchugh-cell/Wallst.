"""Phase-4 policy gates around the existing Phase-3 paper execution service."""

from __future__ import annotations

import json
import os
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from .alerts import AlertManager
from .deployment import DeploymentConfig, ExecutionPlan
from .execution import BatchExecutionResult, PaperExecutionService
from .ledger import Ledger, LedgerConflict
from .market_data import MarketDataProvider
from .models import IntentStatus, Quote, ZERO, ensure_aware
from .phase4_models import (
    OperationMode,
    Phase4Error,
    Phase4Policy,
    PublishedTargetSnapshot,
    SnapshotSigner,
    canonical_bytes,
)
from .phase4_store import (
    RECONCILIATION_READY_SECONDS,
    STREAM_LEASE_SECONDS,
    Phase4Store,
)
from .publisher import StrategyTargetPublisher, collect_assets


class GatedExecutionMarketData(MarketDataProvider):
    """Apply Phase-4 spread/deviation gates to every Phase-3 quote fetch."""

    def __init__(
        self,
        source: MarketDataProvider,
        policy: Phase4Policy,
        research_references: dict[str, Decimal],
        *,
        alerts: AlertManager | None = None,
    ) -> None:
        self.source = source
        self.policy = policy
        self.research_references = research_references
        self.alerts = alerts

    def get_quotes(self, symbols: tuple[str, ...]) -> dict[str, Quote]:
        quotes = self.source.get_quotes(symbols)
        for symbol in symbols:
            try:
                quote = quotes[symbol]
                spread_bps = (quote.ask - quote.bid) / quote.mid * Decimal("10000")
                if spread_bps > self.policy.max_quote_spread_bps:
                    raise Phase4Error(
                        f"{symbol} quote spread {spread_bps:.2f} bps exceeds policy"
                    )
                reference = self.research_references.get(symbol)
                if reference is None or reference <= ZERO:
                    raise Phase4Error(f"{symbol} has no valid signed research reference")
                deviation = abs(quote.mid / reference - Decimal("1")) * Decimal("10000")
                if deviation > self.policy.max_price_deviation_bps:
                    raise Phase4Error(
                        f"{symbol} quote deviation {deviation:.2f} bps exceeds policy"
                    )
            except Exception as exc:
                if self.alerts:
                    self.alerts.emit(
                        "high", "invalid_execution_quote", str(exc),
                        entity_id=symbol, dedupe_key=f"invalid-quote:{symbol}",
                    )
                raise
        return quotes


class Phase4Supervisor:
    def __init__(
        self,
        ledger: Ledger,
        deployment: DeploymentConfig,
        policy: Phase4Policy,
        publisher: StrategyTargetPublisher,
        broker,
        market_data: MarketDataProvider,
        *,
        signer: SnapshotSigner | None = None,
        alerts: AlertManager | None = None,
        clock=None,
    ) -> None:
        policy.validate_deployment(deployment)
        if publisher.policy.to_payload() != policy.to_payload():
            raise Phase4Error("Supervisor and publisher Phase-4 policies differ")
        if publisher.deployment.to_payload() != deployment.to_payload():
            raise Phase4Error("Supervisor and publisher deployments differ")
        self.ledger = ledger
        self.store = Phase4Store(ledger)
        self.deployment = deployment
        self.policy = policy
        self.publisher = publisher
        self.broker = broker
        self.signer = signer
        self.alerts = alerts
        self.clock = clock or ledger.clock
        self.market_data = market_data

    def persist_snapshot(
        self, snapshot: PublishedTargetSnapshot, *, output_path: str | Path | None = None
    ) -> tuple[dict, bool]:
        snapshot.verify(self.policy, self.signer, now=ensure_aware(self.clock(), "publish time"))
        if snapshot.content.get("operation_mode") != self.policy.mode.value:
            raise Phase4Error("Snapshot operating mode does not match active Phase-4 policy")
        # Persistence is an authorization boundary, not a generic blob write:
        # re-check account/config/code/universe/input provenance before the
        # signed envelope becomes the durable publication of record.
        self.publisher.to_execution_targets(snapshot)
        row, created = self.store.publish_snapshot(snapshot)
        if output_path is not None:
            self._write_immutable_snapshot(output_path, snapshot)
        return row, created

    @staticmethod
    def _write_immutable_snapshot(
        path: str | Path, snapshot: PublishedTargetSnapshot
    ) -> None:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            snapshot.to_payload(), indent=2, sort_keys=True, ensure_ascii=True
        ).encode("utf-8") + b"\n"
        try:
            fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        except FileExistsError:
            existing = PublishedTargetSnapshot.from_file(target)
            if canonical_bytes(existing.to_payload()) != canonical_bytes(snapshot.to_payload()):
                raise Phase4Error("Immutable snapshot path already contains different content")
            return
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(target, 0o400)
        except Exception:
            try:
                target.unlink()
            except OSError:
                pass
            raise

    def prepare_plan(
        self,
        snapshot_id: str,
        *,
        confirm_new_equity_session: bool,
    ) -> tuple[ExecutionPlan | None, bool]:
        snapshot = self.store.load_snapshot(snapshot_id)
        try:
            snapshot.verify(self.policy, self.signer, now=ensure_aware(self.clock(), "plan time"))
        except Exception as exc:
            if self.alerts:
                category = "snapshot_expiration" if "expired" in str(exc).lower() else "signature_failure"
                self.alerts.emit(
                    "critical", category, f"Snapshot validation failed: {type(exc).__name__}",
                    entity_id=snapshot_id, dedupe_key=f"{category}:{snapshot_id}",
                )
            raise
        if snapshot.content["operation_mode"] != self.policy.mode.value:
            raise Phase4Error("Stored snapshot operating mode differs from active policy")
        if self.policy.mode == OperationMode.OBSERVE:
            self.ledger.record_audit(
                "phase4_observe_completed", "target_snapshot", snapshot_id,
                {"paper_plan_created": False, "paper_submission_allowed": False},
            )
            return None, False

        if not hasattr(self.broker, "get_asset"):
            raise Phase4Error("Phase-4 broker must provide current asset metadata")
        self.publisher.validate_assets_for_execution(
            snapshot,
            collect_assets(self.broker, self.publisher.required_data_symbols()),
        )
        targets = self.publisher.to_execution_targets(snapshot)
        existing_link = self.store.execution_plan_link_for_snapshot(snapshot_id)
        if existing_link is not None:
            if (
                existing_link["operation_mode"] != self.policy.mode.value
                or bool(existing_link["paper_submission_allowed"])
                != self.policy.mode.can_submit_paper
            ):
                raise Phase4Error("Stored execution-plan link differs from active policy")
            plan = self.ledger.load_execution_plan(existing_link["batch_id"])
            if self.policy.mode == OperationMode.SHADOW:
                self._assert_finalized_shadow(existing_link["batch_id"])
            return plan, False
        references = {
            row["symbol"]: Decimal(str(row["research_reference_price"]))
            for row in snapshot.content["required_target_deltas"]
        }
        gated_data = GatedExecutionMarketData(
            self.market_data, self.policy, references, alerts=self.alerts
        )
        service = PaperExecutionService(
            self.ledger, self.broker, gated_data, clock=self.clock
        )
        paper_allowed = self.policy.mode.can_submit_paper
        try:
            plan, created = service.preview(
                self.deployment,
                targets,
                confirm_new_equity_session=confirm_new_equity_session,
                plan_validator=lambda value, quotes: self._validate_phase4_plan(
                    value,
                    Decimal(str(snapshot.content["account_equity_used_for_sizing"])),
                    str(snapshot.content["expected_execution_session"]),
                    quotes,
                ),
                phase4_link=(snapshot.snapshot_id, self.policy.mode.value, paper_allowed),
            )
            if self.policy.mode == OperationMode.SHADOW:
                approval_reason = "automatic shadow approval; permanently non-submitting plan"
                row = self.ledger.get_execution_batch(plan.batch_id)
                if row["status"] == "previewed":
                    row = self.ledger.approve_execution_batch(
                        plan.batch_id, plan.plan_hash,
                        approved_by=self.policy.publisher_identity,
                        reason=approval_reason,
                    )
                if row["status"] == "approved":
                    row = self.ledger.void_execution_batch(
                        plan.batch_id,
                        operator=self.policy.publisher_identity,
                        reason="shadow artifact finalized; broker submission permanently forbidden",
                    )
                self._assert_finalized_shadow(plan.batch_id)
            return plan, created
        except Exception as exc:
            if self.alerts:
                self.alerts.emit(
                    "high", "risk_rejection", f"Plan preparation failed: {type(exc).__name__}",
                    entity_id=snapshot_id, dedupe_key=f"plan-failed:{snapshot_id}",
                )
            raise

    def _assert_finalized_shadow(self, batch_id: str) -> None:
        row = self.ledger.get_execution_batch(batch_id)
        if (
            row is None
            or row["status"] != "voided"
            or row["approved_by"] != self.policy.publisher_identity
            or row["approval_reason"]
            != "automatic shadow approval; permanently non-submitting plan"
        ):
            raise Phase4Error("Stored shadow batch is not a finalized shadow artifact")

    def _validate_phase4_plan(
        self,
        plan: ExecutionPlan,
        snapshot_equity: Decimal,
        expected_execution_session: str,
        quotes: dict[str, Quote],
    ) -> None:
        if plan.account_id != self.deployment.account_id:
            raise Phase4Error("Execution plan account drifted from deployment")
        if plan.trading_date != expected_execution_session:
            raise Phase4Error("Execution plan is not for the snapshot's signed next session")
        deployed = sum(
            (item.target_quantity * quotes[item.symbol].ask for item in plan.items), ZERO
        )
        if deployed / snapshot_equity > self.policy.max_cash_deployment_pct:
            raise Phase4Error("Execution plan exceeds max_cash_deployment_pct")
        for item in plan.items:
            if item.delta_quantity == ZERO:
                continue
            execution_price = (
                quotes[item.symbol].ask
                if item.delta_quantity > ZERO else quotes[item.symbol].bid
            )
            notional = abs(item.delta_quantity) * execution_price
            if notional < self.policy.min_trade_notional:
                raise Phase4Error(
                    f"{item.symbol} order is below configured minimum trade notional"
                )
            if item.target_weight > self.policy.max_aggregate_ticker_weight:
                raise Phase4Error(f"{item.symbol} exceeds aggregate ticker concentration")

    def approve(
        self,
        batch_id: str,
        plan_hash: str,
        *,
        operator: str,
        reason: str,
    ) -> dict:
        link = self.store.execution_plan_link(batch_id)
        if link is None:
            raise Phase4Error("Batch is not linked to an immutable Phase-4 target snapshot")
        if link["operation_mode"] not in {
            OperationMode.PAPER_MANUAL.value, OperationMode.PAPER_SUPERVISED.value,
        }:
            raise Phase4Error("Observe and shadow plans cannot receive paper approval")
        snapshot = self.store.load_snapshot(link["snapshot_id"])
        snapshot.verify(self.policy, self.signer, now=ensure_aware(self.clock(), "approval time"))
        self.publisher.to_execution_targets(snapshot)
        return self.ledger.approve_execution_batch(
            batch_id, plan_hash, approved_by=operator, reason=reason
        )

    def run_paper(
        self, batch_id: str, *, operator: str, reason: str
    ) -> BatchExecutionResult:
        self._authorize_linked_submission(batch_id)
        if not self.policy.mode.can_submit_paper:
            raise Phase4Error(f"{self.policy.mode.value} mode can never submit paper orders")
        link = self.store.execution_plan_link(batch_id)
        if link is None or not link["paper_submission_allowed"]:
            raise Phase4Error("Batch has no durable Phase-4 paper-submission authority")
        if link["operation_mode"] != self.policy.mode.value:
            raise Phase4Error("Batch operating mode differs from active policy")
        snapshot = self.store.load_snapshot(link["snapshot_id"])
        try:
            snapshot.verify(
                self.policy, self.signer, now=ensure_aware(self.clock(), "execution time")
            )
        except Exception as exc:
            if self.alerts:
                category = "snapshot_expiration" if "expired" in str(exc).lower() else "signature_failure"
                self.alerts.emit(
                    "critical", category, f"Snapshot validation failed: {type(exc).__name__}",
                    entity_id=link["snapshot_id"],
                    dedupe_key=f"{category}:{link['snapshot_id']}",
                )
            raise
        if not hasattr(self.broker, "get_asset"):
            raise Phase4Error("Phase-4 broker must provide current asset metadata")
        self.publisher.validate_assets_for_execution(
            snapshot,
            collect_assets(self.broker, self.publisher.required_data_symbols()),
        )
        self.publisher.to_execution_targets(snapshot)
        stream = self.store.stream_state()
        if stream is None or not stream["connected"] or stream["recovering"]:
            raise Phase4Error(
                "Paper submission is blocked until order-stream REST recovery passes"
            )
        stream_updated = ensure_aware(
            datetime.fromisoformat(stream["updated_at"]), "stream lease time"
        )
        stream_age = (ensure_aware(self.clock(), "stream lease check") - stream_updated).total_seconds()
        if stream_age < -2 or stream_age > STREAM_LEASE_SECONDS:
            raise Phase4Error("Paper submission is blocked because the order-stream lease is stale")
        if self.policy.mode == OperationMode.PAPER_SUPERVISED:
            self._assert_supervised_health()
        references = {
            row["symbol"]: Decimal(str(row["research_reference_price"]))
            for row in snapshot.content["required_target_deltas"]
        }
        service = PaperExecutionService(
            self.ledger,
            self.broker,
            GatedExecutionMarketData(
                self.market_data, self.policy, references, alerts=self.alerts
            ),
            clock=self.clock,
        )
        try:
            result = service.execute(
                batch_id, operator=operator, reason=reason,
                phase4_authorizer=self._authorize_linked_submission,
            )
            if self.alerts:
                for item in result.results:
                    if item.intent_status == IntentStatus.BROKER_REJECTED:
                        self.alerts.emit(
                            "critical", "order_rejection", "Paper broker rejected an OMS order",
                            entity_id=item.intent_id,
                            dedupe_key=f"broker-rejected:{item.intent_id}",
                        )
                    elif item.intent_status == IntentStatus.RISK_REJECTED:
                        self.alerts.emit(
                            "high", "risk_rejection", "OMS pre-trade risk rejected an order",
                            entity_id=item.intent_id,
                            dedupe_key=f"risk-rejected:{item.intent_id}",
                        )
            return result
        except Exception as exc:
            if self.alerts:
                self.alerts.emit(
                    "critical", "paper_execution_failure",
                    f"{type(exc).__name__}: {str(exc)[:500]}",
                    entity_id=batch_id, dedupe_key=f"execution-failed:{batch_id}",
                )
            raise

    def _authorize_linked_submission(self, batch_id: str) -> None:
        """Revalidate every Phase-4 authority boundary immediately before OMS use."""
        if not self.policy.mode.can_submit_paper:
            raise Phase4Error(f"{self.policy.mode.value} mode can never submit paper orders")
        link = self.store.execution_plan_link(batch_id)
        if (
            link is None
            or not bool(link["paper_submission_allowed"])
            or link["operation_mode"] != self.policy.mode.value
        ):
            raise Phase4Error("Batch has no current Phase-4 paper-submission authority")
        snapshot = self.store.load_snapshot(link["snapshot_id"])
        snapshot.verify(
            self.policy, self.signer,
            now=ensure_aware(self.clock(), "submission authorization time"),
        )
        if not hasattr(self.broker, "get_asset"):
            raise Phase4Error("Phase-4 broker must provide current asset metadata")
        self.publisher.validate_assets_for_execution(
            snapshot,
            collect_assets(self.broker, self.publisher.required_data_symbols()),
        )
        self.publisher.to_execution_targets(snapshot)
        control = self.ledger.get_control_state(self.deployment.account_id)
        if control["kill_switch"]:
            raise Phase4Error("Persistent kill switch is engaged")
        latest = self.ledger.latest_reconciliation(self.deployment.account_id)
        if latest is None or not latest["clean"]:
            raise Phase4Error("Paper submission requires a clean reconciliation")
        reconciled_at = ensure_aware(
            datetime.fromisoformat(latest["completed_at"]), "reconciliation time"
        )
        reconciliation_age = (
            ensure_aware(self.clock(), "submission authorization time") - reconciled_at
        ).total_seconds()
        if reconciliation_age < -2 or reconciliation_age > RECONCILIATION_READY_SECONDS:
            raise Phase4Error("Paper submission requires a recent clean reconciliation")
        stream = self.store.stream_state()
        if stream is None or not stream["connected"] or stream["recovering"]:
            raise Phase4Error("Paper submission requires a healthy recovered order stream")
        stream_updated = ensure_aware(
            datetime.fromisoformat(stream["updated_at"]), "stream lease time"
        )
        stream_age = (
            ensure_aware(self.clock(), "stream lease check") - stream_updated
        ).total_seconds()
        if stream_age < -2 or stream_age > STREAM_LEASE_SECONDS:
            raise Phase4Error("Paper submission is blocked because the order-stream lease is stale")
        critical = [
            row for row in self.store.list_alerts(unresolved_only=True)
            if row["severity"] == "critical"
        ]
        if critical:
            raise Phase4Error("Unresolved critical alerts block paper submission")

    def _assert_supervised_health(self) -> None:
        account_id = self.deployment.account_id
        control = self.ledger.get_control_state(account_id)
        if control["kill_switch"]:
            raise Phase4Error("Persistent kill switch is engaged")
        latest = self.ledger.latest_reconciliation(account_id)
        if latest is None or not latest["clean"]:
            raise Phase4Error("Supervised paper mode requires a clean reconciliation")
        stream = self.store.stream_state()
        if stream is None or not stream["connected"] or stream["recovering"]:
            raise Phase4Error("Supervised paper mode requires a healthy recovered order stream")
        critical = [
            row for row in self.store.list_alerts(unresolved_only=True)
            if row["severity"] == "critical"
        ]
        if critical:
            raise Phase4Error("Unresolved critical alerts block supervised paper submission")
