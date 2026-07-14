"""Restart-safe order management for account-level target intents."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal

from .broker import Broker, BrokerError
from .ledger import Ledger
from .models import (
    ACTIVE_ORDER_STATUSES,
    IntentStatus,
    OMSResult,
    OrderRequest,
    OrderStatus,
    OrderType,
    Quote,
    RiskViolation,
    Side,
    TargetPositionIntent,
    TimeInForce,
    ZERO,
    utc_now,
)
from .risk import PreTradeRiskEngine


class ExecutionBlocked(RuntimeError):
    pass


class ReconciliationRequired(ExecutionBlocked):
    pass


class KillSwitchError(ExecutionBlocked):
    pass


class OrderManagementSystem:
    def __init__(
        self,
        ledger: Ledger,
        broker: Broker,
        risk_engine: PreTradeRiskEngine,
        *,
        client_id_namespace: str = "wslab",
        arm_max_age_seconds: int = 900,
        clock=utc_now,
    ) -> None:
        if not client_id_namespace or not all(c.isalnum() or c in "-_" for c in client_id_namespace):
            raise ValueError("client_id_namespace may contain only letters, numbers, '-' and '_'")
        if len(client_id_namespace) > 95:
            raise ValueError("client_id_namespace is too long for deterministic broker client IDs")
        self.ledger = ledger
        self.broker = broker
        self.risk_engine = risk_engine
        self.client_id_namespace = client_id_namespace
        if arm_max_age_seconds <= 0:
            raise ValueError("arm_max_age_seconds must be positive")
        self.arm_max_age_seconds = arm_max_age_seconds
        self.clock = clock

    @property
    def account_id(self) -> str:
        return self.broker.get_account().account_id

    # --- Operational controls ------------------------------------------------

    def arm(self, reason: str, *, max_reconciliation_age_seconds: int = 300) -> None:
        """Arm only after a recent clean reconciliation and an explicit baseline."""
        if not reason.strip():
            raise ValueError("An operator reason is required to arm execution")
        account_id = self.account_id
        self.ledger.assert_account_binding(account_id)
        if not self.ledger.positions_bootstrapped(account_id):
            raise ReconciliationRequired("Position ledger must be bootstrapped before arming")
        control = self.ledger.get_control_state(account_id)
        if control["kill_switch"]:
            raise ExecutionBlocked("Reset the kill switch explicitly before arming")
        latest = self.ledger.latest_reconciliation(account_id)
        if latest is None or not bool(latest["clean"]):
            raise ReconciliationRequired("A clean reconciliation is required before arming")
        completed = datetime.fromisoformat(latest["completed_at"])
        if completed.tzinfo is None:
            completed = completed.replace(tzinfo=timezone.utc)
        if not self.ledger.reconciliation_follows_last_control_change(
            account_id, latest["run_id"]
        ):
            raise ReconciliationRequired(
                "A clean reconciliation newer than the last control change is required"
            )
        age = (self.clock() - completed.astimezone(timezone.utc)).total_seconds()
        if age < 0 or age > max_reconciliation_age_seconds:
            raise ReconciliationRequired("The latest clean reconciliation is stale")
        self.ledger.set_control_state(
            account_id, armed=True, kill_switch=False, reason=reason
        )

    def disarm(self, reason: str) -> None:
        account_id = self.account_id
        control = self.ledger.get_control_state(account_id)
        self.ledger.set_control_state(
            account_id,
            armed=False,
            kill_switch=control["kill_switch"],
            reason=reason,
        )

    def reset_kill_switch(self, reason: str) -> None:
        """Clear a kill only into the disarmed state; arming remains separate."""
        self.ledger.set_control_state(
            self.account_id, armed=False, kill_switch=False, reason=reason
        )

    def engage_kill_switch(self, reason: str, *, cancel_open_orders: bool = True) -> None:
        account_id = self.ledger.bound_account_id()
        if account_id is None:
            account_id = self.account_id
        self.ledger.set_control_state(
            account_id, armed=False, kill_switch=True, reason=reason
        )
        try:
            broker_account_id = self.account_id
        except BrokerError as exc:
            self.ledger.record_audit(
                "kill_switch_broker_identity_failed",
                "account",
                account_id,
                {"error": str(exc)},
            )
            raise KillSwitchError(
                "Kill switch remains engaged; broker account could not be reached"
            ) from exc
        if broker_account_id != account_id:
            raise KillSwitchError(
                "Kill switch remains engaged; authenticated broker account does not match ledger"
            )
        if not cancel_open_orders:
            return
        local_by_client = {
            o["client_order_id"]: o
            for o in self.ledger.list_orders()
            if o["account_id"] == account_id
        }
        failures: set[str] = set()
        try:
            open_orders = self.broker.get_open_orders()
        except BrokerError as exc:
            self.ledger.record_audit(
                "kill_switch_open_order_query_failed",
                "account",
                account_id,
                {"error": str(exc)},
            )
            raise KillSwitchError(
                "Kill switch remains engaged; broker open orders could not be queried"
            ) from exc
        for broker_order in open_orders:
            try:
                canceled = self.broker.cancel_order(broker_order.broker_order_id)
            except BrokerError as exc:
                failures.add(broker_order.broker_order_id)
                self.ledger.record_audit(
                    "kill_switch_cancellation_failed",
                    "broker_order",
                    broker_order.broker_order_id,
                    {"error": str(exc), "client_order_id": broker_order.client_order_id},
                )
                continue
            local = local_by_client.get(canceled.client_order_id)
            if local is not None:
                # A fill can win the race between listing and cancellation.
                # Synchronize broker truth instead of blindly marking canceled.
                self._submit_or_synchronize(local, allow_submit=False)
        try:
            remaining = self.broker.get_open_orders()
        except BrokerError as exc:
            self.ledger.record_audit(
                "kill_switch_cancellation_verification_failed",
                "account",
                account_id,
                {"error": str(exc)},
            )
            raise KillSwitchError(
                "Kill switch remains engaged; broker cancellations could not be verified"
            ) from exc
        for broker_order in remaining:
            failures.add(broker_order.broker_order_id)
        self.ledger.record_audit(
            "kill_switch_cancellation_complete",
            "account",
            account_id,
            {"cancel_open_orders": cancel_open_orders, "failed_order_ids": sorted(failures)},
        )
        if failures:
            raise KillSwitchError(
                f"Kill switch remains engaged; cancellation failed for {len(failures)} order(s)"
            )

    # --- Intent processing ---------------------------------------------------

    def process_intent(
        self,
        intent: TargetPositionIntent,
        *,
        quote: Quote,
        market_open: bool,
        day_start_equity: Decimal,
        high_water_equity: Decimal,
        trading_date: str | None = None,
    ) -> OMSResult:
        try:
            return self._process_intent(
                intent,
                quote=quote,
                market_open=market_open,
                day_start_equity=day_start_equity,
                high_water_equity=high_water_equity,
                trading_date=trading_date,
            )
        except BrokerError as exc:
            self._fail_closed_accounts(
                (intent.account_id,), "broker error during intent processing", exc
            )
            raise

    def _process_intent(
        self,
        intent: TargetPositionIntent,
        *,
        quote: Quote,
        market_open: bool,
        day_start_equity: Decimal,
        high_water_equity: Decimal,
        trading_date: str | None = None,
    ) -> OMSResult:
        account = self.broker.get_account()
        if intent.account_id != account.account_id:
            raise ExecutionBlocked("Intent account does not match broker account")
        self.ledger.assert_account_binding(account.account_id)
        row, created = self.ledger.create_intent(intent)
        intent_id = row["intent_id"]
        status = IntentStatus(row["status"])
        if status in {
            IntentStatus.NOOP,
            IntentStatus.RISK_REJECTED,
            IntentStatus.FILLED,
            IntentStatus.CANCELED,
            IntentStatus.BROKER_REJECTED,
        }:
            return self._result(intent_id, duplicate=not created)
        if not self.ledger.positions_bootstrapped(account.account_id):
            raise ReconciliationRequired("Position ledger has no explicit opening baseline")

        order = self.ledger.get_order_for_intent(intent_id)
        order_was_newly_planned = False
        if order is None:
            self._assert_account_alignment(account.account_id)
            active_for_symbol = [
                existing for existing in self.ledger.list_orders(active_only=True)
                if existing["account_id"] == account.account_id
                and existing["symbol"] == intent.symbol
            ]
            if active_for_symbol:
                raise ExecutionBlocked(
                    f"An active order already exists for {intent.symbol}; reconcile or cancel it first"
                )
            positions = self.broker.get_positions()
            current_qty = next(
                (p.quantity for p in positions if p.symbol == intent.symbol), ZERO
            )
            delta = intent.target_quantity - current_qty
            if delta == ZERO:
                self.ledger.set_intent_status(
                    intent_id, IntentStatus.NOOP, "broker position already equals target"
                )
                return self._result(intent_id, duplicate=not created)

            side = Side.BUY if delta > ZERO else Side.SELL
            request = OrderRequest(
                account_id=intent.account_id,
                client_order_id=f"{self.client_id_namespace}-{intent.idempotency_key[:32]}",
                intent_id=intent_id,
                symbol=intent.symbol,
                side=side,
                quantity=abs(delta),
                reference_price=intent.reference_price,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
            )
            control = self.ledger.get_control_state(account.account_id)
            if control["armed"] and control["updated_at"] is not None:
                armed_at = datetime.fromisoformat(control["updated_at"])
                if armed_at.tzinfo is None:
                    armed_at = armed_at.replace(tzinfo=timezone.utc)
                arm_age = (self.clock() - armed_at.astimezone(timezone.utc)).total_seconds()
                if arm_age < 0 or arm_age > self.arm_max_age_seconds:
                    self.ledger.set_control_state(
                        account.account_id,
                        armed=False,
                        kill_switch=control["kill_switch"],
                        reason="arming session expired before pre-trade risk",
                    )
                    control = self.ledger.get_control_state(account.account_id)
            decision = self.risk_engine.evaluate(
                request,
                quote=quote,
                account=account,
                positions=positions,
                open_order_count=len(self.broker.get_open_orders()),
                daily_turnover=self.ledger.daily_turnover(
                    account.account_id, trading_date or self.clock().date().isoformat()
                ),
                day_start_equity=day_start_equity,
                high_water_equity=high_water_equity,
                armed=control["armed"],
                kill_switch=control["kill_switch"],
                market_open=market_open,
                signal_at=intent.signal_at,
                now=self.clock(),
            )
            if not decision.allowed:
                circuit_breakers = {"DAILY_LOSS", "DRAWDOWN"}
                triggered = sorted(
                    v.code for v in decision.violations if v.code in circuit_breakers
                )
                if triggered:
                    self.ledger.set_control_state(
                        account.account_id,
                        armed=False,
                        kill_switch=True,
                        reason=f"risk circuit breaker: {', '.join(triggered)}",
                    )
                detail = json.dumps([
                    {"code": v.code, "message": v.message} for v in decision.violations
                ], sort_keys=True)
                self.ledger.set_intent_status(
                    intent_id,
                    IntentStatus.RISK_REJECTED,
                    detail,
                    payload={
                        "order_notional": decision.order_notional,
                        "projected_gross_exposure": decision.projected_gross_exposure,
                    },
                )
                return self._result(intent_id, duplicate=not created)
            order, order_was_newly_planned = self.ledger.plan_order_with_created(request)
            self.ledger.set_intent_status(
                intent_id, IntentStatus.ORDER_PENDING, "risk approved; durable order planned"
            )

        # Only the same call that passed current risk checks may submit an
        # unknown broker order. Later retries/restarts synchronize only.
        self._submit_or_synchronize(order, allow_submit=order_was_newly_planned)
        return self._result(intent_id, duplicate=not created)

    def recover_pending(self) -> list[OMSResult]:
        """Recover every risk-approved durable order after a restart.

        Recovery only synchronizes existing client IDs. A broker-missing local
        order remains pending for explicit operator resolution and is never
        automatically resubmitted after a restart.
        """
        results: list[OMSResult] = []
        for order in self.ledger.list_orders(active_only=True):
            try:
                self._submit_or_synchronize(order, allow_submit=False)
            except BrokerError as exc:
                self._fail_closed_accounts(
                    (order["account_id"],), "broker error during pending-order recovery", exc
                )
                raise
            results.append(self._result(order["intent_id"], duplicate=True))
        return results

    def cancel_tracked_order(self, order_id: str, reason: str) -> OMSResult:
        """Cancel one ledger-tracked broker order and synchronize race outcomes."""
        if not reason.strip():
            raise ValueError("An operator reason is required")
        order = self.ledger.get_order(order_id)
        if order is None:
            raise ExecutionBlocked(f"Unknown order: {order_id}")
        if OrderStatus(order["status"]) not in ACTIVE_ORDER_STATUSES:
            raise ExecutionBlocked("Only an active order can be canceled")
        try:
            broker_order = self.broker.get_order_by_client_id(order["client_order_id"])
        except BrokerError as exc:
            self._fail_closed_accounts(
                (order["account_id"],), "broker error during tracked-order cancellation", exc
            )
            raise
        if broker_order is None:
            raise ReconciliationRequired(
                "Broker cannot find this active order; reconcile and resolve it explicitly"
            )
        self.ledger.record_audit(
            "operator_cancel_requested",
            "order",
            order_id,
            {"reason": reason, "broker_order_id": broker_order.broker_order_id},
        )
        try:
            self.broker.cancel_order(broker_order.broker_order_id)
            self._submit_or_synchronize(order, allow_submit=False)
        except BrokerError as exc:
            self._fail_closed_accounts(
                (order["account_id"],), "broker error during tracked-order cancellation", exc
            )
            raise
        return self._result(order["intent_id"], duplicate=True)

    def abandon_missing_order(self, order_id: str, reason: str) -> OMSResult:
        """Auditable operator resolution for a pending order absent at broker.

        It is allowed only while disarmed. The caller must submit a new target
        version after this resolution; an old signal is never resurrected.
        """
        order = self.ledger.get_order(order_id)
        if order is None:
            raise ExecutionBlocked(f"Unknown order: {order_id}")
        control = self.ledger.get_control_state(order["account_id"])
        if control["armed"]:
            raise ExecutionBlocked("Disarm before abandoning a missing order")
        if self.broker.get_order_by_client_id(order["client_order_id"]) is not None:
            raise ExecutionBlocked("Broker still has this order; reconcile it instead")
        self.ledger.abandon_missing_order(order_id, reason)
        self.ledger.set_intent_status(
            order["intent_id"], IntentStatus.CANCELED, "operator abandoned missing broker order"
        )
        return self._result(order["intent_id"], duplicate=True)

    def _submit_or_synchronize(self, order: dict, *, allow_submit: bool) -> None:
        broker_order = self.broker.get_order_by_client_id(order["client_order_id"])
        acknowledged_under_guard = False
        if broker_order is None:
            if not allow_submit:
                self.ledger.record_audit(
                    "pending_order_not_resubmitted",
                    "order",
                    order["order_id"],
                    {"reason": "recovery synchronizes only; explicit fresh intent required"},
                )
                return
            with self.ledger.execution_guard():
                latest = self.ledger.get_order(order["order_id"])
                if latest is None or OrderStatus(latest["status"]) not in ACTIVE_ORDER_STATUSES:
                    raise ExecutionBlocked("Durable order is no longer active")
                control = self.ledger.get_control_state(order["account_id"])
                arm_age = self._control_age_seconds(control)
                if (
                    not control["armed"]
                    or control["kill_switch"]
                    or arm_age is None
                    or arm_age < 0
                    or arm_age > self.arm_max_age_seconds
                ):
                    self.ledger.record_audit(
                        "broker_submit_fenced",
                        "order",
                        order["order_id"],
                        {
                            "armed": control["armed"],
                            "kill_switch": control["kill_switch"],
                            "arm_age_seconds": arm_age,
                        },
                    )
                    raise ExecutionBlocked(
                        "Control state changed or expired before broker submission"
                    )
                # The process may have waited behind disarm/kill or another
                # submitter, so repeat client-ID discovery inside the fence.
                broker_order = self.broker.get_order_by_client_id(order["client_order_id"])
                if broker_order is None:
                    request = self._request_from_row(order)
                    try:
                        broker_order = self.broker.submit_order(request)
                    except BrokerError as exc:
                        self.ledger.record_audit(
                            "broker_submit_ambiguous",
                            "order",
                            order["order_id"],
                            {"client_order_id": order["client_order_id"], "error": str(exc)},
                        )
                        raise
                self.ledger.acknowledge_order(order["order_id"], broker_order)
                acknowledged_under_guard = True

        if not acknowledged_under_guard:
            self.ledger.acknowledge_order(order["order_id"], broker_order)
        for fill in self.broker.get_fills():
            if fill.client_order_id == order["client_order_id"]:
                self.ledger.record_fill(order["order_id"], fill)

        if broker_order.status == OrderStatus.REJECTED:
            self.ledger.set_intent_status(
                order["intent_id"], IntentStatus.BROKER_REJECTED, broker_order.rejection_reason
            )
        elif broker_order.status == OrderStatus.CANCELED:
            self.ledger.set_intent_status(
                order["intent_id"], IntentStatus.CANCELED, "broker order canceled"
            )
        elif (
            broker_order.status == OrderStatus.FILLED
            and self.ledger.filled_quantity_for_order(order["order_id"])
            == Decimal(order["quantity"])
        ):
            self.ledger.set_intent_status(
                order["intent_id"], IntentStatus.FILLED, "broker fill fully recorded"
            )
        else:
            self.ledger.set_intent_status(
                order["intent_id"], IntentStatus.ORDER_SUBMITTED, broker_order.status.value
            )

    def _control_age_seconds(self, control: dict) -> float | None:
        if control["updated_at"] is None:
            return None
        updated = datetime.fromisoformat(control["updated_at"])
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        return (self.clock() - updated.astimezone(timezone.utc)).total_seconds()

    def _fail_closed_accounts(
        self, account_ids: tuple[str, ...], reason: str, exc: Exception
    ) -> None:
        for account_id in dict.fromkeys(account_ids):
            if not account_id:
                continue
            control = self.ledger.get_control_state(account_id)
            self.ledger.set_control_state(
                account_id,
                armed=False,
                kill_switch=control["kill_switch"],
                reason=reason,
            )
            self.ledger.record_audit(
                "execution_failed_closed",
                "account",
                account_id,
                {"operation": reason, "error_type": type(exc).__name__, "error": str(exc)},
            )

    def _assert_account_alignment(self, account_id: str) -> None:
        local = {p.symbol: p.quantity for p in self.ledger.list_positions(account_id)}
        broker_rows = self.broker.get_positions()
        broker = {p.symbol: p.quantity for p in broker_rows}
        expected_cash = self.ledger.expected_cash(account_id)
        broker_cash = self.broker.get_account().cash
        local_open = {
            order["client_order_id"] for order in self.ledger.list_orders(active_only=True)
            if order["account_id"] == account_id
        }
        broker_open_rows = self.broker.get_open_orders()
        broker_open = {order.client_order_id for order in broker_open_rows}
        broker_order_ids = {order.broker_order_id for order in broker_open_rows}
        duplicate_broker_orders = (
            len(broker_open) != len(broker_open_rows)
            or len(broker_order_ids) != len(broker_open_rows)
        )
        if (
            len(broker) != len(broker_rows)
            or duplicate_broker_orders
            or local != broker
            or expected_cash is None
            or expected_cash != broker_cash
            or local_open != broker_open
        ):
            self.disarm("account mismatch detected before order construction")
            self.ledger.record_audit(
                "pretrade_account_mismatch",
                "account",
                account_id,
                {
                    "ledger_positions": local,
                    "broker_positions": broker,
                    "duplicate_broker_position_symbols": len(broker) != len(broker_rows),
                    "expected_cash": expected_cash,
                    "broker_cash": broker_cash,
                    "ledger_open_order_ids": sorted(local_open),
                    "broker_open_order_ids": sorted(broker_open),
                    "duplicate_broker_open_order_ids": duplicate_broker_orders,
                },
            )
            raise ReconciliationRequired(
                "Broker and ledger account state differ; execution was disarmed"
            )

    @staticmethod
    def _request_from_row(order: dict) -> OrderRequest:
        return OrderRequest(
            account_id=order["account_id"],
            client_order_id=order["client_order_id"],
            intent_id=order["intent_id"],
            symbol=order["symbol"],
            side=Side(order["side"]),
            quantity=Decimal(order["quantity"]),
            reference_price=Decimal(order["reference_price"]),
            order_type=OrderType(order["order_type"]),
            time_in_force=TimeInForce(order["time_in_force"]),
            limit_price=Decimal(order["limit_price"]) if order["limit_price"] else None,
        )

    def _result(self, intent_id: str, *, duplicate: bool) -> OMSResult:
        intent = self.ledger.get_intent(intent_id)
        if intent is None:
            raise RuntimeError(f"Missing intent {intent_id}")
        order = self.ledger.get_order_for_intent(intent_id)
        violations: tuple[RiskViolation, ...] = ()
        if intent["status"] == IntentStatus.RISK_REJECTED.value:
            try:
                parsed = json.loads(intent["status_detail"])
                violations = tuple(RiskViolation(v["code"], v["message"]) for v in parsed)
            except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                violations = (RiskViolation("RISK_REJECTED", intent["status_detail"]),)
        return OMSResult(
            intent_id=intent_id,
            intent_status=IntentStatus(intent["status"]),
            duplicate_intent=duplicate,
            order_id=order["order_id"] if order else None,
            client_order_id=order["client_order_id"] if order else None,
            broker_order_id=order["broker_order_id"] if order else None,
            risk_violations=violations,
        )
