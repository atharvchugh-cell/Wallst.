"""Broker-vs-ledger startup and end-of-cycle reconciliation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal

from .broker import Broker
from .ledger import Ledger, LedgerConflict
from .models import (
    ACTIVE_ORDER_STATUSES,
    AccountSnapshot,
    IntentStatus,
    OrderStatus,
    ZERO,
    json_safe,
    utc_now,
)


@dataclass(frozen=True)
class ReconciliationIssue:
    code: str
    message: str
    entity_id: str = ""


@dataclass(frozen=True)
class ReconciliationReport:
    run_id: str
    account_id: str
    clean: bool
    issues: tuple[ReconciliationIssue, ...]
    started_at: str
    completed_at: str


class Reconciler:
    def __init__(
        self,
        ledger: Ledger,
        broker: Broker,
        *,
        client_id_namespace: str = "wslab",
        clock=utc_now,
    ) -> None:
        self.ledger = ledger
        self.broker = broker
        self.client_id_prefix = f"{client_id_namespace}-"
        self.clock = clock

    def bootstrap_positions(self) -> None:
        """Create the explicit one-time position and cash baseline from broker truth."""
        account = self.broker.get_account()
        self.ledger.bootstrap_positions(account, self.broker.get_positions())

    def reconcile(self, *, synchronize_known_orders: bool = True) -> ReconciliationReport:
        started = self.clock()
        try:
            account = self.broker.get_account()
        except Exception as exc:
            self._record_failed_reconciliation(None, exc)
            raise
        try:
            return self._reconcile_account(
                started, account, synchronize_known_orders=synchronize_known_orders
            )
        except Exception as exc:
            self._record_failed_reconciliation(account.account_id, exc)
            raise

    def _reconcile_account(
        self,
        started,
        account: AccountSnapshot,
        *,
        synchronize_known_orders: bool,
    ) -> ReconciliationReport:
        self.ledger.assert_account_binding(account.account_id)
        issues: list[ReconciliationIssue] = []
        if account.status != "ACTIVE":
            issues.append(ReconciliationIssue(
                "ACCOUNT_NOT_ACTIVE",
                f"Broker account status is {account.status}",
                account.account_id,
            ))
        if account.currency != "USD":
            issues.append(ReconciliationIssue(
                "UNSUPPORTED_CURRENCY",
                f"Broker account currency is {account.currency}; only USD is supported",
                account.account_id,
            ))
        if account.trading_blocked or account.account_blocked or account.trade_suspended_by_user:
            issues.append(ReconciliationIssue(
                "BROKER_TRADING_BLOCKED",
                "Broker reports that account trading is blocked or suspended",
                account.account_id,
            ))
        if account.cash < ZERO or account.buying_power < ZERO or account.equity <= ZERO:
            issues.append(ReconciliationIssue(
                "ACCOUNT_FINANCIAL_STATE_INVALID",
                "Broker account has negative cash/buying power or nonpositive equity",
                account.account_id,
            ))
        local_orders = [
            order for order in self.ledger.list_orders()
            if order["account_id"] == account.account_id
        ]
        local_by_client = {o["client_order_id"]: o for o in local_orders}

        if not self.ledger.positions_bootstrapped(account.account_id):
            issues.append(ReconciliationIssue(
                "POSITION_BASELINE_MISSING",
                "No explicit opening broker-position baseline has been recorded",
                account.account_id,
            ))
        if self.ledger.expected_cash(account.account_id) is None:
            issues.append(ReconciliationIssue(
                "CASH_BASELINE_MISSING",
                "No explicit opening broker-cash baseline has been recorded",
                account.account_id,
            ))

        if synchronize_known_orders:
            broker_fills = self.broker.get_fills()
            for local in local_orders:
                broker_order = self.broker.get_order_by_client_id(local["client_order_id"])
                if broker_order is None:
                    continue
                try:
                    self.ledger.acknowledge_order(local["order_id"], broker_order)
                    for fill in broker_fills:
                        if fill.client_order_id == local["client_order_id"]:
                            self.ledger.record_fill(local["order_id"], fill)
                except LedgerConflict as exc:
                    issues.append(ReconciliationIssue(
                        "ORDER_STATE_CONFLICT",
                        str(exc),
                        local["order_id"],
                    ))
                    continue
                filled = self.ledger.filled_quantity_for_order(local["order_id"])
                if broker_order.status == OrderStatus.FILLED and filled == Decimal(local["quantity"]):
                    self.ledger.set_intent_status(
                        local["intent_id"], IntentStatus.FILLED, "reconciliation imported complete fill"
                    )
                elif broker_order.status == OrderStatus.CANCELED:
                    self.ledger.set_intent_status(
                        local["intent_id"], IntentStatus.CANCELED, "reconciliation observed cancellation"
                    )
                elif broker_order.status == OrderStatus.REJECTED:
                    self.ledger.set_intent_status(
                        local["intent_id"], IntentStatus.BROKER_REJECTED,
                        broker_order.rejection_reason,
                    )

        # Re-read local state after importing broker events.
        local_orders = [
            order for order in self.ledger.list_orders()
            if order["account_id"] == account.account_id
        ]
        local_by_client = {o["client_order_id"]: o for o in local_orders}
        for local in local_orders:
            if OrderStatus(local["status"]) not in ACTIVE_ORDER_STATUSES:
                continue
            broker_order = self.broker.get_order_by_client_id(local["client_order_id"])
            if broker_order is None:
                issues.append(ReconciliationIssue(
                    "LOCAL_ACTIVE_ORDER_MISSING",
                    "Ledger has an active order that the broker cannot find by client ID",
                    local["order_id"],
                ))

        broker_open_orders = self.broker.get_open_orders()
        open_broker_ids = [order.broker_order_id for order in broker_open_orders]
        open_client_ids = [order.client_order_id for order in broker_open_orders]
        if (
            len(open_broker_ids) != len(set(open_broker_ids))
            or len(open_client_ids) != len(set(open_client_ids))
        ):
            issues.append(ReconciliationIssue(
                "DUPLICATE_BROKER_OPEN_ORDER",
                "Broker returned duplicate open-order identifiers",
                account.account_id,
            ))
        for broker_order in broker_open_orders:
            if broker_order.client_order_id not in local_by_client:
                code = (
                    "UNTRACKED_SYSTEM_ORDER"
                    if broker_order.client_order_id.startswith(self.client_id_prefix)
                    else "EXTERNAL_OPEN_ORDER"
                )
                issues.append(ReconciliationIssue(
                    code,
                    "Broker has an open order that is absent from the local ledger",
                    broker_order.client_order_id,
                ))

        for fill in self.broker.get_fills():
            if (
                fill.client_order_id.startswith(self.client_id_prefix)
                and fill.client_order_id not in local_by_client
            ):
                issues.append(ReconciliationIssue(
                    "UNTRACKED_SYSTEM_FILL",
                    "Broker has a namespaced fill with no matching local order",
                    fill.fill_id,
                ))

        local_positions = {
            p.symbol: p.quantity for p in self.ledger.list_positions(account.account_id)
        }
        broker_position_rows = self.broker.get_positions()
        broker_symbols = [position.symbol for position in broker_position_rows]
        if len(broker_symbols) != len(set(broker_symbols)):
            issues.append(ReconciliationIssue(
                "DUPLICATE_BROKER_POSITION",
                "Broker returned more than one position row for a symbol",
                account.account_id,
            ))
        for position in broker_position_rows:
            if position.quantity < ZERO:
                issues.append(ReconciliationIssue(
                    "EXISTING_SHORT",
                    f"{position.symbol}: short positions are unsupported",
                    position.symbol,
                ))
            if position.quantity != position.quantity.to_integral_value():
                issues.append(ReconciliationIssue(
                    "FRACTIONAL_POSITION_UNSUPPORTED",
                    f"{position.symbol}: fractional positions are unsupported",
                    position.symbol,
                ))
        broker_positions = {
            position.symbol: position.quantity for position in broker_position_rows
        }
        for symbol in sorted(set(local_positions) | set(broker_positions)):
            local_qty = local_positions.get(symbol, ZERO)
            broker_qty = broker_positions.get(symbol, ZERO)
            if local_qty != broker_qty:
                issues.append(ReconciliationIssue(
                    "POSITION_MISMATCH",
                    f"{symbol}: ledger={local_qty}, broker={broker_qty}",
                    symbol,
                ))

        expected_cash = self.ledger.expected_cash(account.account_id)
        if expected_cash is not None and expected_cash != account.cash:
            issues.append(ReconciliationIssue(
                "CASH_MISMATCH",
                f"ledger={expected_cash}, broker={account.cash}",
                account.account_id,
            ))

        completed = self.clock()
        run_id = f"rec-{completed.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        issue_payload = [json_safe({
            "code": i.code, "message": i.message, "entity_id": i.entity_id
        }) for i in issues]
        self.ledger.record_reconciliation(
            run_id,
            account.account_id,
            started.isoformat(),
            completed.isoformat(),
            not issues,
            issue_payload,
        )
        if issues:
            control = self.ledger.get_control_state(account.account_id)
            self.ledger.set_control_state(
                account.account_id,
                armed=False,
                kill_switch=control["kill_switch"],
                reason=f"reconciliation {run_id} found {len(issues)} issue(s)",
            )
        return ReconciliationReport(
            run_id=run_id,
            account_id=account.account_id,
            clean=not issues,
            issues=tuple(issues),
            started_at=started.isoformat(),
            completed_at=completed.isoformat(),
        )

    def _record_failed_reconciliation(
        self, observed_account_id: str | None, exc: Exception
    ) -> None:
        known = list(self.ledger.known_account_ids())
        try:
            bound = self.ledger.bound_account_id()
        except LedgerConflict:
            bound = None
        if not known and bound:
            known.append(bound)
        if not known and observed_account_id:
            known.append(observed_account_id)
        entity_id = bound or observed_account_id or "unbound-ledger"
        for account_id in dict.fromkeys(known):
            control = self.ledger.get_control_state(account_id)
            self.ledger.set_control_state(
                account_id,
                armed=False,
                kill_switch=control["kill_switch"],
                reason="reconciliation aborted; prior clean state invalidated",
            )
        self.ledger.record_audit(
            "reconciliation_failed",
            "account",
            entity_id,
            {"error_type": type(exc).__name__, "error": str(exc)},
        )
