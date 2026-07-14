"""Phase-3 reviewed paper-batch orchestration.

The service has no scheduler and no live endpoint. A batch must be previewed,
persisted, approved by exact hash, and explicitly executed against a paper
broker. Every execution attempt revalidates account, reconciliation, exchange
clock, quotes, signal age, equity guardrails, and OMS risk limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable

from .broker import Broker
from .deployment import (
    DeploymentConfig,
    DeploymentError,
    ExecutionPlan,
    SleeveTargetSnapshot,
    build_execution_plan,
)
from .ledger import Ledger, LedgerConflict
from .market_data import (
    NEW_YORK,
    MarketDataError,
    MarketDataProvider,
    MarketSession,
    validate_quotes,
    validate_regular_session,
    validate_signal_session,
)
from .models import (
    IntentStatus,
    OMSResult,
    TargetPositionIntent,
    ZERO,
    ensure_aware,
    json_safe,
    utc_now,
)
from .oms import ExecutionBlocked, OrderManagementSystem
from .reconcile import Reconciler
from .risk import PreTradeRiskEngine


@dataclass(frozen=True)
class BatchExecutionResult:
    batch_id: str
    status: str
    results: tuple[OMSResult, ...]
    reconciliation_clean: bool

    def to_payload(self) -> dict[str, Any]:
        return json_safe({
            "batch_id": self.batch_id,
            "status": self.status,
            "results": self.results,
            "reconciliation_clean": self.reconciliation_clean,
        })


class PaperExecutionService:
    def __init__(
        self,
        ledger: Ledger,
        broker: Broker,
        market_data: MarketDataProvider,
        *,
        clock=utc_now,
    ) -> None:
        self.ledger = ledger
        self.broker = broker
        self.market_data = market_data
        self.clock = clock

    def preview(
        self,
        deployment: DeploymentConfig,
        targets: SleeveTargetSnapshot,
        *,
        confirm_new_equity_session: bool,
        plan_validator: Callable[[ExecutionPlan, dict[str, Any]], None] | None = None,
        phase4_link: tuple[str, str, bool] | None = None,
    ) -> tuple[ExecutionPlan, bool]:
        self.ledger.assert_account_binding(deployment.account_id)
        local_control = self.ledger.get_control_state(deployment.account_id)
        if local_control["armed"]:
            self.ledger.set_control_state(
                deployment.account_id, armed=False,
                kill_switch=local_control["kill_switch"],
                reason="Phase-3 preview invalidated prior submit authority",
            )
        account = self.broker.get_account()
        now = ensure_aware(self.clock(), "preview time")
        self._validate_account(account, deployment.account_id, now, deployment)
        self.ledger.assert_account_binding(account.account_id)
        if not self.ledger.positions_bootstrapped(account.account_id):
            raise ExecutionBlocked("Bootstrap and reconcile the dedicated paper ledger first")
        control = self.ledger.get_control_state(account.account_id)
        if control["armed"] or control["kill_switch"]:
            raise ExecutionBlocked("Preview requires a disarmed account with kill clear")
        signal_age = (now - targets.signal_at).total_seconds()
        if signal_age < 0:
            raise DeploymentError("Target signal timestamp is in the future")
        if signal_age > deployment.risk_limits.max_signal_age_seconds:
            raise DeploymentError("Target signal is stale")

        report = Reconciler(self.ledger, self.broker, clock=self.clock).reconcile()
        if not report.clean:
            raise ExecutionBlocked("A clean reconciliation is required before preview")
        if self.broker.get_open_orders() or self.ledger.list_orders(active_only=True):
            raise ExecutionBlocked("Preview requires zero broker and ledger open orders")
        session = self._market_session()
        self._validate_signal_session(targets.signal_at, session.trading_date)
        self._validate_assets(deployment.managed_symbols)
        quotes = self.market_data.get_quotes(deployment.managed_symbols)
        validate_quotes(
            quotes, deployment.managed_symbols,
            now=ensure_aware(self.clock(), "preview quote validation time"),
            max_age_seconds=deployment.risk_limits.quote_max_age_seconds,
        )
        final_report = Reconciler(self.ledger, self.broker, clock=self.clock).reconcile()
        if not final_report.clean:
            raise ExecutionBlocked("Reconciliation changed while preview data was collected")
        if self.broker.get_open_orders() or self.ledger.list_orders(active_only=True):
            raise ExecutionBlocked("An order appeared while preview data was collected")
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        self._validate_account(
            account, deployment.account_id,
            ensure_aware(self.clock(), "preview sizing account time"), deployment,
        )
        # Re-check freshness after the reconciliation/account/position calls.
        # Otherwise a quote that was fresh when received could age beyond the
        # configured boundary before its price is frozen into an approved plan.
        validate_quotes(
            quotes, deployment.managed_symbols,
            now=ensure_aware(self.clock(), "preview sizing quote validation time"),
            max_age_seconds=deployment.risk_limits.quote_max_age_seconds,
        )
        equity = self.ledger.observe_equity(
            account, session.trading_date,
            allow_new_session=confirm_new_equity_session,
        )
        if equity["trading_date"] != session.trading_date:
            raise LedgerConflict("Equity guardrail session does not match exchange trading date")
        plan = build_execution_plan(
            deployment, targets, account=account,
            positions=positions, quotes=quotes,
            trading_date=session.trading_date,
            daily_turnover=self.ledger.daily_turnover(
                account.account_id, session.trading_date
            ),
        )
        if plan_validator is not None:
            plan_validator(plan, quotes)
        _row, created = self.ledger.create_execution_batch(plan, phase4_link=phase4_link)
        return plan, created

    def execute(
        self,
        batch_id: str,
        *,
        operator: str,
        reason: str,
        phase4_authorizer: Callable[[str], None] | None = None,
    ) -> BatchExecutionResult:
        if not operator.strip() or not reason.strip():
            raise ValueError("Execution requires an operator and reason")
        with self.ledger.batch_execution_guard():
            return self._execute_under_guard(
                batch_id, operator=operator, reason=reason,
                phase4_authorizer=phase4_authorizer,
            )

    def settle(self, batch_id: str, *, operator: str, reason: str) -> BatchExecutionResult:
        """Synchronize an already-started batch without submitting new orders."""
        if not operator.strip() or not reason.strip():
            raise ValueError("Settlement requires an operator and reason")
        with self.ledger.batch_execution_guard():
            row = self.ledger.get_execution_batch(batch_id)
            if row is None:
                raise ExecutionBlocked(f"Unknown execution batch: {batch_id}")
            plan = self.ledger.load_execution_plan(batch_id)
            control = self.ledger.get_control_state(plan.account_id)
            if control["armed"]:
                self.ledger.set_control_state(
                    plan.account_id, armed=False,
                    kill_switch=control["kill_switch"],
                    reason=f"settle paper batch {batch_id}",
                )
            if row["status"] not in {"executing", "submitted", "complete", "failed"}:
                raise ExecutionBlocked("Only an already-started batch can be settled")
            account = self.broker.get_account()
            self._validate_account(
                account, plan.account_id,
                ensure_aware(self.clock(), "settlement account time"),
                _PlanDeployment(plan),
            )
            self.ledger.assert_account_binding(account.account_id)
            self.ledger.record_audit(
                "execution_batch_settlement_requested",
                "execution_batch",
                batch_id,
                {"operator": operator.strip()[:100], "reason": reason.strip()[:500]},
            )
            oms = OrderManagementSystem(
                self.ledger, self.broker,
                PreTradeRiskEngine(plan.risk_limits, clock=self.clock),
                clock=self.clock,
            )
            oms.recover_pending()
            report = Reconciler(self.ledger, self.broker, clock=self.clock).reconcile()
            if not report.clean:
                raise ExecutionBlocked("Settlement reconciliation is not clean")
            if row["status"] in {"complete", "failed"}:
                return BatchExecutionResult(batch_id, row["status"], (), report.clean)
            strategy_id = f"deployment:{plan.deployment_id}:aggregate"
            intents = []
            missing = []
            for item in plan.items:
                if item.delta_quantity == ZERO:
                    continue
                intent = self.ledger.find_intent(
                    account_id=plan.account_id,
                    strategy_id=strategy_id,
                    symbol=item.symbol,
                    signal_at=plan.signal_at.isoformat(),
                    target_version=plan.target_version,
                )
                if intent is None:
                    missing.append(item.symbol)
                else:
                    intents.append(intent)
            statuses = {IntentStatus(intent["status"]) for intent in intents}
            rejected = {
                IntentStatus.RISK_REJECTED,
                IntentStatus.BROKER_REJECTED,
                IntentStatus.CANCELED,
            }
            final_status: str | None = None
            if statuses & rejected:
                final_status = "failed"
            elif not missing and statuses <= {IntentStatus.FILLED, IntentStatus.NOOP}:
                final_status = "complete"
            elif missing and self._broker_trading_date() > plan.trading_date:
                final_status = "failed"
            if final_status is not None:
                self.ledger.begin_execution_batch(batch_id)
                self.ledger.set_execution_batch_status(batch_id, final_status)
                return BatchExecutionResult(batch_id, final_status, (), report.clean)
            return BatchExecutionResult(batch_id, row["status"], (), report.clean)

    def _execute_under_guard(
        self,
        batch_id: str,
        *,
        operator: str,
        reason: str,
        phase4_authorizer: Callable[[str], None] | None,
    ) -> BatchExecutionResult:
        row = self.ledger.get_execution_batch(batch_id)
        if row is None:
            raise ExecutionBlocked(f"Unknown execution batch: {batch_id}")
        plan = self.ledger.load_execution_plan(batch_id)
        phase4_link = self.ledger.conn.execute(
            "SELECT * FROM phase4_plan_links WHERE batch_id = ?", (batch_id,)
        ).fetchone()
        if phase4_link is not None and not bool(phase4_link["paper_submission_allowed"]):
            raise ExecutionBlocked(
                f"Phase-4 {phase4_link['operation_mode']} plan is permanently non-submitting"
            )
        if phase4_link is not None:
            if phase4_authorizer is None:
                raise ExecutionBlocked(
                    "Phase-4 linked batches must execute through Phase4Supervisor"
                )
            # Run inside the same cross-process batch guard as submission so a
            # legacy Phase-3 call cannot race or bypass current Phase-4 gates.
            phase4_authorizer(batch_id)
        local_control = self.ledger.get_control_state(plan.account_id)
        if local_control["armed"]:
            self.ledger.set_control_state(
                plan.account_id, armed=False,
                kill_switch=local_control["kill_switch"],
                reason=f"Phase-3 batch {batch_id} started from a disarmed boundary",
            )
        if row["status"] == "complete":
            # Idempotent replay does not contact the broker or claim a fresh
            # reconciliation. Use settle-batch when a current clean assertion
            # is required.
            return BatchExecutionResult(batch_id, "complete", (), False)
        if row["status"] not in {"approved", "executing", "submitted"}:
            raise ExecutionBlocked(f"Batch must be approved before execution; status={row['status']}")
        if not row["approved_at"] or not row["approved_by"] or not row["approval_reason"]:
            raise ExecutionBlocked("Batch approval artifact is incomplete")
        account = self.broker.get_account()
        now = ensure_aware(self.clock(), "execution time")
        deployment_stub = _PlanDeployment(plan)
        self._validate_account(account, plan.account_id, now, deployment_stub)
        self.ledger.assert_account_binding(account.account_id)
        control = self.ledger.get_control_state(account.account_id)
        if control["kill_switch"]:
            raise ExecutionBlocked("Persistent kill switch is engaged")
        signal_age = (now - plan.signal_at).total_seconds()
        if signal_age < 0 or signal_age > plan.risk_limits.max_signal_age_seconds:
            raise ExecutionBlocked("Approved batch signal is future-dated or stale")

        session = self._market_session()
        if session.trading_date != plan.trading_date:
            raise ExecutionBlocked("Approved batch may execute only on its preview trading date")
        symbols = tuple(item.symbol for item in plan.items)
        quotes = self.market_data.get_quotes(symbols)
        validate_quotes(
            quotes, symbols,
            now=ensure_aware(self.clock(), "execution quote validation time"),
            max_age_seconds=plan.risk_limits.quote_max_age_seconds,
        )
        account = self.broker.get_account()
        self._validate_account(
            account, plan.account_id,
            ensure_aware(self.clock(), "execution equity account time"), deployment_stub,
        )
        equity = self.ledger.observe_equity(
            account, session.trading_date, allow_new_session=False
        )

        risk = PreTradeRiskEngine(plan.risk_limits, clock=self.clock)
        oms = OrderManagementSystem(self.ledger, self.broker, risk, clock=self.clock)
        reconciler = Reconciler(self.ledger, self.broker, clock=self.clock)
        results: list[OMSResult] = []
        began = False
        try:
            # Synchronize known client IDs first. Recovery never submits a
            # broker-missing order; reconciliation decides whether resumption
            # is safe.
            oms.recover_pending()
            before = reconciler.reconcile()
            if not before.clean:
                raise ExecutionBlocked("A clean reconciliation is required before execution")
            active_local = self.ledger.list_orders(active_only=True)
            if row["status"] == "approved" and active_local:
                raise ExecutionBlocked("Fresh approved batch requires zero active orders")
            for active_order in active_local:
                active_intent = self.ledger.get_intent(active_order["intent_id"])
                if (
                    active_intent is None
                    or active_intent["target_version"] != plan.target_version
                    or active_intent["reason"] != f"approved paper batch {batch_id}"
                ):
                    raise ExecutionBlocked(
                        "Active order does not belong to the batch being resumed"
                    )
            self._validate_assets(symbols)
            self.ledger.begin_execution_batch(batch_id)
            began = True
            oms.arm(
                f"approved paper batch {batch_id}: {operator.strip()} - {reason.strip()}",
                max_reconciliation_age_seconds=300,
            )
            actionable = [item for item in plan.items if item.delta_quantity != ZERO]
            actionable.sort(key=lambda item: (item.delta_quantity >= ZERO, item.symbol))
            for item in actionable:
                intent = TargetPositionIntent(
                    account_id=plan.account_id,
                    strategy_id=f"deployment:{plan.deployment_id}:aggregate",
                    symbol=item.symbol,
                    target_quantity=item.target_quantity,
                    signal_at=plan.signal_at,
                    target_version=plan.target_version,
                    reference_price=item.reference_price,
                    reason=f"approved paper batch {batch_id}",
                )
                result = oms.process_intent(
                    intent,
                    quote=quotes[item.symbol],
                    market_open=session.is_open,
                    day_start_equity=Decimal(equity["day_start_equity"]),
                    high_water_equity=Decimal(equity["high_water_equity"]),
                    trading_date=session.trading_date,
                )
                results.append(result)
                if result.intent_status in {
                    IntentStatus.RISK_REJECTED,
                    IntentStatus.BROKER_REJECTED,
                    IntentStatus.CANCELED,
                }:
                    # Do not compound a partial-portfolio failure by sending
                    # later items after any member of the approved batch fails.
                    break
                if result.intent_status in {
                    IntentStatus.ORDER_PENDING,
                    IntentStatus.ORDER_SUBMITTED,
                }:
                    # Keep at most one unresolved order. This prevents later
                    # items from double-counting cash/buying power or exposure
                    # that an accepted asynchronous order has reserved but not
                    # yet reflected in positions/fills.
                    break
            after = reconciler.reconcile()
            if not after.clean:
                raise ExecutionBlocked("Post-submission reconciliation is not clean")
            rejected = {
                IntentStatus.RISK_REJECTED,
                IntentStatus.BROKER_REJECTED,
                IntentStatus.CANCELED,
            }
            if any(result.intent_status in rejected for result in results):
                status = "failed"
            elif any(result.intent_status in {
                IntentStatus.ORDER_PENDING, IntentStatus.ORDER_SUBMITTED,
            } for result in results):
                status = "submitted"
            else:
                status = "complete"
            self.ledger.set_execution_batch_status(batch_id, status)
            return BatchExecutionResult(batch_id, status, tuple(results), after.clean)
        except Exception as exc:
            if began:
                self.ledger.record_execution_batch_error(batch_id, str(exc))
            raise
        finally:
            # Never depend on another broker call to remove submit authority.
            latest_control = self.ledger.get_control_state(plan.account_id)
            if latest_control["armed"]:
                self.ledger.set_control_state(
                    plan.account_id,
                    armed=False,
                    kill_switch=latest_control["kill_switch"],
                    reason=f"paper batch {batch_id} execution attempt ended",
                )

    def _market_session(self) -> MarketSession:
        getter = getattr(self.broker, "get_market_clock", None)
        if getter is None:
            raise MarketDataError("Broker does not provide an exchange clock")
        raw = getter()
        try:
            session = MarketSession(
                raw.timestamp, raw.is_open, raw.next_open, raw.next_close
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise MarketDataError("Broker exchange-clock response is malformed") from exc
        return validate_regular_session(
            session, now=ensure_aware(self.clock(), "exchange-clock receipt time")
        )

    def _validate_signal_session(self, signal_at: datetime, execution_date: str) -> None:
        getter = getattr(self.broker, "get_market_calendar", None)
        if getter is None:
            raise MarketDataError("Broker does not provide an exchange calendar")
        execution_day = date.fromisoformat(execution_date)
        signal_day = signal_at.astimezone(NEW_YORK).date()
        if signal_day >= execution_day:
            raise DeploymentError(
                "Daily-close strategy signal must precede the execution trading date"
            )
        try:
            days = getter(signal_day, execution_day)
            validate_signal_session(
                tuple(days), signal_at=signal_at, execution_date=execution_date
            )
        except MarketDataError:
            raise
        except Exception as exc:
            raise MarketDataError("Broker exchange-calendar response is malformed") from exc

    def _broker_trading_date(self) -> str:
        getter = getattr(self.broker, "get_market_clock", None)
        if getter is None:
            raise MarketDataError("Broker does not provide an exchange clock")
        raw = getter()
        try:
            timestamp = ensure_aware(raw.timestamp, "broker clock timestamp")
            now = ensure_aware(self.clock(), "broker clock receipt time")
            age = (now - timestamp).total_seconds()
            if age < -2 or age > 30:
                raise ValueError("broker clock timestamp is future-dated or stale")
            return timestamp.astimezone(NEW_YORK).date().isoformat()
        except (AttributeError, TypeError, ValueError) as exc:
            raise MarketDataError("Broker exchange-clock response is malformed") from exc

    def _validate_assets(self, symbols: tuple[str, ...]) -> None:
        getter = getattr(self.broker, "get_asset", None)
        if getter is None:
            return
        for symbol in symbols:
            asset = getter(symbol)
            if (
                getattr(asset, "symbol", None) != symbol
                or getattr(asset, "asset_class", None) != "us_equity"
                or getattr(asset, "status", None) != "active"
                or getattr(asset, "tradable", None) is not True
            ):
                raise ExecutionBlocked(f"Managed symbol is not an active tradable US equity: {symbol}")

    @staticmethod
    def _validate_account(
        account: Any,
        expected_account_id: str,
        now: datetime,
        deployment: Any,
    ) -> None:
        if account.account_id != expected_account_id:
            raise ExecutionBlocked("Deployment account does not match authenticated paper account")
        if account.status != "ACTIVE" or account.currency != "USD":
            raise ExecutionBlocked("Paper account must be ACTIVE and denominated in USD")
        if account.trading_blocked or account.account_blocked or account.trade_suspended_by_user:
            raise ExecutionBlocked("Paper account reports trading blocked or suspended")
        if account.cash < ZERO or account.equity <= ZERO or account.buying_power < ZERO:
            raise ExecutionBlocked("Paper account returned invalid cash, equity, or buying power")
        age = (now - account.as_of).total_seconds()
        if age < 0 or age > deployment.risk_limits.account_max_age_seconds:
            raise ExecutionBlocked("Paper account snapshot is future-dated or stale")


class _PlanDeployment:
    """Minimal view used to share account validation for a stored plan."""

    def __init__(self, plan: ExecutionPlan) -> None:
        self.risk_limits = plan.risk_limits
