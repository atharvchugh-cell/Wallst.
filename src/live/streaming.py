"""Durable Alpaca-paper order-stream supervision and REST recovery."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Callable, Iterable, Protocol

from .alerts import AlertManager
from .alpaca_paper import ALPACA_ORDER_STATUS, AlpacaPaperConfig
from .broker import Broker
from .ledger import Ledger, LedgerConflict
from .models import (
    BrokerOrder,
    Fill,
    IntentStatus,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
    ZERO,
    as_decimal,
    ensure_aware,
    json_safe,
)
from .oms import OrderManagementSystem
from .phase4_models import Phase4Error, canonical_bytes
from .phase4_store import (
    CONTINUOUS_RECONCILE_SECONDS,
    PHASE4_STREAM_NAME,
    Phase4Store,
)
from .reconcile import Reconciler, ReconciliationReport
from .risk import PreTradeRiskEngine


ALPACA_PAPER_STREAM_URL = "wss://paper-api.alpaca.markets/stream"
STREAM_NAME = PHASE4_STREAM_NAME
# Events whose order snapshots can be represented by the append-only OMS.
# Alpaca also documents trade_bust/trade_correct; those require reversal
# accounting that this long-only ledger deliberately doesn't implement, so
# they fail closed into REST reconciliation rather than being misapplied.
ALPACA_TRADE_UPDATE_EVENTS = frozenset({
    "accepted",
    "calculated",
    "canceled",
    "done_for_day",
    "expired",
    "fill",
    "held",
    "new",
    "order_cancel_rejected",
    "order_replace_rejected",
    "partial_fill",
    "pending_cancel",
    "pending_new",
    "pending_replace",
    "rejected",
    "replaced",
    "stopped",
    "suspended",
})


@dataclass(frozen=True)
class OrderStreamEvent:
    event_id: str
    event_type: str
    broker_order: BrokerOrder
    fills: tuple[Fill, ...] = ()
    sequence: int | None = None

    def payload(self) -> dict:
        return json_safe({
            "event_id": self.event_id,
            "event_type": self.event_type,
            "sequence": self.sequence,
            "broker_order": self.broker_order,
            "fills": self.fills,
        })


class StreamEventSource(Protocol):
    def events(self) -> Iterable[OrderStreamEvent]: ...


class OrderStreamSupervisor:
    def __init__(
        self,
        ledger: Ledger,
        broker: Broker,
        *,
        alerts: AlertManager | None = None,
        clock=None,
    ) -> None:
        self.ledger = ledger
        self.store = Phase4Store(ledger)
        self.broker = broker
        self.alerts = alerts
        self.clock = clock or ledger.clock

    def connected(self) -> dict:
        return self.store.set_stream_state(
            STREAM_NAME, connected=True, recovering=True
        )

    def heartbeat(self) -> dict:
        state = self.store.stream_state(STREAM_NAME)
        if state is None or not state["connected"] or state["recovering"]:
            raise Phase4Error("Cannot renew an unhealthy order-stream lease")
        account = self.broker.get_account()
        latest = self.ledger.latest_reconciliation(account.account_id)
        due = latest is None
        if latest is not None:
            completed = ensure_aware(
                datetime.fromisoformat(latest["completed_at"]),
                "continuous reconciliation time",
            )
            age = (ensure_aware(self.clock(), "heartbeat time") - completed).total_seconds()
            due = age < -2 or age >= CONTINUOUS_RECONCILE_SECONDS
        if due:
            report = Reconciler(self.ledger, self.broker, clock=self.clock).reconcile()
            if not report.clean:
                self._reconciliation_alerts(report)
                self._disarm("continuous stream reconciliation found a mismatch")
                self.store.set_stream_state(STREAM_NAME, connected=True, recovering=True)
                raise Phase4Error("Continuous order-stream reconciliation is not clean")
        return self.store.set_stream_state(
            STREAM_NAME, connected=True, recovering=False
        )

    def disconnected(self, reason: str) -> dict:
        row = self.store.set_stream_state(
            STREAM_NAME, connected=False, recovering=True, disconnected=True
        )
        self._disarm("order stream disconnected; REST recovery required")
        self._emit_alert(
            "critical", "broker_disconnection", reason[:500],
            entity_id=STREAM_NAME, dedupe_key="broker-stream-disconnected",
        )
        return row

    def process(self, event: OrderStreamEvent) -> str:
        state = self.store.stream_state(STREAM_NAME)
        last_sequence = state["last_sequence"] if state else None
        out_of_order = (
            event.sequence is not None and last_sequence is not None
            and event.sequence <= last_sequence
        )
        local = next((
            row for row in self.ledger.list_orders()
            if row["client_order_id"] == event.broker_order.client_order_id
        ), None)
        if local is not None and event.broker_order.updated_at < datetime.fromisoformat(
            local["updated_at"]
        ):
            out_of_order = True
        disposition = "out_of_order" if out_of_order else (
            "unknown_order" if local is None else "pending"
        )
        created = self.store.record_stream_event(
            event_id=event.event_id,
            sequence=event.sequence,
            client_order_id=event.broker_order.client_order_id,
            event_type=event.event_type,
            broker_updated_at=event.broker_order.updated_at.isoformat(),
            payload=event.payload(),
            disposition=disposition,
            recovery_stream_name=STREAM_NAME,
        )
        if not created:
            recorded = self.store.stream_event(event.event_id)
            if recorded is not None and recorded["disposition"] == "pending":
                self._recovery_alert("incomplete_stream_event", event.event_id)
                self._disarm("an incompletely applied stream event requires REST recovery")
                self.store.set_stream_state(STREAM_NAME, connected=True, recovering=True)
                raise Phase4Error("Stream event was persisted but not fully applied")
            return "duplicate"
        if out_of_order:
            self._recovery_alert("out_of_order_stream_event", event.event_id)
            self._disarm("out-of-order stream event requires REST recovery")
            self.store.set_stream_state(STREAM_NAME, connected=True, recovering=True)
            return disposition
        if local is None:
            self._recovery_alert("external_or_missing_broker_order", event.event_id)
            self._disarm("stream reported an order absent from the local ledger")
            self.store.set_stream_state(STREAM_NAME, connected=True, recovering=True)
            return disposition
        try:
            self.ledger.acknowledge_order(local["order_id"], event.broker_order)
            for fill in event.fills:
                self.ledger.record_fill(local["order_id"], fill)
            self._synchronize_intent(local, event.broker_order)
        except LedgerConflict:
            self._recovery_alert("stream_state_conflict", event.event_id)
            self._disarm("stream event conflicted with durable order state")
            self.store.set_stream_state(STREAM_NAME, connected=True, recovering=True)
            raise
        report = Reconciler(self.ledger, self.broker, clock=self.clock).reconcile()
        if not report.clean:
            self._reconciliation_alerts(report)
            self._disarm("stream event produced a reconciliation mismatch")
            self.store.set_stream_state(STREAM_NAME, connected=True, recovering=True)
            raise Phase4Error("Stream event reconciliation is not clean")
        self.store.mark_stream_event_disposition(event.event_id, "applied")
        self.store.set_stream_state(
            STREAM_NAME,
            connected=True,
            recovering=bool(state["recovering"]) if state else True,
            last_sequence=event.sequence,
            event_at=event.broker_order.updated_at.isoformat(),
        )
        if event.broker_order.status == OrderStatus.REJECTED:
            self._emit_alert(
                "critical", "order_rejection",
                event.broker_order.rejection_reason or "Paper order rejected",
                entity_id=local["order_id"], dedupe_key=f"order-rejected:{local['order_id']}",
            )
        return "applied" if disposition == "pending" else disposition

    def recover(self, reason: str = "stream reconnect REST recovery") -> ReconciliationReport:
        self._disarm(reason)
        self.store.set_stream_state(STREAM_NAME, connected=True, recovering=True)
        try:
            account = self.broker.get_account()
            self.ledger.assert_account_binding(account.account_id)
            # Required recovery order: account above, then positions/open
            # orders, then bounded recent all-status orders, local client IDs,
            # and fills through OMS/reconciler. Do not prefetch all lifetime
            # orders here: Alpaca caps that endpoint at 500 and the reconciler
            # owns the overlap-safe submission-time watermark. The fetched
            # snapshots are deliberately not trusted in memory; reconciliation
            # fetches again and persists the result.
            self.broker.get_positions()
            self.broker.get_open_orders()
            pre_recovery = Reconciler(
                self.ledger, self.broker, clock=self.clock
            ).reconcile()
            if not pre_recovery.clean:
                self._reconciliation_alerts(pre_recovery)
            OrderManagementSystem(
                self.ledger, self.broker, PreTradeRiskEngine(), clock=self.clock
            ).recover_pending()
            report = Reconciler(self.ledger, self.broker, clock=self.clock).reconcile()
            if not report.clean:
                self._reconciliation_alerts(report)
                raise Phase4Error("REST stream recovery reconciliation is not clean")
            for event in self.store.pending_stream_events():
                self.store.mark_stream_event_disposition(event["event_id"], "recovered")
            self.store.set_stream_state(
                STREAM_NAME, connected=True, recovering=False, recovery_completed=True
            )
            if self.alerts:
                self.alerts.emit(
                    "info", "stream_recovery", "Order stream REST recovery completed",
                    entity_id=report.run_id, dedupe_key=f"stream-recovery:{report.run_id}",
                )
            return report
        except Exception as exc:
            self.store.set_stream_state(STREAM_NAME, connected=False, recovering=True)
            self._recovery_alert("stream_recovery_failed", type(exc).__name__)
            raise

    def supervise(
        self,
        source_factory,
        *,
        max_reconnects: int = 8,
        sleep=time.sleep,
    ) -> None:
        if max_reconnects < 0:
            raise ValueError("max_reconnects cannot be negative")
        attempt = 0
        while True:
            try:
                source = source_factory()
                if getattr(source, "supports_ready_callback", False):
                    ready = False

                    def mark_ready() -> None:
                        nonlocal ready
                        self.connected()
                        self.recover("authenticated stream connection established")
                        ready = True

                    events = source.events(
                        on_ready=mark_ready, on_heartbeat=self.heartbeat
                    )
                else:
                    self.connected()
                    self.recover("stream connection established")
                    ready = True
                    events = source.events()
                for event in events:
                    if not ready:
                        raise Phase4Error("Order stream emitted before authenticated readiness")
                    self.process(event)
                raise Phase4Error("Paper order stream ended")
            except KeyboardInterrupt:
                self.disconnected("operator stopped order stream")
                raise
            except Exception as exc:
                self.disconnected(f"{type(exc).__name__}: stream unavailable")
                if attempt >= max_reconnects:
                    raise Phase4Error("Paper order stream exhausted bounded reconnect attempts") from exc
                delay = min(30.0, 0.5 * (2 ** attempt))
                attempt += 1
                sleep(delay)

    def _synchronize_intent(self, local: dict, order: BrokerOrder) -> None:
        filled = self.ledger.filled_quantity_for_order(local["order_id"])
        if order.status == OrderStatus.FILLED and filled == Decimal(local["quantity"]):
            self.ledger.set_intent_status(
                local["intent_id"], IntentStatus.FILLED, "stream observed complete fill"
            )
        elif order.status == OrderStatus.CANCELED:
            self.ledger.set_intent_status(
                local["intent_id"], IntentStatus.CANCELED, "stream observed cancellation"
            )
        elif order.status == OrderStatus.REJECTED:
            self.ledger.set_intent_status(
                local["intent_id"], IntentStatus.BROKER_REJECTED, order.rejection_reason
            )

    def _disarm(self, reason: str) -> None:
        for account_id in self.ledger.known_account_ids():
            control = self.ledger.get_control_state(account_id)
            self.ledger.set_control_state(
                account_id, armed=False, kill_switch=control["kill_switch"], reason=reason
            )

    def _recovery_alert(self, category: str, entity_id: str) -> None:
        self._emit_alert(
            "critical", category, "New submissions blocked pending explicit reconciliation",
            entity_id=entity_id, dedupe_key=f"{category}:{entity_id}",
        )

    def _reconciliation_alerts(self, report: ReconciliationReport) -> None:
        categories = {
            "CASH_MISMATCH": "cash_mismatch",
            "POSITION_MISMATCH": "unexpected_position",
            "EXTERNAL_OPEN_ORDER": "externally_created_order",
            "EXTERNAL_RECENT_ORDER": "externally_created_order",
            "UNTRACKED_SYSTEM_ORDER": "duplicate_or_untracked_intent",
            "DUPLICATE_BROKER_OPEN_ORDER": "duplicate_intent",
            "DUPLICATE_BROKER_RECENT_ORDER": "duplicate_intent",
        }
        for issue in report.issues:
            category = categories.get(issue.code, "reconciliation_mismatch")
            entity_id = issue.entity_id or report.run_id
            self._emit_alert(
                "critical", category, issue.message,
                entity_id=entity_id,
                dedupe_key=f"stream-reconcile:{issue.code}:{entity_id}",
            )

    def _emit_alert(
        self,
        severity: str,
        category: str,
        message: str,
        *,
        entity_id: str = "",
        dedupe_key: str | None = None,
    ) -> dict:
        if self.alerts is not None:
            return self.alerts.emit(
                severity, category, message,
                entity_id=entity_id, dedupe_key=dedupe_key,
            )
        row, _created = self.store.emit_alert(
            severity, category, message,
            entity_id=entity_id, dedupe_key=dedupe_key,
        )
        return row


class AlpacaPaperTradeUpdateStream:
    """Hard-pinned synchronous Alpaca-paper websocket transport.

    The websocket library is imported only when a stream is actually started,
    keeping all offline Phase-4 tests network-free.
    """

    supports_ready_callback = True

    def __init__(self, config: AlpacaPaperConfig, account_id: str) -> None:
        if config.base_url != "https://paper-api.alpaca.markets":
            raise ValueError("Order stream is hard-pinned to Alpaca paper")
        self.config = config
        self.account_id = account_id

    def events(
        self,
        *,
        on_ready: Callable[[], None] | None = None,
        on_heartbeat: Callable[[], None] | None = None,
    ) -> Iterable[OrderStreamEvent]:
        try:
            from websockets.sync.client import connect
        except ImportError as exc:  # pragma: no cover - environment dependent.
            raise Phase4Error("websockets package is required for Alpaca paper streaming") from exc
        try:
            with connect(
                ALPACA_PAPER_STREAM_URL,
                open_timeout=self.config.timeout_seconds,
                close_timeout=self.config.timeout_seconds,
                proxy=None,
            ) as socket:
                socket.send(json.dumps({
                    "action": "auth",
                    "key": self.config.api_key,
                    "secret": self.config.api_secret,
                }))
                auth = json.loads(socket.recv(timeout=self.config.timeout_seconds))
                if not isinstance(auth, dict) or auth.get("data", {}).get("status") != "authorized":
                    raise Phase4Error("Alpaca paper stream authentication failed")
                socket.send(json.dumps({"action": "listen", "data": {"streams": ["trade_updates"]}}))
                listening = json.loads(socket.recv(timeout=self.config.timeout_seconds))
                if (
                    not isinstance(listening, dict)
                    or listening.get("stream") != "listening"
                    or "trade_updates" not in listening.get("data", {}).get("streams", [])
                ):
                    raise Phase4Error("Alpaca paper trade_updates subscription failed")
                if on_ready is not None:
                    on_ready()
                while True:
                    try:
                        raw = socket.recv(timeout=15.0)
                    except TimeoutError:
                        if on_heartbeat is not None:
                            on_heartbeat()
                        continue
                    yield self._parse(raw)
        except Phase4Error:
            raise
        except Exception as exc:
            # Never surface websocket frames or credentials in transport errors.
            raise Phase4Error(f"Alpaca paper stream transport failed: {type(exc).__name__}") from exc

    def _parse(self, raw: str | bytes) -> OrderStreamEvent:
        try:
            envelope = json.loads(raw)
            if not isinstance(envelope, dict) or envelope.get("stream") != "trade_updates":
                raise ValueError("unexpected stream envelope")
            data_row = envelope["data"]
            if not isinstance(data_row, dict):
                raise TypeError("trade-update data must be an object")
            event_type = data_row["event"]
            if not isinstance(event_type, str) or event_type not in ALPACA_TRADE_UPDATE_EVENTS:
                raise ValueError("unsupported trade-update event")
            row = data_row["order"]
            if not isinstance(row, dict) or row.get("asset_class", "us_equity") != "us_equity":
                raise ValueError("trade-update order must be a US equity")
            status = ALPACA_ORDER_STATUS[str(row["status"]).lower()]
            submitted = ensure_aware(
                datetime.fromisoformat(str(row["submitted_at"]).replace("Z", "+00:00")),
                "submitted_at",
            )
            updated = ensure_aware(
                datetime.fromisoformat(
                    str(row.get("updated_at") or data_row.get("timestamp")).replace("Z", "+00:00")
                ),
                "updated_at",
            )
            order = BrokerOrder(
                broker_order_id=str(row["id"]),
                client_order_id=str(row["client_order_id"]),
                account_id=self.account_id,
                symbol=str(row["symbol"]),
                side=Side(str(row["side"]).lower()),
                quantity=as_decimal(row["qty"]),
                filled_quantity=as_decimal(row.get("filled_qty", "0")),
                status=status,
                submitted_at=submitted,
                updated_at=updated,
                order_type=OrderType(str(row["type"]).lower()),
                time_in_force=TimeInForce(str(row["time_in_force"]).lower()),
                limit_price=(
                    as_decimal(row["limit_price"]) if row.get("limit_price") is not None else None
                ),
                rejection_reason=str(row.get("reject_reason") or "")[:500],
            )
            fills: tuple[Fill, ...] = ()
            if event_type in {"fill", "partial_fill"}:
                fill_id = str(data_row.get("execution_id") or "")
                if not fill_id:
                    fill_id = "stream-fill-" + hashlib.sha256(canonical_bytes(data_row)).hexdigest()[:24]
                fills = (Fill(
                    fill_id=fill_id,
                    broker_order_id=order.broker_order_id,
                    client_order_id=order.client_order_id,
                    account_id=self.account_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=as_decimal(data_row["qty"]),
                    price=as_decimal(data_row["price"]),
                    commission=ZERO,
                    occurred_at=updated,
                ),)
            event_id = str(data_row.get("event_id") or "")
            if not event_id:
                event_id = "stream-event-" + hashlib.sha256(canonical_bytes(data_row)).hexdigest()[:24]
            sequence = data_row.get("sequence")
            if sequence is not None and (isinstance(sequence, bool) or not isinstance(sequence, int)):
                raise ValueError("stream sequence must be integer")
            return OrderStreamEvent(event_id, event_type, order, fills, sequence)
        except (KeyError, TypeError, ValueError) as exc:
            raise Phase4Error("Malformed Alpaca paper trade-update event") from exc
