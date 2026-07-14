"""Broker-neutral value objects for the offline execution foundation.

The research engine emits strategy-specific ``TargetEvent`` objects.  Live
execution deliberately does not consume those objects directly: an upstream
portfolio allocator must first aggregate every sleeve into one account-level
``TargetPositionIntent`` per symbol.  That boundary prevents two strategies
from independently fighting over the same brokerage position.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any


ZERO = Decimal("0")


def as_decimal(value: Decimal | str | int | float) -> Decimal:
    """Convert external numeric input without inheriting binary-float noise."""
    result = value if isinstance(value, Decimal) else Decimal(str(value))
    if not result.is_finite():
        raise ValueError("numeric values must be finite")
    return result


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    OPG = "opg"
    CLS = "cls"
    IOC = "ioc"
    FOK = "fok"


class OrderStatus(str, Enum):
    PENDING_SUBMIT = "pending_submit"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


class IntentStatus(str, Enum):
    CREATED = "created"
    NOOP = "noop"
    RISK_REJECTED = "risk_rejected"
    ORDER_PENDING = "order_pending"
    ORDER_SUBMITTED = "order_submitted"
    FILLED = "filled"
    CANCELED = "canceled"
    BROKER_REJECTED = "broker_rejected"


TERMINAL_INTENT_STATUSES = {
    IntentStatus.NOOP,
    IntentStatus.RISK_REJECTED,
    IntentStatus.FILLED,
    IntentStatus.CANCELED,
    IntentStatus.BROKER_REJECTED,
}


ACTIVE_ORDER_STATUSES = {
    OrderStatus.PENDING_SUBMIT,
    OrderStatus.SUBMITTED,
    OrderStatus.PARTIALLY_FILLED,
}


@dataclass(frozen=True)
class TargetPositionIntent:
    """Immutable desired *account-level* position for one symbol.

    ``target_version`` must change when the producer intentionally revises a
    target for the same signal timestamp.  Reusing a version with different
    content is rejected by the ledger as an idempotency conflict.
    """

    account_id: str
    strategy_id: str
    symbol: str
    target_quantity: Decimal
    signal_at: datetime
    target_version: str
    reference_price: Decimal
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "account_id", self.account_id.strip())
        object.__setattr__(self, "strategy_id", self.strategy_id.strip())
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "target_quantity", as_decimal(self.target_quantity))
        object.__setattr__(self, "reference_price", as_decimal(self.reference_price))
        object.__setattr__(self, "signal_at", ensure_aware(self.signal_at, "signal_at"))
        object.__setattr__(self, "target_version", self.target_version.strip())
        if not self.account_id or not self.strategy_id or not self.symbol or not self.target_version:
            raise ValueError("account_id, strategy_id, symbol, and target_version are required")
        if self.target_quantity < ZERO:
            raise ValueError("target_quantity cannot be negative; phase one never opens shorts")
        if self.reference_price <= ZERO:
            raise ValueError("reference_price must be positive")

    @property
    def idempotency_key(self) -> str:
        canonical = {
            "account_id": self.account_id,
            "signal_at": self.signal_at.isoformat(),
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "target_version": self.target_version,
        }
        payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    as_of: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "bid", as_decimal(self.bid))
        object.__setattr__(self, "ask", as_decimal(self.ask))
        object.__setattr__(self, "last", as_decimal(self.last))
        object.__setattr__(self, "as_of", ensure_aware(self.as_of, "quote.as_of"))
        if not self.symbol or min(self.bid, self.ask, self.last) <= ZERO:
            raise ValueError("quote requires a symbol and positive bid/ask/last")
        if self.ask < self.bid:
            raise ValueError("quote ask cannot be below bid")

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / Decimal("2")


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: Decimal
    avg_price: Decimal
    market_price: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "quantity", as_decimal(self.quantity))
        object.__setattr__(self, "avg_price", as_decimal(self.avg_price))
        object.__setattr__(self, "market_price", as_decimal(self.market_price))
        if not self.symbol:
            raise ValueError("position symbol is required")
        if self.avg_price < ZERO or self.market_price < ZERO:
            raise ValueError("position prices cannot be negative")

    @property
    def market_value(self) -> Decimal:
        return self.quantity * self.market_price


@dataclass(frozen=True)
class AccountSnapshot:
    account_id: str
    cash: Decimal
    equity: Decimal
    buying_power: Decimal
    as_of: datetime
    status: str = "ACTIVE"
    currency: str = "USD"
    trading_blocked: bool = False
    account_blocked: bool = False
    trade_suspended_by_user: bool = False
    last_equity: Decimal | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "account_id", self.account_id.strip())
        object.__setattr__(self, "cash", as_decimal(self.cash))
        object.__setattr__(self, "equity", as_decimal(self.equity))
        object.__setattr__(self, "buying_power", as_decimal(self.buying_power))
        if self.last_equity is not None:
            object.__setattr__(self, "last_equity", as_decimal(self.last_equity))
        object.__setattr__(self, "as_of", ensure_aware(self.as_of, "account.as_of"))
        object.__setattr__(self, "status", self.status.strip().upper())
        object.__setattr__(self, "currency", self.currency.strip().upper())
        if not self.account_id:
            raise ValueError("account_id is required")
        if not self.status or not self.currency:
            raise ValueError("account status and currency are required")
        if not all(isinstance(value, bool) for value in (
            self.trading_blocked, self.account_blocked, self.trade_suspended_by_user
        )):
            raise ValueError("account block and suspension fields must be booleans")
        if self.last_equity is not None and self.last_equity <= ZERO:
            raise ValueError("last_equity must be positive when provided")


@dataclass(frozen=True)
class OrderRequest:
    account_id: str
    client_order_id: str
    intent_id: str
    symbol: str
    side: Side
    quantity: Decimal
    reference_price: Decimal
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: Decimal | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "account_id", self.account_id.strip())
        object.__setattr__(self, "client_order_id", self.client_order_id.strip())
        object.__setattr__(self, "intent_id", self.intent_id.strip())
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "side", Side(self.side))
        object.__setattr__(self, "order_type", OrderType(self.order_type))
        object.__setattr__(self, "time_in_force", TimeInForce(self.time_in_force))
        object.__setattr__(self, "quantity", as_decimal(self.quantity))
        object.__setattr__(self, "reference_price", as_decimal(self.reference_price))
        if self.limit_price is not None:
            object.__setattr__(self, "limit_price", as_decimal(self.limit_price))
        if (
            not self.account_id or not self.client_order_id or not self.intent_id
            or not self.symbol or self.quantity <= ZERO
        ):
            raise ValueError("order request requires IDs and a positive quantity")
        if self.reference_price <= ZERO:
            raise ValueError("reference_price must be positive")
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit orders require limit_price")
        if self.order_type != OrderType.LIMIT and self.limit_price is not None:
            raise ValueError("limit_price is valid only for limit orders")
        if self.limit_price is not None and self.limit_price <= ZERO:
            raise ValueError("limit_price must be positive")


@dataclass(frozen=True)
class BrokerOrder:
    broker_order_id: str
    client_order_id: str
    account_id: str
    symbol: str
    side: Side
    quantity: Decimal
    filled_quantity: Decimal
    status: OrderStatus
    submitted_at: datetime
    updated_at: datetime
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY
    rejection_reason: str = ""
    limit_price: Decimal | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "broker_order_id", self.broker_order_id.strip())
        object.__setattr__(self, "client_order_id", self.client_order_id.strip())
        object.__setattr__(self, "account_id", self.account_id.strip())
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "side", Side(self.side))
        object.__setattr__(self, "status", OrderStatus(self.status))
        object.__setattr__(self, "order_type", OrderType(self.order_type))
        object.__setattr__(self, "time_in_force", TimeInForce(self.time_in_force))
        object.__setattr__(self, "quantity", as_decimal(self.quantity))
        object.__setattr__(self, "filled_quantity", as_decimal(self.filled_quantity))
        if self.limit_price is not None:
            object.__setattr__(self, "limit_price", as_decimal(self.limit_price))
        object.__setattr__(self, "submitted_at", ensure_aware(self.submitted_at, "submitted_at"))
        object.__setattr__(self, "updated_at", ensure_aware(self.updated_at, "updated_at"))
        if not self.broker_order_id or not self.client_order_id or not self.account_id or not self.symbol:
            raise ValueError("broker order requires IDs, account, and symbol")
        if self.quantity <= ZERO or self.filled_quantity < ZERO:
            raise ValueError("broker order quantities are invalid")
        if self.filled_quantity > self.quantity:
            raise ValueError("filled quantity cannot exceed ordered quantity")
        if self.updated_at < self.submitted_at:
            raise ValueError("broker order updated_at cannot precede submitted_at")
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("broker limit orders require limit_price")
        if self.order_type != OrderType.LIMIT and self.limit_price is not None:
            raise ValueError("broker limit_price is valid only for limit orders")
        if self.limit_price is not None and self.limit_price <= ZERO:
            raise ValueError("broker limit_price must be positive")


@dataclass(frozen=True)
class Fill:
    fill_id: str
    broker_order_id: str
    client_order_id: str
    account_id: str
    symbol: str
    side: Side
    quantity: Decimal
    price: Decimal
    commission: Decimal
    occurred_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "fill_id", self.fill_id.strip())
        object.__setattr__(self, "broker_order_id", self.broker_order_id.strip())
        object.__setattr__(self, "client_order_id", self.client_order_id.strip())
        object.__setattr__(self, "account_id", self.account_id.strip())
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "side", Side(self.side))
        object.__setattr__(self, "quantity", as_decimal(self.quantity))
        object.__setattr__(self, "price", as_decimal(self.price))
        object.__setattr__(self, "commission", as_decimal(self.commission))
        object.__setattr__(self, "occurred_at", ensure_aware(self.occurred_at, "occurred_at"))
        if (
            not self.fill_id or not self.broker_order_id or not self.client_order_id
            or not self.account_id or not self.symbol or self.quantity <= ZERO or self.price <= ZERO
        ):
            raise ValueError("fill requires an ID, positive quantity, and positive price")
        if self.commission < ZERO:
            raise ValueError("fill commission cannot be negative")


@dataclass(frozen=True)
class RiskViolation:
    code: str
    message: str


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    violations: tuple[RiskViolation, ...]
    order_notional: Decimal
    projected_gross_exposure: Decimal


@dataclass(frozen=True)
class OMSResult:
    intent_id: str
    intent_status: IntentStatus
    duplicate_intent: bool
    order_id: str | None = None
    client_order_id: str | None = None
    broker_order_id: str | None = None
    risk_violations: tuple[RiskViolation, ...] = field(default_factory=tuple)


def json_safe(value: Any) -> Any:
    """Recursively normalize audit payloads for deterministic JSON storage."""
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return ensure_aware(value, "datetime").isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value
