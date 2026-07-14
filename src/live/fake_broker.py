"""Deterministic, in-process broker used by phase-one tests and demos only."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from decimal import Decimal

from .broker import Broker, BrokerError
from .models import (
    ACTIVE_ORDER_STATUSES,
    AccountSnapshot,
    BrokerOrder,
    Fill,
    OrderRequest,
    OrderStatus,
    Position,
    Quote,
    Side,
    ZERO,
    as_decimal,
    utc_now,
)


class FakeBroker(Broker):
    """A long-only cash broker with idempotent client order IDs.

    ``raise_after_submit_once`` simulates the most dangerous common failure:
    the broker accepts (and may fill) an order but the caller loses the
    acknowledgement.  The OMS must recover by looking up the same client ID,
    never by inventing a new order.
    """

    def __init__(
        self,
        account_id: str = "FAKE-ACCOUNT",
        cash: Decimal | str | int | float = "100000",
        *,
        auto_fill: bool = True,
        clock=utc_now,
    ) -> None:
        self.account_id = account_id
        self._cash = as_decimal(cash)
        self.auto_fill = auto_fill
        self.clock = clock
        self._quotes: dict[str, Quote] = {}
        self._positions: dict[str, Position] = {}
        self._orders_by_client: dict[str, BrokerOrder] = {}
        self._orders_by_broker: dict[str, BrokerOrder] = {}
        self._fills: list[Fill] = []
        self._next_order = 1
        self._next_fill = 1
        self.raise_after_submit_once = False

    @property
    def submission_count(self) -> int:
        return len(self._orders_by_client)

    def set_quote(
        self,
        symbol: str,
        price: Decimal | str | int | float,
        *,
        spread_bps: Decimal | str | int | float = "2",
        as_of: datetime | None = None,
    ) -> Quote:
        symbol = symbol.upper()
        mid = as_decimal(price)
        half = as_decimal(spread_bps) / Decimal("20000")
        quote = Quote(
            symbol=symbol,
            bid=mid * (Decimal("1") - half),
            ask=mid * (Decimal("1") + half),
            last=mid,
            as_of=as_of or self.clock(),
        )
        self._quotes[symbol] = quote
        self._mark_positions()
        return quote

    def get_quote(self, symbol: str) -> Quote:
        try:
            return self._quotes[symbol.upper()]
        except KeyError as exc:
            raise BrokerError(f"No fake quote configured for {symbol}") from exc

    def seed_position(
        self,
        symbol: str,
        quantity: Decimal | str | int | float,
        avg_price: Decimal | str | int | float,
    ) -> None:
        symbol = symbol.upper()
        qty = as_decimal(quantity)
        avg = as_decimal(avg_price)
        if qty < ZERO:
            raise BrokerError("FakeBroker is long-only")
        if qty == ZERO:
            self._positions.pop(symbol, None)
            return
        market = self._quotes[symbol].mid if symbol in self._quotes else avg
        self._positions[symbol] = Position(symbol, qty, avg, market)

    def get_account(self) -> AccountSnapshot:
        self._mark_positions()
        equity = self._cash + sum((p.market_value for p in self._positions.values()), ZERO)
        return AccountSnapshot(
            self.account_id, self._cash, equity, self._cash, self.clock(),
            last_equity=equity,
        )

    def get_positions(self) -> list[Position]:
        self._mark_positions()
        return [self._positions[s] for s in sorted(self._positions)]

    def get_open_orders(self) -> list[BrokerOrder]:
        return [o for o in self._orders_by_client.values() if o.status in ACTIVE_ORDER_STATUSES]

    def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        return self._orders_by_client.get(client_order_id)

    def submit_order(self, request: OrderRequest) -> BrokerOrder:
        if request.account_id != self.account_id:
            raise BrokerError("Order account does not match fake broker account")
        existing = self._orders_by_client.get(request.client_order_id)
        if existing is not None:
            if (
                existing.account_id != request.account_id
                or existing.symbol != request.symbol
                or existing.side != request.side
                or existing.quantity != request.quantity
                or existing.order_type != request.order_type
                or existing.time_in_force != request.time_in_force
                or existing.limit_price != request.limit_price
            ):
                raise BrokerError("Client order ID was reused with different order content")
            return existing
        if request.symbol not in self._quotes:
            raise BrokerError(f"No fake quote configured for {request.symbol}")

        now = self.clock()
        broker_id = f"fake-order-{self._next_order}"
        self._next_order += 1
        order = BrokerOrder(
            broker_order_id=broker_id,
            client_order_id=request.client_order_id,
            account_id=request.account_id,
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            filled_quantity=ZERO,
            status=OrderStatus.SUBMITTED,
            submitted_at=now,
            updated_at=now,
            order_type=request.order_type,
            time_in_force=request.time_in_force,
            limit_price=request.limit_price,
        )
        self._store_order(order)
        if self.auto_fill:
            order = self.fill_order(broker_id)
        if self.raise_after_submit_once:
            self.raise_after_submit_once = False
            raise BrokerError("simulated acknowledgement loss after broker acceptance")
        return order

    def fill_order(
        self,
        broker_order_id: str,
        quantity: Decimal | str | int | float | None = None,
    ) -> BrokerOrder:
        try:
            order = self._orders_by_broker[broker_order_id]
        except KeyError as exc:
            raise BrokerError(f"Unknown fake broker order: {broker_order_id}") from exc
        if order.status not in ACTIVE_ORDER_STATUSES:
            return order
        remaining = order.quantity - order.filled_quantity
        qty = remaining if quantity is None else as_decimal(quantity)
        if qty <= ZERO or qty > remaining:
            raise BrokerError("Fill quantity must be positive and no greater than remaining")

        quote = self._quotes[order.symbol]
        price = quote.ask if order.side == Side.BUY else quote.bid
        position = self._positions.get(order.symbol, Position(order.symbol, ZERO, ZERO, quote.mid))
        if order.side == Side.BUY:
            cost = qty * price
            if cost > self._cash:
                return self._reject(order, "insufficient cash")
            new_qty = position.quantity + qty
            new_avg = ((position.quantity * position.avg_price) + cost) / new_qty
            self._cash -= cost
        else:
            if qty > position.quantity:
                return self._reject(order, "short sales are disabled")
            new_qty = position.quantity - qty
            new_avg = position.avg_price if new_qty > ZERO else ZERO
            self._cash += qty * price

        if new_qty == ZERO:
            self._positions.pop(order.symbol, None)
        else:
            self._positions[order.symbol] = Position(order.symbol, new_qty, new_avg, quote.mid)

        now = self.clock()
        fill = Fill(
            fill_id=f"fake-fill-{self._next_fill}",
            broker_order_id=order.broker_order_id,
            client_order_id=order.client_order_id,
            account_id=order.account_id,
            symbol=order.symbol,
            side=order.side,
            quantity=qty,
            price=price,
            commission=ZERO,
            occurred_at=now,
        )
        self._next_fill += 1
        self._fills.append(fill)
        filled = order.filled_quantity + qty
        status = OrderStatus.FILLED if filled == order.quantity else OrderStatus.PARTIALLY_FILLED
        order = replace(order, filled_quantity=filled, status=status, updated_at=now)
        self._store_order(order)
        return order

    def cancel_order(self, broker_order_id: str) -> BrokerOrder:
        try:
            order = self._orders_by_broker[broker_order_id]
        except KeyError as exc:
            raise BrokerError(f"Unknown fake broker order: {broker_order_id}") from exc
        if order.status in ACTIVE_ORDER_STATUSES:
            order = replace(order, status=OrderStatus.CANCELED, updated_at=self.clock())
            self._store_order(order)
        return order

    def get_fills(self, since: datetime | None = None) -> list[Fill]:
        if since is None:
            return list(self._fills)
        return [fill for fill in self._fills if fill.occurred_at >= since]

    def _reject(self, order: BrokerOrder, reason: str) -> BrokerOrder:
        order = replace(
            order,
            status=OrderStatus.REJECTED,
            rejection_reason=reason,
            updated_at=self.clock(),
        )
        self._store_order(order)
        return order

    def _store_order(self, order: BrokerOrder) -> None:
        self._orders_by_client[order.client_order_id] = order
        self._orders_by_broker[order.broker_order_id] = order

    def _mark_positions(self) -> None:
        for symbol, position in list(self._positions.items()):
            quote = self._quotes.get(symbol)
            if quote is not None:
                self._positions[symbol] = Position(
                    symbol, position.quantity, position.avg_price, quote.mid
                )
