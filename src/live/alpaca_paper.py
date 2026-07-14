"""Paper-only Alpaca Trading API adapter.

This module deliberately pins the endpoint to Alpaca's paper host.  It has no
flag, environment variable, or constructor path that can target a live trading
endpoint.  It implements the narrow broker contract used by phase one; target
aggregation, market-data collection, and scheduling remain separate concerns.
"""

from __future__ import annotations

import math
import os
import ssl
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from decimal import Decimal
from typing import Any
from urllib.parse import quote as url_quote

import requests

from .broker import Broker, BrokerError
from .market_data import MarketCalendarDay, NEW_YORK
from .models import (
    AccountSnapshot,
    BrokerOrder,
    Fill,
    OrderRequest,
    OrderStatus,
    OrderType,
    Position,
    Side,
    TimeInForce,
    ZERO,
    as_decimal,
    ensure_aware,
    utc_now,
)


PAPER_BASE_URL = "https://paper-api.alpaca.markets"
_OPEN_STATUS_MAP = {
    "new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.SUBMITTED,
    "pending_new": OrderStatus.SUBMITTED,
    "accepted_for_bidding": OrderStatus.SUBMITTED,
    "pending_cancel": OrderStatus.SUBMITTED,
    "pending_replace": OrderStatus.SUBMITTED,
    "calculated": OrderStatus.SUBMITTED,
    "held": OrderStatus.SUBMITTED,
    "stopped": OrderStatus.SUBMITTED,
    # Alpaca documents done_for_day as able to receive updates on a later day.
    # Treating it as canceled would release the local active-order guard early.
    "done_for_day": OrderStatus.SUBMITTED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
    "expired": OrderStatus.CANCELED,
    "replaced": OrderStatus.CANCELED,
    "rejected": OrderStatus.REJECTED,
    "suspended": OrderStatus.REJECTED,
}
# Shared by the paper websocket parser; kept as an immutable-by-convention
# module mapping so REST and stream status semantics cannot drift.
ALPACA_ORDER_STATUS = _OPEN_STATUS_MAP


@dataclass(frozen=True)
class AlpacaMarketClock:
    timestamp: datetime
    is_open: bool
    next_open: datetime
    next_close: datetime


@dataclass(frozen=True)
class AlpacaAsset:
    asset_id: str
    symbol: str
    asset_class: str
    status: str
    tradable: bool
    exchange: str = ""
    fractionable: bool = False
    shortable: bool = False
    easy_to_borrow: bool = False
    marginable: bool = False
    name: str = ""


@dataclass(frozen=True)
class AlpacaPaperConfig:
    """Credentials for a paper account only; never write these to the ledger."""

    api_key: str = field(repr=False)
    api_secret: str = field(repr=False)
    base_url: str = PAPER_BASE_URL
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "api_key", self.api_key.strip())
        object.__setattr__(self, "api_secret", self.api_secret.strip())
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))
        if not self.api_key or not self.api_secret:
            raise ValueError("Paper API key and secret are required")
        if self.base_url != PAPER_BASE_URL:
            raise ValueError("AlpacaPaperBroker is hard-pinned to the paper endpoint")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")

    @classmethod
    def from_env(cls) -> "AlpacaPaperConfig":
        live_style = [
            name for name in (
                "ALPACA_API_KEY", "ALPACA_API_SECRET", "APCA_API_KEY_ID",
                "APCA_API_SECRET_KEY", "APCA_API_BASE_URL", "ALPACA_BASE_URL",
            )
            if os.getenv(name)
        ]
        if live_style:
            raise ValueError(
                "Live-style credential variables are forbidden for Phase 4; "
                "use only ALPACA_PAPER_API_KEY and ALPACA_PAPER_API_SECRET"
            )
        key = os.getenv("ALPACA_PAPER_API_KEY", "")
        secret = os.getenv("ALPACA_PAPER_API_SECRET", "")
        if not key or not secret:
            raise ValueError(
                "Set ALPACA_PAPER_API_KEY and ALPACA_PAPER_API_SECRET; live credentials are not accepted"
            )
        return cls(api_key=key, api_secret=secret)


class AlpacaPaperBroker(Broker):
    """Synchronous adapter for Alpaca's individual paper Trading API.

    There are intentionally no automatic HTTP retries.  The OMS owns
    idempotency through client order IDs; retrying a POST below this layer would
    obscure whether the broker accepted an order.
    """

    def __init__(
        self,
        config: AlpacaPaperConfig,
        *,
        session: requests.Session | Any | None = None,
        clock=utc_now,
    ) -> None:
        self.config = config
        self._uses_default_session = session is None
        self.session = session if session is not None else requests.Session()
        if hasattr(self.session, "trust_env"):
            self.session.trust_env = False
        self.clock = clock
        self._order_cache: dict[str, BrokerOrder] = {}
        self._account_id: str | None = None
        self._request_ids: list[str] = []

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.config.api_key,
            "APCA-API-SECRET-KEY": self.config.api_secret,
            "Accept": "application/json",
            "User-Agent": "wallst-strategy-lab/phase2",
        }

    @property
    def request_ids(self) -> tuple[str, ...]:
        """Recent Alpaca request IDs, safe to persist for operational support."""
        return tuple(self._request_ids)

    def drain_request_ids(self) -> tuple[str, ...]:
        values = tuple(self._request_ids)
        self._request_ids.clear()
        return values

    def get_account(self) -> AccountSnapshot:
        payload = _require_object(self._json("GET", "/v2/account"), "account")
        try:
            account = AccountSnapshot(
                account_id=str(payload["id"]),
                cash=as_decimal(payload["cash"]),
                equity=as_decimal(payload["equity"]),
                buying_power=as_decimal(payload["buying_power"]),
                as_of=self.clock(),  # receipt timestamp; API account payload has no as-of field
                status=str(payload["status"]),
                currency=str(payload["currency"]),
                trading_blocked=_strict_bool(payload["trading_blocked"], "trading_blocked"),
                account_blocked=_strict_bool(payload["account_blocked"], "account_blocked"),
                trade_suspended_by_user=_strict_bool(
                    payload["trade_suspended_by_user"], "trade_suspended_by_user"
                ),
                last_equity=as_decimal(payload["last_equity"]),
            )
        except (ArithmeticError, KeyError, TypeError, ValueError) as exc:
            raise BrokerError("Malformed Alpaca paper account response") from exc
        if self._account_id is not None and self._account_id != account.account_id:
            raise BrokerError("Authenticated Alpaca paper account changed during the session")
        self._account_id = account.account_id
        return account

    def get_market_clock(self) -> AlpacaMarketClock:
        payload = _require_object(self._json("GET", "/v2/clock"), "market clock")
        try:
            return AlpacaMarketClock(
                timestamp=_parse_time(payload["timestamp"]),
                is_open=_strict_bool(payload["is_open"], "is_open"),
                next_open=_parse_time(payload["next_open"]),
                next_close=_parse_time(payload["next_close"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BrokerError("Malformed Alpaca paper market-clock response") from exc

    def get_market_calendar(
        self, start: date, end: date
    ) -> tuple[MarketCalendarDay, ...]:
        if (
            not isinstance(start, date) or isinstance(start, datetime)
            or not isinstance(end, date) or isinstance(end, datetime)
            or end < start
            or (end - start).days > 7
        ):
            raise BrokerError("Paper calendar request must be a valid range of at most 7 days")
        payload = self._json(
            "GET", "/v2/calendar",
            params={
                "start": start.isoformat(),
                "end": end.isoformat(),
                "date_type": "TRADING",
            },
        )
        if not isinstance(payload, list) or len(payload) > 8:
            raise BrokerError("Malformed Alpaca paper calendar response")
        result: list[MarketCalendarDay] = []
        try:
            for row in payload:
                if not isinstance(row, dict):
                    raise TypeError("calendar row must be an object")
                trading_date = date.fromisoformat(str(row["date"]))
                if trading_date < start or trading_date > end:
                    raise ValueError("calendar row is outside requested range")
                result.append(MarketCalendarDay(
                    trading_date=trading_date,
                    open_at=_calendar_time(trading_date, row["open"]),
                    close_at=_calendar_time(trading_date, row["close"]),
                ))
        except (KeyError, TypeError, ValueError) as exc:
            raise BrokerError("Malformed Alpaca paper calendar response") from exc
        dates = [row.trading_date for row in result]
        if dates != sorted(dates) or len(dates) != len(set(dates)):
            raise BrokerError("Alpaca paper calendar contains duplicate or unsorted dates")
        return tuple(result)

    def get_asset(self, symbol: str) -> AlpacaAsset:
        normalized = symbol.strip().upper()
        if not normalized:
            raise BrokerError("Asset symbol is required")
        payload = _require_object(
            self._json("GET", f"/v2/assets/{url_quote(normalized, safe='')}", allow_not_found=True),
            "asset",
        )
        try:
            return AlpacaAsset(
                asset_id=str(payload["id"]),
                symbol=str(payload["symbol"]).strip().upper(),
                asset_class=str(payload["class"]).strip().lower(),
                status=str(payload["status"]).strip().lower(),
                tradable=_strict_bool(payload["tradable"], "tradable"),
                exchange=str(payload.get("exchange", "")).strip().upper(),
                fractionable=_strict_bool(payload.get("fractionable", False), "fractionable"),
                shortable=_strict_bool(payload.get("shortable", False), "shortable"),
                easy_to_borrow=_strict_bool(
                    payload.get("easy_to_borrow", False), "easy_to_borrow"
                ),
                marginable=_strict_bool(payload.get("marginable", False), "marginable"),
                name=str(payload.get("name", "")).strip()[:200],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BrokerError("Malformed Alpaca paper asset response") from exc

    def get_positions(self) -> list[Position]:
        payload = self._json("GET", "/v2/positions")
        if not isinstance(payload, list):
            raise BrokerError("Alpaca paper positions response was not a list")
        positions: list[Position] = []
        for row in payload:
            try:
                if not isinstance(row, dict):
                    raise BrokerError("Malformed Alpaca paper position response")
                if row.get("asset_class", "us_equity") != "us_equity":
                    raise BrokerError("Phase two supports US-equity positions only")
                qty = as_decimal(row["qty"])
                if str(row.get("side", "long")).lower() == "short":
                    qty = -abs(qty)
                market_price = row.get("current_price")
                if market_price is None:
                    market_value = as_decimal(row.get("market_value", ZERO))
                    market_price = abs(market_value / qty) if qty != ZERO else row["avg_entry_price"]
                positions.append(Position(
                    symbol=str(row["symbol"]),
                    quantity=qty,
                    avg_price=as_decimal(row["avg_entry_price"]),
                    market_price=as_decimal(market_price),
                ))
            except BrokerError:
                raise
            except (ArithmeticError, KeyError, TypeError, ValueError) as exc:
                raise BrokerError("Malformed Alpaca paper position response") from exc
        symbols = [position.symbol for position in positions]
        if len(symbols) != len(set(symbols)):
            raise BrokerError("Alpaca paper returned duplicate position symbols")
        return positions

    def get_open_orders(self) -> list[BrokerOrder]:
        payload = self._json(
            "GET",
            "/v2/orders",
            params={"status": "open", "direction": "asc", "nested": "false", "limit": "500"},
        )
        if not isinstance(payload, list):
            raise BrokerError("Alpaca paper orders response was not a list")
        if len(payload) >= 500:
            raise BrokerError(
                "Alpaca paper returned the open-order page limit; reconcile with a paginated adapter before trading"
            )
        orders = [self._parse_order(row) for row in payload]
        broker_ids = [order.broker_order_id for order in orders]
        client_ids = [order.client_order_id for order in orders]
        if len(broker_ids) != len(set(broker_ids)) or len(client_ids) != len(set(client_ids)):
            raise BrokerError("Alpaca paper returned duplicate open-order identifiers")
        return orders

    def get_recent_orders(self, since: datetime | None = None) -> list[BrokerOrder]:
        params = {
            "status": "all", "direction": "asc", "nested": "false", "limit": "500",
        }
        if since is not None:
            params["after"] = _alpaca_time(since)
        payload = self._json("GET", "/v2/orders", params=params)
        if not isinstance(payload, list):
            raise BrokerError("Alpaca paper recent-orders response was not a list")
        if len(payload) >= 500:
            raise BrokerError(
                "Alpaca paper returned the all-status order page limit; fail-closed reconciliation requires a narrower watermark"
            )
        orders = [self._parse_order(row) for row in payload]
        broker_ids = [order.broker_order_id for order in orders]
        client_ids = [order.client_order_id for order in orders]
        if len(broker_ids) != len(set(broker_ids)) or len(client_ids) != len(set(client_ids)):
            raise BrokerError("Alpaca paper returned duplicate recent-order identifiers")
        return orders

    def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        payload = self._json(
            "GET",
            "/v2/orders:by_client_order_id",
            params={"client_order_id": client_order_id},
            allow_not_found=True,
        )
        return self._parse_order(payload) if payload is not None else None

    def submit_order(self, request: OrderRequest) -> BrokerOrder:
        account = self.get_account()
        if request.account_id != account.account_id:
            raise BrokerError("Order account does not match the authenticated paper account")
        if account.status != "ACTIVE":
            raise BrokerError(f"Alpaca paper account is not active: {account.status}")
        if account.currency != "USD":
            raise BrokerError("Phase two supports USD paper accounts only")
        if account.trading_blocked or account.account_blocked or account.trade_suspended_by_user:
            raise BrokerError("Alpaca paper account reports trading blocked or suspended")
        if len(request.client_order_id) > 128:
            raise BrokerError("Alpaca client_order_id exceeds 128 characters")
        if request.quantity != request.quantity.to_integral_value():
            raise BrokerError("Phase two Alpaca paper orders require whole shares")
        if request.order_type not in {OrderType.MARKET, OrderType.LIMIT}:
            raise BrokerError("Phase-two Alpaca paper adapter permits only market and limit orders")
        if request.order_type == OrderType.MARKET and request.time_in_force != TimeInForce.DAY:
            raise BrokerError("Phase-two market orders require day time-in-force")
        if request.order_type == OrderType.LIMIT and request.time_in_force not in {
            TimeInForce.DAY, TimeInForce.GTC
        }:
            raise BrokerError("Phase-two limit orders permit only day and GTC time-in-force")
        asset = self.get_asset(request.symbol)
        if asset.symbol != request.symbol or asset.asset_class != "us_equity":
            raise BrokerError("Order symbol did not resolve to the requested US equity")
        if asset.status != "active" or not asset.tradable:
            raise BrokerError("Requested Alpaca paper asset is inactive or not tradable")
        body = {
            "symbol": request.symbol,
            "qty": str(request.quantity),
            "side": request.side.value,
            "type": request.order_type.value,
            "time_in_force": request.time_in_force.value,
            "client_order_id": request.client_order_id,
            "extended_hours": False,
        }
        if request.limit_price is not None:
            body["limit_price"] = str(request.limit_price)
        return self._parse_order(self._json("POST", "/v2/orders", json_body=body))

    def cancel_order(self, broker_order_id: str) -> BrokerOrder:
        encoded_id = url_quote(broker_order_id.strip(), safe="")
        if not encoded_id:
            raise BrokerError("Broker order ID is required")
        response = self._request(
            "DELETE", f"/v2/orders/{encoded_id}", accepted_error_statuses={422}
        )
        order = self._get_order_by_id(broker_order_id)
        if response.status_code == 422 and order.status in {
            OrderStatus.PENDING_SUBMIT, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED
        }:
            raise BrokerError(
                f"Alpaca paper cancellation was rejected while order remained {order.status.value}"
            )
        return order

    def get_fills(self, since: datetime | None = None) -> list[Fill]:
        params: dict[str, str] = {"direction": "asc", "page_size": "100"}
        if since is not None:
            params["after"] = _alpaca_time(since)
        rows: list[dict] = []
        seen_tokens: set[str] = set()
        for _page in range(100):
            payload = self._json("GET", "/v2/account/activities/FILL", params=params)
            if not isinstance(payload, list):
                raise BrokerError("Alpaca paper fill-activity response was not a list")
            if any(not isinstance(row, dict) for row in payload):
                raise BrokerError("Malformed Alpaca paper fill-activity response")
            rows.extend(payload)
            if len(payload) < 100:
                break
            token = str(payload[-1].get("id", ""))
            if not token or token in seen_tokens:
                raise BrokerError("Alpaca paper fill pagination did not advance")
            seen_tokens.add(token)
            params["page_token"] = token
        else:
            raise BrokerError("Alpaca paper fill pagination exceeded the safety limit")

        fills: list[Fill] = []
        for row in rows:
            try:
                order_id = str(row["order_id"])
                order = self._order_cache.get(order_id) or self._get_order_by_id(order_id)
                fills.append(Fill(
                    fill_id=str(row["id"]),
                    broker_order_id=order_id,
                    client_order_id=order.client_order_id,
                    account_id=order.account_id,
                    symbol=str(row["symbol"]),
                    side=Side(str(row["side"]).lower()),
                    quantity=as_decimal(row["qty"]),
                    price=as_decimal(row["price"]),
                    commission=as_decimal(row.get("commission", "0")),
                    occurred_at=_parse_time(row["transaction_time"]),
                ))
            except BrokerError:
                raise
            except (ArithmeticError, KeyError, TypeError, ValueError) as exc:
                raise BrokerError("Malformed Alpaca paper fill-activity response") from exc
        return sorted(fills, key=lambda fill: (fill.occurred_at, fill.fill_id))

    def _get_order_by_id(self, broker_order_id: str) -> BrokerOrder:
        encoded_id = url_quote(broker_order_id.strip(), safe="")
        if not encoded_id:
            raise BrokerError("Broker order ID is required")
        return self._parse_order(self._json("GET", f"/v2/orders/{encoded_id}"))

    def _parse_order(self, row: dict[str, Any]) -> BrokerOrder:
        if not isinstance(row, dict):
            raise BrokerError("Malformed Alpaca paper order response")
        if row.get("asset_class", "us_equity") != "us_equity":
            raise BrokerError("Phase two supports US-equity orders only")
        try:
            raw_status = str(row["status"]).lower()
            status = _OPEN_STATUS_MAP[raw_status]
        except KeyError as exc:
            raise BrokerError(f"Unsupported Alpaca paper order status: {row.get('status')!r}") from exc
        try:
            filled_quantity = as_decimal(row.get("filled_qty") or "0")
            if status == OrderStatus.SUBMITTED and filled_quantity > ZERO:
                status = OrderStatus.PARTIALLY_FILLED
            order = BrokerOrder(
                broker_order_id=str(row["id"]),
                client_order_id=str(row["client_order_id"]),
                account_id=str(row.get("account_id") or self._account_id or self.get_account().account_id),
                symbol=str(row["symbol"]),
                side=Side(str(row["side"]).lower()),
                quantity=as_decimal(row["qty"]),
                filled_quantity=filled_quantity,
                status=status,
                submitted_at=_parse_time(row.get("submitted_at") or row.get("created_at")),
                updated_at=_parse_time(row.get("updated_at") or row.get("submitted_at") or row.get("created_at")),
                order_type=OrderType(str(row.get("type") or row.get("order_type") or "market").lower()),
                time_in_force=TimeInForce(str(row.get("time_in_force") or "day").lower()),
                rejection_reason=str(row.get("reject_reason") or ""),
                limit_price=(
                    as_decimal(row["limit_price"])
                    if row.get("limit_price") is not None else None
                ),
            )
        except (ArithmeticError, KeyError, TypeError, ValueError) as exc:
            raise BrokerError("Malformed Alpaca paper order response") from exc
        cached = self._order_cache.get(order.broker_order_id)
        if cached is not None and order.updated_at < cached.updated_at:
            raise BrokerError("Alpaca paper returned stale broker-order state")
        self._order_cache[order.broker_order_id] = order
        return order

    def _json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        json_body: dict[str, str] | None = None,
        allow_not_found: bool = False,
        accepted_error_statuses: set[int] | None = None,
    ) -> Any | None:
        response = self._request(
            method,
            path,
            params=params,
            json_body=json_body,
            allow_not_found=allow_not_found,
            accepted_error_statuses=accepted_error_statuses,
        )
        if response is None:
            return None
        try:
            return response.json()
        except (TypeError, ValueError) as exc:
            raise BrokerError(f"Alpaca paper {method} {path} returned invalid JSON") from exc

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        json_body: dict[str, str] | None = None,
        allow_not_found: bool = False,
        accepted_error_statuses: set[int] | None = None,
    ) -> Any | None:
        if not path.startswith("/"):
            raise ValueError("API path must begin with '/'")
        tls_problem = _tls_runtime_problem()
        if self._uses_default_session and tls_problem:
            raise BrokerError(f"Alpaca paper network disabled: {tls_problem}")
        try:
            response = self.session.request(
                method,
                f"{self.config.base_url}{path}",
                headers=self._headers,
                params=params,
                json=json_body,
                timeout=self.config.timeout_seconds,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise BrokerError(f"Alpaca paper {method} {path} transport error") from exc
        request_id = response.headers.get("X-Request-ID", "")
        if request_id:
            self._request_ids.append(str(request_id))
            del self._request_ids[:-100]
        if response.status_code == 404 and allow_not_found:
            return None
        if accepted_error_statuses and response.status_code in accepted_error_statuses:
            return response
        if not 200 <= response.status_code < 300:
            suffix = f" request_id={request_id}" if request_id else ""
            raise BrokerError(f"Alpaca paper {method} {path} failed HTTP {response.status_code}{suffix}")
        return response


def _parse_time(value: str | None) -> datetime:
    if not value:
        raise BrokerError("Alpaca paper response omitted a required timestamp")
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return ensure_aware(parsed, "alpaca timestamp")
    except (TypeError, ValueError) as exc:
        raise BrokerError("Alpaca paper response contained an invalid timestamp") from exc


def _calendar_time(trading_date: date, value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("calendar time is missing")
    raw = value.strip()
    if "T" in raw or " " in raw:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            parsed = parsed.replace(tzinfo=NEW_YORK)
        return ensure_aware(parsed, "calendar time")
    parsed_time = time.fromisoformat(raw)
    if parsed_time.tzinfo is not None:
        raise ValueError("calendar wall time must not contain an offset")
    return ensure_aware(
        datetime.combine(trading_date, parsed_time, tzinfo=NEW_YORK), "calendar time"
    )


def _alpaca_time(value: datetime) -> str:
    return ensure_aware(value, "since").isoformat().replace("+00:00", "Z")


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BrokerError(f"Alpaca paper {label} response was not an object")
    return value


def _strict_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a JSON boolean")
    return value


def _tls_runtime_problem() -> str:
    """Return a fail-closed diagnostic for urllib3 v2's supported TLS boundary."""
    if not ssl.OPENSSL_VERSION.startswith("OpenSSL "):
        return (
            f"Python ssl uses unsupported {ssl.OPENSSL_VERSION}; "
            "use an OpenSSL-backed Python runtime"
        )
    if ssl.OPENSSL_VERSION_INFO < (1, 1, 1):
        return (
            f"Python ssl uses unsupported {ssl.OPENSSL_VERSION}; "
            "OpenSSL 1.1.1 or newer is required"
        )
    return ""
