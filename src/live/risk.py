"""Deterministic pre-trade risk checks for long-only cash accounts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from .models import (
    AccountSnapshot,
    OrderRequest,
    Position,
    Quote,
    RiskDecision,
    RiskViolation,
    Side,
    ZERO,
    as_decimal,
    ensure_aware,
    utc_now,
)


@dataclass(frozen=True)
class RiskLimits:
    max_order_notional: Decimal = Decimal("10000")
    max_gross_exposure_pct: Decimal = Decimal("1.00")
    max_symbol_exposure_pct: Decimal = Decimal("0.25")
    max_daily_turnover_pct: Decimal = Decimal("0.50")
    max_daily_loss_pct: Decimal = Decimal("0.02")
    max_drawdown_pct: Decimal = Decimal("0.10")
    max_open_orders: int = 5
    min_cash_buffer: Decimal = ZERO
    quote_max_age_seconds: int = 60
    account_max_age_seconds: int = 60
    max_signal_age_seconds: int = 86400
    price_collar_bps: Decimal = Decimal("100")
    allow_fractional_shares: bool = False
    require_market_open: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_order_notional", "max_gross_exposure_pct", "max_symbol_exposure_pct",
            "max_daily_turnover_pct", "max_daily_loss_pct", "max_drawdown_pct",
            "min_cash_buffer", "price_collar_bps",
        ):
            object.__setattr__(self, name, as_decimal(getattr(self, name)))
        if self.max_order_notional <= ZERO:
            raise ValueError("max_order_notional must be positive")
        if any(getattr(self, name) < ZERO for name in (
            "max_gross_exposure_pct", "max_symbol_exposure_pct", "max_daily_turnover_pct",
            "max_daily_loss_pct", "max_drawdown_pct", "min_cash_buffer", "price_collar_bps",
        )):
            raise ValueError("risk percentages, buffers, and collars cannot be negative")
        if (
            self.max_open_orders < 0 or self.quote_max_age_seconds < 0
            or self.account_max_age_seconds < 0 or self.max_signal_age_seconds < 0
        ):
            raise ValueError("order-count and quote-age limits cannot be negative")


class PreTradeRiskEngine:
    def __init__(self, limits: RiskLimits | None = None, *, clock=utc_now) -> None:
        self.limits = limits or RiskLimits()
        self.clock = clock

    def evaluate(
        self,
        request: OrderRequest,
        *,
        quote: Quote,
        account: AccountSnapshot,
        positions: list[Position],
        open_order_count: int,
        daily_turnover: Decimal,
        day_start_equity: Decimal,
        high_water_equity: Decimal,
        armed: bool,
        kill_switch: bool,
        market_open: bool,
        signal_at: datetime,
        now: datetime | None = None,
        reserved_buy_values: dict[str, Decimal] | None = None,
        reserved_turnover: Decimal = ZERO,
    ) -> RiskDecision:
        now = ensure_aware(now or self.clock(), "risk.now")
        signal_at = ensure_aware(signal_at, "signal_at")
        daily_turnover = as_decimal(daily_turnover)
        reserved_turnover = as_decimal(reserved_turnover)
        day_start_equity = as_decimal(day_start_equity)
        high_water_equity = as_decimal(high_water_equity)
        normalized_reserved_buys: dict[str, Decimal] = {}
        for raw_symbol, raw_value in (reserved_buy_values or {}).items():
            symbol = str(raw_symbol).strip().upper()
            value = as_decimal(raw_value)
            if not symbol or value < ZERO:
                raise ValueError("reserved buy values require symbols and nonnegative notionals")
            normalized_reserved_buys[symbol] = (
                normalized_reserved_buys.get(symbol, ZERO) + value
            )
        if reserved_turnover < ZERO:
            raise ValueError("reserved_turnover cannot be negative")
        reserved_buy_notional = sum(normalized_reserved_buys.values(), ZERO)
        violations: list[RiskViolation] = []

        def reject(code: str, message: str) -> None:
            violations.append(RiskViolation(code, message))

        if request.account_id != account.account_id:
            reject("ACCOUNT_MISMATCH", "Order account does not match broker account")
        if account.status != "ACTIVE":
            reject("ACCOUNT_NOT_ACTIVE", f"Broker account status is {account.status}")
        if account.currency != "USD":
            reject("UNSUPPORTED_CURRENCY", "Phase two supports USD accounts only")
        if account.trading_blocked or account.account_blocked or account.trade_suspended_by_user:
            reject("BROKER_TRADING_BLOCKED", "The broker reports that trading is blocked or suspended")
        if account.cash < ZERO:
            reject("NEGATIVE_CASH", "Phase one cannot operate with negative account cash")
        if account.buying_power < ZERO:
            reject("NEGATIVE_BUYING_POWER", "Broker buying power cannot be negative")
        if not armed:
            reject("NOT_ARMED", "Execution is disarmed")
        if kill_switch:
            reject("KILL_SWITCH", "The persistent kill switch is engaged")
        if self.limits.require_market_open and not market_open:
            reject("MARKET_CLOSED", "Market-open confirmation is required")
        if request.symbol != quote.symbol:
            reject("QUOTE_SYMBOL_MISMATCH", "Quote does not match order symbol")

        age = Decimal(str((now - quote.as_of).total_seconds()))
        if age < ZERO:
            reject("FUTURE_QUOTE", "Quote timestamp is in the future")
        elif age > Decimal(self.limits.quote_max_age_seconds):
            reject("STALE_QUOTE", f"Quote is {age} seconds old")
        account_age = Decimal(str((now - account.as_of).total_seconds()))
        if account_age < ZERO:
            reject("FUTURE_ACCOUNT_SNAPSHOT", "Account snapshot timestamp is in the future")
        elif account_age > Decimal(self.limits.account_max_age_seconds):
            reject("STALE_ACCOUNT_SNAPSHOT", f"Account snapshot is {account_age} seconds old")
        signal_age = Decimal(str((now - signal_at).total_seconds()))
        if signal_age < ZERO:
            reject("FUTURE_SIGNAL", "Signal timestamp is in the future")
        elif signal_age > Decimal(self.limits.max_signal_age_seconds):
            reject("STALE_SIGNAL", f"Signal is {signal_age} seconds old")

        if not self.limits.allow_fractional_shares and request.quantity != request.quantity.to_integral_value():
            reject("FRACTIONAL_DISABLED", "Fractional-share orders are disabled")

        execution_price = quote.ask if request.side == Side.BUY else quote.bid
        risk_price = request.limit_price if request.limit_price is not None else execution_price
        order_notional = request.quantity * risk_price
        collar_bps = (
            abs(risk_price / request.reference_price - Decimal("1")) * Decimal("10000")
            if request.reference_price > ZERO else Decimal("Infinity")
        )
        if collar_bps > self.limits.price_collar_bps:
            reject(
                "PRICE_COLLAR",
                f"Risk price moved {collar_bps:.2f} bps from the intent reference",
            )
        if order_notional > self.limits.max_order_notional:
            reject("ORDER_NOTIONAL", "Order exceeds max_order_notional")
        if open_order_count >= self.limits.max_open_orders:
            reject("OPEN_ORDER_LIMIT", "Maximum number of open orders reached")

        by_symbol = {p.symbol: p for p in positions}
        if any(p.quantity < ZERO for p in positions):
            reject("EXISTING_SHORT", "Phase one cannot operate an account with short positions")
        current = by_symbol.get(request.symbol)
        current_qty = current.quantity if current else ZERO
        signed_qty = request.quantity if request.side == Side.BUY else -request.quantity
        projected_qty = current_qty + signed_qty
        if projected_qty < ZERO:
            reject("SHORT_SALE", "Order would create a short position")

        if request.side == Side.BUY:
            if reserved_buy_notional + order_notional > account.buying_power:
                reject("BUYING_POWER", "Order exceeds broker-reported buying power")
            if (
                account.cash - reserved_buy_notional - order_notional
                < self.limits.min_cash_buffer
            ):
                reject("CASH_BUFFER", "Order would breach the minimum cash buffer")

        projected_values: dict[str, Decimal] = {}
        for p in positions:
            px = quote.mid if p.symbol == request.symbol else p.market_price
            projected_values[p.symbol] = p.quantity * px
        for symbol, value in normalized_reserved_buys.items():
            projected_values[symbol] = projected_values.get(symbol, ZERO) + value
        if projected_qty == ZERO:
            projected_values.pop(request.symbol, None)
        else:
            projected_values[request.symbol] = projected_qty * quote.mid
        gross = sum((abs(v) for v in projected_values.values()), ZERO)
        if account.equity <= ZERO:
            reject("NONPOSITIVE_EQUITY", "Account equity must be positive")
        else:
            if gross / account.equity > self.limits.max_gross_exposure_pct:
                reject("GROSS_EXPOSURE", "Projected gross exposure exceeds its account limit")
            excessive_symbols = sorted(
                symbol for symbol, value in projected_values.items()
                if abs(value) / account.equity > self.limits.max_symbol_exposure_pct
            )
            if excessive_symbols:
                reject("SYMBOL_EXPOSURE", "Projected symbol concentration exceeds its limit")
            if (
                (daily_turnover + reserved_turnover + order_notional) / account.equity
                > self.limits.max_daily_turnover_pct
            ):
                reject("DAILY_TURNOVER", "Projected daily turnover exceeds its limit")

        if day_start_equity <= ZERO:
            reject("DAY_START_EQUITY", "Valid day-start equity is required")
        elif account.equity / day_start_equity - Decimal("1") <= -self.limits.max_daily_loss_pct:
            reject("DAILY_LOSS", "Daily loss limit has been reached")
        if high_water_equity <= ZERO:
            reject("HIGH_WATER_EQUITY", "Valid high-water equity is required")
        elif account.equity / high_water_equity - Decimal("1") <= -self.limits.max_drawdown_pct:
            reject("DRAWDOWN", "Account drawdown limit has been reached")

        return RiskDecision(
            allowed=not violations,
            violations=tuple(violations),
            order_notional=order_notional,
            projected_gross_exposure=gross,
        )
