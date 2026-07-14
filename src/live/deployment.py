"""Strict Phase-3 deployment configuration and sleeve aggregation.

Research sleeves remain independent.  This module is the reviewed boundary
that combines them into one account-level target before the OMS sees an order.
All money and weight arithmetic uses :class:`~decimal.Decimal`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass, fields
from datetime import date, datetime
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import Any

from .models import AccountSnapshot, Position, Quote, ZERO, as_decimal, ensure_aware, json_safe
from .risk import RiskLimits


DEPLOYMENT_SCHEMA_VERSION = 1
TARGET_SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 1_000_000
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.-]{0,14}$")
_DECIMAL_RISK_FIELDS = {
    "max_order_notional", "max_gross_exposure_pct", "max_symbol_exposure_pct",
    "max_daily_turnover_pct", "max_daily_loss_pct", "max_drawdown_pct",
    "min_cash_buffer", "price_collar_bps",
}
_INTEGER_RISK_FIELDS = {
    "max_open_orders", "quote_max_age_seconds", "account_max_age_seconds",
    "max_signal_age_seconds",
}
_BOOLEAN_RISK_FIELDS = {"allow_fractional_shares", "require_market_open"}


class DeploymentError(ValueError):
    """A deployment artifact is unsafe or internally inconsistent."""


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DeploymentError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_strict_json(path: str | Path) -> dict[str, Any]:
    """Load a small JSON object while rejecting duplicate object keys."""
    source = Path(path).expanduser()
    if source.is_symlink():
        raise DeploymentError("Deployment artifacts may not be symbolic links")
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(str(source), flags)
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            file_stat = os.fstat(handle.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                raise DeploymentError("Deployment artifact must be a regular file")
            if file_stat.st_size <= 0 or file_stat.st_size > MAX_CONFIG_BYTES:
                raise DeploymentError(
                    f"Deployment artifact must be between 1 and {MAX_CONFIG_BYTES} bytes"
                )
            text = handle.read(MAX_CONFIG_BYTES + 1)
        if len(text.encode("utf-8")) > MAX_CONFIG_BYTES:
            raise DeploymentError("Deployment artifact changed while it was being read")
        payload = json.loads(
            text,
            parse_float=Decimal,
            object_pairs_hook=_pairs_no_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DeploymentError(f"Invalid JSON deployment artifact: {source}") from exc
    if not isinstance(payload, dict):
        raise DeploymentError("Deployment artifact root must be a JSON object")
    return payload


def _exact_keys(payload: dict[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(payload))
    unknown = sorted(set(payload) - expected)
    if missing or unknown:
        raise DeploymentError(
            f"{label} keys do not match schema; missing={missing}, unknown={unknown}"
        )


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value.strip()):
        raise DeploymentError(f"{label} must match {_IDENTIFIER.pattern}")
    return value.strip()


def _symbol(value: Any, label: str = "symbol") -> str:
    if not isinstance(value, str):
        raise DeploymentError(f"{label} must be a string")
    normalized = value.strip().upper()
    if not _SYMBOL.fullmatch(normalized):
        raise DeploymentError(f"Invalid US-equity {label}: {value!r}")
    return normalized


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool):
        raise DeploymentError(f"{label} must be numeric, not boolean")
    try:
        return as_decimal(value)
    except (ArithmeticError, TypeError, ValueError) as exc:
        raise DeploymentError(f"{label} must be a finite decimal") from exc


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DeploymentError(f"{label} must be a positive integer")
    return value


def _parse_risk_limits(payload: Any) -> RiskLimits:
    if not isinstance(payload, dict):
        raise DeploymentError("risk_limits must be an object")
    expected = {field.name for field in fields(RiskLimits)}
    _exact_keys(payload, expected, "risk_limits")
    values: dict[str, Any] = {}
    for name in _DECIMAL_RISK_FIELDS:
        values[name] = _decimal(payload[name], f"risk_limits.{name}")
    for name in _INTEGER_RISK_FIELDS:
        values[name] = _positive_int(payload[name], f"risk_limits.{name}")
    for name in _BOOLEAN_RISK_FIELDS:
        if not isinstance(payload[name], bool):
            raise DeploymentError(f"risk_limits.{name} must be a JSON boolean")
        values[name] = payload[name]
    limits = RiskLimits(**values)
    if limits.allow_fractional_shares:
        raise DeploymentError("Phase 3 requires allow_fractional_shares=false")
    if not limits.require_market_open:
        raise DeploymentError("Phase 3 requires require_market_open=true")
    if not ZERO < limits.max_gross_exposure_pct <= Decimal("1"):
        raise DeploymentError("max_gross_exposure_pct must be in (0, 1]")
    if not ZERO < limits.max_symbol_exposure_pct <= limits.max_gross_exposure_pct:
        raise DeploymentError(
            "max_symbol_exposure_pct must be positive and no greater than gross exposure"
        )
    if not ZERO < limits.max_daily_turnover_pct <= Decimal("2"):
        raise DeploymentError("max_daily_turnover_pct must be in (0, 2]")
    if not ZERO < limits.max_daily_loss_pct < Decimal("1"):
        raise DeploymentError("max_daily_loss_pct must be in (0, 1)")
    if not ZERO < limits.max_drawdown_pct < Decimal("1"):
        raise DeploymentError("max_drawdown_pct must be in (0, 1)")
    if limits.quote_max_age_seconds > 60 or limits.account_max_age_seconds > 60:
        raise DeploymentError("quote and account maximum ages may not exceed 60 seconds")
    if limits.max_signal_age_seconds > 7 * 24 * 60 * 60:
        raise DeploymentError("max_signal_age_seconds may not exceed 7 calendar days")
    if limits.price_collar_bps > Decimal("500"):
        raise DeploymentError("price_collar_bps may not exceed 500 bps")
    return limits


def risk_limits_payload(limits: RiskLimits) -> dict[str, Any]:
    return {field.name: getattr(limits, field.name) for field in fields(RiskLimits)}


@dataclass(frozen=True)
class DeploymentConfig:
    deployment_id: str
    account_id: str
    allocation_policy: str
    execution_policy: str
    managed_symbols: tuple[str, ...]
    sleeve_weights: tuple[tuple[str, Decimal], ...]
    risk_limits: RiskLimits
    max_batch_orders: int
    schema_version: int = DEPLOYMENT_SCHEMA_VERSION

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "DeploymentConfig":
        _exact_keys(payload, {
            "schema_version", "deployment_id", "account_id", "managed_symbols",
            "allocation_policy", "execution_policy", "sleeve_weights",
            "risk_limits", "max_batch_orders",
        }, "deployment")
        if payload["schema_version"] != DEPLOYMENT_SCHEMA_VERSION:
            raise DeploymentError(
                f"Unsupported deployment schema_version {payload['schema_version']!r}"
            )
        deployment_id = _identifier(payload["deployment_id"], "deployment_id")
        account_id = _identifier(payload["account_id"], "account_id")
        if payload["allocation_policy"] != "rebalance_to_deployment_weights":
            raise DeploymentError(
                "allocation_policy must explicitly be 'rebalance_to_deployment_weights'"
            )
        if payload["execution_policy"] != "next_session_regular_hours_market":
            raise DeploymentError(
                "execution_policy must explicitly be 'next_session_regular_hours_market'"
            )
        raw_symbols = payload["managed_symbols"]
        if not isinstance(raw_symbols, list) or not raw_symbols:
            raise DeploymentError("managed_symbols must be a non-empty array")
        symbols = tuple(_symbol(value, "managed symbol") for value in raw_symbols)
        if len(symbols) != len(set(symbols)):
            raise DeploymentError("managed_symbols contains duplicates after normalization")
        if tuple(sorted(symbols)) != symbols:
            raise DeploymentError("managed_symbols must be sorted for deterministic review")
        if len(symbols) > 100:
            raise DeploymentError("Phase 3 permits at most 100 managed symbols")

        raw_sleeves = payload["sleeve_weights"]
        if not isinstance(raw_sleeves, dict) or not raw_sleeves:
            raise DeploymentError("sleeve_weights must be a non-empty object")
        sleeves: list[tuple[str, Decimal]] = []
        for name, raw_weight in raw_sleeves.items():
            normalized = _identifier(name, "sleeve name")
            weight = _decimal(raw_weight, f"sleeve_weights.{name}")
            if weight <= ZERO or weight > Decimal("1"):
                raise DeploymentError("Every sleeve weight must be in (0, 1]")
            sleeves.append((normalized, weight))
        sleeves.sort()
        if sum((weight for _name, weight in sleeves), ZERO) > Decimal("1"):
            raise DeploymentError("sleeve_weights may not allocate more than 100% of equity")

        limits = _parse_risk_limits(payload["risk_limits"])
        max_batch_orders = _positive_int(payload["max_batch_orders"], "max_batch_orders")
        if max_batch_orders > limits.max_open_orders:
            raise DeploymentError("max_batch_orders may not exceed risk max_open_orders")
        return cls(
            deployment_id, account_id, payload["allocation_policy"],
            payload["execution_policy"], symbols, tuple(sleeves), limits,
            max_batch_orders,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "DeploymentConfig":
        return cls.from_payload(load_strict_json(path))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "deployment_id": self.deployment_id,
            "account_id": self.account_id,
            "allocation_policy": self.allocation_policy,
            "execution_policy": self.execution_policy,
            "managed_symbols": list(self.managed_symbols),
            "sleeve_weights": dict(self.sleeve_weights),
            "risk_limits": risk_limits_payload(self.risk_limits),
            "max_batch_orders": self.max_batch_orders,
        }


@dataclass(frozen=True)
class SleeveTargetSnapshot:
    signal_at: datetime
    target_version: str
    sleeves: tuple[tuple[str, tuple[tuple[str, Decimal], ...]], ...]
    schema_version: int = TARGET_SCHEMA_VERSION

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any], deployment: DeploymentConfig
    ) -> "SleeveTargetSnapshot":
        _exact_keys(payload, {"schema_version", "signal_at", "target_version", "sleeves"}, "targets")
        if payload["schema_version"] != TARGET_SCHEMA_VERSION:
            raise DeploymentError(
                f"Unsupported target schema_version {payload['schema_version']!r}"
            )
        if not isinstance(payload["signal_at"], str):
            raise DeploymentError("signal_at must be an ISO-8601 string")
        try:
            signal_at = ensure_aware(
                datetime.fromisoformat(payload["signal_at"].replace("Z", "+00:00")),
                "signal_at",
            )
        except (TypeError, ValueError) as exc:
            raise DeploymentError("signal_at must be a timezone-aware ISO-8601 timestamp") from exc
        target_version = _identifier(payload["target_version"], "target_version")
        raw_sleeves = payload["sleeves"]
        if not isinstance(raw_sleeves, dict):
            raise DeploymentError("sleeves must be an object")
        expected_sleeves = {name for name, _weight in deployment.sleeve_weights}
        if set(raw_sleeves) != expected_sleeves:
            raise DeploymentError(
                "Target sleeves must exactly match deployment sleeve_weights; "
                f"missing={sorted(expected_sleeves - set(raw_sleeves))}, "
                f"unknown={sorted(set(raw_sleeves) - expected_sleeves)}"
            )
        managed = set(deployment.managed_symbols)
        sleeves: list[tuple[str, tuple[tuple[str, Decimal], ...]]] = []
        for sleeve_name in sorted(raw_sleeves):
            raw_targets = raw_sleeves[sleeve_name]
            if not isinstance(raw_targets, dict):
                raise DeploymentError(f"sleeves.{sleeve_name} must be an object")
            normalized: dict[str, Decimal] = {}
            for raw_symbol, raw_weight in raw_targets.items():
                symbol = _symbol(raw_symbol)
                if symbol in normalized:
                    raise DeploymentError(
                        f"sleeves.{sleeve_name} contains duplicate normalized symbol {symbol}"
                    )
                if symbol not in managed:
                    raise DeploymentError(f"Target symbol {symbol} is outside managed_symbols")
                weight = _decimal(raw_weight, f"sleeves.{sleeve_name}.{symbol}")
                if weight < ZERO or weight > Decimal("1"):
                    raise DeploymentError("Sleeve target weights must be in [0, 1]")
                normalized[symbol] = weight
            if sum(normalized.values(), ZERO) > Decimal("1"):
                raise DeploymentError(f"Sleeve {sleeve_name} targets exceed 100%")
            sleeves.append((sleeve_name, tuple(sorted(normalized.items()))))
        return cls(signal_at, target_version, tuple(sleeves))

    @classmethod
    def from_file(
        cls, path: str | Path, deployment: DeploymentConfig
    ) -> "SleeveTargetSnapshot":
        return cls.from_payload(load_strict_json(path), deployment)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "signal_at": self.signal_at,
            "target_version": self.target_version,
            "sleeves": {
                name: dict(targets) for name, targets in self.sleeves
            },
        }


@dataclass(frozen=True)
class ExecutionPlanItem:
    symbol: str
    target_weight: Decimal
    target_quantity: Decimal
    current_quantity: Decimal
    delta_quantity: Decimal
    reference_price: Decimal
    quote_as_of: datetime

    def to_payload(self) -> dict[str, Any]:
        return json_safe({
            "symbol": self.symbol,
            "target_weight": self.target_weight,
            "target_quantity": self.target_quantity,
            "current_quantity": self.current_quantity,
            "delta_quantity": self.delta_quantity,
            "reference_price": self.reference_price,
            "quote_as_of": self.quote_as_of,
        })


@dataclass(frozen=True)
class ExecutionPlan:
    batch_id: str
    plan_hash: str
    source_hash: str
    deployment_id: str
    account_id: str
    signal_at: datetime
    trading_date: str
    target_version: str
    risk_limits: RiskLimits
    items: tuple[ExecutionPlanItem, ...]
    unmanaged_positions: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "plan_hash": self.plan_hash,
            **self.body_payload(),
        }

    def body_payload(self) -> dict[str, Any]:
        return json_safe({
            "schema_version": 1,
            "source_hash": self.source_hash,
            "deployment_id": self.deployment_id,
            "account_id": self.account_id,
            "signal_at": self.signal_at,
            "trading_date": self.trading_date,
            "target_version": self.target_version,
            "risk_limits": risk_limits_payload(self.risk_limits),
            "items": [item.to_payload() for item in self.items],
            "unmanaged_positions": list(self.unmanaged_positions),
        })

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ExecutionPlan":
        _exact_keys(payload, {
            "batch_id", "plan_hash", "schema_version", "source_hash", "deployment_id",
            "account_id", "signal_at", "trading_date", "target_version", "risk_limits",
            "items", "unmanaged_positions",
        }, "execution plan")
        if payload["schema_version"] != 1:
            raise DeploymentError("Unsupported execution plan schema")
        if not isinstance(payload["batch_id"], str) or not re.fullmatch(
            r"batch-[0-9a-f]{24}", payload["batch_id"]
        ):
            raise DeploymentError("Execution plan batch_id is invalid")
        for name in ("plan_hash", "source_hash"):
            if not isinstance(payload[name], str) or not re.fullmatch(
                r"[0-9a-f]{64}", payload[name]
            ):
                raise DeploymentError(f"Execution plan {name} is invalid")
        deployment_id = _identifier(payload["deployment_id"], "plan.deployment_id")
        account_id = _identifier(payload["account_id"], "plan.account_id")
        if (
            not isinstance(payload["target_version"], str)
            or not payload["target_version"].strip()
            or len(payload["target_version"]) > 128
        ):
            raise DeploymentError("Execution plan target_version is invalid")
        try:
            trading_date = date.fromisoformat(str(payload["trading_date"])).isoformat()
        except ValueError as exc:
            raise DeploymentError("Execution plan trading_date is invalid") from exc
        signal_at = ensure_aware(
            datetime.fromisoformat(str(payload["signal_at"]).replace("Z", "+00:00")),
            "plan.signal_at",
        )
        items: list[ExecutionPlanItem] = []
        if not isinstance(payload["items"], list):
            raise DeploymentError("Execution plan items must be an array")
        if not payload["items"]:
            raise DeploymentError("Execution plan must contain at least one managed symbol")
        for raw in payload["items"]:
            if not isinstance(raw, dict):
                raise DeploymentError("Execution plan item must be an object")
            _exact_keys(raw, {
                "symbol", "target_weight", "target_quantity", "current_quantity",
                "delta_quantity", "reference_price", "quote_as_of",
            }, "execution plan item")
            items.append(ExecutionPlanItem(
                _symbol(raw["symbol"]),
                _decimal(raw["target_weight"], "target_weight"),
                _decimal(raw["target_quantity"], "target_quantity"),
                _decimal(raw["current_quantity"], "current_quantity"),
                _decimal(raw["delta_quantity"], "delta_quantity"),
                _decimal(raw["reference_price"], "reference_price"),
                ensure_aware(
                    datetime.fromisoformat(str(raw["quote_as_of"]).replace("Z", "+00:00")),
                    "quote_as_of",
                ),
            ))
        symbols = [item.symbol for item in items]
        if symbols != sorted(symbols) or len(symbols) != len(set(symbols)):
            raise DeploymentError("Execution plan items must have unique sorted symbols")
        for item in items:
            if item.target_weight < ZERO or item.target_weight > Decimal("1"):
                raise DeploymentError("Execution plan target weight is invalid")
            if item.target_quantity < ZERO or item.current_quantity < ZERO:
                raise DeploymentError("Execution plan quantities cannot be negative")
            if any(
                quantity != quantity.to_integral_value()
                for quantity in (item.target_quantity, item.current_quantity, item.delta_quantity)
            ):
                raise DeploymentError("Execution plan quantities must be whole shares")
            if item.delta_quantity != item.target_quantity - item.current_quantity:
                raise DeploymentError("Execution plan delta does not match target-current")
            if item.reference_price <= ZERO:
                raise DeploymentError("Execution plan reference price must be positive")
        raw_unmanaged = payload["unmanaged_positions"]
        if not isinstance(raw_unmanaged, list):
            raise DeploymentError("unmanaged_positions must be an array")
        unmanaged = tuple(_symbol(value, "unmanaged symbol") for value in raw_unmanaged)
        if unmanaged:
            raise DeploymentError("Executable Phase-3 plan cannot contain unmanaged positions")
        limits = _parse_risk_limits(payload["risk_limits"])
        if sum((item.target_weight for item in items), ZERO) > limits.max_gross_exposure_pct:
            raise DeploymentError("Execution plan weights exceed gross exposure")
        if any(item.target_weight > limits.max_symbol_exposure_pct for item in items):
            raise DeploymentError("Execution plan weight exceeds symbol exposure")
        actionable_count = sum(1 for item in items if item.delta_quantity != ZERO)
        if actionable_count > limits.max_open_orders:
            raise DeploymentError(
                "Execution plan order count exceeds the stored open-order safety limit"
            )
        plan = cls(
            str(payload["batch_id"]), str(payload["plan_hash"]), str(payload["source_hash"]),
            deployment_id, account_id, signal_at,
            trading_date, payload["target_version"].strip(),
            limits, tuple(items), unmanaged,
        )
        computed = canonical_hash(plan.body_payload())
        if computed != plan.plan_hash or plan.batch_id != f"batch-{computed[:24]}":
            raise DeploymentError("Execution plan hash or batch ID is invalid")
        return plan


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def aggregate_sleeves(
    deployment: DeploymentConfig, snapshot: SleeveTargetSnapshot
) -> dict[str, Decimal]:
    allocations = dict(deployment.sleeve_weights)
    combined = {symbol: ZERO for symbol in deployment.managed_symbols}
    for sleeve_name, targets in snapshot.sleeves:
        sleeve_allocation = allocations[sleeve_name]
        for symbol, target_weight in targets:
            combined[symbol] += sleeve_allocation * target_weight
    total = sum(combined.values(), ZERO)
    if total > deployment.risk_limits.max_gross_exposure_pct:
        raise DeploymentError("Aggregated target exceeds max_gross_exposure_pct")
    excessive = [
        symbol for symbol, weight in combined.items()
        if weight > deployment.risk_limits.max_symbol_exposure_pct
    ]
    if excessive:
        raise DeploymentError(
            f"Aggregated targets exceed max_symbol_exposure_pct: {sorted(excessive)}"
        )
    return combined


def build_execution_plan(
    deployment: DeploymentConfig,
    snapshot: SleeveTargetSnapshot,
    *,
    account: AccountSnapshot,
    positions: list[Position],
    quotes: dict[str, Quote],
    trading_date: str,
    daily_turnover: Decimal = ZERO,
) -> ExecutionPlan:
    if account.account_id != deployment.account_id:
        raise DeploymentError("Deployment account does not match authenticated account")
    if account.equity <= ZERO:
        raise DeploymentError("Positive account equity is required")
    daily_turnover = _decimal(daily_turnover, "daily_turnover")
    if daily_turnover < ZERO:
        raise DeploymentError("daily_turnover cannot be negative")
    expected_symbols = set(deployment.managed_symbols)
    if set(quotes) != expected_symbols:
        raise DeploymentError(
            "Quote set must exactly match managed_symbols; "
            f"missing={sorted(expected_symbols - set(quotes))}, "
            f"unknown={sorted(set(quotes) - expected_symbols)}"
        )
    position_by_symbol: dict[str, Position] = {}
    for position in positions:
        if position.symbol in position_by_symbol:
            raise DeploymentError(f"Duplicate broker position for {position.symbol}")
        if position.quantity < ZERO or position.quantity != position.quantity.to_integral_value():
            raise DeploymentError("Phase 3 cannot manage short or fractional positions")
        position_by_symbol[position.symbol] = position
    unmanaged = tuple(sorted(
        symbol for symbol, position in position_by_symbol.items()
        if symbol not in expected_symbols and position.quantity != ZERO
    ))
    if unmanaged:
        raise DeploymentError(
            f"Dedicated paper account contains unmanaged positions: {list(unmanaged)}"
        )

    combined = aggregate_sleeves(deployment, snapshot)
    items: list[ExecutionPlanItem] = []
    target_market_value = ZERO
    for symbol in deployment.managed_symbols:
        quote = quotes[symbol]
        if quote.symbol != symbol:
            raise DeploymentError(f"Quote symbol mismatch for {symbol}")
        target_weight = combined[symbol]
        # Size buys against the ask, conservatively leaving whole-share cash
        # residuals.  The OMS still performs a fresh independent risk check.
        raw_quantity = account.equity * target_weight / quote.ask
        target_quantity = raw_quantity.to_integral_value(rounding=ROUND_FLOOR)
        current_quantity = position_by_symbol.get(
            symbol, Position(symbol, ZERO, ZERO, quote.mid)
        ).quantity
        delta = target_quantity - current_quantity
        target_market_value += target_quantity * quote.ask
        if delta != ZERO:
            execution_price = quote.ask if delta > ZERO else quote.bid
            if abs(delta) * execution_price > deployment.risk_limits.max_order_notional:
                raise DeploymentError(f"Planned {symbol} order exceeds max_order_notional")
        items.append(ExecutionPlanItem(
            symbol, target_weight, target_quantity, current_quantity, delta,
            quote.mid, quote.as_of,
        ))
    if target_market_value + deployment.risk_limits.min_cash_buffer > account.equity:
        raise DeploymentError("Target portfolio would breach min_cash_buffer")
    order_count = sum(1 for item in items if item.delta_quantity != ZERO)
    if order_count > deployment.max_batch_orders:
        raise DeploymentError(
            f"Plan requires {order_count} orders, exceeding max_batch_orders="
            f"{deployment.max_batch_orders}"
        )
    planned_turnover = daily_turnover + sum(
        (
            abs(item.delta_quantity)
            * (quotes[item.symbol].ask if item.delta_quantity > ZERO else quotes[item.symbol].bid)
            for item in items
            if item.delta_quantity != ZERO
        ),
        ZERO,
    )
    if planned_turnover / account.equity > deployment.risk_limits.max_daily_turnover_pct:
        raise DeploymentError(
            "The complete batch would exceed max_daily_turnover_pct at preview prices"
        )
    source_hash = canonical_hash({
        "deployment": deployment.to_payload(), "targets": snapshot.to_payload()
    })
    provisional_version = f"{snapshot.target_version}-{source_hash[:12]}"
    body = json_safe({
        "schema_version": 1,
        "source_hash": source_hash,
        "deployment_id": deployment.deployment_id,
        "account_id": deployment.account_id,
        "signal_at": snapshot.signal_at,
        "trading_date": trading_date,
        "target_version": provisional_version,
        "risk_limits": risk_limits_payload(deployment.risk_limits),
        "items": [item.to_payload() for item in items],
        "unmanaged_positions": list(unmanaged),
    })
    plan_hash = canonical_hash(body)
    return ExecutionPlan(
        batch_id=f"batch-{plan_hash[:24]}",
        plan_hash=plan_hash,
        source_hash=source_hash,
        deployment_id=deployment.deployment_id,
        account_id=deployment.account_id,
        signal_at=snapshot.signal_at,
        trading_date=trading_date,
        target_version=provisional_version,
        risk_limits=deployment.risk_limits,
        items=tuple(items),
        unmanaged_positions=unmanaged,
    )
