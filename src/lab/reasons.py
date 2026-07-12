"""Stable, machine-readable reason codes for every decision the lab traces.

One central taxonomy instead of ad-hoc strings scattered through strategies:
each code has a fixed identifier (never renamed once shipped -- downstream
artifacts key on them), a category, and a human-readable template. The
existing engine `reason` strings ("rebalance", "risk_off", ...) are kept
untouched for backward compatibility; the lab trace carries these richer
codes alongside them.

Templates are filled ONLY from values the strategy/engine actually used at
the decision point (see trace.py) -- explanations are grounded in the traced
numbers, never invented after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReasonCode:
    code: str
    category: str  # "selection" | "rejection" | "regime" | "order" | "risk_control" | "data"
    template: str  # human-readable, .format(**fields)-able with trace fields


_CODES: list[ReasonCode] = [
    # --- Selection outcomes ---
    ReasonCode("SELECTED_TOP_K", "selection",
               "selected: ranked {rank} of {num_ranked} eligible (score {rank_score}), within top {top_k}"),
    ReasonCode("POSITION_RETAINED", "selection",
               "retained: still ranked {rank} of {num_ranked} eligible, within top {top_k}"),
    ReasonCode("RANK_BELOW_CUTOFF", "rejection",
               "rejected: eligible but ranked {rank} of {num_ranked}, below top-{top_k} cutoff"),
    ReasonCode("NEGATIVE_TRAILING_RETURN", "rejection",
               "rejected: trailing {lookback_days}-session return {lookback_return} is not positive "
               "(absolute-momentum filter)"),
    ReasonCode("BELOW_TREND_FILTER", "rejection",
               "rejected: close {close} is not above its {trend_sma_period}-session SMA {trend_sma} "
               "(trend filter)"),
    ReasonCode("ABOVE_TREND_FILTER", "selection",
               "passes trend filter: close {close} above its {trend_sma_period}-session SMA {trend_sma}"),
    ReasonCode("POSITION_EXITED", "selection",
               "exited: no longer selected at this rebalance (target weight set to 0)"),
    ReasonCode("ALREADY_AT_TARGET", "selection",
               "no order needed: already at target weight {target_weight}"),
    # --- Mean-reversion specific ---
    ReasonCode("ENTRY_RSI_OVERSOLD", "selection",
               "entered: RSI {rsi} below entry threshold {rsi_entry}"),
    ReasonCode("RSI_NOT_OVERSOLD", "rejection",
               "not entered: RSI {rsi} not below entry threshold {rsi_entry}"),
    ReasonCode("EXIT_STOP_LOSS", "selection",
               "exited: return since entry {position_return} breached stop {stop_loss_pct}"),
    ReasonCode("EXIT_TIMEOUT", "selection",
               "exited: held {days_held} sessions, max {max_holding_days}"),
    ReasonCode("EXIT_SMA", "selection", "exited: close {close} above SMA {sma}"),
    ReasonCode("EXIT_RSI", "selection", "exited: RSI {rsi} at/above exit threshold {rsi_exit}"),
    ReasonCode("NO_FREE_SLOT", "rejection",
               "not entered: all {max_positions} position slots occupied"),
    ReasonCode("ENTRY_FILTER_BLOCKED", "rejection",
               "not entered: entry filter blocked the candidate (see detail)"),
    # --- Regime ---
    ReasonCode("REGIME_RISK_ON", "regime",
               "risk-on: {regime_ticker} close {regime_close} above its {regime_sma_period}-session "
               "SMA {regime_sma}"),
    ReasonCode("REGIME_RISK_OFF", "regime",
               "risk-off: {regime_ticker} close {regime_close} not above its {regime_sma_period}-session "
               "SMA {regime_sma} -> 100% cash"),
    ReasonCode("CASH_FALLBACK", "selection",
               "cash fallback: only {num_selected} of {top_k} slots filled; remainder stays in cash"),
    # --- Data / eligibility ---
    ReasonCode("INSUFFICIENT_HISTORY", "data",
               "excluded: indicators not yet defined at this date (insufficient history)"),
    ReasonCode("MISSING_SIGNAL_DATA", "data",
               "excluded: no price data available at the signal date"),
    ReasonCode("MISSING_FILL_DATA", "data",
               "order not scheduled: no fill-price data available at the intended fill date"),
    ReasonCode("SIGNAL_ONLY_TICKER", "data",
               "signal-only ticker: used as an input, never tradable"),
    ReasonCode("NOT_IN_UNIVERSE", "data",
               "excluded: not a member of the universe at this date"),
    # --- Schedule ---
    ReasonCode("REBALANCE_NOT_DUE", "selection",
               "no action: rebalance not due at this date (schedule: {schedule})"),
    ReasonCode("REBALANCE_DUE", "selection", "rebalance due at this date (schedule: {schedule})"),
    # --- Risk-control overlays (lab enhancements; all default-off) ---
    ReasonCode("VOLATILITY_TARGET", "risk_control",
               "volatility target: trailing {vol_lookback}-session realized vol {realized_vol} vs "
               "target {vol_target} -> exposure multiplier {exposure_multiplier}"),
    ReasonCode("DRAWDOWN_THROTTLE", "risk_control",
               "drawdown throttle: portfolio drawdown {drawdown} in band {throttle_band} -> "
               "exposure multiplier {exposure_multiplier}"),
    ReasonCode("THROTTLE_RECOVERED", "risk_control",
               "drawdown throttle released: drawdown {drawdown} recovered above {recovery_threshold}"),
    ReasonCode("NO_TRADE_BAND", "risk_control",
               "order suppressed: weight change {weight_change} below no-trade band {band}"),
    ReasonCode("MIN_NOTIONAL_BAND", "risk_control",
               "order suppressed: requested notional {requested_notional} below minimum {min_notional}"),
    ReasonCode("TURNOVER_LIMIT", "risk_control",
               "order deferred: period turnover budget {turnover_budget} exhausted"),
    ReasonCode("CONCENTRATION_LIMIT", "risk_control",
               "weight capped: aggregate {ticker} weight {aggregate_weight} above cap {cap} -> "
               "scaled to {capped_weight}"),
    ReasonCode("RISK_BUDGET_REDUCTION", "risk_control",
               "exposure reduced by risk overlay to multiplier {exposure_multiplier}"),
    ReasonCode("SLEEVE_REALLOCATION", "risk_control",
               "dynamic sleeve allocation: {sleeve} target weight {target_sleeve_weight} vs drifted "
               "{drifted_sleeve_weight} -> transfer {transfer_notional}"),
    ReasonCode("CASH_YIELD_ACCRUAL", "risk_control",
               "cash yield accrued at {annual_rate} annual on cash balance {cash}"),
    # --- Order lifecycle (engine-side) ---
    ReasonCode("ORDER_FILLED", "order", "filled {action} {shares_traded} shares at {fill_price}"),
    ReasonCode("ORDER_SCALED_TO_CASH", "order",
               "buy scaled to available cash (scale factor {buy_scale})"),
    ReasonCode("ORDER_BELOW_MIN_NOTIONAL", "order",
               "no-op: trade below engine minimum notional"),
    ReasonCode("ORDER_DEFERRED", "order", "order deferred (see detail)"),
    ReasonCode("ORDER_STALE", "order", "order stale / never fillable (see detail)"),
]

REASON_CODES: dict[str, ReasonCode] = {rc.code: rc for rc in _CODES}


class UnknownReasonCode(Exception):
    pass


def validate_code(code: str) -> str:
    """Return `code` unchanged if registered; raise otherwise. Trace writers
    call this so an unregistered ad-hoc string can never leak into artifacts."""
    if code not in REASON_CODES:
        raise UnknownReasonCode(
            f"Reason code {code!r} is not registered in src/lab/reasons.py. "
            f"Add it to the central taxonomy instead of using an ad-hoc string."
        )
    return code


def _fmt_value(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def human_reason(code: str, fields: dict) -> str:
    """Render the human-readable explanation for `code` from the traced
    `fields`. Missing fields render as 'n/a' rather than raising -- a trace
    row is allowed to omit fields that don't apply, and the explanation must
    still be printable."""
    rc = REASON_CODES.get(code)
    if rc is None:
        raise UnknownReasonCode(f"Reason code {code!r} is not registered.")

    class _Safe(dict):
        def __missing__(self, key):  # noqa: D105
            return "n/a"

    safe = _Safe({k: _fmt_value(v) for k, v in fields.items() if v is not None})
    return rc.template.format_map(safe)
