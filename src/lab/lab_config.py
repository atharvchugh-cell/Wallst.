"""Declarative configuration for every lab enhancement.

Design rules (non-negotiable, enforced here and by tests):
  - every enhancement is OFF by default; a default-constructed LabConfig is
    the do-nothing configuration, proven behavior-identical to the existing
    portfolio mode;
  - unknown keys FAIL CLOSED (a typo'd flag must never silently no-op);
  - the config is hashable (sha256 over canonical JSON) so every artifact can
    be tied to the exact configuration that produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields


class LabConfigError(Exception):
    pass


@dataclass
class VolTargetConfig:
    """Candidate A: portfolio-level volatility targeting. Exposure multiplier
    = clip(target_vol / realized_vol, min_exposure, max_exposure), where
    realized vol is computed from LAGGED portfolio daily returns only (through
    the signal date). No leverage by default (max_exposure <= 1.0)."""

    enabled: bool = False
    target_annual_vol: float = 0.12
    lookback_days: int = 63
    min_exposure: float = 0.40
    max_exposure: float = 1.00
    smoothing_days: int = 5          # trailing mean of raw multiplier; 1 = none
    adjustment_band: float = 0.05    # re-trade held positions only if multiplier moved this much
    evaluate: str = "weekly"         # "daily" | "weekly" | "month_end"

    def validate(self) -> None:
        if self.target_annual_vol <= 0:
            raise LabConfigError(f"vol_target.target_annual_vol must be > 0, got {self.target_annual_vol}")
        if self.lookback_days < 10:
            raise LabConfigError(f"vol_target.lookback_days must be >= 10, got {self.lookback_days}")
        if not (0.0 <= self.min_exposure <= self.max_exposure):
            raise LabConfigError(
                f"vol_target exposure bounds invalid: min {self.min_exposure}, max {self.max_exposure}"
            )
        if self.max_exposure > 1.0:
            raise LabConfigError(
                f"vol_target.max_exposure {self.max_exposure} > 1.0 would require leverage; "
                f"leverage is not supported."
            )
        if self.smoothing_days < 1:
            raise LabConfigError(f"vol_target.smoothing_days must be >= 1, got {self.smoothing_days}")
        if self.evaluate not in ("daily", "weekly", "month_end"):
            raise LabConfigError(f"vol_target.evaluate must be daily|weekly|month_end, got {self.evaluate!r}")


@dataclass
class DrawdownThrottleConfig:
    """Candidate B: stateful drawdown throttle with hysteresis.

    `levels` maps drawdown thresholds (negative floats, deepest last) to
    exposure multipliers, e.g. [(-0.08, 0.75), (-0.15, 0.50)]: exposure drops
    to 0.75 once portfolio drawdown (computed from equity through the PRIOR
    completed session) breaches -8%, to 0.50 once it breaches -15%. Exposure
    is restored one level at a time only after drawdown recovers above the
    breached threshold + `recovery_margin` AND `cooldown_days` sessions have
    passed since the last state change (hysteresis against whipsaw)."""

    enabled: bool = False
    levels: tuple = ((-0.08, 0.75), (-0.15, 0.50))
    recovery_margin: float = 0.03
    cooldown_days: int = 10
    evaluate: str = "daily"

    def validate(self) -> None:
        if not self.levels:
            raise LabConfigError("drawdown_throttle.levels must not be empty")
        prev_dd = 0.0
        for dd, exp in self.levels:
            if dd >= 0:
                raise LabConfigError(f"drawdown_throttle level threshold must be negative, got {dd}")
            if dd >= prev_dd and prev_dd != 0.0:
                raise LabConfigError("drawdown_throttle.levels must be ordered shallowest to deepest")
            if not (0.0 < exp <= 1.0):
                raise LabConfigError(f"drawdown_throttle exposure must be in (0, 1], got {exp}")
            prev_dd = dd
        if self.recovery_margin < 0:
            raise LabConfigError(f"drawdown_throttle.recovery_margin must be >= 0, got {self.recovery_margin}")
        if self.cooldown_days < 0:
            raise LabConfigError(f"drawdown_throttle.cooldown_days must be >= 0, got {self.cooldown_days}")
        if self.evaluate not in ("daily", "weekly", "month_end"):
            raise LabConfigError(f"drawdown_throttle.evaluate must be daily|weekly|month_end")


@dataclass
class DynamicAllocationConfig:
    """Candidate C: constrained dynamic sleeve allocation. Weights are
    computed from TRAILING sleeve equity data only, clipped to per-sleeve
    bounds, renormalized to 1.0, and applied at most every
    `min_interval_months` months when the drift exceeds `no_trade_threshold`."""

    enabled: bool = False
    method: str = "inverse_vol"      # "inverse_vol" | "erc" | "score"
    lookback_days: int = 126
    min_interval_months: int = 3
    no_trade_threshold: float = 0.05  # skip if max |target - drifted| below this
    bounds: dict = field(default_factory=lambda: {
        "momentum": (0.40, 0.75),
        "sector_rotation": (0.15, 0.50),
        "regime_switch": (0.05, 0.25),
    })

    def validate(self) -> None:
        if self.method not in ("inverse_vol", "erc", "score"):
            raise LabConfigError(f"allocation.method must be inverse_vol|erc|score, got {self.method!r}")
        if self.lookback_days < 20:
            raise LabConfigError(f"allocation.lookback_days must be >= 20, got {self.lookback_days}")
        if self.min_interval_months < 1:
            raise LabConfigError("allocation.min_interval_months must be >= 1")
        if not (0.0 <= self.no_trade_threshold < 1.0):
            raise LabConfigError("allocation.no_trade_threshold must be in [0, 1)")
        lo_sum = sum(lo for lo, _hi in self.bounds.values())
        hi_sum = sum(hi for _lo, hi in self.bounds.values())
        if lo_sum > 1.0 + 1e-9 or hi_sum < 1.0 - 1e-9:
            raise LabConfigError(
                f"allocation.bounds infeasible: lower bounds sum to {lo_sum:.2f}, upper to {hi_sum:.2f}; "
                f"weights could never sum to 1.0."
            )
        for name, (lo, hi) in self.bounds.items():
            if not (0.0 <= lo <= hi <= 1.0):
                raise LabConfigError(f"allocation.bounds[{name!r}] invalid: ({lo}, {hi})")


@dataclass
class NoTradeBandConfig:
    """Candidate D: no-trade bands and turnover controls, applied to orders
    at queue time. Suppressed orders are logged with reason codes, never
    silently dropped. Exit orders (target weight 0) are NEVER suppressed --
    suppressing an exit would leave an unmanaged position."""

    enabled: bool = False
    min_weight_change: float = 0.01
    min_notional: float = 25.0
    monthly_turnover_budget_pct: float | None = None  # gross executed notional / equity per calendar month

    def validate(self) -> None:
        if self.min_weight_change < 0 or self.min_notional < 0:
            raise LabConfigError("no_trade_band thresholds must be >= 0")
        if self.monthly_turnover_budget_pct is not None and self.monthly_turnover_budget_pct <= 0:
            raise LabConfigError("no_trade_band.monthly_turnover_budget_pct must be > 0 when set")


@dataclass
class CashYieldConfig:
    """Candidate E: lagged cash yield accrued daily on each sleeve's actual
    cash balance. Default keeps the existing 0%-cash baseline. A dated series
    (CSV with columns date,annual_rate) uses the most recent rate dated at or
    before the PRIOR session -- future rates are structurally unreachable."""

    enabled: bool = False
    annual_rate: float = 0.0
    rate_series_csv: str | None = None

    def validate(self) -> None:
        if self.annual_rate < 0:
            raise LabConfigError(f"cash_yield.annual_rate must be >= 0, got {self.annual_rate}")
        if self.annual_rate > 0.15:
            raise LabConfigError(f"cash_yield.annual_rate {self.annual_rate} is implausibly high (> 15%)")


@dataclass
class ConcentrationConfig:
    """Candidate F: cap on any single ticker's AGGREGATE weight across all
    sleeves (sleeves can silently double up on the same name). Excess weight
    is scaled down pro-rata across the sleeves holding it, with the freed
    weight left in cash -- never redirected into new names the strategies
    didn't pick."""

    enabled: bool = False
    max_ticker_weight: float = 0.25

    def validate(self) -> None:
        if not (0.0 < self.max_ticker_weight <= 1.0):
            raise LabConfigError(
                f"concentration.max_ticker_weight must be in (0, 1], got {self.max_ticker_weight}"
            )


@dataclass
class LabConfig:
    """Top-level lab configuration. Default construction = every enhancement
    off = behavior-identical to `--strategy portfolio` (proven by the
    baseline equivalence tests)."""

    trace: bool = True  # observational only; proven behavior-neutral either way
    vol_target: VolTargetConfig = field(default_factory=VolTargetConfig)
    drawdown_throttle: DrawdownThrottleConfig = field(default_factory=DrawdownThrottleConfig)
    allocation: DynamicAllocationConfig = field(default_factory=DynamicAllocationConfig)
    no_trade_band: NoTradeBandConfig = field(default_factory=NoTradeBandConfig)
    cash_yield: CashYieldConfig = field(default_factory=CashYieldConfig)
    concentration: ConcentrationConfig = field(default_factory=ConcentrationConfig)
    # Per-sleeve strategy variant overrides: {sleeve_name: variant_label}.
    # Labels resolve through the lab variant registry (src/lab/variants.py);
    # empty dict = shipped baseline strategies.
    sleeve_variants: dict = field(default_factory=dict)

    def validate(self) -> None:
        self.vol_target.validate()
        self.drawdown_throttle.validate()
        self.allocation.validate()
        self.no_trade_band.validate()
        self.cash_yield.validate()
        self.concentration.validate()

    # --- Introspection ---------------------------------------------------------

    def enabled_features(self) -> list[str]:
        out = []
        for name in ("vol_target", "drawdown_throttle", "allocation", "no_trade_band",
                     "cash_yield", "concentration"):
            if getattr(self, name).enabled:
                out.append(name)
        if self.sleeve_variants:
            out.append("sleeve_variants")
        return out

    def any_behavior_change(self) -> bool:
        """True if any setting could change trades/accounting relative to the
        production portfolio mode. `trace` is deliberately excluded -- it is
        observational and proven behavior-neutral."""
        return bool(self.enabled_features())

    # --- Serialization ---------------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        # tuples -> lists for JSON round-tripping
        d["drawdown_throttle"]["levels"] = [list(pair) for pair in self.drawdown_throttle.levels]
        d["allocation"]["bounds"] = {k: list(v) for k, v in self.allocation.bounds.items()}
        return d

    @classmethod
    def from_dict(cls, raw: dict) -> "LabConfig":
        """Strict deserialization: any unknown key at any level raises
        LabConfigError (fail closed -- a typo'd flag must never silently
        no-op), and the result is validated before being returned."""
        if not isinstance(raw, dict):
            raise LabConfigError(f"Lab config must be a JSON object, got {type(raw).__name__}")

        section_types = {
            "vol_target": VolTargetConfig,
            "drawdown_throttle": DrawdownThrottleConfig,
            "allocation": DynamicAllocationConfig,
            "no_trade_band": NoTradeBandConfig,
            "cash_yield": CashYieldConfig,
            "concentration": ConcentrationConfig,
        }
        known_top = set(section_types) | {"trace", "sleeve_variants"}
        unknown = set(raw) - known_top
        if unknown:
            raise LabConfigError(f"Unknown lab config key(s): {sorted(unknown)}. Known: {sorted(known_top)}")

        kwargs: dict = {}
        if "trace" in raw:
            kwargs["trace"] = bool(raw["trace"])
        if "sleeve_variants" in raw:
            sv = raw["sleeve_variants"]
            if not isinstance(sv, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in sv.items()
            ):
                raise LabConfigError("sleeve_variants must map sleeve name -> variant label (strings)")
            kwargs["sleeve_variants"] = dict(sv)

        for section, typ in section_types.items():
            if section not in raw:
                continue
            body = raw[section]
            if not isinstance(body, dict):
                raise LabConfigError(f"Lab config section {section!r} must be an object")
            allowed = {f.name for f in fields(typ)}
            unknown = set(body) - allowed
            if unknown:
                raise LabConfigError(
                    f"Unknown key(s) in lab config section {section!r}: {sorted(unknown)}. "
                    f"Known: {sorted(allowed)}"
                )
            body = dict(body)
            if section == "drawdown_throttle" and "levels" in body:
                body["levels"] = tuple(tuple(pair) for pair in body["levels"])
            if section == "allocation" and "bounds" in body:
                body["bounds"] = {k: tuple(v) for k, v in body["bounds"].items()}
            kwargs[section] = typ(**body)

        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    def config_hash(self) -> str:
        """sha256 over the canonical JSON encoding -- ties every artifact to
        the exact configuration that produced it."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
