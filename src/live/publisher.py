"""Authentic Phase-4 strategy target publisher.

The publisher instantiates the registered strategy classes, calls their own
``prepare`` and ``initial_events`` implementations, and aggregates the full
sleeve targets through Phase 3's reviewed aggregation function.  It does not
restate any momentum, sector-ranking, or regime formula.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .. import config, data
from ..market_view import MarketDataView
from ..portfolio import DEFAULT_PORTFOLIO_WEIGHTS
from ..tournament import STRATEGY_REGISTRY
from .alpaca_paper import AlpacaAsset
from .deployment import (
    DeploymentConfig,
    SleeveTargetSnapshot,
    aggregate_sleeves,
    canonical_hash,
)
from .market_data import MarketCalendarDay
from .models import AccountSnapshot, Position, ZERO, ensure_aware, json_safe
from .phase4_models import (
    Phase4Error,
    Phase4Policy,
    PublishedTargetSnapshot,
    SnapshotSigner,
    account_fingerprint,
    content_hash,
)


SURVIVORSHIP_DISCLOSURE = (
    "The momentum stock universe is the repository's current hand-picked large-cap snapshot, "
    "not point-in-time constituents. Results and selections are survivorship-biased."
)
_EXCHANGES = {"NASDAQ", "NYSE", "ARCA", "AMEX", "BATS", "NYSEARCA", "NYSEAMERICAN"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class HistoricalDataBundle:
    frames: dict[str, pd.DataFrame]
    calendar: pd.DatetimeIndex
    retrieved_at: datetime
    source_id: str
    input_file_hashes: dict[str, str]
    adjusted: bool = True
    interval: str = "1d"


class HistoricalDataSource(Protocol):
    def load(self, symbols: tuple[str, ...], decision_session: date) -> HistoricalDataBundle: ...


class AssetSource(Protocol):
    def get_asset(self, symbol: str) -> AlpacaAsset: ...


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ResearchHistoricalDataSource:
    """Existing yfinance/cache/data-validation machinery, fail-closed for publication."""

    def __init__(self, *, cache_dir: str = config.CACHE_DIR, clock=lambda: datetime.now(timezone.utc)):
        self.cache_dir = cache_dir
        self.clock = clock

    def load(self, symbols: tuple[str, ...], decision_session: date) -> HistoricalDataBundle:
        end = pd.Timestamp(decision_session)
        start = end - pd.Timedelta(days=config.MOMENTUM_WARMUP_CALENDAR_DAYS)
        frames, _dropped = data.get_price_data(
            list(symbols), start, end,
            warmup_calendar_days=0,
            hard_fail_on_missing=True,
            cache_dir=self.cache_dir,
            # A same-day cache may contain an unfinished intraday daily bar.
            # Publication therefore refreshes every required symbol after the
            # authenticated close and fails if that refresh cannot complete.
            force_refresh=True,
        )
        if config.BENCHMARK_TICKER not in frames:
            frames[config.BENCHMARK_TICKER] = data.get_benchmark_data(
                start, end, cache_dir=self.cache_dir
            )
        spy = frames[config.BENCHMARK_TICKER]
        calendar = data.build_canonical_calendar(spy, start, end)
        # Phase 4 validates finality against the authenticated official close,
        # rather than the research helper's conservative "always drop today"
        # rule. After the close, dropping today's bar would make a same-evening
        # month-end publication impossible; a missing/stale source bar still
        # fails the exact through-decision checks below.
        hashes: dict[str, str] = {}
        cache_path = Path(self.cache_dir)
        for symbol in symbols:
            for suffix in (".csv", ".meta.json"):
                path = cache_path / f"{symbol}{suffix}"
                if path.is_file() and not path.is_symlink():
                    hashes[str(path.resolve())] = _sha256_file(path)
        return HistoricalDataBundle(
            frames=frames,
            calendar=calendar,
            retrieved_at=ensure_aware(self.clock(), "historical data retrieval time"),
            source_id="yfinance-adjusted-daily-via-repository-cache",
            input_file_hashes=hashes,
            adjusted=config.YFINANCE_AUTO_ADJUST,
            interval=config.YFINANCE_INTERVAL,
        )


def git_state(repo_root: str | Path) -> tuple[str | None, bool, tuple[str, ...]]:
    root = str(Path(repo_root).resolve())
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
            timeout=10, check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=root, capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Phase4Error("Cannot inspect publisher git state") from exc
    if sha.returncode != 0 or status.returncode != 0:
        raise Phase4Error("Cannot inspect publisher git state")
    entries = tuple(line.rstrip() for line in status.stdout.splitlines() if line.strip())
    return sha.stdout.strip() or None, bool(entries), entries


def git_dirty_file_hashes(repo_root: str | Path) -> dict[str, dict[str, Any]]:
    """Freeze exact bytes for every modified, staged, deleted, or untracked file."""
    root = Path(repo_root).resolve()
    commands = (
        ["git", "diff", "--name-only", "-z"],
        ["git", "diff", "--cached", "--name-only", "-z"],
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
    )
    names: set[str] = set()
    try:
        for command in commands:
            result = subprocess.run(
                command, cwd=str(root), capture_output=True, timeout=10, check=False,
            )
            if result.returncode != 0:
                raise Phase4Error("Cannot enumerate dirty publisher files")
            names.update(
                value.decode("utf-8", "strict")
                for value in result.stdout.split(b"\0") if value
            )
    except (OSError, UnicodeError, subprocess.SubprocessError) as exc:
        raise Phase4Error("Cannot enumerate dirty publisher files") from exc
    if len(names) > 2000:
        raise Phase4Error("Dirty worktree contains too many files to authenticate safely")
    frozen: dict[str, dict[str, Any]] = {}
    for name in sorted(names):
        candidate = root / name
        try:
            resolved = candidate.resolve(strict=False)
            resolved.relative_to(root)
        except (OSError, ValueError) as exc:
            raise Phase4Error("Dirty worktree path escapes the repository") from exc
        if candidate.is_symlink():
            raise Phase4Error("Dirty worktree contains an unauthenticated symbolic link")
        if not candidate.exists():
            frozen[name] = {"state": "deleted"}
            continue
        if not candidate.is_file():
            raise Phase4Error("Dirty worktree contains a non-regular file")
        size = candidate.stat().st_size
        if size > 10_000_000:
            raise Phase4Error("Dirty worktree file exceeds the 10 MB authentication limit")
        frozen[name] = {"state": "file", "size": size, "sha256": _sha256_file(candidate)}
    return frozen


def _asset_payload(asset: AlpacaAsset) -> dict[str, Any]:
    return {
        "asset_id": asset.asset_id,
        "symbol": asset.symbol,
        "asset_class": asset.asset_class,
        "status": asset.status,
        "tradable": asset.tradable,
        "exchange": asset.exchange,
        "fractionable": asset.fractionable,
        "shortable": asset.shortable,
        "easy_to_borrow": asset.easy_to_borrow,
        "marginable": asset.marginable,
        "name": asset.name,
    }


def _company_key(name: str) -> str:
    normalized = re.sub(r"\b(CLASS|COMMON|STOCK|SHARES?|INCORPORATED|INC|CORP|CORPORATION)\b", "", name.upper())
    normalized = re.sub(r"\b[A-Z]\b$", "", normalized)
    return re.sub(r"[^A-Z0-9]", "", normalized)


class StrategyTargetPublisher:
    def __init__(
        self,
        deployment: DeploymentConfig,
        policy: Phase4Policy,
        *,
        repo_root: str | Path,
        signer: SnapshotSigner | None = None,
        clock=lambda: datetime.now(timezone.utc),
    ) -> None:
        policy.validate_deployment(deployment)
        self.deployment = deployment
        self.policy = policy
        self.repo_root = Path(repo_root).resolve()
        self.signer = signer
        self.clock = clock
        if policy.require_signing and signer is None:
            raise Phase4Error("Phase-4 policy requires an operator signing key")
        if signer is not None and signer.key_id != policy.signing_key_id:
            raise Phase4Error("Configured signer key ID does not match Phase-4 policy")

    @staticmethod
    def required_tradable_symbols() -> tuple[str, ...]:
        return tuple(sorted(set(config.MEAN_REVERSION_UNIVERSE) | set(config.SECTOR_ETFS)))

    @classmethod
    def required_data_symbols(cls) -> tuple[str, ...]:
        return tuple(sorted(set(cls.required_tradable_symbols()) | {config.BENCHMARK_TICKER}))

    def publish(
        self,
        *,
        decision_day: MarketCalendarDay,
        execution_day: MarketCalendarDay,
        account: AccountSnapshot,
        positions: list[Position],
        assets: dict[str, AlpacaAsset],
        market_data: HistoricalDataBundle,
        input_file_hashes: dict[str, str] | None = None,
        previous_snapshot: PublishedTargetSnapshot | None = None,
    ) -> PublishedTargetSnapshot:
        now = ensure_aware(self.clock(), "publisher time")
        self._validate_sessions(decision_day, execution_day, now)
        self._validate_account(account, positions)
        git_sha, dirty, dirty_entries = git_state(self.repo_root)
        if dirty and self.policy.dirty_worktree_policy == "reject":
            raise Phase4Error("Dirty worktree policy forbids target publication")
        dirty_file_hashes = git_dirty_file_hashes(self.repo_root) if dirty else {}
        asset_rows = self._validate_assets(assets, previous_snapshot)
        self._validate_market_data(market_data, decision_day, now)

        sleeve_targets, strategy_parameters = self._run_registered_strategies(
            market_data, decision_day.trading_date, account.equity
        )
        signal_at = decision_day.close_at + timedelta(microseconds=1)
        target_artifact = SleeveTargetSnapshot.from_payload(
            {
                "schema_version": 1,
                "signal_at": signal_at.isoformat(),
                "target_version": f"phase4-{decision_day.trading_date.isoformat()}",
                "sleeves": sleeve_targets,
            },
            self.deployment,
        )
        aggregated = aggregate_sleeves(self.deployment, target_artifact)
        estimated_deltas = self._estimated_deltas(
            aggregated, positions, account.equity, market_data, decision_day.trading_date
        )
        all_input_hashes = dict(sorted((input_file_hashes or {}).items()))
        for path, digest in market_data.input_file_hashes.items():
            if path in all_input_hashes and all_input_hashes[path] != digest:
                raise Phase4Error("Publisher input path has conflicting content hashes")
            all_input_hashes[path] = digest
        for path, digest in all_input_hashes.items():
            if (
                not isinstance(path, str) or not path.strip() or "\x00" in path
                or not isinstance(digest, str) or not _SHA256.fullmatch(digest)
            ):
                raise Phase4Error("Publisher input hashes must be named lowercase SHA-256 values")
        all_input_hashes = dict(sorted(all_input_hashes.items()))
        universe_payload = {
            "tradable_symbols": list(self.required_tradable_symbols()),
            "signal_only_symbols": [config.BENCHMARK_TICKER],
            "assets": asset_rows,
            "survivorship_biased": True,
            "survivorship_disclosure": SURVIVORSHIP_DISCLOSURE,
        }
        universe_hash = content_hash(universe_payload)
        content = json_safe({
            "strategy_configuration_version": "fixed-60-35-5-v1",
            "phase4_policy_hash": content_hash(self.policy.to_payload()),
            "deployment_hash": canonical_hash(self.deployment.to_payload()),
            "git_sha": git_sha,
            "dirty_worktree": dirty,
            "dirty_worktree_entries": list(dirty_entries),
            "dirty_worktree_file_hashes": dirty_file_hashes,
            "creation_timestamp": now,
            "decision_session": decision_day.trading_date.isoformat(),
            "decision_close_utc": decision_day.close_at,
            "decision_exchange_timezone": "America/New_York",
            "data_cutoff": decision_day.trading_date.isoformat(),
            "expected_execution_session": execution_day.trading_date.isoformat(),
            "expected_execution_open_utc": execution_day.open_at,
            "account_id_fingerprint": account_fingerprint(
                account.account_id, self.policy.system_id
            ),
            "account_equity_used_for_sizing": account.equity,
            "sleeve_allocations": dict(DEFAULT_PORTFOLIO_WEIGHTS),
            "sleeve_level_targets": sleeve_targets,
            "aggregated_ticker_targets": aggregated,
            "current_broker_positions": [
                {"symbol": p.symbol, "quantity": p.quantity, "avg_price": p.avg_price,
                 "market_price": p.market_price}
                for p in sorted(positions, key=lambda row: row.symbol)
            ],
            "required_target_deltas": estimated_deltas,
            "complete_managed_universe": universe_payload,
            "market_data_provenance": {
                "source_id": market_data.source_id,
                "retrieved_at": market_data.retrieved_at,
                "interval": market_data.interval,
                "bounded_through": decision_day.trading_date.isoformat(),
                "reference_calendar": config.BENCHMARK_TICKER,
            },
            "price_semantics": {
                "strategy_input": "adjusted daily close; research signal only",
                "estimated_delta_price": "adjusted decision-session close; non-executable estimate",
                "execution_sizing": "fresh paper-broker bid/ask in Phase-3 preview",
                "order_type": "regular-hours market/day; not a next-close backtest fill",
            },
            "strategy_parameters": strategy_parameters,
            "universe_snapshot_hash": universe_hash,
            "input_file_hashes": all_input_hashes,
            "publisher_identity": self.policy.publisher_identity,
            "operation_mode": self.policy.mode.value,
            "expiration_time": now + timedelta(seconds=self.policy.snapshot_ttl_seconds),
            "target_snapshot_schema_version": 1,
        })
        snapshot = PublishedTargetSnapshot.create(content, signer=self.signer)
        snapshot.verify(self.policy, self.signer, now=now)
        return snapshot

    def to_execution_targets(self, snapshot: PublishedTargetSnapshot) -> SleeveTargetSnapshot:
        snapshot.verify(self.policy, self.signer, now=ensure_aware(self.clock(), "target conversion"))
        expected_fingerprint = account_fingerprint(
            self.deployment.account_id, self.policy.system_id
        )
        if snapshot.content.get("account_id_fingerprint") != expected_fingerprint:
            raise Phase4Error("Target snapshot belongs to a different paper account")
        if snapshot.content.get("strategy_configuration_version") != "fixed-60-35-5-v1":
            raise Phase4Error("Target snapshot strategy configuration has drifted")
        if snapshot.content.get("phase4_policy_hash") != content_hash(self.policy.to_payload()):
            raise Phase4Error("Target snapshot Phase-4 policy has drifted")
        if snapshot.content.get("deployment_hash") != canonical_hash(self.deployment.to_payload()):
            raise Phase4Error("Target snapshot deployment has drifted")
        if snapshot.content.get("operation_mode") != self.policy.mode.value:
            raise Phase4Error("Target snapshot operation mode has drifted")
        if snapshot.content.get("sleeve_allocations") != json_safe(
            dict(DEFAULT_PORTFOLIO_WEIGHTS)
        ):
            raise Phase4Error("Target snapshot sleeve allocations have drifted")
        if snapshot.content.get("universe_snapshot_hash") != content_hash(
            snapshot.content["complete_managed_universe"]
        ):
            raise Phase4Error("Target snapshot universe has drifted")
        universe = snapshot.content["complete_managed_universe"]
        if (
            universe.get("tradable_symbols") != list(self.required_tradable_symbols())
            or universe.get("signal_only_symbols") != [config.BENCHMARK_TICKER]
        ):
            raise Phase4Error("Target snapshot managed universe differs from registered universe")
        expected_parameters: dict[str, Any] = {}
        for name in ("momentum", "sector_rotation", "regime_switch"):
            spec = STRATEGY_REGISTRY[name]
            strategy = (
                spec.factory(universe=list(config.MEAN_REVERSION_UNIVERSE))
                if name == "momentum" else spec.factory()
            )
            expected_parameters[name] = strategy.describe()["params"]
        if snapshot.content.get("strategy_parameters") != json_safe(expected_parameters):
            raise Phase4Error("Target snapshot strategy parameters have drifted")
        sha, dirty, entries = git_state(self.repo_root)
        if (
            snapshot.content.get("git_sha") != sha
            or snapshot.content.get("dirty_worktree") != dirty
            or snapshot.content.get("dirty_worktree_entries") != list(entries)
        ):
            raise Phase4Error("Publisher code state changed after target publication")
        current_dirty_hashes = git_dirty_file_hashes(self.repo_root) if dirty else {}
        if snapshot.content.get("dirty_worktree_file_hashes") != current_dirty_hashes:
            raise Phase4Error("Dirty publisher file content changed after target publication")
        for raw_path, expected in snapshot.content.get("input_file_hashes", {}).items():
            path = Path(raw_path)
            if not path.is_absolute():
                continue
            if not path.is_file() or path.is_symlink() or _sha256_file(path) != expected:
                raise Phase4Error(f"Publisher input changed after target publication: {path.name}")
        return SleeveTargetSnapshot.from_payload(
            {
                "schema_version": 1,
                "signal_at": snapshot.content["decision_close_utc"],
                "target_version": snapshot.snapshot_id,
                "sleeves": snapshot.content["sleeve_level_targets"],
            },
            self.deployment,
        )

    def validate_assets_for_execution(
        self,
        snapshot: PublishedTargetSnapshot,
        current_assets: dict[str, AlpacaAsset],
    ) -> None:
        """Detect ticker/corporate-action/metadata drift after publication."""
        rows = self._validate_assets(current_assets, snapshot)
        frozen = snapshot.content["complete_managed_universe"]["assets"]
        if rows != frozen:
            raise Phase4Error("Broker asset metadata changed after target publication")

    def _validate_sessions(
        self, decision: MarketCalendarDay, execution: MarketCalendarDay, now: datetime
    ) -> None:
        if now <= decision.close_at:
            raise Phase4Error("Decision session official close has not completed")
        if execution.trading_date <= decision.trading_date:
            raise Phase4Error("Expected execution session must follow the decision session")
        if (decision.trading_date.year, decision.trading_date.month) == (
            execution.trading_date.year, execution.trading_date.month
        ):
            raise Phase4Error("Decision session is not the final exchange session of its month")

    def _validate_account(self, account: AccountSnapshot, positions: list[Position]) -> None:
        if account.account_id != self.deployment.account_id:
            raise Phase4Error("Authenticated paper account does not match deployment")
        if account.equity <= ZERO or account.status != "ACTIVE" or account.currency != "USD":
            raise Phase4Error("Paper account is not valid for target publication")
        managed = set(self.required_tradable_symbols())
        if set(self.deployment.managed_symbols) != managed:
            raise Phase4Error("Deployment managed_symbols must exactly match the Phase-4 trade universe")
        seen: set[str] = set()
        for position in positions:
            if position.symbol in seen:
                raise Phase4Error(f"Duplicate broker position for {position.symbol}")
            seen.add(position.symbol)
            if position.quantity < ZERO or position.quantity != position.quantity.to_integral_value():
                raise Phase4Error("Short and fractional positions are forbidden")
            if position.quantity and position.symbol not in managed:
                raise Phase4Error(f"Unexpected broker position outside managed universe: {position.symbol}")

    def _validate_assets(
        self,
        assets: dict[str, AlpacaAsset],
        previous_snapshot: PublishedTargetSnapshot | None,
    ) -> list[dict[str, Any]]:
        required = set(self.required_data_symbols())
        if set(assets) != required:
            raise Phase4Error(
                f"Broker asset set mismatch; missing={sorted(required-set(assets))}, "
                f"unknown={sorted(set(assets)-required)}"
            )
        rows: list[dict[str, Any]] = []
        asset_ids: set[str] = set()
        company_keys: dict[str, str] = {}
        previous_assets: dict[str, dict] = {}
        if previous_snapshot is not None:
            for row in previous_snapshot.content["complete_managed_universe"]["assets"]:
                previous_assets[str(row["symbol"])] = row
        for symbol in sorted(required):
            asset = assets[symbol]
            if asset.symbol != symbol:
                raise Phase4Error(f"Broker reports ticker change for {symbol}: {asset.symbol}")
            if not asset.asset_id or asset.asset_id in asset_ids:
                raise Phase4Error("Broker asset identifiers are missing or duplicated")
            asset_ids.add(asset.asset_id)
            if asset.asset_class != "us_equity" or asset.status != "active" or not asset.tradable:
                raise Phase4Error(f"{symbol} is not an active tradable US equity")
            if asset.exchange not in _EXCHANGES:
                raise Phase4Error(f"{symbol} has unsupported or missing exchange {asset.exchange!r}")
            if not asset.name:
                raise Phase4Error(f"{symbol} is missing broker security-name metadata")
            key = _company_key(asset.name)
            if key and key in company_keys and company_keys[key] != symbol:
                raise Phase4Error(
                    f"Duplicate share classes require explicit universe resolution: "
                    f"{company_keys[key]} and {symbol}"
                )
            company_keys[key] = symbol
            previous = previous_assets.get(symbol)
            if previous is not None and previous.get("asset_id") != asset.asset_id:
                raise Phase4Error(f"Corporate-action/asset identity changed for {symbol}")
            rows.append(_asset_payload(asset))
        return rows

    def _validate_market_data(
        self, bundle: HistoricalDataBundle, decision_day: MarketCalendarDay, now: datetime
    ) -> None:
        retrieved = ensure_aware(bundle.retrieved_at, "market-data retrieval time")
        if retrieved < decision_day.close_at or retrieved > now + timedelta(seconds=2):
            raise Phase4Error(
                "Historical data retrieval must follow the official close and not be future-dated"
            )
        decision = pd.Timestamp(decision_day.trading_date)
        required = set(self.required_data_symbols())
        if set(bundle.frames) != required:
            raise Phase4Error(
                f"Historical data set mismatch; missing={sorted(required-set(bundle.frames))}, "
                f"unknown={sorted(set(bundle.frames)-required)}"
            )
        if not bundle.adjusted or bundle.interval != "1d":
            raise Phase4Error("Publisher requires adjusted daily research data")
        if (
            bundle.calendar.empty
            or not bundle.calendar.is_monotonic_increasing
            or bundle.calendar.has_duplicates
            or any(value != value.normalize() for value in bundle.calendar)
        ):
            raise Phase4Error("SPY reference calendar is empty, malformed, or duplicated")
        minimum_history = max(
            config.MOMENTUM_TREND_SMA_PERIOD, config.REGIME_SMA_PERIOD
        )
        if len(bundle.calendar) < minimum_history:
            raise Phase4Error(
                f"Publisher requires at least {minimum_history} finalized sessions"
            )
        if bundle.calendar[-1].normalize() != decision:
            raise Phase4Error("SPY reference calendar is incomplete through the decision session")
        if decision not in bundle.calendar:
            raise Phase4Error("Decision session is absent from the reference calendar")
        active_calendar = bundle.calendar[-200:]
        for symbol in sorted(required):
            frame = bundle.frames[symbol]
            if frame.empty or not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
                raise Phase4Error(f"{symbol} history is empty, unsorted, or duplicated")
            if frame.index.max().normalize() != decision or any(frame.index > decision):
                raise Phase4Error(f"{symbol} is stale or extends beyond the decision session")
            missing = active_calendar.difference(frame.index)
            if len(missing):
                raise Phase4Error(f"{symbol} has missing finalized sessions, including {missing[0].date()}")
            required_columns = ["Open", "High", "Low", "Close", "Volume"]
            if any(column not in frame for column in required_columns):
                raise Phase4Error(f"{symbol} is missing OHLCV columns")
            indicator_window = frame.loc[active_calendar, required_columns]
            values = np.asarray(indicator_window, dtype=float)
            if not np.isfinite(values).all():
                raise Phase4Error(f"{symbol} has non-finite values in the indicator window")
            if (indicator_window[["Open", "High", "Low", "Close"]] <= 0).any().any():
                raise Phase4Error(f"{symbol} has nonpositive prices in the indicator window")
            if (indicator_window["Volume"].iloc[:-1] <= 0).any():
                raise Phase4Error(f"{symbol} has nonpositive volume in the indicator window")
            if (
                (indicator_window["High"] < indicator_window[["Open", "Close", "Low"]].max(axis=1)).any()
                or (indicator_window["Low"] > indicator_window[["Open", "Close", "High"]].min(axis=1)).any()
            ):
                raise Phase4Error(f"{symbol} has impossible OHLC geometry")
            tail = indicator_window.loc[decision]
            if len(frame) >= 2:
                prior = frame.iloc[-2]
                if all(
                    float(tail[c]) == float(prior[c])
                    for c in ("Open", "High", "Low", "Close", "Volume")
                ):
                    raise Phase4Error(f"{symbol} appears to contain a carried-forward final bar")
            if tail["Volume"] <= 0:
                raise Phase4Error(
                    f"{symbol} has a carried-forward or nonpositive-volume final bar"
                )

    def _run_registered_strategies(
        self, bundle: HistoricalDataBundle, decision_date: date, equity: Decimal
    ) -> tuple[dict[str, dict[str, Decimal]], dict[str, Any]]:
        decision = pd.Timestamp(decision_date)
        calendar = bundle.calendar[bundle.calendar <= decision]
        if calendar.empty or calendar[-1] != decision:
            raise Phase4Error("Decision date is not the final visible research session")
        results: dict[str, dict[str, Decimal]] = {}
        parameters: dict[str, Any] = {}
        allocations = dict(DEFAULT_PORTFOLIO_WEIGHTS)
        for name in ("momentum", "sector_rotation", "regime_switch"):
            spec = STRATEGY_REGISTRY[name]
            strategy = (
                spec.factory(universe=list(config.MEAN_REVERSION_UNIVERSE))
                if name == "momentum" else spec.factory()
            )
            strategy.reset()
            expected_universe = tuple(strategy.universe)
            relevant = set(expected_universe) | set(strategy.signal_tickers)
            frames = {symbol: bundle.frames[symbol].copy() for symbol in relevant}
            enriched = strategy.prepare(frames, pd.DatetimeIndex([decision]), decision)
            if strategy.dropped_tickers or tuple(strategy.universe) != expected_universe:
                raise Phase4Error(
                    f"{name} dropped or mutated registered-universe members during preparation"
                )
            indicator_columns = {
                "momentum": ("Close", "TrendSMA", "MomentumReturn"),
                "sector_rotation": ("Close", "TrailingReturn_3M"),
                "regime_switch": ("Close", "TrailingReturn_3M"),
            }[name]
            for symbol in expected_universe:
                if symbol not in enriched or decision not in enriched[symbol].index:
                    raise Phase4Error(f"{name} lacks decision-session indicators for {symbol}")
                values = enriched[symbol].loc[decision, list(indicator_columns)]
                if not np.isfinite(np.asarray(values, dtype=float)).all():
                    raise Phase4Error(f"{name} has non-finite decision indicators for {symbol}")
            if name == "regime_switch":
                regime_symbol = strategy.regime_ticker
                values = enriched[regime_symbol].loc[decision, ["Close", "RegimeSMA"]]
                if not np.isfinite(np.asarray(values, dtype=float)).all():
                    raise Phase4Error("regime_switch has a non-finite decision regime signal")
            view = MarketDataView(enriched, decision, calendar)
            events = strategy.initial_events(view, float(equity) * allocations[name])
            universe_targets = {symbol: ZERO for symbol in expected_universe}
            seen: set[str] = set()
            for event in events:
                if event.strategy != name or event.signal_date.normalize() != decision:
                    raise Phase4Error(f"{name} emitted an event for the wrong strategy/session")
                if event.ticker not in universe_targets or event.ticker in seen:
                    raise Phase4Error(f"{name} emitted an invalid or duplicate target")
                if not math.isfinite(event.target_weight) or not 0 <= event.target_weight <= 1:
                    raise Phase4Error(f"{name} emitted an invalid target weight")
                seen.add(event.ticker)
                universe_targets[event.ticker] = Decimal(str(event.target_weight))
            if name == "momentum":
                nonzero = [value for value in universe_targets.values() if value > ZERO]
                if len(nonzero) > config.MOMENTUM_TOP_K or any(
                    value != Decimal("0.2") for value in nonzero
                ):
                    raise Phase4Error("Registered momentum output violates fixed Phase-4 contract")
            else:
                nonzero = [value for value in universe_targets.values() if value > ZERO]
                if name == "sector_rotation" and (
                    len(nonzero) != config.SECTOR_TOP_K
                    or any(value != Decimal("0.5") for value in nonzero)
                ):
                    raise Phase4Error("Registered sector-rotation output violates fixed contract")
                if name == "regime_switch" and nonzero and (
                    len(nonzero) != config.SECTOR_TOP_K
                    or any(value != Decimal("0.5") for value in nonzero)
                ):
                    raise Phase4Error("Registered regime-switch output violates fixed contract")
            results[name] = dict(sorted(universe_targets.items()))
            parameters[name] = strategy.describe()["params"]
        return results, parameters

    def _estimated_deltas(
        self,
        aggregated: dict[str, Decimal],
        positions: list[Position],
        equity: Decimal,
        bundle: HistoricalDataBundle,
        decision_date: date,
    ) -> list[dict[str, Any]]:
        current = {position.symbol: position.quantity for position in positions}
        day = pd.Timestamp(decision_date)
        rows = []
        for symbol in self.required_tradable_symbols():
            price = Decimal(str(float(bundle.frames[symbol].loc[day, "Close"])))
            target_qty = (equity * aggregated[symbol] / price).to_integral_value(
                rounding=ROUND_FLOOR
            )
            current_qty = current.get(symbol, ZERO)
            rows.append({
                "symbol": symbol,
                "target_weight": aggregated[symbol],
                "research_reference_price": price,
                "estimated_target_shares": target_qty,
                "current_broker_shares": current_qty,
                "estimated_required_shares": target_qty - current_qty,
                "executable": False,
            })
        return rows


def collect_assets(source: AssetSource, symbols: tuple[str, ...]) -> dict[str, AlpacaAsset]:
    """Fetch exact asset metadata without silent substitution."""
    return {symbol: source.get_asset(symbol) for symbol in symbols}
