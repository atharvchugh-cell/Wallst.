"""Historical price data: yfinance fetch, metadata-keyed on-disk cache, and
the canonical trading calendar used to drive the backtest walk.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from . import config


class FetchError(Exception):
    """A required ticker could not be fetched. Strategies that need their
    full universe (e.g. sector rotation) let this propagate as a hard
    failure rather than silently continuing with a reduced universe."""


def _cache_paths(ticker: str, cache_dir: str) -> tuple[Path, Path]:
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{ticker}.csv", d / f"{ticker}.meta.json"


def _read_cache(ticker: str, cache_dir: str) -> tuple[pd.DataFrame | None, dict | None]:
    csv_path, meta_path = _cache_paths(ticker, cache_dir)
    if not csv_path.exists() or not meta_path.exists():
        return None, None
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        with open(meta_path) as f:
            meta = json.load(f)
        return df, meta
    except Exception:
        return None, None


def _write_cache(ticker: str, cache_dir: str, df: pd.DataFrame, meta: dict) -> None:
    csv_path, meta_path = _cache_paths(ticker, cache_dir)
    fd, tmp_csv = tempfile.mkstemp(dir=csv_path.parent, suffix=".csv")
    os.close(fd)
    try:
        df.to_csv(tmp_csv)
        shutil.move(tmp_csv, csv_path)
    finally:
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)

    fd, tmp_meta = tempfile.mkstemp(dir=meta_path.parent, suffix=".json")
    os.close(fd)
    try:
        with open(tmp_meta, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        shutil.move(tmp_meta, meta_path)
    finally:
        if os.path.exists(tmp_meta):
            os.remove(tmp_meta)


def _meta_matches(meta: dict, auto_adjust: bool, interval: str) -> bool:
    return (
        meta.get("auto_adjust") == auto_adjust
        and meta.get("interval") == interval
        and meta.get("yfinance_version") == yf.__version__
    )


def _fetch_range(ticker: str, start, end, auto_adjust: bool, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        interval=interval,
        progress=False,
    )
    if df is None or df.empty:
        raise FetchError(f"No data returned for {ticker} ({start} to {end})")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.DatetimeIndex(df.index).normalize()
    return df


def get_price_history(
    ticker: str,
    start,
    end,
    cache_dir: str = config.CACHE_DIR,
    auto_adjust: bool = config.YFINANCE_AUTO_ADJUST,
    interval: str = config.YFINANCE_INTERVAL,
    force_refresh: bool = False,
    refresh_threshold_days: int = config.CACHE_REFRESH_THRESHOLD_DAYS,
) -> pd.DataFrame:
    """Fetch daily OHLCV for one ticker via a metadata-keyed on-disk cache.

    Cache metadata (auto_adjust/interval/yfinance version) is part of the
    cache key: a mismatch forces a refetch even if the file is "fresh".
    Adjusted prices can change retroactively (dividends/splits/corrections),
    so a stale cache -- or one whose start postdates the request -- triggers
    a full re-download rather than a delta append; only a tail extension
    (requested end beyond the cache) does a cheaper delta fetch. A failed
    fetch never corrupts the existing cache (write-temp-then-swap).
    """
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    cached_df, meta = _read_cache(ticker, cache_dir)
    now = datetime.now(timezone.utc)

    needs_full_refresh = (
        cached_df is None
        or meta is None
        or not _meta_matches(meta, auto_adjust, interval)
        or force_refresh
        or start < cached_df.index.min()
        or (now - datetime.fromisoformat(meta["last_full_refresh"])).days >= refresh_threshold_days
    )

    if needs_full_refresh:
        df = _fetch_range(ticker, start, end, auto_adjust, interval)
        meta_out = {
            "ticker": ticker,
            "cached_start": str(df.index.min()),
            "cached_end": str(df.index.max()),
            "yfinance_version": yf.__version__,
            "auto_adjust": auto_adjust,
            "interval": interval,
            "last_full_refresh": now.isoformat(),
            "fetch_success": True,
        }
        _write_cache(ticker, cache_dir, df, meta_out)
        return df.loc[(df.index >= start) & (df.index <= end)]

    if end > cached_df.index.max():
        delta_start = cached_df.index.max() + pd.Timedelta(days=1)
        try:
            delta_df = _fetch_range(ticker, delta_start, end, auto_adjust, interval)
            combined = pd.concat([cached_df, delta_df])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            meta_out = dict(meta)
            meta_out["cached_end"] = str(combined.index.max())
            _write_cache(ticker, cache_dir, combined, meta_out)
            cached_df = combined
        except FetchError:
            pass  # no new bars yet (e.g. weekend) -- fall back to what's cached

    return cached_df.loc[(cached_df.index >= start) & (cached_df.index <= end)]


def get_price_data(
    tickers: list[str],
    start,
    end,
    warmup_calendar_days: int,
    hard_fail_on_missing: bool = False,
    **kwargs,
) -> tuple[dict[str, pd.DataFrame], list[tuple[str, str]]]:
    """Fetch history for a list of tickers, each with `warmup_calendar_days`
    of extra history before `start` for indicator warmup.

    Returns (price_data, dropped) where `dropped` is a list of
    (ticker, reason) pairs. If hard_fail_on_missing, a fetch failure raises
    FetchError instead of being recorded in `dropped` -- used by strategies
    (sector rotation) that require their full universe.
    """
    fetch_start = pd.Timestamp(start) - pd.Timedelta(days=warmup_calendar_days)
    price_data: dict[str, pd.DataFrame] = {}
    dropped: list[tuple[str, str]] = []
    for ticker in tickers:
        try:
            df = get_price_history(ticker, fetch_start, end, **kwargs)
            if df.empty:
                raise FetchError(f"Empty history for {ticker}")
            price_data[ticker] = df
        except Exception as e:
            if hard_fail_on_missing:
                raise FetchError(f"Required ticker {ticker} failed to fetch: {e}") from e
            dropped.append((ticker, str(e)))
    return price_data, dropped


def get_benchmark_data(start, end, ticker: str = config.BENCHMARK_TICKER, **kwargs) -> pd.DataFrame:
    return get_price_history(ticker, start, end, **kwargs)


def compute_sector_effective_start(
    price_data: dict[str, pd.DataFrame], requested_start, lookback_months: int
) -> tuple[pd.Timestamp, dict[str, pd.Timestamp]]:
    """Effective start = the later of the requested start and (the latest ETF
    inception date + `lookback_months`) -- not merely the latest inception
    date, since a freshly-listed ETF also needs `lookback_months` of its own
    history before its trailing return is computable."""
    first_dates = {t: df.index.min() for t, df in price_data.items()}
    latest_inception = max(first_dates.values())
    min_viable_start = latest_inception + pd.DateOffset(months=lookback_months)
    effective_start = max(pd.Timestamp(requested_start), min_viable_start)
    return effective_start, first_dates


def build_canonical_calendar(reference_df: pd.DataFrame, start, end) -> pd.DatetimeIndex:
    """The canonical trading calendar for a run, derived from a reference
    ticker's own trading dates (the benchmark, SPY, which is always fetched
    and has essentially complete modern exchange history). Using a fixed
    reference -- rather than intersecting every universe ticker together --
    means one ticker's genuinely shorter history doesn't silently shrink the
    calendar for everyone else; see `find_gaps` for how that's handled instead.
    """
    idx = reference_df.index
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    return idx[(idx >= start) & (idx <= end)].sort_values()


def find_gaps(df: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Dates in `calendar` that fall within `df`'s own active range (on/after
    its first valid date) but are missing or NaN in `df`. Empty means no gap
    -- a ticker simply starting later than `calendar[0]` is NOT a gap."""
    if df.empty:
        return calendar
    active_calendar = calendar[calendar >= df.index.min()]
    reindexed = df.reindex(active_calendar)
    return reindexed[reindexed["Close"].isna()].index


def exclude_unfinalized_today(calendar: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, bool]:
    """Drop today's date from the calendar if present -- its bar may not be
    finalized yet if the run happens mid-session. Returns (calendar, was_dropped)."""
    today = pd.Timestamp.now().normalize()
    if len(calendar) > 0 and calendar[-1] == today:
        return calendar[:-1], True
    return calendar, False
