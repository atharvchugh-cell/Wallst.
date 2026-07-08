"""Optional larger mean-reversion universe: current US-listed common stocks
with market cap >= $50B, built from Nasdaq Trader's public symbol directory
files plus live yfinance market-cap lookups.

DIAGNOSTICS ONLY -- this module does not change any strategy threshold,
sizing rule, or execution assumption. It only builds an alternate `universe`
list that `MeanReversionStrategy` can be constructed with instead of the
hardcoded default in `config.MEAN_REVERSION_UNIVERSE`. Sector rotation is
untouched -- it always uses the 11 fixed sector ETFs in `config.SECTOR_ETFS`.

SURVIVORSHIP BIAS / NOT POINT-IN-TIME (read before trusting any `us_50b`
result): like the default hardcoded universe, this is a CURRENT SNAPSHOT of
companies that are large *today*, not a point-in-time historical constituent
list. A ticker included here because it is >= $50B market cap right now was
not necessarily >= $50B (or even public) for the entire historical window a
backtest is run over -- membership is decided once, using today's market
cap, and then applied uniformly across every historical date tested. Market
caps also drift with the market (bull/bear cycles, buybacks, splits), so
re-running `--refresh-universe` on a different day can change which tickers
qualify. Treat `us_50b` results the same way as the default universe's
survivorship warning: they validate strategy mechanics on a large, liquid,
currently-large-cap set of names, not a general historical edge.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from . import config

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

DEFAULT_MIN_MARKET_CAP = 50_000_000_000.0
DEFAULT_UNIVERSE_CACHE_PATH = f"{config.CACHE_DIR}/universe_us_50b.csv"

MARKET_CAP_BATCH_SIZE = 50
MARKET_CAP_MAX_RETRIES = 3
MARKET_CAP_RETRY_DELAY_SECONDS = 2.0

# If more than this fraction of candidate tickers fail market-cap lookup,
# warn loudly -- the resulting universe may be missing real >= $50B names
# purely due to fetch/rate-limit trouble, not because they're actually
# smaller than the threshold.
MARKET_CAP_LOOKUP_FAILURE_WARN_FRACTION = 0.5

# Below this many surviving tickers, hard-fail rather than silently running
# a tiny/degenerate mean-reversion backtest (mirrors the existing
# config.MIN_MEAN_REVERSION_UNIVERSE_FRACTION guard for the default universe).
MIN_US_50B_UNIVERSE_SIZE = 5

# Case-insensitive keywords in a security's listed name that mark it as NOT
# a plain common stock -- warrants, rights, units, preferreds, notes, and
# funds/trusts are excluded even though they aren't flagged by the ETF column.
_NON_COMMON_STOCK_KEYWORDS = (
    "warrant", "right", "units", " unit", "preferred", "notes", " note ", "fund", "trust",
)


class UniverseError(Exception):
    """Universe construction failed or produced a degenerate result. Callers
    (cli.py) treat this the same way as data.FetchError -- a hard failure,
    not a silent fallback to a bad universe."""


@dataclass
class UniverseSnapshot:
    """A resolved set of tickers plus enough metadata to audit how it was
    built. `market_caps`/`names`/`exchanges` may be empty (e.g. a minimal
    hand-written `--universe-csv` with only a `ticker` column)."""

    tickers: list[str]
    market_caps: dict[str, float] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
    exchanges: dict[str, str] = field(default_factory=dict)
    num_candidates: int | None = None
    num_dropped_lookup_failed: int | None = None
    snapshot_date: str | None = None
    cache_file: str | None = None

    @property
    def min_market_cap(self) -> float | None:
        return min(self.market_caps.values()) if self.market_caps else None

    @property
    def max_market_cap(self) -> float | None:
        return max(self.market_caps.values()) if self.market_caps else None


@dataclass
class UniverseResolution:
    """What `resolve_mean_reversion_universe` hands back to the CLI: the
    ticker list to actually run with, plus a metadata dict shaped for direct
    inclusion in a run's report."""

    tickers: list[str]
    info: dict


# --- Symbol directory fetch + parse ---------------------------------------

def _fetch_text(url: str, timeout: float = 30.0) -> str:
    """Raises UniverseError (not a raw requests exception) on any network
    failure -- callers (cli.py) already handle UniverseError as a clean
    hard-fail, the same way data.FetchError is handled for yfinance
    fetches, rather than letting a network hiccup crash with a raw
    traceback."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise UniverseError(f"Failed to fetch symbol directory from {url}: {e}") from e
    return resp.text


def fetch_nasdaq_listed_text() -> str:
    return _fetch_text(NASDAQ_LISTED_URL)


def fetch_other_listed_text() -> str:
    return _fetch_text(OTHER_LISTED_URL)


def normalize_yahoo_ticker(symbol: str) -> str:
    """Nasdaq Trader symbol -> Yahoo Finance ticker convention. Yahoo uses a
    dash for share-class/preferred suffixes where Nasdaq Trader uses a dot,
    e.g. `BRK.B` -> `BRK-B`, `BF.B` -> `BF-B`."""
    return symbol.strip().replace(".", "-")


def _read_pipe_delimited(text: str) -> pd.DataFrame:
    """Nasdaq Trader symbol directory files are pipe-delimited with a header
    row and a trailing 'File Creation Time: ...' footer row that isn't part
    of the data and must be dropped before parsing."""
    lines = [line for line in text.splitlines() if line.strip()]
    if lines and lines[-1].lower().startswith("file creation time"):
        lines = lines[:-1]
    if len(lines) < 2:
        return pd.DataFrame()
    return pd.read_csv(StringIO("\n".join(lines)), sep="|", dtype=str)


def _standardize_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Common post-processing for both directory formats once they've been
    renamed to symbol/name/etf/test_issue/exchange columns: drop test
    issues, ETFs, and name-pattern non-common-stock instruments, then
    normalize to Yahoo tickers."""
    if df.empty:
        return pd.DataFrame(columns=["symbol", "yahoo_ticker", "name", "exchange"])

    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df["etf"] = df.get("etf", "N").astype(str).str.strip().str.upper()
    df["test_issue"] = df.get("test_issue", "N").astype(str).str.strip().str.upper()
    df["exchange"] = df.get("exchange", "").astype(str).str.strip()

    df = df[df["symbol"] != ""]
    df = df[df["symbol"].str.lower() != "nan"]
    df = df[df["test_issue"] != "Y"]
    df = df[df["etf"] != "Y"]

    name_lower = df["name"].str.lower()
    non_common_mask = pd.Series(False, index=df.index)
    for keyword in _NON_COMMON_STOCK_KEYWORDS:
        non_common_mask = non_common_mask | name_lower.str.contains(keyword, regex=False, na=False)
    df = df[~non_common_mask]

    df["yahoo_ticker"] = df["symbol"].apply(normalize_yahoo_ticker)
    df = df[df["yahoo_ticker"] != ""]
    return df[["symbol", "yahoo_ticker", "name", "exchange"]].drop_duplicates(subset="yahoo_ticker")


def parse_nasdaq_listed(text: str) -> pd.DataFrame:
    """Parse nasdaqlisted.txt -- columns: Symbol|Security Name|Market
    Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares."""
    df = _read_pipe_delimited(text)
    if df.empty:
        return pd.DataFrame(columns=["symbol", "yahoo_ticker", "name", "exchange"])
    df = df.rename(columns={"Symbol": "symbol", "Security Name": "name", "ETF": "etf", "Test Issue": "test_issue"})
    df["exchange"] = "NASDAQ"
    return _standardize_and_filter(df)


def parse_other_listed(text: str) -> pd.DataFrame:
    """Parse otherlisted.txt -- columns: ACT Symbol|Security Name|Exchange|
    CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol. `Exchange` is a
    single-letter venue code (N=NYSE, A=NYSE American, P=NYSE Arca, etc.)."""
    df = _read_pipe_delimited(text)
    if df.empty:
        return pd.DataFrame(columns=["symbol", "yahoo_ticker", "name", "exchange"])
    df = df.rename(
        columns={"ACT Symbol": "symbol", "Security Name": "name", "Exchange": "exchange",
                 "ETF": "etf", "Test Issue": "test_issue"}
    )
    return _standardize_and_filter(df)


def build_candidate_universe(nasdaq_text: str, other_text: str) -> pd.DataFrame:
    """Combine both parsed directories into one deduplicated candidate list
    of (symbol, yahoo_ticker, name, exchange) rows -- plain common stock,
    no ETFs/test issues/warrants/rights/units/preferreds/notes/funds."""
    nasdaq_df = parse_nasdaq_listed(nasdaq_text)
    other_df = parse_other_listed(other_text)
    combined = pd.concat([nasdaq_df, other_df], ignore_index=True)
    return combined.drop_duplicates(subset="yahoo_ticker").reset_index(drop=True)


# --- Market cap lookup ------------------------------------------------------

def _get_market_cap(ticker: str) -> float | None:
    """Current market cap for one ticker via yfinance. `fast_info` is
    preferred (cheaper/faster than the full `.info` scrape); falls back to
    `.info` if fast_info doesn't have it. Returns None on any failure --
    callers treat None the same as an exception (a failed lookup to retry)."""
    t = yf.Ticker(ticker)
    cap = None
    try:
        cap = t.fast_info.get("marketCap") or t.fast_info.get("market_cap")
    except Exception:
        cap = None
    if not cap:
        try:
            cap = t.info.get("marketCap")
        except Exception:
            cap = None
    return float(cap) if cap else None


def fetch_market_caps(
    tickers: list[str],
    batch_size: int = MARKET_CAP_BATCH_SIZE,
    max_retries: int = MARKET_CAP_MAX_RETRIES,
    retry_delay_seconds: float = MARKET_CAP_RETRY_DELAY_SECONDS,
) -> tuple[dict[str, float], list[str]]:
    """Fetch current market cap per ticker, processed in batches with
    failed tickers retried up to `max_retries` times (with a delay between
    rounds, to ride out transient rate-limiting). Returns
    (market_caps, still_failed_tickers)."""
    remaining = list(tickers)
    market_caps: dict[str, float] = {}

    for attempt in range(max_retries + 1):
        if not remaining:
            break
        failed_this_round: list[str] = []
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start:batch_start + batch_size]
            for ticker in batch:
                try:
                    cap = _get_market_cap(ticker)
                except Exception:
                    cap = None
                if cap is not None and cap > 0:
                    market_caps[ticker] = cap
                else:
                    failed_this_round.append(ticker)
        remaining = failed_this_round
        if remaining and attempt < max_retries:
            time.sleep(retry_delay_seconds)

    return market_caps, remaining


# --- Cache read/write --------------------------------------------------------

def save_universe_cache(snapshot: UniverseSnapshot, cache_path: str) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "ticker": t,
            "name": snapshot.names.get(t, ""),
            "market_cap": snapshot.market_caps.get(t),
            "exchange": snapshot.exchanges.get(t, ""),
            "snapshot_date": snapshot.snapshot_date,
        }
        for t in snapshot.tickers
    ]
    pd.DataFrame(rows, columns=["ticker", "name", "market_cap", "exchange", "snapshot_date"]).to_csv(
        path, index=False
    )


def load_universe_cache(cache_path: str) -> UniverseSnapshot | None:
    """Read a universe snapshot CSV back. Tolerant of minimal hand-written
    CSVs (e.g. a `--universe-csv` with only a `ticker` column) -- only
    `ticker` is required, everything else defaults to empty/unknown."""
    path = Path(cache_path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or "ticker" not in df.columns:
        return None

    df["ticker"] = df["ticker"].astype(str).str.strip()
    df = df[df["ticker"] != ""]
    if df.empty:
        return None

    tickers = df["ticker"].tolist()
    market_caps = (
        {t: float(c) for t, c in zip(df["ticker"], df["market_cap"]) if pd.notna(c)}
        if "market_cap" in df.columns else {}
    )
    names = dict(zip(df["ticker"], df["name"])) if "name" in df.columns else {}
    exchanges = dict(zip(df["ticker"], df["exchange"])) if "exchange" in df.columns else {}
    snapshot_date = (
        str(df["snapshot_date"].iloc[0]) if "snapshot_date" in df.columns and len(df) else "unknown"
    )

    return UniverseSnapshot(
        tickers=tickers,
        market_caps=market_caps,
        names=names,
        exchanges=exchanges,
        num_candidates=len(tickers),
        num_dropped_lookup_failed=0,
        snapshot_date=snapshot_date,
        cache_file=str(path),
    )


# --- Build the >= $50B universe from live data -------------------------------

def build_us_50b_universe(
    min_market_cap: float = DEFAULT_MIN_MARKET_CAP,
    cache_path: str = DEFAULT_UNIVERSE_CACHE_PATH,
) -> UniverseSnapshot:
    """Fetch both Nasdaq Trader symbol directories, filter to plain common
    stock, look up current market cap for every candidate, keep tickers
    >= `min_market_cap`, cache the result, and return it. Hard-fails via
    UniverseError if the resulting universe is too small to backtest
    meaningfully -- see MIN_US_50B_UNIVERSE_SIZE."""
    nasdaq_text = fetch_nasdaq_listed_text()
    other_text = fetch_other_listed_text()
    candidates = build_candidate_universe(nasdaq_text, other_text)

    tickers = candidates["yahoo_ticker"].tolist()
    market_caps, failed = fetch_market_caps(tickers)

    total = len(tickers)
    num_failed = len(failed)
    if total > 0 and num_failed / total > MARKET_CAP_LOOKUP_FAILURE_WARN_FRACTION:
        warnings.warn(
            f"Market-cap lookup failed for {num_failed}/{total} candidate tickers "
            f"({num_failed / total:.0%}) -- yfinance rate limiting or network issues "
            f"are the most likely cause. The resulting >= ${min_market_cap:,.0f} "
            f"universe below may be missing real qualifying names."
        )

    qualifying = {t: cap for t, cap in market_caps.items() if cap >= min_market_cap}
    if len(qualifying) < MIN_US_50B_UNIVERSE_SIZE:
        raise UniverseError(
            f"Only {len(qualifying)} ticker(s) qualified for the >= ${min_market_cap:,.0f} "
            f"universe (minimum required: {MIN_US_50B_UNIVERSE_SIZE}). Refusing to run a "
            f"mean-reversion backtest on a degenerate universe rather than silently produce "
            f"a near-empty backtest. {num_failed}/{total} candidate tickers failed "
            f"market-cap lookup."
        )

    name_by_ticker = dict(zip(candidates["yahoo_ticker"], candidates["name"]))
    exchange_by_ticker = dict(zip(candidates["yahoo_ticker"], candidates["exchange"]))
    snapshot_date = datetime.now(timezone.utc).isoformat()

    snapshot = UniverseSnapshot(
        tickers=sorted(qualifying, key=lambda t: qualifying[t], reverse=True),
        market_caps=qualifying,
        names={t: name_by_ticker.get(t, "") for t in qualifying},
        exchanges={t: exchange_by_ticker.get(t, "") for t in qualifying},
        num_candidates=total,
        num_dropped_lookup_failed=num_failed,
        snapshot_date=snapshot_date,
        cache_file=cache_path,
    )
    save_universe_cache(snapshot, cache_path)
    return snapshot


# --- CLI-facing dispatcher ----------------------------------------------------

def _snapshot_to_info(snapshot: UniverseSnapshot, mode: str) -> dict:
    return {
        "mode": mode,
        "num_selected": len(snapshot.tickers),
        "num_candidates": snapshot.num_candidates,
        "num_dropped_lookup_failed": snapshot.num_dropped_lookup_failed,
        "min_market_cap": snapshot.min_market_cap,
        "max_market_cap": snapshot.max_market_cap,
        "cache_file": snapshot.cache_file,
        "snapshot_date": snapshot.snapshot_date,
    }


def resolve_mean_reversion_universe(
    mode: str = "default",
    csv_path: str | None = None,
    refresh: bool = False,
    cache_path: str = DEFAULT_UNIVERSE_CACHE_PATH,
    min_market_cap: float = DEFAULT_MIN_MARKET_CAP,
) -> UniverseResolution:
    """Single entry point cli.py uses to decide which ticker list
    mean_reversion should run with. `csv_path` (if given) always wins,
    regardless of `mode`, since it's an explicit user override.

    - "default": today's hardcoded config.MEAN_REVERSION_UNIVERSE -- always
      available, no network, byte-identical to every mode's prior behavior.
    - "us_50b": current US-listed common stock >= $50B market cap. Loaded
      from `cache_path` unless `refresh` is set or the cache doesn't exist
      yet, in which case it's rebuilt from live Nasdaq Trader + yfinance
      data (see build_us_50b_universe).
    - explicit csv_path: a user-supplied ticker list (own snapshot, or one
      previously cached by this module).
    """
    if csv_path:
        snapshot = load_universe_cache(csv_path)
        if snapshot is None or not snapshot.tickers:
            raise UniverseError(f"--universe-csv {csv_path!r} is missing, empty, or has no 'ticker' column.")
        return UniverseResolution(tickers=snapshot.tickers, info=_snapshot_to_info(snapshot, "csv"))

    if mode == "default":
        return UniverseResolution(
            tickers=list(config.MEAN_REVERSION_UNIVERSE),
            info={
                "mode": "default", "num_selected": len(config.MEAN_REVERSION_UNIVERSE),
                "num_candidates": None, "num_dropped_lookup_failed": None,
                "min_market_cap": None, "max_market_cap": None,
                "cache_file": None, "snapshot_date": None,
            },
        )

    if mode == "us_50b":
        snapshot = None if refresh else load_universe_cache(cache_path)
        if snapshot is None:
            snapshot = build_us_50b_universe(min_market_cap=min_market_cap, cache_path=cache_path)
        return UniverseResolution(tickers=snapshot.tickers, info=_snapshot_to_info(snapshot, "us_50b"))

    raise ValueError(f"Unknown universe mode: {mode!r}")
