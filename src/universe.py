"""Optional larger mean-reversion universe: current US-listed common stocks
with market cap >= $50B.

DIAGNOSTICS ONLY -- this module does not change any strategy threshold,
sizing rule, or execution assumption. It only builds an alternate `universe`
list that `MeanReversionStrategy` can be constructed with instead of the
hardcoded default in `config.MEAN_REVERSION_UNIVERSE`. Sector rotation is
untouched -- it always uses the 11 fixed sector ETFs in `config.SECTOR_ETFS`.

## How the universe is built (bulk screener, not per-ticker lookups)

Market cap is fetched in BULK via Yahoo Finance's screener API (`yf.screen`
with an `EquityQuery` filtering on `intradaymarketcap >= min_market_cap` and
`region == "us"`) -- a small number of paginated requests (Yahoo caps each
page at 250 results; realistically only a few hundred US companies clear a
$50B bar, so this is typically 1-3 requests total) that return qualifying
tickers AND their market caps directly, with progress printed per page.

This deliberately does NOT check market cap one ticker at a time across the
full ~8,000+-ticker Nasdaq Trader symbol directory. An earlier version of
this module did exactly that (`fetch_market_caps`/`_get_market_cap`, still
present below as a tested utility, but no longer used by the default build
path) and was impractically slow in practice: `yfinance`'s `Ticker.fast_info`
and `.info` can each trigger multiple additional sub-requests (share count,
last price, history metadata) per ticker, so scanning thousands of
candidates took minutes with no progress output and looked like a hang.

The Nasdaq Trader symbol directories (`nasdaqlisted.txt`, `otherlisted.txt`)
are still fetched and parsed (`build_candidate_universe`) -- not to drive
per-ticker lookups anymore, but as a source of company name/exchange
metadata for tickers the screener response doesn't already carry, and
because the same warrant/right/unit/preferred/notes/fund name-pattern
filter used there is also applied to whatever name the screener itself
returns, as a safety net (Yahoo's screener already restricts results to
`quoteType=EQUITY`, which excludes ETFs, but that alone doesn't guarantee
every result is what we'd call plain common stock).

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
from typing import Callable

import pandas as pd
import requests
import yfinance as yf

from . import config

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

DEFAULT_MIN_MARKET_CAP = 50_000_000_000.0
DEFAULT_UNIVERSE_CACHE_PATH = f"{config.CACHE_DIR}/universe_us_50b.csv"

# Yahoo's screener endpoint caps a single request at 250 results; paginate
# via `offset` until a page comes back short (fewer than requested) or this
# many pages have been fetched -- 10*250=2500 is a generous safety cap, far
# more than the realistic few hundred US equities that clear a $50B bar.
SCREENER_PAGE_SIZE = 250
SCREENER_MAX_PAGES = 10

# Legacy/fallback per-ticker lookup knobs -- see fetch_market_caps below.
MARKET_CAP_BATCH_SIZE = 50
MARKET_CAP_MAX_RETRIES = 3
MARKET_CAP_RETRY_DELAY_SECONDS = 2.0

# If more than this fraction of screener results can't be parsed for a
# symbol/market cap, warn loudly -- Yahoo's response schema may have
# changed, and the resulting universe may be missing real >= $50B names.
MARKET_CAP_LOOKUP_FAILURE_WARN_FRACTION = 0.5

# Below this many surviving tickers, hard-fail rather than silently running
# a tiny/degenerate mean-reversion backtest (mirrors the existing
# config.MIN_MEAN_REVERSION_UNIVERSE_FRACTION guard for the default universe).
MIN_US_50B_UNIVERSE_SIZE = 5

# Case-insensitive keywords in a security's listed name that mark it as NOT
# a plain common stock -- warrants, rights, units, preferreds, notes, and
# funds are excluded by name pattern even though they aren't flagged by the
# ETF column. ETFs themselves are excluded via the ETF flag, not by name --
# deliberately NOT excluding "trust" here, since legitimate common-stock
# companies (REITs in particular) routinely have "Trust" in their listed
# name, e.g. "Example Realty Trust Inc. Common Stock".
_NON_COMMON_STOCK_KEYWORDS = (
    "warrant", "right", "units", " unit", "preferred", "notes", " note ", "fund",
)


def _name_is_non_common_stock(name: str) -> bool:
    """Scalar version of the name-pattern check, shared by the Nasdaq
    Trader directory filter (`_standardize_and_filter`, vectorized) and the
    per-quote screener-result filter (`build_us_50b_universe`)."""
    name_lower = (name or "").lower()
    return any(keyword in name_lower for keyword in _NON_COMMON_STOCK_KEYWORDS)


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

    df = df[~df["name"].apply(_name_is_non_common_stock)]

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


# --- Market cap lookup: bulk screener (the default/primary path) -----------

def _build_market_cap_equity_query(min_market_cap: float) -> yf.EquityQuery:
    return yf.EquityQuery(
        "and",
        [
            yf.EquityQuery("gte", ["intradaymarketcap", min_market_cap]),
            yf.EquityQuery("eq", ["region", "us"]),
        ],
    )


def _extract_quote_field(quote: dict, keys: tuple[str, ...]):
    """Yahoo's screener response isn't formally documented/versioned by
    yfinance, so field extraction is deliberately tolerant: try several
    plausible key names in order, and unwrap the common Yahoo
    `{"raw": ..., "fmt": ...}` numeric-field shape if present. Returns None
    (never raises) if nothing usable is found -- callers count that as a
    parse failure for one quote, not a reason to abort the whole build."""
    for key in keys:
        val = quote.get(key)
        if val is None:
            continue
        if isinstance(val, dict):
            val = val.get("raw")
        if val not in (None, ""):
            return val
    return None


def fetch_us_large_cap_quotes(
    min_market_cap: float,
    page_size: int = SCREENER_PAGE_SIZE,
    max_pages: int = SCREENER_MAX_PAGES,
    max_results: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict]:
    """Bulk-fetch raw quote dicts for US equities with market cap >=
    `min_market_cap` via Yahoo Finance's screener (`yf.screen`), paginating
    with `offset` until a page returns fewer results than requested (end of
    results), `max_pages` is reached, or `max_results` total quotes have
    been collected. Raises UniverseError (not a raw yfinance/network
    exception) if any page request fails -- a screener failure hard-fails
    cleanly here rather than silently falling back to the much slower
    per-ticker scan this function replaces."""
    progress = progress or (lambda msg: None)
    query = _build_market_cap_equity_query(min_market_cap)
    quotes: list[dict] = []

    for page in range(max_pages):
        if max_results is not None and len(quotes) >= max_results:
            break
        offset = page * page_size
        page_size_this_call = page_size if max_results is None else min(page_size, max_results - len(quotes))
        progress(f"Querying market-cap screener page {page + 1} (offset {offset}, requesting {page_size_this_call})...")
        try:
            result = yf.screen(
                query, offset=offset, size=page_size_this_call, sortField="intradaymarketcap", sortAsc=False,
            )
        except Exception as e:
            raise UniverseError(
                f"Market-cap screener request failed on page {page + 1}: {e}. This queries Yahoo "
                f"Finance's screener for US equities with market cap >= ${min_market_cap:,.0f} in bulk "
                f"instead of one request per candidate ticker; see src/universe.py's module docstring."
            ) from e
        page_quotes = result.get("quotes", []) if isinstance(result, dict) else []
        quotes.extend(page_quotes)
        progress(f"  page {page + 1}: {len(page_quotes)} quote(s) ({len(quotes)} total so far).")
        if len(page_quotes) < page_size_this_call:
            break

    return quotes


# --- Market cap lookup: legacy per-ticker fallback (NOT the default path) --
#
# Kept as a tested utility (e.g. useful for looking up a small, specific
# ticker list) but no longer called by build_us_50b_universe's default
# path -- see the module docstring for why the bulk screener above replaced
# it as the default. If this WERE used to scan a large candidate list, each
# call is still bounded by yfinance's own internal per-HTTP-request timeout
# (~30s), so it fails a given ticker rather than hanging forever on it --
# the practical problem was volume (thousands of sequential slow calls),
# not any single call being literally unbounded.

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
    max_candidates: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> UniverseSnapshot:
    """Build the >= `min_market_cap` US-equity universe via Yahoo's bulk
    screener (see module docstring for why), cache the result, and return
    it. `max_candidates` is a debug/safety cap on how many screener results
    to consider -- leave it None (the default) for a real run; it exists
    for fast, bounded dev/test runs, not production use. `progress`, if
    given, receives a line of text at each meaningful step (candidate
    count, per-page screener progress, running qualifying count, final
    summary) -- cli.py passes `print`. Hard-fails via UniverseError if the
    resulting universe is too small to backtest meaningfully -- see
    MIN_US_50B_UNIVERSE_SIZE."""
    progress = progress or (lambda msg: None)

    # Nasdaq Trader directories supply supplementary name/exchange metadata
    # ONLY -- the screener call below is what actually determines universe
    # membership. A Nasdaq Trader outage is therefore not fatal on its own:
    # fall back to empty candidate metadata (the screener's own quotes
    # already carry a name/exchange for most results) rather than failing
    # the whole build over a source that no longer drives ticker selection.
    progress("Fetching Nasdaq Trader symbol directories (for company name/exchange metadata)...")
    candidate_names: dict[str, str] = {}
    candidate_exchanges: dict[str, str] = {}
    try:
        nasdaq_text = fetch_nasdaq_listed_text()
        other_text = fetch_other_listed_text()
        candidates = build_candidate_universe(nasdaq_text, other_text)
        candidate_names = dict(zip(candidates["yahoo_ticker"], candidates["name"]))
        candidate_exchanges = dict(zip(candidates["yahoo_ticker"], candidates["exchange"]))
        progress(f"{len(candidates)} candidate US-listed common-stock tickers after Nasdaq Trader parsing/filtering.")
    except UniverseError as e:
        progress(f"Nasdaq Trader directories unavailable ({e}); continuing with screener-only metadata.")

    progress(f"Querying Yahoo Finance market-cap screener for US equities >= ${min_market_cap:,.0f}...")
    quotes = fetch_us_large_cap_quotes(min_market_cap, max_results=max_candidates, progress=progress)
    progress(f"Screener returned {len(quotes)} quote(s) total; parsing and filtering...")

    market_caps: dict[str, float] = {}
    names: dict[str, str] = {}
    exchanges: dict[str, str] = {}
    num_unparseable = 0
    num_excluded_non_common = 0

    for quote in quotes:
        symbol = _extract_quote_field(quote, ("symbol",))
        raw_cap = _extract_quote_field(quote, ("intradaymarketcap", "marketCap", "lastclosemarketcap.lasttwelvemonths"))
        if not symbol or raw_cap is None:
            num_unparseable += 1
            continue
        try:
            cap = float(raw_cap)
        except (TypeError, ValueError):
            num_unparseable += 1
            continue
        if cap < min_market_cap:
            continue

        ticker = normalize_yahoo_ticker(str(symbol))
        name = _extract_quote_field(quote, ("longName", "shortName", "displayName"))
        name = str(name) if name else candidate_names.get(ticker, "")
        if _name_is_non_common_stock(name):
            num_excluded_non_common += 1
            continue

        market_caps[ticker] = cap
        names[ticker] = name
        exchange = _extract_quote_field(quote, ("exchange", "fullExchangeName", "exchDisp"))
        exchanges[ticker] = str(exchange) if exchange else candidate_exchanges.get(ticker, "")
        progress(f"  qualifying so far: {len(market_caps)} (latest: {ticker} = ${cap:,.0f})")

    total_quotes = len(quotes)
    if total_quotes > 0 and num_unparseable / total_quotes > MARKET_CAP_LOOKUP_FAILURE_WARN_FRACTION:
        warnings.warn(
            f"{num_unparseable}/{total_quotes} screener results ({num_unparseable / total_quotes:.0%}) "
            f"could not be parsed for a symbol/market cap -- Yahoo's screener response schema may have "
            f"changed. The resulting >= ${min_market_cap:,.0f} universe below may be incomplete."
        )

    progress(
        f"Done: {len(market_caps)} qualifying tickers, {num_excluded_non_common} excluded as "
        f"non-common-stock by name, {num_unparseable} unparseable screener result(s)."
    )

    if len(market_caps) < MIN_US_50B_UNIVERSE_SIZE:
        raise UniverseError(
            f"Only {len(market_caps)} ticker(s) qualified for the >= ${min_market_cap:,.0f} universe "
            f"(minimum required: {MIN_US_50B_UNIVERSE_SIZE}). Refusing to run a mean-reversion backtest "
            f"on a degenerate universe rather than silently produce a near-empty backtest. Screener "
            f"returned {total_quotes} total result(s): {num_unparseable} unparseable, "
            f"{num_excluded_non_common} excluded as non-common-stock by name."
        )

    snapshot = UniverseSnapshot(
        tickers=sorted(market_caps, key=lambda t: market_caps[t], reverse=True),
        market_caps=market_caps,
        names=names,
        exchanges=exchanges,
        num_candidates=total_quotes,
        num_dropped_lookup_failed=num_unparseable,
        snapshot_date=datetime.now(timezone.utc).isoformat(),
        cache_file=cache_path,
    )
    save_universe_cache(snapshot, cache_path)
    progress(f"Universe cached to {cache_path}.")
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
    max_candidates: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> UniverseResolution:
    """Single entry point cli.py uses to decide which ticker list
    mean_reversion should run with. `csv_path` (if given) always wins,
    regardless of `mode`, since it's an explicit user override.

    - "default": today's hardcoded config.MEAN_REVERSION_UNIVERSE -- always
      available, no network, byte-identical to every mode's prior behavior.
    - "us_50b": current US-listed common stock >= $50B market cap. Loaded
      from `cache_path` unless `refresh` is set or the cache doesn't exist
      yet, in which case it's rebuilt from live data via the bulk market-cap
      screener (see build_us_50b_universe). `max_candidates`/`progress` are
      passed straight through to that rebuild (ignored when loading from
      cache, since there's nothing to page through or report progress on).
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
            snapshot = build_us_50b_universe(
                min_market_cap=min_market_cap, cache_path=cache_path,
                max_candidates=max_candidates, progress=progress,
            )
        return UniverseResolution(tickers=snapshot.tickers, info=_snapshot_to_info(snapshot, "us_50b"))

    raise ValueError(f"Unknown universe mode: {mode!r}")
