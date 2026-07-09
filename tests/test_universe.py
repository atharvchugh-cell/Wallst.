import pandas as pd
import pytest
import requests

from src import config, universe

NASDAQ_LISTED_SAMPLE = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
MSFT|Microsoft Corporation - Common Stock|Q|N|N|100|N|N
ZZZZ|Test Company - Common Stock|Q|Y|N|100|N|N
QQQ|Invesco QQQ Trust|Q|N|N|100|Y|N
AAWW|Atlas Air Worldwide Holdings Warrant|Q|N|N|100|N|N
EXRT|Example Realty Trust Inc. Common Stock|Q|N|N|100|N|N
XPRF|Example Corp Class X Stock|Q|N|N|100|N|N
File Creation Time: 0708202600:00
"""


@pytest.fixture(autouse=True)
def _stub_price_data_available(monkeypatch):
    # build_us_50b_universe's final price-data validation step hits the
    # network by default (yfinance history lookups). Every test in this
    # module that doesn't care about that step should be able to assume
    # price data is available for whatever tickers it's using -- tests
    # that specifically want a ticker to fail this check pass their own
    # `price_data_checker` argument to build_us_50b_universe, which
    # overrides this stub entirely.
    monkeypatch.setattr(universe, "_default_price_data_available", lambda ticker: True)

OTHER_LISTED_SAMPLE = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
BRK.B|Berkshire Hathaway Inc. Class B Common Stock|N|BRK.B|N|100|N|BRK.B
JPM|JPMorgan Chase & Co. Common Stock|N|JPM|N|100|N|JPM
SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY
PSA.PRA|Public Storage Depositary Shares Preferred Series A|N|PSA.PRA|N|100|N|PSA.PRA
UNITX|Some Company Units|N|UNITX|N|100|N|UNITX
XYZN|Example Corp 4.5% Senior Notes due 2030|N|XYZN|N|100|N|XYZN
File Creation Time: 0708202600:00
"""


def test_parse_nasdaq_listed_excludes_test_issue_etf_and_warrant():
    df = universe.parse_nasdaq_listed(NASDAQ_LISTED_SAMPLE)
    symbols = set(df["yahoo_ticker"])
    assert symbols == {"AAPL", "MSFT", "EXRT", "XPRF"}
    assert (df["exchange"] == "NASDAQ").all()


def test_parse_other_listed_excludes_etf_and_preferred_and_units():
    df = universe.parse_other_listed(OTHER_LISTED_SAMPLE)
    symbols = set(df["yahoo_ticker"])
    assert symbols == {"BRK-B", "JPM"}


def test_parse_nasdaq_listed_keeps_common_stock_with_trust_in_name():
    # "Trust" alone must NOT exclude a security -- REITs and other
    # legitimate common-stock companies routinely have "Trust" in their
    # listed name. ETFs (like QQQ, also in this sample) are excluded via
    # the ETF flag, not by name pattern.
    df = universe.parse_nasdaq_listed(NASDAQ_LISTED_SAMPLE)
    symbols = set(df["yahoo_ticker"])
    assert "EXRT" in symbols
    assert "QQQ" not in symbols  # excluded via ETF=Y, not because its name says "Trust"


def test_parse_other_listed_excludes_notes():
    df = universe.parse_other_listed(OTHER_LISTED_SAMPLE)
    assert "XYZN" not in set(df["yahoo_ticker"])


def test_normalize_yahoo_ticker_dot_to_dash():
    assert universe.normalize_yahoo_ticker("BRK.B") == "BRK-B"
    assert universe.normalize_yahoo_ticker("AAPL") == "AAPL"
    assert universe.normalize_yahoo_ticker(" BF.B ") == "BF-B"


def test_fetch_text_wraps_network_failure_in_universe_error(monkeypatch):
    # A raw requests exception (connection refused, proxy blocked, DNS
    # failure, etc.) must surface as UniverseError -- the same clean
    # hard-fail every other data-fetch path in this codebase uses -- not
    # crash the whole CLI with an unhandled traceback.
    def fake_get(url, timeout):
        raise requests.exceptions.ProxyError("simulated proxy failure")

    monkeypatch.setattr(requests, "get", fake_get)
    with pytest.raises(universe.UniverseError, match="Failed to fetch symbol directory"):
        universe.fetch_nasdaq_listed_text()


def test_build_candidate_universe_combines_and_dedupes():
    combined = universe.build_candidate_universe(NASDAQ_LISTED_SAMPLE, OTHER_LISTED_SAMPLE)
    assert set(combined["yahoo_ticker"]) == {"AAPL", "MSFT", "EXRT", "XPRF", "BRK-B", "JPM"}
    assert len(combined) == len(combined["yahoo_ticker"].unique())


def test_fetch_market_caps_filters_failures_and_retries(monkeypatch):
    call_counts = {"AAPL": 0, "MSFT": 0, "BAD": 0}

    def fake_get_market_cap(ticker):
        call_counts[ticker] += 1
        if ticker == "AAPL":
            return 3_000_000_000_000.0
        if ticker == "MSFT":
            return 2_500_000_000_000.0
        raise RuntimeError("simulated fetch failure")

    monkeypatch.setattr(universe, "_get_market_cap", fake_get_market_cap)
    market_caps, failed = universe.fetch_market_caps(
        ["AAPL", "MSFT", "BAD"], max_retries=2, retry_delay_seconds=0.0
    )
    assert market_caps == {"AAPL": 3_000_000_000_000.0, "MSFT": 2_500_000_000_000.0}
    assert failed == ["BAD"]
    # BAD should have been retried max_retries+1 times total; AAPL/MSFT only once (succeeded immediately).
    assert call_counts["BAD"] == 3
    assert call_counts["AAPL"] == 1
    assert call_counts["MSFT"] == 1


# Realistic default longNames for tickers used across build_us_50b_universe
# tests without an explicit `name=` -- these need to share a meaningful word
# with the corresponding Nasdaq Trader sample name above, or the
# identity-mismatch guard added in build_us_50b_universe would spuriously
# flag them (a placeholder like "AAPL Inc. Common Stock" shares nothing in
# common with the Nasdaq Trader name "Apple Inc. - Common Stock").
_DEFAULT_QUOTE_NAMES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
}


def make_quote(symbol, market_cap, name=None, exchange="NMS"):
    return {
        "symbol": symbol,
        "longName": name or _DEFAULT_QUOTE_NAMES.get(symbol, f"{symbol} Inc. Common Stock"),
        "intradaymarketcap": market_cap,
        "exchange": exchange,
    }


def test_fetch_us_large_cap_quotes_paginates_until_short_page(monkeypatch):
    # Two full pages of size 2, then a short (1-item) final page -- pagination
    # must stop after the short page rather than requesting a 4th.
    pages = [
        [make_quote("A", 1e12), make_quote("B", 1e12)],
        [make_quote("C", 1e12), make_quote("D", 1e12)],
        [make_quote("E", 1e12)],
    ]
    calls = []

    def fake_screen(query, offset, size, sortField, sortAsc):
        calls.append((offset, size))
        return {"quotes": pages[offset // size]}

    monkeypatch.setattr(universe.yf, "screen", fake_screen)
    quotes = universe.fetch_us_large_cap_quotes(50e9, page_size=2, max_pages=10)
    assert [q["symbol"] for q in quotes] == ["A", "B", "C", "D", "E"]
    assert len(calls) == 3


def test_fetch_us_large_cap_quotes_respects_max_pages_safety_cap(monkeypatch):
    # Every page comes back "full" (never short) -- must still stop at max_pages.
    def fake_screen(query, offset, size, sortField, sortAsc):
        return {"quotes": [make_quote(f"T{offset}", 1e12)] * size}

    monkeypatch.setattr(universe.yf, "screen", fake_screen)
    quotes = universe.fetch_us_large_cap_quotes(50e9, page_size=2, max_pages=3)
    assert len(quotes) == 6  # 3 pages x 2 each, then stopped


def test_fetch_us_large_cap_quotes_respects_max_results(monkeypatch):
    def fake_screen(query, offset, size, sortField, sortAsc):
        return {"quotes": [make_quote(f"T{offset}-{i}", 1e12) for i in range(size)]}

    monkeypatch.setattr(universe.yf, "screen", fake_screen)
    quotes = universe.fetch_us_large_cap_quotes(50e9, page_size=250, max_pages=10, max_results=5)
    assert len(quotes) == 5


def test_fetch_us_large_cap_quotes_wraps_screener_failure_in_universe_error(monkeypatch):
    def fake_screen(query, offset, size, sortField, sortAsc):
        raise RuntimeError("simulated Yahoo screener outage")

    monkeypatch.setattr(universe.yf, "screen", fake_screen)
    with pytest.raises(universe.UniverseError, match="Market-cap screener request failed"):
        universe.fetch_us_large_cap_quotes(50e9)


def test_fetch_us_large_cap_quotes_reports_progress(monkeypatch):
    def fake_screen(query, offset, size, sortField, sortAsc):
        return {"quotes": [make_quote("A", 1e12)]}

    monkeypatch.setattr(universe.yf, "screen", fake_screen)
    messages = []
    universe.fetch_us_large_cap_quotes(50e9, page_size=250, progress=messages.append)
    assert any("page 1" in m for m in messages)


def test_build_us_50b_universe_filters_by_threshold_and_writes_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(universe, "fetch_nasdaq_listed_text", lambda: NASDAQ_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "fetch_other_listed_text", lambda: OTHER_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "MIN_US_50B_UNIVERSE_SIZE", 2)

    quotes = [
        make_quote("AAPL", 3_000_000_000_000.0),
        make_quote("MSFT", 2_500_000_000_000.0),
        make_quote("BRK-B", 800_000_000_000.0, name="Berkshire Hathaway Inc. Class B"),
    ]
    monkeypatch.setattr(universe, "fetch_us_large_cap_quotes", lambda min_market_cap, **kw: quotes)
    cache_path = str(tmp_path / "universe_us_50b.csv")

    snapshot = universe.build_us_50b_universe(min_market_cap=50e9, cache_path=cache_path)

    assert set(snapshot.tickers) == {"AAPL", "MSFT", "BRK-B"}
    assert snapshot.min_market_cap == pytest.approx(800_000_000_000.0)
    assert snapshot.max_market_cap == pytest.approx(3_000_000_000_000.0)
    assert snapshot.num_dropped_lookup_failed == 0
    assert snapshot.cache_file == cache_path

    # Cache file was actually written and round-trips through load_universe_cache.
    loaded = universe.load_universe_cache(cache_path)
    assert loaded is not None
    assert set(loaded.tickers) == {"AAPL", "MSFT", "BRK-B"}
    assert loaded.market_caps["AAPL"] == pytest.approx(3_000_000_000_000.0)


def test_build_us_50b_universe_raises_by_default_when_nasdaq_trader_unavailable(tmp_path, monkeypatch):
    # Nasdaq Trader directories are now a REQUIRED eligibility gate (not
    # just metadata) -- without them, screener results can't be verified as
    # actually US-listed/tradable (the SPCX case). An outage there must
    # hard-fail by default rather than silently proceeding screener-only.
    def raise_unavailable():
        raise universe.UniverseError("simulated Nasdaq Trader outage")

    monkeypatch.setattr(universe, "fetch_nasdaq_listed_text", raise_unavailable)
    monkeypatch.setattr(universe, "fetch_other_listed_text", raise_unavailable)

    quotes = [make_quote("AAPL", 3_000_000_000_000.0, name="Apple Inc.")]
    monkeypatch.setattr(universe, "fetch_us_large_cap_quotes", lambda min_market_cap, **kw: quotes)

    with pytest.raises(universe.UniverseError, match="Cannot verify US-listed"):
        universe.build_us_50b_universe(min_market_cap=50e9, cache_path=str(tmp_path / "u.csv"))


def test_build_us_50b_universe_allow_screener_only_continues_on_nasdaq_trader_outage(tmp_path, monkeypatch):
    # allow_screener_only=True is the explicit opt-in escape hatch: with it
    # set, a Nasdaq Trader outage no longer blocks the build, falling back
    # to the screener's own name/exchange fields for metadata.
    def raise_unavailable():
        raise universe.UniverseError("simulated Nasdaq Trader outage")

    monkeypatch.setattr(universe, "fetch_nasdaq_listed_text", raise_unavailable)
    monkeypatch.setattr(universe, "fetch_other_listed_text", raise_unavailable)
    monkeypatch.setattr(universe, "MIN_US_50B_UNIVERSE_SIZE", 1)

    quotes = [make_quote("AAPL", 3_000_000_000_000.0, name="Apple Inc.")]
    monkeypatch.setattr(universe, "fetch_us_large_cap_quotes", lambda min_market_cap, **kw: quotes)

    snapshot = universe.build_us_50b_universe(
        min_market_cap=50e9, cache_path=str(tmp_path / "u.csv"), allow_screener_only=True,
    )
    assert snapshot.tickers == ["AAPL"]
    assert snapshot.names["AAPL"] == "Apple Inc."


def test_build_us_50b_universe_excludes_non_common_stock_name_from_screener_result(tmp_path, monkeypatch):
    # XPRF IS present in the Nasdaq Trader candidate set (added under a
    # benign name, "Example Corp Class X Stock" -- passes that directory's
    # own name filter) so it clears the eligibility gate, but the
    # SCREENER's own reported name flags it as Preferred. This proves the
    # name-pattern safety net on screener results catches things the
    # Nasdaq Trader eligibility gate alone would have let through.
    monkeypatch.setattr(universe, "fetch_nasdaq_listed_text", lambda: NASDAQ_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "fetch_other_listed_text", lambda: OTHER_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "MIN_US_50B_UNIVERSE_SIZE", 1)

    quotes = [
        make_quote("AAPL", 3_000_000_000_000.0),
        make_quote("XPRF", 60_000_000_000.0, name="Example Corp Preferred Series A"),
    ]
    monkeypatch.setattr(universe, "fetch_us_large_cap_quotes", lambda min_market_cap, **kw: quotes)

    snapshot = universe.build_us_50b_universe(min_market_cap=50e9, cache_path=str(tmp_path / "u.csv"))
    assert snapshot.tickers == ["AAPL"]
    assert snapshot.num_excluded_non_common == 1
    assert "XPRF" not in snapshot.tickers


def test_build_us_50b_universe_excludes_screener_result_not_in_nasdaq_trader_candidate_set(tmp_path, monkeypatch):
    # Real-world failure mode this eligibility gate exists for: Yahoo's
    # screener returned a ~$2T "Space Exploration Technologies Corp."
    # (SpaceX -- not actually publicly traded) that the backtest then
    # dropped for missing price data. A screener ticker not present in the
    # Nasdaq Trader candidate set must be excluded and counted, not
    # silently included just because it cleared the market-cap bar.
    monkeypatch.setattr(universe, "fetch_nasdaq_listed_text", lambda: NASDAQ_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "fetch_other_listed_text", lambda: OTHER_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "MIN_US_50B_UNIVERSE_SIZE", 1)

    quotes = [
        make_quote("AAPL", 3_000_000_000_000.0),
        make_quote("SPCX", 1_970_000_000_000.0, name="Space Exploration Technologies Corp."),
    ]
    monkeypatch.setattr(universe, "fetch_us_large_cap_quotes", lambda min_market_cap, **kw: quotes)

    snapshot = universe.build_us_50b_universe(min_market_cap=50e9, cache_path=str(tmp_path / "u.csv"))
    assert snapshot.tickers == ["AAPL"]
    assert "SPCX" not in snapshot.tickers
    assert snapshot.num_excluded_not_listed == 1


def test_build_us_50b_universe_keeps_screener_result_present_in_candidate_set(tmp_path, monkeypatch):
    monkeypatch.setattr(universe, "fetch_nasdaq_listed_text", lambda: NASDAQ_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "fetch_other_listed_text", lambda: OTHER_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "MIN_US_50B_UNIVERSE_SIZE", 1)

    quotes = [make_quote("AAPL", 3_000_000_000_000.0)]  # AAPL is in NASDAQ_LISTED_SAMPLE
    monkeypatch.setattr(universe, "fetch_us_large_cap_quotes", lambda min_market_cap, **kw: quotes)

    snapshot = universe.build_us_50b_universe(min_market_cap=50e9, cache_path=str(tmp_path / "u.csv"))
    assert snapshot.tickers == ["AAPL"]
    assert snapshot.num_excluded_not_listed == 0


def test_names_materially_inconsistent_flags_disjoint_names():
    assert universe._names_materially_inconsistent(
        "Space Exploration Technologies Corp.", "Spectral Capital Corp - Common Stock"
    )


def test_names_materially_inconsistent_allows_shared_company_name():
    assert not universe._names_materially_inconsistent("Apple Inc.", "Apple Inc. - Common Stock")
    assert not universe._names_materially_inconsistent(
        "Microsoft Corporation", "Microsoft Corporation - Common Stock"
    )


def test_names_materially_inconsistent_does_not_flag_when_either_name_empty():
    # Nothing to compare against -- this is a defense-in-depth backstop,
    # not the primary eligibility check, so it must not block a result it
    # can't confidently evaluate.
    assert not universe._names_materially_inconsistent("", "Apple Inc. - Common Stock")
    assert not universe._names_materially_inconsistent("Apple Inc.", "")


def test_build_us_50b_universe_excludes_screener_result_with_identity_mismatch(tmp_path, monkeypatch):
    # SPCX IS present in the Nasdaq Trader candidate set this time (unlike
    # test_build_us_50b_universe_excludes_screener_result_not_in_nasdaq_trader_candidate_set
    # above, which covers the "not present at all" case) -- but its Nasdaq
    # Trader security name describes a completely different, unrelated
    # company than the screener's own claimed name. Presence in the
    # candidate set alone isn't proof the screener's quote is actually
    # about that same listing; the identity-mismatch guard must catch this.
    extended_nasdaq = NASDAQ_LISTED_SAMPLE.replace(
        "File Creation Time",
        "SPCX|Spectral Capital Corp - Common Stock|Q|N|N|100|N|N\nFile Creation Time",
    )
    monkeypatch.setattr(universe, "fetch_nasdaq_listed_text", lambda: extended_nasdaq)
    monkeypatch.setattr(universe, "fetch_other_listed_text", lambda: OTHER_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "MIN_US_50B_UNIVERSE_SIZE", 1)

    quotes = [
        make_quote("AAPL", 3_000_000_000_000.0),
        make_quote("SPCX", 1_970_000_000_000.0, name="Space Exploration Technologies Corp."),
    ]
    monkeypatch.setattr(universe, "fetch_us_large_cap_quotes", lambda min_market_cap, **kw: quotes)

    snapshot = universe.build_us_50b_universe(min_market_cap=50e9, cache_path=str(tmp_path / "u.csv"))
    assert snapshot.tickers == ["AAPL"]
    assert "SPCX" not in snapshot.tickers
    assert snapshot.num_excluded_identity_mismatch == 1
    assert snapshot.num_excluded_not_listed == 0  # it WAS in the candidate set


def test_build_us_50b_universe_keeps_screener_result_with_consistent_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(universe, "fetch_nasdaq_listed_text", lambda: NASDAQ_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "fetch_other_listed_text", lambda: OTHER_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "MIN_US_50B_UNIVERSE_SIZE", 1)

    quotes = [make_quote("AAPL", 3_000_000_000_000.0, name="Apple Inc.")]
    monkeypatch.setattr(universe, "fetch_us_large_cap_quotes", lambda min_market_cap, **kw: quotes)

    snapshot = universe.build_us_50b_universe(min_market_cap=50e9, cache_path=str(tmp_path / "u.csv"))
    assert snapshot.tickers == ["AAPL"]
    assert snapshot.num_excluded_identity_mismatch == 0
    # Nasdaq Trader's own name is authoritative -- not the screener's longName.
    assert snapshot.names["AAPL"] == "Apple Inc. - Common Stock"


def test_build_us_50b_universe_excludes_ticker_with_no_recent_price_data(tmp_path, monkeypatch):
    monkeypatch.setattr(universe, "fetch_nasdaq_listed_text", lambda: NASDAQ_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "fetch_other_listed_text", lambda: OTHER_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "MIN_US_50B_UNIVERSE_SIZE", 1)

    quotes = [
        make_quote("AAPL", 3_000_000_000_000.0),
        make_quote("MSFT", 2_500_000_000_000.0),
    ]
    monkeypatch.setattr(universe, "fetch_us_large_cap_quotes", lambda min_market_cap, **kw: quotes)

    # MSFT looks fine by every name/eligibility check but has no usable
    # recent price history -- the last-line-of-defense check this test
    # covers (the actual backtest-time failure mode: "possibly delisted;
    # no price data found").
    def fake_price_checker(ticker):
        return ticker != "MSFT"

    snapshot = universe.build_us_50b_universe(
        min_market_cap=50e9, cache_path=str(tmp_path / "u.csv"), price_data_checker=fake_price_checker,
    )
    assert snapshot.tickers == ["AAPL"]
    assert "MSFT" not in snapshot.tickers
    assert snapshot.num_excluded_no_price_data == 1

    # Excluded-for-no-price-data tickers must not leak into the cached CSV.
    loaded = universe.load_universe_cache(str(tmp_path / "u.csv"))
    assert "MSFT" not in loaded.tickers


def test_build_us_50b_universe_no_validated_window_recorded_without_backtest_window(tmp_path, monkeypatch):
    # Without a specific backtest window, the price-data check falls back
    # to a fixed recent lookback, and no validated window is recorded --
    # there's nothing window-specific to trust or distrust later.
    monkeypatch.setattr(universe, "fetch_nasdaq_listed_text", lambda: NASDAQ_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "fetch_other_listed_text", lambda: OTHER_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "MIN_US_50B_UNIVERSE_SIZE", 1)
    quotes = [make_quote("AAPL", 3_000_000_000_000.0)]
    monkeypatch.setattr(universe, "fetch_us_large_cap_quotes", lambda min_market_cap, **kw: quotes)

    snapshot = universe.build_us_50b_universe(min_market_cap=50e9, cache_path=str(tmp_path / "u.csv"))
    assert snapshot.price_data_validated_start is None
    assert snapshot.price_data_validated_end is None


def test_build_us_50b_universe_validates_price_data_against_requested_backtest_window(tmp_path, monkeypatch):
    # The real SPCX-shaped bug: a ticker can have a perfectly fine RECENT
    # quote (so a fixed recent-lookback price check would pass it) while
    # having no price history at all over the historical window a backtest
    # actually needs. When a specific backtest window is known,
    # build_us_50b_universe must validate against THAT window, not just
    # "recent" -- and it must not be filtered out earlier by the identity
    # or non-common-stock checks, so this reaches the price-data step at all.
    extended_nasdaq = NASDAQ_LISTED_SAMPLE.replace(
        "File Creation Time",
        "SPCX|Spectral Capital Corp - Common Stock|Q|N|N|100|N|N\nFile Creation Time",
    )
    monkeypatch.setattr(universe, "fetch_nasdaq_listed_text", lambda: extended_nasdaq)
    monkeypatch.setattr(universe, "fetch_other_listed_text", lambda: OTHER_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "MIN_US_50B_UNIVERSE_SIZE", 1)

    quotes = [
        make_quote("AAPL", 3_000_000_000_000.0),
        make_quote("SPCX", 1_970_000_000_000.0, name="Spectral Capital Corp."),
    ]
    monkeypatch.setattr(universe, "fetch_us_large_cap_quotes", lambda min_market_cap, **kw: quotes)

    calls = []

    def fake_price_check(ticker, start=None, end=None):
        calls.append((ticker, start, end))
        return ticker != "SPCX"  # SPCX has no data for the requested historical window

    monkeypatch.setattr(universe, "_default_price_data_available", fake_price_check)

    snapshot = universe.build_us_50b_universe(
        min_market_cap=50e9, cache_path=str(tmp_path / "u.csv"),
        price_data_start="2022-01-01", price_data_end="2024-12-31",
    )
    assert snapshot.tickers == ["AAPL"]
    assert "SPCX" not in snapshot.tickers
    assert snapshot.num_excluded_no_price_data == 1
    assert snapshot.num_excluded_identity_mismatch == 0  # reached the price-data step, not filtered earlier

    # The checker was called with the warmup-adjusted requested window, not
    # a fixed recent lookback.
    spcx_calls = [c for c in calls if c[0] == "SPCX"]
    assert len(spcx_calls) == 1
    _, start_arg, end_arg = spcx_calls[0]
    expected_start = pd.Timestamp("2022-01-01") - pd.Timedelta(days=config.MEAN_REVERSION_WARMUP_CALENDAR_DAYS)
    assert start_arg == expected_start
    assert end_arg == pd.Timestamp("2024-12-31")

    # The validated window is recorded on the snapshot...
    assert snapshot.price_data_validated_start == str(expected_start.date())
    assert snapshot.price_data_validated_end == "2024-12-31"

    # ...and excluded-for-no-price-data tickers must not leak into the cache.
    loaded = universe.load_universe_cache(str(tmp_path / "u.csv"))
    assert "SPCX" not in loaded.tickers


def test_window_covers_true_when_cached_window_contains_requested_window():
    assert universe._window_covers(
        "2019-01-01", "2024-12-31", pd.Timestamp("2021-01-01"), pd.Timestamp("2023-01-01")
    )


def test_window_covers_false_when_no_recorded_window():
    assert not universe._window_covers(None, None, pd.Timestamp("2021-01-01"), pd.Timestamp("2023-01-01"))


def test_window_covers_false_when_requested_window_extends_beyond_recorded():
    assert not universe._window_covers(
        "2022-01-01", "2022-12-31", pd.Timestamp("2021-01-01"), pd.Timestamp("2023-01-01")
    )


def test_snapshot_to_info_includes_price_data_validated_window():
    snapshot = universe.UniverseSnapshot(
        tickers=["AAPL"], market_caps={"AAPL": 3e12},
        price_data_validated_start="2020-01-01", price_data_validated_end="2024-12-31",
    )
    info = universe._snapshot_to_info(snapshot, "us_50b")
    assert info["price_data_validated_start"] == "2020-01-01"
    assert info["price_data_validated_end"] == "2024-12-31"


def test_resolve_mean_reversion_universe_us_50b_revalidates_cache_for_uncovered_window(tmp_path, monkeypatch):
    # Requirement: a us_50b cache validated for one historical window must
    # not be silently reused as if it were valid for a materially
    # different one -- this is the actual mechanism that fixes the "SPCX
    # cached, then dropped by the backtest" bug for a run reusing an
    # existing cache (not just a fresh --refresh-universe build).
    monkeypatch.setattr(universe, "MIN_US_50B_UNIVERSE_SIZE", 1)
    cache_path = tmp_path / "universe_us_50b.csv"
    old_snapshot = universe.UniverseSnapshot(
        tickers=["AAPL", "SPCX"],
        market_caps={"AAPL": 3e12, "SPCX": 2e12},
        names={"AAPL": "Apple Inc.", "SPCX": "Spectral Capital Corp."},
        exchanges={"AAPL": "NMS", "SPCX": "NMS"},
        num_excluded_no_price_data=0,
        snapshot_date="2026-01-01T00:00:00+00:00",
        cache_file=str(cache_path),
        price_data_validated_start="2023-01-01",
        price_data_validated_end="2023-12-31",
    )
    universe.save_universe_cache(old_snapshot, str(cache_path))

    calls = []

    def fake_price_checker(ticker):
        calls.append(ticker)
        return ticker != "SPCX"  # SPCX no longer has data for the newly requested window

    resolution = universe.resolve_mean_reversion_universe(
        mode="us_50b", refresh=False, cache_path=str(cache_path),
        price_data_checker=fake_price_checker,
        backtest_start="2021-01-01", backtest_end="2024-12-31",
    )
    assert resolution.tickers == ["AAPL"]
    assert "SPCX" not in resolution.tickers
    assert "SPCX" in calls  # actually re-validated, not blindly trusted
    assert resolution.info["num_excluded_no_price_data"] == 1

    # Cache re-saved with the excluded ticker dropped and the window recorded.
    reloaded = universe.load_universe_cache(str(cache_path))
    assert "SPCX" not in reloaded.tickers
    assert reloaded.price_data_validated_start is not None


def test_resolve_mean_reversion_universe_us_50b_skips_revalidation_when_window_covered(tmp_path):
    # The inverse of the above: when the cache's recorded validated window
    # already covers what this run needs, it must NOT re-check price data
    # (avoiding needless network calls / cache rewrites on every run).
    cache_path = tmp_path / "universe_us_50b.csv"
    snapshot = universe.UniverseSnapshot(
        tickers=["AAPL"], market_caps={"AAPL": 3e12}, names={"AAPL": "Apple Inc."},
        exchanges={"AAPL": "NMS"}, snapshot_date="2026-01-01T00:00:00+00:00",
        cache_file=str(cache_path),
        price_data_validated_start="2019-01-01", price_data_validated_end="2024-12-31",
    )
    universe.save_universe_cache(snapshot, str(cache_path))

    def fail_if_called(ticker):
        raise AssertionError("price_data_checker must not be called when the cached window already covers the request")

    resolution = universe.resolve_mean_reversion_universe(
        mode="us_50b", refresh=False, cache_path=str(cache_path),
        price_data_checker=fail_if_called,
        backtest_start="2022-01-01", backtest_end="2023-12-31",
    )
    assert resolution.tickers == ["AAPL"]


def test_build_us_50b_universe_report_metadata_includes_all_exclusion_counts(tmp_path, monkeypatch):
    extended_nasdaq = NASDAQ_LISTED_SAMPLE.replace(
        "File Creation Time",
        "SPCX|Spectral Capital Corp - Common Stock|Q|N|N|100|N|N\nFile Creation Time",
    )
    monkeypatch.setattr(universe, "fetch_nasdaq_listed_text", lambda: extended_nasdaq)
    monkeypatch.setattr(universe, "fetch_other_listed_text", lambda: OTHER_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "MIN_US_50B_UNIVERSE_SIZE", 1)

    quotes = [
        make_quote("AAPL", 3_000_000_000_000.0),
        make_quote("SPCX", 1_970_000_000_000.0, name="Space Exploration Technologies Corp."),
        make_quote("XPRF", 60_000_000_000.0, name="Example Corp Preferred Series A"),
    ]
    monkeypatch.setattr(universe, "fetch_us_large_cap_quotes", lambda min_market_cap, **kw: quotes)

    snapshot = universe.build_us_50b_universe(min_market_cap=50e9, cache_path=str(tmp_path / "u.csv"))
    assert snapshot.tickers == ["AAPL"]
    assert snapshot.num_excluded_identity_mismatch == 1  # SPCX
    assert snapshot.num_excluded_non_common == 1  # XPRF
    assert snapshot.num_excluded_no_price_data == 0
    assert snapshot.num_duplicate_companies_collapsed == 0

    info = universe._snapshot_to_info(snapshot, "us_50b")
    assert info["num_excluded_identity_mismatch"] == 1
    assert info["num_excluded_no_price_data"] == 0


def test_company_key_strips_corporate_suffix_and_share_class_wording():
    assert universe._company_key("Alphabet Inc. Class A") == universe._company_key("Alphabet Inc. Class C")
    assert universe._company_key("Berkshire Hathaway Inc. Class A Common Stock") == \
        universe._company_key("Berkshire Hathaway Inc. Class B Common Stock")
    assert universe._company_key("Apple Inc. - Common Stock") != universe._company_key("Microsoft Corporation")


def test_company_key_strips_capital_stock_and_new_common_stock_wording():
    # Real live Nasdaq Trader names for the same company's dual share
    # classes don't always agree on which generic word follows the class
    # letter -- "Capital Stock" vs "Common Stock", or a leading "New" --
    # so both must be stripped, not just "Common Stock" alone, or these
    # pairs fail to collapse into a single company key.
    assert universe._company_key("Alphabet Inc. - Class A Common Stock") == \
        universe._company_key("Alphabet Inc. - Class C Capital Stock")
    assert universe._company_key("Berkshire Hathaway Inc. Common Stock") == \
        universe._company_key("Berkshire Hathaway Inc. New Common Stock")


def test_dedupe_by_company_collapses_dual_class_shares_keeps_higher_cap():
    market_caps = {"GOOGL": 2.0e12, "GOOG": 2.1e12, "AAPL": 3.0e12}
    names = {
        "GOOGL": "Alphabet Inc. Class A", "GOOG": "Alphabet Inc. Class C",
        "AAPL": "Apple Inc. - Common Stock",
    }
    exchanges = {"GOOGL": "NMS", "GOOG": "NMS", "AAPL": "NMS"}

    kept_caps, kept_names, kept_exchanges, num_collapsed = universe._dedupe_by_company(
        market_caps, names, exchanges
    )
    assert set(kept_caps) == {"GOOG", "AAPL"}  # GOOG has the higher cap of the pair
    assert num_collapsed == 1
    assert kept_names["GOOG"] == "Alphabet Inc. Class C"


def test_dedupe_by_company_no_duplicates_is_a_no_op():
    market_caps = {"AAPL": 3.0e12, "MSFT": 2.5e12}
    names = {"AAPL": "Apple Inc. - Common Stock", "MSFT": "Microsoft Corporation - Common Stock"}
    exchanges = {"AAPL": "NMS", "MSFT": "NMS"}

    kept_caps, kept_names, kept_exchanges, num_collapsed = universe._dedupe_by_company(
        market_caps, names, exchanges
    )
    assert kept_caps == market_caps
    assert num_collapsed == 0


def test_dedupe_by_company_collapses_live_shape_alphabet_names():
    # Exact live-shape Nasdaq Trader names from the smoke-test bug report --
    # GOOGL/GOOG must collapse to one ticker despite "Common Stock" vs.
    # "Capital Stock".
    market_caps = {"GOOGL": 2.0e12, "GOOG": 2.1e12}
    names = {
        "GOOGL": "Alphabet Inc. - Class A Common Stock",
        "GOOG": "Alphabet Inc. - Class C Capital Stock",
    }
    exchanges = {"GOOGL": "NMS", "GOOG": "NMS"}

    kept_caps, _kept_names, _kept_exchanges, num_collapsed = universe._dedupe_by_company(
        market_caps, names, exchanges
    )
    assert set(kept_caps) == {"GOOG"}  # higher of the two market caps
    assert num_collapsed == 1


def test_dedupe_by_company_collapses_live_shape_berkshire_names():
    # Exact live-shape Nasdaq Trader names from the smoke-test bug report --
    # BRK-A/BRK-B must collapse to one ticker despite "Common Stock" vs.
    # "New Common Stock".
    market_caps = {"BRK-A": 900_000_000_000.0, "BRK-B": 850_000_000_000.0}
    names = {
        "BRK-A": "Berkshire Hathaway Inc. Common Stock",
        "BRK-B": "Berkshire Hathaway Inc. New Common Stock",
    }
    exchanges = {"BRK-A": "N", "BRK-B": "N"}

    kept_caps, _kept_names, _kept_exchanges, num_collapsed = universe._dedupe_by_company(
        market_caps, names, exchanges
    )
    assert set(kept_caps) == {"BRK-A"}  # higher of the two market caps
    assert num_collapsed == 1


def test_build_us_50b_universe_dedupes_dual_class_shares_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(universe, "fetch_nasdaq_listed_text", lambda: NASDAQ_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "fetch_other_listed_text", lambda: OTHER_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "MIN_US_50B_UNIVERSE_SIZE", 1)

    # BRK-B is in the Nasdaq Trader candidate set (via BRK.B in OTHER_LISTED_SAMPLE);
    # BRK-A isn't in the sample fixture, so add it as a distinct quote whose
    # normalized ticker also needs to pass the eligibility gate -- append a
    # matching row for this test via a locally extended candidate text.
    extended_other = OTHER_LISTED_SAMPLE.replace(
        "File Creation Time",
        "BRK.A|Berkshire Hathaway Inc. Class A Common Stock|N|BRK.A|N|100|N|BRK.A\nFile Creation Time",
    )
    monkeypatch.setattr(universe, "fetch_other_listed_text", lambda: extended_other)

    quotes = [
        make_quote("BRK-A", 900_000_000_000.0, name="Berkshire Hathaway Inc. Class A Common Stock"),
        make_quote("BRK-B", 850_000_000_000.0, name="Berkshire Hathaway Inc. Class B Common Stock"),
    ]
    monkeypatch.setattr(universe, "fetch_us_large_cap_quotes", lambda min_market_cap, **kw: quotes)

    snapshot = universe.build_us_50b_universe(min_market_cap=50e9, cache_path=str(tmp_path / "u.csv"))
    assert snapshot.tickers == ["BRK-A"]  # higher of the two market caps
    assert snapshot.num_duplicate_companies_collapsed == 1


def test_build_us_50b_universe_dedupes_live_shape_googl_goog_end_to_end(tmp_path, monkeypatch):
    # Reproduces the exact live-shape smoke-test output: GOOGL and GOOG both
    # surviving the >= $50B universe with real Nasdaq Trader security names
    # ("Common Stock" vs. "Capital Stock") that the old _company_key
    # pattern failed to collapse.
    extended_nasdaq = NASDAQ_LISTED_SAMPLE.replace(
        "File Creation Time",
        "GOOGL|Alphabet Inc. - Class A Common Stock|Q|N|N|100|N|N\n"
        "GOOG|Alphabet Inc. - Class C Capital Stock|Q|N|N|100|N|N\n"
        "File Creation Time",
    )
    monkeypatch.setattr(universe, "fetch_nasdaq_listed_text", lambda: extended_nasdaq)
    monkeypatch.setattr(universe, "fetch_other_listed_text", lambda: OTHER_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "MIN_US_50B_UNIVERSE_SIZE", 1)

    quotes = [
        make_quote("GOOGL", 2_000_000_000_000.0, name="Alphabet Inc."),
        make_quote("GOOG", 2_100_000_000_000.0, name="Alphabet Inc."),
    ]
    monkeypatch.setattr(universe, "fetch_us_large_cap_quotes", lambda min_market_cap, **kw: quotes)

    snapshot = universe.build_us_50b_universe(min_market_cap=50e9, cache_path=str(tmp_path / "u.csv"))
    assert snapshot.tickers == ["GOOG"]  # higher of the two market caps
    assert "GOOGL" not in snapshot.tickers
    assert snapshot.num_duplicate_companies_collapsed == 1


def test_build_us_50b_universe_dedupes_live_shape_brk_a_brk_b_end_to_end(tmp_path, monkeypatch):
    # Reproduces the exact live-shape smoke-test output: BRK-A and BRK-B
    # both surviving the >= $50B universe with real Nasdaq Trader security
    # names ("Common Stock" vs. "New Common Stock") that the old
    # _company_key pattern failed to collapse. Uses a standalone
    # other-listed text (not OTHER_LISTED_SAMPLE, which already has its own
    # BRK.B row under a different name) so there's no duplicate-symbol
    # collision in _standardize_and_filter's drop_duplicates.
    other_listed_with_berkshire = (
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
        "BRK.A|Berkshire Hathaway Inc. Common Stock|N|BRK.A|N|100|N|BRK.A\n"
        "BRK.B|Berkshire Hathaway Inc. New Common Stock|N|BRK.B|N|100|N|BRK.B\n"
        "File Creation Time: 0708202600:00\n"
    )
    monkeypatch.setattr(universe, "fetch_nasdaq_listed_text", lambda: NASDAQ_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "fetch_other_listed_text", lambda: other_listed_with_berkshire)
    monkeypatch.setattr(universe, "MIN_US_50B_UNIVERSE_SIZE", 1)

    quotes = [
        make_quote("BRK-A", 900_000_000_000.0, name="Berkshire Hathaway Inc."),
        make_quote("BRK-B", 850_000_000_000.0, name="Berkshire Hathaway Inc."),
    ]
    monkeypatch.setattr(universe, "fetch_us_large_cap_quotes", lambda min_market_cap, **kw: quotes)

    snapshot = universe.build_us_50b_universe(min_market_cap=50e9, cache_path=str(tmp_path / "u.csv"))
    assert snapshot.tickers == ["BRK-A"]  # higher of the two market caps
    assert "BRK-B" not in snapshot.tickers
    assert snapshot.num_duplicate_companies_collapsed == 1


def test_resolve_mean_reversion_universe_revalidation_passes_window_to_default_checker(
    tmp_path, monkeypatch
):
    # Regression test for the Cycle F bug: when no price_data_checker is
    # injected, cache revalidation must wrap _default_price_data_available
    # with the requested window (start/end), NOT call it with ticker-only
    # (which silently falls back to a recent lookback and misses the
    # SPCX-class problem for historical windows).
    #
    # Shape: cache has AAPL + SPCX, validated for 2023 only. New run
    # requests 2021-06-15 to 2024-12-31 (not covered). Revalidation must
    # call _default_price_data_available(ticker, start=..., end=...) --
    # not _default_price_data_available(ticker) -- so SPCX is excluded for
    # lacking data over the historical window, not passed on the basis of
    # a recent-data check it would have survived.
    monkeypatch.setattr(universe, "MIN_US_50B_UNIVERSE_SIZE", 1)
    cache_path = tmp_path / "universe_us_50b.csv"
    old_snapshot = universe.UniverseSnapshot(
        tickers=["AAPL", "SPCX"],
        market_caps={"AAPL": 3e12, "SPCX": 2e12},
        names={"AAPL": "Apple Inc.", "SPCX": "Spectral Capital Corp."},
        exchanges={"AAPL": "NMS", "SPCX": "NMS"},
        num_excluded_no_price_data=0,
        snapshot_date="2026-01-01T00:00:00+00:00",
        cache_file=str(cache_path),
        price_data_validated_start="2023-01-01",
        price_data_validated_end="2023-12-31",
    )
    universe.save_universe_cache(old_snapshot, str(cache_path))

    calls = []

    def fake_default_checker(ticker, start=None, end=None):
        calls.append((ticker, start, end))
        # SPCX has no data for the old historical window (only recent data)
        return ticker != "SPCX"

    # Override the autouse stub with our recording version
    monkeypatch.setattr(universe, "_default_price_data_available", fake_default_checker)

    # No price_data_checker injected -- must create a window-aware wrapper internally
    resolution = universe.resolve_mean_reversion_universe(
        mode="us_50b", refresh=False, cache_path=str(cache_path),
        backtest_start="2021-06-15", backtest_end="2024-12-31",
    )
    assert "SPCX" not in resolution.tickers
    assert "AAPL" in resolution.tickers

    # _default_price_data_available must have been called with the warmup-
    # adjusted window -- not with start=None (which is the recent-lookback fallback).
    assert len(calls) > 0, "_default_price_data_available was never called (revalidation skipped?)"
    expected_start = pd.Timestamp("2021-06-15") - pd.Timedelta(
        days=config.MEAN_REVERSION_WARMUP_CALENDAR_DAYS
    )
    for ticker, start, end in calls:
        assert start is not None, (
            f"_default_price_data_available called for {ticker!r} with start=None -- "
            f"fell back to recent-lookback instead of checking the requested backtest window"
        )
        assert start == expected_start, (
            f"Expected warmup-adjusted start {expected_start.date()}, got {start}"
        )
        assert end == pd.Timestamp("2024-12-31"), (
            f"Expected end 2024-12-31, got {end}"
        )


def test_neither_fresh_build_nor_revalidation_falls_back_to_recent_lookback_when_window_supplied(
    tmp_path, monkeypatch
):
    # Regression-prevention test: both the fresh-build path (build_us_50b_universe
    # with price_data_start/end) and the cache-revalidation path (resolve_
    # mean_reversion_universe loading a cache with an uncovered window) must
    # call _default_price_data_available with explicit start/end args when a
    # backtest window is supplied -- never with start=None (recent-lookback fallback).

    # --- Fresh build path ---
    extended_nasdaq = NASDAQ_LISTED_SAMPLE.replace(
        "File Creation Time",
        "SPCX|Spectral Capital Corp - Common Stock|Q|N|N|100|N|N\nFile Creation Time",
    )
    monkeypatch.setattr(universe, "fetch_nasdaq_listed_text", lambda: extended_nasdaq)
    monkeypatch.setattr(universe, "fetch_other_listed_text", lambda: OTHER_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "MIN_US_50B_UNIVERSE_SIZE", 1)

    quotes = [make_quote("AAPL", 3e12), make_quote("SPCX", 2e12, name="Spectral Capital Corp.")]
    monkeypatch.setattr(universe, "fetch_us_large_cap_quotes", lambda min_market_cap, **kw: quotes)

    fresh_calls = []

    def fresh_checker(ticker, start=None, end=None):
        fresh_calls.append((ticker, start, end))
        return ticker != "SPCX"

    monkeypatch.setattr(universe, "_default_price_data_available", fresh_checker)

    universe.build_us_50b_universe(
        min_market_cap=50e9, cache_path=str(tmp_path / "fresh.csv"),
        price_data_start="2021-06-15", price_data_end="2024-12-31",
    )
    for ticker, start, end in fresh_calls:
        assert start is not None, (
            f"Fresh build called _default_price_data_available for {ticker!r} with start=None"
        )

    # --- Cache revalidation path ---
    cache_path = tmp_path / "cached.csv"
    old_snapshot = universe.UniverseSnapshot(
        tickers=["AAPL", "SPCX"],
        market_caps={"AAPL": 3e12, "SPCX": 2e12},
        names={"AAPL": "Apple Inc.", "SPCX": "Spectral Capital Corp."},
        exchanges={"AAPL": "NMS", "SPCX": "NMS"},
        num_excluded_no_price_data=0,
        snapshot_date="2026-01-01T00:00:00+00:00",
        cache_file=str(cache_path),
        price_data_validated_start="2023-01-01",
        price_data_validated_end="2023-12-31",
    )
    universe.save_universe_cache(old_snapshot, str(cache_path))

    reval_calls = []

    def reval_checker(ticker, start=None, end=None):
        reval_calls.append((ticker, start, end))
        return ticker != "SPCX"

    monkeypatch.setattr(universe, "_default_price_data_available", reval_checker)

    universe.resolve_mean_reversion_universe(
        mode="us_50b", refresh=False, cache_path=str(cache_path),
        backtest_start="2021-06-15", backtest_end="2024-12-31",
    )
    assert len(reval_calls) > 0, "Revalidation never called _default_price_data_available"
    for ticker, start, end in reval_calls:
        assert start is not None, (
            f"Revalidation called _default_price_data_available for {ticker!r} with start=None"
        )


def test_build_us_50b_universe_counts_malformed_quote_as_unparseable_not_blocking(tmp_path, monkeypatch):
    # A quote missing a symbol or market cap must be counted as a failed/
    # unparseable result and skipped, without blocking the rest of the
    # build -- the practical equivalent, in the new bulk-screener world, of
    # "one bad ticker doesn't hang or abort the whole universe build."
    monkeypatch.setattr(universe, "fetch_nasdaq_listed_text", lambda: NASDAQ_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "fetch_other_listed_text", lambda: OTHER_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "MIN_US_50B_UNIVERSE_SIZE", 1)

    quotes = [
        make_quote("AAPL", 3_000_000_000_000.0),
        {"symbol": "BROKEN"},  # no market cap field at all
        {"intradaymarketcap": 1e12},  # no symbol
    ]
    monkeypatch.setattr(universe, "fetch_us_large_cap_quotes", lambda min_market_cap, **kw: quotes)

    with pytest.warns(UserWarning, match="could not be parsed"):
        snapshot = universe.build_us_50b_universe(min_market_cap=50e9, cache_path=str(tmp_path / "u.csv"))
    assert snapshot.tickers == ["AAPL"]
    assert snapshot.num_dropped_lookup_failed == 2


def test_build_us_50b_universe_hard_fails_when_too_few_qualify(tmp_path, monkeypatch):
    monkeypatch.setattr(universe, "fetch_nasdaq_listed_text", lambda: NASDAQ_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "fetch_other_listed_text", lambda: OTHER_LISTED_SAMPLE)
    # Nothing clears the $50B bar -- resulting universe would be empty.
    quotes = [make_quote("AAPL", 1_000_000_000.0), make_quote("MSFT", 1_000_000_000.0)]
    monkeypatch.setattr(universe, "fetch_us_large_cap_quotes", lambda min_market_cap, **kw: quotes)

    with pytest.raises(universe.UniverseError, match="Only 0 ticker"):
        universe.build_us_50b_universe(min_market_cap=50e9, cache_path=str(tmp_path / "u.csv"))


def test_build_us_50b_universe_warns_on_high_lookup_failure_fraction(tmp_path, monkeypatch):
    monkeypatch.setattr(universe, "fetch_nasdaq_listed_text", lambda: NASDAQ_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "fetch_other_listed_text", lambda: OTHER_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "MIN_US_50B_UNIVERSE_SIZE", 1)

    quotes = [make_quote("AAPL", 3_000_000_000_000.0)] + [{"symbol": f"BAD{i}"} for i in range(5)]
    monkeypatch.setattr(universe, "fetch_us_large_cap_quotes", lambda min_market_cap, **kw: quotes)

    with pytest.warns(UserWarning, match="could not be parsed"):
        snapshot = universe.build_us_50b_universe(
            min_market_cap=50e9, cache_path=str(tmp_path / "u.csv")
        )
    assert snapshot.tickers == ["AAPL"]


def test_build_us_50b_universe_reports_progress_milestones(tmp_path, monkeypatch):
    monkeypatch.setattr(universe, "fetch_nasdaq_listed_text", lambda: NASDAQ_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "fetch_other_listed_text", lambda: OTHER_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "MIN_US_50B_UNIVERSE_SIZE", 1)
    quotes = [make_quote("AAPL", 3_000_000_000_000.0)]
    monkeypatch.setattr(universe, "fetch_us_large_cap_quotes", lambda min_market_cap, **kw: quotes)

    messages = []
    universe.build_us_50b_universe(
        min_market_cap=50e9, cache_path=str(tmp_path / "u.csv"), progress=messages.append
    )
    joined = "\n".join(messages)
    assert "candidate" in joined
    assert "qualifying" in joined


def test_build_us_50b_universe_threads_max_candidates_to_screener(tmp_path, monkeypatch):
    monkeypatch.setattr(universe, "fetch_nasdaq_listed_text", lambda: NASDAQ_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "fetch_other_listed_text", lambda: OTHER_LISTED_SAMPLE)
    monkeypatch.setattr(universe, "MIN_US_50B_UNIVERSE_SIZE", 1)

    received = {}

    def fake_fetch(min_market_cap, max_results=None, progress=None):
        received["max_results"] = max_results
        return [make_quote("AAPL", 3_000_000_000_000.0)]

    monkeypatch.setattr(universe, "fetch_us_large_cap_quotes", fake_fetch)
    universe.build_us_50b_universe(min_market_cap=50e9, cache_path=str(tmp_path / "u.csv"), max_candidates=7)
    assert received["max_results"] == 7


def test_load_universe_cache_tolerant_of_minimal_ticker_only_csv(tmp_path):
    path = tmp_path / "custom_universe.csv"
    pd.DataFrame({"ticker": ["AAPL", "MSFT", "GOOGL"]}).to_csv(path, index=False)

    snapshot = universe.load_universe_cache(str(path))
    assert snapshot is not None
    assert snapshot.tickers == ["AAPL", "MSFT", "GOOGL"]
    assert snapshot.market_caps == {}
    assert snapshot.min_market_cap is None


def test_load_universe_cache_missing_file_returns_none(tmp_path):
    assert universe.load_universe_cache(str(tmp_path / "nope.csv")) is None


def test_resolve_mean_reversion_universe_default_matches_config():
    from src import config

    resolution = universe.resolve_mean_reversion_universe(mode="default")
    assert resolution.tickers == list(config.MEAN_REVERSION_UNIVERSE)
    assert resolution.info["mode"] == "default"
    assert resolution.info["cache_file"] is None


def test_resolve_mean_reversion_universe_csv_override_wins_over_mode(tmp_path):
    path = tmp_path / "custom.csv"
    pd.DataFrame({"ticker": ["NVDA", "AMD"]}).to_csv(path, index=False)

    resolution = universe.resolve_mean_reversion_universe(mode="default", csv_path=str(path))
    assert resolution.tickers == ["NVDA", "AMD"]
    assert resolution.info["mode"] == "csv"


def test_resolve_mean_reversion_universe_csv_missing_raises():
    with pytest.raises(universe.UniverseError):
        universe.resolve_mean_reversion_universe(mode="default", csv_path="/nonexistent/path.csv")


def test_resolve_mean_reversion_universe_us_50b_uses_existing_cache_without_refresh(tmp_path, monkeypatch):
    cache_path = tmp_path / "universe_us_50b.csv"
    pd.DataFrame({
        "ticker": ["AAPL", "MSFT"],
        "name": ["Apple Inc.", "Microsoft Corp."],
        "market_cap": [3e12, 2.5e12],
        "exchange": ["NASDAQ", "NASDAQ"],
        "snapshot_date": ["2026-01-01T00:00:00+00:00"] * 2,
    }).to_csv(cache_path, index=False)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("build_us_50b_universe should not be called when a fresh cache exists")

    monkeypatch.setattr(universe, "build_us_50b_universe", fail_if_called)

    resolution = universe.resolve_mean_reversion_universe(
        mode="us_50b", refresh=False, cache_path=str(cache_path)
    )
    assert resolution.tickers == ["AAPL", "MSFT"]
    assert resolution.info["mode"] == "us_50b"
    assert resolution.info["min_market_cap"] == pytest.approx(2.5e12)
    assert resolution.info["max_market_cap"] == pytest.approx(3e12)


def test_resolve_mean_reversion_universe_us_50b_refresh_rebuilds(tmp_path, monkeypatch):
    cache_path = tmp_path / "universe_us_50b.csv"
    pd.DataFrame({"ticker": ["OLD"], "market_cap": [1e12]}).to_csv(cache_path, index=False)

    rebuilt = universe.UniverseSnapshot(
        tickers=["NEW"], market_caps={"NEW": 1e12}, cache_file=str(cache_path),
    )
    calls = []

    def fake_build(
        min_market_cap, cache_path, max_candidates=None, progress=None,
        allow_screener_only=False, price_data_checker=None,
        price_data_start=None, price_data_end=None,
    ):
        calls.append((min_market_cap, cache_path))
        return rebuilt

    monkeypatch.setattr(universe, "build_us_50b_universe", fake_build)

    resolution = universe.resolve_mean_reversion_universe(
        mode="us_50b", refresh=True, cache_path=str(cache_path)
    )
    assert resolution.tickers == ["NEW"]
    assert len(calls) == 1


def test_resolve_mean_reversion_universe_us_50b_threads_max_candidates_and_progress(tmp_path, monkeypatch):
    cache_path = tmp_path / "universe_us_50b.csv"
    received = {}

    def fake_build(
        min_market_cap, cache_path, max_candidates=None, progress=None,
        allow_screener_only=False, price_data_checker=None,
        price_data_start=None, price_data_end=None,
    ):
        received["max_candidates"] = max_candidates
        received["progress"] = progress
        return universe.UniverseSnapshot(tickers=["NEW"], cache_file=str(cache_path))

    monkeypatch.setattr(universe, "build_us_50b_universe", fake_build)
    my_progress = lambda msg: None  # noqa: E731

    universe.resolve_mean_reversion_universe(
        mode="us_50b", refresh=True, cache_path=str(cache_path),
        max_candidates=42, progress=my_progress,
    )
    assert received["max_candidates"] == 42
    assert received["progress"] is my_progress


def test_resolve_mean_reversion_universe_unknown_mode_raises():
    with pytest.raises(ValueError):
        universe.resolve_mean_reversion_universe(mode="not_a_real_mode")
