import pandas as pd
import pytest
import requests

from src import universe

NASDAQ_LISTED_SAMPLE = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
MSFT|Microsoft Corporation - Common Stock|Q|N|N|100|N|N
ZZZZ|Test Company - Common Stock|Q|Y|N|100|N|N
QQQ|Invesco QQQ Trust|Q|N|N|100|Y|N
AAWW|Atlas Air Worldwide Holdings Warrant|Q|N|N|100|N|N
EXRT|Example Realty Trust Inc. Common Stock|Q|N|N|100|N|N
File Creation Time: 0708202600:00
"""

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
    assert symbols == {"AAPL", "MSFT", "EXRT"}
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
    assert set(combined["yahoo_ticker"]) == {"AAPL", "MSFT", "EXRT", "BRK-B", "JPM"}
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


def make_quote(symbol, market_cap, name=None, exchange="NMS"):
    return {
        "symbol": symbol,
        "longName": name or f"{symbol} Inc. Common Stock",
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


def test_build_us_50b_universe_continues_when_nasdaq_trader_directories_unavailable(tmp_path, monkeypatch):
    # The screener call -- not the Nasdaq Trader directories -- determines
    # universe membership now; Nasdaq Trader only supplies supplementary
    # name/exchange metadata, so its outage must not fail the whole build
    # as long as the screener itself succeeds (each quote already carries
    # its own name/exchange, used here instead of the unavailable fallback).
    def raise_unavailable():
        raise universe.UniverseError("simulated Nasdaq Trader outage")

    monkeypatch.setattr(universe, "fetch_nasdaq_listed_text", raise_unavailable)
    monkeypatch.setattr(universe, "fetch_other_listed_text", raise_unavailable)
    monkeypatch.setattr(universe, "MIN_US_50B_UNIVERSE_SIZE", 1)

    quotes = [make_quote("AAPL", 3_000_000_000_000.0, name="Apple Inc.")]
    monkeypatch.setattr(universe, "fetch_us_large_cap_quotes", lambda min_market_cap, **kw: quotes)

    snapshot = universe.build_us_50b_universe(min_market_cap=50e9, cache_path=str(tmp_path / "u.csv"))
    assert snapshot.tickers == ["AAPL"]
    assert snapshot.names["AAPL"] == "Apple Inc."


def test_build_us_50b_universe_excludes_non_common_stock_name_from_screener_result(tmp_path, monkeypatch):
    # A quote that clears the market-cap bar but whose name marks it as a
    # non-common-stock instrument must still be excluded -- Yahoo's screener
    # scoping to quoteType=EQUITY isn't a guaranteed substitute for the same
    # name-pattern safety net applied to the Nasdaq Trader directories.
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
    assert "XPRF" not in snapshot.tickers


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

    def fake_build(min_market_cap, cache_path, max_candidates=None, progress=None):
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

    def fake_build(min_market_cap, cache_path, max_candidates=None, progress=None):
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
