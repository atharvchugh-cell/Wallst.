import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src import data as data_module


def make_df(dates, start_price=100.0):
    closes = [start_price + i for i in range(len(dates))]
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1000}, index=dates
    )


@pytest.fixture
def tmp_cache_dir(tmp_path):
    return str(tmp_path)


def test_cache_metadata_mismatch_forces_refetch(tmp_cache_dir, monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=10)
    cached_df = make_df(dates)
    meta = {
        "ticker": "AAPL", "cached_start": str(dates.min()), "cached_end": str(dates.max()),
        "yfinance_version": "OLD_VERSION", "auto_adjust": True, "interval": "1d",
        "last_full_refresh": datetime.now(timezone.utc).isoformat(), "fetch_success": True,
    }
    data_module._write_cache("AAPL", tmp_cache_dir, cached_df, meta)

    fetch_calls = []

    def fake_fetch_range(ticker, start, end, auto_adjust, interval):
        fetch_calls.append((ticker, start, end))
        return make_df(pd.bdate_range(start, end))

    monkeypatch.setattr(data_module, "_fetch_range", fake_fetch_range)
    monkeypatch.setattr(data_module.yf, "__version__", "NEW_VERSION")

    data_module.get_price_history("AAPL", dates.min(), dates.max(), cache_dir=tmp_cache_dir)
    assert len(fetch_calls) == 1  # forced refetch despite a "fresh" cache, due to version mismatch


def test_stale_cache_triggers_full_refresh_not_delta(tmp_cache_dir, monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=10)
    cached_df = make_df(dates)
    stale_time = datetime.now(timezone.utc) - timedelta(days=30)
    meta = {
        "ticker": "AAPL", "cached_start": str(dates.min()), "cached_end": str(dates.max()),
        "yfinance_version": data_module.yf.__version__, "auto_adjust": True, "interval": "1d",
        "last_full_refresh": stale_time.isoformat(), "fetch_success": True,
    }
    data_module._write_cache("AAPL", tmp_cache_dir, cached_df, meta)

    fetch_calls = []

    def fake_fetch_range(ticker, start, end, auto_adjust, interval):
        fetch_calls.append((start, end))
        return make_df(pd.bdate_range(start, end))

    monkeypatch.setattr(data_module, "_fetch_range", fake_fetch_range)
    data_module.get_price_history("AAPL", dates.min(), dates.max(), cache_dir=tmp_cache_dir, refresh_threshold_days=7)
    assert len(fetch_calls) == 1
    # A full refresh re-fetches the WHOLE range, not just a small delta tail
    fetched_start, fetched_end = fetch_calls[0]
    assert pd.Timestamp(fetched_start) == dates.min()


def test_fresh_cache_with_extended_end_does_delta_fetch_only(tmp_cache_dir, monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=10)
    cached_df = make_df(dates)
    meta = {
        "ticker": "AAPL", "cached_start": str(dates.min()), "cached_end": str(dates.max()),
        "yfinance_version": data_module.yf.__version__, "auto_adjust": True, "interval": "1d",
        "last_full_refresh": datetime.now(timezone.utc).isoformat(), "fetch_success": True,
    }
    data_module._write_cache("AAPL", tmp_cache_dir, cached_df, meta)

    fetch_calls = []
    new_end = dates.max() + pd.Timedelta(days=10)

    def fake_fetch_range(ticker, start, end, auto_adjust, interval):
        fetch_calls.append((start, end))
        return make_df(pd.bdate_range(start, end), start_price=999.0)

    monkeypatch.setattr(data_module, "_fetch_range", fake_fetch_range)
    result = data_module.get_price_history("AAPL", dates.min(), new_end, cache_dir=tmp_cache_dir)
    assert len(fetch_calls) == 1
    fetched_start, fetched_end = fetch_calls[0]
    # Delta fetch deliberately widens its start to a few days BEFORE the
    # cached end (not the original start, and not the exact missing tail) --
    # yfinance handles very narrow date ranges unreliably in practice, so a
    # single-day delta request is avoided by design. The overlap is deduped
    # on merge; see test_stale_cache_narrow_tail_delta_widens_and_merges.
    assert dates.min() < pd.Timestamp(fetched_start) <= dates.max()
    assert result.index.min() == dates.min()  # old cached rows preserved


def test_failed_fetch_does_not_corrupt_existing_cache(tmp_cache_dir, monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=10)
    cached_df = make_df(dates)
    meta = {
        "ticker": "AAPL", "cached_start": str(dates.min()), "cached_end": str(dates.max()),
        "yfinance_version": data_module.yf.__version__, "auto_adjust": True, "interval": "1d",
        "last_full_refresh": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(), "fetch_success": True,
    }
    data_module._write_cache("AAPL", tmp_cache_dir, cached_df, meta)

    def failing_fetch(ticker, start, end, auto_adjust, interval):
        raise data_module.FetchError("simulated network failure")

    monkeypatch.setattr(data_module, "_fetch_range", failing_fetch)
    with pytest.raises(data_module.FetchError):
        data_module.get_price_history("AAPL", dates.min(), dates.max(), cache_dir=tmp_cache_dir, refresh_threshold_days=7)

    # Original cache file must still be intact and readable
    reloaded_df, reloaded_meta = data_module._read_cache("AAPL", tmp_cache_dir)
    assert reloaded_df is not None
    assert len(reloaded_df) == 10
    assert reloaded_meta["fetch_success"] is True


def test_canonical_calendar_drops_gaps_not_forward_fill():
    ref = pd.DataFrame({"Close": range(5)}, index=pd.bdate_range("2024-01-01", periods=5))
    calendar = data_module.build_canonical_calendar(ref, "2024-01-01", "2024-01-05")

    gappy = pd.DataFrame({"Close": [1.0, 2.0, None, 4.0, 5.0]}, index=calendar)
    gaps = data_module.find_gaps(gappy, calendar)
    assert len(gaps) == 1
    assert gaps[0] == calendar[2]

    late_starting = pd.DataFrame({"Close": [3.0, 4.0, 5.0]}, index=calendar[2:])
    assert len(data_module.find_gaps(late_starting, calendar)) == 0  # starting later isn't a gap


def test_exclude_unfinalized_today_drops_todays_bar():
    today = pd.Timestamp.now().normalize()
    calendar = pd.DatetimeIndex(pd.bdate_range(today - pd.Timedelta(days=5), today))
    trimmed, was_dropped = data_module.exclude_unfinalized_today(calendar)
    if calendar[-1] == today:
        assert was_dropped is True
        assert trimmed[-1] < today
    else:
        assert was_dropped is False


def test_end_date_is_inclusive_in_cache_slice(tmp_cache_dir, monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=10)

    def fake_fetch_range(ticker, start, end, auto_adjust, interval):
        return make_df(pd.bdate_range(start, end))

    monkeypatch.setattr(data_module, "_fetch_range", fake_fetch_range)
    result = data_module.get_price_history("AAPL", dates[0], dates[-1], cache_dir=tmp_cache_dir)
    assert result.index.max() == dates[-1]  # the end date itself is included


def test_fetch_range_requests_one_day_past_end_since_yfinance_end_is_exclusive(monkeypatch):
    # yfinance's `end` appears to exclude that calendar date in practice --
    # inferred from observed backtest runs where the last bar came back short
    # of the requested end, but NOT independently confirmed against a live
    # download (this sandbox has no network egress to finance.yahoo.com). This
    # test only proves _fetch_range asks yf.download for end+1 day; it does
    # not prove yfinance's real exclusivity behavior. That still needs a real
    # local run to confirm.
    captured = {}

    class FakeDownload:
        def __call__(self, ticker, start, end, auto_adjust, interval, progress):
            captured["end"] = end
            dates = pd.bdate_range(start, pd.Timestamp(end) - pd.Timedelta(days=1))
            closes = [100.0] * len(dates)
            return pd.DataFrame(
                {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": [1000] * len(dates)},
                index=dates,
            )

    monkeypatch.setattr(data_module.yf, "download", FakeDownload())
    requested_end = pd.Timestamp("2024-01-10")
    df = data_module._fetch_range("AAPL", pd.Timestamp("2024-01-01"), requested_end, True, "1d")
    assert captured["end"] == requested_end + pd.Timedelta(days=1)
    assert df.index.max() == requested_end  # the requested end date's own bar is present


def test_stale_cache_fallback_warns_instead_of_silent(tmp_cache_dir, monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=10)
    cached_df = make_df(dates)
    meta = {
        "ticker": "AAPL", "cached_start": str(dates.min()), "cached_end": str(dates.max()),
        "yfinance_version": data_module.yf.__version__, "auto_adjust": True, "interval": "1d",
        "last_full_refresh": datetime.now(timezone.utc).isoformat(), "fetch_success": True,
    }
    data_module._write_cache("AAPL", tmp_cache_dir, cached_df, meta)

    def failing_delta_fetch(ticker, start, end, auto_adjust, interval):
        raise data_module.FetchError("simulated network failure")

    monkeypatch.setattr(data_module, "_fetch_range", failing_delta_fetch)
    new_end = dates.max() + pd.Timedelta(days=10)
    with pytest.warns(UserWarning, match="AAPL"):
        result = data_module.get_price_history("AAPL", dates.min(), new_end, cache_dir=tmp_cache_dir)
    # Falls back to what's cached rather than raising -- but the warning above proves it's not silent.
    assert result.index.max() == dates.max()


def test_stale_cache_narrow_tail_delta_widens_and_merges(tmp_cache_dir, monkeypatch):
    # Reproduces the real bug seen on a local run: cache ends 2024-12-30,
    # requested end is 2024-12-31 (one trading day later). A naive delta fetch
    # would request only that single missing day, which yfinance handled
    # unreliably in practice ("possibly delisted; no price data found" on
    # SPY, a highly liquid ticker, for a one-day range). The delta fetch must
    # widen its start well before the cached end so yfinance gets a
    # reasonable multi-day window, then merge/dedup the overlap correctly.
    cached_dates = pd.bdate_range("2024-12-01", "2024-12-30")
    cached_df = make_df(cached_dates)
    meta = {
        "ticker": "SPY", "cached_start": str(cached_dates.min()), "cached_end": str(cached_dates.max()),
        "yfinance_version": data_module.yf.__version__, "auto_adjust": True, "interval": "1d",
        "last_full_refresh": datetime.now(timezone.utc).isoformat(), "fetch_success": True,
    }
    data_module._write_cache("SPY", tmp_cache_dir, cached_df, meta)

    fetch_calls = []

    def fake_fetch_range(ticker, start, end, auto_adjust, interval):
        fetch_calls.append((start, end))
        # Simulate yfinance's observed real-world failure on a too-narrow
        # (<=2 day) range, even though a wider range works fine.
        if (pd.Timestamp(end) - pd.Timestamp(start)).days <= 2:
            raise data_module.FetchError("simulated: possibly delisted; no price data found")
        return make_df(pd.bdate_range(start, end))

    monkeypatch.setattr(data_module, "_fetch_range", fake_fetch_range)
    requested_end = pd.Timestamp("2024-12-31")
    result = data_module.get_price_history("SPY", cached_dates.min(), requested_end, cache_dir=tmp_cache_dir)

    assert len(fetch_calls) == 1
    fetched_start, fetched_end = fetch_calls[0]
    assert (pd.Timestamp(fetched_end) - pd.Timestamp(fetched_start)).days > 2  # wide enough to succeed
    assert pd.Timestamp(fetched_start) <= cached_dates.max()  # deliberately overlaps the cached tail
    assert result.index.max() == requested_end  # 2024-12-31 successfully merged in


def test_benchmark_hard_fails_when_short_of_a_requested_weekday_end(monkeypatch):
    dates = pd.bdate_range("2024-12-01", "2024-12-30")  # short of 2024-12-31 (a Tuesday)

    def fake_get_price_history(ticker, start, end, **kwargs):
        return make_df(dates)

    monkeypatch.setattr(data_module, "get_price_history", fake_get_price_history)
    with pytest.raises(data_module.FetchError, match="SPY"):
        data_module.get_benchmark_data("2024-12-01", "2024-12-31")


def test_benchmark_does_not_hard_fail_when_short_end_is_a_weekend(monkeypatch):
    dates = pd.bdate_range("2024-12-01", "2024-12-27")  # 2024-12-28/29 are a weekend

    def fake_get_price_history(ticker, start, end, **kwargs):
        return make_df(dates)

    monkeypatch.setattr(data_module, "get_price_history", fake_get_price_history)
    result = data_module.get_benchmark_data("2024-12-01", "2024-12-29")  # a Sunday
    assert result.index.max() == dates.max()  # no exception -- weekend shortfall is expected


# --- filter_by_sufficient_history: per-window listing-history guard ------------

def test_filter_by_sufficient_history_keeps_full_history_excludes_young_and_missing():
    window_start = "2019-01-01"
    warmup = 200
    # OLD has data reaching well before the warmup cutoff (2018-06-15-ish);
    # YOUNG IPO'd mid-2021 (long after the window start); GONE has no data at
    # all for this window (empty frame).
    old_dates = pd.bdate_range("2015-01-01", "2024-12-31")
    young_dates = pd.bdate_range("2021-06-01", "2024-12-31")
    price_data = {
        "OLD": make_df(old_dates),
        "YOUNG": make_df(young_dates),
        "GONE": pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]),
    }
    kept, excluded = data_module.filter_by_sufficient_history(
        price_data, ["OLD", "YOUNG", "GONE"], window_start, warmup
    )
    assert kept == ["OLD"]
    excluded_map = dict(excluded)
    assert set(excluded_map) == {"YOUNG", "GONE"}
    assert "insufficient history" in excluded_map["YOUNG"]
    assert "no price data" in excluded_map["GONE"]
    # Every exclusion carries a human-readable reason (never silent).
    assert all(reason for _t, reason in excluded)


def test_filter_by_sufficient_history_is_a_noop_for_a_universe_that_predates_the_window():
    # The default-universe guarantee: every member has deep history, so the
    # filter excludes nobody -- behavior for the hardcoded universe is
    # unchanged by adding this filter.
    dates = pd.bdate_range("1995-01-01", "2024-12-31")
    price_data = {t: make_df(dates) for t in ["AAA", "BBB", "CCC"]}
    kept, excluded = data_module.filter_by_sufficient_history(
        price_data, ["AAA", "BBB", "CCC"], "2019-01-01", 200
    )
    assert kept == ["AAA", "BBB", "CCC"]
    assert excluded == []


def test_filter_by_sufficient_history_tolerance_allows_cutoff_on_a_weekend():
    # A ticker whose first bar is a few days after the exact warmup cutoff
    # (because the cutoff landed on a weekend/holiday) must NOT be wrongly
    # excluded -- the small tolerance covers that.
    window_start = pd.Timestamp("2020-01-02")
    warmup = 100
    required_start = window_start - pd.Timedelta(days=warmup)  # 2019-09-24
    # First bar 3 days after required_start -- within the 7-day tolerance.
    dates = pd.bdate_range(required_start + pd.Timedelta(days=3), "2021-12-31")
    price_data = {"BORDER": make_df(dates)}
    kept, excluded = data_module.filter_by_sufficient_history(
        price_data, ["BORDER"], window_start, warmup
    )
    assert kept == ["BORDER"]
    assert excluded == []


def test_yfinance_logger_quieted_to_reduce_delisted_noise():
    # The noisy per-ticker "possibly delisted; no price data found" lines are
    # suppressed by raising the yfinance logger threshold (importing data.py
    # sets it). Our own FetchError/warnings are unaffected -- exclusions are
    # still reported structurally.
    import logging
    assert logging.getLogger("yfinance").level >= logging.ERROR
