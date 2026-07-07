import pandas as pd
import numpy as np
import pytest

from src.strategies.sector_rotation import SectorRotationStrategy
from src.market_view import MarketDataView
from src.indicators import month_end_dates, trailing_month_end_return
from src import data as data_module


def make_price_df(closes, dates):
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1000},
        index=dates,
    )


def build_trending_universe(dates, trends):
    """trends: dict[ticker] -> per-day drift (deterministic, no noise, so
    trailing returns are exactly controllable)."""
    price_data = {}
    for ticker, drift in trends.items():
        series = 100.0 * np.cumprod(1 + np.full(len(dates), drift))
        price_data[ticker] = make_price_df(series, dates)
    return price_data


def run_strategy(strat, price_data, walk_calendar, full_calendar, sleeve_equity=7500.0):
    strat.reset()
    enriched = strat.prepare(price_data, walk_calendar, start=walk_calendar[0])
    events_by_day = []
    for d in walk_calendar:
        view = MarketDataView(enriched, as_of=d, calendar=walk_calendar)
        events_by_day.append((d, strat.on_day(d, view, sleeve_equity)))
    return events_by_day, enriched


def test_no_lookahead_mutating_every_future_date():
    dates = pd.bdate_range("2023-01-01", periods=400)
    base_trends = {"XLK": 0.0015, "XLF": 0.0003, "XLE": -0.0005}
    walk = dates[130:]
    month_end_signal = month_end_dates(dates[(dates >= walk[0])])[0]
    signal_idx = list(dates).index(month_end_signal)

    def build(mutation_offset=None):
        price_data = build_trending_universe(dates, base_trends)
        if mutation_offset is not None:
            for t in price_data:
                price_data[t].iloc[signal_idx + mutation_offset, price_data[t].columns.get_loc("Close")] *= 1.5
        return price_data

    strat_baseline = SectorRotationStrategy(universe=["XLK", "XLF", "XLE"], lookback_months=3, top_k=2)
    baseline_events, _ = run_strategy(strat_baseline, build(), walk, dates)
    baseline_at_signal = [e for d, evs in baseline_events if d == month_end_signal for e in evs]
    assert len(baseline_at_signal) == 3  # all 3 ETFs get a target event

    for offset in [1, 2, 5, 20]:
        strat = SectorRotationStrategy(universe=["XLK", "XLF", "XLE"], lookback_months=3, top_k=2)
        mutated_events, _ = run_strategy(strat, build(mutation_offset=offset), walk, dates)
        mutated_at_signal = [e for d, evs in mutated_events if d == month_end_signal for e in evs]
        assert len(mutated_at_signal) == len(baseline_at_signal)
        for a, b in zip(
            sorted(baseline_at_signal, key=lambda e: e.ticker), sorted(mutated_at_signal, key=lambda e: e.ticker)
        ):
            assert a.ticker == b.ticker
            assert a.target_weight == pytest.approx(b.target_weight)
            assert a.fill_date == b.fill_date


def test_top_k_ranking_matches_engineered_returns():
    dates = pd.bdate_range("2023-01-01", periods=400)
    trends = {"XLK": 0.002, "XLF": 0.0005, "XLE": -0.001, "XLV": 0.0015}
    walk = dates[130:]
    price_data = build_trending_universe(dates, trends)
    strat = SectorRotationStrategy(universe=list(trends.keys()), lookback_months=3, top_k=2)
    events, _ = run_strategy(strat, price_data, walk, dates)
    first_rebalance_day = next(d for d, evs in events if evs)
    day_events = {e.ticker: e for d, evs in events if d == first_rebalance_day for e in evs}
    assert len(day_events) == 4  # all 4 tickers get a target event
    top_2 = {t for t, e in day_events.items() if e.target_weight > 0}
    assert top_2 == {"XLK", "XLV"}  # two strongest trends
    for t in top_2:
        assert day_events[t].target_weight == pytest.approx(0.5)
    for t in set(trends) - top_2:
        assert day_events[t].target_weight == 0.0


def test_month_end_resolves_correctly_across_weekend():
    # Dec 31 2023 was a Sunday -- last trading day of Dec 2023 is Dec 29 (Fri).
    idx = pd.bdate_range("2023-12-01", "2024-01-05")
    me = month_end_dates(idx)
    assert pd.Timestamp("2023-12-29") in me
    assert pd.Timestamp("2023-12-31") not in me


def test_all_11_target_events_every_rebalance_not_just_transactions():
    dates = pd.bdate_range("2023-01-01", periods=400)
    trends = {f"E{i}": 0.0005 * (i - 5) for i in range(11)}  # 11 synthetic ETFs, varied trends
    walk = dates[130:]
    price_data = build_trending_universe(dates, trends)
    strat = SectorRotationStrategy(universe=list(trends.keys()), lookback_months=3, top_k=3)
    events, _ = run_strategy(strat, price_data, walk, dates)
    rebalance_days = [(d, evs) for d, evs in events if evs]
    assert len(rebalance_days) > 0
    for d, evs in rebalance_days:
        assert len(evs) == 11  # every ETF gets a target event, even weight=0 ones
        nonzero = [e for e in evs if e.target_weight > 0]
        assert len(nonzero) == 3
        assert sum(e.target_weight for e in nonzero) == pytest.approx(1.0)


def test_persistent_top_k_member_still_gets_event_next_rebalance():
    dates = pd.bdate_range("2023-01-01", periods=400)
    # XLK is always the strongest -- stays top-K every month.
    trends = {"XLK": 0.002, "XLF": 0.0003, "XLE": -0.0005}
    walk = dates[130:]
    price_data = build_trending_universe(dates, trends)
    strat = SectorRotationStrategy(universe=list(trends.keys()), lookback_months=3, top_k=2)
    events, _ = run_strategy(strat, price_data, walk, dates)
    rebalance_days = [(d, evs) for d, evs in events if evs]
    assert len(rebalance_days) >= 2
    for d, evs in rebalance_days[:2]:
        xlk_event = next(e for e in evs if e.ticker == "XLK")
        assert xlk_event.target_weight == pytest.approx(0.5)
        assert xlk_event.reason == "rebalance"


def test_effective_start_clipping_uses_inception_plus_lookback():
    price_data = {
        "OLD": make_price_df([100.0] * 10, pd.date_range("2010-01-01", periods=10)),
        "YOUNG": make_price_df([100.0] * 10, pd.date_range("2020-06-01", periods=10)),
    }
    effective_start, first_dates = data_module.compute_sector_effective_start(
        price_data, requested_start="2015-01-01", lookback_months=3
    )
    assert effective_start == pd.Timestamp("2020-06-01") + pd.DateOffset(months=3)
    assert first_dates["YOUNG"] == pd.Timestamp("2020-06-01")

    # If requested start is already later than inception+lookback, no clipping.
    effective_start2, _ = data_module.compute_sector_effective_start(
        price_data, requested_start="2022-01-01", lookback_months=3
    )
    assert effective_start2 == pd.Timestamp("2022-01-01")


def test_fetch_failure_on_required_etf_hard_fails(monkeypatch):
    good_df = make_price_df([100.0] * 5, pd.bdate_range("2020-01-01", periods=5))

    def fake_get_price_history(ticker, start, end, **kwargs):
        if ticker == "BAD_TICKER":
            raise data_module.FetchError(f"No data returned for {ticker}")
        return good_df

    monkeypatch.setattr(data_module, "get_price_history", fake_get_price_history)

    with pytest.raises(data_module.FetchError):
        data_module.get_price_data(
            ["XLK", "BAD_TICKER"], "2020-01-01", "2020-02-01",
            warmup_calendar_days=0, hard_fail_on_missing=True,
        )

    # Mean-reversion-style tolerant mode: the bad ticker is dropped, not fatal.
    price_data, dropped = data_module.get_price_data(
        ["XLK", "BAD_TICKER"], "2020-01-01", "2020-02-01",
        warmup_calendar_days=0, hard_fail_on_missing=False,
    )
    assert "XLK" in price_data
    assert dropped == [("BAD_TICKER", "No data returned for BAD_TICKER")]
