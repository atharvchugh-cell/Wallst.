import numpy as np
import pandas as pd
import pytest

from src.indicators import month_end_dates
from src.market_view import MarketDataView
from src.strategies.momentum import MomentumStrategy


def make_price_df(closes, dates):
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1000},
        index=dates,
    )


def build_trending_universe(dates, trends):
    """trends: dict[ticker] -> constant per-day drift. Deterministic, so
    momentum ranks and SMA relationships are exactly controllable."""
    price_data = {}
    for ticker, drift in trends.items():
        series = 100.0 * np.cumprod(1 + np.full(len(dates), drift))
        price_data[ticker] = make_price_df(series, dates)
    return price_data


def run_strategy(strat, price_data, walk_calendar, sleeve_equity=15000.0):
    strat.reset()
    enriched = strat.prepare(price_data, walk_calendar, start=walk_calendar[0])
    events_by_day = []
    for d in walk_calendar:
        view = MarketDataView(enriched, as_of=d, calendar=walk_calendar)
        events_by_day.append((d, strat.on_day(d, view, sleeve_equity)))
    return events_by_day, enriched


# 600 business days: the first ~400 are warmup (200-day SMA + 126-day
# momentum both valid well before the walk), the last 200 are walked.
DATES = pd.bdate_range("2022-01-03", periods=600)
WALK = DATES[400:]


def test_events_only_at_month_ends():
    trends = {"AAA": 0.002, "BBB": 0.001, "CCC": -0.001}
    events, _ = run_strategy(MomentumStrategy(universe=list(trends), top_k=2), build_trending_universe(DATES, trends), WALK)
    m_ends = set(month_end_dates(WALK))
    for d, evs in events:
        if evs:
            assert d in m_ends, f"events emitted on non-month-end {d.date()}"


def test_top_k_ranking_and_equal_weights():
    trends = {"AAA": 0.003, "BBB": 0.002, "CCC": 0.001, "DDD": 0.0005}
    strat = MomentumStrategy(universe=list(trends), top_k=2)
    events, _ = run_strategy(strat, build_trending_universe(DATES, trends), WALK)
    first_rebalance = next((d, evs) for d, evs in events if evs)
    day_events = {e.ticker: e for e in first_rebalance[1]}
    held = {t for t, e in day_events.items() if e.target_weight > 0}
    assert held == {"AAA", "BBB"}  # the two strongest positive trends
    for t in held:
        assert day_events[t].target_weight == pytest.approx(0.5)  # 1/top_k each
    # Names never held and not selected get NO event (no no-op zero targets).
    assert "CCC" not in day_events
    assert "DDD" not in day_events


def test_absolute_momentum_and_trend_filters_produce_cash_fallback():
    # Only AAA is rising; BBB/CCC decline, so they fail BOTH the 126-day
    # absolute-momentum filter and the close>SMA-200 trend filter. With
    # top_k=2 only 1 name qualifies -- weight stays 1/top_k (50%), the rest
    # of the sleeve stays in cash rather than concentrating into AAA.
    trends = {"AAA": 0.002, "BBB": -0.001, "CCC": -0.002}
    strat = MomentumStrategy(universe=list(trends), top_k=2)
    events, _ = run_strategy(strat, build_trending_universe(DATES, trends), WALK)
    first_rebalance = next((d, evs) for d, evs in events if evs)
    nonzero = [e for e in first_rebalance[1] if e.target_weight > 0]
    assert [e.ticker for e in nonzero] == ["AAA"]
    assert nonzero[0].target_weight == pytest.approx(0.5)  # NOT 1.0


def test_holding_that_fails_filters_gets_zero_target_exit():
    # AAA rises for 500 days then crashes 2%/day: by a later month-end its
    # 126-day return is negative and close < SMA-200, so it must be exited
    # (weight 0, reason rebalance_exit) rather than silently kept.
    rising = 100.0 * np.cumprod(1 + np.full(500, 0.0025))
    crashing = rising[-1] * np.cumprod(1 + np.full(100, -0.02))
    aaa = np.concatenate([rising, crashing])
    price_data = {
        "AAA": make_price_df(aaa, DATES),
        "BBB": make_price_df(100.0 * np.cumprod(1 + np.full(600, 0.001)), DATES),
    }
    strat = MomentumStrategy(universe=["AAA", "BBB"], top_k=2)
    events, _ = run_strategy(strat, price_data, WALK)
    exit_events = [e for _d, evs in events for e in evs if e.ticker == "AAA" and e.target_weight == 0.0]
    assert exit_events, "AAA was never exited despite failing both filters"
    assert exit_events[0].reason == "rebalance_exit"


def test_no_lookahead_mutating_every_future_date():
    trends = {"AAA": 0.002, "BBB": 0.001, "CCC": -0.001}
    signal_date = month_end_dates(WALK)[0]
    signal_idx = list(DATES).index(signal_date)

    def build(mutation_offset=None):
        price_data = build_trending_universe(DATES, trends)
        if mutation_offset is not None:
            for t in price_data:
                price_data[t].iloc[signal_idx + mutation_offset, price_data[t].columns.get_loc("Close")] *= 2.0
        return price_data

    baseline_events, _ = run_strategy(MomentumStrategy(universe=list(trends), top_k=2), build(), WALK)
    baseline_at_signal = sorted(
        (e for d, evs in baseline_events if d == signal_date for e in evs), key=lambda e: e.ticker
    )
    assert baseline_at_signal
    for offset in [1, 2, 10, 50]:
        mutated_events, _ = run_strategy(
            MomentumStrategy(universe=list(trends), top_k=2), build(mutation_offset=offset), WALK
        )
        mutated_at_signal = sorted(
            (e for d, evs in mutated_events if d == signal_date for e in evs), key=lambda e: e.ticker
        )
        assert [(e.ticker, e.target_weight) for e in mutated_at_signal] == [
            (e.ticker, e.target_weight) for e in baseline_at_signal
        ]


def test_initial_events_rebalance_from_warmup_data():
    trends = {"AAA": 0.002, "BBB": -0.001}
    price_data = build_trending_universe(DATES, trends)
    strat = MomentumStrategy(universe=list(trends), top_k=1)
    strat.reset()
    enriched = strat.prepare(price_data, WALK, start=WALK[0])
    warmup_view = MarketDataView(enriched, as_of=DATES[399], calendar=DATES)
    events = strat.initial_events(warmup_view, 15000.0)
    held = [e for e in events if e.target_weight > 0]
    assert [e.ticker for e in held] == ["AAA"]
    assert all(e.fill_date is None for e in events)  # engine assigns the walk's first day


def test_prepare_drops_ticker_with_insufficient_warmup():
    trends = {"AAA": 0.002, "BBB": 0.001}
    price_data = build_trending_universe(DATES, trends)
    price_data["SHORT"] = make_price_df(
        100.0 * np.cumprod(1 + np.full(60, 0.001)), DATES[-60:]
    )  # far too short for SMA-200 + 126d momentum
    strat = MomentumStrategy(universe=["AAA", "BBB", "SHORT"], top_k=2)
    strat.reset()
    strat.prepare(price_data, WALK, start=WALK[0])
    assert strat.universe == ["AAA", "BBB"]
    assert any(t == "SHORT" for t, _reason in strat.dropped_tickers)


def test_fill_dates_are_next_trading_day():
    trends = {"AAA": 0.002, "BBB": 0.001}
    events, _ = run_strategy(MomentumStrategy(universe=list(trends), top_k=1), build_trending_universe(DATES, trends), WALK)
    for d, evs in events:
        for e in evs:
            assert e.fill_date > d
            assert e.fill_date in WALK


def test_describe_discloses_params_and_assumptions():
    strat = MomentumStrategy(universe=["AAA"], top_k=3)
    info = strat.describe()
    assert info["name"] == "momentum"
    assert info["family"] == "momentum"
    assert info["params"]["top_k"] == 3
    assert info["params"]["lookback_trading_days"] == 126
    assert any("survivorship" in a.lower() for a in info["assumptions"])


def test_constructor_validation():
    with pytest.raises(ValueError):
        MomentumStrategy(universe=["AAA"], top_k=0)
    with pytest.raises(ValueError):
        MomentumStrategy(universe=["AAA"], lookback_trading_days=0)
    with pytest.raises(ValueError):
        MomentumStrategy(universe=["AAA"], trend_sma_period=-1)
