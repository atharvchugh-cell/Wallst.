import numpy as np
import pandas as pd
import pytest

from src import config
from src.market_view import MarketDataView
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.mean_reversion_filtered import FilteredMeanReversionStrategy


def make_price_df(closes, dates):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1000},
        index=dates,
    )


DATES = pd.bdate_range("2023-01-02", periods=260)
WALK = DATES[100:]


def declining_series(n_flat=110, n_decline=150, daily=-0.015, start=100.0):
    """Flat warmup then a steady decline -- drives RSI deep under any entry
    threshold without any single crash-shaped week (5-day return stays well
    above -15% at 1.5%/day)."""
    flat = np.full(n_flat, start)
    decline = start * np.cumprod(1 + np.full(n_decline, daily))
    return np.concatenate([flat, decline])


def rising_series(n=260, daily=0.001, start=400.0):
    return start * np.cumprod(1 + np.full(n, daily))


def run_strategy(strat, price_data, walk_calendar, sleeve_equity=7500.0):
    strat.reset()
    enriched = strat.prepare(price_data, walk_calendar, start=walk_calendar[0])
    events_by_day = []
    for d in walk_calendar:
        view = MarketDataView(enriched, as_of=d, calendar=walk_calendar)
        events_by_day.append((d, strat.on_day(d, view, sleeve_equity)))
    return events_by_day


def event_signature(events_by_day):
    """Comparable event stream ignoring the strategy-name label (baseline vs
    filtered strategies stamp different names on otherwise identical events)."""
    return [
        (str(d.date()), e.ticker, round(e.target_weight, 9), str(e.fill_date.date()), e.reason)
        for d, evs in events_by_day for e in evs
    ]


def test_permissive_filters_reproduce_baseline_exactly():
    # With a rising SPY (always above a tiny regime SMA) and an effectively
    # disabled knife guard, the filtered strategy must produce EXACTLY the
    # baseline's event stream -- proving the _entry_allowed hook refactor
    # did not change baseline behavior and the subclass adds only filters.
    price_data = {
        "AAA": make_price_df(declining_series(), DATES),
        "BBB": make_price_df(rising_series(), DATES),
        "SPY": make_price_df(rising_series(start=300.0), DATES),
    }
    baseline = MeanReversionStrategy(universe=["AAA", "BBB"])
    baseline_events = run_strategy(baseline, {k: v.copy() for k, v in price_data.items()}, WALK)

    filtered = FilteredMeanReversionStrategy(
        universe=["AAA", "BBB"], regime_sma_period=2, knife_return_threshold=-0.99,
    )
    filtered_events = run_strategy(filtered, {k: v.copy() for k, v in price_data.items()}, WALK)

    assert event_signature(baseline_events) == event_signature(filtered_events)
    assert any(e for _d, evs in baseline_events for e in evs), "test is vacuous -- no events at all"


def test_regime_gate_blocks_entries_when_spy_below_sma():
    # SPY declines throughout -> always below its regime SMA -> the filtered
    # strategy must never enter, while the baseline on identical stock data does.
    price_data = {
        "AAA": make_price_df(declining_series(), DATES),
        "SPY": make_price_df(declining_series(daily=-0.005), DATES),
    }
    baseline_events = run_strategy(MeanReversionStrategy(universe=["AAA"]), {k: v.copy() for k, v in price_data.items()}, WALK)
    assert any(e.reason == "entry_rsi" for _d, evs in baseline_events for e in evs)

    filtered = FilteredMeanReversionStrategy(universe=["AAA"], regime_sma_period=10)
    filtered_events = run_strategy(filtered, {k: v.copy() for k, v in price_data.items()}, WALK)
    assert not any(e.reason == "entry_rsi" for _d, evs in filtered_events for e in evs)


def test_knife_guard_blocks_established_crash_but_allows_orderly_dip():
    # The guard's contract: reject a candidate whose trailing 5-day return
    # already shows a crash (<= -15%); allow an orderly dip. (Day 1 of a
    # crash is indistinguishable from a dip without foresight -- the guard
    # deliberately makes no claim about it; see the strategy docstring.)
    crash = np.concatenate([np.full(110, 100.0), 100.0 * np.cumprod(1 + np.full(150, -0.05))])
    price_data = {
        "CRASH": make_price_df(crash, DATES),
        "DIP": make_price_df(declining_series(), DATES),
        "SPY": make_price_df(rising_series(start=300.0), DATES),
    }
    filtered = FilteredMeanReversionStrategy(universe=["CRASH", "DIP"], regime_sma_period=10)
    filtered.reset()
    enriched = filtered.prepare(price_data, WALK, start=WALK[0])

    # Direct hook check on a day the crash is established: 10 trading days
    # into the -5%/day decline the 5-day return is ~ -22.6%, far beyond -15%.
    established_crash_day = DATES[120]
    view = MarketDataView(enriched, as_of=established_crash_day, calendar=WALK)
    assert filtered._entry_allowed("CRASH", established_crash_day, view) is False
    assert filtered._entry_allowed("DIP", established_crash_day, view) is True

    # Integration: over the whole walk, no entry event may fire for ANY
    # ticker on a day its own 5-day return had already breached the guard.
    events = run_strategy(filtered, {k: v.copy() for k, v in price_data.items()}, WALK)
    for d, evs in events:
        for e in evs:
            if e.reason == "entry_rsi":
                knife = enriched[e.ticker].loc[d, "KnifeReturn"]
                assert knife > config.FILTERED_MR_KNIFE_RETURN_THRESHOLD, (
                    f"{e.ticker} entered on {d.date()} with 5-day return {knife:.1%}"
                )


def test_exits_are_never_gated_by_the_filters():
    # Force the harshest case: _entry_allowed always False. A position that
    # is already open must STILL exit (stop-loss) -- risk-reducing exits are
    # structurally outside the filter hook.
    price_data = {
        "AAA": make_price_df(declining_series(), DATES),
        "SPY": make_price_df(declining_series(daily=-0.005), DATES),
    }
    strat = FilteredMeanReversionStrategy(universe=["AAA"], regime_sma_period=10)
    strat.reset()
    enriched = strat.prepare(price_data, WALK, start=WALK[0])

    st = strat._state["AAA"]
    st.status = "open"
    st.entry_fill_date = WALK[0]
    st.entry_fill_price = float(enriched["AAA"].loc[WALK[0], "Close"])
    st.days_held = 0

    exit_events = []
    for d in WALK:
        view = MarketDataView(enriched, as_of=d, calendar=WALK)
        exit_events.extend(e for e in strat.on_day(d, view, 7500.0) if e.target_weight == 0.0)
        if exit_events:
            break
    assert exit_events, "open position never exited despite falling price and risk-off regime"
    assert exit_events[0].reason in ("exit_stop_loss", "exit_timeout", "exit_sma", "exit_rsi")


def test_prepare_hard_fails_without_regime_ticker_data():
    price_data = {"AAA": make_price_df(declining_series(), DATES)}
    strat = FilteredMeanReversionStrategy(universe=["AAA"])
    strat.reset()
    with pytest.raises(ValueError, match="requires SPY"):
        strat.prepare(price_data, WALK, start=WALK[0])


def test_spy_is_signal_only_never_in_tradable_universe():
    price_data = {
        "AAA": make_price_df(declining_series(), DATES),
        "SPY": make_price_df(rising_series(start=300.0), DATES),
    }
    strat = FilteredMeanReversionStrategy(universe=["AAA"])
    strat.reset()
    enriched = strat.prepare(price_data, WALK, start=WALK[0])
    assert "SPY" in enriched  # data available to the MarketDataView...
    assert "SPY" not in strat.universe  # ...but structurally untradable
    assert strat.signal_tickers == [config.REGIME_TICKER]


def test_describe_discloses_filter_params_and_inherited_thresholds():
    strat = FilteredMeanReversionStrategy(universe=["AAA"])
    info = strat.describe()
    assert info["name"] == "mean_reversion_filtered"
    assert info["params"]["regime_sma_period"] == config.REGIME_SMA_PERIOD
    assert info["params"]["knife_return_threshold"] == config.FILTERED_MR_KNIFE_RETURN_THRESHOLD
    # Inherited baseline thresholds still disclosed -- nothing hidden.
    assert info["params"]["rsi_entry_threshold"] == config.RSI_ENTRY_THRESHOLD
    assert any("V-bottom" in a for a in info["assumptions"])


def test_constructor_validation():
    with pytest.raises(ValueError):
        FilteredMeanReversionStrategy(universe=["AAA"], regime_sma_period=0)
    with pytest.raises(ValueError):
        FilteredMeanReversionStrategy(universe=["AAA"], knife_lookback_days=0)
    with pytest.raises(ValueError):
        FilteredMeanReversionStrategy(universe=["AAA"], knife_return_threshold=0.05)
