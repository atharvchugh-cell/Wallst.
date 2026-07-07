import pandas as pd
import numpy as np
import pytest

from src.strategies.mean_reversion import MeanReversionStrategy
from src.market_view import MarketDataView


def make_price_df(closes, dates):
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1000},
        index=dates,
    )


def run_strategy(strat, price_data, calendar):
    """Runs via the real prepare() path (natural RSI/SMA computation)."""
    strat.reset()
    enriched = strat.prepare(price_data, calendar, start=calendar[0])
    all_events = []
    for d in calendar:
        view = MarketDataView(enriched, as_of=d, calendar=calendar)
        all_events.append((d, strat.on_day(d, view, sleeve_equity=7500.0)))
    return all_events


def make_enriched(dates, close, rsi, sma):
    return pd.DataFrame({"Close": close, "RSI_14": rsi, "SMA_50": sma}, index=dates)


def run_on_enriched(strat, enriched, calendar, sleeve_equity=7500.0):
    """Runs directly against hand-crafted Close/RSI_14/SMA_50 columns,
    bypassing prepare()'s real indicator computation -- gives exact control
    over test conditions for state-machine behavior (entry/exit priority,
    timeout counting, pending slots) independent of indicator math (which is
    covered separately in test_indicators.py)."""
    strat.reset()
    events_by_day = []
    for d in calendar:
        view = MarketDataView(enriched, as_of=d, calendar=calendar)
        events_by_day.append((d, strat.on_day(d, view, sleeve_equity)))
    return events_by_day


def dip_and_recover_series(n=150):
    """Flat, then a sharp drop (RSI<30), lingers low, then recovers well
    above SMA-50."""
    return np.concatenate([
        np.full(50, 100.0),
        np.linspace(100, 70, 10),
        np.linspace(70, 71, 20),
        np.linspace(71, 130, 40),
        np.full(30, 130.0),
    ])[:n]


def test_no_lookahead_mutating_every_future_date():
    dates = pd.bdate_range("2024-01-01", periods=150)
    base = dip_and_recover_series(150)
    signal_day_idx = 65  # somewhere in the "linger low" oversold phase

    def build(mutation_offset=None, mutation_value=None):
        series = base.copy()
        if mutation_offset is not None:
            series[signal_day_idx + mutation_offset] = mutation_value
        return {"AAPL": make_price_df(series, dates), "MSFT": make_price_df(np.full(150, 200.0), dates)}

    walk = dates[60:]
    strat_baseline = MeanReversionStrategy(universe=["AAPL", "MSFT"])
    baseline_events = run_strategy(strat_baseline, build(), walk)
    baseline_at_signal_day = [e for d, evs in baseline_events if d == dates[signal_day_idx] for e in evs]

    for offset in [1, 2, 5, 10, 30]:
        strat = MeanReversionStrategy(universe=["AAPL", "MSFT"])
        mutated_events = run_strategy(strat, build(mutation_offset=offset, mutation_value=999.0), walk)
        mutated_at_signal_day = [e for d, evs in mutated_events if d == dates[signal_day_idx] for e in evs]
        assert len(mutated_at_signal_day) == len(baseline_at_signal_day)
        for a, b in zip(baseline_at_signal_day, mutated_at_signal_day):
            assert a.ticker == b.ticker
            assert a.target_weight == b.target_weight
            assert a.reason == b.reason
            assert a.fill_date == b.fill_date


def test_fill_date_is_next_trading_day_after_signal():
    dates = pd.bdate_range("2024-01-01", periods=150)
    series = dip_and_recover_series(150)
    price_data = {"AAPL": make_price_df(series, dates), "MSFT": make_price_df(np.full(150, 200.0), dates)}
    walk = dates[60:]
    strat = MeanReversionStrategy(universe=["AAPL", "MSFT"])
    events = run_strategy(strat, price_data, walk)
    all_events = [e for _, evs in events for e in evs]
    assert len(all_events) > 0
    for e in all_events:
        pos = walk.get_loc(e.signal_date)
        assert walk[pos + 1] == e.fill_date


def test_stop_loss_uses_entry_fill_price_not_signal_close():
    dates = pd.bdate_range("2024-01-01", periods=6)
    # day0 signal (RSI 25<30) at close=25 (irrelevant to stop calc);
    # day1 fill at close=20 -> entry_fill_price=20, stop threshold=18.4.
    # day2 close=19: NOT below 18.4 (correct, fill-price-based) but WOULD be
    # below 25*0.92=23 (wrong, if it mistakenly used the signal-day close).
    # day3 close=17: below 18.4 -> stop-loss should now fire.
    close = [25.0, 20.0, 19.0, 17.0, 17.0, 17.0]
    rsi = [25.0, 40.0, 40.0, 40.0, 40.0, 40.0]  # stays <50 so exit_rsi never confounds
    sma = [200.0] * 6  # close always below SMA so exit_sma never confounds
    enriched = {"AAPL": make_enriched(dates, close, rsi, sma)}
    strat = MeanReversionStrategy(universe=["AAPL"], max_concurrent_positions=1)
    events = run_on_enriched(strat, enriched, dates)

    exit_events = [(d, e) for d, evs in events for e in evs if e.target_weight == 0.0]
    assert len(exit_events) == 1
    signal_date, event = exit_events[0]
    assert event.reason == "exit_stop_loss"
    assert signal_date == dates[3]  # not dates[2], confirming fill-price basis


def test_exit_priority_same_day_stop_loss_wins():
    dates = pd.bdate_range("2024-01-01", periods=4)
    close = [25.0, 20.0, 10.0, 10.0]  # day2: crashes to 10 -- below stop (18.4)
    rsi = [25.0, 40.0, 60.0, 60.0]     # day2 RSI=60 -- exit_rsi condition ALSO true
    sma = [200.0, 200.0, 5.0, 5.0]     # day2 close(10) > sma(5) -- exit_sma ALSO true
    enriched = {"AAPL": make_enriched(dates, close, rsi, sma)}
    strat = MeanReversionStrategy(universe=["AAPL"], max_concurrent_positions=1)
    events = run_on_enriched(strat, enriched, dates)
    exit_events = [e for _, evs in events for e in evs if e.target_weight == 0.0]
    assert len(exit_events) == 1
    assert exit_events[0].reason == "exit_stop_loss"  # highest priority wins


def test_exit_timeout_exact_day():
    dates = pd.bdate_range("2024-01-01", periods=30)
    close = [100.0] * 30
    rsi = [25.0] + [40.0] * 29   # only day0 is the entry signal; RSI stays <50 after
    sma = [200.0] * 30            # close always below SMA -- exit_sma never fires
    enriched = {"AAPL": make_enriched(dates, close, rsi, sma)}
    strat = MeanReversionStrategy(universe=["AAPL"], max_concurrent_positions=1, max_holding_days=20)
    events = run_on_enriched(strat, enriched, dates)

    entry_events = [(d, e) for d, evs in events for e in evs if e.reason == "entry_rsi"]
    assert len(entry_events) == 1
    assert entry_events[0][0] == dates[0]
    assert entry_events[0][1].fill_date == dates[1]  # entry_fill_date

    timeout_events = [(d, e) for d, evs in events for e in evs if e.reason == "exit_timeout"]
    assert len(timeout_events) == 1
    timeout_signal_date, timeout_event = timeout_events[0]
    # entry_fill_date = dates[1] counts as days_held=0; days_held reaches 20
    # exactly at dates[1+20]=dates[21] -- that's the timeout SIGNAL date.
    assert timeout_signal_date == dates[21]
    assert timeout_event.fill_date == dates[22]


def test_pending_slot_enforcement_same_day_multi_signal():
    dates = pd.bdate_range("2024-01-01", periods=5)
    # Day0: A and B both signal (RSI<30); B is more oversold (lower RSI) and
    # should win the single free slot. Day1: C also signals, but the slot is
    # already fully consumed by B (pending/open) -- C must NOT get an entry.
    a_close = [100.0] * 5
    a_rsi = [25.0, 40.0, 40.0, 40.0, 40.0]
    b_close = [100.0] * 5
    b_rsi = [20.0, 40.0, 40.0, 40.0, 40.0]  # lower RSI than A
    c_close = [100.0] * 5
    c_rsi = [40.0, 22.0, 40.0, 40.0, 40.0]  # signals on day1, after the slot is taken
    sma = [200.0] * 5
    enriched = {
        "A": make_enriched(dates, a_close, a_rsi, sma),
        "B": make_enriched(dates, b_close, b_rsi, sma),
        "C": make_enriched(dates, c_close, c_rsi, sma),
    }
    strat = MeanReversionStrategy(universe=["A", "B", "C"], max_concurrent_positions=1)
    events = run_on_enriched(strat, enriched, dates)
    entry_events = [e for _, evs in events for e in evs if e.reason == "entry_rsi"]
    assert len(entry_events) == 1
    assert entry_events[0].ticker == "B"


def test_lowest_rsi_tiebreak_with_two_free_slots():
    dates = pd.bdate_range("2024-01-01", periods=3)
    sma = [200.0] * 3
    enriched = {
        "A": make_enriched(dates, [100.0] * 3, [28.0, 40.0, 40.0], sma),  # least oversold
        "B": make_enriched(dates, [100.0] * 3, [15.0, 40.0, 40.0], sma),  # most oversold
        "C": make_enriched(dates, [100.0] * 3, [22.0, 40.0, 40.0], sma),  # middle
    }
    strat = MeanReversionStrategy(universe=["A", "B", "C"], max_concurrent_positions=2)
    events = run_on_enriched(strat, enriched, dates)
    entry_events = [e for _, evs in events for e in evs if e.reason == "entry_rsi"]
    entered_tickers = {e.ticker for e in entry_events}
    assert entered_tickers == {"B", "C"}  # two lowest-RSI candidates, not A


def test_signal_time_sizing_uses_equity_at_signal_not_a_later_value():
    # requested_notional is filled in by the ENGINE (not the strategy), but
    # we can still confirm the strategy reports a consistent target_weight
    # regardless of what sleeve_equity is passed in on a given call, and that
    # different equity values on different days produce different (engine-
    # computed) requested_notional downstream -- see test_engine.py for the
    # full signal-time-sizing invariant through the engine.
    dates = pd.bdate_range("2024-01-01", periods=2)
    sma = [200.0, 200.0]
    enriched = {"A": make_enriched(dates, [100.0, 100.0], [25.0, 40.0], sma)}
    strat = MeanReversionStrategy(universe=["A"], max_concurrent_positions=1)
    events = run_on_enriched(strat, enriched, dates, sleeve_equity=7500.0)
    entry_events = [e for _, evs in events for e in evs if e.reason == "entry_rsi"]
    assert len(entry_events) == 1
    assert entry_events[0].target_weight == pytest.approx(1.0)
    assert entry_events[0].requested_notional is None  # not yet sized -- engine's job
