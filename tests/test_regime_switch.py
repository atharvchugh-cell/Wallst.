import numpy as np
import pandas as pd
import pytest

from src import config
from src.market_view import MarketDataView
from src.strategies.regime_switch import RegimeSwitchStrategy
from src.strategies.sector_rotation import SectorRotationStrategy


def make_price_df(closes, dates):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1000},
        index=dates,
    )


def build_trending(dates, drift, start=100.0):
    return make_price_df(start * np.cumprod(1 + np.full(len(dates), drift)), dates)


DATES = pd.bdate_range("2023-01-02", periods=400)
WALK = DATES[130:]
ETF_TRENDS = {"XLK": 0.002, "XLF": 0.0005, "XLE": -0.001, "XLV": 0.0015}


def build_universe(dates=DATES):
    return {t: build_trending(dates, drift) for t, drift in ETF_TRENDS.items()}


def run_strategy(strat, price_data, walk_calendar, sleeve_equity=7500.0):
    strat.reset()
    enriched = strat.prepare(price_data, walk_calendar, start=walk_calendar[0])
    events_by_day = []
    for d in walk_calendar:
        view = MarketDataView(enriched, as_of=d, calendar=walk_calendar)
        events_by_day.append((d, strat.on_day(d, view, sleeve_equity)))
    return events_by_day


def test_risk_on_matches_plain_sector_rotation_exactly():
    # Rising SPY -> always risk-on -> the hybrid's rebalance decisions must be
    # IDENTICAL to plain sector rotation on the same ETF data (inherited code
    # path, so ticker selection, weights, and fill dates all match).
    price_data = build_universe()
    rotation_events = run_strategy(
        SectorRotationStrategy(universe=list(ETF_TRENDS), lookback_months=3, top_k=2),
        {k: v.copy() for k, v in price_data.items()}, WALK,
    )
    price_data["SPY"] = build_trending(DATES, 0.001, start=400.0)
    hybrid_events = run_strategy(
        RegimeSwitchStrategy(universe=list(ETF_TRENDS), lookback_months=3, top_k=2, regime_sma_period=10),
        price_data, WALK,
    )
    sig = lambda events: [
        (str(d.date()), e.ticker, round(e.target_weight, 9), str(e.fill_date.date()))
        for d, evs in events for e in evs
    ]
    assert sig(rotation_events) == sig(hybrid_events)
    assert any(evs for _d, evs in rotation_events), "test is vacuous -- no rebalances at all"


def test_risk_off_targets_all_cash():
    price_data = build_universe()
    price_data["SPY"] = build_trending(DATES, -0.002, start=400.0)  # falling -> below SMA
    events = run_strategy(
        RegimeSwitchStrategy(universe=list(ETF_TRENDS), lookback_months=3, top_k=2, regime_sma_period=10),
        price_data, WALK,
    )
    all_events = [e for _d, evs in events for e in evs]
    assert all_events, "no rebalance events at all"
    assert all(e.target_weight == 0.0 for e in all_events)
    assert all(e.reason == "risk_off" for e in all_events)


def test_regime_flip_switches_between_rotation_and_cash():
    # SPY rises for the first ~60 walk days then declines steeply: early
    # month-ends are risk-on (nonzero weights), later ones risk-off (all 0).
    flip_idx = 190
    spy_up = 400.0 * np.cumprod(1 + np.full(flip_idx, 0.001))
    spy_down = spy_up[-1] * np.cumprod(1 + np.full(len(DATES) - flip_idx, -0.004))
    price_data = build_universe()
    price_data["SPY"] = make_price_df(np.concatenate([spy_up, spy_down]), DATES)

    events = run_strategy(
        RegimeSwitchStrategy(universe=list(ETF_TRENDS), lookback_months=3, top_k=2, regime_sma_period=10),
        price_data, WALK,
    )
    rebalances = [(d, evs) for d, evs in events if evs]
    assert len(rebalances) >= 3
    first_day, first_evs = rebalances[0]
    last_day, last_evs = rebalances[-1]
    assert any(e.target_weight > 0 for e in first_evs), "expected early rebalance to be risk-on"
    assert all(e.target_weight == 0.0 for e in last_evs), "expected late rebalance to be risk-off"
    assert any(e.reason == "risk_off" for e in last_evs)


def test_no_lookahead_future_spy_mutation_cannot_change_regime():
    price_data = build_universe()
    price_data["SPY"] = build_trending(DATES, 0.001, start=400.0)
    strat = RegimeSwitchStrategy(universe=list(ETF_TRENDS), lookback_months=3, top_k=2, regime_sma_period=10)
    baseline = run_strategy(strat, {k: v.copy() for k, v in price_data.items()}, WALK)
    first_rebalance_day = next(d for d, evs in baseline if evs)
    signal_idx = list(DATES).index(first_rebalance_day)

    mutated = {k: v.copy() for k, v in price_data.items()}
    # Crash SPY on every date AFTER the signal date -- decisions at the
    # signal date must be unchanged.
    mutated["SPY"].iloc[signal_idx + 1:, mutated["SPY"].columns.get_loc("Close")] *= 0.5
    strat2 = RegimeSwitchStrategy(universe=list(ETF_TRENDS), lookback_months=3, top_k=2, regime_sma_period=10)
    mutated_events = run_strategy(strat2, mutated, WALK)

    base_sig = [(e.ticker, e.target_weight) for d, evs in baseline if d == first_rebalance_day for e in evs]
    mut_sig = [(e.ticker, e.target_weight) for d, evs in mutated_events if d == first_rebalance_day for e in evs]
    assert base_sig == mut_sig


def test_prepare_hard_fails_without_regime_ticker_data():
    strat = RegimeSwitchStrategy(universe=list(ETF_TRENDS), lookback_months=3, top_k=2)
    strat.reset()
    with pytest.raises(ValueError, match="requires SPY"):
        strat.prepare(build_universe(), WALK, start=WALK[0])


def test_describe_discloses_regime_params_and_inherited_rotation_params():
    strat = RegimeSwitchStrategy(universe=list(ETF_TRENDS), lookback_months=3, top_k=2)
    info = strat.describe()
    assert info["name"] == "regime_switch"
    assert info["family"] == "regime_switch"
    assert info["params"]["regime_sma_period"] == config.REGIME_SMA_PERIOD
    assert info["params"]["top_k"] == 2  # inherited rotation knob still disclosed
    assert info["params"]["risk_off_allocation"] == "100% cash (earns 0%)"
    assert any("V-bottom" in a for a in info["assumptions"])
    assert strat.signal_tickers == [config.REGIME_TICKER]


def test_constructor_validation():
    with pytest.raises(ValueError):
        RegimeSwitchStrategy(universe=list(ETF_TRENDS), regime_sma_period=0)
