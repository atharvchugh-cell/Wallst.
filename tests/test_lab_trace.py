"""Decision-trace correctness.

Proves the trace is faithful and grounded:
  - one record per universe ticker at every momentum/sector rebalance
    (rejections are visible, not just selections);
  - every reason code matches the branch the strategy actually took, checked
    against the traced numeric values;
  - decision identities are unique;
  - every filled order maps back to a decision, and traced target weights
    reconcile to the actual target events;
  - future-data mutation cannot change any trace row dated <= T;
  - unavailable values stay null rather than being fabricated;
  - row ordering is deterministic.
"""

import numpy as np
import pandas as pd
import pytest

from src import portfolio, tournament
from src.lab.lab_config import LabConfig
from src.lab.portfolio_engine import run_lab_portfolio
from src.lab.trace import DecisionRecorder, decisions_frame, orders_frame, selection_funnel

FULL_DATES = pd.bdate_range("1990-01-01", "2024-12-31")


def trending_df(seed, drift=0.01):
    rng = np.random.default_rng(seed)
    base = 100.0 + np.cumsum(rng.normal(drift, 0.5, size=len(FULL_DATES)))
    base = np.clip(base, 10.0, None)
    return pd.DataFrame(
        {"Open": base, "High": base, "Low": base, "Close": base, "Volume": 1000}, index=FULL_DATES
    )


@pytest.fixture
def mocked_data(monkeypatch):
    frames: dict[str, pd.DataFrame] = {}

    def get_frame(t):
        if t not in frames:
            frames[t] = trending_df(seed=abs(hash(t)) % (2**31))
        return frames[t].copy()

    def fake_get_price_data(tickers, start, end, warmup_calendar_days, hard_fail_on_missing, **kw):
        return {t: get_frame(t) for t in tickers}, []

    def fake_get_benchmark_data(start, end, **kw):
        return get_frame("SPY")

    monkeypatch.setattr(tournament.data, "get_price_data", fake_get_price_data)
    monkeypatch.setattr(tournament.data, "get_benchmark_data", fake_get_benchmark_data)
    return frames


RUN_KWARGS = dict(
    total_capital=15000.0, start="2022-01-01", end="2024-12-31", cost_bps=5.0,
    fractional_shares=True, refresh_cache=False,
)


def run_traced(pairs=None, lab_config=None):
    pairs = pairs or portfolio.parse_portfolio_weights(None)
    rec = DecisionRecorder(lab_config_hash="trace_test")
    run = run_lab_portfolio(pairs, lab_config=lab_config or LabConfig(), recorder=rec, **RUN_KWARGS)
    return run


# --- Completeness: rejections are visible ------------------------------------------

def test_every_momentum_universe_ticker_traced_each_rebalance(mocked_data):
    run = run_traced()
    df = decisions_frame(run.trace)
    mom = df[df["sleeve"] == "momentum"]
    assert not mom.empty
    universe = set(next(p.strategy.universe for p in run.prepared if p.name == "momentum"))
    # At each rebalance date, every tradable universe ticker has exactly one row.
    for date, g in mom[mom["ticker"] != "CASH"].groupby("decision_date"):
        traded = g[g["tradable"] != False]  # noqa: E712
        assert set(traded["ticker"]) == universe, f"{date}: {set(traded['ticker'])} != universe"
        assert len(traded) == len(universe), f"{date}: duplicate ticker rows"


def test_selection_funnel_stages_are_monotone(mocked_data):
    run = run_traced()
    funnel = selection_funnel(run.trace)
    mom = funnel[funnel["sleeve"] == "momentum"]
    assert not mom.empty
    # considered >= had_data >= eligible >= selected, every rebalance.
    assert (mom["considered"] >= mom["had_data"]).all()
    assert (mom["had_data"] >= mom["eligible"]).all()
    assert (mom["eligible"] >= mom["selected"]).all()
    # Momentum selects at most top_k.
    assert (mom["selected"] <= 5).all()


# --- Grounding: reason code matches the branch -------------------------------------

def test_reason_codes_match_traced_values(mocked_data):
    run = run_traced()
    df = decisions_frame(run.trace)
    mom = df[(df["sleeve"] == "momentum") & (df["ticker"] != "CASH")]

    below_trend = mom[mom["reason_code"] == "BELOW_TREND_FILTER"]
    assert (below_trend["close"] <= below_trend["trend_sma"] + 1e-9).all()
    assert (below_trend["eligible"] == False).all()  # noqa: E712

    neg = mom[mom["reason_code"] == "NEGATIVE_TRAILING_RETURN"]
    assert (neg["lookback_return"] <= 0 + 1e-12).all()

    selected = mom[mom["reason_code"].isin(["SELECTED_TOP_K", "POSITION_RETAINED"])]
    assert (selected["selected"] == True).all()  # noqa: E712
    assert (selected["rank"] <= selected["top_k"]).all()
    assert np.allclose(selected["target_weight"].values, 1.0 / 5)

    below_cut = mom[mom["reason_code"] == "RANK_BELOW_CUTOFF"]
    assert (below_cut["rank"] > below_cut["top_k"]).all()
    assert (below_cut["eligible"] == True).all()  # noqa: E712


def test_sector_selected_rows_have_top_ranks(mocked_data):
    run = run_traced()
    df = decisions_frame(run.trace)
    sr = df[(df["sleeve"] == "sector_rotation") & (df["selected"] == True)]  # noqa: E712
    assert not sr.empty
    assert (sr["rank"] <= sr["top_k"]).all()
    assert (sr["reason_code"] == "SELECTED_TOP_K").all()


# --- Uniqueness & order/decision reconciliation ------------------------------------

def test_decision_ids_are_unique(mocked_data):
    run = run_traced()
    df = decisions_frame(run.trace)
    assert df["decision_id"].is_unique


def test_every_filled_order_maps_to_a_decision(mocked_data):
    run = run_traced()
    dec = decisions_frame(run.trace)
    orders = orders_frame(run.trace)
    assert not orders.empty
    # Every order's (sleeve, ticker, signal_date) must appear in the decision
    # trace (the decision that emitted the target that later filled).
    dec_keys = set(zip(dec["sleeve"], dec["ticker"], pd.to_datetime(dec["signal_date"])))
    for _i, o in orders.iterrows():
        key = (o["sleeve"], o["ticker"], pd.to_datetime(o["signal_date"]))
        assert key in dec_keys, f"orphan order with no decision: {key}"


def test_traced_target_weights_reconcile_to_target_events(mocked_data):
    """Every nonzero target weight a strategy emitted appears in the decision
    trace with the same weight (selection rows), and vice versa."""
    run = run_traced()
    dec = decisions_frame(run.trace)
    for prepared in run.prepared:
        name = prepared.name
        sleeve_dec = dec[(dec["sleeve"] == name) & (dec["selected"] == True)]  # noqa: E712
        # Actual nonzero target events from the sleeve result.
        sleeve = next(s for s in run.portfolio.sleeves if s.strategy == name)
        nonzero_events = {
            (e.ticker, pd.Timestamp(e.signal_date))
            for e in sleeve.result.target_events if e.target_weight > 0
        }
        traced_selected = {
            (r["ticker"], pd.Timestamp(r["signal_date"])) for _i, r in sleeve_dec.iterrows()
        }
        assert nonzero_events == traced_selected, f"{name}: selection trace != nonzero targets"


# --- No lookahead through the trace ------------------------------------------------

def test_future_mutation_cannot_change_trace_through_T(mocked_data):
    run_before = run_traced()
    dec_before = decisions_frame(run_before.trace)
    cutoff = pd.Timestamp("2023-06-30")

    for t, df in list(mocked_data.items()):
        mask = df.index > cutoff
        df.loc[mask, ["Open", "High", "Low", "Close"]] *= 2.5
        mocked_data[t] = df

    run_after = run_traced()
    dec_after = decisions_frame(run_after.trace)

    b = dec_before[dec_before["decision_date"] <= cutoff].reset_index(drop=True)
    a = dec_after[dec_after["decision_date"] <= cutoff].reset_index(drop=True)
    assert len(b) > 20
    # Compare the decision-defining columns (grounded values through T).
    cols = ["decision_id", "sleeve", "ticker", "reason_code", "selected", "eligible",
            "close", "lookback_return", "trend_sma", "rank", "target_weight"]
    pd.testing.assert_frame_equal(b[cols], a[cols], check_dtype=False)


# --- Grounding: nulls stay null ----------------------------------------------------

def test_unavailable_values_stay_null(mocked_data):
    run = run_traced()
    df = decisions_frame(run.trace)
    # CASH fallback rows carry no price/rank fields.
    cash = df[df["ticker"] == "CASH"]
    if not cash.empty:
        assert cash["close"].isna().all()
        assert cash["rank"].isna().all()
    # Regime signal rows (SPY, tradable=False) carry no rank/lookback fields.
    regime = df[df["reason_code"].isin(["REGIME_RISK_ON", "REGIME_RISK_OFF"])]
    regime_spy = regime[regime["ticker"] == "SPY"]
    if not regime_spy.empty:
        assert regime_spy["rank"].isna().all()
        assert regime_spy["lookback_return"].isna().all()


# --- Determinism -------------------------------------------------------------------

def test_trace_ordering_is_deterministic(mocked_data):
    a = decisions_frame(run_traced().trace)
    b = decisions_frame(run_traced().trace)
    pd.testing.assert_frame_equal(a, b)


# --- Regime risk-off is captured when the market is below trend --------------------

def test_regime_risk_off_is_traced_in_a_downtrend(monkeypatch):
    """With SPY in a sustained downtrend below its 200-day SMA, the regime
    sleeve must record REGIME_RISK_OFF at month-end signal dates."""
    frames: dict[str, pd.DataFrame] = {}

    def spy_downtrend():
        # Rise then fall so the 200-day SMA is above price in the test window.
        n = len(FULL_DATES)
        up = np.linspace(50, 300, n // 2)
        down = np.linspace(300, 60, n - n // 2)
        base = np.concatenate([up, down])
        return pd.DataFrame(
            {"Open": base, "High": base, "Low": base, "Close": base, "Volume": 1000},
            index=FULL_DATES,
        )

    def get_frame(t):
        if t not in frames:
            frames[t] = spy_downtrend() if t == "SPY" else trending_df(abs(hash(t)) % (2**31), drift=-0.02)
        return frames[t].copy()

    def fake_get_price_data(tickers, start, end, warmup_calendar_days, hard_fail_on_missing, **kw):
        return {t: get_frame(t) for t in tickers}, []

    def fake_get_benchmark_data(start, end, **kw):
        return get_frame("SPY")

    monkeypatch.setattr(tournament.data, "get_price_data", fake_get_price_data)
    monkeypatch.setattr(tournament.data, "get_benchmark_data", fake_get_benchmark_data)

    rec = DecisionRecorder(lab_config_hash="regime_test")
    run = run_lab_portfolio(
        [("regime_switch", 1.0)], 15000.0, "2024-01-01", "2024-12-31", 5.0, True, False,
        recorder=rec,
    )
    df = decisions_frame(run.trace)
    risk_off = df[df["reason_code"] == "REGIME_RISK_OFF"]
    assert not risk_off.empty, "expected REGIME_RISK_OFF rows in a sustained downtrend"
    # The SPY signal rows carry the regime close/SMA that justified the call.
    spy_rows = risk_off[risk_off["ticker"] == "SPY"]
    assert not spy_rows.empty
    assert (spy_rows["human_reason"].str.contains("risk-off")).all()
