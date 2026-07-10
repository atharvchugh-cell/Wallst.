"""Tests for --strategy paper: a persistent, reloadable forward paper-trading
ledger for the fixed 60/35/5 portfolio.

The hard properties proven here (requirements 21 & 22):
  - no lookahead: mutating FUTURE prices cannot change an already-processed
    date's signals, fills, or equity;
  - the existing one-trading-day signal-to-fill lag holds (no same-day fill);
  - repeated runs are idempotent -- no duplicated signals, orders, fills,
    transaction costs, or equity rows;
  - sleeves never share cash and capital is never double-counted; the initial
    60/35/5 allocation is exact ($9,000 / $5,250 / $750 of $15,000);
  - state survives a process restart (it is pure on-disk JSON);
  - corrupted / incomplete state fails closed and is never overwritten;
  - future/unfinalized dates are rejected; non-trading dates resolve correctly;
  - reset requires explicit confirmation and backs up first;
  - state writes are atomic (no temp files leak);
  - a pending order remains pending / is reported stale rather than silently
    filled when its fill price is missing;
  - account and sleeve totals reconcile;
  - and a deterministic multi-day simulation fills a pending order on the
    correct later session with everything reconciling after a reload.

NOTE: this is paper trading only -- the module never connects to a broker.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import cli, paper, tournament

FULL_DATES = pd.bdate_range("1990-01-01", "2024-12-31")


def _make_frame(seed, drift=0.02):
    rng = np.random.default_rng(seed)
    base = 100.0 + np.cumsum(rng.normal(drift, 0.5, size=len(FULL_DATES)))
    base = np.clip(base, 10.0, None)
    return pd.DataFrame(
        {"Open": base, "High": base, "Low": base, "Close": base, "Volume": 1000}, index=FULL_DATES
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Deterministic synthetic price frames spanning 1990-2024, wired into every
    data accessor the paper path touches. `frames` is mutable so tests can alter
    FUTURE bars and prove no leakage into already-processed dates. Fetches are
    sliced to <= end, mirroring the real 'never read prices after the paper
    date' guarantee at the data layer."""
    frames = {}

    def frame(t):
        if t not in frames:
            frames[t] = _make_frame(abs(hash(t)) % (2**31))
        return frames[t]

    def fake_price(tickers, start, end, warmup_calendar_days, hard_fail_on_missing, **kw):
        fs = pd.Timestamp(start) - pd.Timedelta(days=warmup_calendar_days)
        out = {}
        for t in tickers:
            df = frame(t)
            out[t] = df.loc[(df.index >= fs) & (df.index <= pd.Timestamp(end))].copy()
        return out, []

    def fake_bench(start, end, **kw):
        df = frame("SPY")
        return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))].copy()

    monkeypatch.setattr(paper.data, "get_price_data", fake_price)
    monkeypatch.setattr(paper.data, "get_benchmark_data", fake_bench)
    monkeypatch.setattr(tournament.data, "get_price_data", fake_price)
    monkeypatch.setattr(tournament.data, "get_benchmark_data", fake_bench)

    class Env:
        def __init__(self):
            self.frames = frames
            self.frame = frame
            self.tmp_path = tmp_path
            self.n = 0

        def dir(self, name=None):
            name = name or f"acct{self.n}"
            self.n += 1
            return str(tmp_path / name)

        def init(self, d, capital=15000.0, start="2023-01-01", weights=None):
            paper.init_account(
                d, capital, start, weights or paper.DEFAULT_PORTFOLIO_WEIGHTS,
                5.0, True, "default", None, None,
            )
            return d

    return Env()


# --- Initialization & exact allocation -----------------------------------------

def test_init_creates_all_cash_60_35_5_account(env):
    d = env.init(env.dir())
    cfg, st = paper.load_account(d)
    assert st["starting_capital"] == pytest.approx(15000.0)
    assert st["last_processed_date"] is None
    sleeves = st["sleeves"]
    assert sleeves["momentum"]["allocated_capital"] == pytest.approx(9000.0)
    assert sleeves["sector_rotation"]["allocated_capital"] == pytest.approx(5250.0)
    assert sleeves["regime_switch"]["allocated_capital"] == pytest.approx(750.0)
    for s in sleeves.values():
        assert s["cash"] == pytest.approx(s["allocated_capital"])
        assert s["equity"] == pytest.approx(s["allocated_capital"])
        assert s["positions"] == []
    assert sum(s["allocated_capital"] for s in sleeves.values()) == pytest.approx(15000.0)
    assert st["total_equity"] == pytest.approx(15000.0)


def test_init_refuses_to_clobber_existing_account(env):
    d = env.init(env.dir())
    with pytest.raises(paper.PaperError):
        env.init(d)


def test_init_rejects_bad_weights(env):
    with pytest.raises(Exception):
        paper.init_account(
            env.dir(), 15000.0, "2023-01-01", [("momentum", 0.6), ("sector_rotation", 0.5)],
            5.0, True, "default", None, None,
        )


# --- One-day lag / no same-day fill --------------------------------------------

def test_pending_order_fills_on_next_session_not_same_day(env):
    d = env.init(env.dir())
    paper.advance(d, target_date="2023-01-31")
    _cfg, st = paper.load_account(d)
    assert len(st["pending_orders"]) > 0
    for o in st["pending_orders"]:
        assert pd.Timestamp(o["scheduled_fill_date"]) > pd.Timestamp(o["signal_date"])
        assert o["signal_date"] == "2023-01-31"
    assert not any(f["fill_date"] == "2023-01-31" for f in st["completed_trades"])


# --- Idempotency ---------------------------------------------------------------

def test_reprocessing_same_date_is_a_noop(env):
    d = env.init(env.dir())
    paper.advance(d, target_date="2023-01-20")
    _cfg, st1 = paper.load_account(d)
    before = (
        st1["num_runs"], len(st1["completed_trades"]),
        len(st1["portfolio_equity_history"]), st1["transaction_costs_total"],
    )
    res = paper.advance(d, target_date="2023-01-20")
    assert res["processed"] == []
    assert "Already processed" in res["message"]
    _cfg, st2 = paper.load_account(d)
    after = (
        st2["num_runs"], len(st2["completed_trades"]),
        len(st2["portfolio_equity_history"]), st2["transaction_costs_total"],
    )
    assert before == after
    dates = [r["date"] for r in st2["portfolio_equity_history"]]
    assert len(dates) == len(set(dates))


def test_no_duplicate_fill_ids(env):
    d = env.init(env.dir())
    paper.advance(d, target_date="2023-02-15")
    _cfg, st = paper.load_account(d)
    ids = [f["id"] for f in st["completed_trades"]]
    assert len(ids) == len(set(ids))


# --- No lookahead: future mutations can't change processed dates ---------------

def test_future_price_mutation_cannot_change_processed_history(env):
    cutoff = "2023-02-28"
    d1 = env.init(env.dir("before"))
    paper.advance(d1, target_date=cutoff)
    _cfg, st_before = paper.load_account(d1)
    hist_before = {r["date"]: r["equity"] for r in st_before["portfolio_equity_history"]}

    cutoff_ts = pd.Timestamp(cutoff)
    for t, df in list(env.frames.items()):
        mask = df.index > cutoff_ts
        df.loc[mask, ["Open", "High", "Low", "Close"]] *= 3.0

    d2 = env.init(env.dir("after"))
    paper.advance(d2, target_date=cutoff)
    _cfg, st_after = paper.load_account(d2)
    hist_after = {r["date"]: r["equity"] for r in st_after["portfolio_equity_history"]}

    assert len(hist_before) > 20
    for date, eq in hist_before.items():
        assert hist_after[date] == pytest.approx(eq), f"equity changed on {date}"


# --- No shared cash / reconciliation -------------------------------------------

def test_totals_reconcile_and_no_shared_cash(env):
    d = env.init(env.dir())
    paper.advance(d, target_date="2023-03-15")
    recon = paper.reconcile_saved_state(d)
    assert recon["ok"], recon["checks"]
    _cfg, st = paper.load_account(d)
    assert sum(s["equity"] for s in st["sleeves"].values()) == pytest.approx(st["total_equity"])
    for s in st["sleeves"].values():
        assert s["cash"] >= -0.01
    assert sum(s["allocated_capital"] for s in st["sleeves"].values()) == pytest.approx(15000.0)


# --- State survives restart ----------------------------------------------------

def test_state_survives_restart_and_continues(env):
    d = env.init(env.dir())
    paper.advance(d, target_date="2023-01-20")
    _cfg, st = paper.load_account(d)
    equity_at_stop = st["total_equity"]
    last = st["last_processed_date"]

    _cfg2, st2 = paper.load_account(d)
    assert st2["total_equity"] == pytest.approx(equity_at_stop)
    assert st2["last_processed_date"] == last

    res = paper.advance(d)
    assert res["processed"]
    assert pd.Timestamp(res["processed"][0]["paper_date"]) > pd.Timestamp(last)


# --- Corrupted / incomplete state fails closed ---------------------------------

def test_corrupted_state_fails_closed_and_is_not_overwritten(env):
    d = env.init(env.dir())
    paper.advance(d, target_date="2023-01-10")
    spath = paper.state_path(d)
    spath.write_text("{ this is not valid json")
    raw = spath.read_text()
    with pytest.raises(paper.PaperStateError):
        paper.load_account(d)
    with pytest.raises(paper.PaperStateError):
        paper.advance(d, target_date="2023-01-11")
    assert spath.read_text() == raw


def test_incomplete_state_fails_closed(env):
    d = env.init(env.dir())
    st = json.loads(paper.state_path(d).read_text())
    del st["sleeves"]
    paper.state_path(d).write_text(json.dumps(st))
    with pytest.raises(paper.PaperStateError):
        paper.load_account(d)


def test_schema_version_mismatch_fails_closed(env):
    d = env.init(env.dir())
    st = json.loads(paper.state_path(d).read_text())
    st["schema_version"] = 999
    paper.state_path(d).write_text(json.dumps(st))
    with pytest.raises(paper.PaperStateError):
        paper.load_account(d)


# --- Future dates rejected; non-trading dates resolve --------------------------

def test_future_date_rejected(env):
    d = env.init(env.dir())
    with pytest.raises(paper.PaperError):
        paper.advance(d, target_date="2030-01-01")


def test_non_trading_date_resolves_to_prior_session(env):
    d = env.init(env.dir())
    paper.advance(d, target_date="2023-06-24")  # Saturday -> Friday 2023-06-23
    _cfg, st = paper.load_account(d)
    assert st["last_processed_date"] == "2023-06-23"


def test_date_before_inception_rejected(env):
    d = env.init(env.dir(), start="2023-06-01")
    with pytest.raises(paper.PaperError):
        paper.advance(d, target_date="2023-01-01")


# --- Reset safeguards ----------------------------------------------------------

def test_reset_requires_explicit_confirmation(env):
    d = env.init(env.dir())
    with pytest.raises(paper.PaperError):
        paper.reset_account(d, confirm=False)
    assert paper.state_path(d).exists()


def test_reset_backs_up_then_removes(env):
    d = env.init(env.dir())
    backup = paper.reset_account(d, confirm=True)
    assert (Path(backup) / paper.CONFIG_FILENAME).exists()
    assert (Path(backup) / paper.STATE_FILENAME).exists()
    assert not paper.state_path(d).exists()
    assert not paper.config_path(d).exists()


# --- Atomic writes -------------------------------------------------------------

def test_no_temp_files_leak_after_write(env):
    d = env.init(env.dir())
    paper.advance(d, target_date="2023-01-10")
    assert list(Path(d).glob("*.tmp")) == []


# --- Pending stays pending when the fill session isn't available yet -----------
# (The engine's stale/defer-unfillable path -- "reported, never silently filled"
# when a fill PRICE is missing -- is proven directly in test_engine.py. Here we
# prove the paper-frontier property: an order created on the latest finalized
# session stays PENDING, never filled, until a later session is finalized.)

def test_frontier_pending_order_stays_pending_and_is_not_filled(env):
    # Process up to the latest finalized session in the mocked data (also a
    # month-end -> a rebalance fires, creating pending orders whose fill session
    # is a PROJECTED future date not yet finalized).
    d = env.init(env.dir(), start="2024-11-01")
    paper.advance(d, target_date="2024-12-31")
    _cfg, st = paper.load_account(d)
    assert st["last_processed_date"] == "2024-12-31"
    pend = st["pending_orders"]
    assert pend
    for o in pend:
        assert pd.Timestamp(o["scheduled_fill_date"]) > pd.Timestamp("2024-12-31")
    # No fill exists on/after the (not-yet-finalized) scheduled fill date.
    assert not any(pd.Timestamp(f["fill_date"]) > pd.Timestamp("2024-12-31")
                   for f in st["completed_trades"])

    # A --paper-run finds no further finalized session: it is a no-op, and the
    # pending orders remain pending (unfilled) rather than being force-filled.
    res = paper.advance(d)
    assert res["processed"] == []
    assert "up to date" in res["message"].lower()
    _cfg, st2 = paper.load_account(d)
    assert len(st2["pending_orders"]) == len(pend)
    assert st2["reconciliation"]["ok"]


# --- Deterministic multi-day end-to-end (requirement 22) -----------------------

def test_multi_day_simulation_fills_pending_on_correct_later_session(env):
    d = env.init(env.dir(), capital=15000.0, start="2023-01-01")

    paper.advance(d, target_date="2023-01-31")
    _cfg, st1 = paper.load_account(d)
    assert st1["last_processed_date"] == "2023-01-31"
    assert len(st1["pending_orders"]) >= 1
    sample = st1["pending_orders"][0]
    fill_day = sample["scheduled_fill_date"]
    assert pd.Timestamp(fill_day) > pd.Timestamp("2023-01-31")
    fills_before = len(st1["completed_trades"])

    _cfg, st_reload = paper.load_account(d)
    assert st_reload["last_processed_date"] == "2023-01-31"
    res = paper.advance(d)
    assert res["processed"][0]["paper_date"] == fill_day == "2023-02-01"

    _cfg, st2 = paper.load_account(d)
    new_fills = [f for f in st2["completed_trades"] if f["fill_date"] == fill_day]
    assert len(new_fills) >= 1
    assert len(st2["completed_trades"]) > fills_before
    assert st2["pending_orders"] == [] or all(
        pd.Timestamp(o["scheduled_fill_date"]) > pd.Timestamp(fill_day)
        for o in st2["pending_orders"]
    )
    assert st2["reconciliation"]["ok"]
    assert sum(s["equity"] for s in st2["sleeves"].values()) == pytest.approx(st2["total_equity"])
    for s in st2["sleeves"].values():
        mv = sum(p["market_value"] for p in s["positions"])
        assert s["cash"] + mv == pytest.approx(s["equity"])
    assert st2["transaction_costs_total"] > 0


# --- Existing modes/flags undisturbed ------------------------------------------

def test_existing_strategy_choices_and_defaults_unchanged():
    for choice in ["mean_reversion", "sector_rotation", "both", "compare", "robustness",
                   "tournament", "portfolio", "walk_forward"]:
        args = cli.parse_args(["--strategy", choice])
        assert args.strategy == choice
        assert args.paper_init is False
        assert args.paper_run is False
        assert args.paper_date is None
        assert args.paper_state_dir == paper.DEFAULT_PAPER_STATE_DIR
    args = cli.parse_args(["--strategy", "paper"])
    assert args.strategy == "paper"


def test_cli_paper_requires_an_action(env):
    d = env.dir()
    rc = cli.main(["--strategy", "paper", "--paper-state-dir", d])
    assert rc == 1
    assert not paper.account_exists(d)
