"""Tests for --strategy paper: a persistent, reloadable, incremental forward
paper-trading ledger for the fixed 60/35/5 portfolio, backed by SQLite
(src/paper_db.py) rather than a full-history replay.

The hard properties proven here:
  - true incrementality: each session only fills PREVIOUSLY-PERSISTED pending
    orders and appends brand-new same-session signals; historical price
    revisions can never rewrite already-committed signals/orders/fills/costs/
    cash/equity rows;
  - the existing one-trading-day signal-to-fill lag holds (no same-day fill);
  - repeated runs are idempotent;
  - sleeves never share cash and capital is never double-counted; the initial
    60/35/5 allocation is exact;
  - state survives a REAL process restart (subprocess, not just object reuse);
  - corrupted / incomplete ledger fails closed and is never overwritten or
    silently repaired;
  - future/unfinalized dates are rejected; non-trading dates resolve;
  - reset requires explicit confirmation and backs up first;
  - a runtime strategy/config fingerprint drift refuses to advance while every
    read-only command keeps working;
  - a scheduled fill date is holiday-aware (src/nyse_calendar.py), not a
    weekend-only projection;
  - every signal/order/fill has a stable surrogate ID (never derived solely
    from ticker/side/date), with foreign-key linkage order->signal,
    fill->order, and duplicate signals are rejected by a DB constraint;
  - a mid-session exception rolls back EVERY table write for that session
    (nothing partial ever survives);
  - reconciliation independently rebuilds cash/positions/avg-cost/realized P&L
    from the immutable fills ledger and catches tampering;
  - a stale (unfillable) order is reported, never silently filled;
  - concurrent runs are blocked by a file lock;
  - every run summary carries the full required field list.

NOTE: this is paper trading only -- the module never connects to a broker.
"""

import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import cli, config, nyse_calendar, paper, paper_db, portfolio, tournament

FULL_DATES = pd.bdate_range("1990-01-01", "2024-12-31")
REPO_ROOT = Path(__file__).resolve().parent.parent
SUBPROCESS_HELPER = Path(__file__).resolve().parent / "_paper_mock_cli.py"


def _seed(ticker: str) -> int:
    """SHA-256-derived deterministic seed -- NOT Python's randomized hash(),
    which is salted per-process and would make fixtures non-reproducible."""
    return int(hashlib.sha256(ticker.encode()).hexdigest()[:8], 16) % (2**31)


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
    bars and prove immutability of already-committed ledger rows."""
    frames = {}

    def frame(t):
        if t not in frames:
            frames[t] = _make_frame(_seed(t))
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


def _conn(d):
    return paper_db.connect(d)


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
    assert cfg["config_fingerprint"]


def test_init_refuses_to_clobber_existing_account(env):
    d = env.init(env.dir())
    with pytest.raises(paper.PaperError):
        env.init(d)


def test_init_rejects_bad_weights(env):
    with pytest.raises(portfolio.PortfolioError):
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


# --- Immutable ledger: historical price revisions can't rewrite committed rows -

def test_historical_price_mutation_after_fill_does_not_rewrite_ledger(env):
    """Process a signal and its fill, mutate HISTORICAL prices (dated before
    that fill), advance another session, and verify every previously
    persisted signal/order/fill/cost/cash/equity record is unchanged
    field-for-field. This is the core guarantee of an append-only
    incremental ledger: market-data revisions can never rewrite history."""
    d = env.init(env.dir())
    paper.advance(d, target_date="2023-01-31")  # signals + pending orders created
    paper.advance(d)  # 2023-02-01: those pending orders fill
    _cfg, before = paper.load_account(d)
    assert before["completed_trades"], "test requires at least one fill to have happened"

    before_signals = _dump_table(d, "signals")
    before_orders = _dump_table(d, "orders")
    before_fills = _dump_table(d, "fills")
    before_equity = _dump_table(d, "equity_history")
    before_sleeve_equity = _dump_table(d, "sleeve_equity_history")
    before_positions = _dump_table(d, "positions")

    # Mutate EVERY historical bar up to and including the fill date.
    fill_date = pd.Timestamp(before["last_processed_date"])
    for t, df in list(env.frames.items()):
        mask = df.index <= fill_date
        df.loc[mask, ["Open", "High", "Low", "Close"]] *= 7.0

    paper.advance(d)  # process the next session; must not touch prior rows

    after_signals = _dump_table(d, "signals")
    after_orders = _dump_table(d, "orders")
    after_fills = _dump_table(d, "fills")
    after_equity = _dump_table(d, "equity_history")
    after_sleeve_equity = _dump_table(d, "sleeve_equity_history")

    assert after_signals[: len(before_signals)] == before_signals
    assert after_orders[: len(before_orders)] == before_orders
    assert after_fills[: len(before_fills)] == before_fills
    assert after_equity[: len(before_equity)] == before_equity
    assert after_sleeve_equity[: len(before_sleeve_equity)] == before_sleeve_equity
    # New rows may have been appended, but nothing already there was altered.
    assert len(after_signals) >= len(before_signals)
    assert len(after_fills) >= len(before_fills)


def _dump_table(state_dir, table_name):
    conn = _conn(state_dir)
    try:
        rows = conn.execute(f"SELECT * FROM {table_name} ORDER BY rowid").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


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


# --- Persisted pending orders are exactly what fills next session --------------

def test_persisted_pending_orders_are_exactly_filled_next_session(env):
    d = env.init(env.dir())
    paper.advance(d, target_date="2023-01-31")
    _cfg, st1 = paper.load_account(d)
    pending_before = {(o["sleeve"], o["ticker"], o["scheduled_fill_date"]) for o in st1["pending_orders"]}
    assert pending_before

    res = paper.advance(d)
    fill_date = res["processed"][0]["paper_date"]
    _cfg, st2 = paper.load_account(d)
    filled_keys = {(f["sleeve"], f["ticker"], f["fill_date"]) for f in st2["completed_trades"]
                   if f["fill_date"] == fill_date}
    expected = {(s, t, fd) for (s, t, fd) in pending_before if fd == fill_date}
    assert expected
    assert filled_keys == expected
    # And no order that was pending remains pending after its scheduled date.
    assert all(pd.Timestamp(o["scheduled_fill_date"]) > pd.Timestamp(fill_date)
               for o in st2["pending_orders"])


# --- State survives a REAL process restart --------------------------------------

def test_state_survives_subprocess_restart_and_continues(tmp_path):
    d = str(tmp_path / "subproc_acct")
    r1 = subprocess.run(
        [sys.executable, str(SUBPROCESS_HELPER), "--strategy", "paper", "--paper-state-dir", d,
         "--paper-init", "--paper-start", "2023-01-01", "--capital", "15000"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
    )
    assert r1.returncode == 0, r1.stdout + r1.stderr

    r2 = subprocess.run(
        [sys.executable, str(SUBPROCESS_HELPER), "--strategy", "paper", "--paper-state-dir", d,
         "--paper-date", "2023-01-20"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=180,
    )
    assert r2.returncode == 0, r2.stdout + r2.stderr

    # Read back in THIS process (no mocking needed -- pure DB reads).
    _cfg, st_mid = paper.load_account(d)
    assert st_mid["last_processed_date"] == "2023-01-20"
    equity_at_stop = st_mid["total_equity"]

    r3 = subprocess.run(
        [sys.executable, str(SUBPROCESS_HELPER), "--strategy", "paper", "--paper-state-dir", d,
         "--paper-run"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
    )
    assert r3.returncode == 0, r3.stdout + r3.stderr

    _cfg, st_final = paper.load_account(d)
    assert pd.Timestamp(st_final["last_processed_date"]) > pd.Timestamp("2023-01-20")
    assert st_final["num_runs"] == st_mid["num_runs"] + 1
    assert st_final["reconciliation"]["ok"]
    # The third (separate) process built on the second's persisted state: the
    # equity history it appended must extend, not replace, the earlier run's.
    assert len(st_final["portfolio_equity_history"]) > len(st_mid["portfolio_equity_history"])
    assert isinstance(st_final["total_equity"], float) and st_final["total_equity"] > 0


# --- Corrupted / incomplete ledger fails closed ---------------------------------

def test_corrupted_database_fails_closed_and_is_not_overwritten(env):
    d = env.init(env.dir())
    paper.advance(d, target_date="2023-01-10")
    dbfile = paper_db.db_path(d)
    raw_before = dbfile.read_bytes()
    dbfile.write_bytes(b"this is not a valid sqlite file, deliberately corrupted")
    with pytest.raises(paper.PaperStateError):
        paper.load_account(d)
    with pytest.raises(paper.PaperStateError):
        paper.advance(d, target_date="2023-01-11")
    assert dbfile.read_bytes() != raw_before  # we wrote garbage ourselves
    # But the module never rewrote it further -- still exactly our garbage.
    assert dbfile.read_bytes() == b"this is not a valid sqlite file, deliberately corrupted"


def test_missing_meta_keys_fails_closed(env):
    d = env.init(env.dir())
    conn = _conn(d)
    conn.execute("DELETE FROM meta WHERE key = 'config_fingerprint'")
    conn.close()
    with pytest.raises(paper.PaperStateError):
        paper.load_account(d)


def test_schema_version_mismatch_fails_closed(env):
    d = env.init(env.dir())
    conn = _conn(d)
    conn.execute("UPDATE meta SET value = '999' WHERE key = 'schema_version'")
    conn.close()
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


# --- Reset safeguards ------------------------------------------------------------

def test_reset_requires_explicit_confirmation(env):
    d = env.init(env.dir())
    with pytest.raises(paper.PaperError):
        paper.reset_account(d, confirm=False)
    assert paper.account_exists(d)


def test_reset_backs_up_then_removes(env):
    d = env.init(env.dir())
    backup = paper.reset_account(d, confirm=True)
    assert (Path(backup) / paper_db.DB_FILENAME).exists()
    assert not paper.account_exists(d)


# --- Atomic writes / DB integrity -------------------------------------------------

def test_no_temp_files_leak_after_write(env):
    d = env.init(env.dir())
    paper.advance(d, target_date="2023-01-10")
    assert list(Path(d).glob("*.tmp")) == []


def test_database_passes_integrity_check_after_advance(env):
    d = env.init(env.dir())
    paper.advance(d, target_date="2023-01-20")
    conn = _conn(d)
    result = conn.execute("PRAGMA integrity_check").fetchone()[0]
    conn.close()
    assert result == "ok"


# --- Frontier pending order stays pending ----------------------------------------

def test_frontier_pending_order_stays_pending_and_is_not_filled(env):
    d = env.init(env.dir(), start="2024-11-01")
    paper.advance(d, target_date="2024-12-31")
    _cfg, st = paper.load_account(d)
    assert st["last_processed_date"] == "2024-12-31"
    pend = st["pending_orders"]
    assert pend
    for o in pend:
        assert pd.Timestamp(o["scheduled_fill_date"]) > pd.Timestamp("2024-12-31")
    assert not any(pd.Timestamp(f["fill_date"]) > pd.Timestamp("2024-12-31")
                   for f in st["completed_trades"])

    res = paper.advance(d)
    assert res["processed"] == []
    assert "up to date" in res["message"].lower()
    _cfg, st2 = paper.load_account(d)
    assert len(st2["pending_orders"]) == len(pend)
    assert st2["reconciliation"]["ok"]


# --- Deterministic multi-day end-to-end ------------------------------------------

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


# --- Config fingerprint drift refuses to advance; read-only still works --------

def test_fingerprint_drift_refuses_to_advance(env, monkeypatch):
    d = env.init(env.dir())
    paper.advance(d, target_date="2023-01-10")
    monkeypatch.setattr(paper, "SIGNAL_EXECUTION_VERSION", 999)
    with pytest.raises(paper.PaperFingerprintError):
        paper.advance(d, target_date="2023-01-11")
    # Nothing was persisted despite the attempted advance (load_account is a
    # pure DB read -- it needs no price-data mocks, so no unpatching needed).
    _cfg, st = paper.load_account(d)
    assert st["last_processed_date"] == "2023-01-10"


def test_read_only_commands_work_under_fingerprint_drift(env, monkeypatch):
    d = env.init(env.dir())
    paper.advance(d, target_date="2023-01-10")
    monkeypatch.setattr(paper, "SIGNAL_EXECUTION_VERSION", 999)
    cfg, st = paper.load_account(d)  # must not raise
    assert st["last_processed_date"] == "2023-01-10"
    recon = paper.reconcile_saved_state(d)  # must not raise
    assert recon["ok"]


def test_cli_status_works_under_fingerprint_drift(env, monkeypatch):
    d = env.init(env.dir())
    paper.advance(d, target_date="2023-01-10")
    monkeypatch.setattr(paper, "SIGNAL_EXECUTION_VERSION", 999)
    rc = cli.main(["--strategy", "paper", "--paper-state-dir", d, "--paper-status"])
    assert rc == 0
    rc = cli.main(["--strategy", "paper", "--paper-state-dir", d, "--paper-run"])
    assert rc == 1  # advancing must be refused


# --- Holiday-aware scheduled fills (src/nyse_calendar.py integration) ----------

def test_project_next_weekday_no_longer_exists():
    """Requirement 6: _project_next_weekday (weekend-only projection) must be
    fully replaced by the holiday-aware src.nyse_calendar.next_nyse_session."""
    assert not hasattr(paper, "_project_next_weekday")
    source = Path(paper.__file__).read_text()
    assert "_project_next_weekday" not in source
    assert "nyse_calendar.next_nyse_session" in source


def test_advance_uses_holiday_aware_calendar_for_frontier_projection(env, monkeypatch):
    """White-box: at the frontier (no further finalized session), advance()
    must ask src.nyse_calendar for the next session, not a naive weekday-only
    projection. Patch it to a sentinel and confirm any new pending order
    created on the frontier session carries that sentinel as its
    scheduled_fill_date."""
    d = env.init(env.dir(), start="2023-01-01")
    paper.advance(d, target_date="2023-01-31")  # not yet the frontier of FULL_DATES

    calls = []
    sentinel = pd.Timestamp("2099-06-15")

    def spy(d_):
        calls.append(pd.Timestamp(d_))
        return sentinel

    monkeypatch.setattr(paper.nyse_calendar, "next_nyse_session", spy)
    # Advance all the way to FULL_DATES' last available session (the true
    # frontier). It's a Dec 31 month-end, so a rebalance fires and creates
    # pending orders whose scheduled fill lands on the projected next session.
    paper.advance(d, target_date=str(FULL_DATES[-1].date()))
    assert calls, "nyse_calendar.next_nyse_session was never called at the frontier"
    _cfg, st = paper.load_account(d)
    assert st["pending_orders"], "the frontier month-end rebalance must create pending orders"
    assert any(o["scheduled_fill_date"] == str(sentinel.date()) for o in st["pending_orders"]), (
        "a pending order created on the frontier session must carry the holiday-aware "
        "projected fill date from nyse_calendar.next_nyse_session, not a weekday-only guess"
    )


# --- Stable IDs / FK linkage / duplicate-signal DB constraint ------------------

def test_stable_ids_survive_reload(env):
    d = env.init(env.dir())
    paper.advance(d, target_date="2023-01-31")
    _cfg, st1 = paper.load_account(d)
    ids1 = sorted(o["id"] for o in st1["pending_orders"])
    _cfg, st2 = paper.load_account(d)
    ids2 = sorted(o["id"] for o in st2["pending_orders"])
    assert ids1 == ids2


def test_duplicate_signal_rejected_by_db_constraint(env):
    d = env.init(env.dir())
    conn = _conn(d)
    conn.execute("BEGIN")
    paper_db.insert_signal(conn, "momentum", "AAPL", "2023-01-05", "2023-01-05", 0.2, 1000.0,
                            100.0, 9000.0, "test", "2023-01-06")
    conn.execute("COMMIT")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("BEGIN")
        paper_db.insert_signal(conn, "momentum", "AAPL", "2023-01-05", "2023-01-05", 0.3, 1500.0,
                                100.0, 9000.0, "duplicate", "2023-01-06")
        conn.execute("COMMIT")
    conn.close()


def test_order_and_fill_reference_stable_signal_and_order_ids(env):
    d = env.init(env.dir())
    paper.advance(d, target_date="2023-01-31")
    paper.advance(d)  # fills happen
    conn = _conn(d)
    orders = conn.execute("SELECT id, signal_id FROM orders").fetchall()
    signal_ids = {r["id"] for r in conn.execute("SELECT id FROM signals")}
    assert orders
    for o in orders:
        assert o["signal_id"] in signal_ids
    fills = conn.execute("SELECT id, order_id FROM fills").fetchall()
    order_ids = {r["id"] for r in orders}
    assert fills
    for f in fills:
        assert f["order_id"] in order_ids
    conn.close()


def test_same_ticker_multiple_sessions_gets_distinct_ids_no_collision(env):
    """Requirement 7: IDs are surrogate autoincrement keys, NOT derived from
    (ticker, side, date) -- so the SAME ticker traded across multiple sessions
    (two monthly rebalances one month apart) produces DISTINCT signal / order /
    fill rows with distinct IDs, never a collision that would overwrite the
    earlier transaction."""
    d = env.init(env.dir())
    # Two consecutive month-end rebalances -> the same momentum/sector tickers
    # are signalled and filled twice, on different sessions.
    paper.advance(d, target_date="2023-01-31")
    paper.advance(d)  # fill Jan rebalance
    paper.advance(d, target_date="2023-02-28")
    paper.advance(d)  # fill Feb rebalance

    conn = _conn(d)
    # Find a ticker that was filled on two different dates.
    rows = conn.execute(
        "SELECT ticker, COUNT(DISTINCT fill_date) nd, COUNT(*) n, COUNT(DISTINCT id) nid "
        "FROM fills WHERE shares > 0 GROUP BY sleeve, ticker HAVING nd >= 2"
    ).fetchall()
    conn.close()
    assert rows, "expected at least one ticker filled on two different sessions"
    for r in rows:
        # Every physical fill of that ticker is its own row with its own id --
        # no collision collapsed the two same-ticker fills into one.
        assert r["nid"] == r["n"], f"{r['ticker']}: {r['n']} fills but only {r['nid']} distinct ids"


def test_order_ids_are_surrogate_keys_not_date_derived(env):
    """An order's ID is an ``ORD-<autoincrement>`` surrogate key with no
    ticker/side/date structure, so a holiday shifting the scheduled fill date
    cannot change or collide the row that holds the order."""
    d = env.init(env.dir())
    paper.advance(d, target_date="2023-01-31")
    _cfg, st = paper.load_account(d)
    ids = [o["id"] for o in st["pending_orders"]]
    assert ids and all(i.startswith("ORD-") for i in ids)
    assert len(ids) == len(set(ids))
    for o in st["pending_orders"]:
        # No component of the (ticker, side, date) tuple appears in the id.
        assert o["ticker"] not in o["id"]
        assert o["scheduled_fill_date"] not in o["id"]
        assert str(o["intended_side"]) not in o["id"]


# --- Semantic state corruption fails closed / independent reconciliation -------

def test_reconcile_detects_altered_fill(env):
    d = env.init(env.dir())
    paper.advance(d, target_date="2023-01-31")
    paper.advance(d)  # fills happen
    conn = _conn(d)
    row = conn.execute("SELECT id FROM fills WHERE shares > 0 LIMIT 1").fetchone()
    assert row is not None
    conn.execute("BEGIN")
    conn.execute("UPDATE fills SET fill_price = fill_price * 5.0 WHERE id = ?", (row["id"],))
    conn.execute("COMMIT")
    conn.close()
    recon = paper.reconcile_saved_state(d)
    assert not recon["ok"]
    failed = {c["check"] for c in recon["checks"] if not c["ok"]}
    assert "fill_rows_not_altered" in failed


def test_reconcile_detects_deleted_fill_row(env):
    d = env.init(env.dir())
    paper.advance(d, target_date="2023-01-31")
    paper.advance(d)
    conn = _conn(d)
    row = conn.execute("SELECT id FROM fills LIMIT 1").fetchone()
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("BEGIN")
    conn.execute("DELETE FROM fills WHERE id = ?", (row["id"],))
    conn.execute("COMMIT")
    conn.close()
    recon = paper.reconcile_saved_state(d)
    assert not recon["ok"]
    failed = {c["check"] for c in recon["checks"] if not c["ok"]}
    assert "filled_orders_have_a_fill" in failed


def test_reconcile_detects_negative_cash_beyond_tolerance(env):
    d = env.init(env.dir())
    paper.advance(d, target_date="2023-01-10")
    conn = _conn(d)
    conn.execute("BEGIN")
    conn.execute("UPDATE sleeves SET cash = -5000.0 WHERE name = 'momentum'")
    conn.execute("COMMIT")
    conn.close()
    recon = paper.reconcile_saved_state(d)
    assert not recon["ok"]
    failed = {c["check"] for c in recon["checks"] if not c["ok"]}
    assert "momentum:cash_not_negative" in failed


def test_advance_refuses_on_top_of_inconsistent_ledger(env):
    d = env.init(env.dir())
    paper.advance(d, target_date="2023-01-10")
    conn = _conn(d)
    conn.execute("BEGIN")
    conn.execute("UPDATE sleeves SET allocated_capital = allocated_capital + 1.0 WHERE name = 'momentum'")
    conn.execute("COMMIT")
    conn.close()
    with pytest.raises(paper.PaperStateError):
        paper.advance(d, target_date="2023-01-11")


# --- Transactional rollback: a mid-session exception loses nothing -------------

def test_exception_midsession_rolls_back_everything(env):
    d = env.init(env.dir())
    paper.advance(d, target_date="2023-01-31")
    _cfg, before = paper.load_account(d)
    before_counts = (
        before["last_processed_date"], len(before["completed_trades"]),
        len(before["pending_orders"]), len(before["portfolio_equity_history"]),
    )

    def boom(*a, **kw):
        raise RuntimeError("INJECTED FAILURE mid-session")

    # Manual save/restore (not pytest's monkeypatch): this test's monkeypatch
    # fixture instance is shared with the `env` fixture's price-data mocks, so
    # calling monkeypatch.undo() here would also wipe those out mid-test.
    original = tournament.prepare_sector_plan
    tournament.prepare_sector_plan = boom
    try:
        with pytest.raises(RuntimeError, match="INJECTED FAILURE"):
            paper.advance(d)
    finally:
        tournament.prepare_sector_plan = original

    _cfg, after = paper.load_account(d)
    after_counts = (
        after["last_processed_date"], len(after["completed_trades"]),
        len(after["pending_orders"]), len(after["portfolio_equity_history"]),
    )
    assert before_counts == after_counts

    conn = _conn(d)
    n_sleeve_eq = conn.execute(
        "SELECT COUNT(*) c FROM sleeve_equity_history WHERE session_date > '2023-01-31'"
    ).fetchone()["c"]
    n_signals = conn.execute(
        "SELECT COUNT(*) c FROM signals WHERE signal_date > '2023-01-31'"
    ).fetchone()["c"]
    n_processed = conn.execute(
        "SELECT COUNT(*) c FROM processed_sessions WHERE session_date > '2023-01-31'"
    ).fetchone()["c"]
    conn.close()
    assert n_sleeve_eq == 0
    assert n_signals == 0
    assert n_processed == 0

    # The account is not corrupted -- it can advance normally afterward.
    res = paper.advance(d)
    assert res["processed"]
    assert res["processed"][0]["reconciliation_ok"]


# --- Concurrent-run locking ------------------------------------------------------

def test_concurrent_advance_raises_lock_error(env):
    import fcntl

    d = env.init(env.dir())
    lock_path = Path(d) / ".paper_lock"
    f = open(lock_path, "w")
    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(paper.PaperLockError):
            paper.advance(d, target_date="2023-01-10")
    finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


# --- Stale-order persistence ------------------------------------------------------

def test_stale_order_recorded_when_price_missing_at_fill(env, monkeypatch):
    d = env.init(env.dir())
    paper.advance(d, target_date="2023-01-31")
    _cfg, st = paper.load_account(d)
    assert st["pending_orders"]
    target = st["pending_orders"][0]
    fill_date = pd.Timestamp(target["scheduled_fill_date"])
    ticker = target["ticker"]

    # Blank out that ticker's price on the scheduled fill date only.
    df = env.frame(ticker)
    df.loc[fill_date, ["Open", "High", "Low", "Close"]] = float("nan")

    paper.advance(d)  # process the fill date
    _cfg, st2 = paper.load_account(d)
    stale = [o for o in st2["rejected_stale_orders"]
             if o["sleeve"] == target["sleeve"] and o["ticker"] == ticker]
    assert stale, "expected the order to be recorded stale, not silently filled"
    assert not any(f["sleeve"] == target["sleeve"] and f["ticker"] == ticker
                   and f["fill_date"] == target["scheduled_fill_date"] and f["shares"] > 0
                   for f in st2["completed_trades"])
    assert st2["reconciliation"]["ok"]


# --- Sector sleeve not yet active on the account's early sessions ----------------

def test_sector_sleeve_not_yet_active_does_not_crash_first_sessions(env):
    """A sector-plan sleeve whose effective start (latest ETF inception +
    lookback months) postdates the account's early sessions must not crash the
    run -- it holds cash until it becomes active while the other sleeves trade
    normally. Regression for the inverted [effective_start, exec_end] window
    that would otherwise raise ValueError and, being caught + rolled back,
    leave the account permanently unable to complete its first --paper-run."""
    # Trim XLK so the sector effective start clips forward to 2023-02-01
    # (its inception 2022-11-01 + 3-month lookback), after the account's
    # January sessions but with XLK data present as of them (so prepare_sector_plan
    # itself succeeds -- this specifically exercises the window-inversion guard).
    full = env.frame("XLK")
    env.frames["XLK"] = full.loc[full.index >= pd.Timestamp("2022-11-01")].copy()

    d = env.init(env.dir(), start="2023-01-03")
    res = paper.advance(d, target_date="2023-01-31")  # all sessions < 2023-02-01
    assert res["processed"], "should process the early sessions without crashing"
    _cfg, st = paper.load_account(d)
    assert st["reconciliation"]["ok"], st["reconciliation"]["checks"]
    for name in ("sector_rotation", "regime_switch"):
        s = st["sleeves"][name]
        assert s["positions"] == [], f"{name} should hold no positions before it is active"
        assert s["cash"] == pytest.approx(s["allocated_capital"])
    # The stock-plan momentum sleeve is active from inception and did trade.
    assert any(f["sleeve"] == "momentum" for f in st["completed_trades"])


# --- Required daily-summary fields ------------------------------------------------

REQUIRED_SUMMARY_FIELDS = {
    "paper_date", "data_cutoff_date", "num_new_signals", "num_pending_created",
    "num_fills", "num_stale", "total_equity", "cumulative_return", "daily_return",
    "transaction_costs_run", "reconciliation_ok", "warnings",
}


def test_run_summary_has_all_required_fields(env):
    d = env.init(env.dir())
    res = paper.advance(d, target_date="2023-01-10")
    summary = res["processed"][0]
    missing = REQUIRED_SUMMARY_FIELDS - set(summary)
    assert not missing, f"run summary missing fields: {missing}"

    _cfg, st = paper.load_account(d)
    log_entry = st["run_log"][-1]
    missing_log = REQUIRED_SUMMARY_FIELDS - set(log_entry)
    assert not missing_log, f"persisted run_log entry missing fields: {missing_log}"


# --- Existing modes/flags undisturbed --------------------------------------------

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
