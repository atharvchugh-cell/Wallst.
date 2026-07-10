"""Tests for --strategy portfolio: a single account allocated across weighted,
fully-independent (non-rebalanced) strategy sleeves.

The hard properties proven here:
  - a 100%-of-one-strategy portfolio is byte-identical to running that
    strategy standalone (no wrapper side effects),
  - the 60/35/5 split allocates $9,000 / $5,250 / $750 of a $15,000 account,
  - the portfolio equity curve equals the plain sum of sleeve equity curves on
    the common date intersection,
  - no cash is shared or double-counted across sleeves,
  - weights that don't sum to 1 (or are negative/duplicated/unknown) are
    rejected,
  - existing CLI modes/defaults are undisturbed,
  - and mutating FUTURE prices cannot change past portfolio equity
    (no lookahead is introduced by the combination layer).
"""

import numpy as np
import pandas as pd
import pytest

from src import cli, portfolio, tournament

FULL_DATES = pd.bdate_range("1990-01-01", "2024-12-31")


def shared_trending_df(seed=42, drift=0.01):
    rng = np.random.default_rng(seed)
    base = 100.0 + np.cumsum(rng.normal(drift, 0.5, size=len(FULL_DATES)))
    base = np.clip(base, 10.0, None)
    return pd.DataFrame(
        {"Open": base, "High": base, "Low": base, "Close": base, "Volume": 1000}, index=FULL_DATES
    )


@pytest.fixture
def mocked_data(monkeypatch):
    """Deterministic per-ticker synthetic frames spanning 1990-2024, shared by
    every data accessor the portfolio path touches (tournament sleeve fetches
    AND the portfolio-level SPY benchmark fetch)."""
    frames: dict[str, pd.DataFrame] = {}

    def get_frame(t):
        if t not in frames:
            frames[t] = shared_trending_df(seed=abs(hash(t)) % (2**31))
        return frames[t].copy()

    def fake_get_price_data(tickers, start, end, warmup_calendar_days, hard_fail_on_missing, **kw):
        return {t: get_frame(t) for t in tickers}, []

    def fake_get_benchmark_data(start, end, **kw):
        return get_frame("SPY")

    monkeypatch.setattr(tournament.data, "get_price_data", fake_get_price_data)
    monkeypatch.setattr(tournament.data, "get_benchmark_data", fake_get_benchmark_data)
    monkeypatch.setattr(portfolio.data, "get_benchmark_data", fake_get_benchmark_data)
    return frames


# --- Weight parsing & validation -----------------------------------------------

def test_parse_defaults_to_60_35_5():
    assert portfolio.parse_portfolio_weights(None) == [
        ("momentum", 0.60), ("sector_rotation", 0.35), ("regime_switch", 0.05)
    ]


def test_parse_explicit_string():
    parsed = portfolio.parse_portfolio_weights("momentum=0.5, sector_rotation=0.5")
    assert parsed == [("momentum", 0.5), ("sector_rotation", 0.5)]


@pytest.mark.parametrize("spec", ["momentum", "momentum=", "momentum=abc", "=0.5"])
def test_parse_rejects_malformed_tokens(spec):
    with pytest.raises(portfolio.PortfolioError):
        portfolio.parse_portfolio_weights(spec)


def test_validate_accepts_valid_weights():
    portfolio.validate_portfolio_weights(
        [("momentum", 0.60), ("sector_rotation", 0.35), ("regime_switch", 0.05)]
    )  # sums to 1.0 -- no raise


def test_validate_rejects_weights_not_summing_to_one():
    # 0.60 + 0.35 + 0.10 = 1.05
    with pytest.raises(portfolio.PortfolioError, match="sum to 1.0"):
        portfolio.validate_portfolio_weights(
            [("momentum", 0.60), ("sector_rotation", 0.35), ("regime_switch", 0.10)]
        )
    # ...and a shortfall is rejected too (0.90).
    with pytest.raises(portfolio.PortfolioError, match="sum to 1.0"):
        portfolio.validate_portfolio_weights([("momentum", 0.60), ("sector_rotation", 0.30)])


def test_validate_rejects_negative_weight():
    with pytest.raises(portfolio.PortfolioError, match="non-negative"):
        portfolio.validate_portfolio_weights([("momentum", 1.20), ("sector_rotation", -0.20)])


def test_validate_rejects_duplicate_strategy():
    with pytest.raises(portfolio.PortfolioError, match="Duplicate"):
        portfolio.validate_portfolio_weights([("momentum", 0.6), ("momentum", 0.4)])


def test_validate_rejects_unknown_strategy():
    with pytest.raises(portfolio.PortfolioError, match="Unknown"):
        portfolio.validate_portfolio_weights([("bogus_strategy", 1.0)])


def test_validate_accepts_tiny_floating_point_slack():
    # 0.1 + 0.2 + 0.7 == 0.9999999999999999 in float, within tolerance.
    portfolio.validate_portfolio_weights(
        [("momentum", 0.1), ("sector_rotation", 0.2), ("regime_switch", 0.7)]
    )


def test_allocate_capital_is_proportional():
    alloc = portfolio.allocate_capital(
        [("momentum", 0.60), ("sector_rotation", 0.35), ("regime_switch", 0.05)], 15000.0
    )
    assert alloc == [("momentum", 9000.0), ("sector_rotation", 5250.0), ("regime_switch", 750.0)]
    assert sum(c for _n, c in alloc) == pytest.approx(15000.0)


# --- Capital allocation correctness --------------------------------------------

def test_60_35_5_initial_sleeve_capital_is_correct(mocked_data):
    pf = portfolio.run_portfolio(
        portfolio.parse_portfolio_weights(None), 15000.0, "2023-01-01", "2024-12-31",
        cost_bps=5.0, fractional_shares=True, refresh_cache=False, output_dir="output",
        write_sleeve_artifacts=False,
    )
    by_name = {s.strategy: s for s in pf.sleeves}
    assert by_name["momentum"].allocated_capital == pytest.approx(9000.0)
    assert by_name["sector_rotation"].allocated_capital == pytest.approx(5250.0)
    assert by_name["regime_switch"].allocated_capital == pytest.approx(750.0)
    # The engine received exactly that capital per sleeve.
    assert by_name["momentum"].result.capital == pytest.approx(9000.0)
    assert by_name["sector_rotation"].result.capital == pytest.approx(5250.0)
    assert by_name["regime_switch"].result.capital == pytest.approx(750.0)
    # Allocations sum to the full account -- no capital invented or lost.
    assert sum(s.allocated_capital for s in pf.sleeves) == pytest.approx(15000.0)
    assert pf.combined_result.capital == pytest.approx(15000.0)


# --- 100%-of-one-strategy equals standalone ------------------------------------

@pytest.mark.parametrize("strategy_name", ["momentum", "sector_rotation", "regime_switch"])
def test_100pct_single_strategy_matches_standalone(mocked_data, strategy_name):
    """A portfolio that is 100% one strategy must produce exactly that
    strategy's standalone equity curve (the wrapper adds no behavior and does
    not alter the capital the sleeve sees)."""
    spec = tournament.STRATEGY_REGISTRY[strategy_name]
    standalone = tournament.run_tournament_sleeve(
        spec, "2023-01-01", "2024-12-31", 15000.0, 5.0, True, False, "output",
        write_artifacts=False,
    )

    pf = portfolio.run_portfolio(
        [(strategy_name, 1.0)], 15000.0, "2023-01-01", "2024-12-31",
        cost_bps=5.0, fractional_shares=True, refresh_cache=False, output_dir="output",
        write_sleeve_artifacts=False,
    )

    # The single sleeve's curve and the combined portfolio curve are both
    # exactly the standalone curve.
    pd.testing.assert_series_equal(
        pf.sleeves[0].result.equity_curve, standalone.result.equity_curve, check_names=False
    )
    pd.testing.assert_series_equal(
        pf.combined_result.equity_curve, standalone.result.equity_curve, check_names=False
    )
    assert pf.metrics["total_return"] == pytest.approx(standalone.metrics["total_return"])
    assert pf.metrics["final_equity"] == pytest.approx(standalone.metrics["final_equity"])


# --- Combined == sum of sleeves; no shared cash --------------------------------

def test_combined_equity_equals_sum_of_sleeve_equities(mocked_data):
    pf = portfolio.run_portfolio(
        portfolio.parse_portfolio_weights(None), 15000.0, "2023-01-01", "2024-12-31",
        cost_bps=5.0, fractional_shares=True, refresh_cache=False, output_dir="output",
        write_sleeve_artifacts=False,
    )
    common = pf.combined_result.equity_curve.index
    manual_sum = sum(s.result.equity_curve.reindex(common) for s in pf.sleeves)
    pd.testing.assert_series_equal(
        pf.combined_result.equity_curve, manual_sum, check_names=False
    )
    # Reconciliation identities at the common window end.
    portfolio_final = float(pf.combined_result.equity_curve.iloc[-1])
    assert sum(s.final_value for s in pf.sleeves) == pytest.approx(portfolio_final)
    assert sum(s.ending_weight for s in pf.sleeves) == pytest.approx(1.0)
    assert sum(s.pnl_contribution for s in pf.sleeves) == pytest.approx(portfolio_final - 15000.0)


def test_no_shared_cash_or_double_counted_capital(mocked_data):
    """Each sleeve in the portfolio behaves EXACTLY as if run standalone with
    only its allocated slice of capital -- proving no sleeve ever borrowed cash
    from a sibling and no dollar was counted twice."""
    pairs = portfolio.parse_portfolio_weights(None)
    pf = portfolio.run_portfolio(
        pairs, 15000.0, "2023-01-01", "2024-12-31",
        cost_bps=5.0, fractional_shares=True, refresh_cache=False, output_dir="output",
        write_sleeve_artifacts=False,
    )
    for sleeve in pf.sleeves:
        spec = tournament.STRATEGY_REGISTRY[sleeve.strategy]
        standalone = tournament.run_tournament_sleeve(
            spec, "2023-01-01", "2024-12-31", sleeve.allocated_capital, 5.0, True, False,
            "output", write_artifacts=False,
        )
        pd.testing.assert_series_equal(
            sleeve.result.equity_curve, standalone.result.equity_curve, check_names=False
        )


def test_portfolio_is_not_full_capital_per_sleeve(mocked_data):
    """Guard against the tournament-style mistake of giving every sleeve the
    FULL account: the 60% momentum sleeve must start at $9,000, not $15,000."""
    pf = portfolio.run_portfolio(
        portfolio.parse_portfolio_weights(None), 15000.0, "2023-01-01", "2024-12-31",
        cost_bps=5.0, fractional_shares=True, refresh_cache=False, output_dir="output",
        write_sleeve_artifacts=False,
    )
    momentum = next(s for s in pf.sleeves if s.strategy == "momentum")
    assert momentum.result.capital == pytest.approx(9000.0)
    assert momentum.result.capital != pytest.approx(15000.0)
    # The three sleeves' starting capital sums to the account, not 3x it.
    assert sum(s.result.capital for s in pf.sleeves) == pytest.approx(15000.0)


def test_zero_weight_sleeve_is_skipped_not_run(mocked_data):
    pf = portfolio.run_portfolio(
        [("momentum", 1.0), ("sector_rotation", 0.0)], 15000.0, "2023-01-01", "2024-12-31",
        cost_bps=5.0, fractional_shares=True, refresh_cache=False, output_dir="output",
        write_sleeve_artifacts=False,
    )
    assert [s.strategy for s in pf.sleeves] == ["momentum"]
    assert pf.skipped_zero_weight == ["sector_rotation"]
    assert pf.sleeves[0].allocated_capital == pytest.approx(15000.0)


# --- No lookahead introduced by the portfolio layer ----------------------------

def test_no_lookahead_future_prices_cannot_change_past_portfolio_equity(mocked_data):
    """Run the portfolio, then mutate every sleeve's price data for dates AFTER
    a cutoff and re-run. Portfolio equity on/before the cutoff must be
    unchanged -- the combination layer introduces no way for future data to
    leak into past equity."""
    pairs = portfolio.parse_portfolio_weights(None)
    kwargs = dict(
        total_capital=15000.0, start="2023-01-01", end="2024-12-31", cost_bps=5.0,
        fractional_shares=True, refresh_cache=False, output_dir="output",
        write_sleeve_artifacts=False,
    )
    pf_before = portfolio.run_portfolio(pairs, **kwargs)
    equity_before = pf_before.combined_result.equity_curve.copy()

    cutoff = pd.Timestamp("2024-01-01")
    # Mutate ONLY future close values (keep the calendar/index identical so the
    # common window is unchanged); pre-cutoff data and warmup are untouched.
    for t, df in list(mocked_data.items()):
        mask = df.index > cutoff
        df.loc[mask, ["Open", "High", "Low", "Close"]] *= 3.0
        mocked_data[t] = df

    pf_after = portfolio.run_portfolio(pairs, **kwargs)
    equity_after = pf_after.combined_result.equity_curve

    up_to_cutoff = equity_before.index[equity_before.index <= cutoff]
    assert len(up_to_cutoff) > 50  # sanity: the pre-cutoff span is substantial
    pd.testing.assert_series_equal(
        equity_before.reindex(up_to_cutoff), equity_after.reindex(up_to_cutoff), check_names=False
    )


# --- CLI end-to-end ------------------------------------------------------------

def test_cli_portfolio_end_to_end_writes_all_artifacts(mocked_data, tmp_path):
    exit_code = cli.main([
        "--strategy", "portfolio", "--start", "2023-01-01", "--end", "2024-12-31",
        "--capital", "15000", "--output-dir", str(tmp_path),
    ])
    assert exit_code == 0
    p_dirs = list(tmp_path.glob("*_portfolio_*"))
    assert len(p_dirs) == 1
    run_dir = p_dirs[0]
    for artifact in [
        "portfolio_report.txt", "portfolio_summary.csv", "portfolio_equity.csv",
        "portfolio_sleeves.csv", "portfolio.json",
    ]:
        assert (run_dir / artifact).exists(), f"missing {artifact}"

    # portfolio_equity.csv columns visibly sum to the portfolio column.
    eq = pd.read_csv(run_dir / "portfolio_equity.csv", index_col="date")
    sleeve_cols = [c for c in eq.columns if c != "portfolio"]
    assert set(sleeve_cols) == {"momentum", "sector_rotation", "regime_switch"}
    np.testing.assert_allclose(eq[sleeve_cols].sum(axis=1).values, eq["portfolio"].values, rtol=1e-9)

    # portfolio_summary.csv has a portfolio row and a SPY row with excess return.
    summary = pd.read_csv(run_dir / "portfolio_summary.csv")
    assert set(summary["strategy"]) == {"momentum", "sector_rotation", "regime_switch", "portfolio", "SPY"}
    assert "excess_return" in summary.columns

    # portfolio_sleeves.csv reconciliation total row.
    sleeves = pd.read_csv(run_dir / "portfolio_sleeves.csv")
    total_row = sleeves[sleeves["strategy"] == "PORTFOLIO_TOTAL"].iloc[0]
    assert total_row["allocated_capital"] == pytest.approx(15000.0)
    assert total_row["ending_weight"] == pytest.approx(1.0)

    text = (run_dir / "portfolio_report.txt").read_text()
    assert "STATIC ALLOCATION" in text
    assert "no rebalancing" in text.lower() or "no cash is ever transferred" in text.lower()


def test_cli_portfolio_respects_explicit_weights(mocked_data, tmp_path):
    exit_code = cli.main([
        "--strategy", "portfolio", "--start", "2023-01-01", "--end", "2024-12-31",
        "--capital", "10000", "--output-dir", str(tmp_path),
        "--portfolio-weights", "momentum=0.5,sector_rotation=0.5",
    ])
    assert exit_code == 0
    sleeves = pd.read_csv(list(tmp_path.glob("*_portfolio_*"))[0] / "portfolio_sleeves.csv")
    row_by = {r["strategy"]: r for _i, r in sleeves.iterrows()}
    assert row_by["momentum"]["allocated_capital"] == pytest.approx(5000.0)
    assert row_by["sector_rotation"]["allocated_capital"] == pytest.approx(5000.0)
    assert "regime_switch" not in row_by


def test_cli_portfolio_rejects_bad_weights(mocked_data, tmp_path):
    # Sums to 1.10 -> validation failure -> exit code 1, no artifacts.
    exit_code = cli.main([
        "--strategy", "portfolio", "--start", "2023-01-01", "--end", "2023-12-31",
        "--output-dir", str(tmp_path),
        "--portfolio-weights", "momentum=0.60,sector_rotation=0.50",
    ])
    assert exit_code == 1
    assert list(tmp_path.glob("*_portfolio_*")) == []


# --- Existing modes/defaults undisturbed ---------------------------------------

def test_existing_strategy_choices_and_defaults_unchanged():
    # Adding "portfolio" must not disturb any pre-existing --strategy choice.
    for choice in ["mean_reversion", "sector_rotation", "both", "compare", "robustness", "tournament"]:
        args = cli.parse_args(["--strategy", choice])
        assert args.strategy == choice
        # The new flag defaults OFF for every other mode -- no behavior change.
        assert args.portfolio_weights is None
    # And the new mode is selectable with its own default weights.
    args = cli.parse_args(["--strategy", "portfolio"])
    assert args.strategy == "portfolio"
    assert args.portfolio_weights is None
