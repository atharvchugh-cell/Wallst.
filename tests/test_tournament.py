import numpy as np
import pandas as pd
import pytest

from src import cli, config, tournament
from src.reporting import write_tournament_report

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
    return frames


# --- Registry & configuration sanity ------------------------------------------

def test_registry_contains_all_five_strategies_with_expected_plans():
    reg = tournament.STRATEGY_REGISTRY
    assert set(reg) == {
        "mean_reversion", "mean_reversion_filtered", "momentum", "sector_rotation", "regime_switch",
    }
    assert reg["mean_reversion"].data_plan == "stock" and reg["mean_reversion"].uses_stock_universe
    assert reg["momentum"].data_plan == "stock" and reg["momentum"].uses_stock_universe
    assert reg["mean_reversion_filtered"].uses_stock_universe
    assert reg["sector_rotation"].data_plan == "sector" and not reg["sector_rotation"].uses_stock_universe
    assert reg["regime_switch"].data_plan == "sector"


def test_param_sensitivity_variants_are_small_disclosed_and_constructible():
    for name, variants in tournament.PARAM_SENSITIVITY_VARIANTS.items():
        assert name in tournament.STRATEGY_REGISTRY
        assert 1 <= len(variants) <= 4, f"{name}: sweeps must stay small ({len(variants)} variants)"
        spec = tournament.STRATEGY_REGISTRY[name]
        for label, overrides, rationale, _warmup in variants:
            assert rationale, f"{name}/{label}: every variant needs a written rationale"
            if spec.uses_stock_universe:
                spec.factory(universe=["AAPL", "MSFT"], **overrides)  # must not raise
            else:
                spec.factory(**overrides)


def test_parse_tournament_windows():
    assert tournament.parse_tournament_windows(None, "2022-01-01", "2024-12-31") == [
        ("full", "2022-01-01", "2024-12-31")
    ]
    regimes = tournament.parse_tournament_windows("regimes", "2022-01-01", "2024-12-31")
    assert regimes == tournament.REGIME_WINDOWS
    assert len(regimes) == 6
    custom = tournament.parse_tournament_windows("2020-01-01:2020-12-31, 2022-01-01:2022-12-31", "x", "y")
    assert custom == [("2020-2020", "2020-01-01", "2020-12-31"), ("2022-2022", "2022-01-01", "2022-12-31")]


# --- Robustness scoring ---------------------------------------------------------

def test_robustness_components_math():
    all_window_metrics = {
        "w1": {"A": {"total_return": 0.10, "max_drawdown": -0.05},
               "B": {"total_return": -0.02, "max_drawdown": -0.20}},
        "w2": {"A": {"total_return": 0.02, "max_drawdown": -0.10},
               "B": {"total_return": 0.08, "max_drawdown": -0.04}},
    }
    spy_by_window = {"w1": {"total_return": 0.05}, "w2": {"total_return": 0.05}}
    comps = tournament.robustness_components(all_window_metrics, spy_by_window)

    a = comps["A"]
    assert a["num_windows"] == 2
    assert a["pct_windows_beats_spy_return"] == pytest.approx(0.5)  # beats in w1 only
    assert a["pct_windows_positive_return"] == pytest.approx(1.0)
    assert a["worst_window_max_drawdown"] == pytest.approx(-0.10)
    assert a["robustness_score"] == pytest.approx(0.75)
    assert a["return_dispersion"] == pytest.approx(np.std([0.10, 0.02]))

    b = comps["B"]
    assert b["pct_windows_beats_spy_return"] == pytest.approx(0.5)  # beats in w2 only
    assert b["pct_windows_positive_return"] == pytest.approx(0.5)
    assert b["worst_window_max_drawdown"] == pytest.approx(-0.20)
    assert b["robustness_score"] == pytest.approx(0.5)


def test_robustness_components_skips_missing_windows_without_counting_them_as_losses():
    all_window_metrics = {
        "w1": {"A": {"total_return": 0.10, "max_drawdown": -0.05}},
        "w2": {},  # A missing here entirely (e.g. failed run)
    }
    spy_by_window = {"w1": {"total_return": 0.05}, "w2": {"total_return": 0.05}}
    comps = tournament.robustness_components(all_window_metrics, spy_by_window)
    assert comps["A"]["num_windows"] == 1
    assert comps["A"]["pct_windows_beats_spy_return"] == pytest.approx(1.0)


# --- Generic runner equivalence with the incumbent standalone runners -----------

def test_generic_runner_reproduces_mean_reversion_sleeve_equity(mocked_data, tmp_path):
    start, end = "2023-01-01", "2023-12-31"
    result_cli, metrics_cli, _cfg, _dir = cli.run_mean_reversion_sleeve(
        start, end, capital=15000.0, cost_bps=5.0, fractional_shares=True,
        refresh_cache=False, output_dir=str(tmp_path),
    )
    run = tournament.run_tournament_sleeve(
        tournament.STRATEGY_REGISTRY["mean_reversion"], start, end,
        capital=15000.0, cost_bps=5.0, fractional_shares=True, refresh_cache=False,
        output_dir=str(tmp_path), write_artifacts=False,
    )
    pd.testing.assert_series_equal(result_cli.equity_curve, run.result.equity_curve)
    assert metrics_cli["total_return"] == pytest.approx(run.metrics["total_return"])
    assert metrics_cli["num_trades"] == run.metrics["num_trades"]


def test_generic_runner_reproduces_sector_rotation_sleeve_equity(mocked_data, tmp_path):
    start, end = "2023-01-01", "2023-12-31"
    result_cli, metrics_cli, _cfg, _dir = cli.run_sector_rotation_sleeve(
        start, end, capital=15000.0, cost_bps=5.0, fractional_shares=True,
        refresh_cache=False, output_dir=str(tmp_path),
    )
    run = tournament.run_tournament_sleeve(
        tournament.STRATEGY_REGISTRY["sector_rotation"], start, end,
        capital=15000.0, cost_bps=5.0, fractional_shares=True, refresh_cache=False,
        output_dir=str(tmp_path), write_artifacts=False,
    )
    pd.testing.assert_series_equal(result_cli.equity_curve, run.result.equity_curve)
    assert metrics_cli["total_return"] == pytest.approx(run.metrics["total_return"])


def test_generic_runner_hard_fails_on_gapped_non_benchmark_signal_ticker(mocked_data, tmp_path, monkeypatch):
    # SPY itself can never trip this check -- it DEFINES the canonical
    # calendar, so its holes become the calendar's holes (existing, documented
    # benchmark property). The guard exists for any future non-benchmark
    # signal ticker, exercised here via a stub strategy that declares one.
    from src import data as data_module
    from src.strategies.mean_reversion import MeanReversionStrategy

    class StubSignalStrategy(MeanReversionStrategy):
        name = "stub_signal"
        signal_tickers = ["QQQ"]

        def prepare(self, price_data, calendar, start):
            enriched = super().prepare(price_data, calendar, start)
            enriched["QQQ"] = price_data["QQQ"].copy()
            return enriched

    frames = mocked_data
    real_get_price_data = tournament.data.get_price_data

    def gappy_qqq_price_data(tickers, start, end, warmup_calendar_days, hard_fail_on_missing, **kw):
        out, dropped = real_get_price_data(tickers, start, end, warmup_calendar_days, hard_fail_on_missing, **kw)
        if "QQQ" in out:
            df = out["QQQ"]
            out["QQQ"] = df.drop(df.index[(df.index >= "2023-06-01") & (df.index <= "2023-06-10")])
        return out, dropped

    monkeypatch.setattr(tournament.data, "get_price_data", gappy_qqq_price_data)
    spec = tournament.StrategySpec(
        name="stub_signal", factory=StubSignalStrategy, data_plan="stock",
        warmup_calendar_days=200, uses_stock_universe=True,
    )
    with pytest.raises(data_module.FetchError, match="Signal ticker QQQ"):
        tournament.run_tournament_sleeve(
            spec, "2023-01-01", "2023-12-31",
            capital=15000.0, cost_bps=5.0, fractional_shares=True, refresh_cache=False,
            output_dir=str(tmp_path), write_artifacts=False,
        )


# --- Report writer ----------------------------------------------------------------

def test_write_tournament_report_produces_expected_artifacts(tmp_path):
    metrics_by_window = {
        "w1": {
            "momentum": {"total_return": 0.12, "max_drawdown": -0.08, "sharpe_ratio": 1.1,
                         "win_rate": 0.6, "num_trades": 10, "best_year": 0.2, "worst_year": -0.05},
            "SPY": {"total_return": 0.10, "max_drawdown": -0.12, "sharpe_ratio": 0.9},
        },
        "w2": {
            "momentum": {"total_return": -0.02, "max_drawdown": -0.15, "sharpe_ratio": -0.2},
            "SPY": {"total_return": 0.01, "max_drawdown": -0.10, "sharpe_ratio": 0.1},
        },
    }
    robustness = tournament.robustness_components(
        {w: {s: m for s, m in wm.items() if s != "SPY"} for w, wm in metrics_by_window.items()},
        {w: wm.get("SPY", {}) for w, wm in metrics_by_window.items()},
    )
    run_dir = write_tournament_report(
        metrics_by_window,
        {"w1": "2022-01-01 to 2022-12-31", "w2": "2023-01-01 to 2023-12-31"},
        {"momentum": {"name": "momentum", "family": "momentum", "universe_size": 25,
                      "params": {"top_k": 5}, "assumptions": ["survivorship-biased universe"]}},
        {"requested_start": "2022-01-01", "requested_end": "2023-12-31", "capital": 15000.0,
         "cost_bps": 5.0, "strategies": ["momentum"]},
        robustness=robustness,
        cost_sensitivity={"momentum": {0.0: {"total_return": 0.12, "excess_return": 0.02, "cost_drag_pct": 0.0},
                                        20.0: {"total_return": 0.05, "excess_return": -0.05, "cost_drag_pct": 0.04}}},
        param_sensitivity={"momentum": {"baseline": {"total_return": 0.12, "sharpe_ratio": 1.1,
                                                      "max_drawdown": -0.08, "excess_return": 0.02},
                                        "top_k_3": {"total_return": 0.02, "sharpe_ratio": 0.3,
                                                    "max_drawdown": -0.12, "excess_return": -0.08}}},
        param_rationale={"momentum": {"baseline": "defaults", "top_k_3": "more concentrated"}},
        failures=[("w2", "sector_rotation", "simulated failure")],
        output_dir=str(tmp_path),
    )
    assert (run_dir / "tournament_report.txt").exists()
    assert (run_dir / "tournament_summary.csv").exists()
    assert (run_dir / "tournament.json").exists()
    assert (run_dir / "cost_sensitivity.csv").exists()
    assert (run_dir / "param_sensitivity.csv").exists()

    text = (run_dir / "tournament_report.txt").read_text()
    assert "robustness_score = mean(" in text          # formula disclosed inline
    assert "edge vs SPY disappears at higher costs" in text  # sign-flip warning fired
    assert "beat-SPY conclusion FLIPS" in text         # param-fragility warning fired
    assert "NEVER auto-selected" in text
    assert "simulated failure" in text                 # failures surfaced, not hidden
    assert "survivorship-biased universe" in text      # describe() assumptions printed

    summary = pd.read_csv(run_dir / "tournament_summary.csv")
    assert set(summary["strategy"]) == {"momentum", "SPY"}
    assert set(summary["window"]) == {"w1", "w2"}


# --- CLI end-to-end ------------------------------------------------------------------

def test_cli_tournament_end_to_end(mocked_data, tmp_path):
    exit_code = cli.main([
        "--strategy", "tournament", "--start", "2023-01-01", "--end", "2024-12-31",
        "--capital", "15000", "--output-dir", str(tmp_path),
    ])
    assert exit_code == 0
    t_dirs = list(tmp_path.glob("*_tournament_*"))
    assert len(t_dirs) == 1
    summary = pd.read_csv(t_dirs[0] / "tournament_summary.csv")
    assert set(summary["strategy"]) == set(tournament.STRATEGY_REGISTRY) | {"SPY"}
    # Single window -> no robustness section, but the table must be complete.
    assert set(summary["window"]) == {"full"}
    # Every required metric column is present in the shared table.
    for col in ["total_return", "cagr", "max_drawdown", "sharpe_ratio", "sortino_ratio",
                "calmar_ratio", "win_rate", "num_trades", "total_turnover",
                "days_with_any_position_pct", "best_month", "worst_month",
                "best_year", "worst_year", "excess_return"]:
        assert col in summary.columns


def test_cli_tournament_subset_and_multi_window(mocked_data, tmp_path):
    exit_code = cli.main([
        "--strategy", "tournament", "--start", "2022-01-01", "--end", "2024-12-31",
        "--capital", "15000", "--output-dir", str(tmp_path),
        "--tournament-strategies", "momentum,sector_rotation",
        "--tournament-windows", "2022-01-01:2022-12-31,2023-01-01:2024-12-31",
    ])
    assert exit_code == 0
    t_dirs = list(tmp_path.glob("*_tournament_*"))
    summary = pd.read_csv(t_dirs[0] / "tournament_summary.csv")
    assert set(summary["strategy"]) == {"momentum", "sector_rotation", "SPY"}
    assert set(summary["window"]) == {"2022-2022", "2023-2024"}
    text = (t_dirs[0] / "tournament_report.txt").read_text()
    assert "Cross-window robustness" in text  # multi-window -> robustness section present


def test_cli_tournament_rejects_unknown_strategy(mocked_data, tmp_path):
    exit_code = cli.main([
        "--strategy", "tournament", "--start", "2023-01-01", "--end", "2023-12-31",
        "--output-dir", str(tmp_path), "--tournament-strategies", "momentum,bogus",
    ])
    assert exit_code == 1


def test_cli_tournament_rejects_bad_cost_list(mocked_data, tmp_path):
    exit_code = cli.main([
        "--strategy", "tournament", "--start", "2023-01-01", "--end", "2023-12-31",
        "--output-dir", str(tmp_path), "--tournament-cost-bps-list", "5,-1",
    ])
    assert exit_code == 1


def test_cli_existing_strategy_choices_unchanged():
    # Guard: adding "tournament" must not have disturbed the existing modes.
    for choice in ["mean_reversion", "sector_rotation", "both", "compare", "robustness"]:
        args = cli.parse_args(["--strategy", choice])
        assert args.strategy == choice
    # And the new tournament flags default OFF/None -- no behavior change
    # for any pre-existing invocation.
    args = cli.parse_args(["--strategy", "mean_reversion"])
    assert args.tournament_strategies is None
    assert args.tournament_windows is None
    assert args.tournament_cost_bps_list is None
    assert args.tournament_param_sensitivity is False
