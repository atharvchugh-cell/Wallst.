import pandas as pd
import pytest

from src.engine import run_backtest, combine_results
from src.metrics import compute_all_metrics, spy_standalone_metrics
from src.reporting import compute_sleeve_contribution, write_comparison_report, write_robustness_report
from src.robustness import ALLOCATION_MIXES, blend_metrics
from src.strategies.base import Strategy, TargetEvent


def make_price_df(closes, dates):
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1000},
        index=dates,
    )


class ScriptedStrategy(Strategy):
    name = "scripted"

    def __init__(self, universe, schedule):
        super().__init__()
        self.universe = universe
        self.schedule = schedule

    def on_day(self, day, market, sleeve_equity, filled_tickers=frozenset()):
        events = []
        for ticker, fill_date, weight, reason in self.schedule.get(day, []):
            events.append(
                TargetEvent(
                    strategy=self.name, ticker=ticker, signal_date=day, fill_date=fill_date,
                    target_weight=weight, sizing_price=market.close(ticker, day), reason=reason,
                )
            )
        return events


def _build_mr_sr_combined():
    """Two simple, independently-run sleeves with fully overlapping ranges
    and known price paths, combined -- mirrors what cli.py's `compare` mode
    assembles, but with hand-computable numbers."""
    dates = pd.bdate_range("2024-01-01", periods=6)
    # Mean-reversion sleeve: buys A at fill (close=100), price doubles -> gains.
    price_mr = {"A": make_price_df([100.0, 100.0, 200.0, 200.0, 200.0, 200.0], dates)}
    schedule_mr = {dates[0]: [("A", dates[1], 1.0, "entry")]}
    strat_mr = ScriptedStrategy(["A"], schedule_mr)
    mr_result = run_backtest(strat_mr, price_mr, dates, dates[0], dates[-1], capital=1000.0, cost_bps=0.0)

    # Sector-rotation sleeve: buys B at fill (close=100), price is flat -> no gain.
    price_sr = {"B": make_price_df([100.0] * 6, dates)}
    schedule_sr = {dates[0]: [("B", dates[1], 1.0, "entry")]}
    strat_sr = ScriptedStrategy(["B"], schedule_sr)
    sr_result = run_backtest(strat_sr, price_sr, dates, dates[0], dates[-1], capital=1000.0, cost_bps=0.0)

    combined = combine_results(mr_result, sr_result)
    return mr_result, sr_result, combined, dates


def test_compute_sleeve_contribution_known_values():
    mr_result, sr_result, combined, dates = _build_mr_sr_combined()

    contribution = compute_sleeve_contribution(mr_result, sr_result, combined)

    # A doubled (1000 -> 2000): +1000 dollar gain. B stayed flat: +0.
    assert contribution["mr_dollar_gain"] == pytest.approx(1000.0)
    assert contribution["sr_dollar_gain"] == pytest.approx(0.0)
    # All of the combined gain came from mean_reversion.
    assert contribution["mr_return_contribution_pct"] == pytest.approx(1.0)
    assert contribution["sr_return_contribution_pct"] == pytest.approx(0.0)
    # Zero costs in both sleeves (cost_bps=0.0) -- contribution % undefined, not divide-by-zero.
    assert contribution["mr_transaction_costs"] == pytest.approx(0.0)
    assert contribution["sr_transaction_costs"] == pytest.approx(0.0)
    assert pd.isna(contribution["mr_cost_contribution_pct"])


def test_compute_sleeve_contribution_cost_split():
    dates = pd.bdate_range("2024-01-01", periods=4)
    price_a = {"A": make_price_df([100.0] * 4, dates)}
    price_b = {"B": make_price_df([100.0] * 4, dates)}
    # A trades twice (in, then out) -- pays cost twice. B trades once (in only).
    schedule_a = {
        dates[0]: [("A", dates[1], 1.0, "entry")],
        dates[1]: [("A", dates[2], 0.0, "exit")],
    }
    schedule_b = {dates[0]: [("B", dates[1], 1.0, "entry")]}
    strat_a = ScriptedStrategy(["A"], schedule_a)
    strat_b = ScriptedStrategy(["B"], schedule_b)
    result_a = run_backtest(strat_a, price_a, dates, dates[0], dates[-1], capital=1000.0, cost_bps=100.0)
    result_b = run_backtest(strat_b, price_b, dates, dates[0], dates[-1], capital=1000.0, cost_bps=100.0)
    combined = combine_results(result_a, result_b)

    contribution = compute_sleeve_contribution(result_a, result_b, combined)
    assert len(result_a.transactions) == 2
    assert len(result_b.transactions) == 1
    assert contribution["mr_transaction_costs"] > contribution["sr_transaction_costs"] > 0
    total_costs = contribution["mr_transaction_costs"] + contribution["sr_transaction_costs"]
    assert contribution["mr_cost_contribution_pct"] == pytest.approx(
        contribution["mr_transaction_costs"] / total_costs
    )


def test_write_comparison_report_creates_expected_files(tmp_path):
    mr_result, sr_result, combined, dates = _build_mr_sr_combined()
    mr_metrics = compute_all_metrics(mr_result)
    sr_metrics = compute_all_metrics(sr_result)
    combined_metrics = compute_all_metrics(combined)
    spy_close = pd.Series([50.0, 51.0, 52.0, 53.0, 54.0, 55.0], index=dates)
    spy_metrics = spy_standalone_metrics(spy_close)

    metrics_by_label = {
        "mean_reversion": mr_metrics, "sector_rotation": sr_metrics, "both": combined_metrics, "SPY": spy_metrics,
    }
    equity_by_label = {
        "mean_reversion": mr_result.equity_curve, "sector_rotation": sr_result.equity_curve,
        "both": combined.equity_curve, "SPY": spy_close,
    }
    ranges_by_label = {k: "n/a" for k in metrics_by_label}
    contribution = compute_sleeve_contribution(mr_result, sr_result, combined)
    run_config = {"requested_start": "2024-01-01", "requested_end": "2024-01-08", "capital": 2000.0, "cost_bps": 0.0}

    run_dir = write_comparison_report(
        metrics_by_label, equity_by_label, ranges_by_label, years=[2024], run_config=run_config,
        contribution=contribution, output_dir=str(tmp_path),
    )

    assert (run_dir / "comparison.csv").exists()
    assert (run_dir / "comparison.txt").exists()
    assert (run_dir / "annual_returns.csv").exists()
    assert (run_dir / "monthly_returns.csv").exists()
    assert (run_dir / "comparison.json").exists()

    comparison_df = pd.read_csv(run_dir / "comparison.csv")
    assert list(comparison_df.columns) == ["metric", "mean_reversion", "sector_rotation", "both", "SPY"]
    assert "Total return" in comparison_df["metric"].values
    assert "Sharpe" in comparison_df["metric"].values
    assert "Correlation to SPY" in comparison_df["metric"].values

    annual_df = pd.read_csv(run_dir / "annual_returns.csv")
    assert list(annual_df["year"]) == [2024]
    assert "mean_reversion" in annual_df.columns
    # A doubled over the window -- 2024's annual return for mean_reversion should be +100%.
    assert annual_df.loc[0, "mean_reversion"] == pytest.approx(1.0)

    monthly_df = pd.read_csv(run_dir / "monthly_returns.csv")
    assert "SPY" in monthly_df.columns

    txt = (run_dir / "comparison.txt").read_text()
    assert "Strategy Comparison Report" in txt
    assert "Strategy contribution" in txt
    assert "diagnostics" in txt.lower()


def _build_one_robustness_window(mr_result, sr_result, spy_close):
    # blend_metrics requires `capital` to match the capital both sleeves
    # were actually run at (1000.0, from _build_mr_sr_combined) -- it no
    # longer renormalizes each curve to an arbitrary hypothetical capital.
    mr_metrics = compute_all_metrics(mr_result)
    sr_metrics = compute_all_metrics(sr_result)
    alloc_metrics = {
        label: blend_metrics(sr_result, sr_metrics, mr_result, mr_metrics, w_sr, w_mr, capital=1000.0,
                              benchmark_close=spy_close)
        for label, w_sr, w_mr in ALLOCATION_MIXES
    }
    alloc_metrics["SPY"] = spy_standalone_metrics(spy_close)
    return alloc_metrics


def test_write_robustness_report_creates_expected_files(tmp_path):
    mr_result, sr_result, _combined, dates = _build_mr_sr_combined()
    spy_close = pd.Series([50.0, 51.0, 49.0, 52.0, 53.0, 55.0], index=dates)

    all_window_metrics = {
        "2024-2024": _build_one_robustness_window(mr_result, sr_result, spy_close),
    }
    window_ranges = {"2024-2024": f"{dates[0].date()} to {dates[-1].date()}"}
    run_config = {"requested_start": "2024-01-01", "requested_end": "2024-01-08", "capital": 2000.0, "cost_bps": 0.0}

    run_dir = write_robustness_report(all_window_metrics, window_ranges, run_config, output_dir=str(tmp_path))

    assert (run_dir / "robustness_summary.csv").exists()
    assert (run_dir / "robustness_summary.txt").exists()
    assert (run_dir / "robustness_rankings.csv").exists()
    assert (run_dir / "robustness_heatmap_data.csv").exists()
    assert (run_dir / "robustness_summary.json").exists()

    summary_df = pd.read_csv(run_dir / "robustness_summary.csv")
    assert set(summary_df["window"]) == {"2024-2024"}
    assert set(summary_df["allocation"]) == {label for label, _, _ in ALLOCATION_MIXES} | {"SPY"}
    assert "total_return" in summary_df.columns

    rankings_df = pd.read_csv(run_dir / "robustness_rankings.csv")
    assert len(rankings_df) == len(ALLOCATION_MIXES)  # SPY excluded from the ranked-allocation table
    assert "avg_rank_total_return" in rankings_df.columns
    assert "pct_windows_beats_spy_return" in rankings_df.columns

    heatmap_df = pd.read_csv(run_dir / "robustness_heatmap_data.csv")
    assert "2024-2024" in heatmap_df.columns
    assert "SPY" in heatmap_df["allocation"].values

    txt = (run_dir / "robustness_summary.txt").read_text()
    assert "Robustness Testing Report" in txt
    assert "Best allocation per window" in txt
    assert "Average rank across windows" in txt
    assert "beats SPY" in txt
    assert "drawdown protection justify its cost drag" in txt
    # 100% mean-reversion allocation gained (A doubled) -- must appear as the
    # 2024-2024 window's best-by-total-return allocation.
    assert "0% sector / 100% mean-reversion" in txt


def test_write_robustness_report_pure_allocations_match_underlying_sleeve(tmp_path):
    # Sanity-check the report end to end: the 100%-sector-only row's total
    # return in the CSV must equal sector_rotation's own standalone return.
    mr_result, sr_result, _combined, dates = _build_mr_sr_combined()
    spy_close = pd.Series([50.0] * 6, index=dates)
    all_window_metrics = {"W": _build_one_robustness_window(mr_result, sr_result, spy_close)}
    window_ranges = {"W": "n/a"}
    run_config = {"requested_start": "2024-01-01", "requested_end": "2024-01-08", "capital": 2000.0, "cost_bps": 0.0}

    run_dir = write_robustness_report(all_window_metrics, window_ranges, run_config, output_dir=str(tmp_path))
    summary_df = pd.read_csv(run_dir / "robustness_summary.csv")
    sr_only_row = summary_df[
        (summary_df["window"] == "W") & (summary_df["allocation"] == "100% sector / 0% mean-reversion")
    ].iloc[0]
    from src.metrics import total_return
    assert sr_only_row["total_return"] == pytest.approx(total_return(sr_result.equity_curve))
