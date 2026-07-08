import pandas as pd
import pytest

from src.engine import BacktestResult
from src.metrics import total_return as total_return_fn
from src.robustness import (
    ALLOCATION_MIXES,
    DEFAULT_ROBUSTNESS_WINDOWS,
    average_ranks,
    beats_spy_fraction,
    blend_equity_curve,
    blend_metrics,
    mean_reversion_tradeoff,
    rank_allocations,
)


def make_result(equity_values, dates, capital=1000.0):
    equity = pd.Series(equity_values, index=dates)
    return BacktestResult(
        strategy_name="test", capital=capital, start=dates[0], end=dates[-1], equity_curve=equity,
        target_events=[], transactions=[], trades=[], positions=[], dropped_tickers=[],
        universe=[], cost_bps=0.0, fractional_shares=True,
    )


def test_blend_equity_curve_pure_sector_matches_sector_alone():
    # 100%-weighted blend must reproduce the sector sleeve's RAW equity
    # curve exactly (no rescaling to a different capital, no normalizing
    # away its own starting value).
    dates = pd.bdate_range("2024-01-01", periods=5)
    sr = make_result([100.0, 110.0, 121.0, 121.0, 133.1], dates)
    mr = make_result([100.0, 90.0, 90.0, 81.0, 81.0], dates)

    blended, window_start, window_end = blend_equity_curve(sr, mr, w_sr=1.0, w_mr=0.0)
    pd.testing.assert_series_equal(blended, sr.equity_curve, check_names=False)
    assert window_start == dates[0]
    assert window_end == dates[-1]


def test_blend_equity_curve_pure_mean_reversion_matches_mean_reversion_alone():
    dates = pd.bdate_range("2024-01-01", periods=5)
    sr = make_result([100.0, 110.0, 121.0, 121.0, 133.1], dates)
    mr = make_result([100.0, 90.0, 90.0, 81.0, 81.0], dates)

    blended, _, _ = blend_equity_curve(sr, mr, w_sr=0.0, w_mr=1.0)
    pd.testing.assert_series_equal(blended, mr.equity_curve, check_names=False)


def test_blend_equity_curve_50_50_equals_half_scaled_raw_sum():
    # 50/50 blended equity must equal 0.5*sector_equity + 0.5*mean_reversion_equity
    # over the common index -- a direct sum of raw (not normalized) curves.
    dates = pd.bdate_range("2024-01-01", periods=3)
    sr = make_result([100.0, 200.0, 200.0], dates)  # doubles
    mr = make_result([100.0, 100.0, 100.0], dates)  # flat

    blended, _, _ = blend_equity_curve(sr, mr, w_sr=0.5, w_mr=0.5)
    expected = 0.5 * sr.equity_curve + 0.5 * mr.equity_curve
    pd.testing.assert_series_equal(blended, expected, check_names=False)
    assert blended.iloc[1] == pytest.approx(150.0)
    assert blended.iloc[0] == pytest.approx(100.0)


def test_blend_equity_curve_uses_intersection_of_dates():
    dates_sr = pd.bdate_range("2024-01-01", periods=5)
    dates_mr = pd.bdate_range("2024-01-03", periods=5)  # starts 2 days later
    sr = make_result([100.0] * 5, dates_sr)
    mr = make_result([100.0] * 5, dates_mr)
    _, window_start, window_end = blend_equity_curve(sr, mr, 0.5, 0.5)
    assert window_start == dates_mr[0]  # later of the two starts
    assert window_end == dates_sr[-1] if dates_sr[-1] < dates_mr[-1] else dates_mr[-1]


def test_blend_equity_curve_preserves_first_day_opening_cost():
    # Synthetic sector sleeve whose first equity point is BELOW starting
    # capital because of an opening transaction cost (capital=1000.0 but
    # the first recorded equity point is 995.0, i.e. a $5 day-0 fill cost).
    dates = pd.bdate_range("2024-01-01", periods=4)
    sr = make_result([995.0, 1005.0, 1015.0, 1025.0], dates, capital=1000.0)
    mr = make_result([1000.0, 1000.0, 1000.0, 1000.0], dates, capital=1000.0)  # never trades, no cost

    blended, window_start, _ = blend_equity_curve(sr, mr, w_sr=1.0, w_mr=0.0)
    # The 100%-sector blend must start from the RAW 995.0 first-day economics,
    # not be reset to a clean 1000.0 (full capital, cost-free) starting point.
    assert blended.loc[window_start] == pytest.approx(995.0)
    assert blended.loc[window_start] != pytest.approx(1000.0)


def test_blend_metrics_total_return_preserves_initial_cost_drag():
    # total_return for a 100%-sector allocation must be computed against the
    # true ALLOCATED starting capital (1000.0), not the sleeve's own first
    # recorded equity point (995.0) -- so the $5 opening cost shows up as
    # negative drag baked into the return, not silently dropped.
    dates = pd.bdate_range("2024-01-01", periods=4)
    sr = make_result([995.0, 1005.0, 1015.0, 1045.0], dates, capital=1000.0)
    mr = make_result([1000.0, 1000.0, 1000.0, 1000.0], dates, capital=1000.0)
    sr_metrics = {"total_turnover": 1000.0, "total_transaction_costs": 5.0, "num_transactions": 1,
                  "num_trades": 0, "average_capital_invested_pct": 1.0}
    mr_metrics = {"total_turnover": 0.0, "total_transaction_costs": 0.0, "num_transactions": 0,
                  "num_trades": 0, "average_capital_invested_pct": 0.0}

    blended = blend_metrics(sr, sr_metrics, mr, mr_metrics, w_sr=1.0, w_mr=0.0, capital=1000.0)
    # (1045 - 1000) / 1000 = 4.5%, NOT (1045 - 995) / 995 = ~5.03% (which
    # would silently exclude the opening cost from the reported return).
    assert blended["total_return"] == pytest.approx(0.045)


def test_blend_metrics_pure_sector_matches_standalone_sector_metrics():
    # On a simple known path, a 100% sector / 0% mean-reversion row must
    # match sector_rotation's own standalone total_return computed the same
    # way (basis = the true starting capital both sleeves were run at).
    dates = pd.bdate_range("2024-01-01", periods=4)
    sr = make_result([1000.0, 1100.0, 1210.0, 1331.0], dates, capital=1000.0)
    mr = make_result([1000.0, 950.0, 950.0, 900.0], dates, capital=1000.0)
    sr_metrics = {"total_turnover": 500.0, "total_transaction_costs": 5.0, "num_transactions": 4,
                  "num_trades": 2, "average_capital_invested_pct": 0.9}
    mr_metrics = {"total_turnover": 800.0, "total_transaction_costs": 12.0, "num_transactions": 10,
                  "num_trades": 5, "average_capital_invested_pct": 0.6}

    blended_100_sr = blend_metrics(sr, sr_metrics, mr, mr_metrics, w_sr=1.0, w_mr=0.0, capital=1000.0)
    assert blended_100_sr["total_return"] == pytest.approx(total_return_fn(sr.equity_curve, start_capital=1000.0))
    assert blended_100_sr["total_transaction_costs"] == pytest.approx(5.0)
    assert blended_100_sr["num_transactions"] == 4


def test_blend_metrics_pure_mean_reversion_matches_standalone_mean_reversion_metrics():
    dates = pd.bdate_range("2024-01-01", periods=4)
    sr = make_result([1000.0, 1100.0, 1210.0, 1331.0], dates, capital=1000.0)
    mr = make_result([1000.0, 950.0, 950.0, 900.0], dates, capital=1000.0)
    sr_metrics = {"total_turnover": 500.0, "total_transaction_costs": 5.0, "num_transactions": 4,
                  "num_trades": 2, "average_capital_invested_pct": 0.9}
    mr_metrics = {"total_turnover": 800.0, "total_transaction_costs": 12.0, "num_transactions": 10,
                  "num_trades": 5, "average_capital_invested_pct": 0.6}

    blended_100_mr = blend_metrics(sr, sr_metrics, mr, mr_metrics, w_sr=0.0, w_mr=1.0, capital=1000.0)
    assert blended_100_mr["total_return"] == pytest.approx(total_return_fn(mr.equity_curve, start_capital=1000.0))
    assert blended_100_mr["total_transaction_costs"] == pytest.approx(12.0)
    assert blended_100_mr["num_transactions"] == 10


def test_blend_metrics_cost_and_turnover_are_capital_weighted():
    dates = pd.bdate_range("2024-01-01", periods=3)
    sr = make_result([100.0, 100.0, 100.0], dates)
    mr = make_result([100.0, 100.0, 100.0], dates)
    sr_metrics = {"total_turnover": 1000.0, "total_transaction_costs": 10.0, "num_transactions": 4,
                  "num_trades": 2, "average_capital_invested_pct": 1.0}
    mr_metrics = {"total_turnover": 2000.0, "total_transaction_costs": 40.0, "num_transactions": 8,
                  "num_trades": 4, "average_capital_invested_pct": 0.5}

    blended = blend_metrics(sr, sr_metrics, mr, mr_metrics, w_sr=0.75, w_mr=0.25, capital=1000.0)
    assert blended["total_turnover"] == pytest.approx(0.75 * 1000.0 + 0.25 * 2000.0)
    assert blended["total_transaction_costs"] == pytest.approx(0.75 * 10.0 + 0.25 * 40.0)
    assert blended["cost_drag_pct"] == pytest.approx(blended["total_transaction_costs"] / 1000.0)
    # Trade counts are unaffected by weight as long as weight > 0.
    assert blended["num_transactions"] == 4 + 8
    assert blended["num_trades"] == 2 + 4


def test_blend_metrics_zero_weight_sleeve_contributes_no_trades_or_costs():
    dates = pd.bdate_range("2024-01-01", periods=3)
    sr = make_result([100.0, 100.0, 100.0], dates)
    mr = make_result([100.0, 100.0, 100.0], dates)
    sr_metrics = {"total_turnover": 1000.0, "total_transaction_costs": 10.0, "num_transactions": 4, "num_trades": 2}
    mr_metrics = {"total_turnover": 2000.0, "total_transaction_costs": 40.0, "num_transactions": 8, "num_trades": 4}

    blended = blend_metrics(sr, sr_metrics, mr, mr_metrics, w_sr=1.0, w_mr=0.0, capital=1000.0)
    assert blended["total_turnover"] == pytest.approx(1000.0)  # only sector's turnover
    assert blended["num_transactions"] == 4  # mean_reversion's 8 excluded entirely (0% weight)
    assert blended["num_trades"] == 2


def test_allocation_mixes_weights_sum_to_one():
    for label, w_sr, w_mr in ALLOCATION_MIXES:
        assert w_sr + w_mr == pytest.approx(1.0), label


def test_default_robustness_windows_are_valid_date_ranges():
    for label, start, end in DEFAULT_ROBUSTNESS_WINDOWS:
        assert pd.Timestamp(start) < pd.Timestamp(end), label


def test_rank_allocations_descending_higher_is_better():
    window_metrics = {
        "A": {"total_return": 0.10},
        "B": {"total_return": 0.30},
        "C": {"total_return": 0.20},
    }
    ranks = rank_allocations(window_metrics, "total_return")
    assert ranks == {"B": 1, "C": 2, "A": 3}


def test_rank_allocations_max_drawdown_less_negative_is_better():
    window_metrics = {
        "A": {"max_drawdown": -0.30},
        "B": {"max_drawdown": -0.10},  # smallest magnitude -- best
        "C": {"max_drawdown": -0.20},
    }
    ranks = rank_allocations(window_metrics, "max_drawdown")
    assert ranks == {"B": 1, "C": 2, "A": 3}


def test_rank_allocations_skips_missing_and_nan_values():
    window_metrics = {
        "A": {"total_return": 0.10},
        "B": {"total_return": None},
        "C": {"total_return": float("nan")},
    }
    ranks = rank_allocations(window_metrics, "total_return")
    assert ranks == {"A": 1}


def test_average_ranks_across_windows():
    all_window_metrics = {
        "W1": {"A": {"total_return": 0.30}, "B": {"total_return": 0.10}},  # A rank1, B rank2
        "W2": {"A": {"total_return": 0.05}, "B": {"total_return": 0.20}},  # A rank2, B rank1
    }
    avg = average_ranks(all_window_metrics, "total_return")
    assert avg["A"] == pytest.approx(1.5)
    assert avg["B"] == pytest.approx(1.5)


def test_beats_spy_fraction_counts_correctly():
    all_window_metrics = {
        "W1": {"A": {"total_return": 0.30}, "B": {"total_return": 0.05}},
        "W2": {"A": {"total_return": 0.01}, "B": {"total_return": 0.02}},
    }
    spy_by_window = {"W1": {"total_return": 0.10}, "W2": {"total_return": 0.10}}
    result = beats_spy_fraction(all_window_metrics, spy_by_window, "total_return")
    assert result["A"] == pytest.approx(0.5)  # beat SPY in W1 only
    assert result["B"] == pytest.approx(0.0)  # never beat SPY


def test_mean_reversion_tradeoff_flags_worth_it_correctly():
    baseline_label = ALLOCATION_MIXES[0][0]  # 100% sector
    other_label = ALLOCATION_MIXES[2][0]  # 50/50
    all_window_metrics = {
        "W1": {
            baseline_label: {"max_drawdown": -0.20, "cost_drag_pct": 0.01},
            other_label: {"max_drawdown": -0.10, "cost_drag_pct": 0.03},  # dd improved 0.10, cost up 0.02 -> worth it
        },
        "W2": {
            baseline_label: {"max_drawdown": -0.20, "cost_drag_pct": 0.01},
            other_label: {"max_drawdown": -0.19, "cost_drag_pct": 0.05},  # dd improved 0.01, cost up 0.04 -> not worth it
        },
    }
    rows = mean_reversion_tradeoff(all_window_metrics)
    by_window = {r["window"]: r for r in rows if r["allocation"] == other_label}
    assert by_window["W1"]["worth_it"] is True
    assert by_window["W1"]["drawdown_improvement_pts"] == pytest.approx(0.10)
    assert by_window["W2"]["worth_it"] is False
