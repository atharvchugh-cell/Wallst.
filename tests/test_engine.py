import pandas as pd
import pytest

from src.engine import run_backtest, combine_results, EventValidationError, EPSILON_SHARES
from src.strategies.base import Strategy, TargetEvent


def make_price_df(closes, dates):
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1000},
        index=dates,
    )


class ScriptedStrategy(Strategy):
    """A test-only strategy that emits exactly the TargetEvents given in
    `schedule` (dict: signal_date -> list of (ticker, fill_date, weight, reason))
    on the matching day, and nothing else. Lets engine tests inject precise
    scenarios without going through real strategy decision logic."""

    name = "scripted"

    def __init__(self, universe, schedule):
        super().__init__()
        self.universe = universe
        self.schedule = schedule

    def on_day(self, day, market, sleeve_equity):
        events = []
        for ticker, fill_date, weight, reason in self.schedule.get(day, []):
            events.append(
                TargetEvent(
                    strategy=self.name, ticker=ticker, signal_date=day, fill_date=fill_date,
                    target_weight=weight, sizing_price=market.close(ticker, day), reason=reason,
                )
            )
        return events


def test_zero_cost_known_price_path_exact_equity():
    dates = pd.bdate_range("2024-01-01", periods=10)
    close = [100.0, 100.0, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0]
    price_data = {"A": make_price_df(close, dates)}
    schedule = {dates[0]: [("A", dates[1], 1.0, "entry")]}
    strat = ScriptedStrategy(["A"], schedule)
    result = run_backtest(strat, price_data, dates, dates[0], dates[-1], capital=1000.0, cost_bps=0.0)
    # Fully invested at day1 close=100 -> at day2 close=110, equity=1100 (10% up)
    assert result.equity_curve.iloc[-1] == pytest.approx(1100.0, rel=1e-9)


def test_nonzero_cost_deducted_exactly():
    dates = pd.bdate_range("2024-01-01", periods=3)
    close = [100.0, 100.0, 100.0]
    price_data = {"A": make_price_df(close, dates)}
    schedule = {dates[0]: [("A", dates[1], 1.0, "entry")]}
    strat = ScriptedStrategy(["A"], schedule)
    result = run_backtest(strat, price_data, dates, dates[0], dates[-1], capital=1000.0, cost_bps=100.0)  # 1%
    # target_weight=1.0 requests ALL of equity as notional, but cost is
    # additional on top -- cash-constrained scaling caps notional so that
    # notional*(1+rate) == capital, i.e. notional = 1000/1.01. Since price is
    # unchanged, final equity == capital - cost == capital * (1/1.01).
    cost_rate = 100.0 / 10000.0
    expected_equity = 1000.0 / (1.0 + cost_rate)
    assert result.equity_curve.iloc[-1] == pytest.approx(expected_equity, rel=1e-9)


def test_no_daily_rebalance_shares_untouched_between_events():
    dates = pd.bdate_range("2024-01-01", periods=10)
    close = [100.0, 100.0, 105.0, 110.0, 120.0, 130.0, 100.0, 90.0, 80.0, 70.0]
    price_data = {"A": make_price_df(close, dates)}
    schedule = {dates[0]: [("A", dates[1], 1.0, "entry")]}  # ONLY one event, ever
    strat = ScriptedStrategy(["A"], schedule)
    result = run_backtest(strat, price_data, dates, dates[0], dates[-1], capital=1000.0, cost_bps=0.0)
    shares_series = [p["shares"] for p in result.positions if p["ticker"] == "A"]
    # shares fixed at 10.0 (1000/100) from the fill day onward -- price moves
    # a lot afterward but no rebalancing transaction should ever occur again.
    assert shares_series[0] == 0.0  # before fill (day0)
    for s in shares_series[1:]:
        assert s == pytest.approx(10.0)
    assert len(result.transactions) == 1  # only the single entry, nothing else


def test_duplicate_event_raises():
    dates = pd.bdate_range("2024-01-01", periods=5)
    price_data = {"A": make_price_df([100.0] * 5, dates)}
    schedule = {
        dates[0]: [("A", dates[2], 1.0, "entry"), ("A", dates[2], 1.0, "entry")],  # duplicate same fill_date
    }
    strat = ScriptedStrategy(["A"], schedule)
    with pytest.raises(EventValidationError):
        run_backtest(strat, price_data, dates, dates[0], dates[-1], capital=1000.0)


def test_fill_date_not_after_signal_date_raises():
    dates = pd.bdate_range("2024-01-01", periods=5)
    price_data = {"A": make_price_df([100.0] * 5, dates)}
    schedule = {dates[2]: [("A", dates[2], 1.0, "entry")]}  # fill == signal, invalid
    strat = ScriptedStrategy(["A"], schedule)
    with pytest.raises(EventValidationError):
        run_backtest(strat, price_data, dates, dates[0], dates[-1], capital=1000.0)


def test_fill_date_outside_calendar_raises():
    dates = pd.bdate_range("2024-01-01", periods=5)
    price_data = {"A": make_price_df([100.0] * 5, dates)}
    bogus_future_date = pd.Timestamp("2030-01-01")
    schedule = {dates[0]: [("A", bogus_future_date, 1.0, "entry")]}
    strat = ScriptedStrategy(["A"], schedule)
    with pytest.raises(EventValidationError):
        run_backtest(strat, price_data, dates, dates[0], dates[-1], capital=1000.0)


def test_average_cost_partial_sell_accounting():
    dates = pd.bdate_range("2024-01-01", periods=6)
    close = [100.0, 100.0, 100.0, 200.0, 200.0, 200.0]
    price_data = {"A": make_price_df(close, dates)}
    schedule = {
        dates[0]: [("A", dates[1], 1.0, "entry")],       # buy full, fills @100
        dates[2]: [("A", dates[3], 0.5, "trim")],         # trim to 50% weight, fills @200
        dates[4]: [("A", dates[5], 0.0, "exit")],         # full exit, fills @200
    }
    strat = ScriptedStrategy(["A"], schedule)
    result = run_backtest(strat, price_data, dates, dates[0], dates[-1], capital=1000.0, cost_bps=0.0)

    trade_types = [t.event_type for t in result.trades]
    assert trade_types == ["partial_sell", "full_exit"]
    # avg_cost_basis at the trim (day3): bought 10 shares @100 -> basis=100
    partial = result.trades[0]
    assert partial.avg_cost_basis == pytest.approx(100.0)
    assert partial.sale_price == pytest.approx(200.0)
    assert partial.realized_pnl > 0
    full_exit = result.trades[1]
    assert full_exit.avg_cost_basis == pytest.approx(100.0)  # unchanged after partial sell


def test_cash_constrained_buy_scaling():
    dates = pd.bdate_range("2024-01-01", periods=4)
    close_a = [100.0, 100.0, 100.0, 100.0]
    close_b = [100.0, 100.0, 100.0, 100.0]
    price_data = {"A": make_price_df(close_a, dates), "B": make_price_df(close_b, dates)}
    # Two simultaneous buy events that together exceed available cash
    schedule = {dates[0]: [("A", dates[1], 0.7, "buyA"), ("B", dates[1], 0.7, "buyB")]}
    strat = ScriptedStrategy(["A", "B"], schedule)
    result = run_backtest(strat, price_data, dates, dates[0], dates[-1], capital=1000.0, cost_bps=0.0)
    total_spent = sum(tx.executed_notional for tx in result.transactions)
    assert total_spent <= 1000.0 + 1e-6
    # Both buys should be scaled down proportionally (same requested weight -> same executed notional)
    notional_a = [tx.executed_notional for tx in result.transactions if tx.ticker == "A"][0]
    notional_b = [tx.executed_notional for tx in result.transactions if tx.ticker == "B"][0]
    assert notional_a == pytest.approx(notional_b)
    assert result.transactions[0].actual_weight_after < 0.7  # scaled below the request


def test_fractional_shares_off_rounds_down():
    dates = pd.bdate_range("2024-01-01", periods=3)
    close = [3.0, 3.0, 3.0]  # 1000/3 = 333.33 shares if fractional; 333 if not
    price_data = {"A": make_price_df(close, dates)}
    schedule = {dates[0]: [("A", dates[1], 1.0, "entry")]}
    strat = ScriptedStrategy(["A"], schedule)
    result = run_backtest(
        strat, price_data, dates, dates[0], dates[-1], capital=1000.0, cost_bps=0.0, fractional_shares=False
    )
    assert result.transactions[0].shares_traded == 333.0
    assert result.transactions[0].shares_traded == float(int(result.transactions[0].shares_traded))


def test_combine_results_intersection_alignment():
    dates_a = pd.bdate_range("2024-01-01", periods=10)
    dates_b = pd.bdate_range("2024-01-05", periods=10)  # starts later
    price_a = {"A": make_price_df([100.0] * 10, dates_a)}
    price_b = {"B": make_price_df([50.0] * 10, dates_b)}
    strat_a = ScriptedStrategy(["A"], {})
    strat_b = ScriptedStrategy(["B"], {})
    result_a = run_backtest(strat_a, price_a, dates_a, dates_a[0], dates_a[-1], capital=1000.0)
    result_b = run_backtest(strat_b, price_b, dates_b, dates_b[0], dates_b[-1], capital=500.0)
    combined = combine_results(result_a, result_b)
    assert combined.start == max(dates_a[0], dates_b[0])
    assert combined.equity_curve.index.min() == max(dates_a[0], dates_b[0])
    expected_first_value = result_a.equity_curve.loc[combined.start] + result_b.equity_curve.loc[combined.start]
    assert combined.equity_curve.iloc[0] == pytest.approx(expected_first_value)


def test_combine_results_never_shares_cash_pool():
    dates = pd.bdate_range("2024-01-01", periods=5)
    price_a = {"A": make_price_df([100.0] * 5, dates)}
    price_b = {"B": make_price_df([100.0] * 5, dates)}
    schedule_a = {dates[0]: [("A", dates[1], 1.0, "entry")]}
    schedule_b = {dates[0]: [("B", dates[1], 1.0, "entry")]}
    strat_a = ScriptedStrategy(["A"], schedule_a)
    strat_b = ScriptedStrategy(["B"], schedule_b)
    result_a = run_backtest(strat_a, price_a, dates, dates[0], dates[-1], capital=7500.0, cost_bps=0.0)
    result_b = run_backtest(strat_b, price_b, dates, dates[0], dates[-1], capital=7500.0, cost_bps=0.0)
    # Each sleeve independently invests its OWN full capital -- if cash were
    # shared, one of these would be starved.
    assert result_a.transactions[0].executed_notional == pytest.approx(7500.0)
    assert result_b.transactions[0].executed_notional == pytest.approx(7500.0)


def test_turnover_and_cost_drag_known_sequence():
    dates = pd.bdate_range("2024-01-01", periods=4)
    price_data = {"A": make_price_df([100.0] * 4, dates)}
    schedule = {
        dates[0]: [("A", dates[1], 1.0, "entry")],
        dates[2]: [("A", dates[3], 0.0, "exit")],
    }
    strat = ScriptedStrategy(["A"], schedule)
    cost_rate = 10.0 / 10000.0  # 0.1%
    result = run_backtest(strat, price_data, dates, dates[0], dates[-1], capital=1000.0, cost_bps=10.0)
    from src.metrics import total_transaction_costs, total_turnover, cost_drag_pct
    # Buy is cash-constrained the same way as the cost test above: notional
    # scales to 1000/(1+rate) so notional+cost == capital. Price is unchanged
    # at the exit, so the sell notional equals the same buy notional exactly.
    buy_notional = 1000.0 / (1.0 + cost_rate)
    expected_turnover = 2 * buy_notional
    assert total_turnover(result.transactions) == pytest.approx(expected_turnover, rel=1e-6)
    costs = total_transaction_costs(result.transactions)
    assert costs == pytest.approx(expected_turnover * cost_rate, rel=1e-6)
    assert cost_drag_pct(result.transactions, 1000.0) == pytest.approx(costs / 1000.0)


def test_missing_price_at_scheduled_fill_date_raises():
    dates = pd.bdate_range("2024-01-01", periods=5)
    price_data = {"A": make_price_df([100.0] * 5, dates)}
    # Schedule a fill on a date that IS in the calendar but doesn't exist in
    # A's own price frame (simulate a data hole) by truncating A's frame.
    truncated = {"A": price_data["A"].drop(price_data["A"].index[3])}
    schedule = {dates[0]: [("A", dates[3], 1.0, "entry")]}
    strat = ScriptedStrategy(["A"], schedule)
    with pytest.raises(EventValidationError):
        run_backtest(strat, truncated, dates, dates[0], dates[-1], capital=1000.0)
