import pandas as pd
import pytest

from src.market_view import MarketDataView, LookaheadError


@pytest.fixture
def frames():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    return {
        "AAPL": pd.DataFrame(
            {"Close": range(10), "RSI_14": [float(i) for i in range(10)]}, index=idx
        )
    }


@pytest.mark.parametrize("offset_days", [1, 2, 5, 9])
def test_close_raises_beyond_as_of(frames, offset_days):
    idx = frames["AAPL"].index
    as_of = idx[0]
    future_date = as_of + pd.Timedelta(days=offset_days)
    view = MarketDataView(frames, as_of=as_of, calendar=idx)
    with pytest.raises(LookaheadError):
        view.close("AAPL", date=future_date)


@pytest.mark.parametrize("offset_days", [1, 3, 8])
def test_indicator_raises_beyond_as_of(frames, offset_days):
    idx = frames["AAPL"].index
    as_of = idx[0]
    future_date = as_of + pd.Timedelta(days=offset_days)
    view = MarketDataView(frames, as_of=as_of, calendar=idx)
    with pytest.raises(LookaheadError):
        view.indicator("AAPL", "RSI_14", date=future_date)


def test_close_and_indicator_allowed_at_or_before_as_of(frames):
    idx = frames["AAPL"].index
    as_of = idx[5]
    view = MarketDataView(frames, as_of=as_of, calendar=idx)
    assert view.close("AAPL") == 5
    assert view.close("AAPL", date=idx[2]) == 2
    assert view.indicator("AAPL", "RSI_14") == 5.0


def test_history_bounded_at_as_of(frames):
    idx = frames["AAPL"].index
    as_of = idx[4]
    view = MarketDataView(frames, as_of=as_of, calendar=idx)
    hist = view.history("AAPL")
    assert hist.index.max() == as_of
    assert len(hist) == 5  # idx[0..4]

    hist_lookback = view.history("AAPL", lookback=2)
    assert len(hist_lookback) == 2
    assert list(hist_lookback.values) == [3, 4]


def test_has_data(frames):
    idx = frames["AAPL"].index
    view = MarketDataView(frames, as_of=idx[3], calendar=idx)
    assert view.has_data("AAPL", idx[2]) is True
    assert view.has_data("AAPL", idx[5]) is False  # beyond as_of
    assert view.has_data("MSFT", idx[0]) is False  # unknown ticker


def test_next_trading_day(frames):
    idx = frames["AAPL"].index
    view = MarketDataView(frames, as_of=idx[3], calendar=idx)
    assert view.next_trading_day() == idx[4]
    assert view.next_trading_day(idx[8]) == idx[9]
    assert view.next_trading_day(idx[9]) is None  # last date in calendar


def test_is_month_end_true_and_false_within_range():
    idx = pd.bdate_range("2024-01-29", "2024-02-05")  # Jan 29,30,31 (Wed-Fri), Feb 1,2,5
    frames = {"X": pd.DataFrame({"Close": range(len(idx))}, index=idx)}
    view = MarketDataView(frames, as_of=idx[0], calendar=idx)
    jan_31 = pd.Timestamp("2024-01-31")
    assert jan_31 in idx
    assert view.is_month_end(jan_31) is True
    assert view.is_month_end(pd.Timestamp("2024-01-29")) is False


def test_is_month_end_false_for_last_date_in_dataset():
    # Last date in the fetched range can't be confirmed as a true month-end
    # without a following trading day in the dataset -- must be False even
    # if it happens to be a real month-end, per documented simplification.
    idx = pd.bdate_range("2024-01-29", "2024-01-31")
    frames = {"X": pd.DataFrame({"Close": range(len(idx))}, index=idx)}
    view = MarketDataView(frames, as_of=idx[-1], calendar=idx)
    assert idx[-1] == pd.Timestamp("2024-01-31")
    assert view.is_month_end(idx[-1]) is False


def test_unknown_ticker_raises_key_error(frames):
    idx = frames["AAPL"].index
    view = MarketDataView(frames, as_of=idx[0], calendar=idx)
    with pytest.raises(KeyError):
        view.close("MSFT")
