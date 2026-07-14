"""Read-only Alpaca IEX snapshot adapter tests; no network is used."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.live.alpaca_data import (
    DATA_BASE_URL,
    AlpacaDataConfig,
    AlpacaPaperMarketData,
)
from src.live.market_data import MarketDataError


NOW = datetime(2026, 7, 14, 14, 0, tzinfo=timezone.utc)


class Response:
    def __init__(self, status_code, payload, headers=None):
        self.status_code = status_code
        self.payload = payload
        self.headers = headers or {}

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class Session:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.response


def snapshot(symbol="SPY"):
    return {
        symbol: {
            "latestQuote": {"bp": 99.9, "ap": 100.1, "t": "2026-07-14T14:00:00Z"},
            "latestTrade": {"p": 100, "t": "2026-07-14T13:59:59Z"},
        }
    }


def test_data_adapter_is_hard_pinned_to_data_host_and_iex():
    with pytest.raises(ValueError, match="hard-pinned"):
        AlpacaDataConfig("key", "secret", base_url="https://example.com")
    with pytest.raises(ValueError, match="IEX"):
        AlpacaDataConfig("key", "secret", feed="sip")


def test_snapshot_maps_quote_and_uses_oldest_required_timestamp():
    session = Session(Response(200, snapshot(), {"X-Request-ID": "data-request-1"}))
    adapter = AlpacaPaperMarketData(
        AlpacaDataConfig("paper-key", "paper-secret"), session=session, clock=lambda: NOW
    )
    quote = adapter.get_quotes(("SPY",))["SPY"]
    assert quote.bid == Decimal("99.9")
    assert quote.ask == Decimal("100.1")
    assert quote.last == Decimal("100")
    assert quote.as_of == datetime(2026, 7, 14, 13, 59, 59, tzinfo=timezone.utc)
    method, url, kwargs = session.calls[0]
    assert (method, url) == ("GET", f"{DATA_BASE_URL}/v2/stocks/snapshots")
    assert kwargs["params"] == {"symbols": "SPY", "feed": "iex", "currency": "USD"}
    assert kwargs["allow_redirects"] is False
    assert kwargs["headers"]["APCA-API-KEY-ID"] == "paper-key"
    assert adapter.drain_request_ids() == ("data-request-1",)


@pytest.mark.parametrize(
    "payload,match",
    [
        ({}, "exactly cover"),
        ({"SPY": {}}, "Malformed"),
        ({"SPY": {"latestQuote": {"bp": 101, "ap": 100, "t": "2026-07-14T14:00:00Z"},
                   "latestTrade": {"p": 100, "t": "2026-07-14T14:00:00Z"}}}, "Malformed"),
    ],
)
def test_snapshot_missing_or_malformed_data_fails_closed(payload, match):
    adapter = AlpacaPaperMarketData(
        AlpacaDataConfig("key", "secret"), session=Session(Response(200, payload))
    )
    with pytest.raises(MarketDataError, match=match):
        adapter.get_quotes(("SPY",))


def test_http_error_does_not_echo_response_or_credentials():
    adapter = AlpacaPaperMarketData(
        AlpacaDataConfig("paper-key", "paper-secret"),
        session=Session(Response(403, {"message": "paper-secret"})),
    )
    with pytest.raises(MarketDataError) as exc:
        adapter.get_quotes(("SPY",))
    assert "403" in str(exc.value)
    assert "paper-secret" not in str(exc.value)


def test_future_component_timestamp_cannot_hide_behind_older_component():
    payload = snapshot()
    payload["SPY"]["latestQuote"]["t"] = "2026-07-14T14:00:10Z"
    adapter = AlpacaPaperMarketData(
        AlpacaDataConfig("key", "secret"),
        session=Session(Response(200, payload)),
        clock=lambda: NOW,
    )
    with pytest.raises(MarketDataError, match="Malformed"):
        adapter.get_quotes(("SPY",))
