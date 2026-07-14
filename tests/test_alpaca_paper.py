"""Contract tests for the paper-only Alpaca adapter; no network is used."""

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

import src.live.alpaca_paper as alpaca_module
from src.live.alpaca_paper import AlpacaPaperBroker, AlpacaPaperConfig, PAPER_BASE_URL
from src.live.broker import BrokerError
from src.live.models import OrderRequest, OrderStatus, OrderType, Side, TimeInForce
from src.live import paper_cli


NOW = datetime(2026, 7, 14, 14, 0, tzinfo=timezone.utc)


class Response:
    def __init__(self, status_code, payload=None, headers=None):
        self.status_code = status_code
        self.payload = payload
        self.headers = headers or {}

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class StubSession:
    def __init__(self, routes):
        self.routes = {key: list(values) for key, values in routes.items()}
        self.calls = []

    def request(self, method, url, **kwargs):
        assert url.startswith(PAPER_BASE_URL)
        path = url[len(PAPER_BASE_URL):]
        self.calls.append((method, path, kwargs))
        try:
            return self.routes[(method, path)].pop(0)
        except (KeyError, IndexError) as exc:
            raise AssertionError(f"No stub route for {method} {path}") from exc


def order_payload(**overrides):
    payload = {
        "id": "broker-order-1",
        "client_order_id": "wslab-order-1",
        "account_id": "paper-account",
        "symbol": "SPY",
        "side": "buy",
        "qty": "10",
        "filled_qty": "0",
        "status": "new",
        "submitted_at": "2026-07-14T14:00:00Z",
        "updated_at": "2026-07-14T14:00:00Z",
        "type": "market",
        "time_in_force": "day",
    }
    payload.update(overrides)
    return payload


def account_payload(**overrides):
    payload = {
        "id": "paper-account",
        "cash": "10000",
        "equity": "10000",
        "buying_power": "10000",
        "last_equity": "10000",
        "status": "ACTIVE",
        "currency": "USD",
        "trading_blocked": False,
        "account_blocked": False,
        "trade_suspended_by_user": False,
    }
    payload.update(overrides)
    return payload


def asset_payload(**overrides):
    payload = {
        "id": "asset-spy",
        "symbol": "SPY",
        "class": "us_equity",
        "status": "active",
        "tradable": True,
    }
    payload.update(overrides)
    return payload


def broker(routes):
    return AlpacaPaperBroker(
        AlpacaPaperConfig("paper-key", "paper-secret"),
        session=StubSession(routes),
        clock=lambda: NOW,
    )


def test_adapter_rejects_every_nonpaper_endpoint_and_missing_credentials(monkeypatch):
    with pytest.raises(ValueError, match="hard-pinned"):
        AlpacaPaperConfig("key", "secret", base_url="https://api.alpaca.markets")
    monkeypatch.delenv("ALPACA_PAPER_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_API_SECRET", raising=False)
    with pytest.raises(ValueError, match="ALPACA_PAPER"):
        AlpacaPaperConfig.from_env()


def test_paper_config_repr_never_exposes_credentials():
    rendered = repr(AlpacaPaperConfig("paper-key", "paper-secret"))
    assert "paper-key" not in rendered
    assert "paper-secret" not in rendered


def test_account_uses_paper_url_and_authenticated_headers_only():
    adapter = broker({
        ("GET", "/v2/account"): [Response(200, account_payload())],
    })
    account = adapter.get_account()
    assert account.account_id == "paper-account"
    assert account.cash == Decimal("10000")
    method, path, kwargs = adapter.session.calls[0]
    assert (method, path) == ("GET", "/v2/account")
    assert kwargs["headers"]["APCA-API-KEY-ID"] == "paper-key"
    assert kwargs["headers"]["APCA-API-SECRET-KEY"] == "paper-secret"
    assert kwargs["timeout"] == 10.0
    assert kwargs["allow_redirects"] is False


def test_market_calendar_maps_official_session_and_rejects_bad_geometry():
    adapter = broker({
        ("GET", "/v2/calendar"): [Response(200, [{
            "date": "2026-07-13", "open": "09:30", "close": "16:00",
        }])],
    })
    days = adapter.get_market_calendar(date(2026, 7, 13), date(2026, 7, 14))
    assert len(days) == 1
    assert days[0].trading_date == date(2026, 7, 13)
    assert days[0].close_at == datetime(2026, 7, 13, 20, 0, tzinfo=timezone.utc)
    _method, _path, kwargs = adapter.session.calls[0]
    assert kwargs["params"] == {
        "start": "2026-07-13", "end": "2026-07-14", "date_type": "TRADING",
    }

    malformed = broker({
        ("GET", "/v2/calendar"): [Response(200, [{
            "date": "2026-07-13", "open": "16:00", "close": "09:30",
        }])],
    })
    with pytest.raises(BrokerError, match="Malformed"):
        malformed.get_market_calendar(date(2026, 7, 13), date(2026, 7, 14))


def test_submit_maps_order_and_sends_deterministic_client_id():
    adapter = broker({
        ("GET", "/v2/account"): [Response(200, account_payload())],
        ("GET", "/v2/assets/SPY"): [Response(200, asset_payload())],
        ("POST", "/v2/orders"): [Response(200, order_payload())],
    })
    order = adapter.submit_order(OrderRequest(
        "paper-account", "wslab-order-1", "intent-1", "SPY", Side.BUY,
        Decimal("10"), Decimal("100"), OrderType.MARKET, TimeInForce.DAY,
    ))
    assert order.status == OrderStatus.SUBMITTED
    assert order.client_order_id == "wslab-order-1"
    _method, _path, kwargs = adapter.session.calls[2]
    assert kwargs["json"] == {
        "symbol": "SPY", "qty": "10", "side": "buy", "type": "market",
        "time_in_force": "day", "client_order_id": "wslab-order-1", "extended_hours": False,
    }


def test_client_lookup_404_is_safe_and_other_errors_do_not_echo_response_payload():
    adapter = broker({
        ("GET", "/v2/orders:by_client_order_id"): [Response(404, {"message": "not found"})],
    })
    assert adapter.get_order_by_client_id("missing") is None

    rejected = broker({("GET", "/v2/account"): [Response(401, {"message": "paper-secret"})]})
    with pytest.raises(BrokerError) as exc:
        rejected.get_account()
    assert "401" in str(exc.value)
    assert "paper-secret" not in str(exc.value)


def test_fill_activity_is_mapped_through_broker_order_and_since_filter():
    adapter = broker({
        ("GET", "/v2/account/activities/FILL"): [Response(200, [{
            "id": "fill-1", "order_id": "broker-order-1", "symbol": "SPY", "side": "buy",
            "qty": "2", "price": "101.25", "transaction_time": "2026-07-14T14:01:00Z",
        }])],
        ("GET", "/v2/orders/broker-order-1"): [Response(200, order_payload(filled_qty="2", status="filled"))],
    })
    fills = adapter.get_fills(since=NOW)
    assert len(fills) == 1
    assert fills[0].client_order_id == "wslab-order-1"
    assert fills[0].quantity == Decimal("2")
    activity_call = adapter.session.calls[0]
    assert activity_call[2]["params"]["after"] == "2026-07-14T14:00:00Z"


def test_open_order_page_limit_and_unsupported_status_fail_closed():
    adapter = broker({("GET", "/v2/orders"): [Response(200, [order_payload()] * 500)]})
    with pytest.raises(BrokerError, match="page limit"):
        adapter.get_open_orders()

    bad = broker({("GET", "/v2/orders:by_client_order_id"): [Response(200, order_payload(status="mystery"))]})
    with pytest.raises(BrokerError, match="Unsupported"):
        bad.get_order_by_client_id("wslab-order-1")


def test_paper_cli_requires_explicit_network_confirmation():
    with pytest.raises(SystemExit) as exc:
        paper_cli.main(["check"])
    assert exc.value.code == 2


def test_account_readiness_flags_are_preserved_and_malformed_flags_fail_closed():
    blocked = broker({
        ("GET", "/v2/account"): [Response(200, account_payload(trading_blocked=True))],
    })
    assert blocked.get_account().trading_blocked is True

    malformed = broker({
        ("GET", "/v2/account"): [Response(200, account_payload(trading_blocked="false"))],
    })
    with pytest.raises(BrokerError, match="Malformed"):
        malformed.get_account()


def test_submit_fails_before_post_for_blocked_account_or_untradable_asset():
    blocked = broker({
        ("GET", "/v2/account"): [Response(200, account_payload(account_blocked=True))],
    })
    request = OrderRequest(
        "paper-account", "wslab-order-1", "intent-1", "SPY", Side.BUY,
        Decimal("10"), Decimal("100"), OrderType.MARKET, TimeInForce.DAY,
    )
    with pytest.raises(BrokerError, match="blocked"):
        blocked.submit_order(request)
    assert all(call[0] != "POST" for call in blocked.session.calls)

    untradable = broker({
        ("GET", "/v2/account"): [Response(200, account_payload())],
        ("GET", "/v2/assets/SPY"): [Response(200, asset_payload(tradable=False))],
    })
    with pytest.raises(BrokerError, match="not tradable"):
        untradable.submit_order(request)
    assert all(call[0] != "POST" for call in untradable.session.calls)


@pytest.mark.parametrize("status", ["done_for_day", "stopped"])
def test_nonterminal_alpaca_statuses_remain_active(status):
    adapter = broker({
        ("GET", "/v2/orders:by_client_order_id"): [
            Response(200, order_payload(status=status))
        ],
    })
    assert adapter.get_order_by_client_id("wslab-order-1").status == OrderStatus.SUBMITTED


def test_done_for_day_with_existing_fill_remains_partially_filled():
    adapter = broker({
        ("GET", "/v2/orders:by_client_order_id"): [
            Response(200, order_payload(status="done_for_day", filled_qty="2"))
        ],
    })
    order = adapter.get_order_by_client_id("wslab-order-1")
    assert order.status == OrderStatus.PARTIALLY_FILLED
    assert order.filled_quantity == Decimal("2")


def test_cancel_422_adopts_fill_race_but_rejects_still_open_order():
    filled = broker({
        ("DELETE", "/v2/orders/broker-order-1"): [Response(422)],
        ("GET", "/v2/orders/broker-order-1"): [
            Response(200, order_payload(status="filled", filled_qty="10"))
        ],
    })
    assert filled.cancel_order("broker-order-1").status == OrderStatus.FILLED

    still_open = broker({
        ("DELETE", "/v2/orders/broker-order-1"): [Response(422)],
        ("GET", "/v2/orders/broker-order-1"): [Response(200, order_payload())],
    })
    with pytest.raises(BrokerError, match="remained submitted"):
        still_open.cancel_order("broker-order-1")


def test_market_clock_and_successful_request_ids_are_exposed():
    adapter = broker({
        ("GET", "/v2/clock"): [Response(200, {
            "timestamp": "2026-07-14T14:00:00Z",
            "is_open": True,
            "next_open": "2026-07-15T13:30:00Z",
            "next_close": "2026-07-14T20:00:00Z",
        }, headers={"X-Request-ID": "request-123"})],
    })
    clock = adapter.get_market_clock()
    assert clock.is_open is True
    assert adapter.drain_request_ids() == ("request-123",)
    assert adapter.request_ids == ()


def test_malformed_numeric_and_fill_rows_fail_as_sanitized_broker_errors():
    malformed_account = broker({
        ("GET", "/v2/account"): [Response(200, account_payload(cash="NaN"))],
    })
    with pytest.raises(BrokerError, match="Malformed"):
        malformed_account.get_account()

    malformed_fills = broker({
        ("GET", "/v2/account/activities/FILL"): [Response(200, ["not-an-object"])],
    })
    with pytest.raises(BrokerError, match="Malformed"):
        malformed_fills.get_fills()


def test_redirects_and_stale_order_snapshots_fail_closed():
    redirected = broker({
        ("GET", "/v2/account"): [
            Response(302, headers={"Location": "https://example.invalid/steal"})
        ],
    })
    with pytest.raises(BrokerError, match="HTTP 302"):
        redirected.get_account()
    assert redirected.session.calls[0][2]["allow_redirects"] is False

    stale = broker({
        ("GET", "/v2/orders:by_client_order_id"): [
            Response(200, order_payload(updated_at="2026-07-14T14:02:00Z")),
            Response(200, order_payload(updated_at="2026-07-14T14:01:00Z")),
        ],
    })
    stale.get_order_by_client_id("wslab-order-1")
    with pytest.raises(BrokerError, match="stale"):
        stale.get_order_by_client_id("wslab-order-1")


def test_account_identity_and_duplicate_broker_rows_fail_closed():
    changed_account = broker({
        ("GET", "/v2/account"): [
            Response(200, account_payload()),
            Response(200, account_payload(id="different-paper-account")),
        ],
    })
    changed_account.get_account()
    with pytest.raises(BrokerError, match="changed during the session"):
        changed_account.get_account()

    position_row = {
        "symbol": "SPY", "qty": "1", "side": "long",
        "avg_entry_price": "100", "current_price": "100",
        "asset_class": "us_equity",
    }
    duplicate_positions = broker({
        ("GET", "/v2/positions"): [Response(200, [position_row, position_row])],
    })
    with pytest.raises(BrokerError, match="duplicate position"):
        duplicate_positions.get_positions()

    duplicate_orders = broker({
        ("GET", "/v2/orders"): [Response(200, [order_payload(), order_payload()])],
    })
    with pytest.raises(BrokerError, match="duplicate open-order"):
        duplicate_orders.get_open_orders()


def test_paper_cli_offline_status_never_constructs_a_broker(tmp_path, monkeypatch, capsys):
    def forbidden():
        raise AssertionError("offline status must not construct a broker")

    monkeypatch.setattr(paper_cli, "_broker", forbidden)
    rc = paper_cli.main([
        "status", "--db", str(tmp_path / "ledger.sqlite3"), "--account-id", "paper-account",
    ])
    assert rc == 0
    assert '"positions_bootstrapped": false' in capsys.readouterr().out


def test_real_network_session_fails_closed_on_unsupported_tls(monkeypatch):
    monkeypatch.setattr(
        alpaca_module, "_tls_runtime_problem", lambda: "unsupported test TLS runtime"
    )
    adapter = AlpacaPaperBroker(AlpacaPaperConfig("paper-key", "paper-secret"))
    assert adapter.session.trust_env is False
    with pytest.raises(BrokerError, match="network disabled"):
        adapter.get_account()
