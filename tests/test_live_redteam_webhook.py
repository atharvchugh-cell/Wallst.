"""Adversarial tests for DNS-pinned Phase-4 webhook delivery."""

from types import SimpleNamespace
import socket

import pytest
import requests
from requests.exceptions import InvalidURL, ProxyError, SSLError

import src.live.alerts as alerts_module
from src.live.alerts import WebhookConfig, WebhookSink, _PinnedHTTPSAdapter


PUBLIC_IP = "93.184.216.34"
WEBHOOK_HOST = "alerts.example.com"
WEBHOOK_URL = f"https://{WEBHOOK_HOST}/phase4"


def alert_payload():
    return {
        "alert_id": "alert-redteam-1",
        "severity": "critical",
        "category": "reconciliation_mismatch",
        "message": "paper account requires operator review",
        "entity_id": "PAPER",
        "occurrence_count": 1,
    }


def address_info(*addresses):
    rows = []
    for address in addresses:
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        sockaddr = (address, 443, 0, 0) if family == socket.AF_INET6 else (address, 443)
        rows.append((family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr))
    return rows


class RecordingSession(requests.Session):
    def __init__(self):
        super().__init__()
        self.sent = []

    def send(self, request, **kwargs):
        self.sent.append({
            "request": request,
            "kwargs": kwargs,
            "adapter": self.get_adapter(request.url),
        })
        return SimpleNamespace(status_code=204)


def config():
    return WebhookConfig(
        WEBHOOK_URL, timeout_seconds=3.25, allowed_hosts=(WEBHOOK_HOST,)
    )


def test_webhook_delivery_mounts_numeric_ip_adapter_and_preserves_controls():
    session = RecordingSession()
    resolver_calls = []

    def resolver(host, port, **kwargs):
        resolver_calls.append((host, port, kwargs))
        # A second lookup would model a DNS-rebinding opportunity and must not
        # occur in WebhookSink before handing off to the numeric-IP adapter.
        if len(resolver_calls) > 1:
            return address_info("127.0.0.1")
        return address_info(PUBLIC_IP)

    WebhookSink(config(), session=session, resolver=resolver).send(alert_payload())

    assert len(resolver_calls) == 1
    sent = session.sent[0]
    adapter = sent["adapter"]
    assert isinstance(adapter, _PinnedHTTPSAdapter)
    assert adapter.pinned_ip == PUBLIC_IP
    assert adapter.hostname == WEBHOOK_HOST
    assert sent["request"].headers["Host"] == WEBHOOK_HOST
    assert sent["kwargs"]["timeout"] == 3.25
    assert sent["kwargs"]["allow_redirects"] is False
    assert not sent["kwargs"]["proxies"]
    assert session.trust_env is False


def test_pinned_adapter_connects_to_ip_but_keeps_sni_and_hostname_verification():
    adapter = _PinnedHTTPSAdapter(WEBHOOK_HOST, PUBLIC_IP)
    request = requests.Request("POST", WEBHOOK_URL).prepare()

    pool = adapter.get_connection_with_tls_context(
        request, verify=True, proxies={}, cert=None
    )
    adapter.add_headers(request)

    assert pool.host == PUBLIC_IP
    assert pool.port == 443
    assert pool.assert_hostname == WEBHOOK_HOST
    assert pool.conn_kw["server_hostname"] == WEBHOOK_HOST
    assert pool.cert_reqs == "CERT_REQUIRED"
    assert request.headers["Host"] == WEBHOOK_HOST


def test_pinned_adapter_refuses_tls_disable_proxy_and_origin_changes():
    adapter = _PinnedHTTPSAdapter(WEBHOOK_HOST, PUBLIC_IP)
    request = requests.Request("POST", WEBHOOK_URL).prepare()

    with pytest.raises(SSLError, match="requires TLS verification"):
        adapter.get_connection_with_tls_context(request, verify=False, proxies={})
    with pytest.raises(ProxyError, match="forbids proxies"):
        adapter.get_connection_with_tls_context(
            request, verify=True, proxies={"https": "http://proxy.example:8080"}
        )
    changed = requests.Request("POST", "https://other.example.com/phase4").prepare()
    with pytest.raises(InvalidURL, match="different origin"):
        adapter.get_connection_with_tls_context(changed, verify=True, proxies={})


@pytest.mark.parametrize(
    "addresses",
    [
        ("127.0.0.1",),
        (PUBLIC_IP, "10.0.0.7"),
        ("169.254.169.254",),
        ("::1",),
    ],
)
def test_webhook_rejects_private_or_mixed_dns_answers_before_request(addresses):
    session = RecordingSession()
    sink = WebhookSink(
        config(), session=session, resolver=lambda *_args, **_kwargs: address_info(*addresses)
    )

    with pytest.raises(RuntimeError, match="non-public"):
        sink.send(alert_payload())
    assert session.sent == []


def test_default_network_session_fails_closed_on_unsupported_tls_runtime(monkeypatch):
    resolved = False

    def resolver(*_args, **_kwargs):
        nonlocal resolved
        resolved = True
        return address_info(PUBLIC_IP)

    monkeypatch.setattr(
        alerts_module, "_tls_runtime_problem", lambda: "unsupported test TLS runtime"
    )
    sink = WebhookSink(config(), resolver=resolver)

    with pytest.raises(RuntimeError, match="network disabled"):
        sink.send(alert_payload())
    assert resolved is False


def test_ipv6_pin_uses_numeric_pool_and_bracketed_http_host():
    public_ipv6 = "2001:4860:4860::8888"
    adapter = _PinnedHTTPSAdapter(public_ipv6, public_ipv6)
    request = requests.Request("POST", f"https://[{public_ipv6}]/phase4").prepare()

    pool = adapter.get_connection_with_tls_context(request, verify=True, proxies={})
    adapter.add_headers(request)

    assert pool.host == public_ipv6
    assert pool.assert_hostname == public_ipv6
    assert pool.conn_kw["server_hostname"] == public_ipv6
    assert request.headers["Host"] == f"[{public_ipv6}]"
