"""Broker-neutral Phase-4 alerting with durable deduplication."""

from __future__ import annotations

import json
import ipaddress
import math
import os
import socket
import sys
import threading
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import InvalidURL, ProxyError, SSLError
from requests.utils import select_proxy

from .alpaca_paper import _tls_runtime_problem
from .phase4_store import Phase4Store


class AlertSink(Protocol):
    def send(self, alert: dict[str, Any]) -> None: ...


class StructuredConsoleSink:
    def __init__(self, stream=None) -> None:
        self.stream = stream or sys.stderr

    def send(self, alert: dict[str, Any]) -> None:
        safe = {
            "type": "phase4_alert",
            "alert_id": alert["alert_id"],
            "severity": alert["severity"],
            "category": alert["category"],
            "message": alert["message"],
            "entity_id": alert["entity_id"],
            "occurrence_count": alert["occurrence_count"],
        }
        print(json.dumps(safe, sort_keys=True, separators=(",", ":")), file=self.stream)


@dataclass(frozen=True)
class WebhookConfig:
    url: str
    timeout_seconds: float = 5.0
    allowed_hosts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        parsed = urlsplit(self.url)
        hostname = (parsed.hostname or "").rstrip(".").lower()
        if (
            parsed.scheme != "https" or not hostname or parsed.username or parsed.password
            or parsed.fragment or hostname == "localhost" or hostname.endswith(".localhost")
            or hostname.endswith(".local")
        ):
            raise ValueError("Alert webhook must be a credential-free public HTTPS URL")
        try:
            literal = ipaddress.ip_address(hostname)
        except ValueError:
            literal = None
        if literal is not None and not literal.is_global:
            raise ValueError("Alert webhook IP must be globally routable")
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("Alert webhook port is invalid") from exc
        if port not in {None, 443}:
            raise ValueError("Alert webhook must use the standard HTTPS port")
        allowed = tuple(value.rstrip(".").lower() for value in self.allowed_hosts)
        if not allowed or hostname not in allowed:
            raise ValueError("Alert webhook hostname must be explicitly allowlisted")
        object.__setattr__(self, "allowed_hosts", allowed)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Webhook timeout must be positive and finite")

    @classmethod
    def from_env(cls) -> "WebhookConfig | None":
        value = os.getenv("WSLAB_PHASE4_ALERT_WEBHOOK_URL", "").strip()
        allowed = tuple(
            item.strip().lower()
            for item in os.getenv("WSLAB_PHASE4_ALERT_WEBHOOK_HOST_ALLOWLIST", "").split(",")
            if item.strip()
        )
        return cls(value, allowed_hosts=allowed) if value else None


def _canonical_hostname(value: str) -> str:
    raw = value.rstrip(".").lower()
    try:
        return raw.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("Alert webhook hostname is invalid") from exc


def _host_header(hostname: str) -> str:
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return hostname
    return f"[{address}]" if address.version == 6 else str(address)


class _PinnedHTTPSAdapter(HTTPAdapter):
    """Connect to one validated IP while authenticating the original host.

    The pool's host is the numeric address, so socket creation cannot perform
    a second hostname lookup. ``server_hostname`` supplies TLS SNI and
    ``assert_hostname`` keeps certificate verification bound to the original
    allowlisted hostname. There is intentionally no insecure fallback.
    """

    def __init__(self, hostname: str, pinned_ip: str) -> None:
        self.hostname = _canonical_hostname(hostname)
        address = ipaddress.ip_address(pinned_ip)
        if not address.is_global:
            raise ValueError("Pinned webhook IP must be globally routable")
        self.pinned_ip = str(address)
        super().__init__(max_retries=0)

    def _assert_request_origin(self, request) -> None:
        parsed = urlsplit(request.url)
        request_hostname = _canonical_hostname(parsed.hostname or "")
        try:
            port = parsed.port
        except ValueError as exc:
            raise InvalidURL("Alert webhook port is invalid", request=request) from exc
        if (
            parsed.scheme.lower() != "https"
            or request_hostname != self.hostname
            or port not in {None, 443}
        ):
            raise InvalidURL("Pinned webhook adapter refused a different origin", request=request)

    def get_connection_with_tls_context(
        self, request, verify, proxies=None, cert=None
    ):
        self._assert_request_origin(request)
        if select_proxy(request.url, proxies or {}):
            raise ProxyError("Pinned webhook transport forbids proxies", request=request)
        if verify is False:
            raise SSLError("Pinned webhook transport requires TLS verification", request=request)
        try:
            host_params, pool_kwargs = self.build_connection_pool_key_attributes(
                request, verify, cert
            )
        except ValueError as exc:
            raise InvalidURL(exc, request=request) from exc
        host_params.update({"scheme": "https", "host": self.pinned_ip, "port": 443})
        pool_kwargs.update({
            "server_hostname": self.hostname,
            "assert_hostname": self.hostname,
        })
        return self.poolmanager.connection_from_host(
            **host_params, pool_kwargs=pool_kwargs
        )

    def add_headers(self, request, **kwargs) -> None:
        super().add_headers(request, **kwargs)
        # urllib3 would otherwise derive Host from the numeric connection-pool
        # address. Keep HTTP virtual-host routing bound to the TLS hostname.
        request.headers["Host"] = _host_header(self.hostname)


class WebhookSink:
    def __init__(self, config: WebhookConfig, *, session=None, resolver=None) -> None:
        self.config = config
        self._uses_default_session = session is None
        self.session = session or requests.Session()
        if hasattr(self.session, "trust_env"):
            self.session.trust_env = False
        if not hasattr(self.session, "mount"):
            raise TypeError("Webhook session must support requests transport adapters")
        self._resolver = resolver or socket.getaddrinfo
        self._send_lock = threading.Lock()

    def send(self, alert: dict[str, Any]) -> None:
        with self._send_lock:
            if self._uses_default_session:
                tls_problem = _tls_runtime_problem()
                if tls_problem:
                    raise RuntimeError(f"Alert webhook network disabled: {tls_problem}")
            hostname = _canonical_hostname(urlsplit(self.config.url).hostname or "")
            try:
                info = self._resolver(
                    hostname, 443, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
                )
                addresses: list[str] = []
                for item in info:
                    address = str(ipaddress.ip_address(item[4][0]))
                    if address not in addresses:
                        addresses.append(address)
            except (IndexError, OSError, TypeError, ValueError) as exc:
                raise RuntimeError("Alert webhook DNS resolution failed") from exc
            if not addresses or any(
                not ipaddress.ip_address(value).is_global for value in addresses
            ):
                raise RuntimeError("Alert webhook resolved to a non-public address")

            # Pin this delivery to the first resolver-preferred public address.
            # The adapter still authenticates ``hostname`` through SNI and the
            # certificate SAN/CN, and rejects proxying or disabled TLS checks.
            adapter = _PinnedHTTPSAdapter(hostname, addresses[0])
            prepared_url = requests.Request("POST", self.config.url).prepare().url
            self.session.mount(prepared_url, adapter)

            # No broker credential or arbitrary header is ever attached.
            payload = {
                "alert_id": alert["alert_id"],
                "severity": alert["severity"],
                "category": alert["category"],
                "message": alert["message"],
                "entity_id": alert["entity_id"],
                "occurrence_count": alert["occurrence_count"],
            }
            try:
                response = self.session.request(
                    "POST", self.config.url, json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Host": _host_header(hostname),
                        "User-Agent": "wallst-strategy-lab/phase4",
                    },
                    timeout=self.config.timeout_seconds, allow_redirects=False,
                )
            except requests.RequestException as exc:
                raise RuntimeError("Alert webhook transport failed") from exc
            if not 200 <= response.status_code < 300:
                raise RuntimeError(f"Alert webhook failed HTTP {response.status_code}")


class AlertManager:
    def __init__(self, store: Phase4Store, sinks: tuple[AlertSink, ...] = ()) -> None:
        self.store = store
        self.sinks = sinks

    def emit(
        self,
        severity: str,
        category: str,
        message: str,
        *,
        entity_id: str = "",
        dedupe_key: str | None = None,
    ) -> dict:
        alert, created, upgraded = self.store.emit_alert_with_transition(
            severity, category, message, entity_id=entity_id, dedupe_key=dedupe_key
        )
        # Repeats are counted durably but do not flood operators. Escalations
        # are delivered separately by the monitoring/health path. A severity
        # upgrade is itself a new paging event and must not wait for the later
        # age-based escalation sweep.
        if created or upgraded:
            self.deliver(alert)
        return alert

    def deliver(self, alert: dict[str, Any]) -> None:
        """Deliver an already-durable alert or escalation to configured sinks."""
        for sink in self.sinks:
            try:
                sink.send(alert)
            except Exception as exc:
                self.store.emit_alert(
                    "high", "alert_delivery_failure",
                    f"{type(sink).__name__} failed: {type(exc).__name__}",
                    entity_id=alert["alert_id"],
                    dedupe_key=f"alert-delivery:{type(sink).__name__}",
                )
