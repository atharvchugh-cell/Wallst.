"""Broker-neutral Phase-4 alerting with durable deduplication."""

from __future__ import annotations

import json
import ipaddress
import math
import os
import socket
import sys
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

import requests

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


class WebhookSink:
    def __init__(self, config: WebhookConfig, *, session=None) -> None:
        self.config = config
        self.session = session or requests.Session()
        if hasattr(self.session, "trust_env"):
            self.session.trust_env = False

    def send(self, alert: dict[str, Any]) -> None:
        hostname = urlsplit(self.config.url).hostname or ""
        try:
            addresses = {
                item[4][0]
                for item in socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
            }
        except OSError as exc:
            raise RuntimeError("Alert webhook DNS resolution failed") from exc
        if not addresses or any(not ipaddress.ip_address(value).is_global for value in addresses):
            raise RuntimeError("Alert webhook resolved to a non-public address")
        # No broker credential or arbitrary header is ever attached.
        payload = {
            "alert_id": alert["alert_id"],
            "severity": alert["severity"],
            "category": alert["category"],
            "message": alert["message"],
            "entity_id": alert["entity_id"],
            "occurrence_count": alert["occurrence_count"],
        }
        response = self.session.request(
            "POST", self.config.url, json=payload,
            headers={"Content-Type": "application/json", "User-Agent": "wallst-strategy-lab/phase4"},
            timeout=self.config.timeout_seconds, allow_redirects=False,
        )
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
        alert, created = self.store.emit_alert(
            severity, category, message, entity_id=entity_id, dedupe_key=dedupe_key
        )
        # Repeats are counted durably but do not flood operators. Escalations
        # are delivered separately by the monitoring/health path.
        if created:
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
