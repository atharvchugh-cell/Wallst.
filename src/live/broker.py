"""Execution-broker contract.

Phase one ships :class:`src.live.fake_broker.FakeBroker`; phase two implements
this same contract for an isolated paper adapter. The interface stays narrow so
broker-specific objects do not leak into the OMS or risk engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from .models import AccountSnapshot, BrokerOrder, Fill, OrderRequest, Position


class BrokerError(RuntimeError):
    pass


class Broker(ABC):
    @abstractmethod
    def get_account(self) -> AccountSnapshot:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> list[Position]:
        raise NotImplementedError

    @abstractmethod
    def get_open_orders(self) -> list[BrokerOrder]:
        raise NotImplementedError

    @abstractmethod
    def get_recent_orders(self, since: datetime | None = None) -> list[BrokerOrder]:
        """Return recent orders in every status, including terminal orders."""
        raise NotImplementedError

    @abstractmethod
    def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        raise NotImplementedError

    @abstractmethod
    def submit_order(self, request: OrderRequest) -> BrokerOrder:
        """Submit idempotently by ``request.client_order_id``."""
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> BrokerOrder:
        raise NotImplementedError

    @abstractmethod
    def get_fills(self, since: datetime | None = None) -> list[Fill]:
        raise NotImplementedError
