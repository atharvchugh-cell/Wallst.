"""Broker-neutral execution foundation and isolated paper operations.

Phase one provides the durable OMS and deterministic fake broker. Phase two
adds an Alpaca paper-only adapter. Phase three adds reviewed aggregation,
market data, immutable approval batches, and explicitly confirmed paper
execution. No live-money broker endpoint is present.
"""

from .alpaca_paper import (
    AlpacaAsset,
    AlpacaMarketClock,
    AlpacaPaperBroker,
    AlpacaPaperConfig,
)
from .fake_broker import FakeBroker
from .alpaca_data import AlpacaPaperMarketData, AlpacaDataConfig
from .deployment import DeploymentConfig, SleeveTargetSnapshot
from .execution import PaperExecutionService
from .ledger import Ledger
from .models import TargetPositionIntent
from .oms import OrderManagementSystem
from .reconcile import Reconciler
from .risk import PreTradeRiskEngine, RiskLimits

__all__ = [
    "FakeBroker",
    "AlpacaPaperMarketData",
    "AlpacaDataConfig",
    "AlpacaPaperBroker",
    "AlpacaPaperConfig",
    "AlpacaAsset",
    "AlpacaMarketClock",
    "Ledger",
    "DeploymentConfig",
    "SleeveTargetSnapshot",
    "PaperExecutionService",
    "OrderManagementSystem",
    "PreTradeRiskEngine",
    "Reconciler",
    "RiskLimits",
    "TargetPositionIntent",
]
