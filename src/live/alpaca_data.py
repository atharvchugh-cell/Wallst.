"""Read-only Alpaca IEX snapshots for Phase-3 paper execution.

This adapter is hard-pinned to Alpaca's market-data host and explicitly asks
for the IEX feed available to paper-only accounts.  It has no order methods.
"""

from __future__ import annotations

import math
import requests
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .alpaca_paper import AlpacaPaperConfig, _tls_runtime_problem
from .market_data import MarketDataError, MarketDataProvider
from .models import Quote, as_decimal, ensure_aware, utc_now


DATA_BASE_URL = "https://data.alpaca.markets"
DATA_FEED = "iex"


@dataclass(frozen=True)
class AlpacaDataConfig:
    api_key: str = field(repr=False)
    api_secret: str = field(repr=False)
    base_url: str = DATA_BASE_URL
    feed: str = DATA_FEED
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "api_key", self.api_key.strip())
        object.__setattr__(self, "api_secret", self.api_secret.strip())
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))
        object.__setattr__(self, "feed", self.feed.strip().lower())
        if not self.api_key or not self.api_secret:
            raise ValueError("Paper API key and secret are required for market data")
        if self.base_url != DATA_BASE_URL:
            raise ValueError("AlpacaPaperMarketData is hard-pinned to the data endpoint")
        if self.feed != DATA_FEED:
            raise ValueError("Phase 3 paper market data is hard-pinned to the IEX feed")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")

    @classmethod
    def from_paper_env(cls) -> "AlpacaDataConfig":
        paper = AlpacaPaperConfig.from_env()
        return cls(paper.api_key, paper.api_secret)


class AlpacaPaperMarketData(MarketDataProvider):
    def __init__(
        self,
        config: AlpacaDataConfig,
        *,
        session: requests.Session | Any | None = None,
        clock=utc_now,
    ) -> None:
        self.config = config
        self._uses_default_session = session is None
        self.session = session if session is not None else requests.Session()
        if hasattr(self.session, "trust_env"):
            self.session.trust_env = False
        self.clock = clock
        self._request_ids: list[str] = []

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.config.api_key,
            "APCA-API-SECRET-KEY": self.config.api_secret,
            "Accept": "application/json",
            "User-Agent": "wallst-strategy-lab/phase3",
        }

    def drain_request_ids(self) -> tuple[str, ...]:
        values = tuple(self._request_ids)
        self._request_ids.clear()
        return values

    def get_quotes(self, symbols: tuple[str, ...]) -> dict[str, Quote]:
        if not symbols or len(symbols) > 100 or len(set(symbols)) != len(symbols):
            raise MarketDataError("Request must contain 1-100 unique symbols")
        response = self._request({
            "symbols": ",".join(symbols), "feed": self.config.feed, "currency": "USD"
        })
        try:
            payload = response.json()
        except (TypeError, ValueError) as exc:
            raise MarketDataError("Alpaca market-data response was not valid JSON") from exc
        if not isinstance(payload, dict):
            raise MarketDataError("Alpaca market-data response was not an object")
        if set(payload) != set(symbols):
            raise MarketDataError(
                "Alpaca snapshot response did not exactly cover requested symbols"
            )
        result: dict[str, Quote] = {}
        for symbol in symbols:
            row = payload[symbol]
            if not isinstance(row, dict):
                raise MarketDataError(f"Malformed Alpaca snapshot for {symbol}")
            try:
                latest_quote = row["latestQuote"]
                latest_trade = row["latestTrade"]
                if not isinstance(latest_quote, dict) or not isinstance(latest_trade, dict):
                    raise TypeError("snapshot quote/trade must be objects")
                quote_time = _parse_time(latest_quote["t"])
                trade_time = _parse_time(latest_trade["t"])
                received_at = ensure_aware(self.clock(), "market-data receipt time")
                if any(
                    (timestamp - received_at).total_seconds() > 2
                    for timestamp in (quote_time, trade_time)
                ):
                    raise ValueError("snapshot timestamp is in the future")
                # The conservative timestamp makes the risk engine reject the
                # snapshot if either the BBO or last trade is stale.
                result[symbol] = Quote(
                    symbol=symbol,
                    bid=as_decimal(latest_quote["bp"]),
                    ask=as_decimal(latest_quote["ap"]),
                    last=as_decimal(latest_trade["p"]),
                    as_of=min(quote_time, trade_time),
                )
            except (ArithmeticError, KeyError, TypeError, ValueError) as exc:
                raise MarketDataError(f"Malformed Alpaca snapshot for {symbol}") from exc
        return result

    def _request(self, params: dict[str, str]) -> Any:
        tls_problem = _tls_runtime_problem()
        if self._uses_default_session and tls_problem:
            raise MarketDataError(f"Alpaca market-data network disabled: {tls_problem}")
        path = "/v2/stocks/snapshots"
        try:
            response = self.session.request(
                "GET", f"{self.config.base_url}{path}", headers=self._headers,
                params=params, timeout=self.config.timeout_seconds, allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise MarketDataError("Alpaca market-data GET snapshots transport error") from exc
        request_id = response.headers.get("X-Request-ID", "")
        if request_id:
            self._request_ids.append(str(request_id))
            del self._request_ids[:-100]
        if not 200 <= response.status_code < 300:
            suffix = f" request_id={request_id}" if request_id else ""
            raise MarketDataError(
                f"Alpaca market-data GET snapshots failed HTTP {response.status_code}{suffix}"
            )
        return response


def _parse_time(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("missing market-data timestamp")
    return ensure_aware(
        datetime.fromisoformat(value.replace("Z", "+00:00")), "market-data timestamp"
    )
