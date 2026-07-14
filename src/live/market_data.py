"""Broker-neutral market-data and exchange-session validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from .models import Quote, ensure_aware, utc_now


NEW_YORK = ZoneInfo("America/New_York")


class MarketDataError(RuntimeError):
    pass


class MarketDataProvider(ABC):
    @abstractmethod
    def get_quotes(self, symbols: tuple[str, ...]) -> dict[str, Quote]:
        raise NotImplementedError


@dataclass(frozen=True)
class MarketCalendarDay:
    trading_date: date
    open_at: datetime
    close_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.trading_date, date) or isinstance(self.trading_date, datetime):
            raise ValueError("calendar trading_date must be a date")
        object.__setattr__(self, "open_at", ensure_aware(self.open_at, "calendar.open_at"))
        object.__setattr__(self, "close_at", ensure_aware(self.close_at, "calendar.close_at"))
        local_open = self.open_at.astimezone(NEW_YORK)
        local_close = self.close_at.astimezone(NEW_YORK)
        if (
            local_open.date() != self.trading_date
            or local_close.date() != self.trading_date
            or self.close_at <= self.open_at
        ):
            raise ValueError("calendar session times are inconsistent with trading_date")


@dataclass(frozen=True)
class MarketSession:
    timestamp: datetime
    is_open: bool
    next_open: datetime
    next_close: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_aware(self.timestamp, "session.timestamp"))
        object.__setattr__(self, "next_open", ensure_aware(self.next_open, "session.next_open"))
        object.__setattr__(self, "next_close", ensure_aware(self.next_close, "session.next_close"))
        if not isinstance(self.is_open, bool):
            raise ValueError("session.is_open must be boolean")

    @property
    def trading_date(self) -> str:
        return self.timestamp.astimezone(NEW_YORK).date().isoformat()


def validate_regular_session(
    session: MarketSession,
    *,
    now: datetime | None = None,
    max_clock_age_seconds: int = 30,
    future_tolerance_seconds: int = 2,
) -> MarketSession:
    """Fail closed unless the exchange clock proves regular US hours are open."""
    now = ensure_aware(now or utc_now(), "session validation time")
    age = (now - session.timestamp).total_seconds()
    if age < -future_tolerance_seconds:
        raise MarketDataError("Exchange clock timestamp is in the future")
    if age > max_clock_age_seconds:
        raise MarketDataError("Exchange clock response is stale")
    if not session.is_open:
        raise MarketDataError("US equity market is closed")
    local = session.timestamp.astimezone(NEW_YORK)
    if local.weekday() >= 5 or not (time(9, 30) <= local.time() < time(16, 0)):
        raise MarketDataError("Exchange clock open flag is outside regular US equity hours")
    close_local = session.next_close.astimezone(NEW_YORK)
    if session.next_close <= session.timestamp or close_local.date() != local.date():
        raise MarketDataError("Exchange clock next_close is inconsistent with the open session")
    if session.next_open <= session.timestamp:
        raise MarketDataError("Exchange clock next_open did not advance")
    return session


def validate_quotes(
    quotes: dict[str, Quote],
    symbols: tuple[str, ...],
    *,
    now: datetime,
    max_age_seconds: int,
    future_tolerance_seconds: int = 2,
) -> None:
    expected = set(symbols)
    if set(quotes) != expected:
        raise MarketDataError(
            "Market-data response did not exactly cover requested symbols; "
            f"missing={sorted(expected - set(quotes))}, "
            f"unknown={sorted(set(quotes) - expected)}"
        )
    now = ensure_aware(now, "quote validation time")
    for symbol in symbols:
        quote = quotes[symbol]
        if quote.symbol != symbol:
            raise MarketDataError(f"Market-data symbol mismatch for {symbol}")
        age = (now - quote.as_of).total_seconds()
        if age < -future_tolerance_seconds:
            raise MarketDataError(f"Quote for {symbol} is in the future")
        if age > max_age_seconds:
            raise MarketDataError(f"Quote for {symbol} is stale")


def validate_signal_session(
    days: tuple[MarketCalendarDay, ...],
    *,
    signal_at: datetime,
    execution_date: str,
) -> MarketCalendarDay:
    """Prove that a daily-close signal follows the immediately prior session."""
    signal_at = ensure_aware(signal_at, "signal_at")
    try:
        execution_day = date.fromisoformat(execution_date)
    except (TypeError, ValueError) as exc:
        raise MarketDataError("Execution trading date is invalid") from exc
    dates = [day.trading_date for day in days]
    if dates != sorted(dates) or len(dates) != len(set(dates)):
        raise MarketDataError("Market calendar contains duplicate or unsorted sessions")
    if execution_day not in dates:
        raise MarketDataError("Market calendar does not confirm the execution session")
    signal_local = signal_at.astimezone(NEW_YORK)
    prior = [day for day in days if day.trading_date < execution_day]
    if not prior or prior[-1].trading_date != signal_local.date():
        raise MarketDataError(
            "Daily-close signal is not from the immediately preceding exchange session"
        )
    signal_day = prior[-1]
    if signal_at < signal_day.close_at:
        raise MarketDataError("Daily-close signal predates the official session close")
    return signal_day
