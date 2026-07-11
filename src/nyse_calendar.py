"""A NYSE holiday-aware session calendar, computed from fixed calendar rules
(no network access, no dependency on price data).

The paper ledger uses two different calendars for two different purposes:
  - The FINALIZED session calendar (data.build_canonical_calendar, driven by
    actual SPY bars) tells us which sessions have already happened and have
    settled prices -- it is inherently holiday-aware because there is simply
    no bar on a closed day.
  - This module tells us the next NYSE session *before* it has a bar yet --
    i.e. projecting forward from the most recently processed/finalized
    session to schedule a pending order's fill date. That projection must
    still skip weekends AND exchange holidays, or a pending order scheduled
    to fill on (say) the Friday after Thanksgiving would carry the wrong
    date. `next_nyse_session` is the single source of truth for that
    projection, replacing an earlier weekday-only `_project_next_weekday`
    that did not know about holidays.

Holiday rules (observed per standard NYSE convention: a holiday falling on
Saturday is observed the preceding Friday; on Sunday, the following Monday):
  New Year's Day, Martin Luther King Jr. Day (3rd Monday of January),
  Washington's Birthday / Presidents Day (3rd Monday of February), Good
  Friday (2 days before Easter Sunday), Memorial Day (last Monday of May),
  Juneteenth National Independence Day (June 19, observed from 2022),
  Independence Day (July 4), Labor Day (1st Monday of September),
  Thanksgiving Day (4th Thursday of November), Christmas Day (December 25).
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import pandas as pd

# Juneteenth became an NYSE holiday starting in 2022; earlier years must not
# treat June 19 as a closure (NYSE was open on e.g. 2021-06-18/21).
JUNETEENTH_FIRST_YEAR = 2022


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th occurrence (1-indexed) of `weekday` (Monday=0) in `month`."""
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    d = d + timedelta(days=offset)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """The last occurrence of `weekday` (Monday=0) in `month`."""
    if month == 12:
        next_month_first = date(year + 1, 1, 1)
    else:
        next_month_first = date(year, month + 1, 1)
    d = next_month_first - timedelta(days=1)
    offset = (d.weekday() - weekday) % 7
    return d - timedelta(days=offset)


def _easter_sunday(year: int) -> date:
    """Anonymous Gregorian / Meeus-Jones-Butcher algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _observed(d: date) -> date:
    """Saturday holidays are observed the preceding Friday; Sunday holidays
    the following Monday. Weekday holidays are observed on the day itself."""
    if d.weekday() == 5:  # Saturday
        return d - timedelta(days=1)
    if d.weekday() == 6:  # Sunday
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=None)
def nyse_holidays_for_year(year: int) -> frozenset[date]:
    """The observed NYSE market-closure dates for one calendar year."""
    holidays = {
        _observed(date(year, 1, 1)),               # New Year's Day
        _nth_weekday(year, 1, 0, 3),                # MLK Day
        _nth_weekday(year, 2, 0, 3),                # Washington's Birthday
        _easter_sunday(year) - timedelta(days=2),   # Good Friday
        _last_weekday(year, 5, 0),                  # Memorial Day
        _observed(date(year, 7, 4)),                # Independence Day
        _nth_weekday(year, 9, 0, 1),                # Labor Day
        _nth_weekday(year, 11, 3, 4),                # Thanksgiving
        _observed(date(year, 12, 25)),              # Christmas Day
    }
    if year >= JUNETEENTH_FIRST_YEAR:
        holidays.add(_observed(date(year, 6, 19)))  # Juneteenth
    return frozenset(holidays)


def is_nyse_holiday(d) -> bool:
    d = pd.Timestamp(d)
    return d.date() in nyse_holidays_for_year(d.year)


def is_nyse_session(d) -> bool:
    """True if `d` is a weekday and not an observed NYSE holiday. Does NOT
    know about unscheduled closures (e.g. a one-off national day of
    mourning) -- those are rare and, if they occur, the finalized-session
    calendar (driven by actual price bars) is what governs real settled
    sessions; this function only projects the ORDINARY schedule forward for
    scheduling a not-yet-finalized fill date."""
    d = pd.Timestamp(d)
    if d.dayofweek >= 5:  # Saturday=5, Sunday=6
        return False
    return not is_nyse_holiday(d)


def next_nyse_session(d) -> pd.Timestamp:
    """The next NYSE trading session strictly after `d`, skipping weekends
    and holidays."""
    nxt = pd.Timestamp(d).normalize() + pd.Timedelta(days=1)
    while not is_nyse_session(nxt):
        nxt += pd.Timedelta(days=1)
    return nxt
