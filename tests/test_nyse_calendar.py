"""Tests for the NYSE holiday-aware session calendar (src/nyse_calendar.py),
which replaces the earlier weekday-only frontier projection. Dates below are
well-documented NYSE market holidays, verified independently via the
Anonymous Gregorian Easter algorithm for the movable Good Friday case."""

import pandas as pd
import pytest

from src import nyse_calendar as cal


# --- Named holidays (requirement: MLK, Good Friday, July 4 observed, Thanksgiving, Christmas observed) --

def test_mlk_day_is_a_holiday():
    assert cal.is_nyse_holiday("2024-01-15")  # 3rd Monday of Jan 2024
    assert not cal.is_nyse_session("2024-01-15")


def test_good_friday_is_a_holiday():
    for d in ["2024-03-29", "2025-04-18", "2026-04-03"]:
        assert cal.is_nyse_holiday(d), d
        assert not cal.is_nyse_session(d), d


def test_july_4th_observed_on_preceding_friday_when_saturday():
    # July 4, 2020 fell on a Saturday -> observed Friday July 3, 2020.
    assert not cal.is_nyse_holiday("2020-07-04")  # the actual Saturday isn't separately flagged
    assert cal.is_nyse_holiday("2020-07-03")
    assert not cal.is_nyse_session("2020-07-03")
    # An ordinary (non-observed) July 4th, e.g. 2024 (Thursday), is a holiday on the day itself.
    assert cal.is_nyse_holiday("2024-07-04")


def test_thanksgiving_is_a_holiday():
    assert cal.is_nyse_holiday("2024-11-28")  # 4th Thursday of Nov 2024
    assert not cal.is_nyse_session("2024-11-28")


def test_christmas_observed_on_preceding_friday_when_saturday():
    # Christmas Day 2021 fell on a Saturday -> observed Friday Dec 24, 2021.
    assert cal.is_nyse_holiday("2021-12-24")
    assert not cal.is_nyse_session("2021-12-24")


def test_christmas_observed_on_following_monday_when_sunday():
    # Christmas Day 2022 fell on a Sunday -> observed Monday Dec 26, 2022.
    assert cal.is_nyse_holiday("2022-12-26")
    assert not cal.is_nyse_session("2022-12-26")


def test_juneteenth_only_a_holiday_from_2022_onward():
    assert cal.is_nyse_holiday("2022-06-20")  # 2022-06-19 is a Sunday -> observed Monday
    assert cal.is_nyse_session("2021-06-18")  # NYSE was open; Juneteenth wasn't yet a market holiday
    assert cal.is_nyse_session("2021-06-21")


# --- Ordinary weekends -----------------------------------------------------

def test_weekends_are_not_sessions():
    assert not cal.is_nyse_session("2024-06-01")  # Saturday
    assert not cal.is_nyse_session("2024-06-02")  # Sunday
    assert cal.is_nyse_session("2024-06-03")      # Monday


# --- next_nyse_session skips weekends AND holidays --------------------------

def test_next_session_skips_thanksgiving():
    nxt = cal.next_nyse_session("2024-11-27")  # Wed before Thanksgiving
    assert nxt == pd.Timestamp("2024-11-29")   # skips Thu 11/28 (holiday)


def test_next_session_skips_weekend_and_holiday_combo():
    # 2022: Juneteenth (June 19) is a Sunday -> observed Monday 6/20, so
    # Friday 6/17 -> next session skips the weekend AND the observed Monday
    # holiday, landing on Tuesday 6/21.
    nxt = cal.next_nyse_session("2022-06-17")
    assert nxt == pd.Timestamp("2022-06-21")


def test_next_session_across_christmas_and_new_year():
    nxt = cal.next_nyse_session("2024-12-24")  # Tue before Christmas (Wed 12/25 holiday)
    assert nxt == pd.Timestamp("2024-12-26")


def test_next_session_is_strictly_after_input():
    nxt = cal.next_nyse_session("2024-06-03")
    assert nxt > pd.Timestamp("2024-06-03")
    assert nxt == pd.Timestamp("2024-06-04")


@pytest.mark.parametrize("d", ["2024-01-15", "2024-03-29", "2024-11-28"])
def test_next_session_never_lands_on_a_holiday(d):
    nxt = cal.next_nyse_session(d)
    assert cal.is_nyse_session(nxt)
