"""Boundary tests for the TCCCPR calling-hours gate — the window is
[CALLING_HOURS_START, CALLING_HOURS_END) = [10:00, 19:00) IST by default."""

from datetime import datetime

from app.compliance.calling_hours import within_calling_hours


def _at(hour: int, minute: int) -> datetime:
    return datetime(2026, 7, 20, hour, minute)  # naive -> assumed IST


def test_09_59_is_outside_the_window():
    assert within_calling_hours(_at(9, 59)) is False


def test_10_00_is_the_first_included_minute():
    assert within_calling_hours(_at(10, 0)) is True


def test_18_59_is_the_last_included_minute():
    assert within_calling_hours(_at(18, 59)) is True


def test_19_00_is_outside_the_window():
    assert within_calling_hours(_at(19, 0)) is False


def test_midday_is_inside():
    assert within_calling_hours(_at(14, 30)) is True


def test_midnight_is_outside():
    assert within_calling_hours(_at(0, 0)) is False


def test_accepts_a_tz_aware_datetime_in_a_different_zone():
    import pytz
    utc = pytz.utc
    # 10:00 IST == 04:30 UTC (IST is UTC+5:30)
    assert within_calling_hours(utc.localize(datetime(2026, 7, 20, 4, 30))) is True
    assert within_calling_hours(utc.localize(datetime(2026, 7, 20, 4, 29))) is False
