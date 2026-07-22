"""Boundary tests for the calling-hours gate.

These pin the GATE'S LOGIC, not the deployed window. The window is
operator-configured (`CALLING_HOURS_START` / `CALLING_HOURS_END`) and this
deployment currently runs it fully open — see CLAUDE.md. Tests that asserted
"09:59 is outside" therefore started failing the moment the owner widened the
window, which was the tests reporting a configuration choice as a code defect.

So every test below sets the window it is testing. That keeps the half-open
`[start, end)` semantics, the IST conversion and the timezone handling under
test no matter what the operator's `.env` says — which is the property that
actually matters, since the gate is what campaign_tick and the worker both
consult before dialling anyone.
"""

from datetime import datetime

import pytest

from app.compliance import calling_hours
from app.compliance.calling_hours import within_calling_hours


@pytest.fixture
def window(monkeypatch):
    """Set the calling window for one test, independent of .env."""
    def _set(start: int, end: int) -> None:
        monkeypatch.setattr(calling_hours, "CALLING_HOURS_START", start)
        monkeypatch.setattr(calling_hours, "CALLING_HOURS_END", end)
    return _set


def _at(hour: int, minute: int) -> datetime:
    return datetime(2026, 7, 20, hour, minute)  # naive -> assumed IST


def test_the_start_hour_is_the_first_included_minute(window):
    window(10, 19)
    assert within_calling_hours(_at(10, 0)) is True


def test_the_minute_before_the_start_hour_is_outside(window):
    window(10, 19)
    assert within_calling_hours(_at(9, 59)) is False


def test_the_last_minute_before_the_end_hour_is_included(window):
    window(10, 19)
    assert within_calling_hours(_at(18, 59)) is True


def test_the_end_hour_itself_is_outside(window):
    """Half-open [start, end): 19:00 with end=19 must be refused. An
    inclusive end would dial for a whole extra hour."""
    window(10, 19)
    assert within_calling_hours(_at(19, 0)) is False


def test_midday_is_inside(window):
    window(10, 19)
    assert within_calling_hours(_at(14, 30)) is True


def test_midnight_is_outside_a_daytime_window(window):
    window(10, 19)
    assert within_calling_hours(_at(0, 0)) is False


def test_a_fully_open_window_admits_every_hour(window):
    """The configuration this deployment currently runs. 00:00-24:00 must
    admit every hour including both ends of the day — if the half-open
    comparison excluded 00:00 or 23:59, the operator would get silent gaps
    at exactly the hours they widened the window to reach."""
    window(0, 24)
    for hour in (0, 1, 9, 12, 19, 22, 23):
        assert within_calling_hours(_at(hour, 0)) is True, f"{hour}:00 was refused"
    assert within_calling_hours(_at(23, 59)) is True


def test_a_narrow_window_still_bounds_correctly(window):
    window(14, 15)
    assert within_calling_hours(_at(13, 59)) is False
    assert within_calling_hours(_at(14, 0)) is True
    assert within_calling_hours(_at(14, 59)) is True
    assert within_calling_hours(_at(15, 0)) is False


def test_a_tz_aware_datetime_is_converted_before_comparing(window):
    """The gate is defined in IST but callers may hand it any zone. 10:00 IST
    is 04:30 UTC; comparing the raw UTC hour would refuse the whole morning."""
    import pytz

    window(10, 19)
    utc = pytz.utc
    assert within_calling_hours(utc.localize(datetime(2026, 7, 20, 4, 30))) is True
    assert within_calling_hours(utc.localize(datetime(2026, 7, 20, 4, 29))) is False
