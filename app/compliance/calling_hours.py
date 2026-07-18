"""compliance/calling_hours.py — TCCCPR calling-hours gate.

Pure logic: no I/O, no Redis, no DB. app/scheduler.py's campaign_tick calls
within_calling_hours() before enqueueing anything; leads outside the window
are deferred to the next tick, not dropped.

Window is [CALLING_HOURS_START, CALLING_HOURS_END) — start inclusive, end
exclusive, so with the default 10/19 a lead can be dialled at exactly 10:00
IST but not at exactly 19:00 IST.
"""

from __future__ import annotations

from datetime import datetime

import pytz

from app.config import CALLING_HOURS_END, CALLING_HOURS_START, CALLING_HOURS_TZ


def within_calling_hours(now: datetime | None = None) -> bool:
    """*now*: an explicit datetime for testability. Naive datetimes are
    assumed to already be in the calling-hours timezone (IST by default);
    tz-aware datetimes are converted. Defaults to the real current time."""
    tz = pytz.timezone(CALLING_HOURS_TZ)
    if now is None:
        now = datetime.now(tz)
    elif now.tzinfo is not None:
        now = now.astimezone(tz)
    else:
        now = tz.localize(now)

    return CALLING_HOURS_START <= now.hour < CALLING_HOURS_END
