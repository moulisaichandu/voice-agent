"""compliance/consent.py — Consent-basis validation.

Every dial must record which consent basis and timestamp authorised it,
retained for audit (migrations/0001_init.sql's leads.consent_basis/consent_at).
Pure logic: takes already-loaded lead fields, no I/O.
"""

from __future__ import annotations

from datetime import datetime

import pytz

from app.config import CALLING_HOURS_TZ

VALID_CONSENT_BASES = frozenset({"explicit", "inferred"})


def normalize_consent_at(value: datetime) -> datetime:
    """Return consent timestamps in the configured calling timezone.

    Spreadsheet dates often arrive without an offset. Treating those as the
    server's local timezone made the same consent mean different instants on a
    developer laptop, a UTC container, and production. Naive values are
    explicitly interpreted as the business timezone; aware values are
    converted there.
    """
    tz = pytz.timezone(CALLING_HOURS_TZ)
    if value.tzinfo is None:
        return tz.localize(value)
    return value.astimezone(tz)


def has_valid_consent(consent_basis: str | None, consent_at: datetime | None) -> bool:
    """A lead is dial-eligible on consent grounds only if both an allowed
    basis AND a real timestamp are present — one without the other is not a
    valid audit trail. A future-dated timestamp is rejected outright: it can
    only mean a data-entry/clock error, never a legitimate consent record."""
    if consent_basis not in VALID_CONSENT_BASES:
        return False
    if consent_at is None:
        return False
    normalized = normalize_consent_at(consent_at)
    return normalized <= datetime.now(normalized.tzinfo)
