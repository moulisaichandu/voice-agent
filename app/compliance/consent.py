"""compliance/consent.py — Consent-basis validation.

Every dial must record which consent basis and timestamp authorised it,
retained for audit (migrations/0001_init.sql's leads.consent_basis/consent_at).
Pure logic: takes already-loaded lead fields, no I/O.
"""

from __future__ import annotations

from datetime import datetime

VALID_CONSENT_BASES = frozenset({"explicit", "inferred"})


def has_valid_consent(consent_basis: str | None, consent_at: datetime | None) -> bool:
    """A lead is dial-eligible on consent grounds only if both an allowed
    basis AND a real timestamp are present — one without the other is not a
    valid audit trail. A future-dated timestamp is rejected outright: it can
    only mean a data-entry/clock error, never a legitimate consent record."""
    if consent_basis not in VALID_CONSENT_BASES:
        return False
    if consent_at is None:
        return False
    now = datetime.now(consent_at.tzinfo) if consent_at.tzinfo else datetime.now()
    return consent_at <= now
