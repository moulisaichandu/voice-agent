"""compliance/dnd.py — Phone normalization + DND-set membership check.

Normalization is pure and shared by every path that touches a phone number
(Sheets sync, DND scrub, lead matching) so they can never silently disagree
about a lead's canonical phone — the exact class of bug that would otherwise
let a DND-listed number slip through because one path formatted it differently
than another. India-only (+91), per the project's confirmed scope — unlike
ai-voice-agent's leadfiles.normalize_phone, no configurable country code.

The actual Redis DND-SET LOOKUP is deliberately NOT here — that's I/O, done by
app/telephony/worker.py via `sismember` (O(1), not `smembers`, which would
pull the whole set over the wire on every dequeue). This module only decides
"is this string plausibly a phone number, and in what canonical form" and
"does this canonical phone appear in an already-fetched set" — both pure,
both unit-testable with no Redis running.
"""

from __future__ import annotations

import re

_DIGITS = re.compile(r"\D")


def normalize_phone_e164(raw: object) -> str | None:
    """Return a dialable +91E.164 string, or None if *raw* isn't a plausible
    Indian phone number. Deliberately strict — a bare 7-9 digit run (an
    account ID, a partial number) is rejected rather than dialled as junk."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    has_plus = s.startswith("+")
    digits = _DIGITS.sub("", s)

    if has_plus:
        # Only accept a +91 number here — this project dials India only.
        return f"+{digits}" if digits.startswith("91") and len(digits) == 12 else None

    # A leading trunk '0' before a 10-digit national number.
    local = digits[1:] if (digits.startswith("0") and len(digits) == 11) else digits
    if len(local) == 10:
        return f"+91{local}"
    # Already has the country code but no '+' (e.g. "919876543210").
    if digits.startswith("91") and len(digits) == 12:
        return f"+{digits}"
    return None


def is_dnd(phone_e164: str, dnd_set: set[str] | frozenset[str]) -> bool:
    """Membership check against an already-fetched DND set — no I/O here.
    Callers fetch the set once (or check with Redis SISMEMBER directly for
    the O(1) live path) rather than passing the whole set per lookup in
    production; this signature exists for the pure/testable half of that."""
    return phone_e164 in dnd_set
