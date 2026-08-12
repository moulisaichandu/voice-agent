"""telephony/sarvam_circuit_breaker.py — manual-clear-only breaker for a
Sarvam account that has run out of credits.

Sarvam has no programmatic balance/credits API (only a web dashboard at
dashboard.sarvam.ai/usage), so unlike ElevenLabs' account-status check
(elevenlabs_client.cached_subscription_status) this cannot be proactive: a
call has to actually fail with the "insufficient credits" signal before this
trips. That happened live — an STT stream returned
`{'message': 'Insufficient credits'}` and died, and every one of the active
campaigns used Sarvam, so every scheduled dial from that point would have rung
a real lead's phone and failed within seconds with nothing refusing to try.

See docs/superpowers/specs/2026-08-12-backend-account-health-gate-design.md for
the full design, and for why auto-recovery was deliberately rejected: Sarvam
gives no way to proactively confirm credits have been topped up, so a timer
would either resume into the same failure (too short) or stay paused long
after the account was healthy again (too long). A human confirming "I topped
up" is the only reliable signal.

No TTL on the Redis key, for that reason: this stays tripped until
POST /admin/system/sarvam-circuit-breaker/clear is called.
"""

from __future__ import annotations

from app import redis_client

_BREAKER_KEY = "sarvam:circuit_breaker"


async def trip(reason: str) -> None:
    """Trip the breaker with *reason*, unless it is already tripped.

    NX (set-if-not-exists) makes this first-trip-wins, so the ORIGINAL failure
    reason survives repeated tripping. Several calls typically fail the same
    way in the minutes before anyone notices, and the first one is the
    informative one — a later, vaguer message overwriting it would throw away
    the evidence an operator actually needs.
    """
    r = redis_client.get_redis()
    await r.set(_BREAKER_KEY, reason, nx=True)


async def tripped_reason() -> str | None:
    """The reason the breaker is tripped, or None if it is not."""
    r = redis_client.get_redis()
    return await r.get(_BREAKER_KEY)


async def clear() -> None:
    """The only way out — an admin confirming the account is healthy again.

    Mirrors the manual PATCH /admin/campaigns/{id} {"active": false} that was
    used by hand as the stopgap this replaces: the same "a human explicitly
    says this is fixed now" shape.
    """
    r = redis_client.get_redis()
    await r.delete(_BREAKER_KEY)
