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

# Sarvam says "out of credits" in at least three different ways, all observed
# from the same account state:
#
#   "Insufficient credits"                          (error frame, at design time)
#   "No credits available."                         (TTS error frame, code 402)
#   "Credits exhausted. Visit the API Dashboard..." (STT close frame, code 1003)
#
# The first version of this matched the literal string "insufficient credit"
# and fired on NONE of the last two — it failed open on the first real
# exhaustion after shipping, which is precisely the risk the design doc named.
# Matching the word "credit" is wording-independent without being a blanket
# "any error trips it": Sarvam does not use the word for anything else, and a
# blanket net would halt every campaign over a transient blip.
_CREDITS_MARKER = "credit"


def looks_like_credits_exhausted(message: object) -> bool:
    """Whether *message* is Sarvam saying the account has no credits left.

    Takes `object`, not `str`, because the callers fall back to the WHOLE
    event dict when a frame carries no 'message' key — and that fallback is
    searched too, rather than dismissed as "not a string".

    That is deliberate, and it is the second lesson from the same outage. This
    code has already lost a credits signal once to a key rename: it read
    data['error'] while Sarvam sent data['message'], so the check ran on a
    dict and could never fire. Searching the stringified frame means the next
    rename costs a scruffy log line, not a silent failure to stop dialling.

    The cost is a Sarvam error frame that mentions credit without being an
    exhaustion — a quota figure, say — refusing to dial. That is a fair price:
    an error frame is already a failed call, and this errs toward the failure
    an admin can see and clear rather than the one that rings real leads.
    """
    return _CREDITS_MARKER in str(message).lower()


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
