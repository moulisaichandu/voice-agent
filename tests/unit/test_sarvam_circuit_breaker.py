"""Unit tests for telephony/sarvam_circuit_breaker.py — run against the REAL
local Redis, same convention as test_worker_redis_primitives.py: the point
is verifying the actual trip/clear semantics, not that a mock echoes back
what it was told.

Needs `docker compose --profile dev up -d redis` (or any local Redis on
localhost:6379); tests/conftest.py points every test at DB index 15.
"""

from app import redis_client
from app.telephony import sarvam_circuit_breaker as breaker


async def _clean():
    r = redis_client.get_redis()
    await r.delete(breaker._BREAKER_KEY)


async def test_a_fresh_breaker_is_not_tripped():
    await _clean()
    assert await breaker.tripped_reason() is None


async def test_trip_then_tripped_reason_round_trip():
    await _clean()
    await breaker.trip("Insufficient credits")
    assert await breaker.tripped_reason() == "Insufficient credits"


async def test_a_second_trip_does_not_overwrite_the_first_reason():
    """First-trip-wins: several calls failing the same way in the minutes
    before anyone notices must not lose the ORIGINAL reason to a later,
    possibly less-informative one."""
    await _clean()
    await breaker.trip("Insufficient credits")
    await breaker.trip("a different later reason")
    assert await breaker.tripped_reason() == "Insufficient credits"


async def test_clear_removes_the_trip():
    await _clean()
    await breaker.trip("Insufficient credits")
    await breaker.clear()
    assert await breaker.tripped_reason() is None


async def test_clear_on_an_already_clear_breaker_is_a_noop():
    await _clean()
    await breaker.clear()  # must not raise
    assert await breaker.tripped_reason() is None


async def test_the_breaker_has_no_expiry():
    """Manual-clear-only is the design decision this key encodes, not an
    oversight — Sarvam offers no way to confirm a top-up programmatically, so
    a TTL would either resume into the same failure (too short) or stay paused
    long after the account recovered (too long). A TTL appearing here later
    would silently reintroduce exactly that."""
    await _clean()
    await breaker.trip("Insufficient credits")
    r = redis_client.get_redis()
    # -1 is redis for "key exists, no expiry set"; -2 means it is gone.
    assert await r.ttl(breaker._BREAKER_KEY) == -1


# ── recognising a credits failure, whatever Sarvam calls it today ────────────
#
# The first version matched the literal string "insufficient credit", taken
# from the one wording observed when the breaker was designed. A live call on
# 2026-08-13 produced two DIFFERENT wordings from the same account state, and
# the breaker fired on neither:
#
#   TTS error frame : "No credits available."            (code 402)
#   STT close frame : "Credits exhausted. Visit the API Dashboard..."  (1003)
#
# The design doc names this exact risk — "soft coupling to Sarvam's exact
# wording... fails open, not closed". It failed open on the first real
# exhaustion after shipping.

import pytest


@pytest.mark.parametrize("message", [
    "Insufficient credits",                                    # original wording
    "No credits available.",                                   # live, TTS 402
    "Credits exhausted. Visit the API Dashboard to review "
    "and manage your subscription.",                           # live, STT 1003
    "INSUFFICIENT CREDIT",                                     # case
    "Your account has run out of credit",
])
def test_every_observed_credits_wording_is_recognised(message):
    assert breaker.looks_like_credits_exhausted(message)


@pytest.mark.parametrize("message", [
    "bad language code",
    "invalid speaker",
    "Error in Pipeline : 1 validation error for SarvamAppRequest",
    "",
    None,
])
def test_unrelated_failures_are_not_credits(message):
    """Still scoped. Tripping on any failure would halt every Sarvam campaign
    over a transient blip — a call that would have succeeded on retry."""
    assert not breaker.looks_like_credits_exhausted(message)


def test_a_non_string_is_not_credits():
    """The callers fall back to the whole event dict when a frame carries no
    message, so this must not raise on one."""
    assert not breaker.looks_like_credits_exhausted({"type": "error"})


def test_a_credits_message_under_an_unexpected_key_is_still_recognised():
    """Fail-closed against the OTHER half of this outage. The callers fall
    back to the whole event dict when there is no 'message' key — and this
    code has already lost a credits signal exactly that way once, reading
    data['error'] while Sarvam sent data['message'].

    Searching the stringified frame means the next rename costs an ugly log
    line rather than a breaker that silently never fires."""
    assert breaker.looks_like_credits_exhausted(
        {"type": "error", "data": {"detail": "Credits exhausted."}}
    )


def test_a_dict_with_no_credits_wording_is_not_credits():
    """The other side of that: searching the whole frame must not become
    "any error frame trips it"."""
    assert not breaker.looks_like_credits_exhausted(
        {"type": "error", "data": {"detail": "bad language code"}}
    )
