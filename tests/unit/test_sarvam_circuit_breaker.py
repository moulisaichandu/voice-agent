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
