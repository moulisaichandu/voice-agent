"""Tests for the Redis-side primitives in telephony/worker.py — run against
the REAL local Redis (already up, unprofiled in docker-compose — unlike
Postgres, not gated behind --profile dev), because the whole point is
verifying the hand-written Lua scripts are actually atomic/correct, not that
a mock echoes back what I told it to. Not marked @pytest.mark.integration:
Redis is treated as always-available base infrastructure here, consistent
with conftest.py already pointing REDIS_URL at a real local instance.
"""

import pytest

from app import redis_client
from app.telephony import worker


@pytest.fixture(autouse=True)
async def _clean_keys():
    """Every test in this file uses the real, fixed key names — clean up
    before AND after so tests can't see each other's leftover state."""
    r = redis_client.get_redis()
    keys = [worker.QUEUE_KEY, worker.PROCESSING_KEY, worker.LIVE_COUNT_KEY]
    await r.delete(*keys)
    yield
    await r.delete(*keys)


# ── concurrency semaphore ───────────────────────────────────────────────────

async def test_acquire_slot_succeeds_under_capacity(monkeypatch):
    monkeypatch.setattr(worker, "MAX_CONCURRENT_CALLS", 2)
    assert await worker.try_acquire_call_slot() is True
    assert await worker.try_acquire_call_slot() is True


async def test_acquire_slot_fails_at_capacity(monkeypatch):
    monkeypatch.setattr(worker, "MAX_CONCURRENT_CALLS", 1)
    assert await worker.try_acquire_call_slot() is True
    assert await worker.try_acquire_call_slot() is False  # at cap


async def test_release_frees_a_slot_for_reacquisition(monkeypatch):
    monkeypatch.setattr(worker, "MAX_CONCURRENT_CALLS", 1)
    assert await worker.try_acquire_call_slot() is True
    assert await worker.try_acquire_call_slot() is False
    await worker.release_call_slot()
    assert await worker.try_acquire_call_slot() is True  # slot freed


async def test_release_never_drives_the_counter_negative():
    """A double-release (a bug, or a retried webhook not caught by its own
    idempotency key) must not let the semaphore over-admit indefinitely."""
    r = redis_client.get_redis()
    await worker.release_call_slot()
    await worker.release_call_slot()
    await worker.release_call_slot()
    count = await r.get(worker.LIVE_COUNT_KEY)
    assert count in (None, "0", 0)  # floored, never negative


# ── queue mechanics: enqueue / reserve / ack ────────────────────────────────

async def test_enqueue_reserve_ack_round_trip():
    await worker.enqueue_lead("lead-1")
    lead_id = await worker._reserve_next(timeout_s=1)
    assert lead_id == "lead-1"

    # Popped item sits in the processing list until acked.
    r = redis_client.get_redis()
    assert await r.lrange(worker.PROCESSING_KEY, 0, -1) == ["lead-1"]

    await worker._ack("lead-1")
    assert await r.lrange(worker.PROCESSING_KEY, 0, -1) == []


async def test_reserve_times_out_with_no_work():
    lead_id = await worker._reserve_next(timeout_s=1)
    assert lead_id is None


async def test_reserve_is_fifo():
    await worker.enqueue_lead("first")
    await worker.enqueue_lead("second")
    assert await worker._reserve_next(timeout_s=1) == "first"
    assert await worker._reserve_next(timeout_s=1) == "second"


# ── reaper ───────────────────────────────────────────────────────────────────

async def test_reaper_requeues_stuck_processing_items():
    """Simulates a worker that popped an item (BLMOVE succeeded, so it's in
    calls:processing) and then crashed before acking — the reaper must put it
    back on the queue so a healthy worker picks it up next."""
    r = redis_client.get_redis()
    await r.lpush(worker.PROCESSING_KEY, "stuck-lead")

    n = await worker.reaper_sweep(timeout_s=0)

    assert n == 1
    assert await r.lrange(worker.PROCESSING_KEY, 0, -1) == []
    assert await r.lrange(worker.QUEUE_KEY, 0, -1) == ["stuck-lead"]


async def test_reaper_is_a_noop_when_nothing_is_stuck():
    n = await worker.reaper_sweep(timeout_s=0)
    assert n == 0
