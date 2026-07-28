from uuid import uuid4

"""Tests for the Redis-side primitives in telephony/worker.py — run against
the REAL local Redis (already up, unprofiled in docker-compose — unlike
Postgres, not gated behind --profile dev), because the whole point is
verifying the hand-written Lua scripts are actually atomic/correct, not that
a mock echoes back what I told it to. Not marked @pytest.mark.integration:
Redis is treated as always-available base infrastructure here, consistent
with conftest.py already pointing REDIS_URL at a real local instance.
"""

import pytest

from app import redis_client, scheduler
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


# ── calls stranded by a process that died mid-call ───────────────────────────
#
# _release_slot_once's docstring records this failure being found in
# production: "calls:live:count has no TTL and nothing resets it... at
# MAX_CONCURRENT_CALLS the worker stopped dialling altogether while still
# heartbeating, and only a manual DEL recovered it. Found in production with
# 4 of 10 slots already gone."
#
# Keying the guard per-attempt fixed the RETRY leak. It cannot fix this one:
# if the process dies mid-call — a deploy, a crash, an OOM — neither
# /calls/stream's finally nor /calls/hangup ever runs, so the slot is never
# returned and the lead sits in 'calling' forever. Found live: one lead stuck
# in 'calling' for 22 hours with a slot permanently gone, after a rebuild.
#
# A call cannot outlive CALL_MAX_DURATION_S, so anything older than that plus
# slack is definitionally over.

async def test_a_call_that_cannot_still_be_running_returns_its_slot(monkeypatch):
    released = {"n": 0}
    stranded = [uuid4(), uuid4()]

    async def fake_stale(older_than_s):
        assert older_than_s > 0
        return list(stranded)

    async def fake_mark(lead_id, status):
        assert status == "failed"
        return True  # the guarded transition actually fired

    async def fake_release():
        released["n"] += 1

    monkeypatch.setattr(worker.leads_db, "stale_calling_leads", fake_stale)
    monkeypatch.setattr(worker.leads_db, "mark_result_if_calling", fake_mark)
    monkeypatch.setattr(worker, "release_call_slot", fake_release)

    reaped = await worker.reap_stranded_calls(600)

    assert reaped == 2
    assert released["n"] == 2, "each stranded call holds exactly one slot"


async def test_a_slot_is_returned_only_by_the_run_that_un_stranded_it(monkeypatch):
    """Idempotence, and it comes from the guarded DB transition rather than a
    separate bookkeeping flag: mark_result_if_calling only succeeds for a lead
    still in 'calling', so a second sweep over the same lead releases nothing
    and cannot decrement the counter twice into over-admission."""
    released = {"n": 0}
    lead = uuid4()

    async def fake_stale(older_than_s):
        return [lead]

    async def fake_mark(lead_id, status):
        return False  # somebody else already resolved it

    async def fake_release():
        released["n"] += 1

    monkeypatch.setattr(worker.leads_db, "stale_calling_leads", fake_stale)
    monkeypatch.setattr(worker.leads_db, "mark_result_if_calling", fake_mark)
    monkeypatch.setattr(worker, "release_call_slot", fake_release)

    assert await worker.reap_stranded_calls(600) == 0
    assert released["n"] == 0


async def test_the_window_passed_in_is_the_window_queried(monkeypatch):
    """The worker does not decide how old is too old — the scheduler does, and
    passes it in. This only pins that it is forwarded rather than ignored."""
    seen = {}

    async def fake_stale(older_than_s):
        seen["window"] = older_than_s
        return []

    monkeypatch.setattr(worker.leads_db, "stale_calling_leads", fake_stale)
    await worker.reap_stranded_calls(1234)
    assert seen["window"] == 1234


async def test_the_sweeper_never_reaps_a_call_that_could_still_be_live(monkeypatch):
    """THE safety property, and it belongs to the scheduler because that is
    where the window is chosen. Reaping a live call frees a slot that is
    genuinely in use and marks a lead failed while a real person is still
    talking to it."""
    from app.config import CALL_MAX_DURATION_S
    seen = {}

    async def fake_reap(older_than_s):
        seen["window"] = older_than_s
        return 0

    async def noop(*a, **kw):
        return 0

    monkeypatch.setattr(scheduler.worker, "reap_stranded_calls", fake_reap)
    monkeypatch.setattr(scheduler.worker, "reaper_sweep", noop)

    await scheduler.retry_sweeper()

    assert seen["window"] > CALL_MAX_DURATION_S, (
        "the window must clear a call's maximum duration, with slack"
    )


async def test_the_sweeper_runs_it(monkeypatch):
    """It has to be wired to something that actually ticks, or it is dead code
    and the slots stay gone."""
    called = {"n": 0}

    async def fake_reap(older_than_s):
        called["n"] += 1
        return 0

    async def noop(*a, **kw):
        return 0

    monkeypatch.setattr(scheduler.worker, "reap_stranded_calls", fake_reap)
    monkeypatch.setattr(scheduler.worker, "reaper_sweep", noop)

    await scheduler.retry_sweeper()

    assert called["n"] == 1
