"""End-to-end pipeline integration tests — the Phase 4 verification.

Drives a real lead through the WHOLE chain against real Postgres AND real
Redis: campaign_tick -> Redis queue -> worker reserve -> (simulated crash) ->
reaper -> reprocess, asserting no double-dial. Only ElevenLabs itself is
mocked (no real phone calls, obviously); every queue operation, DB
transition, and guard is genuine.

This is the test that proves the crash-safety design actually works, rather
than each piece working in isolation under mocks.
"""

import uuid
from types import SimpleNamespace

import pytest

from app import redis_client, scheduler
from app.db import calls as calls_db
from app.db import campaigns as campaigns_db
from app.db import leads as leads_db
from app.db.pool import get_pool
from app.telephony import worker

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
async def _isolate(request):
    """Redis keys AND cross-test DB state both need clearing.

    campaign_tick's due_leads() is deliberately global (it dials every active
    campaign's pending leads) — so leftover 'pending' leads from an earlier
    test in this persistent dev database would be picked up by a later test's
    campaign_tick and processed INSTEAD of the lead under test. Deactivating
    all pre-existing campaigns before each test uses the real query semantics
    (due_leads joins on c.active = true) to scope each test to its own data.

    Also clears any dialing:* locks — a leaked lock would block a later test's
    lead from dialing for DIALING_LOCK_TTL_S. (Finding that leak in the first
    place is what surfaced the real worker bug this fixture no longer masks.)
    """
    r = redis_client.get_redis()
    keys = [worker.QUEUE_KEY, worker.PROCESSING_KEY, worker.LIVE_COUNT_KEY,
            worker.DND_SET_KEY, "sheets:writeback:queue"]

    async def _clear():
        await r.delete(*keys)
        locks = [k async for k in r.scan_iter(match="dialing:*")]
        if locks:
            await r.delete(*locks)

    await _clear()
    pool = await get_pool()
    await pool.execute("update campaigns set active = false")
    yield
    await _clear()


@pytest.fixture
async def campaign():
    return await campaigns_db.create_campaign(
        name=f"pipeline-{uuid.uuid4().hex[:8]}", mode="twoway",
        agent_id="agent_pipeline_test", max_attempts=3,
    )


@pytest.fixture
async def lead(campaign):
    return await leads_db.upsert_lead(
        sheet_row=2, name="Ravi", phone_e164="+91" + uuid.uuid4().hex[:10],
        campaign_id=campaign.campaign_id,
    )


def _patch_dial_path(monkeypatch):
    """Mock ONLY the external boundaries: ElevenLabs placement + preflight.
    Everything else (Redis, Postgres, all guards) stays real.

    conversation_id must be globally unique per placement: calls has a UNIQUE
    constraint on it and this dev database persists across tests, so a fixed
    prefix would collide between test runs (and it did — the constraint
    correctly caught it).
    """
    async def no_preflight_error(agent_id, phone_id):
        return None

    monkeypatch.setattr(worker.preflight_module, "preflight", no_preflight_error)
    monkeypatch.setattr(worker, "ELEVENLABS_AGENT_PHONE_NUMBER_ID", "phone_test")

    placed = []

    def fake_place(**kwargs):
        placed.append(kwargs)
        return SimpleNamespace(
            success=True, message="ok",
            conversation_id=f"conv_{uuid.uuid4().hex}", call_sid=f"sid_{uuid.uuid4().hex[:8]}",
        )

    monkeypatch.setattr(worker.elevenlabs_client, "place_outbound_call", fake_place)
    return placed


async def _count_calls_for_lead(lead_id) -> int:
    pool = await get_pool()
    return await pool.fetchval("select count(*) from calls where lead_id = $1", lead_id)


# ── happy path ───────────────────────────────────────────────────────────────

async def test_full_pipeline_enqueue_reserve_dial(campaign, lead, monkeypatch):
    placed = _patch_dial_path(monkeypatch)
    monkeypatch.setattr(scheduler, "within_calling_hours", lambda: True)

    await scheduler.campaign_tick()

    # The lead really landed on the real Redis queue and moved to 'queued'.
    r = redis_client.get_redis()
    assert str(lead.lead_id) in await r.lrange(worker.QUEUE_KEY, 0, -1)
    assert (await leads_db.get_lead(lead.lead_id)).status == "queued"

    reserved = await worker._reserve_next(timeout_s=2)
    assert reserved == str(lead.lead_id)
    await worker.process_one(reserved)

    assert len(placed) == 1
    assert placed[0]["dynamic_variables"] == {"lead_id": str(lead.lead_id)}
    refetched = await leads_db.get_lead(lead.lead_id)
    assert refetched.status == "calling"
    assert refetched.attempts == 1
    assert await _count_calls_for_lead(lead.lead_id) == 1
    # Processing list drained — the ack really happened.
    assert await r.lrange(worker.PROCESSING_KEY, 0, -1) == []


# ── the crash-safety scenarios ───────────────────────────────────────────────

async def test_crash_before_reserve_is_recovered_by_reaper_and_dialled_once(
    campaign, lead, monkeypatch,
):
    """Worker popped the lead (BLMOVE succeeded) then died before doing
    anything. The reaper must requeue it, and the retry must dial EXACTLY
    once — the whole point of the BLMOVE+ack design over bare BRPOP."""
    placed = _patch_dial_path(monkeypatch)
    monkeypatch.setattr(scheduler, "within_calling_hours", lambda: True)
    await scheduler.campaign_tick()

    # Reserve, then simulate a hard crash: never process, never ack.
    reserved = await worker._reserve_next(timeout_s=2)
    assert reserved == str(lead.lead_id)
    r = redis_client.get_redis()
    assert await r.lrange(worker.PROCESSING_KEY, 0, -1) == [str(lead.lead_id)]

    # Nothing was dialled, and the lead is still 'queued' (crash was pre-reserve).
    assert placed == []
    assert (await leads_db.get_lead(lead.lead_id)).status == "queued"

    # The reaper recovers it.
    n = await worker.reaper_sweep(timeout_s=0)
    assert n == 1
    assert await r.lrange(worker.QUEUE_KEY, 0, -1) == [str(lead.lead_id)]

    # A healthy worker picks it up and dials — exactly once.
    reserved_again = await worker._reserve_next(timeout_s=2)
    await worker.process_one(reserved_again)

    assert len(placed) == 1
    assert await _count_calls_for_lead(lead.lead_id) == 1
    assert (await leads_db.get_lead(lead.lead_id)).attempts == 1


async def test_crash_after_reserving_the_lead_does_not_double_dial(
    campaign, lead, monkeypatch,
):
    """The nastier case: the worker got as far as mark_calling (and possibly
    placed a call) before dying. On retry, the WHERE status='queued' guard
    must reject the second attempt — otherwise the lead gets called twice."""
    placed = _patch_dial_path(monkeypatch)
    monkeypatch.setattr(scheduler, "within_calling_hours", lambda: True)
    await scheduler.campaign_tick()

    reserved = await worker._reserve_next(timeout_s=2)
    await worker.process_one(reserved)
    assert len(placed) == 1
    assert (await leads_db.get_lead(lead.lead_id)).status == "calling"

    # Simulate the lead somehow being requeued after the call was already
    # placed (e.g. a reaper false positive, or a duplicate enqueue).
    await worker.enqueue_lead(str(lead.lead_id))
    reserved_again = await worker._reserve_next(timeout_s=2)
    await worker.process_one(reserved_again)

    # The reserve-before-act guard rejected it — no second call placed.
    assert len(placed) == 1
    assert await _count_calls_for_lead(lead.lead_id) == 1
    assert (await leads_db.get_lead(lead.lead_id)).attempts == 1


# ── DND + concurrency, end to end ────────────────────────────────────────────

async def test_dnd_scrub_blocks_the_dial_end_to_end(campaign, lead, monkeypatch):
    """dnd_refresh propagates a Supabase dnd flag into the real Redis set,
    and the worker's dial-time SISMEMBER check must then block the call."""
    placed = _patch_dial_path(monkeypatch)
    await leads_db.mark_dnd(lead.lead_id)
    await scheduler.dnd_refresh()

    r = redis_client.get_redis()
    assert await r.sismember(worker.DND_SET_KEY, lead.phone_e164)

    await worker.enqueue_lead(str(lead.lead_id))
    reserved = await worker._reserve_next(timeout_s=2)
    await worker.process_one(reserved)

    assert placed == []
    assert await _count_calls_for_lead(lead.lead_id) == 0


async def test_concurrency_cap_defers_the_second_lead(campaign, monkeypatch):
    """With MAX_CONCURRENT_CALLS=1, a second lead must be requeued rather
    than dialled — and must NOT burn an attempt for a call never placed."""
    placed = _patch_dial_path(monkeypatch)
    monkeypatch.setattr(worker, "MAX_CONCURRENT_CALLS", 1)

    lead_a = await leads_db.upsert_lead(
        sheet_row=2, name="A", phone_e164="+91" + uuid.uuid4().hex[:10],
        campaign_id=campaign.campaign_id,
    )
    lead_b = await leads_db.upsert_lead(
        sheet_row=3, name="B", phone_e164="+91" + uuid.uuid4().hex[:10],
        campaign_id=campaign.campaign_id,
    )
    for lead_x in (lead_a, lead_b):
        await leads_db.mark_queued(lead_x.lead_id)
        await worker.enqueue_lead(str(lead_x.lead_id))

    await worker.process_one(await worker._reserve_next(timeout_s=2))
    await worker.process_one(await worker._reserve_next(timeout_s=2))

    assert len(placed) == 1                      # only one call got through
    b = await leads_db.get_lead(lead_b.lead_id)
    assert b.status == "pending"                  # released, not stuck 'calling'
    assert b.attempts == 0                        # no attempt burned

    # Once the first call ends (webhook releases the slot), B can dial.
    await worker.release_call_slot()
    await leads_db.mark_queued(lead_b.lead_id)
    await worker.process_one(await worker._reserve_next(timeout_s=2))
    assert len(placed) == 2


# ── webhook -> sheets write-back queue, end to end ───────────────────────────

async def test_transcript_webhook_records_and_queues_writeback(campaign, lead, monkeypatch):
    """The real webhook handler against the real DB + Redis: records the
    transcript, releases the concurrency slot, and queues the Sheets
    write-back."""
    from app.webhooks import elevenlabs as wh

    placed = _patch_dial_path(monkeypatch)
    monkeypatch.setattr(scheduler, "within_calling_hours", lambda: True)
    await scheduler.campaign_tick()
    await worker.process_one(await worker._reserve_next(timeout_s=2))
    assert len(placed) == 1

    call = await calls_db.get_latest_call_for_lead(lead.lead_id)
    conversation_id = call.el_conversation_id

    await wh._handle_post_call_transcription({
        "conversation_id": conversation_id,
        "status": "done",
        "transcript": [
            {"role": "agent", "message": "నమస్కారం, Digital Brolly నుండి."},
            {"role": "user", "message": "chెప్పండి"},
        ],
        "analysis": {"transcript_summary": "Interested."},
    })

    updated = await calls_db.get_call_by_conversation_id(conversation_id)
    assert updated.status == "done"
    assert updated.turns == 2
    assert updated.transcript[0].role == "agent"
    assert updated.transcript[1].role == "lead"   # ElevenLabs "user" -> our "lead"
    assert updated.summary == "Interested."

    r = redis_client.get_redis()
    assert await r.lrange("sheets:writeback:queue", 0, -1) == [str(lead.lead_id)]

    # A retried webhook must not double-record or double-queue.
    await wh._handle_post_call_transcription({
        "conversation_id": conversation_id, "status": "done",
        "transcript": [], "analysis": {},
    })
    assert await r.lrange("sheets:writeback:queue", 0, -1) == [str(lead.lead_id)]
    still = await calls_db.get_call_by_conversation_id(conversation_id)
    assert still.turns == 2                       # not overwritten by the retry
