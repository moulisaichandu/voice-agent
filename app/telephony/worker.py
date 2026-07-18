"""telephony/worker.py — The Redis-queue call worker.

Fixes several distinct safety issues in the blueprint's own worker sketch
(§5.1), which was unsafe as literally written:

  1. BRPOP is at-most-once with no ack — a crash between popping a lead_id and
     durably recording it lost the lead with zero trace it was ever attempted.
     Fixed with BLMOVE into a `calls:processing` list; a reaper (scheduler.py)
     requeues anything stuck there past PROCESSING_REAPER_TIMEOUT_S.

  2. No reserve-before-act ordering — the lead must transition to `calling`
     (guarded by WHERE status='queued', see db/leads.py) BEFORE the call is
     placed, not after, so a crash mid-placement can't get double-dialled by
     a later retry.

  3. `phone in await r.smembers(...)` pulls the WHOLE DND set over the wire on
     every dequeue. Fixed: `SISMEMBER`, O(1).

  4. The rate-limit key was set only AFTER a successful call, leaving a window
     where a duplicate enqueue could double-dial before the guard existed.
     Fixed: a `dialing:<phone>` NX lock acquired BEFORE placing the call.
     NOTE: that lock is explicitly DELETED on every path that bails without
     actually placing a call (preflight failure, at-capacity, placement
     error). It exists to stop the same number being dialled twice
     concurrently — if no call went out, holding it just blocks a legitimate
     retry for DIALING_LOCK_TTL_S. Only a genuinely in-flight call leaves it
     set, to expire on its own.

  5. Busy-loop `rpush(...); sleep(2)` on concurrency-full. Fixed: a Redis
     counting semaphore (INCR/DECR via an atomic Lua script), acquired here
     and released by the webhook handler once the call actually ENDS (not
     when it's merely placed — "placed" != "finished", and the slot must
     stay held for the call's real duration to enforce MAX_CONCURRENT_CALLS).

  6. No exception handling around call placement — one API error killed the
     whole worker loop silently. Fixed: process_one() wraps the full
     reserve -> dial -> record sequence; any failure is turned into a lead
     status update, and the `finally` block always acks the processing-list
     entry so only a genuine crash (not a handled failure) leaves work for
     the reaper.
"""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from app import redis_client
from app.config import (
    DIALING_LOCK_TTL_S,
    ELEVENLABS_AGENT_PHONE_NUMBER_ID,
    MAX_CONCURRENT_CALLS,
)
from app.db import calls as calls_db
from app.db import campaigns as campaigns_db
from app.db import leads as leads_db
from app.telephony import elevenlabs_client
from app.telephony import preflight as preflight_module

logger = logging.getLogger(__name__)

QUEUE_KEY = "calls:queue"
PROCESSING_KEY = "calls:processing"
LIVE_COUNT_KEY = "calls:live:count"
DND_SET_KEY = "dnd:numbers"

_ACQUIRE_SLOT_LUA = """
local count = tonumber(redis.call('GET', KEYS[1]) or '0')
if count >= tonumber(ARGV[1]) then
  return 0
end
redis.call('INCR', KEYS[1])
return 1
"""

# Floors at 0 so a double-release (a bug, or a retried webhook that isn't
# caught by its own idempotency key for some reason) can't drive the count
# negative, which would let the semaphore over-admit indefinitely.
_RELEASE_SLOT_LUA = """
local n = tonumber(redis.call('GET', KEYS[1]) or '0')
if n > 0 then redis.call('DECR', KEYS[1]) end
return 1
"""


async def try_acquire_call_slot() -> bool:
    r = redis_client.get_redis()
    result = await r.eval(_ACQUIRE_SLOT_LUA, 1, LIVE_COUNT_KEY, MAX_CONCURRENT_CALLS)
    return bool(result)


async def release_call_slot() -> None:
    """Called by app/webhooks/elevenlabs.py once a call's outcome is recorded
    (post_call_transcription or call_initiation_failure) — that is the actual
    end of the call's lifetime, not the moment it was placed."""
    r = redis_client.get_redis()
    await r.eval(_RELEASE_SLOT_LUA, 1, LIVE_COUNT_KEY)


async def enqueue_lead(lead_id: str) -> None:
    """Called by scheduler.py's campaign_tick for each due lead."""
    r = redis_client.get_redis()
    await r.lpush(QUEUE_KEY, lead_id)


async def reaper_sweep(timeout_s: int) -> int:
    """Requeue anything that's been in calls:processing without being acked
    for longer than *timeout_s* — evidence the worker that popped it crashed
    before handling it. Returns the number of items requeued. Deliberately
    simple (no per-item timestamps) for the pilot's scale: it requeues the
    WHOLE processing list's current contents on the assumption a healthy
    worker acks within seconds, so anything still there when the sweeper runs
    (on a multi-minute interval — see config.RETRY_SWEEP_MINUTES) is stuck."""
    r = redis_client.get_redis()
    n = 0
    while True:
        lead_id = await r.rpoplpush(PROCESSING_KEY, QUEUE_KEY)
        if lead_id is None:
            break
        n += 1
    if n:
        logger.warning(f"[worker] reaper requeued {n} stuck lead(s) from {PROCESSING_KEY}")
    return n


async def _reserve_next(timeout_s: int = 2) -> str | None:
    r = redis_client.get_redis()
    return await r.blmove(QUEUE_KEY, PROCESSING_KEY, timeout=timeout_s, src="RIGHT", dest="LEFT")


async def _ack(lead_id: str) -> None:
    r = redis_client.get_redis()
    await r.lrem(PROCESSING_KEY, 1, lead_id)


async def process_one(lead_id: str) -> None:
    """The full reserve -> dial -> record sequence for one lead_id already
    popped into calls:processing. Always acks in `finally` — see module
    docstring point 6."""
    try:
        lead = await leads_db.get_lead(UUID(lead_id))
        if lead is None:
            logger.warning(f"[worker] lead {lead_id} not found — dropping")
            return

        r = redis_client.get_redis()
        if await r.sismember(DND_SET_KEY, lead.phone_e164):
            await leads_db.mark_dnd(lead.lead_id)
            return

        if lead.campaign_id is None:
            logger.warning(f"[worker] lead {lead_id} has no campaign — skipping")
            return
        campaign = await campaigns_db.get_campaign(lead.campaign_id)
        if campaign is None or not campaign.active:
            logger.warning(f"[worker] lead {lead_id}: campaign missing/inactive — skipping")
            return

        # Reserve-before-act (module docstring point 2).
        if not await leads_db.mark_calling(lead.lead_id):
            logger.info(f"[worker] lead {lead_id} already left 'queued' — skipping")
            return

        # In-flight per-number lock (module docstring point 4). Any early
        # return past this point that releases the reservation must put the
        # lead back to 'pending' — it never actually got attempted, so
        # attempts must NOT have been incremented (record_dial_attempt()
        # hasn't run yet at any of these points).
        lock_key = f"dialing:{lead.phone_e164}"
        if not await r.set(lock_key, "1", nx=True, ex=DIALING_LOCK_TTL_S):
            logger.warning(f"[worker] lead {lead_id}: already dialing {lead.phone_e164}")
            await leads_db.mark_result(lead.lead_id, "pending")
            return

        reachability_error = await preflight_module.preflight(
            campaign.agent_id, ELEVENLABS_AGENT_PHONE_NUMBER_ID or "",
        )
        if reachability_error:
            logger.error(f"[worker] preflight failed, not dialing: {reachability_error}")
            # Release the dialing lock: it exists to stop the same number
            # being dialled twice concurrently, and nothing was dialled here.
            # Leaving it set would block this lead from retrying for the full
            # DIALING_LOCK_TTL_S even though no call was ever placed.
            await r.delete(lock_key)
            await leads_db.mark_result(lead.lead_id, "pending")
            return

        if not await try_acquire_call_slot():
            # At capacity — put the lead back and let a later tick retry
            # rather than busy-loop (module docstring point 5). Same as
            # above: no call placed, so the dialing lock must not linger.
            await r.delete(lock_key)
            await leads_db.mark_result(lead.lead_id, "pending")
            await enqueue_lead(lead_id)
            await asyncio.sleep(1)
            return

        # From here on, a slot is held — every remaining exit path must
        # either transfer ownership of that slot to the webhook handler
        # (successful placement) or release it itself (any failure).
        await leads_db.record_dial_attempt(lead.lead_id)
        try:
            result = elevenlabs_client.place_outbound_call(
                agent_id=campaign.agent_id,
                agent_phone_number_id=ELEVENLABS_AGENT_PHONE_NUMBER_ID or "",
                to_number=lead.phone_e164,
                dynamic_variables={"lead_id": str(lead.lead_id)},
            )
        except Exception as exc:
            # No call connected, so the dialing lock must not linger either —
            # a retry (this lead is under max_attempts) would otherwise be
            # blocked for DIALING_LOCK_TTL_S for no reason.
            await release_call_slot()
            await r.delete(lock_key)
            logger.error(f"[worker] call placement raised for {lead_id}: "
                         f"{type(exc).__name__}: {exc}")
            await leads_db.mark_result(lead.lead_id, "failed")
            return

        if not result.success or not result.conversation_id:
            await release_call_slot()
            await r.delete(lock_key)
            logger.error(f"[worker] call placement unsuccessful for {lead_id}: "
                         f"{result.message}")
            await leads_db.mark_result(lead.lead_id, "failed")
            return

        await calls_db.create_call(
            lead_id=lead.lead_id, campaign_id=campaign.campaign_id, mode=campaign.mode,
            el_conversation_id=result.conversation_id, provider_call_id=result.call_sid,
        )
        # The slot is now owned by the webhook handler, released when the
        # call's real outcome is recorded — not here.
    finally:
        await _ack(lead_id)


async def run_worker(stop_event: asyncio.Event) -> None:
    """The worker loop: BLMOVE, process, repeat, until *stop_event* is set.
    A short BLMOVE timeout (2s) keeps the stop check responsive without
    busy-polling."""
    logger.info("[worker] started")
    while not stop_event.is_set():
        lead_id = await _reserve_next(timeout_s=2)
        if lead_id is None:
            continue
        await process_one(lead_id)
    logger.info("[worker] stopped")
