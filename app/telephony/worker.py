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
     whole worker loop silently. Fixed: process_one() catches placement
     failures specifically and turns them into a lead status update.

  7. process_one() had `try/finally` with no `except`, while claiming here
     that "any failure is turned into a lead status update". It wasn't: an
     error anywhere OUTSIDE the placement call (get_lead, sismember,
     mark_calling, preflight, create_call) propagated — and the `finally`
     acked anyway, deleting the lead from calls:processing. That left it
     status='calling', in no queue and in no processing list, so the reaper
     couldn't see it and due_leads() (status='pending' only) never
     re-selected it: silently un-callable forever, the exact loss BLMOVE+ack
     exists to prevent. Fixed: an outer `except` undoes whatever the partial
     attempt did (slot, dialing lock, reservation) and, if recovery itself
     fails, deliberately does NOT ack — leaving the entry for the reaper.
     A call that was actually placed is never released back to 'pending';
     the webhook owns it from that point.

  8. run_worker()'s loop body was unguarded. As a background asyncio task
     (app/main.py's lifespan), an escaping exception killed the coroutine
     while the process kept serving health checks — nothing dialled again
     until a human noticed and restarted. Fixed: per-iteration try/except
     with exponential backoff, and CancelledError re-raised so shutdown
     still works.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

from app import languages, redis_client
from app.compliance.calling_hours import within_calling_hours
from app.config import DIALING_LOCK_TTL_S, MAX_CONCURRENT_CALLS
from app.db import calls as calls_db
from app.db import campaigns as campaigns_db
from app.db import leads as leads_db
from app.telephony import plivo_client
from app.telephony import preflight as preflight_module

logger = logging.getLogger(__name__)

QUEUE_KEY = "calls:queue"
PROCESSING_KEY = "calls:processing"
LIVE_COUNT_KEY = "calls:live:count"
DND_SET_KEY = "dnd:numbers"

# Written once per loop iteration (~every 2s, the BLMOVE timeout below) — the
# admin readiness check's only way to know the worker is actually alive, not
# just configured. Earlier this session the worker died silently while
# /health kept returning 200; WORKER_ENABLED alone would just repeat the
# config back, which is exactly what lied then. TTL is generous versus the
# write cadence so one slow iteration doesn't flap readiness to critical.
HEARTBEAT_KEY = "worker:heartbeat"
HEARTBEAT_TTL_S = 30

# How long this attempt's Plivo CallUUID stays readable. It only has to
# outlive the call itself, since call_routes' slot-release guard reads it when
# the stream ends and when Plivo posts the hangup. Matches
# call_routes._SLOT_GUARD_TTL_S deliberately — the two expire together.
CALL_UUID_TTL_S = 3600

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


async def add_to_dnd_set(phone_e164: str) -> None:
    """Put a number into the dial-time DND set NOW, rather than waiting for
    scheduler.dnd_refresh's 06:00 cron.

    process_one checks this set with SISMEMBER on every dequeue, and it was
    the only thing catching a person who opted out on one campaign while still
    being a dial-eligible row on another. Since the set was written solely by
    a once-a-day job, an opt-out recorded at 10:15 did not reach the scrub
    until 06:00 the NEXT morning — a full day of calling someone who had
    already asked you to stop. Writing it here closes that window to zero."""
    r = redis_client.get_redis()
    await r.sadd(DND_SET_KEY, phone_e164)


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


async def reap_orphaned_queued_leads() -> int:
    """Return leads that are 'queued' in Postgres but present in neither Redis list.

    campaign_tick reserves a lead as 'queued' and then relies solely on a
    calls:queue entry to carry it forward. Nothing anywhere moved 'queued' back
    to 'pending'; due_leads() selects only 'pending'; reaper_sweep inspects only
    calls:processing and reap_stranded_calls only 'calling'. So losing the
    Redis list — a FLUSHALL (which CLAUDE.md documents as safe), a recreated
    container without its volume, or maxmemory eviction — stranded every queued
    lead permanently.

    Membership, not a timestamp, is the test: BLMOVE moves an id from the queue
    to the processing list atomically, so a live lead is always in exactly one
    of them. Anything in neither has no carrier left.

    The one race is a lead between mark_queued and its lpush, which this could
    reset to 'pending'; the lpush then still happens and the worker's own
    mark_calling guard rejects it harmlessly, and the lead is re-enqueued on the
    next tick. Recoverable, unlike the state this repairs.
    """
    r = redis_client.get_redis()
    carried = set(await r.lrange(QUEUE_KEY, 0, -1)) | set(
        await r.lrange(PROCESSING_KEY, 0, -1))
    n = 0
    for lead_id in await leads_db.queued_lead_ids():
        if lead_id in carried:
            continue
        if await leads_db.release_orphaned_queued_lead(lead_id):
            n += 1
    if n:
        logger.warning(
            f"[worker] returned {n} lead(s) to 'pending' that were reserved as "
            "'queued' but had no entry in either Redis list — the queue was "
            "flushed or lost while they were waiting."
        )
    return n


# Lower calls:live:count to a target, atomically, and never raise it. Done in
# Lua so the read and the write cannot interleave with a worker acquiring a
# slot between them — which would otherwise clobber a legitimate acquisition.
_RECONCILE_SLOTS_LUA = """
local cur = tonumber(redis.call('GET', KEYS[1]) or '0')
local target = tonumber(ARGV[1])
if cur > target then
  redis.call('SET', KEYS[1], target)
  return cur - target
end
return 0
"""


async def reconcile_live_slots() -> int:
    """Bring calls:live:count back down to the number of calls that exist.

    The companion to reap_stranded_calls, which recovers a slot by resolving
    the lead still holding it. That cannot help when the lead was already
    resolved by another path and only the counter was left behind: the slot is
    then orphaned, attributable to nothing, and permanent. Observed live
    immediately after shipping the reaper — 0 leads calling, 0 stale,
    calls:live:count = 1.

    Only ever LOWERS. Raising it would invent occupancy and throttle dialling
    for a reason that does not exist, which is a worse failure than the one
    this is fixing and reached by a job meant to be a safety net.

    The DB count is read before the Lua runs, so a call starting in that window
    could see the counter lowered by one more than it should be. That is
    bounded, self-correcting on the next release, and vastly preferable to
    capacity that is lost forever — the failure _release_slot_once's docstring
    records reaching 4 of 10 slots in production.
    """
    live = await leads_db.count_calling_leads()
    r = redis_client.get_redis()
    recovered = int(await r.eval(_RECONCILE_SLOTS_LUA, 1, LIVE_COUNT_KEY, live) or 0)
    if recovered:
        logger.warning(
            f"[worker] recovered {recovered} orphaned concurrency slot(s): "
            f"calls:live:count claimed more calls than the {live} actually in "
            "flight. Left alone, each one permanently reduces how many leads "
            "can be dialled at once."
        )
    return recovered


async def reap_stranded_calls(older_than_s: int) -> int:
    """Return the concurrency slots of calls whose process died mid-call.

    _release_slot_once keys its dedupe guard per ATTEMPT, which fixed slots
    leaking across RETRIES. It cannot fix this: if the process itself dies
    mid-call, neither /calls/stream's finally nor /calls/hangup ever runs, so
    the slot is never returned at all. calls:live:count has no TTL and nothing
    else resets it, so every crashed call costs a permanent slot — the failure
    _release_slot_once's docstring records being found in production with 4 of
    10 already gone, recoverable only by a manual DEL.

    Idempotent by construction, with no extra bookkeeping: the slot is released
    only when mark_result_if_calling actually transitions the lead, and that is
    guarded on the lead still being 'calling'. A second sweep over the same
    lead transitions nothing and therefore releases nothing, so repeated runs
    can never decrement the counter into over-admission.

    'failed' is the honest outcome — the call really did fail — and it lets the
    ordinary retry path pick the lead up again, which is what should happen to
    someone whose call was cut short by a deploy.
    """
    stranded = await leads_db.stale_calling_leads(older_than_s)
    reaped = 0
    for lead_id in stranded:
        try:
            if await leads_db.mark_result_if_calling(lead_id, "failed"):
                await release_call_slot()
                reaped += 1
        except Exception as exc:  # noqa: BLE001 - one bad row must not stop the sweep
            logger.error(f"[worker] could not reap stranded call for lead "
                         f"{lead_id}: {type(exc).__name__}: {exc}")
    if reaped:
        logger.warning(
            f"[worker] reaped {reaped} call(s) stranded by a process that died "
            "mid-call: their concurrency slots are back and their leads are "
            "eligible for retry."
        )
    return reaped


async def _reserve_next(timeout_s: int = 2) -> str | None:
    r = redis_client.get_redis()
    return await r.blmove(QUEUE_KEY, PROCESSING_KEY, timeout=timeout_s, src="RIGHT", dest="LEFT")


async def _ack(lead_id: str) -> None:
    r = redis_client.get_redis()
    await r.lrem(PROCESSING_KEY, 1, lead_id)


async def process_one(lead_id: str) -> None:
    """The full reserve -> dial -> record sequence for one lead_id already
    popped into calls:processing. See module docstring points 6 and 7."""
    # Tracked so the unhandled-failure path (point 7) knows exactly what has
    # to be undone: a partially-completed attempt must not leave the lead
    # reserved, the number locked, or a concurrency slot held.
    reserved = False
    lock_key: str | None = None
    slot_held = False
    call_placed = False
    safe_to_ack = True

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
            await leads_db.mark_result(lead.lead_id, "pending")
            return
        campaign = await campaigns_db.get_campaign(lead.campaign_id)
        if campaign is None or not campaign.active:
            logger.warning(
                f"[worker] lead {lead_id}: campaign missing/inactive — returning "
                "to pending rather than stranding it in queued"
            )
            await leads_db.mark_result(lead.lead_id, "pending")
            return

        # Calling hours, re-checked HERE and not only in campaign_tick.
        #
        # CLAUDE.md makes the 10:00-19:00 IST window a hard rule, but the only
        # check used to be at enqueue time in scheduler.campaign_tick — and
        # queued leads outlive that check. calls:queue is a Redis list with AOF
        # persistence on a named volume, and the backend has
        # restart: unless-stopped, so a batch queued at 18:58 survives a crash
        # and gets dialled the moment the process comes back, whatever the
        # hour. retry_sweeper's reaper (which runs 24/7) can requeue a stuck
        # lead into the same situation. Both dial real people in the middle of
        # the night with nothing to stop them.
        #
        # Back to 'pending' rather than returning with the lead left 'queued':
        # nothing else ever transitions 'queued' -> 'pending', so leaving it
        # would strand the lead permanently (due_leads selects 'pending'
        # only). No attempt is burned — record_dial_attempt hasn't run.
        if not within_calling_hours():
            logger.warning(
                f"[worker] lead {lead_id}: outside calling hours at dial time — "
                "returning it to 'pending' instead of dialling"
            )
            await leads_db.mark_result(lead.lead_id, "pending")
            return

        # Reserve-before-act (module docstring point 2).
        if not await leads_db.mark_calling(lead.lead_id):
            # Usually a benign race: another worker moved this lead on. But it
            # is ALSO how a crashed reservation looks — a lead left 'calling'
            # with no attempt recorded, which the reaper requeues and this
            # guard then rejects, acking it away into no queue and no
            # processing list. release_undialed_calling_lead can tell the two
            # apart (see its docstring) and only ever frees the crashed one.
            if await leads_db.release_undialed_calling_lead(lead.lead_id):
                logger.warning(
                    f"[worker] lead {lead_id} was stuck in 'calling' with no "
                    "dial attempt recorded — a reservation whose process died. "
                    "Released back to 'pending'."
                )
            else:
                logger.info(f"[worker] lead {lead_id} already left 'queued' — skipping")
            return
        reserved = True

        # In-flight per-number lock (module docstring point 4). Any early
        # return past this point that releases the reservation must put the
        # lead back to 'pending' — it never actually got attempted, so
        # attempts must NOT have been incremented (record_dial_attempt()
        # hasn't run yet at any of these points).
        if not await r.set(f"dialing:{lead.phone_e164}", "1", nx=True, ex=DIALING_LOCK_TTL_S):
            logger.warning(f"[worker] lead {lead_id}: already dialing {lead.phone_e164}")
            await leads_db.mark_result(lead.lead_id, "pending")
            reserved = False
            return
        lock_key = f"dialing:{lead.phone_e164}"

        # for_call() is the single shared resolution app/telephony/
        # call_routes.py's _call_language() also uses — see its docstring
        # for why passing campaign.language (a catalogue token, not an ISO
        # code) straight to preflight would silently refuse every dial for
        # a code-mixed ("tinglish"/"hinglish") campaign.
        call_iso_code, _ = languages.for_call(lead.language_pref, campaign.language)
        reachability_error = await preflight_module.preflight(
            campaign.agent_id, call_iso_code, mode=campaign.mode,
        )
        if reachability_error:
            logger.error(f"[worker] preflight failed, not dialing: {reachability_error}")
            # Release the dialing lock: it exists to stop the same number
            # being dialled twice concurrently, and nothing was dialled here.
            # Leaving it set would block this lead from retrying for the full
            # DIALING_LOCK_TTL_S even though no call was ever placed.
            await r.delete(lock_key)
            lock_key = None
            await leads_db.mark_result(lead.lead_id, "pending")
            reserved = False
            return

        if not await try_acquire_call_slot():
            # At capacity — put the lead back and let a later tick retry
            # rather than busy-loop (module docstring point 5). Same as
            # above: no call placed, so the dialing lock must not linger.
            await r.delete(lock_key)
            lock_key = None
            await leads_db.mark_result(lead.lead_id, "pending")
            reserved = False
            await enqueue_lead(lead_id)
            await asyncio.sleep(1)
            return

        # From here on, a slot is held — every remaining exit path must
        # either transfer ownership of that slot to the webhook handler
        # (successful placement) or release it itself (any failure).
        slot_held = True
        await leads_db.record_dial_attempt(lead.lead_id)
        # Plivo dials; it then streams the call audio to /calls/stream, which
        # bridges it to the ElevenLabs agent (app/telephony/bridge.py). The
        # agent's dynamic variables — including a one-way campaign's script —
        # are supplied there, at bridge time, because that is where the
        # conversation actually starts.
        try:
            result = await plivo_client.place_outbound_call(
                to_number=lead.phone_e164,
                lead_id=str(lead.lead_id),
            )
        except Exception as exc:
            # No call connected, so the dialing lock must not linger either —
            # a retry (this lead is under max_attempts) would otherwise be
            # blocked for DIALING_LOCK_TTL_S for no reason.
            await release_call_slot()
            slot_held = False
            await r.delete(lock_key)
            lock_key = None
            logger.error(f"[worker] call placement raised for {lead_id}: "
                         f"{type(exc).__name__}: {exc}")
            await leads_db.mark_result(lead.lead_id, "failed")
            reserved = False
            return

        if not result.success:
            await release_call_slot()
            slot_held = False
            await r.delete(lock_key)
            lock_key = None
            logger.error(f"[worker] call placement unsuccessful for {lead_id}: "
                         f"{result.message}")
            await leads_db.mark_result(lead.lead_id, "failed")
            reserved = False
            return

        # A real call is now ringing. Past this line the lead must NEVER be
        # released back to 'pending' by the failure path below — that would
        # let a later tick dial someone who is already on the phone.
        call_placed = True

        # Record THIS attempt's provider call id, so the slot-release guard in
        # call_routes can tell attempt 2 apart from attempt 1. Keyed writes
        # here rather than relying solely on /calls/answer because this is the
        # same code path that acquired the slot, so it is authoritative even
        # if Plivo's answer callback never arrives or its form fails to parse.
        # Best-effort: a Redis blip here must not fail a call that is already
        # ringing, and the guard degrades to lead_id scope without it.
        # Falls back to a locally generated id when Plivo returns 200 with no
        # request_uuid (plivo_client sets success=True with call_uuid=None for
        # a body that fails to parse). Without one, the release guard degrades
        # to lead_id scope, which cannot tell attempt 2 from attempt 1 and
        # silently skips the second release — the slot leak this key exists to
        # prevent. Any per-attempt-unique value does that job; it only has to
        # be the SAME one for both teardown paths, which it is, because they
        # both read it from here.
        attempt_id = result.call_uuid or f"attempt-{uuid4().hex}"
        try:
            await r.set(f"call:uuid:{lead_id}", attempt_id, ex=CALL_UUID_TTL_S)
        except Exception:
            logger.warning(f"[worker] could not record call uuid for {lead_id}")

        # el_conversation_id is NULL here on purpose: with Plivo placing the
        # call there is no ElevenLabs conversation yet — one is created when
        # the audio bridge connects, and call_routes.py fills it in then.
        await calls_db.create_call(
            lead_id=lead.lead_id, campaign_id=campaign.campaign_id, mode=campaign.mode,
            el_conversation_id=None, provider_call_id=result.call_uuid,
        )
        # The slot is now owned by the call routes, released when the call
        # actually ends (stream closes, or Plivo posts the hangup) — not here.

    except Exception:
        # Module docstring point 7. Without this, ANY unhandled error (a
        # Postgres blip in get_lead/create_call, a Redis drop in sismember)
        # propagated straight through the `finally` below — which still
        # acked, deleting the lead from calls:processing. The lead was left
        # status='calling', in no queue and in no processing list, so the
        # reaper could not see it and due_leads() (status='pending' only)
        # never re-selected it: silently un-callable forever. That is exactly
        # the loss BLMOVE+ack exists to prevent.
        logger.exception(f"[worker] unhandled error processing lead {lead_id}")
        try:
            if call_placed:
                # The call is live. Leave the reservation and the slot alone —
                # the webhook owns both from here. Releasing either would risk
                # a double-dial or an over-admitting semaphore.
                logger.error(
                    f"[worker] lead {lead_id}: call was already placed when this "
                    "failed — leaving it 'calling' for the transcript webhook to "
                    "resolve. If create_call() is what failed, the calls row is "
                    "missing and this call's transcript cannot be matched back."
                )
            else:
                if slot_held:
                    await release_call_slot()
                if lock_key:
                    await redis_client.get_redis().delete(lock_key)
                if reserved:
                    # Back to 'pending', NOT 'failed': nothing was dialled, so
                    # this must stay eligible for the next campaign_tick and
                    # must not burn an attempt (see db/leads.py).
                    await leads_db.mark_result(UUID(lead_id), "pending")
        except Exception:
            # Recovery itself failed — the backing store is likely down. Do
            # NOT ack: leaving the entry in calls:processing is precisely the
            # case the reaper was built for.
            logger.exception(
                f"[worker] recovery failed for lead {lead_id} — leaving it in "
                f"{PROCESSING_KEY} for the reaper rather than acking it away"
            )
            safe_to_ack = False
    finally:
        if safe_to_ack:
            await _ack(lead_id)


_MAX_LOOP_BACKOFF_S = 30


async def run_worker(stop_event: asyncio.Event) -> None:
    """The worker loop: BLMOVE, process, repeat, until *stop_event* is set.
    A short BLMOVE timeout (2s) keeps the stop check responsive without
    busy-polling. Started as a background asyncio task from app.main's
    lifespan by default (see config.WORKER_ENABLED) — see __main__ below for
    running it as a standalone process instead.

    The loop body is guarded (module docstring point 8): as a background
    task, an escaping exception would kill this coroutine while the rest of
    the process — health checks included — carried on looking healthy, and
    nothing would ever dial again until someone noticed and restarted. A
    transient Redis/Postgres blip must cost a retry, not the dialer.
    """
    logger.info("[worker] started")
    consecutive_failures = 0
    while not stop_event.is_set():
        try:
            await redis_client.get_redis().set(
                HEARTBEAT_KEY, datetime.now(timezone.utc).isoformat(), ex=HEARTBEAT_TTL_S,
            )
            lead_id = await _reserve_next(timeout_s=2)
            if lead_id is not None:
                await process_one(lead_id)
            consecutive_failures = 0
        except asyncio.CancelledError:
            # Real shutdown, not a fault — never swallow it into the backoff.
            raise
        except Exception:
            consecutive_failures += 1
            backoff = min(2 ** consecutive_failures, _MAX_LOOP_BACKOFF_S)
            logger.exception(
                f"[worker] loop iteration failed ({consecutive_failures} in a row) "
                f"— retrying in {backoff}s"
            )
            # Back off so a sustained outage doesn't spin the CPU or flood the
            # log, while a one-off blip still recovers within seconds.
            await asyncio.sleep(backoff)
    logger.info("[worker] stopped")


if __name__ == "__main__":  # pragma: no cover - manual/standalone entrypoint
    # For running the worker as its own process/container instead of
    # in-process with the API — see the commented-out `worker` service in
    # docker-compose.yml. Set WORKER_ENABLED=false on the `backend` service
    # if you do this, to avoid two redundant (harmless, just wasteful)
    # consumers of calls:queue.
    import signal

    async def _standalone() -> None:
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except NotImplementedError:
                pass  # Windows: no add_signal_handler for SIGTERM; Ctrl+C still works
        await run_worker(stop_event)

    asyncio.run(_standalone())
