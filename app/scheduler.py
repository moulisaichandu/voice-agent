"""scheduler.py — APScheduler jobs: the time & compliance engine.

Runs in-process, started from app.main's lifespan (see config.SCHEDULER_ENABLED
— only ONE running instance may have this on; APScheduler jobs would otherwise
fire twice, per the blueprint's own "one scheduler, one instance" warning).

  campaign_tick        — pop due leads into the Redis queue, inside calling
                          hours only
  retry_sweeper        — recover crashed-worker items (the reaper) AND
                          requeue business-failed leads still under
                          max_attempts
  dnd_refresh          — sync the Redis DND set from Supabase's leads.dnd flags
  sheets_sync          — pull leads from the Google Sheet into Supabase
  transcript_reconcile — drain sheets:writeback:queue (pushed by
                          app/webhooks/elevenlabs.py), writing each call's
                          outcome back to its lead's Sheet row. This is the
                          QUEUE'S ONLY CONSUMER — no separate always-running
                          worker was built for it (unlike calls:queue's
                          worker.py) since a periodic drain is sufficient at
                          this project's pilot scale. That means write-back
                          latency is bounded by TRANSCRIPT_RECONCILE_MINUTES
                          (default matches the blueprint's own 30-minute
                          interval) — an explicit, documented tradeoff, not
                          an oversight; lower it in .env for faster write-back.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from uuid import UUID

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app import redis_client
from app.compliance.calling_hours import within_calling_hours
from app.config import (
    CALL_MAX_DURATION_S,
    CALLING_HOURS_TZ,
    CAMPAIGN_TICK_SECONDS,
    MAX_CONCURRENT_CALLS,
    PROCESSING_REAPER_TIMEOUT_S,
    RETRY_SWEEP_MINUTES,
    SHEETS_SYNC_MINUTES,
    TRANSCRIPT_RECONCILE_MINUTES,
)
from app.db import leads as leads_db
from app.sheets import client as sheets_client
from app.sheets import sync as sheets_sync_module
from app.sheets import writeback as sheets_writeback
from app.telephony import worker

logger = logging.getLogger(__name__)

# Cap per run so a large backlog doesn't fire an unbounded burst of Sheets API
# calls in one tick — the rest drains on the next interval instead.
_WRITEBACK_BATCH_SIZE = 25

_scheduler: AsyncIOScheduler | None = None

# Sheets sync failures previously only reached a log line — invisible to
# anyone not tailing the container. The admin API's /admin/sheets-status
# reads this key so "did the last sync work, and when" is answerable from the
# console instead of shelling in. Redis is the right home per CLAUDE.md:
# ephemeral, rebuildable, and this is the one place that already writes it.
SHEETS_LAST_SYNC_KEY = "sheets:last_sync"


async def campaign_tick(campaign_id: UUID | None = None) -> int:
    """Pop due leads into the Redis queue — but only inside calling hours.
    Batch size is a simple MAX_CONCURRENT_CALLS per tick (not an exact
    free-slot computation): the worker's own semaphore is what actually gates
    dialing, so over-enqueueing by a small amount just means leads sit
    'queued' briefly rather than anything unsafe. At this project's pilot
    scale (~10-20 concurrent lines) that's a fine tradeoff against the extra
    complexity of computing exact free slots here too.

    Returns the number of leads actually queued (0 outside calling hours, or
    if due_leads() had nothing eligible) — used by the admin API's
    "trigger now" endpoint to report something meaningful to the caller. This
    is deliberately the SAME function the scheduler calls on its own
    interval, not a separate bypass — running it on-demand still goes through
    every compliance gate (calling hours, DND, consent, max_attempts).

    *campaign_id* restricts the tick to one campaign. The scheduler always
    calls this with None (every active campaign); the admin API passes an id
    so that a human clicking "dial now" on one campaign's page cannot start
    calling a different campaign's leads. Scoping only narrows — no gate is
    relaxed for a scoped run."""
    if not within_calling_hours():
        return 0
    due = await leads_db.due_leads(limit=MAX_CONCURRENT_CALLS, campaign_id=campaign_id)
    queued = 0
    for lead in due:
        if await leads_db.mark_queued(lead.lead_id):
            await worker.enqueue_lead(str(lead.lead_id))
            queued += 1
    return queued


async def retry_sweeper() -> None:
    """Two distinct recovery jobs share this one scheduled slot:
      1. Crash recovery — anything still in calls:processing past
         PROCESSING_REAPER_TIMEOUT_S means the worker that popped it died
         before acking; requeue it (see worker.reaper_sweep's docstring).
      2. Business retry — leads whose calls didn't connect (no_answer/failed,
         under their campaign's max_attempts) are already left in a
         non-terminal DB status by the worker; nothing further is needed here
         beyond the reaper today, since due_leads() already re-selects them
         on the next campaign_tick once they're back to 'pending'.
    """
    await worker.reaper_sweep(PROCESSING_REAPER_TIMEOUT_S)
    # 3. Slot recovery — a call whose process died mid-flight never ran either
    #    teardown path, so its concurrency slot was never returned and its lead
    #    is still 'calling'. Bounded by CALL_MAX_DURATION_S plus slack, so a
    #    genuinely live call is never touched.
    await worker.reap_stranded_calls(CALL_MAX_DURATION_S + _STRANDED_SLACK_S)
    # 4. ...and any slot left behind with no lead to attribute it to, which
    #    step 3 structurally cannot see.
    await worker.reconcile_live_slots()


async def dnd_refresh() -> None:
    """Sync the Redis DND set from Supabase. NOTE: this does NOT integrate
    with India's real NCPR/DND registry — no such API was specified anywhere
    in the blueprint, and doing so requires business registration/compliance
    arrangements outside this codebase's scope. This only propagates
    `leads.dnd = true` flags already recorded in Supabase (e.g. from a lead
    explicitly asking not to be called again) into Redis for the worker's
    O(1) dial-time check."""
    phones = await leads_db.dnd_phones()
    r = redis_client.get_redis()
    if phones:
        await r.sadd(worker.DND_SET_KEY, *phones)


async def sheets_sync() -> None:
    r = redis_client.get_redis()
    now_iso = datetime.now(timezone.utc).isoformat()

    reason = sheets_client.unconfigured_reason()
    if reason:
        # A settings gap, not an outage. Logged at WARNING once per sweep with
        # the fix in it, rather than as an ERROR that looks like something
        # broke; the Sheet is a human-friendly mirror, and the app is fully
        # functional without it.
        logger.warning(f"[scheduler] skipping sheets_sync — {reason}")
        record = {"at": now_iso, "synced": None, "error": reason}
        try:
            await r.set(SHEETS_LAST_SYNC_KEY, json.dumps(record))
        except Exception:
            pass
        return

    try:
        synced = await sheets_sync_module.sheets_sync()
        record = {"at": now_iso, "synced": synced, "error": None}
    except Exception as exc:
        # A Sheets outage/misconfiguration must not crash the scheduler loop
        # (which also runs campaign_tick/retry_sweeper/dnd_refresh) — log and
        # let the next interval retry.
        logger.error(f"[scheduler] sheets_sync failed: {type(exc).__name__}: {exc}")
        record = {"at": now_iso, "synced": None, "error": f"{type(exc).__name__}: {exc}"}

    try:
        await r.set(SHEETS_LAST_SYNC_KEY, json.dumps(record))
    except Exception:
        pass  # Redis itself may be what's down; the log line above still landed


# How long past a call's maximum duration before it is certainly over. Slack
# absorbs clock skew and the drain PlivoCall.run() allows for trailing audio.
_STRANDED_SLACK_S = 120

_WRITEBACK_QUEUE = "sheets:writeback:queue"
# Reserve-then-ack processing list, mirroring the calls worker's BLMOVE pattern
# (app/telephony/worker.py). A lead_id is moved here BEFORE its write-back runs
# and only removed after the write-back succeeds or is re-queued, so a crash
# mid-drain leaves it recoverable instead of dropping the Sheet write-back.
_WRITEBACK_PROCESSING = "sheets:writeback:processing"


async def transcript_reconcile() -> None:
    """Drain up to _WRITEBACK_BATCH_SIZE lead_ids from sheets:writeback:queue,
    writing each one's latest call outcome back to their Sheet row.

    Reserve-then-ack, not rpop: the old code popped a whole batch off the queue
    at once (at-most-once), so a crash/restart mid-drain permanently lost those
    Sheet write-backs — the calls worker uses BLMOVE+ack+reaper to avoid exactly
    this. Here each lead_id is reserved onto a processing list before its
    write-back and removed only after success (or an explicit re-queue). A run
    first reclaims anything a previous crashed run stranded in the processing
    list. write_back_lead overwrites the Sheet row idempotently, so an
    at-least-once re-run after a crash-between-write-and-ack is harmless."""
    r = redis_client.get_redis()

    reason = sheets_client.unconfigured_reason()
    if reason:
        # Leave the queue completely alone. Every transcript in it is still
        # wanted; it just cannot be delivered until the Sheet is set up, and
        # draining it into certain failure would mean nine doomed API calls and
        # nine ERROR lines every sweep, forever.
        try:
            waiting = await r.llen(_WRITEBACK_QUEUE)
        except Exception:
            waiting = "?"
        logger.warning(
            f"[scheduler] {waiting} transcript(s) are queued for the Sheet and "
            f"waiting: {reason} Nothing is lost — they will be written as soon "
            "as it is configured."
        )
        return

    # Reclaim items a previous crashed run left mid-flight. Only one instance of
    # this job runs at a time (APScheduler max_instances=1), so nothing else is
    # holding them.
    while await r.lmove(_WRITEBACK_PROCESSING, _WRITEBACK_QUEUE, "LEFT", "RIGHT"):
        pass

    # Reserve a fixed snapshot of up to BATCH ids into the processing list FIRST,
    # so a failure re-queued below is retried on the NEXT run, not re-processed
    # in a hot loop this run.
    reserved: list[str] = []
    for _ in range(_WRITEBACK_BATCH_SIZE):
        lead_id_str = await r.lmove(_WRITEBACK_QUEUE, _WRITEBACK_PROCESSING, "RIGHT", "LEFT")
        if lead_id_str is None:
            break
        reserved.append(lead_id_str)

    ok, failed = 0, 0
    for lead_id_str in reserved:
        try:
            success = await sheets_writeback.write_back_lead(UUID(lead_id_str))
        except Exception as exc:
            logger.error(f"[scheduler] write-back raised for lead {lead_id_str}: "
                         f"{type(exc).__name__}: {exc}")
            success = False
        if success:
            ok += 1
        else:
            failed += 1
            await r.lpush(_WRITEBACK_QUEUE, lead_id_str)  # retry on the next run
        # Ack: remove from the processing list now that it is either done or
        # safely back on the queue.
        await r.lrem(_WRITEBACK_PROCESSING, 1, lead_id_str)

    if ok or failed:
        logger.info(f"[scheduler] transcript_reconcile: {ok} written back, "
                    f"{failed} requeued")


def start() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    tz = pytz.timezone(CALLING_HOURS_TZ)
    sched = AsyncIOScheduler(timezone=tz)
    sched.add_job(campaign_tick, "interval", seconds=CAMPAIGN_TICK_SECONDS, id="campaign_tick")
    sched.add_job(retry_sweeper, "interval", minutes=RETRY_SWEEP_MINUTES, id="retry_sweeper")
    sched.add_job(dnd_refresh, CronTrigger(hour=6, minute=0), id="dnd_refresh")
    sched.add_job(sheets_sync, "interval", minutes=SHEETS_SYNC_MINUTES, id="sheets_sync")
    sched.add_job(transcript_reconcile, "interval", minutes=TRANSCRIPT_RECONCILE_MINUTES,
                  id="transcript_reconcile")
    sched.start()
    _scheduler = sched
    logger.info("[scheduler] started: campaign_tick, retry_sweeper, dnd_refresh, "
                "sheets_sync, transcript_reconcile")
    return sched


def shutdown() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
