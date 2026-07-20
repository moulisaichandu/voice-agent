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

import logging
from uuid import UUID

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app import redis_client
from app.compliance.calling_hours import within_calling_hours
from app.config import (
    CALLING_HOURS_TZ,
    CAMPAIGN_TICK_SECONDS,
    MAX_CONCURRENT_CALLS,
    PROCESSING_REAPER_TIMEOUT_S,
    RETRY_SWEEP_MINUTES,
    SHEETS_SYNC_MINUTES,
    TRANSCRIPT_RECONCILE_MINUTES,
)
from app.db import leads as leads_db
from app.sheets import sync as sheets_sync_module
from app.sheets import writeback as sheets_writeback
from app.telephony import worker

logger = logging.getLogger(__name__)

# Cap per run so a large backlog doesn't fire an unbounded burst of Sheets API
# calls in one tick — the rest drains on the next interval instead.
_WRITEBACK_BATCH_SIZE = 25

_scheduler: AsyncIOScheduler | None = None


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
    try:
        await sheets_sync_module.sheets_sync()
    except Exception as exc:
        # A Sheets outage/misconfiguration must not crash the scheduler loop
        # (which also runs campaign_tick/retry_sweeper/dnd_refresh) — log and
        # let the next interval retry.
        logger.error(f"[scheduler] sheets_sync failed: {type(exc).__name__}: {exc}")


async def transcript_reconcile() -> None:
    """Drains up to _WRITEBACK_BATCH_SIZE lead_ids from sheets:writeback:queue,
    writing each one's latest call outcome back to their Sheet row. A failed
    write (row/phone unresolvable, Sheets API error) re-queues the lead_id for
    the next run rather than dropping it."""
    r = redis_client.get_redis()
    raw = await r.rpop("sheets:writeback:queue", _WRITEBACK_BATCH_SIZE)
    if not raw:
        return
    lead_ids = raw if isinstance(raw, list) else [raw]

    ok, failed = 0, 0
    for lead_id_str in lead_ids:
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
            await r.lpush("sheets:writeback:queue", lead_id_str)

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
