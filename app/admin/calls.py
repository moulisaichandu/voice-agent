"""admin/calls.py — Call history and the manual campaign_tick trigger.

See app/admin/__init__.py for the router assembly and auth dependency shared
by every admin submodule.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app import scheduler
from app.db import calls as calls_db
from app.db import campaigns as campaigns_db
from app.db.models import Call

router = APIRouter()


class TickResult(BaseModel):
    queued: int


@router.get("/campaigns/{campaign_id}/calls", response_model=list[Call])
async def list_calls(campaign_id: UUID) -> list[Call]:
    if await campaigns_db.get_campaign(campaign_id) is None:
        raise HTTPException(status_code=404, detail="campaign not found")
    return await calls_db.list_calls_for_campaign(campaign_id)


@router.get("/calls", response_model=list[Call])
async def recent_calls(limit: int = Query(default=50, ge=1, le=200)) -> list[Call]:
    """Most recent calls across every campaign — the call-history page. Not
    scoped to active campaigns: a call that already happened is history
    regardless of whether its campaign is still active."""
    return await calls_db.list_recent_calls(limit=limit)


@router.post("/campaigns/{campaign_id}/trigger-tick", response_model=TickResult)
async def trigger_tick_for_campaign(campaign_id: UUID) -> TickResult:
    """Dial-eligible leads for THIS campaign only.

    Scoped deliberately. The unscoped variant below exists too, but a human
    clicking "dial now" from one campaign's page must not start calling a
    different campaign's leads — with real credentials that is real calls to
    people the operator did not intend to contact, and there is no recalling
    a placed call. The scope only narrows the selection; every compliance
    gate still applies (see scheduler.campaign_tick)."""
    if await campaigns_db.get_campaign(campaign_id) is None:
        raise HTTPException(status_code=404, detail="campaign not found")
    queued = await scheduler.campaign_tick(campaign_id=campaign_id)
    return TickResult(queued=queued)


@router.post("/trigger-tick", response_model=TickResult)
async def trigger_tick() -> TickResult:
    """Runs campaign_tick() across EVERY active campaign, right now, instead
    of waiting for the scheduler's next interval (up to CAMPAIGN_TICK_SECONDS
    away). THIS CAN PLACE REAL PHONE CALLS if ELEVENLABS_API_KEY and a real
    agent/phone number are configured and the worker is running
    (config.WORKER_ENABLED) — it is the exact same function the scheduler
    calls automatically, so every compliance gate (calling hours, DND,
    consent, max_attempts) still applies. There is no test-only bypass here.

    Prefer the per-campaign route above for anything a human triggers; this
    one is for exercising the scheduler's real, unscoped behaviour."""
    queued = await scheduler.campaign_tick()
    return TickResult(queued=queued)
