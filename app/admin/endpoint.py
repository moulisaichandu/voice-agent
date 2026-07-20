"""admin/endpoint.py — Minimal admin/test API: campaigns, leads, calls, and a
manual campaign_tick trigger.

Built to support a local test frontend (a way to create a campaign, add
leads, and see calls/transcripts without hand-editing Supabase or waiting on
a real Google Sheet) — NOT a full production admin surface. Every write here
goes through the exact same validated paths the rest of the app uses
(create_campaign's AI-disclosure/script checks, upsert_lead's normal
upsert-on-phone+campaign semantics, campaign_tick's compliance gates) — there
is no bypass of any hard rule anywhere in this module.

Protected by APP_AUTH_TOKEN when set (see require_admin_auth) — open by
default in local dev, matching .env.example's blank default. This makes
APP_AUTH_TOKEN load-bearing for the first time; previously it was read in
app/config.py but only consulted by app.main's CORS-wildcard fail-fast check,
never actually enforced on a request.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from app import scheduler
from app.compliance.dnd import normalize_phone_e164
from app.config import APP_AUTH_TOKEN
from app.db import calls as calls_db
from app.db import campaigns as campaigns_db
from app.db import leads as leads_db
from app.db.models import Call, Campaign, Lead


async def require_admin_auth(authorization: str | None = Header(default=None)) -> None:
    if not APP_AUTH_TOKEN:
        return
    if authorization != f"Bearer {APP_AUTH_TOKEN}":
        raise HTTPException(status_code=401, detail="unauthorized")


router = APIRouter(prefix="/admin", tags=["Admin"],
                   dependencies=[Depends(require_admin_auth)])


class CampaignCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    mode: Literal["oneway", "twoway"]
    agent_id: str = Field(min_length=1)
    script: str | None = None
    max_attempts: int = Field(default=2, ge=1, le=10)


class LeadCreate(BaseModel):
    phone: str = Field(min_length=1, description="Any reasonable Indian phone "
                       "format — normalized server-side to +91E.164")
    name: str | None = None
    language_pref: str = "auto"
    consent_basis: Literal["explicit", "inferred"] | None = None
    consent_at: datetime | None = None


class TickResult(BaseModel):
    queued: int


@router.get("/campaigns", response_model=list[Campaign])
async def list_campaigns(include_inactive: bool = False) -> list[Campaign]:
    """Active campaigns only by default. A dev database that's had the
    integration suite run against it accumulates hundreds of throwaway
    `test-*`/`pipeline-*` campaigns (tests/integration/test_pipeline.py
    deactivates every pre-existing campaign before each test), which buries
    the real ones — so the default is the useful view, not the complete one.
    Pass include_inactive=true for everything."""
    if include_inactive:
        return await campaigns_db.list_campaigns()
    return await campaigns_db.list_active_campaigns()


@router.post("/campaigns", response_model=Campaign, status_code=201)
async def create_campaign(body: CampaignCreate) -> Campaign:
    try:
        return await campaigns_db.create_campaign(
            name=body.name, mode=body.mode, agent_id=body.agent_id,
            script=body.script, max_attempts=body.max_attempts,
        )
    except ValueError as exc:
        # The oneway-needs-a-script and AI-disclosure checks in
        # create_campaign() raise ValueError — surfaced as a 422 the frontend
        # can show inline, not a 500.
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/campaigns/{campaign_id}/leads", response_model=list[Lead])
async def list_leads(campaign_id: UUID) -> list[Lead]:
    if await campaigns_db.get_campaign(campaign_id) is None:
        raise HTTPException(status_code=404, detail="campaign not found")
    return await leads_db.list_leads_for_campaign(campaign_id)


@router.post("/campaigns/{campaign_id}/leads", response_model=Lead, status_code=201)
async def add_lead(campaign_id: UUID, body: LeadCreate) -> Lead:
    if await campaigns_db.get_campaign(campaign_id) is None:
        raise HTTPException(status_code=404, detail="campaign not found")
    phone = normalize_phone_e164(body.phone)
    if not phone:
        raise HTTPException(status_code=400,
                            detail=f"{body.phone!r} is not a valid Indian phone number")
    return await leads_db.upsert_lead(
        sheet_row=None, name=body.name, phone_e164=phone, campaign_id=campaign_id,
        language_pref=body.language_pref,
        consent_basis=body.consent_basis, consent_at=body.consent_at,
    )


@router.get("/campaigns/{campaign_id}/calls", response_model=list[Call])
async def list_calls(campaign_id: UUID) -> list[Call]:
    if await campaigns_db.get_campaign(campaign_id) is None:
        raise HTTPException(status_code=404, detail="campaign not found")
    return await calls_db.list_calls_for_campaign(campaign_id)


@router.post("/trigger-tick", response_model=TickResult)
async def trigger_tick() -> TickResult:
    """Runs campaign_tick() right now instead of waiting for the scheduler's
    next interval (up to CAMPAIGN_TICK_SECONDS away). THIS CAN PLACE REAL
    PHONE CALLS if ELEVENLABS_API_KEY and a real agent/phone number are
    configured and the worker is running (config.WORKER_ENABLED) — it is the
    exact same function the scheduler calls automatically, so every
    compliance gate (calling hours, DND, consent, max_attempts) still
    applies. There is no test-only bypass of any of them here."""
    queued = await scheduler.campaign_tick()
    return TickResult(queued=queued)
