"""admin/campaigns.py — Campaign CRUD for the admin/test API.

Every write goes through the exact same validated paths the rest of the app
uses (create_campaign's AI-disclosure/script checks) — there is no bypass of
any hard rule here. See app/admin/__init__.py for the router assembly and
auth dependency shared by every admin submodule.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app import config as app_config
from app.db import campaigns as campaigns_db
from app.db.models import Campaign, CampaignLanguage

router = APIRouter()


class CampaignCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    mode: Literal["oneway", "twoway"] = Field(
        default="twoway",
        description="Still ADMIN-SET at creation time, never inferred per call "
                    "(CLAUDE.md hard rule) — this default just makes the safer "
                    "of the two the one you get without asking.",
    )
    agent_id: str | None = Field(
        default=None,
        description="Omit to use the agent configured for this mode in .env "
                    "(ELEVENLABS_TWOWAY_AGENT_ID / ELEVENLABS_ONEWAY_AGENT_ID). "
                    "An explicit value still wins.",
    )
    script: str | None = None
    language: CampaignLanguage = Field(
        default="auto",
        description="The language this campaign's calls run in. 'auto' sends "
                    "no override and uses whatever the ElevenLabs agent is "
                    "configured with — the behaviour of every campaign created "
                    "before this field existed. A lead's own language column "
                    "still overrides this per row.",
    )
    max_attempts: int = Field(default=2, ge=1, le=10)


# Which env var backs which mode. Never cross-fall-back: dialling a one-way
# campaign with the two-way agent would put a lead in a conversation the
# campaign was never designed for, and vice versa.
_AGENT_ID_BY_MODE = {
    "oneway": "ELEVENLABS_ONEWAY_AGENT_ID",
    "twoway": "ELEVENLABS_TWOWAY_AGENT_ID",
}


def _resolve_agent_id(mode: str, supplied: str | None) -> str:
    """The agent this campaign dials with. Derived from config so an operator
    doesn't hand-copy an `agent_...` string the server already knows — but
    never guessed: an unset variable is a 422 naming it, not a placeholder."""
    if supplied and supplied.strip():
        return supplied.strip()
    var_name = _AGENT_ID_BY_MODE[mode]
    configured = getattr(app_config, var_name, None)
    if not configured:
        raise HTTPException(
            status_code=422,
            detail=f"No agent_id given and {var_name} is not set in .env, so there "
                   f"is no agent to dial {mode} campaigns with. Set {var_name}, or "
                   f"pass agent_id explicitly.",
        )
    return configured


class CampaignActiveUpdate(BaseModel):
    active: bool


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
    agent_id = _resolve_agent_id(body.mode, body.agent_id)
    try:
        return await campaigns_db.create_campaign(
            name=body.name, mode=body.mode, agent_id=agent_id,
            script=body.script, max_attempts=body.max_attempts,
            language=body.language,
        )
    except ValueError as exc:
        # The oneway-needs-a-script and AI-disclosure checks in
        # create_campaign() raise ValueError — surfaced as a 422 the frontend
        # can show inline, not a 500.
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.patch("/campaigns/{campaign_id}", response_model=Campaign)
async def update_campaign_active(campaign_id: UUID, body: CampaignActiveUpdate) -> Campaign:
    """Activate/deactivate — the reversible, one-click lever. Leads and call
    history are untouched; campaign_tick and due_leads() simply stop
    selecting this campaign's leads while inactive (see db/leads.due_leads)."""
    updated = await campaigns_db.set_campaign_active(campaign_id, body.active)
    if updated is None:
        raise HTTPException(status_code=404, detail="campaign not found")
    return updated
