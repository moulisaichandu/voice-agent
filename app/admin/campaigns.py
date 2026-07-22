"""admin/campaigns.py — Campaign CRUD for the admin/test API.

Every write goes through the exact same validated paths the rest of the app
uses (create_campaign's AI-disclosure/script checks) — there is no bypass of
any hard rule here. See app/admin/__init__.py for the router assembly and
auth dependency shared by every admin submodule.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app import config as app_config
from app import languages as languages_module
from app.db import campaigns as campaigns_db
from app.db.models import Campaign, CampaignLanguage
from app.telephony.elevenlabs_client import agent_language_support

router = APIRouter()
logger = logging.getLogger(__name__)


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


async def _check_language_support(agent_id: str, language: str) -> None:
    """Move ElevenLabs' language-support check from the first dial (see
    app/telephony/preflight.py's identical gate, which stays the
    authoritative check applied again at dial time) to campaign creation, so
    an operator learns an agent can't speak a language before importing leads
    and recording consent for a campaign that can never ring.

    'auto' sends no override and works on any agent — it must not gain a new
    way to fail, so it returns immediately without an API call.

    Fails OPEN: if the ElevenLabs lookup itself raises (network, auth, an SDK
    shape change), the campaign is still created and a warning is logged.
    This check is a convenience that surfaces a problem earlier; it must
    never be the reason an operator can't create a campaign when the agent is
    actually fine.
    """
    if language == languages_module.AUTO:
        return

    iso = languages_module.iso_code(language)

    # Synchronous SDK call, off the event loop — this route shares its loop
    # with every live audio bridge, exactly like preflight.py's identical
    # call.
    try:
        support = await asyncio.to_thread(agent_language_support, agent_id)
    except Exception as exc:
        logger.warning(
            f"[create_campaign] could not read agent {agent_id}'s language "
            f"support ({type(exc).__name__}: {exc}) — creating the campaign anyway"
        )
        return

    if not support["override_allowed"]:
        raise HTTPException(
            status_code=422,
            detail=(
                f"This campaign is set to '{language}', but agent {agent_id} "
                "does not allow the language override. ElevenLabs rejects an "
                "override for a field that isn't enabled, so every call would "
                "fail. Open the agent's Security tab in the ElevenLabs "
                "dashboard and enable the 'language' override."
            ),
        )
    if support["languages"] and iso not in support["languages"]:
        # Distinguish "you haven't configured it yet" from "it cannot be
        # configured". Sending an operator to look for a Telugu option that
        # ElevenLabs does not offer wastes their time and makes them doubt
        # themselves rather than the tool.
        if not languages_module.elevenlabs_can_speak(iso):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"ElevenLabs Agents does not support {languages_module.display(language)} "
                    "at all — it is not one of the languages the platform offers, "
                    "so no dashboard setting, model change or plan upgrade will "
                    "enable it. Choose a different language for this campaign."
                ),
            )
        raise HTTPException(
            status_code=422,
            detail=(
                f"This campaign is set to '{language}', but agent {agent_id} "
                f"is not configured for it (it has: "
                f"{', '.join(sorted(support['languages'])) or 'none'}). Add "
                "it under Additional Languages in the agent's settings in "
                "the ElevenLabs dashboard, or choose a different language "
                "for this campaign."
            ),
        )


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
    await _check_language_support(agent_id, body.language)
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
