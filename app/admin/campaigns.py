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

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from app import config as app_config
from app import languages as languages_module
from app.db import campaigns as campaigns_db
from app.db.models import Campaign, CampaignLanguage
from app.telephony import sarvam_llm
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


def _resolve_agent_id(
    mode: str, supplied: str | None, language: str = languages_module.AUTO,
) -> str:
    """The agent this campaign dials with. Derived from config so an operator
    doesn't hand-copy an `agent_...` string the server already knows — but
    never guessed: an unset variable is a 422 naming it, not a placeholder."""
    if supplied and supplied.strip():
        return supplied.strip()
    backend = languages_module.backend_for(language)
    if backend != languages_module.ELEVENLABS:
        # Campaign.agent_id is retained as a required database field for
        # compatibility with existing migrations, but these backends do not
        # use an ElevenLabs agent. Store an explicit backend marker instead of
        # rejecting a valid Telugu/OpenAI campaign for a missing EL key.
        return backend
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


def _check_twoway_capable(mode: str, language: str) -> None:
    """Refuse a two-way campaign whose language routes to a backend that
    hasn't been PROVEN on a live call yet.

    The te/tinglish backends (see app/languages.py's backend_for()) are at
    different stages: openai_bridge.py implements two-way fully — turn
    detection, RAG, barge-in truncation — and sarvam_bridge.py does not
    implement it at all yet. Neither has what matters most, which is a human
    having placed a real call and confirmed it behaves correctly. That is what
    each backend's own two-way flag records; languages_module.twoway_enabled()
    owns which flag belongs to which backend, so this function and
    app/telephony/preflight.py cannot drift apart about it.

    Routing a two-way campaign to an unverified conversational path risks the
    exact "campaigns.mode silently produces calls leads couldn't respond to"
    failure CLAUDE.md documents from the sibling project, which is why mode is
    admin-set and never inferred — the same caution applies to a backend that
    has never been heard on a real line.

    Checked at CREATION, where the operator can still choose 'oneway' or a
    different language — not at dial time, when the only options left are
    "let it ring wrong" or "silently downgrade the mode the operator
    explicitly chose" (also not acceptable — CLAUDE.md again).

    Pure logic, no I/O: nothing to fail open on, unlike
    _check_language_support below.
    """
    if mode != "twoway":
        return
    backend = languages_module.backend_for(language)
    if not languages_module.twoway_enabled(backend):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Two-way calling for {languages_module.display(language)} "
                f"is not yet enabled on its voice backend "
                f"({languages_module.backend_display(backend)}) — a live call "
                "has to confirm it before real campaigns use it. One-way "
                "calling IS available for this language today (mode='oneway'), "
                "or choose a different language for a two-way campaign."
            ),
        )


async def _check_language_support(agent_id: str, language: str) -> None:
    """Move ElevenLabs' language-support check from the first dial (see
    app/telephony/preflight.py's identical gate, which stays the
    authoritative check applied again at dial time) to campaign creation, so
    an operator learns an agent can't speak a language before importing leads
    and recording consent for a campaign that can never ring.

    'auto' sends no override and works on any agent — it must not gain a new
    way to fail, so it returns immediately without an API call.

    Only meaningful for ElevenLabs-backed languages. For anything else (today:
    te/tinglish) the ElevenLabs agent is never dialled to carry this call at
    all — app/telephony/call_routes._backend_bridge() sends it to
    openai_bridge instead — so asking THIS agent whether it speaks Telugu
    answers a question that has nothing to do with whether the campaign will
    work, and answers it wrong: no ElevenLabs agent speaks Telugu (see
    app/languages.py's ELEVENLABS_AGENT_LANGUAGES), so this would reject every
    such campaign outright, including the one-way ones the OpenAI backend
    exists to carry.

    Fails OPEN: if the ElevenLabs lookup itself raises (network, auth, an SDK
    shape change), the campaign is still created and a warning is logged.
    This check is a convenience that surfaces a problem earlier; it must
    never be the reason an operator can't create a campaign when the agent is
    actually fine.
    """
    if language == languages_module.AUTO:
        return
    if languages_module.backend_for(language) != languages_module.ELEVENLABS:
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
async def create_campaign(
    body: CampaignCreate, background_tasks: BackgroundTasks,
) -> Campaign:
    _check_twoway_capable(body.mode, body.language)
    agent_id = _resolve_agent_id(body.mode, body.agent_id, body.language)
    await _check_language_support(agent_id, body.language)
    try:
        campaign = await campaigns_db.create_campaign(
            name=body.name, mode=body.mode, agent_id=agent_id,
            script=body.script, max_attempts=body.max_attempts,
            language=body.language,
        )
        # Sarvam rendering is an optional cache warm-up and can take long
        # enough for the console proxy to time out. Respond as soon as the
        # campaign is durable; Starlette runs this task after sending the 201.
        background_tasks.add_task(_warm_rendered_script, campaign)
        return campaign
    except ValueError as exc:
        # The oneway-needs-a-script and AI-disclosure checks in
        # create_campaign() raise ValueError — surfaced as a 422 the frontend
        # can show inline, not a 500.
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def _warm_rendered_script(campaign: Campaign) -> None:
    """Render this campaign's script into Telugu now, so no lead waits for it.

    Rendering measured 14-22s against the live API — Sarvam's chat models are
    reasoning models and cannot be told to hurry. The result is cached per
    campaign, so only the FIRST call of a campaign would pay it, but that lead
    answers the phone and hears up to twenty-two seconds of silence. They hang
    up, and because ONEWAY_MAX_SILENT_S is 30s the watchdog never even notices:
    it is recorded as a delivered call.

    Deliberately best-effort. The campaign is what the operator asked for; the
    warm cache is a nicety. A Sarvam outage at creation time must not cost them
    the campaign, because the dial path still renders on demand — just slowly.
    """
    script = (campaign.script or "").strip()
    if not script:
        # NOTHING to warm. A scriptless two-way call speaks
        # sarvam_prompts.DEFAULT_TWOWAY_TELUGU — a constant — and never calls
        # the renderer at all (see sarvam_bridge). Warming
        # DEFAULT_TWOWAY_SCRIPT anyway cost ~93s of Sarvam reasoning per
        # campaign, the credits for it, and — observed live 2026-08-31,
        # seconds after a campaign was created — a FALSE "[compliance] the
        # rendered Telugu did not disclose" warning, from a render nothing
        # would ever read. That warning has to stay rare enough to believe.
        return
    if languages_module.backend_for(campaign.language) != languages_module.SARVAM:
        # ElevenLabs renders nothing and OpenAI Realtime renders as it speaks.
        # Warming either would pay Sarvam to translate a script no Sarvam call
        # will ever read.
        return
    try:
        await sarvam_llm.render(
            script, language_style=languages_module.style(campaign.language),
        )
    except Exception as exc:  # noqa: BLE001 - never lose a campaign over a cache
        logger.warning(
            f"[sarvam-llm] could not pre-render campaign {campaign.campaign_id}'s "
            f"script ({type(exc).__name__}: {exc}). The campaign is created; the "
            "first call will render on demand and its lead may hear a long "
            "pause before the agent speaks."
        )


@router.patch("/campaigns/{campaign_id}", response_model=Campaign)
async def update_campaign_active(campaign_id: UUID, body: CampaignActiveUpdate) -> Campaign:
    """Activate/deactivate — the reversible, one-click lever. Leads and call
    history are untouched; campaign_tick and due_leads() simply stop
    selecting this campaign's leads while inactive (see db/leads.due_leads)."""
    updated = await campaigns_db.set_campaign_active(campaign_id, body.active)
    if updated is None:
        raise HTTPException(status_code=404, detail="campaign not found")
    return updated
