"""telephony/preflight.py — Validate the whole chain BEFORE dialling a campaign.

NOT a straight port of ai-voice-agent's calling.preflight(): the failure
surface here is structurally different. There, Plivo answers a call directly
and fetches our own answer URL — a dead tunnel meant every call rang and
dropped instantly. Here, the phone provider (Plivo, connected via a SIP trunk)
talks directly to ElevenLabs, not to this server, so a dead PUBLIC_BASE_URL
does NOT stop a call from connecting — it silently breaks two things instead:
(a) the two-way agent's /rag/search tool calls, so every course question gets
the "no relevant material" fallback, and (b) the transcript webhook, which
degrades to ElevenLabs' own retry-then-give-up rather than failing loud on our
side. Checked here, once per campaign_tick batch, so those failure modes are
caught before a whole campaign runs half-blind rather than discovered
lead-by-lead (or not at all, since neither failure mode stops calls from
connecting).

Also checks the specific agent_id/agent_phone_number_id the campaign about to
run will use — so a typo'd or deleted agent fails ONCE, loudly, instead of
identically on every lead in the campaign.
"""

from __future__ import annotations

import httpx

from app.config import ELEVENLABS_API_KEY, PUBLIC_BASE_URL
from app.telephony.elevenlabs_client import agent_exists, phone_number_exists

_HEALTH_CHECK_TIMEOUT_S = 10.0


async def preflight(agent_id: str, agent_phone_number_id: str) -> str | None:
    """Returns None if everything checks out; otherwise a human-readable
    reason not to dial — mirrors ai-voice-agent's calling.preflight() contract
    (None = go, str = don't, with the reason)."""
    if not ELEVENLABS_API_KEY:
        return ("ELEVENLABS_API_KEY is not set, so no call can be placed. Set it "
                "in .env.")

    if PUBLIC_BASE_URL:
        url = f"{PUBLIC_BASE_URL.rstrip('/')}/health"
        try:
            async with httpx.AsyncClient(timeout=_HEALTH_CHECK_TIMEOUT_S) as client:
                resp = await client.get(url)
        except httpx.RequestError as exc:
            return (f"PUBLIC_BASE_URL ({PUBLIC_BASE_URL}) is unreachable "
                    f"({type(exc).__name__}). Two-way calls would get no course "
                    "answers (the /rag/search tool can't be reached) and transcript "
                    "webhooks would degrade to retry-then-fail. Is the tunnel "
                    "(ngrok/cloudflared) running?")
        if resp.status_code != 200:
            return (f"PUBLIC_BASE_URL ({PUBLIC_BASE_URL}) returned HTTP "
                    f"{resp.status_code} instead of 200 at /health. Something other "
                    "than this app is answering, or it's misconfigured.")
    else:
        return ("PUBLIC_BASE_URL is not set. Two-way calls need it for the "
                "/rag/search tool and the transcript webhook to reach this server.")

    # Blank ids are rejected BEFORE the API is consulted. A GET for an empty
    # id degenerates to the collection endpoint (".../phone-numbers/"), which
    # returns 200 — so phone_number_exists("") answers True and an unset
    # ELEVENLABS_AGENT_PHONE_NUMBER_ID sails through the very check meant to
    # catch it. Observed live: the dial then went out and failed at ElevenLabs
    # with "Document with id  not found", burning an attempt per lead for a
    # call that never had any chance of connecting. An unset env var is the
    # most likely misconfiguration there is, so it must fail here, cheaply.
    if not agent_id or not agent_id.strip():
        return ("No agent_id is set for this campaign. Set campaigns.agent_id "
                "(admin/seed) to a real ElevenLabs agent id.")

    if not agent_phone_number_id or not agent_phone_number_id.strip():
        return ("ELEVENLABS_AGENT_PHONE_NUMBER_ID is not set, so there is no "
                "number to dial from. Register your number in ElevenLabs (for a "
                "Plivo number, via a SIP trunk) and put the resulting id in .env.")

    # These two hit the ElevenLabs API, so they can fail for reasons that are
    # neither "exists" nor "doesn't exist" — a revoked key, a rate limit, an
    # outage. This function's contract is None = go, str = don't-dial-because,
    # so those become a reason rather than an exception: letting them escape
    # would abort process_one mid-reserve instead of cleanly declining to
    # dial (see worker.py's docstring point 7 for why that matters).
    try:
        if not agent_exists(agent_id):
            return (f"ElevenLabs agent_id {agent_id!r} was not found. Check "
                    "ELEVENLABS_ONEWAY_AGENT_ID / ELEVENLABS_TWOWAY_AGENT_ID in .env, "
                    "or whether the agent was deleted in the ElevenLabs dashboard. "
                    "(A voice id is not an agent id — they look similar.)")

        if not phone_number_exists(agent_phone_number_id):
            return (f"ElevenLabs agent_phone_number_id {agent_phone_number_id!r} was "
                    "not found. Check ELEVENLABS_AGENT_PHONE_NUMBER_ID in .env, or "
                    "whether the number was unlinked in the ElevenLabs dashboard.")
    except Exception as exc:
        return (f"Could not verify the agent/phone number with ElevenLabs "
                f"({type(exc).__name__}: {exc}). Not dialling while the platform's "
                "state is unknown — a call placed against an unverified agent can't "
                "be reasoned about.")

    return None
