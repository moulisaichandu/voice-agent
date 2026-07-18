"""telephony/preflight.py — Validate the whole chain BEFORE dialling a campaign.

NOT a straight port of ai-voice-agent's calling.preflight(): the failure
surface here is structurally different. There, Plivo answers a call and fetches
our own answer URL — a dead tunnel meant every call rang and dropped instantly.
Here, Exotel talks directly to ElevenLabs, not to this server, so a dead
PUBLIC_BASE_URL does NOT stop a call from connecting — it silently breaks two
things instead: (a) the two-way agent's /rag/search tool calls, so every course
question gets the "no relevant material" fallback, and (b) the transcript
webhook, which degrades to ElevenLabs' own retry-then-give-up rather than
failing loud on our side. Checked here, once per campaign_tick batch, so those
failure modes are caught before a whole campaign runs half-blind rather than
discovered lead-by-lead (or not at all, since neither failure mode stops calls
from connecting).

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

    if not agent_exists(agent_id):
        return (f"ElevenLabs agent_id {agent_id!r} was not found. Check "
                "ELEVENLABS_ONEWAY_AGENT_ID / ELEVENLABS_TWOWAY_AGENT_ID in .env, or "
                "whether the agent was deleted in the ElevenLabs dashboard.")

    if not phone_number_exists(agent_phone_number_id):
        return (f"ElevenLabs agent_phone_number_id {agent_phone_number_id!r} was not "
                "found. Check ELEVENLABS_AGENT_PHONE_NUMBER_ID in .env, or whether the "
                "Exotel number was unlinked in the ElevenLabs dashboard.")

    return None
