"""telephony/preflight.py — Validate the whole chain BEFORE dialling a campaign.

With Plivo placing the call and streaming the audio here, a reachable
PUBLIC_BASE_URL is now LOAD-BEARING, not merely nice to have. Plivo fetches
the answer URL from the public internet the instant the lead picks up; if it
can't reach us, the lead's phone still RINGS and the call is cut the moment
they answer — with nothing in our logs, because the request never arrived.
That is the exact silent failure the sibling ai-voice-agent project documents,
and it costs real calls to real people.

So this round-trips the REAL answer URL: one request proves the tunnel is up,
this server is up, the webhook token matches, and <Stream> XML is served.

Checked once per campaign_tick batch rather than per lead, so a broken chain
fails once, loudly, instead of identically on every lead.
"""

from __future__ import annotations

import asyncio

import httpx

from app.config import (
    CALL_WEBHOOK_SECRET,
    ELEVENLABS_API_KEY,
    PLIVO_AUTH_ID,
    PLIVO_AUTH_TOKEN,
    PLIVO_FROM_NUMBER,
    PUBLIC_BASE_URL,
)
from app.telephony.elevenlabs_client import agent_exists

_HEALTH_CHECK_TIMEOUT_S = 10.0


async def preflight(agent_id: str) -> str | None:
    """None = safe to dial; a string = the human-readable reason not to.

    Never raises: an exception here would abort process_one mid-reservation
    instead of cleanly declining to dial (see worker.py's docstring point 7).
    """
    if not ELEVENLABS_API_KEY:
        return "ELEVENLABS_API_KEY is not set, so no agent can answer. Set it in .env."

    if not (PLIVO_AUTH_ID and PLIVO_AUTH_TOKEN):
        return ("PLIVO_AUTH_ID / PLIVO_AUTH_TOKEN are not set, so no call can be "
                "placed. These are Plivo's REST API credentials.")

    if not PLIVO_FROM_NUMBER:
        return ("PLIVO_FROM_NUMBER is not set — there is no number to dial from. "
                "Use the E.164 number you own in Plivo, e.g. +918035383564.")

    if not agent_id or not agent_id.strip():
        return ("No agent_id is set for this campaign. Set campaigns.agent_id to a "
                "real ElevenLabs agent id.")

    if not PUBLIC_BASE_URL:
        return ("PUBLIC_BASE_URL is not set, so Plivo has no way to reach this "
                "server. Every call would ring and then drop the instant it was "
                "answered. Set it to the public HTTPS URL of this backend.")

    # Round-trip the real answer URL rather than /health: this validates the
    # webhook token and the XML too, which /health cannot.
    url = (f"{PUBLIC_BASE_URL.rstrip('/')}/calls/answer"
           f"?token={CALL_WEBHOOK_SECRET or ''}&lead=preflight")
    try:
        async with httpx.AsyncClient(timeout=_HEALTH_CHECK_TIMEOUT_S) as client:
            resp = await client.post(url)
    except httpx.RequestError as exc:
        return (f"PUBLIC_BASE_URL ({PUBLIC_BASE_URL}) is unreachable "
                f"({type(exc).__name__}). Plivo could not fetch the answer URL, so "
                "every call would ring and drop on answer. Is the tunnel running?")

    if resp.status_code == 403:
        return ("The answer URL rejected our own token (HTTP 403). "
                "CALL_WEBHOOK_SECRET doesn't match what this server is running "
                "with — recreate the container after changing it. Plivo's calls "
                "would be refused the same way.")
    if resp.status_code != 200:
        hint = (" That's what a dead or offline tunnel returns."
                if resp.status_code in (404, 502, 503, 504) else "")
        return (f"The answer URL returned HTTP {resp.status_code} instead of 200."
                f"{hint} Plivo needs a 200 with <Stream> XML, so calls would drop "
                f"on answer. URL: {PUBLIC_BASE_URL}/calls/answer")
    if "<Stream" not in resp.text:
        return ("The answer URL returned 200 but no <Stream> XML — something other "
                "than this app is answering on that URL (a tunnel error page, or "
                f"another service?). URL: {PUBLIC_BASE_URL}/calls/answer")

    # Cheapest last: this is the only check that costs an ElevenLabs API call.
    #
    # Off the event loop: agent_exists is a SYNCHRONOUS SDK call, and this
    # function runs per dial on the same loop that carries every live audio
    # bridge — so calling it inline froze the audio of every call in progress
    # for a full ElevenLabs round-trip, every time the worker dialled anyone.
    # bridge._signed_url wraps its SDK call for exactly this reason.
    try:
        if not await asyncio.to_thread(agent_exists, agent_id):
            return (f"ElevenLabs agent_id {agent_id!r} was not found. Check "
                    "campaigns.agent_id and ELEVENLABS_{ONEWAY,TWOWAY}_AGENT_ID in "
                    ".env, or whether the agent was deleted. (A voice id is not an "
                    "agent id — they look similar.)")
    except Exception as exc:
        return (f"Could not verify the agent with ElevenLabs ({type(exc).__name__}: "
                f"{exc}). Not dialling while the platform's state is unknown.")

    return None
