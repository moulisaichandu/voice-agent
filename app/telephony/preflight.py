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
import logging

import httpx

from app import languages as languages_module
from app.config import (
    CALL_WEBHOOK_SECRET,
    CONVERSATION_LLM_PROVIDER,
    ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE_ID,
    OPENAI_API_KEY,
    PLIVO_AUTH_ID,
    PLIVO_AUTH_TOKEN,
    PLIVO_FROM_NUMBER,
    SARVAM_API_KEY,
)
from app.telephony import conversation_llm, public_url, sarvam_circuit_breaker
from app.telephony.elevenlabs_client import (
    GOOD_SUBSCRIPTION_STATUSES,
    agent_exists,
    agent_language_support,
    cached_subscription_status,
)

_HEALTH_CHECK_TIMEOUT_S = 10.0

logger = logging.getLogger(__name__)


async def preflight(
    agent_id: str, language: str | None = None, mode: str | None = None,
) -> str | None:
    """None = safe to dial; a string = the human-readable reason not to.

    Never raises: an exception here would abort process_one mid-reservation
    instead of cleanly declining to dial (see worker.py's docstring point 7).

    *mode* defaults to None so every existing caller (and every test written
    before this parameter existed) keeps working unchanged — None never
    equals 'twoway', so the guard below simply never fires for them.
    """
    # Refuse a two-way call on a backend that hasn't been PROVEN on a live
    # call yet — unconditional, checked first, and before anything else can
    # fail open. This is pure local logic with no I/O, unlike every check
    # below it, so there is no outage to fail open on.
    #
    # app/telephony/openai_bridge.py fully implements two-way conversation
    # (turn detection, RAG, barge-in truncation). OPENAI_TWOWAY_ENABLED
    # (app/config.py) is the operator's confirmation that a human has placed
    # a real call on it and judged it correct — the same discipline CLAUDE.md
    # already applies to AI disclosure and to campaigns.mode never being
    # inferred: a lead-facing behaviour this consequential doesn't go live on
    # code review alone. app/admin/campaigns.py's _check_twoway_capable()
    # already refuses this combination at CREATION time, but that cannot
    # catch a campaign that already existed before the flag was flipped
    # either way, so this dial-time copy is still load-bearing.
    #
    # 'auto' (language=None) always resolves to ELEVENLABS (see
    # languages.backend_for_iso), so a two-way 'auto' campaign — every
    # campaign created before this feature existed — can never trip this.
    twoway_backend = languages_module.backend_for_iso(language)
    if mode == "twoway" and not languages_module.twoway_enabled(twoway_backend):
        # The flag and the backend's NAME both come from languages_module
        # rather than being written here. This message used to hardcode
        # "OpenAI Realtime" and read OPENAI_TWOWAY_ENABLED directly, which
        # silently became wrong the moment Telugu moved to Sarvam: an operator
        # would be told to check a backend their campaign does not use.
        display = languages_module.backend_display(twoway_backend)
        flag = f"{twoway_backend.split('_')[0].upper()}_TWOWAY_ENABLED"
        return (
            f"This campaign is 'twoway' in '{language}'. That backend "
            f"({display}) has not been confirmed for two-way calling — "
            f"{flag} is not set, and a live call needs to confirm it works "
            "before real campaigns use it. One-way calling IS available for "
            f"'{language}' today. Change this campaign's "
            "mode to 'oneway', or choose a different language for a two-way "
            "campaign."
        )

    # Only the backend that actually carries this call needs ElevenLabs. The
    # previous unconditional check made every Sarvam Telugu campaign fail
    # when ElevenLabs was intentionally not configured.
    if (languages_module.backend_for_iso(language) == languages_module.ELEVENLABS
            and not ELEVENLABS_API_KEY):
        return "ELEVENLABS_API_KEY is not set, so no agent can answer. Set it in .env."

    if not (PLIVO_AUTH_ID and PLIVO_AUTH_TOKEN):
        return ("PLIVO_AUTH_ID / PLIVO_AUTH_TOKEN are not set, so no call can be "
                "placed. These are Plivo's REST API credentials.")

    if not PLIVO_FROM_NUMBER:
        return ("PLIVO_FROM_NUMBER is not set — there is no number to dial from. "
                "Use the E.164 number you own in Plivo, e.g. +918035383564.")

    if (languages_module.backend_for_iso(language) == languages_module.ELEVENLABS
            and (not agent_id or not agent_id.strip())):
        return ("No agent_id is set for this campaign. Set campaigns.agent_id to a "
                "real ElevenLabs agent id.")

    # Re-resolve the tunnel FIRST. This runs once per campaign_tick batch,
    # immediately before any dialling, which is the one moment being current
    # actually matters: a quick tunnel mints a new hostname whenever cloudflared
    # restarts, and nothing else re-resolves for the life of the process. A
    # stale hostname here means every lead's phone rings and the call drops on
    # answer, with nothing in our logs. See app/telephony/public_url.py.
    base_url = await public_url.refresh()

    if not base_url:
        return ("PUBLIC_BASE_URL is not set, so Plivo has no way to reach this "
                "server. Every call would ring and then drop the instant it was "
                "answered. Set it to the public HTTPS URL of this backend.")

    # Round-trip the real answer URL rather than /health: this validates the
    # webhook token and the XML too, which /health cannot.
    url = (f"{base_url.rstrip('/')}/calls/answer"
           f"?token={CALL_WEBHOOK_SECRET or ''}&lead=preflight")
    try:
        async with httpx.AsyncClient(timeout=_HEALTH_CHECK_TIMEOUT_S) as client:
            resp = await client.post(url)
    except httpx.RequestError as exc:
        return (f"PUBLIC_BASE_URL ({base_url}) is unreachable "
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
                f"on answer. URL: {base_url}/calls/answer")
    if "<Stream" not in resp.text:
        return ("The answer URL returned 200 but no <Stream> XML — something other "
                "than this app is answering on that URL (a tunnel error page, or "
                f"another service?). URL: {base_url}/calls/answer")

    # Validate the provider-specific prerequisites before touching ElevenLabs.
    # Sarvam and OpenAI campaigns do not have an ElevenLabs agent at all.
    call_backend = languages_module.backend_for_iso(language)
    if call_backend == languages_module.SARVAM:
        if not SARVAM_API_KEY:
            return (
                f"This campaign dials in '{language}', which runs on the "
                "Sarvam backend, but SARVAM_API_KEY is not set. Get one from "
                "https://dashboard.sarvam.ai and set it in .env. To fall back "
                "to the previous backend instead, set "
                "TELUGU_BACKEND=openai_realtime."
            )
        breaker_reason = await sarvam_circuit_breaker.tripped_reason()
        if breaker_reason:
            return (
                f"Sarvam reported {breaker_reason!r} on a recent call, and "
                "dialling on this backend has been paused rather than risk "
                "repeating it on every other lead. Check credits at "
                "dashboard.sarvam.ai/usage, top up if needed, then clear it: "
                "POST /admin/system/sarvam-circuit-breaker/clear."
            )
        if mode == "twoway" and CONVERSATION_LLM_PROVIDER == conversation_llm.SARVAM:
            return (
                f"This campaign dials in '{language}' as a two-way call. "
                "CONVERSATION_LLM_PROVIDER is set to 'sarvam', but Sarvam's "
                "chat models are too slow for a live conversational turn. Set "
                "CONVERSATION_LLM_PROVIDER=openai in .env and restart."
            )
        if mode == "twoway":
            missing = conversation_llm.missing_key_name()
            if missing:
                return (
                    f"This campaign dials in '{language}' as a two-way call, "
                    f"but {missing} is not set. Set {missing} in .env and restart."
                )
        return None

    if call_backend == languages_module.OPENAI_REALTIME:
        if not OPENAI_API_KEY:
            return (
                f"This campaign dials in '{language}', which runs on the "
                "OpenAI Realtime backend, but OPENAI_API_KEY is not set. "
                "Set it in .env."
            )
        return None

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

    # The agent's own configuration, last: only reached when the chain is
    # otherwise sound. Both the language and the forced voice are answered by
    # ONE read of the agent, so they share it.
    #
    # Which questions are worth asking depends entirely on which backend will
    # actually carry this call. An OpenAI-backed language has no ElevenLabs
    # agent, no language preset and no override switch — running the ElevenLabs
    # checks below against it would refuse the campaign for a reason unrelated
    # to whether it can dial, and a forced ElevenLabs voice is irrelevant to a
    # call openai_bridge.py carries. See app/languages.py's backend_for_iso()
    # for why this needs the ISO lookup rather than backend_for()'s token
    # lookup. Checked OUTSIDE the `if language:` below because a forced voice
    # must not drag an OpenAI-backed call into the ElevenLabs branch;
    # backend_for_iso(None) is ELEVENLABS, so an 'auto' campaign never enters.
    # ElevenLabs billing applies to every call this backend carries, whatever
    # the language or voice override — so this runs BEFORE the
    # 'auto'-with-no-override early return just below. Placing it after would
    # skip billing for exactly the commonest campaign shape there is, which is
    # the one most likely to be dialling when an invoice goes unpaid.
    #
    # Fails closed, like agent_exists() above: a check that cannot answer is
    # not evidence the account is fine.
    try:
        sub = await cached_subscription_status()
    except Exception as exc:
        return (f"Could not check ElevenLabs' billing status "
                f"({type(exc).__name__}: {exc}). Not dialling while the "
                "account's state is unknown.")
    if sub["status"] not in GOOD_SUBSCRIPTION_STATUSES:
        return (
            f"ElevenLabs account status is {sub['status']!r} — no conversation "
            "can run until this is resolved at elevenlabs.io (almost always a "
            "payment issue). The agent WebSocket refuses the handshake in this "
            "state, so every call would ring, connect, and go silent."
        )

    # Nothing overridden -> nothing to validate, and NO API call. This is what
    # keeps an 'auto' campaign with no forced voice byte-identical to what it
    # was before the voice override existed: bridge.py sends no
    # conversation_config_override key at all, so ElevenLabs has nothing to
    # reject, and such a campaign must not gain a new way to not dial.
    if not language and not ELEVENLABS_VOICE_ID:
        return None

    try:
        support = await asyncio.to_thread(agent_language_support, agent_id)
    except Exception as exc:
        # Deliberately NOT a refusal. These checks are an extra guard over a
        # correctly-configured agent; an SDK or API change must not become
        # a dial-stopping outage when the call would have worked fine.
        # Every check above still applies.
        logger.warning(
            f"[preflight] could not read agent {agent_id}'s configuration "
            f"({type(exc).__name__}: {exc}) — dialling anyway"
        )
        return None

    # The voice first: once ELEVENLABS_VOICE_ID is set it is sent on EVERY
    # ElevenLabs-backed call, including 'auto', so an agent that doesn't allow
    # it fails 100% of its calls — a strictly larger blast radius than any
    # language gate below. .get(), not a subscript: preflight never raising is
    # a hard contract (worker.py would abort mid-reservation), and only the
    # to_thread call above is inside the try.
    if ELEVENLABS_VOICE_ID and not support.get("voice_override_allowed", False):
        return (
            f"ELEVENLABS_VOICE_ID is set ({ELEVENLABS_VOICE_ID}), so every call "
            f"sends a voice override, but agent {agent_id} does not allow it. "
            "ElevenLabs rejects an override for a field that isn't enabled, so "
            "every call would fail on answer. Open that agent's Security tab and "
            "enable the 'voice_id' override (Overrides -> TTS -> Voice ID), or "
            "clear ELEVENLABS_VOICE_ID in .env to use the agent's own voice. "
            "Run scripts/check_agent_language_support.py to see both agents."
        )

    if language:
        if not support["override_allowed"]:
            return (
                f"This campaign dials in '{language}', but agent {agent_id} does "
                "not allow the language override. ElevenLabs rejects an override "
                "for a field that isn't enabled, so every call would fail. Open "
                "the agent's Security tab and enable the 'language' override."
            )
        if support["languages"] and language not in support["languages"]:
            # Two very different failures wear the same shape here, and telling
            # them apart is the difference between a 30-second fix and an hour
            # spent hunting for a dashboard setting that does not exist.
            if not languages_module.elevenlabs_can_speak(language):
                return (
                    f"ElevenLabs Agents does not support '{language}' at all — it "
                    "is not one of the languages the platform offers, so no "
                    "dashboard setting, model change or plan upgrade will enable "
                    f"it. Agent {agent_id} cannot dial this campaign. Use a "
                    "backend that supports this language, or change the "
                    "campaign's language."
                )
            return (
                f"Agent {agent_id} is not configured for '{language}' "
                f"(it has: {', '.join(sorted(support['languages'])) or 'none'}). "
                "Add it under Additional Languages in the agent's settings."
            )
        model = (support["tts_model"] or "").lower()
        if language in languages_module.V3_ONLY_ISO and model and "v3" not in model:
            return (
                f"Agent {agent_id} runs the '{support['tts_model']}' TTS model, "
                f"which cannot speak '{language}' — Flash/Turbo v2.5's 32 "
                "languages do not include Telugu. The call would produce garbled "
                "audio and fall back to English. Switch the agent's model to "
                "Eleven v3 Conversational."
            )

    return None
