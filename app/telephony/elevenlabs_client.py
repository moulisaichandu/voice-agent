"""telephony/elevenlabs_client.py — Thin wrapper over the ElevenLabs SDK.

Written against the REAL SDK shapes (elevenlabs==2.58.0), verified by direct
introspection rather than assumed from the blueprint's prose. This project's
phone number is a Plivo number connected to ElevenLabs via a SIP trunk — the
SDK has no dedicated Plivo integration (only exotel, twilio, and generic
sip_trunk), so calls go through the sip_trunk namespace:
  client.conversational_ai.sip_trunk.outbound_call(
      agent_id, agent_phone_number_id, to_number,
      conversation_initiation_client_data=...,   # carries dynamic_variables
  ) -> SipTrunkOutboundCallResponse(success, message, conversation_id, sip_call_id)
  client.conversational_ai.agents.get(agent_id)               -> raises NotFoundError if missing
  client.conversational_ai.phone_numbers.get(phone_number_id) -> raises NotFoundError if missing
  client.webhooks.construct_event(rawBody, sig_header, secret) -> dict (raises on bad signature)

(agents.get/phone_numbers.get/webhooks.construct_event are all provider-agnostic
in the SDK — nothing about them is Exotel/Twilio/SIP-trunk-specific.)
"""

from __future__ import annotations

import asyncio
import json

from elevenlabs.client import ElevenLabs
from elevenlabs.core.api_error import ApiError
from elevenlabs.types.conversation_initiation_client_data_request_input import (
    ConversationInitiationClientDataRequestInput,
)

from app import redis_client
from app.config import ELEVENLABS_API_KEY

_client: ElevenLabs | None = None


def get_client() -> ElevenLabs:
    global _client
    if _client is None:
        if not ELEVENLABS_API_KEY:
            raise RuntimeError("ELEVENLABS_API_KEY is not set — cannot call ElevenLabs.")
        _client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    return _client


class OutboundCallResult:
    def __init__(self, success: bool, message: str | None,
                 conversation_id: str | None, call_sid: str | None):
        self.success = success
        self.message = message
        self.conversation_id = conversation_id
        self.call_sid = call_sid


def place_outbound_call(
    *, agent_id: str, agent_phone_number_id: str, to_number: str,
    dynamic_variables: dict | None = None,
) -> OutboundCallResult:
    """Place an outbound call over the Plivo-via-SIP-trunk number linked in
    ElevenLabs. dynamic_variables should include lead_id — see
    app/db/calls.py's docstring on why: it's what lets the transcript webhook
    round-trip back to the right lead/call row without depending on a
    separate, easy-to-lose webhook_id field."""
    client = get_client()
    init_data = ConversationInitiationClientDataRequestInput(
        dynamic_variables=dynamic_variables or {},
    )
    resp = client.conversational_ai.sip_trunk.outbound_call(
        agent_id=agent_id,
        agent_phone_number_id=agent_phone_number_id,
        to_number=to_number,
        conversation_initiation_client_data=init_data,
    )
    return OutboundCallResult(
        success=resp.success, message=resp.message,
        conversation_id=resp.conversation_id, call_sid=resp.sip_call_id,
    )


def _missing(exc: ApiError) -> bool:
    """Whether an ApiError means "this id doesn't exist" rather than "the call
    failed".

    Checks the STATUS CODE, not the exception class. Verified against the live
    API: `agents.get()` on a missing id raises a bare ApiError with
    status_code=404 — NOT the SDK's NotFoundError, even though that class
    exists and subclasses ApiError. Catching NotFoundError alone (as this did
    originally) therefore never fired, so a typo'd or deleted agent id raised
    straight out of preflight instead of failing it cleanly. Unit tests that
    mock NotFoundError pass either way, which is exactly why this survived
    until it was run against the real API.
    """
    return getattr(exc, "status_code", None) == 404


def agent_exists(agent_id: str) -> bool:
    """Used by preflight — a typo'd/deleted agent_id should fail ONCE, loudly,
    before a whole campaign dials, not identically on every lead.

    Only a 404 means "no such agent". Anything else (401 bad key, 429 rate
    limit, 5xx) is re-raised: reporting those as "agent not found" would send
    whoever reads the preflight message hunting for a deleted agent that is
    actually fine.
    """
    try:
        get_client().conversational_ai.agents.get(agent_id)
        return True
    except ApiError as exc:
        if _missing(exc):
            return False
        raise


def phone_number_exists(agent_phone_number_id: str) -> bool:
    try:
        get_client().conversational_ai.phone_numbers.get(agent_phone_number_id)
        return True
    except ApiError as exc:
        if _missing(exc):
            return False
        raise


def subscription_status() -> dict:
    """The account's billing status — used by the admin readiness check.

    A non-active status blocks every conversation at the PLATFORM level, not
    just this app: verified live (2026-07) when the account went `past_due`
    and the agent WebSocket refused the handshake with "This request cannot
    be processed due to a payment issue." No retry or backoff on our side
    fixes that — only the invoice being paid does — so it is worth a readiness
    check of its own rather than only surfacing as an inexplicable dial
    failure later.

    Most callers should use cached_subscription_status() below instead — this
    is the uncached primitive it wraps.
    """
    sub = get_client().user.subscription.get()
    return {
        "status": sub.status,
        "character_count": sub.character_count,
        "character_limit": sub.character_limit,
    }


GOOD_SUBSCRIPTION_STATUSES = {"active", "trialing"}

_SUBSCRIPTION_CACHE_KEY = "elevenlabs:subscription_status"
# Matches _READINESS_CACHE_TTL_S in app/admin/system.py — the same 60s window
# the dashboard's other expensive check already uses, not a new number to tune.
_SUBSCRIPTION_CACHE_TTL_S = 60


async def cached_subscription_status(*, force_refresh: bool = False) -> dict:
    """subscription_status(), cached in Redis for _SUBSCRIPTION_CACHE_TTL_S.

    Lives here rather than in the readiness dashboard because there are now
    TWO callers: app/admin/system.py, and app/telephony/preflight.py's
    dial-time gate. With the caching stranded in the dashboard, preflight
    could not reach it, and a poll plus a dial-check inside the same window
    would each pay for their own ElevenLabs round-trip.

    force_refresh bypasses the cache, for POST /admin/readiness/refresh —
    "I just paid the invoice, tell me now" is the one case where waiting out
    a TTL is exactly wrong.

    Redis failures are swallowed on BOTH sides: it is the ephemeral speed
    layer here, never a dependency that can take the answer away. An API
    failure is NOT swallowed — the caller decides what an unreachable
    ElevenLabs means (preflight refuses to dial; the dashboard reports "could
    not check"), and caching a failure as though it were a healthy result
    would let a dead account dial.
    """
    r = redis_client.get_redis()
    if not force_refresh:
        try:
            raw = await r.get(_SUBSCRIPTION_CACHE_KEY)
        except Exception:  # noqa: BLE001 - a cache outage is just a cache miss
            raw = None
        if raw:
            try:
                return json.loads(raw)
            except (TypeError, ValueError):
                pass  # corrupted entry — fall through to a fresh call
    # Off the event loop: subscription_status is a SYNCHRONOUS SDK call, and
    # this runs on the loop driving live audio bridges.
    result = await asyncio.to_thread(subscription_status)
    try:
        await r.set(_SUBSCRIPTION_CACHE_KEY, json.dumps(result),
                    ex=_SUBSCRIPTION_CACHE_TTL_S)
    except Exception:  # noqa: BLE001 - best-effort; a write failure is not a call failure
        pass
    return result


def agent_language_support(agent_id: str) -> dict:
    """What languages this agent can actually be asked to speak, and in which voice.

    Returns {"override_allowed": bool, "languages": set[str],
             "tts_model": str | None, "voice_id": str | None,
             "voice_override_allowed": bool, "preset_voice_ids": dict[str, str]}.

    The voice keys exist because app/telephony/bridge.py can force a voice on
    every call (config's ELEVENLABS_VOICE_ID): `voice_override_allowed` is what
    preflight checks before dialling, since ElevenLabs rejects the conversation
    outright if the Security tab doesn't allow the field. `voice_id` and
    `preset_voice_ids` are diagnostics — a per-language preset PINS a voice for
    that language, so changing the agent's base voice in the dashboard cannot
    affect it, which is the usual reason a voice change appears not to work.

    Read with getattr chains rather than direct attribute access on purpose:
    this reaches four levels into an SDK response whose shape is not part of
    any contract we control, and a missing intermediate must degrade to "I
    don't know" rather than raise inside preflight — see this module's
    docstring on why the SDK's type stubs are not trusted here.

    Paths verified 2026-07-22 by dumping a real two-way agent's config
    (elevenlabs==2.58.0):
      agent.conversation_config.agent.language              -> "en"
      agent.conversation_config.language_presets             -> {} (dict, key
                                                                  = iso code)
      agent.conversation_config.tts.model_id                -> "eleven_flash_v2"
      agent.platform_settings.overrides.conversation_config_override
          .agent.language                                   -> False

    Voice paths, verified the same way against elevenlabs==2.58.0's types:
      agent.conversation_config.tts.voice_id                -> the base voice
      agent.conversation_config.language_presets[iso]
          .overrides.tts.voice_id                           -> a PINNED
                                                               per-language voice
      agent.platform_settings.overrides.conversation_config_override
          .tts.voice_id                                     -> bool, the
                                                               Security-tab
                                                               allowlist flag
    """
    agent = get_client().conversational_ai.agents.get(agent_id)

    conv = getattr(agent, "conversation_config", None)
    agent_cfg = getattr(conv, "agent", None)
    tts_cfg = getattr(conv, "tts", None)

    languages_: set[str] = set()
    default = getattr(agent_cfg, "language", None)
    if default:
        languages_.add(str(default))
    presets = getattr(conv, "language_presets", None) or {}
    try:
        languages_.update(str(k) for k in presets)
    except TypeError:
        pass

    # A voice pinned for ONE language by a preset. Same defensive walk as
    # above: an unexpected shape must yield "none found", never raise.
    preset_voices: dict[str, str] = {}
    try:
        for iso, preset in presets.items():
            pinned = getattr(
                getattr(getattr(preset, "overrides", None), "tts", None),
                "voice_id", None,
            )
            if pinned:
                preset_voices[str(iso)] = str(pinned)
    except (AttributeError, TypeError):
        pass

    overrides = getattr(getattr(agent, "platform_settings", None), "overrides", None)
    ov_conv = getattr(overrides, "conversation_config_override", None)
    ov_agent = getattr(ov_conv, "agent", None)
    ov_tts = getattr(ov_conv, "tts", None)

    return {
        "override_allowed": bool(getattr(ov_agent, "language", False)),
        "languages": languages_,
        "tts_model": getattr(tts_cfg, "model_id", None),
        "voice_id": getattr(tts_cfg, "voice_id", None),
        "voice_override_allowed": bool(getattr(ov_tts, "voice_id", False)),
        "preset_voice_ids": preset_voices,
    }


def construct_webhook_event(raw_body: str, sig_header: str, secret: str) -> dict:
    """Verifies the ElevenLabs-Signature HMAC and returns the parsed payload.
    Raises on a missing/invalid/expired signature — callers must not catch
    broadly and continue; an unverified webhook must be rejected outright."""
    return get_client().webhooks.construct_event(raw_body, sig_header, secret)
