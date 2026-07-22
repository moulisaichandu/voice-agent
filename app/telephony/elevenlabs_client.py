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

from elevenlabs.client import ElevenLabs
from elevenlabs.core.api_error import ApiError
from elevenlabs.types.conversation_initiation_client_data_request_input import (
    ConversationInitiationClientDataRequestInput,
)

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
    """
    sub = get_client().user.subscription.get()
    return {
        "status": sub.status,
        "character_count": sub.character_count,
        "character_limit": sub.character_limit,
    }


def construct_webhook_event(raw_body: str, sig_header: str, secret: str) -> dict:
    """Verifies the ElevenLabs-Signature HMAC and returns the parsed payload.
    Raises on a missing/invalid/expired signature — callers must not catch
    broadly and continue; an unverified webhook must be rejected outright."""
    return get_client().webhooks.construct_event(raw_body, sig_header, secret)
