"""telephony/elevenlabs_client.py — Thin wrapper over the ElevenLabs SDK.

Written against the REAL SDK shapes (elevenlabs==2.58.0), verified by direct
introspection rather than assumed from the blueprint's prose:
  client.conversational_ai.exotel.outbound_call(
      agent_id, agent_phone_number_id, to_number,
      conversation_initiation_client_data=...,   # carries dynamic_variables
  ) -> ExotelOutboundCallResponse(success, message, conversation_id, call_sid)
  client.conversational_ai.agents.get(agent_id)               -> raises NotFoundError if missing
  client.conversational_ai.phone_numbers.get(phone_number_id) -> raises NotFoundError if missing
  client.webhooks.construct_event(rawBody, sig_header, secret) -> dict (raises on bad signature)
"""

from __future__ import annotations

from elevenlabs.client import ElevenLabs
from elevenlabs.errors import NotFoundError
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
    """Place an outbound call via Exotel. dynamic_variables should include
    lead_id — see app/db/calls.py's docstring on why: it's what lets the
    transcript webhook round-trip back to the right lead/call row without
    depending on a separate, easy-to-lose webhook_id field."""
    client = get_client()
    init_data = ConversationInitiationClientDataRequestInput(
        dynamic_variables=dynamic_variables or {},
    )
    resp = client.conversational_ai.exotel.outbound_call(
        agent_id=agent_id,
        agent_phone_number_id=agent_phone_number_id,
        to_number=to_number,
        conversation_initiation_client_data=init_data,
    )
    return OutboundCallResult(
        success=resp.success, message=resp.message,
        conversation_id=resp.conversation_id, call_sid=resp.call_sid,
    )


def agent_exists(agent_id: str) -> bool:
    """Used by preflight — a typo'd/deleted agent_id should fail ONCE, loudly,
    before a whole campaign dials, not identically on every lead."""
    try:
        get_client().conversational_ai.agents.get(agent_id)
        return True
    except NotFoundError:
        return False


def phone_number_exists(agent_phone_number_id: str) -> bool:
    try:
        get_client().conversational_ai.phone_numbers.get(agent_phone_number_id)
        return True
    except NotFoundError:
        return False


def construct_webhook_event(raw_body: str, sig_header: str, secret: str) -> dict:
    """Verifies the ElevenLabs-Signature HMAC and returns the parsed payload.
    Raises on a missing/invalid/expired signature — callers must not catch
    broadly and continue; an unverified webhook must be rejected outright."""
    return get_client().webhooks.construct_event(raw_body, sig_header, secret)
