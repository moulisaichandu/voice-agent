"""telephony/plivo_client.py — Place outbound calls via Plivo's Voice API.

Plivo's STANDARD Voice API, not SIP trunking. The flow:

    calls.create(answer_url=...)   → Plivo dials the lead
    lead answers                   → Plivo GETs/POSTs our answer_url
    we return <Stream> XML         → Plivo opens a WebSocket to us
    app/telephony/bridge.py        → bridges that audio to the ElevenLabs agent

This is the same mechanism the sibling ai-voice-agent project uses, and it
needs nothing beyond PLIVO_AUTH_ID/PLIVO_AUTH_TOKEN. The alternative — having
ElevenLabs place the call — requires the number to be registered inside
ElevenLabs over a SIP trunk (Plivo's Zentrunk), which is provisioned
per-account and was not enabled on ours.

Uses httpx rather than the `plivo` SDK: the SDK is synchronous, and this
server runs the call worker, the scheduler and the audio bridges on one event
loop, so a blocking HTTP call here would stall live calls. httpx is already a
dependency (see preflight.py).
"""

from __future__ import annotations

import logging
from urllib.parse import quote

import httpx

from app.config import (
    CALL_RING_TIMEOUT_S,
    CALL_WEBHOOK_SECRET,
    PLIVO_AUTH_ID,
    PLIVO_AUTH_TOKEN,
    PLIVO_FROM_NUMBER,
    PUBLIC_BASE_URL,
)
from app.telephony import public_url

logger = logging.getLogger(__name__)

_API_ROOT = "https://api.plivo.com/v1/Account"
_TIMEOUT_S = 20.0


class PlivoCallResult:
    """Mirrors telephony.elevenlabs_client.OutboundCallResult so the worker's
    handling of a placement result is identical whichever transport placed it."""

    def __init__(self, success: bool, message: str | None, call_uuid: str | None):
        self.success = success
        self.message = message
        self.call_uuid = call_uuid


def is_configured() -> bool:
    return bool(PLIVO_AUTH_ID and PLIVO_AUTH_TOKEN and PLIVO_FROM_NUMBER and PUBLIC_BASE_URL)


def _webhook_urls(lead_id: str) -> tuple[str, str]:
    """answer_url and hangup_url for one lead, carrying the shared secret.

    lead_id travels in the query string because it is how the answer and
    stream webhooks — which Plivo calls, not us — find their way back to the
    right lead and call row. It is a UUID we generated, not a secret.
    """
    # public_url.base(), not the import-time PUBLIC_BASE_URL — see
    # app/telephony/public_url.py. A stale hostname here means the lead's phone
    # rings and the call drops the instant they answer, with nothing in our
    # logs, because Plivo's request never reaches us at all.
    base = public_url.base().rstrip("/")
    token = quote(CALL_WEBHOOK_SECRET or "")
    lead = quote(lead_id)
    return (f"{base}/calls/answer?token={token}&lead={lead}",
            f"{base}/calls/hangup?token={token}&lead={lead}")


async def place_outbound_call(*, to_number: str, lead_id: str) -> PlivoCallResult:
    """Dial *to_number*, pointing Plivo at this server's answer URL.

    Returns a result rather than raising for an API-level rejection, so the
    worker can mark the lead failed and move on; genuine transport errors
    (timeout, DNS) still raise and are handled by the worker's placement
    try/except.
    """
    if not is_configured():
        return PlivoCallResult(
            success=False,
            message=("Plivo is not fully configured — needs PLIVO_AUTH_ID, "
                     "PLIVO_AUTH_TOKEN, PLIVO_FROM_NUMBER and PUBLIC_BASE_URL."),
            call_uuid=None,
        )

    answer_url, hangup_url = _webhook_urls(lead_id)
    # Plivo rejects a leading '+' on these fields.
    payload = {
        "from": (PLIVO_FROM_NUMBER or "").lstrip("+"),
        "to": to_number.lstrip("+"),
        "answer_url": answer_url,
        "answer_method": "POST",
        "hangup_url": hangup_url,
        "hangup_method": "POST",
        "ring_timeout": CALL_RING_TIMEOUT_S,
    }

    url = f"{_API_ROOT}/{PLIVO_AUTH_ID}/Call/"
    async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
        resp = await client.post(url, json=payload, auth=(PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN))

    if resp.status_code not in (200, 201, 202):
        # Don't log the body blindly — Plivo echoes request fields, including
        # the answer_url with its webhook token in the query string.
        logger.error(f"[plivo] call placement rejected: HTTP {resp.status_code}")
        return PlivoCallResult(
            success=False,
            message=f"Plivo returned HTTP {resp.status_code}",
            call_uuid=None,
        )

    try:
        body = resp.json()
    except ValueError:
        body = {}
    call_uuid = body.get("request_uuid") or body.get("call_uuid")
    return PlivoCallResult(success=True, message=body.get("message"), call_uuid=call_uuid)


async def hangup(call_uuid: str) -> None:
    """Force-end a live call. Best-effort: used on timeout, where failing to
    hang up must not itself raise into the caller's cleanup path."""
    if not (call_uuid and is_configured()):
        return
    url = f"{_API_ROOT}/{PLIVO_AUTH_ID}/Call/{quote(call_uuid)}/"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            await client.delete(url, auth=(PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN))
    except Exception as exc:  # pragma: no cover - best effort
        logger.error(f"[plivo] hangup failed for {call_uuid}: {type(exc).__name__}: {exc}")
