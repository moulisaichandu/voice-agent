"""telephony/call_routes.py — The Plivo-facing webhooks.

    POST /calls/answer   Plivo fetches this when the lead picks up; we return
                         the <Stream> XML that opens the audio WebSocket.
    WS   /calls/stream   The call audio itself, bridged to the ElevenLabs agent.
    POST /calls/hangup   Why the call ended — including calls that never
                         connected, which otherwise vanish with no reason.

Plivo cannot send APP_AUTH_TOKEN, so these authenticate with
CALL_WEBHOOK_SECRET in the query string, compared with compare_digest.

Concurrency-slot accounting lives here because this is where a call actually
ENDS. The worker acquires a slot before dialling; exactly one of the paths
below must release it, and a call can plausibly hit both (the stream ends,
then Plivo posts the hangup), so release is guarded by a Redis SET NX key —
double-releasing would let the semaphore over-admit and quietly break the
MAX_CONCURRENT_CALLS cap.
"""

from __future__ import annotations

import logging
import secrets
from uuid import UUID

from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import Response

from app import redis_client
from app.config import CALL_WEBHOOK_SECRET
from app.db import calls as calls_db
from app.db import campaigns as campaigns_db
from app.db import leads as leads_db
from app.telephony import bridge as bridge_module
from app.telephony import worker as telephony_worker

router = APIRouter(tags=["Telephony"])
logger = logging.getLogger(__name__)

# Long enough to outlast any single call (CALL_MAX_DURATION_S plus slack), so
# the guard is still present when a late hangup arrives after the stream ended.
_SLOT_GUARD_TTL_S = 3600


def _authorized(scope) -> bool:
    """Constant-time comparison of the shared secret.

    An unset secret means open, matching the sibling project's dev behaviour —
    but app/main.py refuses to boot with a public URL and no secret, so this
    only ever happens on a local, non-public server.
    """
    if not CALL_WEBHOOK_SECRET:
        return True
    token = scope.query_params.get("token") or ""
    return secrets.compare_digest(token, CALL_WEBHOOK_SECRET)


async def _release_slot_once(lead_id: str) -> None:
    """Release this call's concurrency slot, at most once per lead."""
    r = redis_client.get_redis()
    first = await r.set(f"slot:released:{lead_id}", "1", nx=True, ex=_SLOT_GUARD_TTL_S)
    if first:
        await telephony_worker.release_call_slot()


@router.post("/calls/answer")
async def answer(request: Request) -> Response:
    """Return the <Stream> XML that connects the call's audio to us."""
    lead_id = request.query_params.get("lead", "")
    if not _authorized(request):
        logger.warning(f"[calls] /calls/answer rejected (bad token) lead={lead_id}")
        return Response(status_code=403, content="forbidden")

    # Plivo posts the live CallUUID here; recording it is what makes a
    # force-hangup possible later.
    try:
        form = await request.form()
        call_uuid = form.get("CallUUID")
    except Exception:
        call_uuid = None
    if call_uuid and lead_id:
        r = redis_client.get_redis()
        await r.set(f"call:uuid:{lead_id}", call_uuid, ex=_SLOT_GUARD_TTL_S)

    logger.info(f"[calls] answered lead={lead_id} CallUUID={call_uuid}")
    return Response(content=bridge_module.answer_xml(lead_id),
                    media_type="application/xml")


@router.websocket("/calls/stream")
async def stream(ws: WebSocket) -> None:
    """The call's audio, bridged to the ElevenLabs agent."""
    lead_id = ws.query_params.get("lead", "")
    if not _authorized(ws):
        logger.warning(f"[calls] /calls/stream rejected (bad token) lead={lead_id}")
        await ws.close(code=1008)
        return

    await ws.accept()

    try:
        lead = await leads_db.get_lead(UUID(lead_id))
    except (ValueError, AttributeError):
        lead = None
    if lead is None or lead.campaign_id is None:
        logger.error(f"[calls] /calls/stream: no lead for {lead_id!r} — dropping the call")
        await ws.close()
        await _release_slot_once(lead_id)
        return

    campaign = await campaigns_db.get_campaign(lead.campaign_id)
    if campaign is None:
        logger.error(f"[calls] /calls/stream: no campaign for lead {lead_id} — dropping")
        await ws.close()
        await _release_slot_once(lead_id)
        return

    dynamic_variables = {"lead_id": str(lead.lead_id)}
    if lead.name:
        dynamic_variables["lead_name"] = lead.name
    if campaign.mode == "oneway" and campaign.script:
        dynamic_variables["script"] = campaign.script

    outcome = {"status": "failed", "turns": 0, "transcript": [],
               "conversation_id": None}
    try:
        outcome = await bridge_module.bridge(
            ws,
            agent_id=campaign.agent_id,
            lead_id=str(lead.lead_id),
            dynamic_variables=dynamic_variables,
            one_way=(campaign.mode == "oneway"),
        )
    except Exception as exc:
        # A failed bridge must still fall through to recording + slot release
        # below, or the lead is left 'calling' and a slot leaks for TTL.
        logger.exception(f"[calls] bridge failed for lead {lead_id}: "
                         f"{type(exc).__name__}: {exc}")
    finally:
        await _finalise_call(lead, outcome)
        await _release_slot_once(lead_id)


async def _finalise_call(lead, outcome: dict) -> None:
    """Record the transcript and queue the Sheets write-back.

    This replaces what the post-call transcript webhook used to do. That
    webhook still exists and is still verified, but with the bridge in the
    media path the transcript is already here the moment the call ends — no
    round-trip, and no dependency on ElevenLabs reaching us.
    """
    conversation_id = outcome.get("conversation_id")
    try:
        if conversation_id:
            await calls_db.set_conversation_and_record(
                lead_id=lead.lead_id,
                el_conversation_id=conversation_id,
                status=outcome.get("status") or "unknown",
                turns=outcome.get("turns") or 0,
                transcript=outcome.get("transcript") or [],
                summary=None,
            )
        await leads_db.mark_result(
            lead.lead_id, "done" if outcome.get("status") == "done" else "failed"
        )
    except Exception as exc:
        logger.exception(f"[calls] could not record outcome for lead "
                         f"{lead.lead_id}: {type(exc).__name__}: {exc}")
        return

    try:
        r = redis_client.get_redis()
        await r.lpush("sheets:writeback:queue", str(lead.lead_id))
    except Exception as exc:
        # Never block or fail a call on the Sheets mirror — CLAUDE.md's rule.
        logger.error(f"[calls] could not queue Sheets write-back for "
                     f"{lead.lead_id}: {type(exc).__name__}: {exc}")


@router.post("/calls/hangup")
async def hangup(request: Request) -> Response:
    """Record why a call ended, and release the slot if no stream ever ran.

    Plivo posts here for EVERY call, including ones that never connected (the
    lead rejected it, the carrier dropped it, our answer URL was unreachable).
    Those never reach /calls/stream, so without this their concurrency slot
    would be held until its guard expired. Always 200, never raises — an error
    here makes Plivo retry-storm.
    """
    lead_id = request.query_params.get("lead", "")
    if not _authorized(request):
        logger.warning(f"[calls] /calls/hangup rejected (bad token) lead={lead_id}")
        return Response(status_code=403, content="forbidden")

    try:
        form = await request.form()
        fields = {k: form.get(k) for k in
                  ("CallUUID", "CallStatus", "HangupCause", "Duration")}
    except Exception:
        fields = {}

    duration = (fields.get("Duration") or "0").strip() or "0"
    if duration in ("0", "0.0"):
        # Duration 0 means the call never carried audio — it died on answer.
        # That is the failure worth shouting about; it is what a dead tunnel
        # or an unreachable answer URL looks like from Plivo's side.
        logger.warning(f"[calls] CALL ENDED WITHOUT CONNECTING lead={lead_id} "
                       f"status={fields.get('CallStatus')} "
                       f"cause={fields.get('HangupCause')}")
        # Back to 'pending', NOT a 'no_answer' status: that string isn't in
        # db/models.py's LeadStatus, and since the DB column is unconstrained
        # text it would write fine and then fail validation on every later
        # read of that row. 'pending' is also the correct retry semantics —
        # due_leads() re-selects it, still bounded by max_attempts because
        # record_dial_attempt() already counted this attempt.
        try:
            if lead_id:
                await leads_db.mark_result(UUID(lead_id), "pending")
        except Exception:
            logger.exception(f"[calls] could not release lead {lead_id} for retry")
    else:
        logger.info(f"[calls] call ended lead={lead_id} duration={duration}s "
                    f"cause={fields.get('HangupCause')}")

    if lead_id:
        await _release_slot_once(lead_id)
    return Response(status_code=200, content="ok")
