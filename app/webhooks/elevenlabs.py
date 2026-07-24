"""webhooks/elevenlabs.py — POST /webhooks/elevenlabs, the transcript receiver.

Real, documented payload shape (verified against ElevenLabs' docs, not
guessed):
    {"type": "post_call_transcription", "event_timestamp": ..., "data": {
        "conversation_id": "...", "agent_id": "...", "status": "done",
        "transcript": [{"role": "agent"|"user", "message": "...", ...}],
        "metadata": {"call_duration_secs": ...},
        "analysis": {"transcript_summary": "...", "call_successful": "..."},
        "conversation_initiation_client_data": {
            "dynamic_variables": {"lead_id": "..."}
        }
    }}
Three event types total: post_call_transcription (handled below, retried up
to 5x by ElevenLabs), post_call_audio and call_initiation_failure (both
acknowledged but not deeply processed yet — no feature needs them).

ElevenLabs' role is "user" for the person on the call; this project's own
vocabulary is "lead" (see CLAUDE.md/db/models.py) — mapped explicitly below
rather than leaking ElevenLabs' terminology into stored data.
"""

from __future__ import annotations

import logging

from elevenlabs.errors import BadRequestError
from fastapi import APIRouter, HTTPException, Request

from app import redis_client
from app.config import ELEVENLABS_WEBHOOK_SECRET
from app.db import calls as calls_db
from app.db.models import TranscriptTurn
from app.telephony.elevenlabs_client import construct_webhook_event

router = APIRouter(tags=["Webhooks"])
logger = logging.getLogger(__name__)

# How long a processed conversation_id is remembered — long enough to outlast
# ElevenLabs' own retry window (up to 5 retries with exponential backoff) so a
# genuine retry after that finds no idempotency key and is (harmlessly)
# reprocessed rather than remembered forever.
_IDEMPOTENCY_TTL_S = 6 * 3600

_ROLE_MAP = {"agent": "agent", "user": "lead"}


def _map_transcript(raw_turns: list[dict]) -> list[TranscriptTurn]:
    turns = []
    for t in raw_turns or []:
        role = _ROLE_MAP.get(t.get("role"))
        message = t.get("message")
        if role and message:
            turns.append(TranscriptTurn(role=role, text=message))
    return turns


async def _handle_post_call_transcription(data: dict) -> None:
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        logger.warning("[webhooks] post_call_transcription with no conversation_id — dropping")
        return

    r = redis_client.get_redis()
    idem_key = f"idem:post_call_transcription:{conversation_id}"
    # SET ... NX: only the FIRST caller to see this conversation_id gets True
    # back — a retried webhook (ElevenLabs retries up to 5x) finds the key
    # already set and skips reprocessing, without needing a DB round-trip
    # first just to check.
    is_first = await r.set(idem_key, "1", nx=True, ex=_IDEMPOTENCY_TTL_S)
    if not is_first:
        logger.info(f"[webhooks] duplicate transcript for {conversation_id} — skipping")
        return

    # NOTE: this handler deliberately does NOT release the concurrency slot.
    # It used to, back when ElevenLabs placed the call and this webhook was
    # the only signal a call had ended. Now Plivo places the call and
    # app/telephony/call_routes.py owns slot release (guarded so the stream
    # end and Plivo's hangup post can't both release). This webhook still
    # fires for the same conversation, so releasing here as well would
    # double-release and let the semaphore over-admit past
    # MAX_CONCURRENT_CALLS.

    turns = _map_transcript(data.get("transcript"))
    summary = (data.get("analysis") or {}).get("transcript_summary")

    call = await calls_db.record_transcript(
        el_conversation_id=conversation_id,
        status=data.get("status") or "unknown",
        turns=len(turns),
        transcript=turns,
        summary=summary,
    )
    if call is None:
        # The call row should already exist (created at dial time) — if it
        # doesn't, either the worker never got to create_call(), or this
        # conversation_id doesn't belong to us. Log loudly; don't silently drop.
        logger.error(f"[webhooks] no call row for conversation_id={conversation_id} "
                     "— transcript could not be recorded")
        return

    # Sheets write-back is a QUEUE, not a direct call from here — several
    # two-way calls finishing near-simultaneously would otherwise risk
    # concurrent 429s from the Sheets API. app/sheets/sync.py (Phase 5)
    # consumes this list.
    if call.lead_id:
        await r.lpush("sheets:writeback:queue", str(call.lead_id))


async def _handle_call_initiation_failure(data: dict) -> None:
    # Unlike post_call_transcription's shape (confirmed against ElevenLabs'
    # docs), this event's exact field names are NOT verified — `.get()` is
    # used throughout so a wrong guess degrades to logging less detail rather
    # than raising. Confirm against a real payload before relying on this for
    # anything beyond "a call failed to initiate."
    conversation_id = data.get("conversation_id")
    logger.warning(f"[webhooks] call_initiation_failure conversation_id={conversation_id}: "
                    f"{data.get('reason') or data}")
    # Slot release is owned by app/telephony/call_routes.py — see the note in
    # _handle_post_call_transcription. Releasing here too would double-release.
    if conversation_id:
        await calls_db.record_transcript(
            el_conversation_id=conversation_id, status="failed", turns=0,
            transcript=[], summary=data.get("reason"),
        )


@router.post("/webhooks/elevenlabs")
async def elevenlabs_webhook(request: Request) -> dict:
    raw_body = (await request.body()).decode("utf-8")
    sig_header = request.headers.get("ElevenLabs-Signature", "")

    try:
        event = construct_webhook_event(raw_body, sig_header, ELEVENLABS_WEBHOOK_SECRET or "")
    except (BadRequestError, ValueError) as exc:
        # Deliberately generic 401 body — don't echo back WHY verification
        # failed (missing vs malformed vs stale timestamp), which would help
        # an attacker iterate toward a forged signature.
        #
        # ValueError is caught alongside BadRequestError because the SDK does
        # int(timestamp) on the attacker-supplied `t=` field BEFORE any HMAC
        # comparison: a non-numeric timestamp raises a bare ValueError, which
        # would otherwise escape as a 500 (and a stack trace) for an unverified,
        # unauthenticated request.
        logger.warning(f"[webhooks] signature verification failed: {exc}")
        raise HTTPException(status_code=401, detail="invalid signature") from exc

    event_type = event.get("type")
    data = event.get("data") or {}

    if event_type == "post_call_transcription":
        await _handle_post_call_transcription(data)
    elif event_type == "call_initiation_failure":
        await _handle_call_initiation_failure(data)
    elif event_type == "post_call_audio":
        pass  # acknowledged, not processed — no feature needs raw audio yet
    else:
        logger.warning(f"[webhooks] unrecognised event type: {event_type!r}")

    return {"status": "ok"}
