"""db/calls.py — CRUD for the calls table.

el_conversation_id is how the transcript webhook (keyed by ElevenLabs'
conversation_id) maps back to a call/lead. The UNIQUE constraint on it
(migrations/0001_init.sql) is schema-level defense in depth alongside the
Redis idempotency key (idem:post_call_transcription:{conversation_id}) — a
retried webhook cannot create a duplicate row even if the Redis key expired.
"""

from __future__ import annotations

import json
from uuid import UUID

from app.db.models import Call, TranscriptTurn
from app.db.pool import get_pool


def _row_to_call(row) -> Call:
    d = dict(row)
    if d.get("transcript"):
        raw = json.loads(d["transcript"]) if isinstance(d["transcript"], str) else d["transcript"]
        d["transcript"] = [TranscriptTurn(**t) for t in raw]
    return Call(**d)


async def create_call(
    *, lead_id: UUID, campaign_id: UUID, mode: str,
    el_conversation_id: str | None = None, provider_call_id: str | None = None,
) -> Call:
    """Called by the worker right after (successfully) placing the call —
    el_conversation_id is usually known immediately from the placement API
    response; pass None if it arrives later and call set_conversation_id()."""
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        insert into calls (lead_id, campaign_id, mode, el_conversation_id,
                            provider_call_id, started_at)
        values ($1, $2, $3, $4, $5, now())
        returning *
        """,
        lead_id, campaign_id, mode, el_conversation_id, provider_call_id,
    )
    return _row_to_call(row)


async def set_conversation_and_record(
    *, lead_id: UUID, el_conversation_id: str, status: str, turns: int,
    transcript: list[TranscriptTurn], summary: str | None,
) -> Call | None:
    """Attach an ElevenLabs conversation_id to this lead's most recent call row
    and record the transcript, in one statement.

    Needed by the audio-bridge path (app/telephony/call_routes.py): with Plivo
    placing the call, the row is created at dial time when no ElevenLabs
    conversation exists yet — that id only appears once the bridge connects.
    record_transcript() can't be used because it looks the row up BY
    conversation_id, which is precisely what is still missing.

    Scoped to the newest call for the lead so a retry updates the attempt it
    belongs to rather than an older one.
    """
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        update calls
        set el_conversation_id = $2, status = $3, turns = $4,
            transcript = $5::jsonb, summary = $6, ended_at = now()
        where call_id = (
            select call_id from calls where lead_id = $1
            order by created_at desc limit 1
        )
        returning *
        """,
        lead_id, el_conversation_id, status, turns,
        json.dumps([t.model_dump() for t in transcript]), summary,
    )
    return _row_to_call(row) if row else None


async def get_latest_call_for_lead(lead_id: UUID) -> Call | None:
    """Used by app/sheets/writeback.py — the most recent call attempt is what
    gets written back to the lead's Sheet row."""
    pool = await get_pool()
    row = await pool.fetchrow(
        "select * from calls where lead_id = $1 order by created_at desc limit 1",
        lead_id,
    )
    return _row_to_call(row) if row else None


async def list_calls_for_campaign(campaign_id: UUID) -> list[Call]:
    """Used by the admin API's per-campaign call/transcript list view."""
    pool = await get_pool()
    rows = await pool.fetch(
        "select * from calls where campaign_id = $1 order by created_at desc", campaign_id,
    )
    return [_row_to_call(r) for r in rows]


async def get_call_by_conversation_id(el_conversation_id: str) -> Call | None:
    pool = await get_pool()
    row = await pool.fetchrow(
        "select * from calls where el_conversation_id = $1", el_conversation_id
    )
    return _row_to_call(row) if row else None


async def record_transcript(
    *, el_conversation_id: str, status: str, turns: int,
    transcript: list[TranscriptTurn], summary: str | None,
) -> Call | None:
    """Called by the post_call_transcription webhook handler. Idempotent by
    construction: if this conversation_id was already recorded, the caller
    should have already short-circuited via the Redis idempotency key before
    reaching here — this just does the write."""
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        update calls
        set status = $2, turns = $3,
            transcript = $4::jsonb, summary = $5, ended_at = now()
        where el_conversation_id = $1
        returning *
        """,
        el_conversation_id, status, turns,
        json.dumps([t.model_dump() for t in transcript]), summary,
    )
    return _row_to_call(row) if row else None
