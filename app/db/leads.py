"""db/leads.py — CRUD + campaign-worthy queries for the leads table."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from app.db.models import Lead
from app.db.pool import get_pool


def _row_to_lead(row) -> Lead:
    return Lead(**dict(row))


async def upsert_lead(
    *,
    sheet_row: int | None,
    name: str | None,
    phone_e164: str,
    campaign_id: UUID,
    language_pref: str = "auto",
    consent_basis: str | None = None,
    consent_at: datetime | None = None,
) -> Lead:
    """Insert or update, keyed on (phone_e164, campaign_id) — re-importing the
    same sheet never creates duplicates or resets a lead already in progress."""
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        insert into leads (sheet_row, name, phone_e164, campaign_id,
                            language_pref, consent_basis, consent_at)
        values ($1, $2, $3, $4, $5, $6, $7)
        on conflict (phone_e164, campaign_id) do update
            set sheet_row = excluded.sheet_row,
                name = excluded.name,
                language_pref = excluded.language_pref,
                consent_basis = excluded.consent_basis,
                consent_at = excluded.consent_at
        returning *
        """,
        sheet_row, name, phone_e164, campaign_id, language_pref, consent_basis, consent_at,
    )
    return _row_to_lead(row)


async def get_lead(lead_id: UUID) -> Lead | None:
    pool = await get_pool()
    row = await pool.fetchrow("select * from leads where lead_id = $1", lead_id)
    return _row_to_lead(row) if row else None


async def get_lead_by_phone(phone_e164: str) -> Lead | None:
    """Used by app/sheets/writeback.py's fallback full scan when a cached
    sheet_row no longer matches — see CLAUDE.md's hard rule on this."""
    pool = await get_pool()
    row = await pool.fetchrow(
        "select * from leads where phone_e164 = $1 order by created_at desc limit 1",
        phone_e164,
    )
    return _row_to_lead(row) if row else None


async def due_leads(limit: int) -> list[Lead]:
    """Leads eligible for dialling right now: pending, not DND, under their
    campaign's max_attempts, on an active campaign."""
    pool = await get_pool()
    rows = await pool.fetch(
        """
        select l.* from leads l
        join campaigns c on c.campaign_id = l.campaign_id
        where l.status = 'pending' and l.dnd = false and c.active = true
          and l.attempts < c.max_attempts
        order by l.created_at
        limit $1
        """,
        limit,
    )
    return [_row_to_lead(r) for r in rows]


async def mark_queued(lead_id: UUID) -> bool:
    """Guarded transition: only a currently-pending lead can be queued, so a
    duplicate scheduler tick can't double-enqueue the same lead."""
    pool = await get_pool()
    result = await pool.execute(
        "update leads set status = 'queued' where lead_id = $1 and status = 'pending'",
        lead_id,
    )
    return result.endswith("1")


async def mark_calling(lead_id: UUID) -> bool:
    """Reserve-before-act: the worker calls this BEFORE placing the call, not
    after — see app/telephony/worker.py's docstring for why. Guarded by
    WHERE status='queued' so a duplicate pop from the Redis processing list
    can't double-transition (and therefore can't double-dial) the same lead.

    Deliberately does NOT increment `attempts` — that happens in
    record_dial_attempt(), only once a call is ACTUALLY placed. If this
    reservation is later released back to 'pending' (preflight failed, the
    per-number dialing lock was held, at concurrency capacity — see worker.py),
    no real attempt happened, and counting one here would let a lead get
    silently exhausted against max_attempts by pure infrastructure hiccups
    rather than real no-answer/failed calls."""
    pool = await get_pool()
    result = await pool.execute(
        "update leads set status = 'calling' where lead_id = $1 and status = 'queued'",
        lead_id,
    )
    return result.endswith("1")


async def record_dial_attempt(lead_id: UUID) -> None:
    """Called by the worker right before place_outbound_call() — this is the
    ONLY place `attempts` increments, so it counts calls actually placed with
    ElevenLabs, not reservation attempts that got released before dialling."""
    pool = await get_pool()
    await pool.execute(
        "update leads set attempts = attempts + 1, last_called_at = now() "
        "where lead_id = $1",
        lead_id,
    )


async def mark_result(lead_id: UUID, status: str) -> None:
    pool = await get_pool()
    await pool.execute("update leads set status = $2 where lead_id = $1", lead_id, status)


async def mark_dnd(lead_id: UUID) -> None:
    pool = await get_pool()
    await pool.execute(
        "update leads set dnd = true, status = 'dnd' where lead_id = $1", lead_id
    )


async def dnd_phones() -> list[str]:
    """Every phone number currently flagged dnd=true in Supabase — used by
    scheduler.dnd_refresh() to (re)populate Redis's dnd:numbers set. See that
    function's docstring: this is NOT a real NCPR/TRAI registry integration,
    only a propagation of flags already recorded here."""
    pool = await get_pool()
    rows = await pool.fetch("select distinct phone_e164 from leads where dnd = true")
    return [r["phone_e164"] for r in rows]
