"""db/leads.py — CRUD + campaign-worthy queries for the leads table."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from app.compliance.consent import has_valid_consent
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
    dial_order: int | None = None,
) -> Lead:
    """Insert or update, keyed on (phone_e164, campaign_id) — re-importing the
    same sheet never creates duplicates or resets a lead already in progress.

    *dial_order* is the lead's position in the file it was imported from; see
    migrations/0002 and due_leads(). It is preserved rather than overwritten
    when the caller passes None, so a Google-Sheet sync of a lead that
    originally came from an uploaded file doesn't silently wipe its position
    and drop it to the back of the calling order."""
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        insert into leads (sheet_row, name, phone_e164, campaign_id,
                            language_pref, consent_basis, consent_at, dial_order)
        values ($1, $2, $3, $4, $5, $6, $7, $8)
        on conflict (phone_e164, campaign_id) do update
            set sheet_row = excluded.sheet_row,
                name = excluded.name,
                language_pref = excluded.language_pref,
                -- coalesced, like dial_order below: sheets_sync passes NULL
                -- for a Sheet with no consent columns, and overwriting a
                -- consent captured by the file-import path silently makes a
                -- dialable lead undialable. A background job must never be
                -- able to clear a compliance field.
                consent_basis = coalesce(excluded.consent_basis, leads.consent_basis),
                consent_at = coalesce(excluded.consent_at, leads.consent_at),
                dial_order = coalesce(excluded.dial_order, leads.dial_order)
        returning *
        """,
        sheet_row, name, phone_e164, campaign_id, language_pref, consent_basis,
        consent_at, dial_order,
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


async def due_leads(limit: int, campaign_id: UUID | None = None) -> list[Lead]:
    """Leads eligible for dialling right now: pending, not DND, under their
    campaign's max_attempts, on an active campaign, AND with a recorded valid
    consent basis + timestamp (app.compliance.consent.has_valid_consent) — a
    lead with no consent record on file is exactly as dial-ineligible as a
    DND one.

    Ordered by dial_order (a lead's position in the file it was uploaded from)
    so a campaign calls its list top-to-bottom the way the operator wrote it,
    falling back to created_at for leads that came from the Google Sheet or the
    single-lead form and therefore have no file position. See migrations/0002 —
    created_at alone could not express this, because a bulk import writes every
    row in the same instant and upsert deliberately never rewrites created_at.

    The consent check is deliberately pure Python, not duplicated into SQL,
    so there's exactly one place (compliance/consent.py, already unit-tested)
    that decides what counts as valid consent. That means over-fetching
    candidates before filtering: a batch with some consent-ineligible rows
    would otherwise silently return fewer than `limit` even though more
    eligible leads exist further down the same query. campaign_tick's own
    batch size is already a soft target (see its docstring), so returning
    somewhat fewer than `limit` on a heavily consent-gated batch is consistent
    with that existing tradeoff, not a new one.

    *campaign_id* narrows the selection to one campaign, for the admin API's
    per-campaign trigger. It only ever NARROWS: every gate above (pending,
    not DND, active campaign, under max_attempts, valid consent) still
    applies, so a scoped trigger can never dial someone a global tick
    wouldn't have."""
    pool = await get_pool()
    candidate_limit = max(limit * 5, limit + 20)
    rows = await pool.fetch(
        """
        select l.* from leads l
        join campaigns c on c.campaign_id = l.campaign_id
        where l.status = 'pending' and l.dnd = false and c.active = true
          and l.attempts < c.max_attempts
          and ($2::uuid is null or l.campaign_id = $2)
        order by l.dial_order nulls last, l.created_at
        limit $1
        """,
        candidate_limit, campaign_id,
    )
    eligible = [r for r in rows if has_valid_consent(r["consent_basis"], r["consent_at"])]
    return [_row_to_lead(r) for r in eligible[:limit]]


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


async def mark_result_if_calling(lead_id: UUID, status: str) -> bool:
    """Guarded transition, in the same family as mark_queued/mark_calling:
    only a lead still in 'calling' can be resolved, and only once.

    Two independent code paths report the end of the same call —
    call_routes._finalise_call (from the audio stream closing) and
    /calls/hangup (from Plivo's post) — with no ordering guarantee between
    them. Both used the unguarded mark_result, so the later one silently
    overwrote the earlier. Plivo reports Duration 0 for a call that connected
    but lasted under a second, and hangup writes 'pending' on Duration 0: a
    completed call could therefore be reset to 'pending' and dialled again,
    or a genuine no-connect marked 'failed' and never retried.

    With the guard, the first writer wins and the second no-ops — correct in
    both orders, since either outcome is a truthful description of a call that
    both connected and ended almost immediately. Returns whether this call is
    the one that resolved the lead."""
    pool = await get_pool()
    result = await pool.execute(
        "update leads set status = $2 where lead_id = $1 and status = 'calling'",
        lead_id, status,
    )
    return result.endswith("1")


async def queued_lead_ids() -> list[str]:
    """Every lead currently reserved as 'queued', as strings.

    Strings because the only caller compares them against Redis list members,
    which are strings. See worker.reap_orphaned_queued_leads.
    """
    pool = await get_pool()
    rows = await pool.fetch("select lead_id from leads where status = 'queued'")
    return [str(r["lead_id"]) for r in rows]


async def release_orphaned_queued_lead(lead_id) -> bool:
    """Return a 'queued' lead to 'pending' so due_leads() can select it again.

    Guarded on 'queued' so this can never disturb a lead that has moved on to
    'calling' since the caller took its snapshot.
    """
    pool = await get_pool()
    result = await pool.execute(
        "update leads set status = 'pending' "
        "where lead_id = $1 and status = 'queued'",
        UUID(str(lead_id)),
    )
    return result.endswith("1")


async def release_undialed_calling_lead(lead_id: UUID) -> bool:
    """Return a lead reserved for a call that was never actually placed.

    The crashed-reservation case. worker.process_one moves a lead to 'calling'
    BEFORE dialling, then runs preflight, then record_dial_attempt, then places
    the call. A process death anywhere in that window leaves the lead 'calling'
    with last_called_at NULL — and nothing could recover it:
    stale_calling_leads() has no clock to measure it by, and the reaper's
    requeue is rejected by mark_calling's `where status='queued'` guard, so the
    lead was acked away into no queue and no processing list.

    `last_called_at is null` is the safety guard, not an optimisation: that
    column is written by record_dial_attempt, which runs BEFORE placement, so a
    NULL there means no call can possibly be in flight. A lead whose attempt was
    really placed stays 'calling' and is left to reap_stranded_calls, which is
    bounded by CALL_MAX_DURATION_S and is the right owner for it.
    """
    pool = await get_pool()
    result = await pool.execute(
        "update leads set status = 'pending' "
        "where lead_id = $1 and status = 'calling' and last_called_at is null",
        lead_id,
    )
    return result.endswith("1")


async def requeue_failed_leads(*, cooldown_s: int) -> list[UUID]:
    """Return 'failed' leads that still have attempts left to the dialling pool.

    Without this, `max_attempts` is dead for exactly the calls it exists for.
    call_routes._finalise_call writes 'failed' for any call that connected but
    did not run its course, and due_leads() selects only 'pending' — so nothing
    anywhere moved a lead back, and a single provider failure retired it
    permanently at attempt 1 of 2.

    Observed live 2026-08: an ElevenLabs account in `past_due` refuses the agent
    WebSocket AFTER Plivo connects the leg, so every English/Hindi lead was
    marked 'failed' and silently removed from the campaign for good.

    *cooldown_s* keeps a provider outage from becoming a tight redial loop
    against the same people — a lead is only eligible once that long has passed
    since its last attempt. `attempts < c.max_attempts` is the same predicate
    due_leads() uses, so exhausted leads stay terminal and this can never become
    an infinite redial.
    """
    pool = await get_pool()
    rows = await pool.fetch(
        """
        update leads l set status = 'pending'
        from campaigns c
        where c.campaign_id = l.campaign_id
          and l.status = 'failed'
          and l.attempts < c.max_attempts
          and (l.last_called_at is null
               or l.last_called_at < now() - ($1::int * interval '1 second'))
        returning l.lead_id
        """,
        cooldown_s,
    )
    return [r["lead_id"] for r in rows]


async def mark_dnd(lead_id: UUID) -> str | None:
    """Flag this lead — and every OTHER lead row sharing its phone number — as
    do-not-call. Returns the phone that was opted out (None if the lead is
    gone), so the caller can push it into the Redis dial-time set.

    Cross-campaign on purpose. `leads` is unique (phone_e164, campaign_id), so
    one person enrolled in two campaigns is two rows, and due_leads' `l.dnd =
    false` filter is per ROW. Flagging only the row the operator clicked left
    the person fully dial-eligible on every other campaign — they ask to never
    be called again, and the next tick calls them from the other campaign. A
    person opts out of being called, not out of one spreadsheet row."""
    pool = await get_pool()
    row = await pool.fetchrow("select phone_e164 from leads where lead_id = $1", lead_id)
    if row is None:
        return None
    phone = row["phone_e164"]
    await pool.execute(
        "update leads set dnd = true, status = 'dnd' where phone_e164 = $1", phone
    )
    return phone


async def list_leads_for_campaign(campaign_id: UUID) -> list[Lead]:
    """Used by the admin API's per-campaign lead list view."""
    pool = await get_pool()
    rows = await pool.fetch(
        "select * from leads where campaign_id = $1 order by created_at desc", campaign_id,
    )
    return [_row_to_lead(r) for r in rows]


async def lead_status_counts(campaign_id: UUID | None = None) -> dict[str, int]:
    """Lead counts grouped by status, optionally scoped to one campaign — the
    admin dashboard's pipeline view. A status with zero leads is simply
    absent from the result; callers fill in zero for anything missing."""
    pool = await get_pool()
    rows = await pool.fetch(
        """
        select status, count(*) as n from leads
        where $1::uuid is null or campaign_id = $1
        group by status
        """,
        campaign_id,
    )
    return {r["status"]: r["n"] for r in rows}


async def lead_status_counts_for_active_campaigns() -> dict[str, int]:
    """Same idea as lead_status_counts(), scoped to active campaigns only —
    the admin dashboard's pipeline view. A dev database that's had the
    integration suite run against it accumulates hundreds of leads on
    deactivated test-*/pipeline-* campaigns (see admin/campaigns.list_campaigns'
    docstring for the identical problem solved there), which would otherwise
    dominate the numbers and make the dashboard look nothing like reality."""
    pool = await get_pool()
    rows = await pool.fetch(
        """
        select l.status, count(*) as n
        from leads l
        join campaigns c on c.campaign_id = l.campaign_id
        where c.active = true
        group by l.status
        """
    )
    return {r["status"]: r["n"] for r in rows}


async def search_leads_by_phone(phone_e164: str) -> list[Lead]:
    """Every lead across every campaign with this phone number — the admin
    API's global lookup ("someone rang back, what do we know about them?").
    Exact match, not fuzzy: callers should normalize with
    compliance.dnd.normalize_phone_e164() first, and a normalized E.164
    number is unambiguous."""
    pool = await get_pool()
    rows = await pool.fetch(
        "select * from leads where phone_e164 = $1 order by created_at desc", phone_e164,
    )
    return [_row_to_lead(r) for r in rows]


async def dnd_phones() -> list[str]:
    """Every phone number currently flagged dnd=true in Supabase — used by
    scheduler.dnd_refresh() to (re)populate Redis's dnd:numbers set. See that
    function's docstring: this is NOT a real NCPR/TRAI registry integration,
    only a propagation of flags already recorded here."""
    pool = await get_pool()
    rows = await pool.fetch("select distinct phone_e164 from leads where dnd = true")
    return [r["phone_e164"] for r in rows]


async def stale_calling_leads(older_than_s: int) -> list[UUID]:
    """Leads still marked 'calling' whose call cannot still be in progress.

    A call is bounded by CALL_MAX_DURATION_S, so a lead left in 'calling' for
    longer than that plus slack is not a live call — it is a call whose process
    died before it could record an outcome (a deploy, a crash, an OOM). Neither
    /calls/stream's finally nor /calls/hangup runs in that case, so nothing
    releases the concurrency slot or resolves the lead, and both stay stuck
    forever. See worker.reap_stranded_calls.

    last_called_at is set by record_dial_attempt when the attempt is placed, so
    it is the right clock: it exists for every attempt, answered or not.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT lead_id FROM leads
             WHERE status = 'calling'
               AND last_called_at IS NOT NULL
               AND last_called_at < now() - ($1 || ' seconds')::interval
            """,
            str(int(older_than_s)),
        )
    return [r["lead_id"] for r in rows]


async def count_calling_leads() -> int:
    """How many calls are genuinely in flight right now.

    A concurrency slot is only legitimately held while its lead is 'calling',
    so this is the ceiling calls:live:count can honestly have. See
    worker.reconcile_live_slots.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT count(*) FROM leads WHERE status = 'calling'") or 0
