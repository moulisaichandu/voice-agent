"""db/campaigns.py — CRUD for the campaigns table.

mode is admin-set here, at creation time, never inferred by an LLM per call —
see CLAUDE.md's hard rules and migrations/0001_init.sql's CHECK constraint.

AI disclosure is validated here too, at the same creation-time chokepoint —
CLAUDE.md requires it as the FIRST line of every script, enforced by a test,
not just review. See app/compliance/disclosure.py.
"""

from __future__ import annotations

from uuid import UUID

from app.compliance.disclosure import has_ai_disclosure
from app.db.models import Campaign, CampaignMode
from app.db.pool import get_pool


def _row_to_campaign(row) -> Campaign:
    return Campaign(**dict(row))


async def create_campaign(
    *, name: str, mode: CampaignMode, agent_id: str, script: str | None = None,
    max_attempts: int = 2, language: str = "auto",
) -> Campaign:
    # Validated BEFORE any DB access, deliberately: a rejected script must
    # never reach the calls table, and callers get a fast, DB-free failure.
    if mode == "oneway" and not (script and script.strip()):
        raise ValueError(
            "A one-way campaign requires a script — it's the only thing said "
            "on the call."
        )
    if script is not None and not has_ai_disclosure(script):
        raise ValueError(
            "campaign script must disclose AI involvement in its first "
            "sentence (CLAUDE.md hard rule) — e.g. start with 'This is an AI "
            "voice assistant calling on behalf of...' or a Telugu/Tinglish "
            "equivalent containing 'AI' or 'కృత్రిమ మేధ'."
        )

    pool = await get_pool()
    row = await pool.fetchrow(
        """
        insert into campaigns (name, mode, agent_id, script, max_attempts, language)
        values ($1, $2, $3, $4, $5, $6)
        returning *
        """,
        name, mode, agent_id, script, max_attempts, language,
    )
    return _row_to_campaign(row)


async def get_campaign(campaign_id: UUID) -> Campaign | None:
    pool = await get_pool()
    row = await pool.fetchrow("select * from campaigns where campaign_id = $1", campaign_id)
    return _row_to_campaign(row) if row else None


async def list_active_campaigns() -> list[Campaign]:
    pool = await get_pool()
    rows = await pool.fetch("select * from campaigns where active = true order by created_at")
    return [_row_to_campaign(r) for r in rows]


async def list_campaigns() -> list[Campaign]:
    """All campaigns, active or not — used by the admin API's campaign list
    view. list_active_campaigns() stays separate since campaign_tick's
    dial-eligibility query is intentionally scoped to active ones only."""
    pool = await get_pool()
    rows = await pool.fetch("select * from campaigns order by created_at desc")
    return [_row_to_campaign(r) for r in rows]


async def set_campaign_active(campaign_id: UUID, active: bool) -> Campaign | None:
    """Toggle whether a campaign is dial-eligible. Reversible and low-stakes —
    unlike removing a lead, deactivating a campaign touches no history; its
    leads and calls are untouched, and due_leads()/campaign_tick simply stop
    selecting them while active=false."""
    pool = await get_pool()
    row = await pool.fetchrow(
        "update campaigns set active = $2 where campaign_id = $1 returning *",
        campaign_id, active,
    )
    return _row_to_campaign(row) if row else None


async def get_campaign_by_name(name: str) -> Campaign | None:
    """Used by app/sheets/sync.py to resolve a Sheet row's "Campaign" column
    (a human-typed name) to a campaign_id. Most-recently-created wins if a
    name was reused across campaigns, matching leads.get_lead_by_phone's
    same tie-breaking convention."""
    pool = await get_pool()
    row = await pool.fetchrow(
        "select * from campaigns where name = $1 order by created_at desc limit 1", name,
    )
    return _row_to_campaign(row) if row else None
