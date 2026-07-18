"""db/campaigns.py — CRUD for the campaigns table.

mode is admin-set here, at creation time, never inferred by an LLM per call —
see CLAUDE.md's hard rules and migrations/0001_init.sql's CHECK constraint.
"""

from __future__ import annotations

from uuid import UUID

from app.db.models import Campaign, CampaignMode
from app.db.pool import get_pool


def _row_to_campaign(row) -> Campaign:
    return Campaign(**dict(row))


async def create_campaign(
    *, name: str, mode: CampaignMode, agent_id: str, script: str | None = None,
    max_attempts: int = 2,
) -> Campaign:
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        insert into campaigns (name, mode, agent_id, script, max_attempts)
        values ($1, $2, $3, $4, $5)
        returning *
        """,
        name, mode, agent_id, script, max_attempts,
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
