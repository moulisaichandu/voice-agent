"""Unit tests for app/db/campaigns.py's creation-time validation. Both checks
run BEFORE the DB pool is touched, so these raise without a real database —
the happy path (a campaign actually gets inserted) is covered by
tests/integration/test_db.py, which needs real Postgres.
"""

import pytest

from app.db import campaigns as campaigns_db


async def test_oneway_campaign_without_a_script_is_rejected():
    with pytest.raises(ValueError, match="requires a script"):
        await campaigns_db.create_campaign(name="x", mode="oneway", agent_id="agent_1")


async def test_oneway_campaign_with_a_blank_script_is_rejected():
    with pytest.raises(ValueError, match="requires a script"):
        await campaigns_db.create_campaign(
            name="x", mode="oneway", agent_id="agent_1", script="   ",
        )


async def test_script_without_ai_disclosure_is_rejected():
    with pytest.raises(ValueError, match="disclose AI"):
        await campaigns_db.create_campaign(
            name="x", mode="oneway", agent_id="agent_1",
            script="Namaste! We're calling about the course.",
        )


async def test_twoway_campaign_with_no_script_is_not_blocked_by_disclosure_check(monkeypatch):
    """mode=twoway doesn't require a script at all — CLAUDE.md's disclosure
    rule only bites when a script is actually stored (script is nullable and,
    per its own SQL comment, a 'one-way message template'). Proven by
    monkeypatching get_pool to a sentinel failure: reaching it means
    validation let this call through rather than raising ValueError first."""
    async def sentinel_pool():
        raise RuntimeError("sentinel: reached the DB layer")

    monkeypatch.setattr(campaigns_db, "get_pool", sentinel_pool)
    with pytest.raises(RuntimeError, match="sentinel"):
        await campaigns_db.create_campaign(name="x", mode="twoway", agent_id="agent_1")
