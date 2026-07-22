"""Unit tests for admin/calls.py — the DB layer and scheduler are mocked at
the app.admin.calls module's imported names, so these run with no Docker.
Confirms trigger-tick still calls the real scheduler.campaign_tick — not a
test-only bypass of any compliance gate.
"""

from datetime import datetime, timezone
from uuid import uuid4

from app.admin import calls as admin_calls


def _campaign(**overrides):
    defaults = dict(
        campaign_id=uuid4(), name="Demo", mode="twoway", agent_id="agent_1",
        script=None, max_attempts=2, active=True, created_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    from app.db.models import Campaign
    return Campaign(**defaults)


def _call(**overrides):
    defaults = dict(
        call_id=uuid4(), lead_id=uuid4(), campaign_id=uuid4(),
        el_conversation_id=None, provider_call_id="plivo_1", mode="twoway",
        status="done", turns=2, started_at=None, ended_at=None,
        transcript=None, summary=None, created_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    from app.db.models import Call
    return Call(**defaults)


# ── campaign-scoped calls ────────────────────────────────────────────────────

def test_list_calls_404s_for_an_unknown_campaign(client, monkeypatch):
    async def not_found(cid):
        return None

    monkeypatch.setattr(admin_calls.campaigns_db, "get_campaign", not_found)

    r = client.get(f"/admin/campaigns/{uuid4()}/calls")
    assert r.status_code == 404


# ── cross-campaign call history ──────────────────────────────────────────────

def test_recent_calls_returns_calls_across_campaigns(client, monkeypatch):
    calls = [_call(campaign_id=uuid4()), _call(campaign_id=uuid4())]
    captured = {}

    async def fake_recent(limit):
        captured["limit"] = limit
        return calls

    monkeypatch.setattr(admin_calls.calls_db, "list_recent_calls", fake_recent)

    r = client.get("/admin/calls")
    assert r.status_code == 200
    assert len(r.json()) == 2
    assert captured["limit"] == 50  # the default


def test_recent_calls_respects_the_limit_param(client, monkeypatch):
    captured = {}

    async def fake_recent(limit):
        captured["limit"] = limit
        return []

    monkeypatch.setattr(admin_calls.calls_db, "list_recent_calls", fake_recent)

    r = client.get("/admin/calls", params={"limit": 5})
    assert r.status_code == 200
    assert captured["limit"] == 5


def test_recent_calls_rejects_an_out_of_range_limit(client, monkeypatch):
    async def must_not_be_called(limit):
        raise AssertionError("must not query the DB for an invalid limit")

    monkeypatch.setattr(admin_calls.calls_db, "list_recent_calls", must_not_be_called)

    r = client.get("/admin/calls", params={"limit": 10_000})
    assert r.status_code == 422


# ── trigger-tick ─────────────────────────────────────────────────────────────

def test_scoped_trigger_only_ticks_that_campaign(client, monkeypatch):
    """A "dial now" button lives on one campaign's page. It must not start
    calling another campaign's leads — with real credentials that is real
    calls to people the operator never intended to contact."""
    campaign = _campaign()
    got = {}

    async def get_campaign(cid):
        return campaign

    async def fake_tick(campaign_id=None):
        got["campaign_id"] = campaign_id
        return 2

    monkeypatch.setattr(admin_calls.campaigns_db, "get_campaign", get_campaign)
    monkeypatch.setattr(admin_calls.scheduler, "campaign_tick", fake_tick)

    r = client.post(f"/admin/campaigns/{campaign.campaign_id}/trigger-tick")

    assert r.status_code == 200
    assert r.json() == {"queued": 2}
    assert got["campaign_id"] == campaign.campaign_id


def test_scoped_trigger_404s_for_an_unknown_campaign(client, monkeypatch):
    async def not_found(cid):
        return None

    async def must_not_tick(campaign_id=None):
        raise AssertionError("must not dial for a campaign that doesn't exist")

    monkeypatch.setattr(admin_calls.campaigns_db, "get_campaign", not_found)
    monkeypatch.setattr(admin_calls.scheduler, "campaign_tick", must_not_tick)

    assert client.post(f"/admin/campaigns/{uuid4()}/trigger-tick").status_code == 404


def test_trigger_tick_calls_the_real_campaign_tick_not_a_bypass(client, monkeypatch):
    """This must be scheduler.campaign_tick() itself — the same function the
    APScheduler job calls — so calling hours/DND/consent/max_attempts are
    never skipped just because a human clicked a test button."""
    calls = {"n": 0}

    async def fake_tick():
        calls["n"] += 1
        return 3

    monkeypatch.setattr(admin_calls.scheduler, "campaign_tick", fake_tick)

    r = client.post("/admin/trigger-tick")
    assert r.status_code == 200
    assert r.json() == {"queued": 3}
    assert calls["n"] == 1
