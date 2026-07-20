"""Unit tests for admin/endpoint.py — the DB layer and scheduler are mocked
at the app.admin.endpoint module's imported names, so these run with no
Docker. Confirms: routing/validation/error-mapping behavior, AND that no
compliance gate is bypassed (create_campaign's own ValueError still surfaces,
trigger-tick still calls the real campaign_tick — not some test-only path).
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.admin import endpoint as admin_endpoint


def _campaign(**overrides):
    defaults = dict(
        campaign_id=uuid4(), name="Demo", mode="twoway", agent_id="agent_1",
        script=None, max_attempts=2, active=True, created_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    from app.db.models import Campaign
    return Campaign(**defaults)


def _lead(**overrides):
    defaults = dict(
        lead_id=uuid4(), sheet_row=None, name="Ravi", phone_e164="+919876543210",
        language_pref="auto", campaign_id=uuid4(), consent_basis=None, consent_at=None,
        dnd=False, status="pending", attempts=0, last_called_at=None,
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    from app.db.models import Lead
    return Lead(**defaults)


# ── campaigns ────────────────────────────────────────────────────────────────

def test_list_campaigns_defaults_to_active_only(client, monkeypatch):
    """A dev DB that's had the integration suite run against it holds
    hundreds of deactivated throwaway campaigns — the default view must not
    drown the real ones in them."""
    async def fake_active():
        return [_campaign()]

    async def fake_all():
        raise AssertionError("must not fetch inactive campaigns by default")

    monkeypatch.setattr(admin_endpoint.campaigns_db, "list_active_campaigns", fake_active)
    monkeypatch.setattr(admin_endpoint.campaigns_db, "list_campaigns", fake_all)

    r = client.get("/admin/campaigns")
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_list_campaigns_include_inactive_returns_everything(client, monkeypatch):
    async def fake_active():
        raise AssertionError("must not use the active-only query when asked for all")

    async def fake_all():
        return [_campaign(), _campaign(active=False), _campaign(mode="oneway")]

    monkeypatch.setattr(admin_endpoint.campaigns_db, "list_active_campaigns", fake_active)
    monkeypatch.setattr(admin_endpoint.campaigns_db, "list_campaigns", fake_all)

    r = client.get("/admin/campaigns?include_inactive=true")
    assert r.status_code == 200
    assert len(r.json()) == 3


def test_create_campaign_delegates_to_create_campaign(client, monkeypatch):
    created = {}

    async def fake_create(**kwargs):
        created.update(kwargs)
        return _campaign(**kwargs)

    monkeypatch.setattr(admin_endpoint.campaigns_db, "create_campaign", fake_create)

    r = client.post("/admin/campaigns", json={
        "name": "Test Campaign", "mode": "twoway", "agent_id": "agent_x",
    })
    assert r.status_code == 201
    assert created["name"] == "Test Campaign"
    assert created["mode"] == "twoway"


def test_create_campaign_surfaces_validation_errors_as_422_not_500(client, monkeypatch):
    """The disclosure/oneway-script checks live in db/campaigns.create_campaign
    itself (see app/compliance/disclosure.py) — this endpoint must not
    swallow or bypass that ValueError, just translate it to an HTTP error."""
    async def fake_create(**kwargs):
        raise ValueError("campaign script must disclose AI involvement in its first sentence")

    monkeypatch.setattr(admin_endpoint.campaigns_db, "create_campaign", fake_create)

    r = client.post("/admin/campaigns", json={
        "name": "Bad Campaign", "mode": "oneway", "agent_id": "agent_x",
        "script": "Hello, we're calling about your course.",
    })
    assert r.status_code == 422
    assert "disclose AI" in r.json()["detail"]


def test_create_campaign_rejects_invalid_mode_before_reaching_the_db(client, monkeypatch):
    async def boom(**kwargs):
        raise AssertionError("should never be called — pydantic should reject first")

    monkeypatch.setattr(admin_endpoint.campaigns_db, "create_campaign", boom)

    r = client.post("/admin/campaigns", json={
        "name": "x", "mode": "sideways", "agent_id": "agent_x",
    })
    assert r.status_code == 422


# ── leads ────────────────────────────────────────────────────────────────────

def test_list_leads_404s_for_an_unknown_campaign(client, monkeypatch):
    async def not_found(cid):
        return None

    monkeypatch.setattr(admin_endpoint.campaigns_db, "get_campaign", not_found)

    r = client.get(f"/admin/campaigns/{uuid4()}/leads")
    assert r.status_code == 404


def test_add_lead_normalizes_the_phone_before_storing(client, monkeypatch):
    campaign = _campaign()

    async def get_campaign(cid):
        return campaign

    captured = {}

    async def fake_upsert(**kwargs):
        captured.update(kwargs)
        return _lead(campaign_id=campaign.campaign_id, phone_e164=kwargs["phone_e164"])

    monkeypatch.setattr(admin_endpoint.campaigns_db, "get_campaign", get_campaign)
    monkeypatch.setattr(admin_endpoint.leads_db, "upsert_lead", fake_upsert)

    r = client.post(f"/admin/campaigns/{campaign.campaign_id}/leads", json={
        "phone": "09876543210", "name": "Sita",
    })
    assert r.status_code == 201
    assert captured["phone_e164"] == "+919876543210"


def test_add_lead_rejects_an_unparseable_phone_number(client, monkeypatch):
    campaign = _campaign()

    async def get_campaign(cid):
        return campaign

    monkeypatch.setattr(admin_endpoint.campaigns_db, "get_campaign", get_campaign)

    r = client.post(f"/admin/campaigns/{campaign.campaign_id}/leads", json={
        "phone": "123", "name": "Bad Number",
    })
    assert r.status_code == 400


# ── calls ────────────────────────────────────────────────────────────────────

def test_list_calls_404s_for_an_unknown_campaign(client, monkeypatch):
    async def not_found(cid):
        return None

    monkeypatch.setattr(admin_endpoint.campaigns_db, "get_campaign", not_found)

    r = client.get(f"/admin/campaigns/{uuid4()}/calls")
    assert r.status_code == 404


# ── trigger-tick ─────────────────────────────────────────────────────────────

def test_trigger_tick_calls_the_real_campaign_tick_not_a_bypass(client, monkeypatch):
    """This must be scheduler.campaign_tick() itself — the same function the
    APScheduler job calls — so calling hours/DND/consent/max_attempts are
    never skipped just because a human clicked a test button."""
    calls = {"n": 0}

    async def fake_tick():
        calls["n"] += 1
        return 3

    monkeypatch.setattr(admin_endpoint.scheduler, "campaign_tick", fake_tick)

    r = client.post("/admin/trigger-tick")
    assert r.status_code == 200
    assert r.json() == {"queued": 3}
    assert calls["n"] == 1


# ── auth ─────────────────────────────────────────────────────────────────────

def test_admin_routes_are_open_when_app_auth_token_is_unset(client, monkeypatch):
    """conftest.py sets APP_AUTH_TOKEN='' — matches .env.example's local-dev
    default. Confirms the dependency is a true no-op in that configuration."""
    async def fake_list():
        return []

    monkeypatch.setattr(admin_endpoint.campaigns_db, "list_campaigns", fake_list)
    r = client.get("/admin/campaigns")
    assert r.status_code == 200


async def test_admin_routes_require_the_bearer_token_when_configured(monkeypatch):
    monkeypatch.setattr(admin_endpoint, "APP_AUTH_TOKEN", "secret-token")

    with pytest.raises(Exception):
        # require_admin_auth is a plain function — call it directly rather
        # than spinning up a TestClient, since APP_AUTH_TOKEN is read at
        # request time (a FastAPI Header dependency), not import time.
        await admin_endpoint.require_admin_auth(authorization=None)


async def test_admin_routes_accept_the_correct_bearer_token_when_configured(monkeypatch):
    monkeypatch.setattr(admin_endpoint, "APP_AUTH_TOKEN", "secret-token")
    await admin_endpoint.require_admin_auth(authorization="Bearer secret-token")  # must not raise
