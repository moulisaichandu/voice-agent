"""Unit tests for admin/system.py — readiness, stats, config, Sheets status.

Everything I/O-touching is mocked at the app.admin.system module's imported
names, so these run with no Docker. The readiness tests are the important
ones: this endpoint exists specifically because every real blocker hit while
bringing this project up was invisible until a call failed, so its caching
and its can_dial logic are worth pinning precisely.
"""

import json
from datetime import datetime, timezone
from uuid import uuid4

from app.admin import system as admin_system


class _FakeRedis:
    def __init__(self, store: dict | None = None):
        self.store: dict[str, str] = store or {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value
        return True

    async def llen(self, key):
        return 0


class _FakePool:
    async def fetchval(self, query):
        return 1


def _campaign(**overrides):
    defaults = dict(
        campaign_id=uuid4(), name="Demo", mode="twoway", agent_id="agent_1",
        script=None, max_attempts=2, active=True, created_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    from app.db.models import Campaign
    return Campaign(**defaults)


def _mock_all_healthy(monkeypatch, *, redis: _FakeRedis | None = None) -> _FakeRedis:
    """Every dependency reports "fine" — individual tests then knock out
    exactly the one thing they're testing.

    Only stamps a default heartbeat into a FRESH redis (no `redis=` passed) —
    a caller supplying their own instance is deliberately controlling its
    exact contents (e.g. no heartbeat key, to test that failure), and
    silently adding one back would defeat the point."""
    r = redis
    if r is None:
        r = _FakeRedis()
        r.store[admin_system.worker_module.HEARTBEAT_KEY] = "2026-01-01T00:00:00+00:00"
    monkeypatch.setattr(admin_system.redis_client, "get_redis", lambda: r)

    async def redis_ok():
        return True

    monkeypatch.setattr(admin_system.redis_client, "is_available", redis_ok)

    async def fake_get_pool():
        return _FakePool()

    monkeypatch.setattr(admin_system, "get_pool", fake_get_pool)
    monkeypatch.setattr(admin_system, "within_calling_hours", lambda: True)

    for name in admin_system._REQUIRED_FOR_DIALING:
        monkeypatch.setattr(admin_system.app_config, name, "set-for-test")

    async def fake_active_campaigns():
        return [_campaign()]

    monkeypatch.setattr(admin_system.campaigns_db, "list_active_campaigns", fake_active_campaigns)

    async def fake_preflight(agent_id, language=None, mode=None):
        return None  # None = passing, per preflight()'s own contract

    monkeypatch.setattr(admin_system.preflight_module, "preflight", fake_preflight)

    def fake_subscription_status():
        return {"status": "active", "character_count": 100, "character_limit": 10_000}

    monkeypatch.setattr(admin_system.elevenlabs_client, "subscription_status",
                        fake_subscription_status)
    return r


# ── readiness: overall shape ─────────────────────────────────────────────────

def test_readiness_can_dial_true_when_everything_passes(client, monkeypatch):
    _mock_all_healthy(monkeypatch)

    r = client.get("/admin/readiness")

    assert r.status_code == 200
    body = r.json()
    assert body["can_dial"] is True
    assert {c["status"] for c in body["checks"]} == {"good"}


def test_readiness_can_dial_false_when_the_worker_has_no_heartbeat(client, monkeypatch):
    redis = _FakeRedis()  # no heartbeat key set
    _mock_all_healthy(monkeypatch, redis=redis)

    r = client.get("/admin/readiness")

    body = r.json()
    assert body["can_dial"] is False
    worker_check = next(c for c in body["checks"] if c["key"] == "worker")
    assert worker_check["status"] == "critical"
    assert "No heartbeat" in worker_check["detail"]


def test_readiness_can_dial_false_on_elevenlabs_billing_past_due(client, monkeypatch):
    """The exact failure mode hit this session: a past_due account refuses
    the agent WebSocket handshake, which nothing on our side can retry past."""
    _mock_all_healthy(monkeypatch)

    def past_due():
        return {"status": "past_due", "character_count": 100, "character_limit": 10_000}

    monkeypatch.setattr(admin_system.elevenlabs_client, "subscription_status", past_due)

    r = client.get("/admin/readiness")
    body = r.json()
    assert body["can_dial"] is False
    billing_check = next(c for c in body["checks"] if c["key"] == "billing")
    assert billing_check["status"] == "critical"
    assert "past_due" in billing_check["detail"]


def test_readiness_outside_calling_hours_is_a_warning_not_a_fault(client, monkeypatch):
    _mock_all_healthy(monkeypatch)
    monkeypatch.setattr(admin_system, "within_calling_hours", lambda: False)

    r = client.get("/admin/readiness")
    body = r.json()
    hours_check = next(c for c in body["checks"] if c["key"] == "calling_hours")
    assert hours_check["status"] == "warning"
    assert body["can_dial"] is False  # nothing WILL dial right now, honestly


def test_readiness_reports_missing_config_by_name(client, monkeypatch):
    _mock_all_healthy(monkeypatch)
    monkeypatch.setattr(admin_system.app_config, "PLIVO_FROM_NUMBER", None)

    r = client.get("/admin/readiness")
    config_check = next(c for c in r.json()["checks"] if c["key"] == "config")
    assert config_check["status"] == "critical"
    assert "PLIVO_FROM_NUMBER" in config_check["detail"]


def test_readiness_with_no_active_campaigns_warns_instead_of_running_preflight(client, monkeypatch):
    _mock_all_healthy(monkeypatch)

    async def no_campaigns():
        return []

    async def must_not_be_called(agent_id):
        raise AssertionError("must not run preflight with no campaign to check")

    monkeypatch.setattr(admin_system.campaigns_db, "list_active_campaigns", no_campaigns)
    monkeypatch.setattr(admin_system.preflight_module, "preflight", must_not_be_called)

    r = client.get("/admin/readiness")
    preflight_check = next(c for c in r.json()["checks"] if c["key"] == "preflight")
    assert preflight_check["status"] == "warning"
    assert "No active campaigns" in preflight_check["detail"]


def test_readiness_preflight_passes_the_campaign_language(client, monkeypatch):
    """_compute_preflight() used to call preflight(campaign.agent_id) with no
    language at all, so the readiness dashboard stayed green for a campaign
    misconfigured for its language — the dial path (worker.py) already passed
    it and refused correctly, but the operator never saw why leads were
    silently cycling queued -> calling -> pending.

    There is no specific lead at dashboard time, only a campaign, so this must
    resolve with the campaign's language alone (languages.resolve(None, ...)),
    the correct campaign-level question for a readiness check. A 'tinglish'
    campaign must reach preflight as the ISO code 'te', not the raw token —
    preflight only understands ISO codes.

    Also pins that the campaign's mode reaches preflight — the two-way
    dial-time guard needs it and this is the readiness dashboard's only call
    site for preflight()."""
    _mock_all_healthy(monkeypatch)

    async def fake_active_campaigns():
        return [_campaign(language="tinglish", mode="twoway")]

    monkeypatch.setattr(admin_system.campaigns_db, "list_active_campaigns", fake_active_campaigns)

    received = {}

    async def capturing_preflight(agent_id, language=None, mode=None):
        received["agent_id"] = agent_id
        received["language"] = language
        received["mode"] = mode
        return None

    monkeypatch.setattr(admin_system.preflight_module, "preflight", capturing_preflight)

    r = client.get("/admin/readiness")

    assert r.status_code == 200
    assert received["language"] == "te"
    assert received["mode"] == "twoway"


def test_readiness_database_down_is_critical(client, monkeypatch):
    _mock_all_healthy(monkeypatch)

    class _BrokenPool:
        async def fetchval(self, query):
            raise ConnectionError("connection refused")

    async def fake_get_pool():
        return _BrokenPool()

    monkeypatch.setattr(admin_system, "get_pool", fake_get_pool)

    r = client.get("/admin/readiness")
    db_check = next(c for c in r.json()["checks"] if c["key"] == "database")
    assert db_check["status"] == "critical"


# ── readiness: caching ───────────────────────────────────────────────────────

def test_readiness_caches_the_expensive_checks_across_requests(client, monkeypatch):
    """preflight and billing cost an HTTP round-trip and an ElevenLabs API
    call respectively — polled by the dashboard, they must not run on every
    single request."""
    _mock_all_healthy(monkeypatch)
    calls = {"preflight": 0, "billing": 0}

    async def counted_preflight(agent_id, language=None, mode=None):
        calls["preflight"] += 1
        return None

    def counted_billing():
        calls["billing"] += 1
        return {"status": "active", "character_count": 1, "character_limit": 2}

    monkeypatch.setattr(admin_system.preflight_module, "preflight", counted_preflight)
    monkeypatch.setattr(admin_system.elevenlabs_client, "subscription_status", counted_billing)

    client.get("/admin/readiness")
    r2 = client.get("/admin/readiness")

    assert calls == {"preflight": 1, "billing": 1}
    preflight_check = next(c for c in r2.json()["checks"] if c["key"] == "preflight")
    assert preflight_check["cached"] is True


def test_readiness_refresh_bypasses_the_cache(client, monkeypatch):
    _mock_all_healthy(monkeypatch)
    calls = {"preflight": 0}

    async def counted_preflight(agent_id, language=None, mode=None):
        calls["preflight"] += 1
        return None

    monkeypatch.setattr(admin_system.preflight_module, "preflight", counted_preflight)

    client.get("/admin/readiness")
    r = client.post("/admin/readiness/refresh")

    assert calls["preflight"] == 2
    preflight_check = next(c for c in r.json()["checks"] if c["key"] == "preflight")
    assert preflight_check["cached"] is False


def test_readiness_cache_survives_a_redis_hiccup_on_write(client, monkeypatch):
    """If caching the result fails (Redis blip), the check itself must still
    be reported — _check_redis() surfaces the outage separately."""
    _mock_all_healthy(monkeypatch)

    class _WriteBrokenRedis(_FakeRedis):
        async def set(self, key, value, ex=None):
            raise ConnectionError("redis down mid-write")

    monkeypatch.setattr(admin_system.redis_client, "get_redis", lambda: _WriteBrokenRedis())

    r = client.get("/admin/readiness")
    assert r.status_code == 200
    preflight_check = next(c for c in r.json()["checks"] if c["key"] == "preflight")
    assert preflight_check["status"] == "good"


# ── stats ────────────────────────────────────────────────────────────────────

def test_stats_combines_lead_status_counts_and_calls_today(client, monkeypatch):
    async def fake_status_counts():
        return {"pending": 3, "done": 5, "failed": 1}

    captured = {}

    async def fake_call_counts(since, active_campaigns_only=False):
        captured["active_campaigns_only"] = active_campaigns_only
        return {"total": 4, "with_transcript": 2}

    monkeypatch.setattr(admin_system.leads_db, "lead_status_counts_for_active_campaigns",
                        fake_status_counts)
    monkeypatch.setattr(admin_system.calls_db, "call_counts_since", fake_call_counts)

    r = client.get("/admin/stats")

    assert r.status_code == 200
    body = r.json()
    assert body["leads_by_status"] == {"pending": 3, "done": 5, "failed": 1}
    assert body["calls_today"] == 4
    assert body["calls_today_with_transcript"] == 2
    # Scoped to active campaigns — a dev DB's stale test-*/pipeline-*
    # campaigns must not dominate the dashboard's numbers (see the DB
    # function's own docstring; reproduced live: 378 "pending" leads and 66
    # "calls today" before this scoping was added).
    assert captured["active_campaigns_only"] is True


# ── config ───────────────────────────────────────────────────────────────────

def test_config_never_emits_a_secret_value(client, monkeypatch):
    monkeypatch.setattr(admin_system.app_config, "ELEVENLABS_API_KEY", "sk_super_secret_value")
    monkeypatch.setattr(admin_system.app_config, "PLIVO_AUTH_TOKEN", "another_secret")

    r = client.get("/admin/config")
    body = json.dumps(r.json())

    assert "sk_super_secret_value" not in body
    assert "another_secret" not in body
    api_key_var = next(v for v in r.json()["variables"] if v["name"] == "ELEVENLABS_API_KEY")
    assert api_key_var["is_set"] is True
    assert api_key_var["value"] is None


def test_config_reports_unset_secrets_as_not_set(client, monkeypatch):
    monkeypatch.setattr(admin_system.app_config, "OPENAI_API_KEY", None)

    r = client.get("/admin/config")
    var = next(v for v in r.json()["variables"] if v["name"] == "OPENAI_API_KEY")
    assert var["is_set"] is False


def test_config_reports_actual_values_for_tuning_knobs(client, monkeypatch):
    monkeypatch.setattr(admin_system.app_config, "MAX_CONCURRENT_CALLS", 7)

    r = client.get("/admin/config")
    var = next(v for v in r.json()["variables"] if v["name"] == "MAX_CONCURRENT_CALLS")
    assert var["value"] == 7


# ── sheets status ────────────────────────────────────────────────────────────

def test_sheets_status_not_configured_when_no_sheet_id(client, monkeypatch):
    monkeypatch.setattr(admin_system.app_config, "GOOGLE_SHEET_ID", None)
    monkeypatch.setattr(admin_system.redis_client, "get_redis", lambda: _FakeRedis())

    r = client.get("/admin/sheets-status")
    assert r.status_code == 200
    assert r.json()["configured"] is False


def test_sheets_status_reads_the_last_sync_from_redis(client, monkeypatch):
    monkeypatch.setattr(admin_system.app_config, "GOOGLE_SHEET_ID", "sheet-123")
    record = {"at": "2026-01-01T00:00:00+00:00", "synced": 12, "error": None}
    redis = _FakeRedis({admin_system.scheduler_module.SHEETS_LAST_SYNC_KEY: json.dumps(record)})
    monkeypatch.setattr(admin_system.redis_client, "get_redis", lambda: redis)

    r = client.get("/admin/sheets-status")
    body = r.json()
    assert body["configured"] is True
    assert body["last_synced_count"] == 12
    assert body["last_error"] is None


def test_sheets_status_reports_the_last_sync_error(client, monkeypatch):
    monkeypatch.setattr(admin_system.app_config, "GOOGLE_SHEET_ID", "sheet-123")
    record = {"at": "2026-01-01T00:00:00+00:00", "synced": None, "error": "RuntimeError: boom"}
    redis = _FakeRedis({admin_system.scheduler_module.SHEETS_LAST_SYNC_KEY: json.dumps(record)})
    monkeypatch.setattr(admin_system.redis_client, "get_redis", lambda: redis)

    r = client.get("/admin/sheets-status")
    assert r.json()["last_error"] == "RuntimeError: boom"


def test_sheets_status_handles_no_sync_having_run_yet(client, monkeypatch):
    monkeypatch.setattr(admin_system.app_config, "GOOGLE_SHEET_ID", "sheet-123")
    monkeypatch.setattr(admin_system.redis_client, "get_redis", lambda: _FakeRedis())

    r = client.get("/admin/sheets-status")
    body = r.json()
    assert body["configured"] is True
    assert body["last_sync_at"] is None
