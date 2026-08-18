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

    async def set(self, key, value, ex=None, nx=False):
        # nx is not decoration: sarvam_circuit_breaker.trip() relies on
        # set-if-not-exists for its first-trip-wins semantics, so a fake that
        # ignored it would let a later, vaguer reason overwrite the original
        # and the test would never notice.
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    async def delete(self, key):
        return 1 if self.store.pop(key, None) is not None else 0

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

    # A stable tunnel. The rotation check reports a CHANGE, so "healthy" means
    # the same hostname it saw last time — which is what a named tunnel, or an
    # unrestarted cloudflared, actually looks like.
    async def stable_url():
        return "https://stable.trycloudflare.com"

    r.store[admin_system._LAST_PUBLIC_URL_KEY] = "https://stable.trycloudflare.com"
    monkeypatch.setattr(admin_system.public_url, "refresh", stable_url)

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
    """can_dial answers "is this system CAPABLE of dialling", not "will it
    dial in the next second" — outside the calling-hours window is the
    normal, correct state of a healthy system for ~14 hours of every day, and
    the calling_hours check already says so clearly on its own. can_dial must
    not also go false here, or a perfectly healthy overnight system becomes
    indistinguishable from a genuinely broken one — the exact defect this
    fix removes. See the can_dial comment in app/admin/system.py."""
    _mock_all_healthy(monkeypatch)
    monkeypatch.setattr(admin_system, "within_calling_hours", lambda: False)

    r = client.get("/admin/readiness")
    body = r.json()
    hours_check = next(c for c in body["checks"] if c["key"] == "calling_hours")
    assert hours_check["status"] == "warning"
    assert body["can_dial"] is True


def test_can_dial_true_when_one_campaign_is_blocked_among_several(client, monkeypatch):
    """One misconfigured campaign among several healthy ones does not stop
    the SYSTEM from dialling — the other campaigns still can. This is the
    operator's actual reported case: a 'twoway'-in-'te' test campaign made
    the whole dashboard read can_dial: false even though two other campaigns,
    and every infrastructure check, were fine."""
    _mock_all_healthy(monkeypatch)

    async def three_campaigns():
        return [
            _campaign(name="Healthy One", agent_id="agent_1", mode="oneway"),
            _campaign(name="Healthy Two", agent_id="agent_2", mode="oneway"),
            _campaign(name="Broken Telugu", agent_id="agent_3", mode="twoway", language="te"),
        ]

    monkeypatch.setattr(admin_system.campaigns_db, "list_active_campaigns", three_campaigns)

    async def fails_only_broken(agent_id, language=None, mode=None):
        if agent_id == "agent_3":
            return "twoway in te is not supported by this backend"
        return None

    monkeypatch.setattr(admin_system.preflight_module, "preflight", fails_only_broken)

    r = client.get("/admin/readiness")
    body = r.json()
    preflight_check = next(c for c in body["checks"] if c["key"] == "preflight")
    assert preflight_check["status"] == "warning"
    assert body["can_dial"] is True


def test_can_dial_false_when_every_active_campaign_is_blocked(client, monkeypatch):
    """Contrast with the mixed case above: when NOTHING can dial, can_dial
    must still say so — this fix narrows what gates it, it does not disable
    the gate."""
    _mock_all_healthy(monkeypatch)

    async def two_campaigns():
        return [
            _campaign(name="One", agent_id="agent_1"),
            _campaign(name="Two", agent_id="agent_2"),
        ]

    monkeypatch.setattr(admin_system.campaigns_db, "list_active_campaigns", two_campaigns)

    async def always_fails(agent_id, language=None, mode=None):
        return "tunnel is down"

    monkeypatch.setattr(admin_system.preflight_module, "preflight", always_fails)

    r = client.get("/admin/readiness")
    body = r.json()
    assert body["can_dial"] is False


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


def test_preflight_all_active_campaigns_passing_is_good(client, monkeypatch):
    _mock_all_healthy(monkeypatch)

    async def three_campaigns():
        return [
            _campaign(name="Alpha", agent_id="agent_1", mode="twoway"),
            _campaign(name="Beta", agent_id="agent_2", mode="oneway"),
            _campaign(name="Gamma", agent_id="agent_3", mode="oneway"),
        ]

    monkeypatch.setattr(admin_system.campaigns_db, "list_active_campaigns", three_campaigns)

    async def always_passes(agent_id, language=None, mode=None):
        return None

    monkeypatch.setattr(admin_system.preflight_module, "preflight", always_passes)

    r = client.get("/admin/readiness")
    preflight_check = next(c for c in r.json()["checks"] if c["key"] == "preflight")
    assert preflight_check["status"] == "good"
    assert "3" in preflight_check["detail"]


def test_preflight_mixed_pass_fail_is_a_warning_naming_the_blocked_campaign(client, monkeypatch):
    """The operator's actual reported case: one misconfigured campaign among
    several healthy ones must read as 'one campaign needs attention', not as
    a system-wide failure."""
    _mock_all_healthy(monkeypatch)

    async def three_campaigns():
        return [
            _campaign(name="Healthy One", agent_id="agent_1", mode="oneway"),
            _campaign(name="Healthy Two", agent_id="agent_2", mode="oneway"),
            _campaign(name="Broken Telugu", agent_id="agent_3", mode="twoway", language="te"),
        ]

    monkeypatch.setattr(admin_system.campaigns_db, "list_active_campaigns", three_campaigns)

    async def fails_only_broken(agent_id, language=None, mode=None):
        if agent_id == "agent_3":
            return ("This campaign is 'twoway' in 'te', but that language's voice "
                    "backend (OpenAI Realtime) only supports one-way calls today")
        return None

    monkeypatch.setattr(admin_system.preflight_module, "preflight", fails_only_broken)

    r = client.get("/admin/readiness")
    body = r.json()
    preflight_check = next(c for c in body["checks"] if c["key"] == "preflight")
    assert preflight_check["status"] == "warning"
    assert "Broken Telugu" in preflight_check["detail"]
    assert "2" in preflight_check["detail"]  # 2 of 3 can still dial
    # can_dial semantics for this exact scenario are pinned separately below,
    # by test_can_dial_true_when_one_campaign_is_blocked_among_several
    # (Fix 2) — this test is about the preflight check's own status/detail.


def test_preflight_all_active_campaigns_failing_same_reason_states_it_once(client, monkeypatch):
    _mock_all_healthy(monkeypatch)

    async def three_campaigns():
        return [
            _campaign(name="One", agent_id="agent_1"),
            _campaign(name="Two", agent_id="agent_2"),
            _campaign(name="Three", agent_id="agent_3"),
        ]

    monkeypatch.setattr(admin_system.campaigns_db, "list_active_campaigns", three_campaigns)

    async def all_fail_same_way(agent_id, language=None, mode=None):
        return "PUBLIC_BASE_URL is unreachable (ConnectError). Is the tunnel running?"

    monkeypatch.setattr(admin_system.preflight_module, "preflight", all_fail_same_way)

    r = client.get("/admin/readiness")
    preflight_check = next(c for c in r.json()["checks"] if c["key"] == "preflight")
    assert preflight_check["status"] == "critical"
    # Stated once, not once per campaign — three campaigns failing the same
    # way must not produce three repeats of an infrastructure-level message.
    assert preflight_check["detail"].count("tunnel running") == 1
    assert r.json()["can_dial"] is False


def test_preflight_one_active_campaign_blocked_names_it_not_infrastructure(client, monkeypatch):
    """The exact operator report this fix exists for: ONE active campaign,
    misconfigured as 'twoway' in 'te'. "All active campaigns fail" and "my
    one campaign is misconfigured" are the same observation when there is
    only one campaign — the count carries no evidence of a shared cause, so
    the message must point at the campaign, not assert an infrastructure
    fault."""
    _mock_all_healthy(monkeypatch)

    async def one_campaign():
        return [_campaign(name="Broken Telugu", agent_id="agent_3", mode="twoway",
                          language="te")]

    monkeypatch.setattr(admin_system.campaigns_db, "list_active_campaigns", one_campaign)

    async def fails(agent_id, language=None, mode=None):
        return ("This campaign is 'twoway' in 'te', but that language's voice backend "
                "(OpenAI Realtime) only supports one-way calls today")

    monkeypatch.setattr(admin_system.preflight_module, "preflight", fails)

    r = client.get("/admin/readiness")
    body = r.json()
    preflight_check = next(c for c in body["checks"] if c["key"] == "preflight")
    assert preflight_check["status"] == "critical"
    assert "Broken Telugu" in preflight_check["detail"]
    assert "infrastructure" not in preflight_check["detail"].lower()
    assert body["can_dial"] is False


def test_preflight_many_campaigns_sharing_one_config_failing_is_not_infrastructure(
    client, monkeypatch,
):
    """Several campaigns that all share the SAME (agent_id, language, mode)
    are, for evidence purposes, one configuration wearing several names —
    "all of them fail" still carries no information about a shared cause,
    because there is only one distinct configuration under test. Must not be
    reported as infrastructure."""
    _mock_all_healthy(monkeypatch)

    async def five_campaigns_same_shape():
        return [_campaign(name=f"Clone {i}", agent_id="agent_1", mode="oneway")
                for i in range(5)]

    monkeypatch.setattr(admin_system.campaigns_db, "list_active_campaigns",
                        five_campaigns_same_shape)

    async def fails(agent_id, language=None, mode=None):
        return "misconfigured agent_id"

    monkeypatch.setattr(admin_system.preflight_module, "preflight", fails)

    r = client.get("/admin/readiness")
    body = r.json()
    preflight_check = next(c for c in body["checks"] if c["key"] == "preflight")
    assert preflight_check["status"] == "critical"
    assert "infrastructure" not in preflight_check["detail"].lower()
    assert "misconfigured agent_id" in preflight_check["detail"]
    assert body["can_dial"] is False


def test_preflight_different_configs_same_reason_may_claim_infrastructure(client, monkeypatch):
    """Contrast with the two tests above: when campaigns with DIFFERENT
    (agent_id, language, mode) configurations all fail for the exact same
    reason, that reason cannot be caused by any one campaign's own
    configuration — that IS evidence of a shared/infrastructure cause, and
    is safe to describe as such."""
    _mock_all_healthy(monkeypatch)

    async def three_distinct_campaigns():
        return [
            _campaign(name="One", agent_id="agent_1", mode="oneway"),
            _campaign(name="Two", agent_id="agent_2", mode="twoway"),
            _campaign(name="Three", agent_id="agent_3", mode="oneway", language="te"),
        ]

    monkeypatch.setattr(admin_system.campaigns_db, "list_active_campaigns",
                        three_distinct_campaigns)

    async def all_fail_same_way(agent_id, language=None, mode=None):
        return "PUBLIC_BASE_URL is unreachable (ConnectError). Is the tunnel running?"

    monkeypatch.setattr(admin_system.preflight_module, "preflight", all_fail_same_way)

    r = client.get("/admin/readiness")
    body = r.json()
    preflight_check = next(c for c in body["checks"] if c["key"] == "preflight")
    assert preflight_check["status"] == "critical"
    assert "infrastructure" in preflight_check["detail"].lower()
    assert body["can_dial"] is False


def test_preflight_dedups_shared_configuration_into_one_call(client, monkeypatch):
    """Many campaigns commonly share the same agent/language/mode — the dev
    DB scenario CLAUDE.md and app/admin/campaigns.py's list_campaigns
    docstring both call out (hundreds of throwaway test-*/pipeline-*
    campaigns). preflight() is a real ElevenLabs + HTTP round-trip; it must
    run once per distinct (agent_id, language, mode), not once per campaign."""
    _mock_all_healthy(monkeypatch)

    async def five_campaigns_same_shape():
        return [_campaign(name=f"Clone {i}", agent_id="agent_1", mode="oneway")
                for i in range(5)]

    monkeypatch.setattr(admin_system.campaigns_db, "list_active_campaigns",
                        five_campaigns_same_shape)

    calls = {"n": 0}

    async def counted(agent_id, language=None, mode=None):
        calls["n"] += 1
        return None

    monkeypatch.setattr(admin_system.preflight_module, "preflight", counted)

    r = client.get("/admin/readiness")
    assert calls["n"] == 1
    preflight_check = next(c for c in r.json()["checks"] if c["key"] == "preflight")
    assert preflight_check["status"] == "good"
    assert "5" in preflight_check["detail"]


def test_preflight_truncates_beyond_the_configuration_cap_and_says_so(client, monkeypatch):
    """Bounds the work: a database full of active campaigns with genuinely
    distinct configurations must not make this endpoint hang making dozens of
    real preflight() round-trips. A truncation must be visible in the
    detail — a silent cap would read as 'everything checked out fine'."""
    _mock_all_healthy(monkeypatch)
    monkeypatch.setattr(admin_system, "_MAX_PREFLIGHT_CONFIGURATIONS", 2)

    async def four_distinct_campaigns():
        return [_campaign(name=f"C{i}", agent_id=f"agent_{i}", mode="oneway")
                for i in range(4)]

    monkeypatch.setattr(admin_system.campaigns_db, "list_active_campaigns",
                        four_distinct_campaigns)

    calls = {"n": 0}

    async def counted(agent_id, language=None, mode=None):
        calls["n"] += 1
        return None

    monkeypatch.setattr(admin_system.preflight_module, "preflight", counted)

    r = client.get("/admin/readiness")
    assert calls["n"] == 2  # capped at _MAX_PREFLIGHT_CONFIGURATIONS
    preflight_check = next(c for c in r.json()["checks"] if c["key"] == "preflight")
    assert "not checked" in preflight_check["detail"].lower()
    assert "2" in preflight_check["detail"]  # the 2 skipped campaigns


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


def test_readiness_recomputes_malformed_cached_check(client, monkeypatch):
    redis = _FakeRedis({"readiness:preflight": "not-json"})
    _mock_all_healthy(monkeypatch, redis=redis)

    r = client.get("/admin/readiness")

    assert r.status_code == 200
    preflight_check = next(c for c in r.json()["checks"] if c["key"] == "preflight")
    assert preflight_check["status"] == "good"
    assert preflight_check["cached"] is False


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


def test_sheets_status_does_not_500_on_malformed_cache(client, monkeypatch):
    monkeypatch.setattr(admin_system.app_config, "GOOGLE_SHEET_ID", "sheet-123")
    redis = _FakeRedis({admin_system.scheduler_module.SHEETS_LAST_SYNC_KEY: "not-json"})
    monkeypatch.setattr(admin_system.redis_client, "get_redis", lambda: redis)

    r = client.get("/admin/sheets-status")

    assert r.status_code == 200
    assert "malformed" in r.json()["last_error"]


async def test_billing_check_reuses_a_cache_warmed_by_elevenlabs_client_directly(
    client, monkeypatch
):
    """THE property the whole move exists to guarantee: preflight.py and the
    readiness dashboard must not each pay for their own ElevenLabs round-trip
    inside the same cache window. Before the move they could not share one —
    the caching lived in admin/system.py, where preflight cannot reach it."""
    redis = _FakeRedis()
    _mock_all_healthy(monkeypatch, redis=redis)
    calls = {"n": 0}

    def counted():
        calls["n"] += 1
        return {"status": "active", "character_count": 1, "character_limit": 2}

    monkeypatch.setattr(admin_system.elevenlabs_client, "subscription_status", counted)

    # Simulates preflight.py having warmed the cache moments earlier.
    await admin_system.elevenlabs_client.cached_subscription_status()
    assert calls["n"] == 1

    r = client.get("/admin/readiness")

    assert calls["n"] == 1, "the readiness dashboard must reuse the warm cache"
    billing_check = next(c for c in r.json()["checks"] if c["key"] == "billing")
    assert billing_check["status"] == "good"


# ── readiness: the Sarvam circuit breaker ────────────────────────────────────
#
# Tonight's whole problem was that this state was invisible until a real call
# failed. The dashboard is where "can this system dial right now, and if not
# why" is supposed to be answerable.

async def test_readiness_can_dial_false_when_sarvam_circuit_breaker_is_tripped(
    client, monkeypatch
):
    """The reactive half of the design: a call that already failed with
    Sarvam's "insufficient credits" signal must stop every later Sarvam-backed
    dial until an admin clears it."""
    redis = _FakeRedis()
    redis.store[admin_system.worker_module.HEARTBEAT_KEY] = "2026-01-01T00:00:00+00:00"
    _mock_all_healthy(monkeypatch, redis=redis)
    await admin_system.sarvam_circuit_breaker.trip("Insufficient credits")

    r = client.get("/admin/readiness")

    body = r.json()
    assert body["can_dial"] is False
    check = next(c for c in body["checks"] if c["key"] == "sarvam_circuit_breaker")
    assert check["status"] == "critical"
    assert "Insufficient credits" in check["detail"]


async def test_the_tripped_check_says_how_to_clear_it(client, monkeypatch):
    """Manual-clear-only means the dashboard has to carry the way out, or an
    operator is shown a red light with no switch."""
    redis = _FakeRedis()
    redis.store[admin_system.worker_module.HEARTBEAT_KEY] = "2026-01-01T00:00:00+00:00"
    _mock_all_healthy(monkeypatch, redis=redis)
    await admin_system.sarvam_circuit_breaker.trip("Insufficient credits")

    r = client.get("/admin/readiness")

    check = next(c for c in r.json()["checks"] if c["key"] == "sarvam_circuit_breaker")
    assert "sarvam-circuit-breaker/clear" in check["detail"]
    assert "dashboard.sarvam.ai" in check["detail"]


async def test_readiness_sarvam_circuit_breaker_good_when_not_tripped(
    client, monkeypatch
):
    _mock_all_healthy(monkeypatch)

    r = client.get("/admin/readiness")

    check = next(c for c in r.json()["checks"] if c["key"] == "sarvam_circuit_breaker")
    assert check["status"] == "good"


async def test_an_unreadable_breaker_is_reported_not_swallowed(client, monkeypatch):
    """A Redis failure must not take down the whole dashboard, and must not
    quietly read as "not tripped" either — the dashboard exists to answer this
    question, so "I could not tell" is the honest answer."""
    _mock_all_healthy(monkeypatch)

    async def boom():
        raise ConnectionError("redis is down")

    monkeypatch.setattr(admin_system.sarvam_circuit_breaker, "tripped_reason", boom)

    r = client.get("/admin/readiness")

    assert r.status_code == 200
    check = next(c for c in r.json()["checks"] if c["key"] == "sarvam_circuit_breaker")
    assert check["status"] == "critical"
    assert "Could not check" in check["detail"]


async def test_clear_sarvam_circuit_breaker_endpoint_clears_it(client, monkeypatch):
    redis = _FakeRedis()
    monkeypatch.setattr(admin_system.redis_client, "get_redis", lambda: redis)
    await admin_system.sarvam_circuit_breaker.trip("Insufficient credits")

    r = client.post("/admin/system/sarvam-circuit-breaker/clear")

    assert r.status_code == 200
    assert await admin_system.sarvam_circuit_breaker.tripped_reason() is None


async def test_clearing_an_untripped_breaker_is_harmless(client, monkeypatch):
    """An operator who clicks it twice, or clears one that recovered on its
    own, must not get an error back."""
    redis = _FakeRedis()
    monkeypatch.setattr(admin_system.redis_client, "get_redis", lambda: redis)

    r = client.post("/admin/system/sarvam-circuit-breaker/clear")

    assert r.status_code == 200
    assert await admin_system.sarvam_circuit_breaker.tripped_reason() is None


# ── the tunnel rotates in one direction only ────────────────────────────────
#
# public_url.refresh() re-resolves the hostname this app hands to Plivo, so
# dialling keeps working and preflight stays green after a rotation. But the
# ElevenLabs agent's course-lookup tool URL and its transcript webhook are
# configured against that same rotating hostname in the ElevenLabs dashboard,
# and nothing re-points or checks them. The failure is silent and specific:
# calls connect, the dashboard is green, and the agent simply cannot answer
# questions about the courses. The module's own docstring records 22 distinct
# hostnames minted on this machine.

async def test_a_rotated_tunnel_is_reported_rather_than_passing_silently(monkeypatch):
    from app.admin import system as sys_mod

    seen = {"stored": "https://old-hostname.trycloudflare.com"}

    class _R:
        async def get(self, key):
            return seen["stored"]

        async def set(self, key, value, **kw):
            seen["stored"] = value
            return True

    async def resolve():
        return "https://brand-new-hostname.trycloudflare.com"

    monkeypatch.setattr(sys_mod.redis_client, "get_redis", lambda: _R())
    monkeypatch.setattr(sys_mod.public_url, "refresh", resolve)

    check = await sys_mod._check_public_url_rotation()

    assert check["status"] == "warning"
    assert "ElevenLabs" in check["detail"]
    assert "brand-new-hostname" in check["detail"]
    # ...and the new value is remembered, so it reports once, not forever.
    assert seen["stored"] == "https://brand-new-hostname.trycloudflare.com"


async def test_an_unchanged_tunnel_is_quiet(monkeypatch):
    from app.admin import system as sys_mod

    class _R:
        async def get(self, key):
            return "https://same-hostname.trycloudflare.com"

        async def set(self, key, value, **kw):
            return True

    async def resolve():
        return "https://same-hostname.trycloudflare.com"

    monkeypatch.setattr(sys_mod.redis_client, "get_redis", lambda: _R())
    monkeypatch.setattr(sys_mod.public_url, "refresh", resolve)

    assert (await sys_mod._check_public_url_rotation())["status"] == "good"
