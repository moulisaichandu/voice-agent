"""Unit tests for admin/campaigns.py — the DB layer is mocked at the
app.admin.campaigns module's imported names, so these run with no Docker.
Confirms routing/validation/error-mapping AND that no compliance gate is
bypassed (create_campaign's own ValueError still surfaces as a 422).
"""

from datetime import datetime, timezone
from uuid import uuid4

from app.admin import campaigns as admin_campaigns


def _campaign(**overrides):
    defaults = dict(
        campaign_id=uuid4(), name="Demo", mode="twoway", agent_id="agent_1",
        script=None, language="auto", max_attempts=2, active=True,
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    from app.db.models import Campaign
    return Campaign(**defaults)


def test_list_campaigns_defaults_to_active_only(client, monkeypatch):
    """A dev DB that's had the integration suite run against it holds
    hundreds of deactivated throwaway campaigns — the default view must not
    drown the real ones in them."""
    async def fake_active():
        return [_campaign()]

    async def fake_all():
        raise AssertionError("must not fetch inactive campaigns by default")

    monkeypatch.setattr(admin_campaigns.campaigns_db, "list_active_campaigns", fake_active)
    monkeypatch.setattr(admin_campaigns.campaigns_db, "list_campaigns", fake_all)

    r = client.get("/admin/campaigns")
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_list_campaigns_include_inactive_returns_everything(client, monkeypatch):
    async def fake_active():
        raise AssertionError("must not use the active-only query when asked for all")

    async def fake_all():
        return [_campaign(), _campaign(active=False), _campaign(mode="oneway")]

    monkeypatch.setattr(admin_campaigns.campaigns_db, "list_active_campaigns", fake_active)
    monkeypatch.setattr(admin_campaigns.campaigns_db, "list_campaigns", fake_all)

    r = client.get("/admin/campaigns?include_inactive=true")
    assert r.status_code == 200
    assert len(r.json()) == 3


def test_create_campaign_delegates_to_create_campaign(client, monkeypatch):
    created = {}

    async def fake_create(**kwargs):
        created.update(kwargs)
        return _campaign(**kwargs)

    monkeypatch.setattr(admin_campaigns.campaigns_db, "create_campaign", fake_create)

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

    monkeypatch.setattr(admin_campaigns.campaigns_db, "create_campaign", fake_create)

    r = client.post("/admin/campaigns", json={
        "name": "Bad Campaign", "mode": "oneway", "agent_id": "agent_x",
        "script": "Hello, we're calling about your course.",
    })
    assert r.status_code == 422
    assert "disclose AI" in r.json()["detail"]


def test_create_campaign_rejects_invalid_mode_before_reaching_the_db(client, monkeypatch):
    async def boom(**kwargs):
        raise AssertionError("should never be called — pydantic should reject first")

    monkeypatch.setattr(admin_campaigns.campaigns_db, "create_campaign", boom)

    r = client.post("/admin/campaigns", json={
        "name": "x", "mode": "sideways", "agent_id": "agent_x",
    })
    assert r.status_code == 422


# ── agent_id derivation ──────────────────────────────────────────────────────

def test_agent_id_defaults_to_the_configured_twoway_agent(client, monkeypatch):
    """The console no longer asks for an agent id — the server already knows it
    from .env, so the operator isn't hand-copying an `agent_...` string."""
    created = {}

    async def fake_create(**kwargs):
        created.update(kwargs)
        return _campaign(**kwargs)

    monkeypatch.setattr(admin_campaigns.campaigns_db, "create_campaign", fake_create)
    monkeypatch.setattr(admin_campaigns.app_config, "ELEVENLABS_TWOWAY_AGENT_ID", "agent_two")
    monkeypatch.setattr(admin_campaigns.app_config, "ELEVENLABS_ONEWAY_AGENT_ID", "agent_one")

    r = client.post("/admin/campaigns", json={"name": "Just a name"})

    assert r.status_code == 201
    assert created["agent_id"] == "agent_two"
    # mode is admin-set at creation; the default is the SAFER of the two,
    # per CLAUDE.md. Nothing infers it per call.
    assert created["mode"] == "twoway"


def test_agent_id_defaults_per_mode_and_never_crosses_over(client, monkeypatch):
    """A one-way campaign must never be dialled with the two-way agent — that
    would drop a lead into a conversation the campaign wasn't designed for."""
    created = {}

    async def fake_create(**kwargs):
        created.update(kwargs)
        return _campaign(**kwargs)

    monkeypatch.setattr(admin_campaigns.campaigns_db, "create_campaign", fake_create)
    monkeypatch.setattr(admin_campaigns.app_config, "ELEVENLABS_TWOWAY_AGENT_ID", "agent_two")
    monkeypatch.setattr(admin_campaigns.app_config, "ELEVENLABS_ONEWAY_AGENT_ID", "agent_one")

    r = client.post("/admin/campaigns", json={
        "name": "One-way run", "mode": "oneway",
        "script": "This is an AI voice assistant from Digital Brolly.",
    })

    assert r.status_code == 201
    assert created["agent_id"] == "agent_one"


def test_an_explicit_agent_id_still_wins(client, monkeypatch):
    """Kept so the API stays usable without .env agents configured."""
    created = {}

    async def fake_create(**kwargs):
        created.update(kwargs)
        return _campaign(**kwargs)

    monkeypatch.setattr(admin_campaigns.campaigns_db, "create_campaign", fake_create)
    monkeypatch.setattr(admin_campaigns.app_config, "ELEVENLABS_TWOWAY_AGENT_ID", "agent_two")

    r = client.post("/admin/campaigns", json={"name": "x", "agent_id": "agent_explicit"})

    assert r.status_code == 201
    assert created["agent_id"] == "agent_explicit"


def test_missing_agent_config_is_a_422_naming_the_variable(client, monkeypatch):
    """Never fall back to a placeholder or the other mode's agent: a campaign
    with a bogus agent_id fails at preflight with a confusing error, long after
    the operator has moved on."""
    async def boom(**kwargs):
        raise AssertionError("must not create a campaign with no resolvable agent")

    monkeypatch.setattr(admin_campaigns.campaigns_db, "create_campaign", boom)
    monkeypatch.setattr(admin_campaigns.app_config, "ELEVENLABS_TWOWAY_AGENT_ID", None)

    r = client.post("/admin/campaigns", json={"name": "No agent anywhere"})

    assert r.status_code == 422
    assert "ELEVENLABS_TWOWAY_AGENT_ID" in r.json()["detail"]


def test_a_blank_agent_id_is_treated_as_absent(client, monkeypatch):
    """An empty string from a form field must fall back to config, not be
    written to the NOT NULL agent_id column as ''."""
    created = {}

    async def fake_create(**kwargs):
        created.update(kwargs)
        return _campaign(**kwargs)

    monkeypatch.setattr(admin_campaigns.campaigns_db, "create_campaign", fake_create)
    monkeypatch.setattr(admin_campaigns.app_config, "ELEVENLABS_TWOWAY_AGENT_ID", "agent_two")

    r = client.post("/admin/campaigns", json={"name": "x", "agent_id": "   "})

    assert r.status_code == 201
    assert created["agent_id"] == "agent_two"


# ── activate/deactivate ──────────────────────────────────────────────────────

def test_deactivate_campaign_round_trips(client, monkeypatch):
    campaign = _campaign(active=True)
    captured = {}

    async def fake_set_active(campaign_id, active):
        captured["campaign_id"] = campaign_id
        captured["active"] = active
        return _campaign(campaign_id=campaign_id, active=active)

    monkeypatch.setattr(admin_campaigns.campaigns_db, "set_campaign_active", fake_set_active)

    r = client.patch(f"/admin/campaigns/{campaign.campaign_id}", json={"active": False})

    assert r.status_code == 200
    assert r.json()["active"] is False
    assert captured == {"campaign_id": campaign.campaign_id, "active": False}


def test_activate_campaign_round_trips(client, monkeypatch):
    campaign = _campaign(active=False)

    async def fake_set_active(campaign_id, active):
        return _campaign(campaign_id=campaign_id, active=active)

    monkeypatch.setattr(admin_campaigns.campaigns_db, "set_campaign_active", fake_set_active)

    r = client.patch(f"/admin/campaigns/{campaign.campaign_id}", json={"active": True})
    assert r.status_code == 200
    assert r.json()["active"] is True


def test_update_campaign_active_404s_for_an_unknown_campaign(client, monkeypatch):
    async def fake_set_active(campaign_id, active):
        return None  # set_campaign_active returns None when no row matched

    monkeypatch.setattr(admin_campaigns.campaigns_db, "set_campaign_active", fake_set_active)

    r = client.patch(f"/admin/campaigns/{uuid4()}", json={"active": False})
    assert r.status_code == 404


# ── language ─────────────────────────────────────────────────────────────────

def test_create_campaign_defaults_to_auto_language(client, monkeypatch):
    """The safety default. Every campaign created before this field existed
    reads back as 'auto', which sends no override — so nothing that works
    today changes."""
    created = {}

    async def fake_create(**kwargs):
        created.update(kwargs)
        return _campaign()

    monkeypatch.setattr(admin_campaigns.campaigns_db, "create_campaign", fake_create)
    monkeypatch.setattr(admin_campaigns.app_config, "ELEVENLABS_TWOWAY_AGENT_ID", "agent_1")

    r = client.post("/admin/campaigns", json={"name": "Demo"})
    assert r.status_code == 201
    assert created["language"] == "auto"


def test_create_campaign_persists_the_chosen_language(client, monkeypatch):
    """Tinglish routes to a one-way-only backend (see the two-way guard tests
    below), so this exercises mode='oneway' explicitly rather than the
    (two-way) default."""
    created = {}

    async def fake_create(**kwargs):
        created.update(kwargs)
        return _campaign(mode="oneway", language="tinglish")

    monkeypatch.setattr(admin_campaigns.campaigns_db, "create_campaign", fake_create)
    monkeypatch.setattr(admin_campaigns.app_config, "ELEVENLABS_ONEWAY_AGENT_ID", "agent_1")
    monkeypatch.setattr(admin_campaigns, "agent_language_support", lambda agent_id: _support())

    r = client.post("/admin/campaigns", json={
        "name": "Demo", "mode": "oneway", "language": "tinglish",
        "script": "This is an automated AI call from Digital Brolly.",
    })
    assert r.status_code == 201
    assert created["language"] == "tinglish"
    assert r.json()["language"] == "tinglish"


def test_an_unknown_language_is_rejected_at_the_boundary(client, monkeypatch):
    """422 here, not a mid-call ElevenLabs error. An unknown language must not
    reach the initiation frame — the failure would be a dropped call."""
    async def fake_create(**kwargs):
        raise AssertionError("must not reach the DB with an invalid language")

    monkeypatch.setattr(admin_campaigns.campaigns_db, "create_campaign", fake_create)
    monkeypatch.setattr(admin_campaigns.app_config, "ELEVENLABS_TWOWAY_AGENT_ID", "agent_1")

    r = client.post("/admin/campaigns", json={"name": "Demo", "language": "klingon"})
    assert r.status_code == 422


# ── language SUPPORT gate (creation time) ───────────────────────────────────
#
# ElevenLabs Agents does not support every catalogue language — Telugu isn't
# accepted at all — so an operator picking one the agent can't speak would
# otherwise only find out at the first dial, after leads are imported and
# consent recorded. This moves that failure to creation time. Mirrors
# app/telephony/preflight.py's identical, already-shipped dial-time gate,
# which stays the authoritative check and applies the same rules again.

def _support(**overrides):
    support = {"override_allowed": True, "languages": {"en", "te", "hi"},
               "tts_model": "eleven_v3_conversational"}
    support.update(overrides)
    return support


def test_create_campaign_does_not_consult_elevenlabs_for_auto(client, monkeypatch):
    """'auto' sends no override and works on any agent — it must not gain a
    new way to fail, and must not cost an API call."""
    async def fake_create(**kwargs):
        return _campaign()

    def boom(agent_id):
        raise AssertionError("must not query language support for an auto campaign")

    monkeypatch.setattr(admin_campaigns.campaigns_db, "create_campaign", fake_create)
    monkeypatch.setattr(admin_campaigns.app_config, "ELEVENLABS_TWOWAY_AGENT_ID", "agent_1")
    monkeypatch.setattr(admin_campaigns, "agent_language_support", boom)

    r = client.post("/admin/campaigns", json={"name": "Demo", "language": "auto"})
    assert r.status_code == 201


def test_create_campaign_succeeds_when_the_agent_supports_the_language(client, monkeypatch):
    """'hi' (not 'tinglish'/'te'): this exercises the ElevenLabs
    language-support-success path, which only applies to ElevenLabs-backed
    languages — te/tinglish skip it entirely (see the tests above)."""
    async def fake_create(**kwargs):
        return _campaign(language="hi")

    monkeypatch.setattr(admin_campaigns.campaigns_db, "create_campaign", fake_create)
    monkeypatch.setattr(admin_campaigns.app_config, "ELEVENLABS_TWOWAY_AGENT_ID", "agent_1")
    monkeypatch.setattr(admin_campaigns, "agent_language_support", lambda agent_id: _support())

    r = client.post("/admin/campaigns", json={"name": "Demo", "language": "hi"})
    assert r.status_code == 201


def test_create_campaign_rejects_a_language_the_agent_does_not_support(client, monkeypatch):
    async def boom(**kwargs):
        raise AssertionError("must not create a campaign for a language the agent can't speak")

    monkeypatch.setattr(admin_campaigns.campaigns_db, "create_campaign", boom)
    monkeypatch.setattr(admin_campaigns.app_config, "ELEVENLABS_TWOWAY_AGENT_ID", "agent_1")
    monkeypatch.setattr(
        admin_campaigns, "agent_language_support",
        lambda agent_id: _support(languages={"en"}),
    )

    # Hindi deliberately: ElevenLabs DOES offer Hindi, so this is the
    # "you haven't configured it yet" branch, whose advice is actionable.
    # Telugu takes a different branch — see the regression test below.
    r = client.post("/admin/campaigns", json={"name": "Demo", "language": "hi"})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "hi" in detail
    assert "Additional Languages" in detail


def test_create_campaign_rejects_when_the_override_is_disabled(client, monkeypatch):
    async def boom(**kwargs):
        raise AssertionError("must not create a campaign whose override the agent rejects")

    monkeypatch.setattr(admin_campaigns.campaigns_db, "create_campaign", boom)
    monkeypatch.setattr(admin_campaigns.app_config, "ELEVENLABS_TWOWAY_AGENT_ID", "agent_1")
    monkeypatch.setattr(
        admin_campaigns, "agent_language_support",
        lambda agent_id: _support(override_allowed=False),
    )

    r = client.post("/admin/campaigns", json={"name": "Demo", "language": "hi"})
    assert r.status_code == 422
    assert "Security" in r.json()["detail"]


def test_create_campaign_is_not_blocked_by_a_language_lookup_failure(client, monkeypatch):
    """Fail open: an ElevenLabs outage or SDK change must never be the reason
    an operator cannot create a campaign for an agent that is actually fine —
    preflight() remains the authoritative gate at dial time."""
    async def fake_create(**kwargs):
        return _campaign(language="hi")

    def boom(agent_id):
        raise RuntimeError("ElevenLabs is down")

    monkeypatch.setattr(admin_campaigns.campaigns_db, "create_campaign", fake_create)
    monkeypatch.setattr(admin_campaigns.app_config, "ELEVENLABS_TWOWAY_AGENT_ID", "agent_1")
    monkeypatch.setattr(admin_campaigns, "agent_language_support", boom)

    r = client.post("/admin/campaigns", json={"name": "Demo", "language": "hi"})
    assert r.status_code == 201


def test_check_language_support_skips_elevenlabs_entirely_for_a_non_elevenlabs_backend(
    client, monkeypatch
):
    """REGRESSION / supersedes the old "Telugu is unsupported" test. Before
    app/telephony/openai_bridge.py existed, ElevenLabs was the only backend,
    so ANY Telugu campaign was correctly rejected here — the API rejects 'te'
    outright. Now that te/tinglish route to the OpenAI Realtime backend (see
    app/languages.py's backend_for()), asking the ElevenLabs agent whether it
    speaks Telugu is asking the wrong question: that agent is never dialled
    for this call at all. Consulting it anyway would block the very campaigns
    this backend exists to carry."""
    async def fake_create(**kwargs):
        return _campaign(mode="oneway", language="te")

    # A COUNTER, not a raising stub: _check_language_support fails OPEN on an
    # exception from agent_language_support, which would swallow a raising
    # stub too and make this pass for the wrong reason (an API failure that
    # merely happened to occur) rather than for the right one (never called).
    calls = []

    def spy(agent_id):
        calls.append(agent_id)
        return _support()

    monkeypatch.setattr(admin_campaigns.campaigns_db, "create_campaign", fake_create)
    monkeypatch.setattr(admin_campaigns.app_config, "ELEVENLABS_ONEWAY_AGENT_ID", "agent_1")
    monkeypatch.setattr(admin_campaigns, "agent_language_support", spy)

    r = client.post("/admin/campaigns", json={
        "name": "Telugu Oneway", "mode": "oneway", "language": "te",
        "script": "This is an automated AI call from Digital Brolly.",
    })
    assert r.status_code == 201
    assert calls == [], "must not consult ElevenLabs for a language it never dials with"


# ── two-way requires a backend that can hold a conversation ─────────────────
#
# openai_bridge (the Telugu/Tinglish backend) fully implements two-way
# conversation, but each backend's own *_TWOWAY_ENABLED flag gates real use
# on a human having
# confirmed it on a live call first — the same discipline CLAUDE.md already
# applies to AI disclosure and to mode never being inferred. With the flag
# off (the default), routing a two-way campaign there is refused HERE, at
# creation, where the operator can act on it; with it on, the campaign is
# created like any other.

def test_create_campaign_rejects_twoway_telugu_with_a_next_step(client, monkeypatch):
    async def boom(**kwargs):
        raise AssertionError(
            "must not create a two-way campaign before the backend is enabled"
        )

    monkeypatch.setattr(admin_campaigns.campaigns_db, "create_campaign", boom)
    monkeypatch.setattr(admin_campaigns.app_config, "ELEVENLABS_TWOWAY_AGENT_ID", "agent_1")
    monkeypatch.setattr(admin_campaigns.languages_module, "SARVAM_TWOWAY_ENABLED", False)

    r = client.post("/admin/campaigns", json={
        "name": "Telugu Twoway", "mode": "twoway", "language": "te",
    })
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "two-way" in detail.lower() and "not" in detail.lower()
    assert "one-way" in detail.lower(), (
        "the operator needs a next step, not a dead end"
    )


def test_create_campaign_allows_twoway_telugu_once_the_flag_is_enabled(client, monkeypatch):
    """The flip side: the backend's two-way flag is the operator's own
    confirmation that a live call has been judged and the guard should stand
    aside — proves the flag genuinely gates the check rather than the check
    being unconditional with the flag as dead code."""
    created = {}

    async def fake_create(**kwargs):
        created.update(kwargs)
        return _campaign(mode="twoway", language="te")

    monkeypatch.setattr(admin_campaigns.campaigns_db, "create_campaign", fake_create)
    monkeypatch.setattr(admin_campaigns.app_config, "ELEVENLABS_TWOWAY_AGENT_ID", "agent_1")
    monkeypatch.setattr(admin_campaigns.languages_module, "SARVAM_TWOWAY_ENABLED", True)

    r = client.post("/admin/campaigns", json={
        "name": "Telugu Twoway", "mode": "twoway", "language": "te",
    })
    assert r.status_code == 201
    assert created["mode"] == "twoway" and created["language"] == "te"


def test_create_campaign_rejects_twoway_tinglish_too(client, monkeypatch):
    """Tinglish shares Telugu's backend — same flag-gated limitation. Pins
    the flag explicitly rather than relying on its default,
    since a real deployment's .env (loaded via load_dotenv()) can set it
    either way independent of what this test means to check."""
    async def boom(**kwargs):
        raise AssertionError("tinglish shares te's backend and its limitation")

    monkeypatch.setattr(admin_campaigns.campaigns_db, "create_campaign", boom)
    monkeypatch.setattr(admin_campaigns.app_config, "ELEVENLABS_TWOWAY_AGENT_ID", "agent_1")
    monkeypatch.setattr(admin_campaigns.languages_module, "SARVAM_TWOWAY_ENABLED", False)

    r = client.post("/admin/campaigns", json={
        "name": "Tinglish Twoway", "mode": "twoway", "language": "tinglish",
    })
    assert r.status_code == 422


def test_create_campaign_allows_oneway_telugu(client, monkeypatch):
    """The operator's next step from the rejection above must actually work."""
    async def fake_create(**kwargs):
        return _campaign(mode="oneway", language="te")

    monkeypatch.setattr(admin_campaigns.campaigns_db, "create_campaign", fake_create)
    monkeypatch.setattr(admin_campaigns.app_config, "ELEVENLABS_ONEWAY_AGENT_ID", "agent_1")

    r = client.post("/admin/campaigns", json={
        "name": "Telugu Oneway", "mode": "oneway", "language": "te",
        "script": "This is an automated AI call from Digital Brolly.",
    })
    assert r.status_code == 201


def test_create_campaign_still_allows_twoway_hindi(client, monkeypatch):
    """The production path must not move: Hindi is ElevenLabs-backed and
    two-way-capable there, so the new guard must not touch it."""
    async def fake_create(**kwargs):
        return _campaign(mode="twoway", language="hi")

    monkeypatch.setattr(admin_campaigns.campaigns_db, "create_campaign", fake_create)
    monkeypatch.setattr(admin_campaigns.app_config, "ELEVENLABS_TWOWAY_AGENT_ID", "agent_1")
    monkeypatch.setattr(
        admin_campaigns, "agent_language_support", lambda agent_id: _support()
    )

    r = client.post("/admin/campaigns", json={
        "name": "Hindi Twoway", "mode": "twoway", "language": "hi",
    })
    assert r.status_code == 201


def test_create_campaign_still_allows_twoway_auto(client, monkeypatch):
    async def fake_create(**kwargs):
        return _campaign(mode="twoway", language="auto")

    monkeypatch.setattr(admin_campaigns.campaigns_db, "create_campaign", fake_create)
    monkeypatch.setattr(admin_campaigns.app_config, "ELEVENLABS_TWOWAY_AGENT_ID", "agent_1")

    r = client.post("/admin/campaigns", json={"name": "Auto Twoway", "mode": "twoway"})
    assert r.status_code == 201


# ── which voice backend a campaign will actually run on ──────────────────────
#
# The dashboard used to compute this client-side from the language token. That
# stopped being able to tell the truth when TELUGU_BACKEND made the mapping an
# operator decision: a browser cannot see .env, so a rolled-back deployment
# would have shown every Telugu campaign running on a backend it no longer
# used. It is now computed where the answer actually lives.

def test_a_campaign_reports_the_backend_that_will_carry_it(client, monkeypatch):
    async def fake_active():
        return [_campaign(language="te"), _campaign(language="en")]

    monkeypatch.setattr(admin_campaigns.campaigns_db, "list_active_campaigns", fake_active)

    rows = client.get("/admin/campaigns").json()
    assert [r["voice_backend"] for r in rows] == ["Sarvam", "ElevenLabs"]


def test_the_reported_backend_follows_the_rollback_switch(client, monkeypatch):
    """The whole reason this moved server-side. With TELUGU_BACKEND rolled
    back, a Telugu campaign really is running on OpenAI Realtime, and an
    operator diagnosing a bad call needs the dashboard to say so."""
    async def fake_active():
        return [_campaign(language="te")]

    monkeypatch.setattr(admin_campaigns.campaigns_db, "list_active_campaigns", fake_active)
    monkeypatch.setattr(admin_campaigns.languages_module, "TELUGU_BACKEND",
                        admin_campaigns.languages_module.OPENAI_REALTIME)

    rows = client.get("/admin/campaigns").json()
    assert rows[0]["voice_backend"] == "OpenAI Realtime"
