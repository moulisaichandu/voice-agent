"""Unit tests for telephony/call_routes.py — the Plivo-facing webhooks.

These carry the security boundary (Plivo can't send APP_AUTH_TOKEN, so
CALL_WEBHOOK_SECRET is all that guards them) and the concurrency-slot
accounting, which is easy to get subtly wrong: a call can plausibly hit both
the stream-end and the hangup post, and double-releasing would let the
semaphore over-admit past MAX_CONCURRENT_CALLS.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.telephony import call_routes


class _FakeRedis:
    """SET NX + LPUSH, enough for slot-guard and write-back queue behaviour."""

    def __init__(self):
        self.store: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}

    async def set(self, key, value, nx=False, ex=None):
        if nx and key in self.store:
            return None  # real Redis returns None when NX finds the key set
        self.store[key] = value
        return True

    async def get(self, key):
        return self.store.get(key)

    async def lpush(self, key, value):
        self.lists.setdefault(key, []).insert(0, value)


@pytest.fixture
def redis(monkeypatch):
    r = _FakeRedis()
    monkeypatch.setattr(call_routes.redis_client, "get_redis", lambda: r)
    return r


@pytest.fixture
def released(monkeypatch):
    calls = []

    async def fake_release():
        calls.append(1)

    monkeypatch.setattr(call_routes.telephony_worker, "release_call_slot", fake_release)
    return calls


# ── auth ─────────────────────────────────────────────────────────────────────

def test_authorized_accepts_the_matching_token(monkeypatch):
    monkeypatch.setattr(call_routes, "CALL_WEBHOOK_SECRET", "s3cret")
    scope = SimpleNamespace(query_params={"token": "s3cret"})
    assert call_routes._authorized(scope) is True


@pytest.mark.parametrize("token", ["", "wrong", "s3cre", "s3cret "])
def test_authorized_rejects_anything_else(monkeypatch, token):
    monkeypatch.setattr(call_routes, "CALL_WEBHOOK_SECRET", "s3cret")
    scope = SimpleNamespace(query_params={"token": token})
    assert call_routes._authorized(scope) is False


def test_authorized_is_open_when_no_secret_is_configured(monkeypatch):
    """Matches the sibling project's dev behaviour. Safe only because
    app/main.py refuses to boot with a public URL and no secret."""
    monkeypatch.setattr(call_routes, "CALL_WEBHOOK_SECRET", None)
    assert call_routes._authorized(SimpleNamespace(query_params={})) is True


# ── slot accounting ──────────────────────────────────────────────────────────

async def test_slot_is_released_once_per_call_attempt(redis, released):
    """The stream ending and Plivo's hangup post can BOTH fire for one call.
    Releasing twice would let the semaphore over-admit and quietly break the
    MAX_CONCURRENT_CALLS cap."""
    lead_id = str(uuid4())
    redis.store[f"call:uuid:{lead_id}"] = "call-uuid-1"

    await call_routes._release_slot_once(lead_id)
    await call_routes._release_slot_once(lead_id)
    await call_routes._release_slot_once(lead_id)

    assert released == [1]


async def test_a_retry_of_the_same_lead_releases_its_own_slot(redis, released):
    """REGRESSION. The guard used to be keyed on lead_id with a 1h TTL, while a
    retry follows ~CAMPAIGN_TICK_SECONDS (60s) later — so attempt 2's release
    always found attempt 1's key and skipped. calls:live:count has no TTL and
    nothing resets it, so every retried lead permanently burned a slot; at
    MAX_CONCURRENT_CALLS the dialler stopped dialling entirely while still
    reporting itself healthy. Found live with 4 of 10 slots already gone."""
    lead_id = str(uuid4())

    redis.store[f"call:uuid:{lead_id}"] = "call-uuid-attempt-1"
    await call_routes._release_slot_once(lead_id)
    await call_routes._release_slot_once(lead_id)  # duplicate for attempt 1

    # The worker places a second attempt and records its new provider call id.
    redis.store[f"call:uuid:{lead_id}"] = "call-uuid-attempt-2"
    await call_routes._release_slot_once(lead_id)
    await call_routes._release_slot_once(lead_id)  # duplicate for attempt 2

    assert released == [1, 1], "each attempt must release exactly one slot"


async def test_the_recorded_uuid_beats_an_explicitly_passed_one(redis, released):
    """REGRESSION. The explicit argument used to win, on the reasoning that
    /calls/hangup gets its CallUUID straight from Plivo's form and so has the
    freshest value. But the worker records Plivo's `request_uuid` and the
    hangup form carries its `CallUUID` — two different identifiers for one
    call. Preferring the argument meant the stream's release keyed its guard
    on request_uuid while the hangup post keyed on CallUUID, so both released
    the same slot and calls:live:count over-admitted past
    MAX_CONCURRENT_CALLS. Agreement matters more than freshness."""
    lead_id = str(uuid4())
    redis.store[f"call:uuid:{lead_id}"] = "request-uuid"

    # The stream releases with no argument; the hangup post passes Plivo's
    # CallUUID. Both must resolve to the SAME guard key.
    await call_routes._release_slot_once(lead_id)
    await call_routes._release_slot_once(lead_id, call_uuid="plivo-call-uuid")

    assert released == [1]
    assert "slot:released:request-uuid" in redis.store
    assert "slot:released:plivo-call-uuid" not in redis.store


async def test_the_caller_argument_no_longer_changes_the_token(redis, released):
    """Per-attempt scoping now comes from the RECORDED id, not the argument.

    This test used to assert the opposite — that a passed call_uuid was
    preferred over lead_id when nothing was recorded, on the reasoning that it
    scopes better per attempt. It does, but only for the caller that has one:
    /calls/stream passes none. So the two teardown paths resolved different
    tokens for the same call and released the slot twice, over-admitting past
    MAX_CONCURRENT_CALLS permanently — the exact failure this guard exists to
    prevent, reintroduced by the arm meant to improve it.

    worker.process_one now records an attempt id unconditionally (falling back
    to a generated one when Plivo returns no request_uuid), so per-attempt
    scoping is preserved where it can actually be relied on, and the fallback
    is one both paths reach identically.
    """
    lead_id = str(uuid4())

    await call_routes._release_slot_once(lead_id, call_uuid="plivo-call-uuid")
    await call_routes._release_slot_once(lead_id)

    assert released == [1], "the two teardown paths released the slot twice"
    assert f"slot:released:{lead_id}" in redis.store


async def test_slot_release_falls_back_to_lead_id_with_no_attempt_id(redis, released):
    """A call that never reached /calls/answer has no recorded uuid. Degrading
    to the old lead-scoped behaviour still beats skipping the release."""
    lead_id = str(uuid4())

    await call_routes._release_slot_once(lead_id)
    await call_routes._release_slot_once(lead_id)

    assert released == [1]
    assert f"slot:released:{lead_id}" in redis.store


async def test_different_leads_each_release_their_own_slot(redis, released):
    await call_routes._release_slot_once(str(uuid4()))
    await call_routes._release_slot_once(str(uuid4()))
    assert released == [1, 1]


# ── finalising a call ────────────────────────────────────────────────────────

def _lead():
    return SimpleNamespace(lead_id=uuid4(), name="Ravi", phone_e164="+919876543210",
                           campaign_id=uuid4())


async def test_finalise_records_transcript_and_queues_writeback(redis, monkeypatch):
    lead = _lead()
    recorded = {}

    async def fake_record(**kwargs):
        recorded.update(kwargs)
        return SimpleNamespace(**kwargs)

    statuses = []

    async def fake_mark(lead_id, status):
        statuses.append(status)

    monkeypatch.setattr(call_routes.calls_db, "set_conversation_and_record", fake_record)
    monkeypatch.setattr(call_routes.leads_db, "mark_result_if_calling", fake_mark)

    await call_routes._finalise_call(lead, {
        "status": "done", "turns": 2, "transcript": [], "conversation_id": "conv_1",
    })

    assert recorded["el_conversation_id"] == "conv_1"
    assert recorded["turns"] == 2
    assert statuses == ["done"]
    assert redis.lists["sheets:writeback:queue"] == [str(lead.lead_id)]


async def test_finalise_marks_failed_when_the_bridge_did_not_complete(redis, monkeypatch):
    lead = _lead()
    statuses = []

    async def fake_record(**kwargs):
        return None

    async def fake_mark(lead_id, status):
        statuses.append(status)

    monkeypatch.setattr(call_routes.calls_db, "set_conversation_and_record", fake_record)
    monkeypatch.setattr(call_routes.leads_db, "mark_result_if_calling", fake_mark)

    await call_routes._finalise_call(lead, {
        "status": "failed", "turns": 0, "transcript": [], "conversation_id": "conv_2",
    })

    assert statuses == ["failed"]


async def test_finalise_records_the_call_even_with_no_conversation_id(redis, monkeypatch):
    """REGRESSION. This write used to be skipped when the bridge produced no
    conversation_id — which is exactly what an ElevenLabs account in `past_due`
    looks like, since the socket is refused before any
    conversation_initiation_metadata arrives. The calls row was left at
    status=NULL/ended_at=NULL forever and the Sheet was mirrored blank, so the
    failure with the most urgent cause was the one that looked like nothing had
    happened at all."""
    lead = _lead()
    recorded = {}

    async def fake_record(**kwargs):
        recorded.update(kwargs)
        return SimpleNamespace(**kwargs)

    async def fake_mark(lead_id, status):
        pass

    monkeypatch.setattr(call_routes.calls_db, "set_conversation_and_record", fake_record)
    monkeypatch.setattr(call_routes.leads_db, "mark_result_if_calling", fake_mark)

    await call_routes._finalise_call(lead, {
        "status": "failed", "turns": 0, "transcript": [], "conversation_id": None,
    })

    assert recorded, "the call row must be updated even with no conversation id"
    assert recorded["el_conversation_id"] is None
    assert recorded["status"] == "failed"


# ── ending the Plivo leg ─────────────────────────────────────────────────────

async def test_the_plivo_leg_is_hung_up_with_the_real_call_uuid(redis, monkeypatch):
    """REGRESSION. plivo_client.hangup() was written, correct, and never
    called. The answer XML sets keepCallAlive="true", so closing our WebSocket
    is not itself a hangup — a one-way call that had finished its message, or
    any call that hit CALL_MAX_DURATION_S, was left for the carrier to time
    out and billed the whole way."""
    lead_id = str(uuid4())
    redis.store[f"call:plivo_uuid:{lead_id}"] = "plivo-call-uuid"
    # The slot token is a DIFFERENT id; hanging up with it would 404.
    redis.store[f"call:uuid:{lead_id}"] = "plivo-request-uuid"
    hung_up = []

    async def fake_hangup(call_uuid):
        hung_up.append(call_uuid)

    monkeypatch.setattr(call_routes.plivo_client, "hangup", fake_hangup)

    await call_routes._end_plivo_leg(lead_id)

    assert hung_up == ["plivo-call-uuid"]


async def test_no_hangup_is_attempted_when_the_call_was_never_answered(redis, monkeypatch):
    """No CallUUID means /calls/answer never ran, so there is no leg up."""
    hung_up = []

    async def fake_hangup(call_uuid):
        hung_up.append(call_uuid)

    monkeypatch.setattr(call_routes.plivo_client, "hangup", fake_hangup)

    await call_routes._end_plivo_leg(str(uuid4()))

    assert hung_up == []


async def test_a_failed_hangup_lookup_never_raises(redis, monkeypatch):
    """This sits between recording the outcome and releasing the slot, neither
    of which may be skipped because a best-effort hangup failed."""
    async def boom(key):
        raise RuntimeError("redis down")

    monkeypatch.setattr(redis, "get", boom)

    await call_routes._end_plivo_leg(str(uuid4()))  # must not raise


async def test_releasing_the_slot_never_raises_when_redis_is_down(redis, released, monkeypatch):
    """REGRESSION: /calls/hangup's contract is 'always 200, never raises' — an
    error makes Plivo retry-storm — and /calls/stream's cleanup runs this in a
    finally. A Redis failure while releasing the slot must degrade to a logged
    skip, not an exception that escapes the handler."""
    async def boom(*a, **k):
        raise RuntimeError("redis unavailable")

    monkeypatch.setattr(redis, "get", boom)

    await call_routes._release_slot_once(str(uuid4()))  # must not raise


# ── /calls/stream: cleanup must run on EVERY exit path ───────────────────────
# keepCallAlive keeps the Plivo leg up when our socket closes, and the worker
# has already reserved a concurrency slot, so every early exit must (a) end the
# Plivo leg and (b) release the slot — or the lead is stranded 'calling', a slot
# leaks, and a silent leg bills on.


class _FakeWS:
    """The minimum WebSocket surface stream() touches."""

    def __init__(self, lead_id: str):
        self.query_params = {"lead": lead_id, "token": ""}
        self.accepted = False
        self.close_code = "unset"

    async def accept(self):
        self.accepted = True

    async def close(self, code=None):
        self.close_code = code


class _FakeRequest:
    def __init__(self, lead_id: str, form_data: dict | None = None):
        self.query_params = {"lead": lead_id, "token": ""}
        self._form = form_data or {}

    async def form(self):
        return self._form


async def test_a_db_error_resolving_the_call_still_cleans_up(redis, released, monkeypatch):
    """REGRESSION (critical): a transient DB error while the call is connecting
    must not skip cleanup. The old code caught only UUID ValueError/AttributeError
    and left get_campaign unguarded, so a real asyncpg error threw out of the
    handler before the finally that releases the slot and ends the Plivo leg —
    stranding the lead 'calling' forever and leaking a slot and a billed leg."""
    monkeypatch.setattr(call_routes, "CALL_WEBHOOK_SECRET", None)

    async def boom(_lead_id):
        raise RuntimeError("connection pool exhausted")

    monkeypatch.setattr(call_routes.leads_db, "get_lead", boom)

    hung_up = []

    async def fake_hangup(uuid):
        hung_up.append(uuid)

    monkeypatch.setattr(call_routes.plivo_client, "hangup", fake_hangup)

    lead_id = str(uuid4())
    await redis.set(call_routes.CALL_PLIVO_UUID_KEY_FMT.format(lead_id=lead_id),
                    "plivo-uuid")

    await call_routes.stream(_FakeWS(lead_id))  # must NOT raise

    assert released == [1], "the slot must be released even when the lookup errors"
    assert hung_up == ["plivo-uuid"], "the Plivo leg must be hung up too"


async def test_a_missing_lead_hangs_up_the_plivo_leg(redis, released, monkeypatch):
    """REGRESSION: a lead that vanished between /calls/answer and /calls/stream
    released its slot but never had its Plivo leg hung up — keepCallAlive keeps
    the leg up when our socket closes, so it billed on, silent."""
    monkeypatch.setattr(call_routes, "CALL_WEBHOOK_SECRET", None)

    async def no_lead(_lead_id):
        return None

    monkeypatch.setattr(call_routes.leads_db, "get_lead", no_lead)

    hung_up = []

    async def fake_hangup(uuid):
        hung_up.append(uuid)

    monkeypatch.setattr(call_routes.plivo_client, "hangup", fake_hangup)

    lead_id = str(uuid4())
    await redis.set(call_routes.CALL_PLIVO_UUID_KEY_FMT.format(lead_id=lead_id),
                    "plivo-uuid")

    await call_routes.stream(_FakeWS(lead_id))

    assert hung_up == ["plivo-uuid"], "the Plivo leg must be hung up on a missing lead"
    assert released == [1]


async def test_answer_still_serves_the_stream_xml_when_redis_is_down(redis, monkeypatch):
    """REGRESSION: recording the Plivo CallUUID is best-effort — a Redis blip
    must not fail the answer webhook (500). The call can still proceed via the
    <Stream> XML; only the later force-hangup ability is lost."""
    monkeypatch.setattr(call_routes, "CALL_WEBHOOK_SECRET", None)

    async def boom(*a, **k):
        raise RuntimeError("redis down")

    monkeypatch.setattr(redis, "set", boom)

    resp = await call_routes.answer(_FakeRequest(str(uuid4()),
                                                 form_data={"CallUUID": "plivo-uuid"}))
    assert resp.status_code == 200
    assert b"<Stream" in resp.body


async def test_a_sheets_failure_never_breaks_the_call_outcome(redis, monkeypatch):
    """CLAUDE.md: never block a call on a Sheets write. The transcript must
    still be recorded even if queueing the mirror fails."""
    lead = _lead()
    recorded = {}

    async def fake_record(**kwargs):
        recorded.update(kwargs)
        return SimpleNamespace(**kwargs)

    async def fake_mark(lead_id, status):
        pass

    async def boom(key, value):
        raise RuntimeError("redis down")

    monkeypatch.setattr(call_routes.calls_db, "set_conversation_and_record", fake_record)
    monkeypatch.setattr(call_routes.leads_db, "mark_result_if_calling", fake_mark)
    monkeypatch.setattr(redis, "lpush", boom)

    await call_routes._finalise_call(lead, {
        "status": "done", "turns": 1, "transcript": [], "conversation_id": "conv_3",
    })  # must not raise

    assert recorded["el_conversation_id"] == "conv_3"


# ── language resolution on the dial path ─────────────────────────────────────

def _lead_and_campaign(lead_language="auto", campaign_language="auto"):
    from datetime import datetime, timezone
    from uuid import uuid4

    from app.db.models import Campaign, Lead

    campaign_id = uuid4()
    lead = Lead(
        lead_id=uuid4(), phone_e164="+919876543210", campaign_id=campaign_id,
        language_pref=lead_language, created_at=datetime.now(timezone.utc),
    )
    campaign = Campaign(
        campaign_id=campaign_id, name="Demo", mode="twoway", agent_id="agent_1",
        language=campaign_language, created_at=datetime.now(timezone.utc),
    )
    return lead, campaign


def test_build_call_language_uses_the_campaign_default():
    lang, dyn = call_routes._call_language(*_lead_and_campaign(campaign_language="te"))
    assert lang == "te"
    assert dyn["language"] == "Telugu"
    assert "Telugu" in dyn["language_style"]


def test_build_call_language_lets_the_lead_override_the_campaign():
    """A mixed-language file must work without splitting it into separate
    campaigns — the row's own value wins, exactly as consent already does."""
    lang, dyn = call_routes._call_language(
        *_lead_and_campaign(lead_language="hi", campaign_language="te")
    )
    assert lang == "hi"
    assert dyn["language"] == "Hindi"


def test_build_call_language_maps_tinglish_to_telugu():
    """Speech recognition and synthesis must run as Telugu; the code-mixing
    is carried by the style text, not by the ISO code."""
    lang, dyn = call_routes._call_language(*_lead_and_campaign(campaign_language="tinglish"))
    assert lang == "te"
    assert "English" in dyn["language_style"]


def test_build_call_language_sends_nothing_for_auto():
    """The safety default: an untouched campaign adds no language variables
    and requests no override."""
    lang, dyn = call_routes._call_language(*_lead_and_campaign())
    assert lang is None
    assert dyn == {}


# ── backend dispatch ─────────────────────────────────────────────────────────
#
# The bridge FUNCTION for a call's language, not a branch: both bridges take
# identical arguments and populate the identical outcome dict, so everything
# downstream of the dial (_finalise_call, Sheets write-back, slot release)
# never learns which one ran. See app/languages.py's backend_for() for which
# languages route where.

def test_backend_bridge_picks_sarvam_for_telugu():
    assert call_routes._backend_bridge("te") is call_routes.sarvam_bridge.bridge


def test_backend_bridge_picks_sarvam_for_tinglish():
    assert call_routes._backend_bridge("tinglish") is call_routes.sarvam_bridge.bridge


@pytest.mark.parametrize("token", ["te", "tinglish"])
def test_backend_bridge_follows_the_telugu_rollback_switch(monkeypatch, token):
    """The rollback has to reach the dial path, not just the language table.
    A switch that changed what preflight validated but not which bridge ran
    would be worse than no switch at all."""
    monkeypatch.setattr(call_routes.languages, "TELUGU_BACKEND",
                        call_routes.languages.OPENAI_REALTIME)
    assert call_routes._backend_bridge(token) is call_routes.openai_bridge.bridge


def test_every_backend_a_language_can_route_to_has_a_bridge():
    """_backend_bridge falls back to the ElevenLabs bridge for anything it does
    not recognise, which is right for a hand-built token and WRONG for a real
    backend somebody forgot to wire — ElevenLabs cannot speak Telugu, so that
    fallback would dial a lead into a wall. This catches the omission here
    instead of on a live call."""
    for backend in (call_routes.languages.ELEVENLABS,
                    call_routes.languages.OPENAI_REALTIME,
                    call_routes.languages.SARVAM):
        assert backend in call_routes._BRIDGE_BY_BACKEND


@pytest.mark.parametrize("token", ["auto", "en", "hi", "hinglish"])
def test_backend_bridge_keeps_the_production_path_on_elevenlabs(token):
    """THE most important test in this task. auto/en/hi/hinglish must keep
    dialling through the ElevenLabs bridge exactly as they do today — routing
    any of them to the OpenAI backend (one-way only, see openai_bridge's
    module docstring) would silence every existing two-way campaign and every
    production one-way campaign in these languages."""
    assert call_routes._backend_bridge(token) is call_routes.bridge_module.bridge


# ── the disclosure that was actually spoken ──────────────────────────────────

def _turn(role, text):
    from app.db.models import TranscriptTurn
    return TranscriptTurn(role=role, text=text)


def test_a_spoken_telugu_disclosure_passes(caplog):
    outcome = {"transcript": [_turn("agent", "నమస్తే! ఇది కృత్రిమ మేధ ద్వారా చేసే కాల్.")]}
    assert call_routes._spoken_disclosure_ok(outcome) is True


def test_a_missing_spoken_disclosure_is_flagged():
    """The model dropped the legally-required disclosure. The call already
    happened — this is a detective control, and its job is to make sure the
    operator finds out."""
    outcome = {"transcript": [_turn("agent", "నమస్తే! మా కొత్త కోర్సు గురించి చెప్తాను.")]}
    assert call_routes._spoken_disclosure_ok(outcome) is False


def test_only_the_first_agent_turn_counts():
    """Disclosing in turn three is not disclosing. The rule is first line."""
    outcome = {"transcript": [
        _turn("agent", "నమస్తే! మా కొత్త కోర్సు గురించి చెప్తాను."),
        _turn("agent", "ఇది కృత్రిమ మేధ ద్వారా చేసే కాల్."),
    ]}
    assert call_routes._spoken_disclosure_ok(outcome) is False


def test_a_call_with_no_agent_turns_is_not_flagged():
    """No speech means no call worth judging — the lead heard nothing, so
    there is no disclosure failure to report, just a failed call."""
    assert call_routes._spoken_disclosure_ok({"transcript": []}) is True


async def test_a_failed_disclosure_check_still_records_the_call_and_queues_writeback(
    redis, monkeypatch, caplog
):
    """The whole point of a DETECTIVE control: it must never become a
    preventive one. A missing spoken disclosure must be logged loudly, but the
    call recording and the Sheets write-back queue must both still happen
    exactly as if the disclosure had passed."""
    lead = _lead()
    recorded = {}

    async def fake_record(**kwargs):
        recorded.update(kwargs)
        return SimpleNamespace(**kwargs)

    async def fake_mark(lead_id, status):
        pass

    monkeypatch.setattr(call_routes.calls_db, "set_conversation_and_record", fake_record)
    monkeypatch.setattr(call_routes.leads_db, "mark_result_if_calling", fake_mark)

    with caplog.at_level("ERROR"):
        await call_routes._finalise_call(lead, {
            "status": "done", "turns": 1, "conversation_id": "conv_4",
            "transcript": [_turn("agent", "నమస్తే! మా కొత్త కోర్సు గురించి చెప్తాను.")],
        })

    assert recorded["el_conversation_id"] == "conv_4"
    assert redis.lists["sheets:writeback:queue"] == [str(lead.lead_id)]
    assert any(
        record.levelname == "ERROR" and str(lead.lead_id) in record.message
        for record in caplog.records
    )


# ── one call must release exactly one slot ──────────────────────────────────
#
# _release_slot_once resolved its dedupe token as `recorded or call_uuid or
# lead_id`. When the recorded value is missing — Plivo returned 200 with a body
# that carried no request_uuid, or the best-effort Redis write blipped — the
# stream path (which passes no call_uuid) fell through to lead_id while the
# hangup path fell through to Plivo's CallUUID. Two different guard keys, two
# decrements of calls:live:count for one call, permanently over-admitting past
# MAX_CONCURRENT_CALLS. The docstring already argued the principle: whichever
# id wins matters far less than both paths choosing the SAME one.

async def test_stream_and_hangup_release_one_slot_when_no_attempt_id_was_recorded(
        monkeypatch):
    from app.telephony import call_routes as cr

    store: dict[str, str] = {}
    releases = []

    class _R:
        async def get(self, key):
            return store.get(key)

        async def set(self, key, value, nx=False, ex=None):
            if nx and key in store:
                return False
            store[key] = value
            return True

    async def fake_release():
        releases.append(1)

    monkeypatch.setattr(cr.redis_client, "get_redis", lambda: _R())
    monkeypatch.setattr(cr.telephony_worker, "release_call_slot", fake_release)

    # the stream's finally, which knows only the lead
    await cr._release_slot_once("lead-1")
    # then Plivo's hangup post, which carries its own CallUUID
    await cr._release_slot_once("lead-1", call_uuid="plivo-call-uuid-abc")

    assert len(releases) == 1, (
        f"one call released {len(releases)} slots — calls:live:count now "
        "over-admits permanently"
    )
