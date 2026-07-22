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


async def test_an_explicit_call_uuid_is_used_when_nothing_was_recorded(redis, released):
    """The argument is still the fallback — better than degrading straight to
    lead_id scope, which can't tell one attempt from the next."""
    lead_id = str(uuid4())

    await call_routes._release_slot_once(lead_id, call_uuid="plivo-call-uuid")
    await call_routes._release_slot_once(lead_id, call_uuid="plivo-call-uuid")

    assert released == [1]
    assert "slot:released:plivo-call-uuid" in redis.store


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
