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

async def test_slot_is_released_once_per_lead(redis, released):
    """The stream ending and Plivo's hangup post can BOTH fire for one call.
    Releasing twice would let the semaphore over-admit and quietly break the
    MAX_CONCURRENT_CALLS cap."""
    lead_id = str(uuid4())

    await call_routes._release_slot_once(lead_id)
    await call_routes._release_slot_once(lead_id)
    await call_routes._release_slot_once(lead_id)

    assert released == [1]


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
    monkeypatch.setattr(call_routes.leads_db, "mark_result", fake_mark)

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
    monkeypatch.setattr(call_routes.leads_db, "mark_result", fake_mark)

    await call_routes._finalise_call(lead, {
        "status": "failed", "turns": 0, "transcript": [], "conversation_id": "conv_2",
    })

    assert statuses == ["failed"]


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
    monkeypatch.setattr(call_routes.leads_db, "mark_result", fake_mark)
    monkeypatch.setattr(redis, "lpush", boom)

    await call_routes._finalise_call(lead, {
        "status": "done", "turns": 1, "transcript": [], "conversation_id": "conv_3",
    })  # must not raise

    assert recorded["el_conversation_id"] == "conv_3"
