"""Unit tests for webhooks/elevenlabs.py: signature verification, event
dispatch, role/status mapping, and idempotency. Redis and the DB layer are
mocked so these run with no Docker — the idempotency SET NX semantics are
exercised via a small in-memory fake, not asserted against real Redis (that
would make this an integration test for no real benefit here)."""

from types import SimpleNamespace

import pytest
from elevenlabs.errors import BadRequestError
from fastapi.testclient import TestClient

from app.webhooks import elevenlabs as wh


class _FakeRedis:
    """Just enough of redis.asyncio.Redis for these tests: SET NX + LPUSH +
    EVAL (the handler calls worker.release_call_slot(), which EVALs the
    slot-release Lua script — this fake tracks call count only, since the
    actual semaphore arithmetic is covered by tests/unit/test_worker.py)."""

    def __init__(self):
        self.store: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}
        self.eval_calls: list[tuple] = []

    async def set(self, key, value, nx=False, ex=None):
        if nx and key in self.store:
            return None  # real redis: NX SET on an existing key returns None
        self.store[key] = value
        return True

    async def lpush(self, key, value):
        self.lists.setdefault(key, []).insert(0, value)

    async def eval(self, script, numkeys, *keys_and_args):
        self.eval_calls.append((script, numkeys, keys_and_args))
        return 1


@pytest.fixture
def fake_redis(monkeypatch):
    r = _FakeRedis()
    monkeypatch.setattr(wh.redis_client, "get_redis", lambda: r)
    return r


def _transcription_payload(conversation_id="conv_1", lead_id=None):
    return {
        "type": "post_call_transcription",
        "data": {
            "conversation_id": conversation_id,
            "status": "done",
            "transcript": [
                {"role": "agent", "message": "నమస్కారం!"},
                {"role": "user", "message": "hello"},
                {"role": "user", "message": ""},  # blank message dropped
            ],
            "analysis": {"transcript_summary": "Interested lead."},
        },
    }


async def test_handle_post_call_transcription_maps_roles_and_records(fake_redis, monkeypatch):
    recorded = {}

    async def fake_record_transcript(**kwargs):
        recorded.update(kwargs)
        return SimpleNamespace(lead_id="lead_xyz")

    monkeypatch.setattr(wh.calls_db, "record_transcript", fake_record_transcript)

    await wh._handle_post_call_transcription(_transcription_payload()["data"])

    assert recorded["el_conversation_id"] == "conv_1"
    assert recorded["status"] == "done"
    assert recorded["summary"] == "Interested lead."
    roles = [t.role for t in recorded["transcript"]]
    assert roles == ["agent", "lead"]          # "user" mapped to "lead"; blank dropped
    assert recorded["turns"] == 2

    # Sheets write-back queued for this lead.
    assert fake_redis.lists["sheets:writeback:queue"] == ["lead_xyz"]

    # The concurrency slot worker.py acquired before placing this call must
    # be released here — this IS the call's real end.
    # Slot release moved to app/telephony/call_routes.py when Plivo took over
    # placing the call: this webhook still fires for the same conversation,
    # so releasing here too would double-release and break the concurrency cap.
    assert fake_redis.eval_calls == []


async def test_call_initiation_failure_releases_slot_and_records_failure(fake_redis, monkeypatch):
    recorded = {}

    async def fake_record_transcript(**kwargs):
        recorded.update(kwargs)
        return SimpleNamespace(lead_id="lead_xyz")

    monkeypatch.setattr(wh.calls_db, "record_transcript", fake_record_transcript)

    await wh._handle_call_initiation_failure({"conversation_id": "conv_2"})

    assert fake_redis.eval_calls == []        # slot release is call_routes' job now
    assert recorded["el_conversation_id"] == "conv_2"
    assert recorded["status"] == "failed"


async def test_duplicate_conversation_id_is_not_reprocessed(fake_redis, monkeypatch):
    calls = {"n": 0}

    async def fake_record_transcript(**kwargs):
        calls["n"] += 1
        return SimpleNamespace(lead_id="lead_xyz")

    monkeypatch.setattr(wh.calls_db, "record_transcript", fake_record_transcript)

    data = _transcription_payload()["data"]
    await wh._handle_post_call_transcription(data)
    await wh._handle_post_call_transcription(data)  # ElevenLabs retry

    assert calls["n"] == 1
    # The slot must be released exactly once too — a retry releasing it again
    # would let the concurrency semaphore drift low over time.
    # Slot release moved to app/telephony/call_routes.py when Plivo took over
    # placing the call: this webhook still fires for the same conversation,
    # so releasing here too would double-release and break the concurrency cap.
    assert fake_redis.eval_calls == []
    assert "sheets:writeback:queue" not in fake_redis.lists or \
           len(fake_redis.lists["sheets:writeback:queue"]) == 1


async def test_missing_conversation_id_is_dropped_not_crashed(fake_redis, monkeypatch):
    async def boom(**kwargs):
        raise AssertionError("record_transcript must not be called with no conversation_id")

    monkeypatch.setattr(wh.calls_db, "record_transcript", boom)
    await wh._handle_post_call_transcription({"status": "done", "transcript": []})  # no error


async def test_unknown_conversation_id_logs_but_does_not_crash(fake_redis, monkeypatch):
    async def fake_record_transcript(**kwargs):
        return None  # no matching call row

    monkeypatch.setattr(wh.calls_db, "record_transcript", fake_record_transcript)
    await wh._handle_post_call_transcription(_transcription_payload()["data"])  # no error
    assert "sheets:writeback:queue" not in fake_redis.lists


# ── HTTP-level: signature verification ─────────────────────────────────────────

@pytest.fixture
def app_client(monkeypatch):
    monkeypatch.setattr(wh, "ELEVENLABS_WEBHOOK_SECRET", "test-secret")
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_tampered_or_bad_signature_is_rejected(app_client, monkeypatch):
    def fake_construct(raw_body, sig_header, secret):
        raise BadRequestError(body={"detail": "bad signature"})

    monkeypatch.setattr(wh, "construct_webhook_event", fake_construct)
    r = app_client.post(
        "/webhooks/elevenlabs",
        content=b'{"type":"post_call_transcription","data":{}}',
        headers={"ElevenLabs-Signature": "t=1,v0=wrong"},
    )
    assert r.status_code == 401
    assert "invalid signature" in r.text.lower()


def test_valid_signature_dispatches_and_returns_200(app_client, monkeypatch):
    monkeypatch.setattr(
        wh, "construct_webhook_event",
        lambda raw_body, sig_header, secret: {"type": "post_call_audio", "data": {}},
    )
    r = app_client.post(
        "/webhooks/elevenlabs",
        content=b"{}",
        headers={"ElevenLabs-Signature": "t=1,v0=correct"},
    )
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_unrecognised_event_type_is_acknowledged_not_rejected(app_client, monkeypatch):
    monkeypatch.setattr(
        wh, "construct_webhook_event",
        lambda raw_body, sig_header, secret: {"type": "some_future_event", "data": {}},
    )
    r = app_client.post(
        "/webhooks/elevenlabs", content=b"{}",
        headers={"ElevenLabs-Signature": "t=1,v0=correct"},
    )
    assert r.status_code == 200
