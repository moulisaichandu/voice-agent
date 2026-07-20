"""Unit tests for telephony/preflight.py — each failure mode covered
individually, mirroring ai-voice-agent/backend/tests/test_preflight.py's
per-failure-mode structure, plus the new agent/phone-number-not-found cases
this project's preflight adds that the sibling's didn't need."""

import httpx
import pytest

from app.telephony import preflight as pf


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class _FakeAsyncClient:
    def __init__(self, *, status_code=200, raise_error=None):
        self._status_code = status_code
        self._raise_error = raise_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        if self._raise_error:
            raise self._raise_error
        return _FakeResponse(self._status_code)


def _patch_healthy_agent_and_phone(monkeypatch):
    monkeypatch.setattr(pf, "agent_exists", lambda agent_id: True)
    monkeypatch.setattr(pf, "phone_number_exists", lambda phone_id: True)


async def test_missing_api_key(monkeypatch):
    monkeypatch.setattr(pf, "ELEVENLABS_API_KEY", None)
    result = await pf.preflight("agent_1", "phone_1")
    assert result is not None
    assert "ELEVENLABS_API_KEY" in result


async def test_missing_public_base_url(monkeypatch):
    monkeypatch.setattr(pf, "ELEVENLABS_API_KEY", "key")
    monkeypatch.setattr(pf, "PUBLIC_BASE_URL", "")
    result = await pf.preflight("agent_1", "phone_1")
    assert result is not None
    assert "PUBLIC_BASE_URL" in result


async def test_unreachable_public_base_url(monkeypatch):
    monkeypatch.setattr(pf, "ELEVENLABS_API_KEY", "key")
    monkeypatch.setattr(pf, "PUBLIC_BASE_URL", "https://dead.ngrok.dev")
    monkeypatch.setattr(
        pf.httpx, "AsyncClient",
        lambda **kw: _FakeAsyncClient(raise_error=httpx.ConnectError("refused")),
    )
    result = await pf.preflight("agent_1", "phone_1")
    assert result is not None
    assert "unreachable" in result
    assert "dead.ngrok.dev" in result


async def test_public_base_url_returns_non_200(monkeypatch):
    monkeypatch.setattr(pf, "ELEVENLABS_API_KEY", "key")
    monkeypatch.setattr(pf, "PUBLIC_BASE_URL", "https://example.ngrok.dev")
    monkeypatch.setattr(pf.httpx, "AsyncClient", lambda **kw: _FakeAsyncClient(status_code=404))
    result = await pf.preflight("agent_1", "phone_1")
    assert result is not None
    assert "404" in result


async def test_agent_not_found(monkeypatch):
    monkeypatch.setattr(pf, "ELEVENLABS_API_KEY", "key")
    monkeypatch.setattr(pf, "PUBLIC_BASE_URL", "https://example.ngrok.dev")
    monkeypatch.setattr(pf.httpx, "AsyncClient", lambda **kw: _FakeAsyncClient(status_code=200))
    monkeypatch.setattr(pf, "agent_exists", lambda agent_id: False)
    result = await pf.preflight("bad_agent", "phone_1")
    assert result is not None
    assert "bad_agent" in result


async def test_phone_number_not_found(monkeypatch):
    monkeypatch.setattr(pf, "ELEVENLABS_API_KEY", "key")
    monkeypatch.setattr(pf, "PUBLIC_BASE_URL", "https://example.ngrok.dev")
    monkeypatch.setattr(pf.httpx, "AsyncClient", lambda **kw: _FakeAsyncClient(status_code=200))
    monkeypatch.setattr(pf, "agent_exists", lambda agent_id: True)
    monkeypatch.setattr(pf, "phone_number_exists", lambda phone_id: False)
    result = await pf.preflight("agent_1", "bad_phone")
    assert result is not None
    assert "bad_phone" in result


@pytest.mark.parametrize("blank", ["", "   ", None])
async def test_blank_phone_number_id_is_rejected_without_calling_the_api(monkeypatch, blank):
    """Regression, found by running it live: phone_numbers.get("") hits the
    COLLECTION endpoint and returns 200, so phone_number_exists("") answers
    True. An unset ELEVENLABS_AGENT_PHONE_NUMBER_ID therefore passed the exact
    check meant to catch it, and the dial went out and failed at ElevenLabs —
    burning an attempt per lead for a call that could never connect."""
    monkeypatch.setattr(pf, "ELEVENLABS_API_KEY", "key")
    monkeypatch.setattr(pf, "PUBLIC_BASE_URL", "https://example.ngrok.dev")
    monkeypatch.setattr(pf.httpx, "AsyncClient", lambda **kw: _FakeAsyncClient(status_code=200))

    def must_not_be_called(*a, **kw):
        raise AssertionError("the API must not be consulted for a blank id")

    monkeypatch.setattr(pf, "agent_exists", must_not_be_called)
    monkeypatch.setattr(pf, "phone_number_exists", must_not_be_called)

    result = await pf.preflight("agent_1", blank)
    assert result is not None
    assert "ELEVENLABS_AGENT_PHONE_NUMBER_ID" in result


@pytest.mark.parametrize("blank", ["", "   ", None])
async def test_blank_agent_id_is_rejected_without_calling_the_api(monkeypatch, blank):
    monkeypatch.setattr(pf, "ELEVENLABS_API_KEY", "key")
    monkeypatch.setattr(pf, "PUBLIC_BASE_URL", "https://example.ngrok.dev")
    monkeypatch.setattr(pf.httpx, "AsyncClient", lambda **kw: _FakeAsyncClient(status_code=200))

    def must_not_be_called(*a, **kw):
        raise AssertionError("the API must not be consulted for a blank id")

    monkeypatch.setattr(pf, "agent_exists", must_not_be_called)

    result = await pf.preflight(blank, "phone_1")
    assert result is not None
    assert "agent_id" in result


async def test_api_failure_becomes_a_dont_dial_reason_not_an_exception(monkeypatch):
    """preflight's contract is None = go, str = don't-dial-because. A revoked
    key or an ElevenLabs outage must land as a reason: letting it escape
    aborts process_one mid-reserve instead of cleanly declining to dial."""
    monkeypatch.setattr(pf, "ELEVENLABS_API_KEY", "key")
    monkeypatch.setattr(pf, "PUBLIC_BASE_URL", "https://example.ngrok.dev")
    monkeypatch.setattr(pf.httpx, "AsyncClient", lambda **kw: _FakeAsyncClient(status_code=200))

    def boom(agent_id):
        raise RuntimeError("ElevenLabs is down")

    monkeypatch.setattr(pf, "agent_exists", boom)

    result = await pf.preflight("agent_1", "phone_1")

    assert result is not None
    assert "Could not verify" in result
    assert "ElevenLabs is down" in result


async def test_all_checks_pass_returns_none(monkeypatch):
    monkeypatch.setattr(pf, "ELEVENLABS_API_KEY", "key")
    monkeypatch.setattr(pf, "PUBLIC_BASE_URL", "https://example.ngrok.dev")
    monkeypatch.setattr(pf.httpx, "AsyncClient", lambda **kw: _FakeAsyncClient(status_code=200))
    _patch_healthy_agent_and_phone(monkeypatch)
    result = await pf.preflight("agent_1", "phone_1")
    assert result is None
