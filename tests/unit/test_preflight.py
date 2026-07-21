"""Unit tests for telephony/preflight.py — one test per failure mode.

Preflight matters more under the Plivo-bridge architecture than it did when
ElevenLabs placed the calls: Plivo fetches the answer URL from the public
internet the moment the lead picks up, so an unreachable PUBLIC_BASE_URL means
the phone RINGS and the call is cut on answer, with nothing in our logs. These
tests pin every reason it must decline to dial.
"""

import httpx
import pytest

from app.telephony import preflight as pf


class _FakeResponse:
    def __init__(self, status_code, text='<Response><Stream>ws://x</Stream></Response>'):
        self.status_code = status_code
        self.text = text


class _FakeAsyncClient:
    def __init__(self, *, status_code=200, text=None, raise_error=None):
        self._status_code = status_code
        self._text = text
        self._raise_error = raise_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url):
        if self._raise_error:
            raise self._raise_error
        if self._text is not None:
            return _FakeResponse(self._status_code, self._text)
        return _FakeResponse(self._status_code)


def _configured(monkeypatch, **overrides):
    """Everything set correctly; individual tests knock one thing out."""
    values = {
        "ELEVENLABS_API_KEY": "key",
        "PLIVO_AUTH_ID": "auth-id",
        "PLIVO_AUTH_TOKEN": "auth-token",
        "PLIVO_FROM_NUMBER": "+918035383564",
        "PUBLIC_BASE_URL": "https://example.trycloudflare.com",
        "CALL_WEBHOOK_SECRET": "s3cret",
    }
    values.update(overrides)
    for name, value in values.items():
        monkeypatch.setattr(pf, name, value)
    monkeypatch.setattr(pf, "agent_exists", lambda agent_id: True)
    monkeypatch.setattr(pf.httpx, "AsyncClient", lambda **kw: _FakeAsyncClient())


async def test_missing_elevenlabs_api_key(monkeypatch):
    _configured(monkeypatch, ELEVENLABS_API_KEY=None)
    result = await pf.preflight("agent_1")
    assert result and "ELEVENLABS_API_KEY" in result


@pytest.mark.parametrize("missing", ["PLIVO_AUTH_ID", "PLIVO_AUTH_TOKEN"])
async def test_missing_plivo_credentials(monkeypatch, missing):
    _configured(monkeypatch, **{missing: None})
    result = await pf.preflight("agent_1")
    assert result and "PLIVO_AUTH" in result


async def test_missing_from_number(monkeypatch):
    _configured(monkeypatch, PLIVO_FROM_NUMBER=None)
    result = await pf.preflight("agent_1")
    assert result and "PLIVO_FROM_NUMBER" in result


@pytest.mark.parametrize("blank", ["", "   ", None])
async def test_blank_agent_id_is_rejected_without_calling_the_api(monkeypatch, blank):
    _configured(monkeypatch)

    def must_not_be_called(agent_id):
        raise AssertionError("the API must not be consulted for a blank id")

    monkeypatch.setattr(pf, "agent_exists", must_not_be_called)
    result = await pf.preflight(blank)
    assert result and "agent_id" in result


async def test_missing_public_base_url(monkeypatch):
    """Now load-bearing: without it Plivo cannot fetch the answer URL, and
    every call rings then drops the instant it's answered."""
    _configured(monkeypatch, PUBLIC_BASE_URL="")
    result = await pf.preflight("agent_1")
    assert result and "PUBLIC_BASE_URL" in result


async def test_unreachable_public_base_url(monkeypatch):
    _configured(monkeypatch)
    monkeypatch.setattr(
        pf.httpx, "AsyncClient",
        lambda **kw: _FakeAsyncClient(raise_error=httpx.ConnectError("refused")),
    )
    result = await pf.preflight("agent_1")
    assert result and "unreachable" in result


async def test_answer_url_rejecting_our_own_token(monkeypatch):
    """A 403 means CALL_WEBHOOK_SECRET doesn't match what the running server
    expects — Plivo's calls would be refused identically."""
    _configured(monkeypatch)
    monkeypatch.setattr(pf.httpx, "AsyncClient",
                        lambda **kw: _FakeAsyncClient(status_code=403))
    result = await pf.preflight("agent_1")
    assert result and "CALL_WEBHOOK_SECRET" in result


async def test_answer_url_returns_non_200(monkeypatch):
    _configured(monkeypatch)
    monkeypatch.setattr(pf.httpx, "AsyncClient",
                        lambda **kw: _FakeAsyncClient(status_code=502))
    result = await pf.preflight("agent_1")
    assert result and "502" in result


async def test_answer_url_returns_200_but_not_stream_xml(monkeypatch):
    """A tunnel error page can return 200. Checking the body is what catches
    'something else is answering on that URL'."""
    _configured(monkeypatch)
    monkeypatch.setattr(
        pf.httpx, "AsyncClient",
        lambda **kw: _FakeAsyncClient(status_code=200, text="<html>ngrok offline</html>"),
    )
    result = await pf.preflight("agent_1")
    assert result and "<Stream>" in result


async def test_agent_not_found(monkeypatch):
    _configured(monkeypatch)
    monkeypatch.setattr(pf, "agent_exists", lambda agent_id: False)
    result = await pf.preflight("bad_agent")
    assert result and "bad_agent" in result


async def test_api_failure_becomes_a_dont_dial_reason_not_an_exception(monkeypatch):
    """preflight's contract is None = go, str = don't-dial-because. An
    exception escaping aborts process_one mid-reservation instead of cleanly
    declining (worker.py docstring point 7)."""
    _configured(monkeypatch)

    def boom(agent_id):
        raise RuntimeError("ElevenLabs is down")

    monkeypatch.setattr(pf, "agent_exists", boom)
    result = await pf.preflight("agent_1")
    assert result and "Could not verify" in result and "ElevenLabs is down" in result


async def test_all_checks_pass_returns_none(monkeypatch):
    _configured(monkeypatch)
    assert await pf.preflight("agent_1") is None
