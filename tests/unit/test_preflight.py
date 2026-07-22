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


# ── language gate: can this agent actually speak the campaign's language? ────
#
# REGRESSION context: a real Telugu two-way test failed three ways at once —
# the agent replied in English, mis-heard Telugu speech, and produced garbled
# audio. Root cause: ElevenLabs Agents defaults to Flash v2.5 TTS, whose 32
# languages don't include Telugu (see app/languages.py's V3_ONLY_ISO). That
# misconfiguration is invisible from outside the platform — these tests pin
# the guard that makes it visible before a campaign dials 200 people.

def _support(monkeypatch, **overrides):
    support = {"override_allowed": True, "languages": {"en", "te", "hi"},
               "tts_model": "eleven_v3_conversational"}
    support.update(overrides)
    monkeypatch.setattr(pf, "agent_language_support", lambda agent_id: support)


async def test_preflight_passes_with_no_language_requested(monkeypatch):
    """'auto' campaigns must not gain a new way to fail — nothing is
    overridden, so nothing needs checking, and the API must not even be
    consulted."""
    _configured(monkeypatch)

    def boom(agent_id):
        raise AssertionError("must not query language support for an auto campaign")

    monkeypatch.setattr(pf, "agent_language_support", boom)
    assert await pf.preflight("agent_1") is None


async def test_preflight_passes_when_the_agent_supports_the_language(monkeypatch):
    _configured(monkeypatch)
    _support(monkeypatch)
    assert await pf.preflight("agent_1", "te") is None


async def test_preflight_blocks_when_the_override_is_disabled(monkeypatch):
    """Overrides are off by default and ElevenLabs raises when one arrives
    unannounced — every call in the campaign would fail. Say which box to
    tick, once, instead of failing N calls."""
    _configured(monkeypatch)
    _support(monkeypatch, override_allowed=False)
    result = await pf.preflight("agent_1", "te")
    assert result and "Security" in result


async def test_preflight_blocks_when_the_language_is_not_on_the_agent(monkeypatch):
    _configured(monkeypatch)
    _support(monkeypatch, languages={"en"})
    result = await pf.preflight("agent_1", "te")
    assert result and "Additional Languages" in result


async def test_preflight_blocks_telugu_on_a_v2_model(monkeypatch):
    """REGRESSION for the real failure this feature was built around: Flash
    v2.5's 32 languages don't include Telugu, so the agent produced garbled
    audio and fell back to English. Invisible from the outside."""
    _configured(monkeypatch)
    _support(monkeypatch, tts_model="eleven_flash_v2_5")
    result = await pf.preflight("agent_1", "te")
    assert result and "v3" in result


async def test_hindi_is_fine_on_a_v2_model(monkeypatch):
    """Hindi IS in Flash v2.5's language list — the model gate must be
    specific to the languages that actually need v3, not a blanket rule."""
    _configured(monkeypatch)
    _support(monkeypatch, tts_model="eleven_flash_v2_5")
    assert await pf.preflight("agent_1", "hi") is None


async def test_a_language_lookup_failure_does_not_block_the_campaign(monkeypatch):
    """An unreachable or changed API must not become a dial-stopping outage:
    the language check is an EXTRA guard, and the call can still succeed if
    the agent happens to be configured correctly. Every other preflight check
    still applies."""
    _configured(monkeypatch)

    def boom(agent_id):
        raise RuntimeError("ElevenLabs API changed")

    monkeypatch.setattr(pf, "agent_language_support", boom)
    assert await pf.preflight("agent_1", "te") is None
