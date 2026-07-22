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
    """Hindi, not Telugu: Telugu no longer reaches this ElevenLabs branch at
    all (it routes to the OpenAI Realtime backend — see the backend-split
    tests below), so the ElevenLabs-side happy path is pinned with a language
    that actually stays on this backend."""
    _configured(monkeypatch)
    _support(monkeypatch)
    assert await pf.preflight("agent_1", "hi") is None


async def test_preflight_blocks_when_the_override_is_disabled(monkeypatch):
    """Overrides are off by default and ElevenLabs raises when one arrives
    unannounced — every call in the campaign would fail. Say which box to
    tick, once, instead of failing N calls.

    Hindi, not Telugu: Telugu no longer reaches this ElevenLabs branch at all
    (see the backend-split tests below)."""
    _configured(monkeypatch)
    _support(monkeypatch, override_allowed=False)
    result = await pf.preflight("agent_1", "hi")
    assert result and "Security" in result


async def test_preflight_blocks_when_the_language_is_not_on_the_agent(monkeypatch):
    """Uses Hindi deliberately: ElevenLabs DOES offer Hindi, so "add it under
    Additional Languages" is real, actionable advice. Telugu would be wrong
    here — see the regression test at the end of this file."""
    _configured(monkeypatch)
    _support(monkeypatch, languages={"en"})
    result = await pf.preflight("agent_1", "hi")
    assert result and "Additional Languages" in result


async def test_the_v3_model_gate_still_fires_for_whatever_language_needs_it(monkeypatch):
    """REGRESSION for the real failure this feature was built around: Flash
    v2.5's 32 languages don't include Telugu, so an ElevenLabs agent produced
    garbled audio and fell back to English. Telugu itself no longer reaches
    this branch at all any more — it now dials on the OpenAI Realtime backend
    instead (see the backend-split tests below), so this pins the SAME gate
    with V3_ONLY_ISO forced onto a still-ElevenLabs-backed language, proving
    the check itself (not just Telugu's specific escape from it) is intact
    for whatever language it names next."""
    _configured(monkeypatch)
    _support(monkeypatch, tts_model="eleven_flash_v2_5")
    monkeypatch.setattr(pf.languages_module, "V3_ONLY_ISO", frozenset({"hi"}))
    result = await pf.preflight("agent_1", "hi")
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
    still applies.

    Hindi, not Telugu: Telugu no longer reaches this ElevenLabs branch at all
    (see the backend-split tests below)."""
    _configured(monkeypatch)

    def boom(agent_id):
        raise RuntimeError("ElevenLabs API changed")

    monkeypatch.setattr(pf, "agent_language_support", boom)
    assert await pf.preflight("agent_1", "hi") is None


# ── the message must not send an operator after a setting that cannot exist ──

async def test_an_unsupported_language_says_so_instead_of_naming_a_setting(monkeypatch):
    """REGRESSION. This message used to read "Add it under Additional
    Languages in the agent's settings" for EVERY unconfigured language —
    including Telugu, which ElevenLabs does not offer at all. The operator
    went hunting in the dashboard for an option that cannot exist and
    reasonably concluded they were doing it wrong, not that the tool was.

    Telugu itself no longer reaches this ElevenLabs branch at all (it now
    routes to the OpenAI Realtime backend — see the backend-split tests
    below), so backend_for_iso is forced here to keep exercising this exact
    wording logic on the ISO it was written for."""
    _configured(monkeypatch)
    monkeypatch.setattr(pf.languages_module, "backend_for_iso",
                        lambda iso: pf.languages_module.ELEVENLABS)
    _support(monkeypatch, languages={"en", "hi"})
    result = await pf.preflight("agent_1", "te")
    assert result is not None
    assert "does not support" in result
    assert "Additional Languages" not in result, (
        "must not send the operator to a dashboard setting that cannot exist"
    )


async def test_a_supported_but_unconfigured_language_still_names_the_setting(monkeypatch):
    """The other half of the same branch: Hindi IS offered by ElevenLabs, so
    naming the setting is correct and must survive the fix above."""
    _configured(monkeypatch)
    _support(monkeypatch, languages={"en"})
    result = await pf.preflight("agent_1", "hi")
    assert result is not None
    assert "Additional Languages" in result


# ── the OpenAI backend has entirely different prerequisites ──────────────────
#
# For an OpenAI-backed call every one of the ElevenLabs questions above is
# meaningless — there is no agent, no language preset, no override switch —
# and running them would refuse every Telugu campaign for a reason unrelated
# to whether it can dial.

async def test_a_telugu_campaign_does_not_ask_elevenlabs_anything(monkeypatch):
    """Telugu never touches ElevenLabs. Asking it about an agent that isn't
    involved would refuse every Telugu campaign for a reason that has nothing
    to do with why it can or can't dial."""
    _configured(monkeypatch)
    monkeypatch.setattr(pf, "OPENAI_API_KEY", "sk-test")

    def boom(agent_id):
        raise AssertionError("must not consult ElevenLabs for an OpenAI-backed call")

    monkeypatch.setattr(pf, "agent_language_support", boom)
    assert await pf.preflight("agent_1", "te") is None


async def test_a_telugu_campaign_needs_an_openai_key(monkeypatch):
    _configured(monkeypatch)
    monkeypatch.setattr(pf, "OPENAI_API_KEY", None)
    reason = await pf.preflight("agent_1", "te")
    assert reason is not None
    assert "OPENAI_API_KEY" in reason


async def test_hindi_still_runs_the_elevenlabs_checks(monkeypatch):
    """The other side of the branch: an ElevenLabs-backed language must keep
    every check it had."""
    _configured(monkeypatch)
    _support(monkeypatch, override_allowed=False)
    reason = await pf.preflight("agent_1", "hi")
    assert reason is not None
    assert "Security" in reason


# ── refuse a two-way call the backend cannot hold, AT DIAL TIME ──────────────
#
# openai_bridge.py is ONE-WAY ONLY; two-way is Milestone B. Task 4's
# creation-time guard (app/admin/campaigns.py's _check_twoway_capable) cannot
# catch a campaign that already existed when this feature shipped. Undialled
# today, such a campaign would connect to OpenAI, never speak (no
# turn-detection cue), never listen, sit in dead silence for the full
# CALL_MAX_DURATION_S of billed Plivo airtime, and then be recorded as a
# SUCCESSFUL zero-turn call because max_duration counts as a clean exit —
# worse than the silent-mismatch bug CLAUDE.md warns about by name, because at
# least a monologue is audible.

async def test_a_twoway_openai_backed_campaign_is_refused(monkeypatch):
    _configured(monkeypatch)
    monkeypatch.setattr(pf, "OPENAI_API_KEY", "sk-test")
    reason = await pf.preflight("agent_1", "te", mode="twoway")
    assert reason is not None
    assert "twoway" in reason
    assert "te" in reason
    assert "one-way" in reason.lower()


async def test_a_oneway_openai_backed_campaign_is_not_refused(monkeypatch):
    _configured(monkeypatch)
    monkeypatch.setattr(pf, "OPENAI_API_KEY", "sk-test")
    assert await pf.preflight("agent_1", "te", mode="oneway") is None


async def test_a_twoway_elevenlabs_backed_campaign_is_not_refused(monkeypatch):
    """This combination works in production today (every existing two-way
    campaign) and must not be caught by the new guard."""
    _configured(monkeypatch)
    _support(monkeypatch)
    assert await pf.preflight("agent_1", "hi", mode="twoway") is None


async def test_an_auto_twoway_campaign_is_not_refused(monkeypatch):
    """'auto' sends no language override at all and always stays on
    ElevenLabs — it must not gain a new way to fail."""
    _configured(monkeypatch)
    assert await pf.preflight("agent_1", None, mode="twoway") is None


async def test_the_twoway_guard_needs_no_mode_argument_to_keep_working(monkeypatch):
    """Existing callers that don't pass mode= must keep working exactly as
    before — mode defaults to something that never triggers the guard."""
    _configured(monkeypatch)
    monkeypatch.setattr(pf, "OPENAI_API_KEY", "sk-test")
    assert await pf.preflight("agent_1", "te") is None
