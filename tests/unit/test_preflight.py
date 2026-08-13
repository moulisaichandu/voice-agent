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
        # No forced voice by default: config.py calls load_dotenv(), so without
        # pinning this a real .env value would change what preflight checks
        # depending on whose machine runs the suite. Voice tests set it.
        "ELEVENLABS_VOICE_ID": None,
        # Same reason: whichever machine runs this suite may have
        # CONVERSATION_LLM_PROVIDER=sarvam in its own .env, which would trip
        # the new twoway/Sarvam guard below in every unrelated test.
        "CONVERSATION_LLM_PROVIDER": "openai",
    }
    values.update(overrides)
    # preflight no longer reads a module-level PUBLIC_BASE_URL: it resolves the
    # live tunnel through public_url.refresh() so a hostname that rotated while
    # the process was running cannot be handed to Plivo. Tests still set it by
    # name here; it just drives the resolver instead of a global.
    base_url = values.pop("PUBLIC_BASE_URL", "")
    for name, value in values.items():
        monkeypatch.setattr(pf, name, value)

    async def _resolve_base():
        return base_url or ""

    monkeypatch.setattr(pf.public_url, "refresh", _resolve_base)
    monkeypatch.setattr(pf, "agent_exists", lambda agent_id: True)
    monkeypatch.setattr(pf.httpx, "AsyncClient", lambda **kw: _FakeAsyncClient())

    # Both account-health gates healthy by default, so every pre-existing test
    # keeps meaning "everything is configured correctly" now that preflight
    # consults them too.
    async def _good_billing(*, force_refresh: bool = False):
        return {"status": "active", "character_count": 1, "character_limit": 2}

    monkeypatch.setattr(pf, "cached_subscription_status", _good_billing)

    async def _not_tripped():
        return None

    monkeypatch.setattr(pf.sarvam_circuit_breaker, "tripped_reason", _not_tripped)


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
               "tts_model": "eleven_v3_conversational",
               "voice_override_allowed": True, "voice_id": "voice_base",
               "preset_voice_ids": {}}
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
    monkeypatch.setattr(pf, "SARVAM_API_KEY", "sk-test")

    def boom(agent_id):
        raise AssertionError("must not consult ElevenLabs for a Sarvam-backed call")

    monkeypatch.setattr(pf, "agent_language_support", boom)
    assert await pf.preflight("agent_1", "te") is None


async def test_a_telugu_campaign_needs_a_sarvam_key(monkeypatch):
    _configured(monkeypatch)
    monkeypatch.setattr(pf, "SARVAM_API_KEY", None)
    reason = await pf.preflight("agent_1", "te")
    assert reason is not None
    assert "SARVAM_API_KEY" in reason


async def test_a_rolled_back_telugu_campaign_needs_the_openai_key_instead(monkeypatch):
    """The rollback switch has to change WHICH prerequisites are checked.
    Checking Sarvam's key for a call openai_bridge is about to place would
    refuse every rolled-back Telugu campaign for a key it does not need — and
    the reverse would let one dial with no credentials at all."""
    _configured(monkeypatch)
    monkeypatch.setattr(pf.languages_module, "TELUGU_BACKEND",
                        pf.languages_module.OPENAI_REALTIME)
    monkeypatch.setattr(pf, "SARVAM_API_KEY", None)
    monkeypatch.setattr(pf, "OPENAI_API_KEY", "sk-test")

    def boom(agent_id):
        raise AssertionError("must not consult ElevenLabs for an OpenAI-backed call")

    monkeypatch.setattr(pf, "agent_language_support", boom)
    assert await pf.preflight("agent_1", "te") is None

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


# ── the forced voice (ELEVENLABS_VOICE_ID) must be validated too ──────────────
#
# Once set, bridge.py sends a tts.voice_id override on EVERY ElevenLabs-backed
# call — including 'auto', which previously sent no override at all and so was
# never validated here. An agent that doesn't allow the field rejects 100% of
# its calls, so this gate has to fire for auto campaigns as well.

_VOICE = "ohvvU75FpBEB8fdaLOMh"


async def test_a_forced_voice_is_validated_even_on_an_auto_campaign(monkeypatch):
    """The gap this closes: 'auto' skipped agent validation entirely, so a
    forced voice would have been sent unchecked and failed every call on
    answer, with nothing in our logs explaining why."""
    _configured(monkeypatch, ELEVENLABS_VOICE_ID=_VOICE)
    _support(monkeypatch, voice_override_allowed=False)

    reason = await pf.preflight("agent_1")  # no language: an auto campaign

    assert reason is not None
    assert "Security" in reason
    assert _VOICE in reason


async def test_an_auto_campaign_with_a_forced_voice_dials_when_allowed(monkeypatch):
    _configured(monkeypatch, ELEVENLABS_VOICE_ID=_VOICE)
    _support(monkeypatch)
    assert await pf.preflight("agent_1") is None


async def test_a_voice_lookup_failure_does_not_block_an_auto_campaign(monkeypatch):
    """Fail-open is preserved on the NEW code path: this check is an extra
    guard over a correctly-configured agent, and an SDK/API outage must not
    become a dial-stopping outage of its own."""
    _configured(monkeypatch, ELEVENLABS_VOICE_ID=_VOICE)

    def boom(agent_id):
        raise RuntimeError("ElevenLabs API down")

    monkeypatch.setattr(pf, "agent_language_support", boom)
    assert await pf.preflight("agent_1") is None


async def test_a_forced_voice_never_consults_elevenlabs_for_a_telugu_call(monkeypatch):
    """A Telugu call is carried by sarvam_bridge.py, which sends no ElevenLabs
    override at all — so a forced ElevenLabs voice is irrelevant to it and must
    not make preflight interrogate an agent that will never carry the call."""
    _configured(monkeypatch, ELEVENLABS_VOICE_ID=_VOICE, SARVAM_API_KEY="sk-test")

    def boom(agent_id):
        raise AssertionError("must not query ElevenLabs for a Sarvam-backed call")

    monkeypatch.setattr(pf, "agent_language_support", boom)
    assert await pf.preflight("agent_1", "te") is None


async def test_the_language_checks_still_run_when_a_voice_is_forced(monkeypatch):
    """The new voice gate must not swallow the pre-existing language gates."""
    _configured(monkeypatch, ELEVENLABS_VOICE_ID=_VOICE)
    _support(monkeypatch, override_allowed=False)

    reason = await pf.preflight("agent_1", "hi")

    assert reason is not None
    assert "language override" in reason


# ── refuse a two-way call on a backend not yet PROVEN on a live call ─────────
#
# openai_bridge.py fully implements two-way conversation, but
# OPENAI_TWOWAY_ENABLED gates real use on a human having confirmed it on a
# live call — CLAUDE.md's discipline for anything this consequential to get
# wrong. Task 4's creation-time guard (app/admin/campaigns.py's
# _check_twoway_capable) cannot catch a campaign that already existed before
# the flag was flipped either way, so this dial-time copy is still
# load-bearing. With the flag off, such a campaign would connect to OpenAI,
# get an unverified conversational experience, and if that experience were
# actually broken (as the sibling project's own unresolved barge-in bug shows
# can happen silently), sit in dead silence for the full CALL_MAX_DURATION_S
# and be recorded as a SUCCESSFUL zero-turn call because max_duration counts
# as a clean exit — worse than the silent-mismatch bug CLAUDE.md warns about
# by name, because at least a monologue is audible.

def _telugu_backend(monkeypatch, backend):
    """Point Telugu at *backend* and satisfy that backend's credential check,
    so a test can isolate the two-way gate from everything else."""
    monkeypatch.setattr(pf.languages_module, "TELUGU_BACKEND", backend)
    monkeypatch.setattr(pf, "SARVAM_API_KEY", "sk-test")
    monkeypatch.setattr(pf, "OPENAI_API_KEY", "sk-test")


def _set_twoway(monkeypatch, backend, enabled):
    flag = ("SARVAM_TWOWAY_ENABLED"
            if backend == pf.languages_module.SARVAM else "OPENAI_TWOWAY_ENABLED")
    monkeypatch.setattr(pf.languages_module, flag, enabled)


_TELUGU_BACKENDS = ["sarvam", "openai_realtime"]


@pytest.mark.parametrize("backend", _TELUGU_BACKENDS)
async def test_a_twoway_telugu_campaign_is_refused_while_its_flag_is_off(
    monkeypatch, backend
):
    _configured(monkeypatch)
    _telugu_backend(monkeypatch, backend)
    _set_twoway(monkeypatch, backend, False)
    reason = await pf.preflight("agent_1", "te", mode="twoway")
    assert reason is not None
    assert "twoway" in reason
    assert "te" in reason
    assert "one-way" in reason.lower()


@pytest.mark.parametrize("backend", _TELUGU_BACKENDS)
async def test_the_refusal_names_the_backend_that_would_actually_run(
    monkeypatch, backend
):
    """The message used to hardcode "OpenAI Realtime". An operator told to
    check a backend their campaign does not use has been sent to the wrong
    place to fix a real problem."""
    _configured(monkeypatch)
    _telugu_backend(monkeypatch, backend)
    _set_twoway(monkeypatch, backend, False)
    reason = await pf.preflight("agent_1", "te", mode="twoway")
    assert pf.languages_module.backend_display(backend) in reason


@pytest.mark.parametrize("backend", _TELUGU_BACKENDS)
async def test_a_twoway_telugu_campaign_dials_once_its_flag_is_on(
    monkeypatch, backend
):
    """The flip side: the flag genuinely gates the check, not just documents
    an intention — proves the guard reads the flag rather than always
    refusing regardless of it."""
    _configured(monkeypatch)
    _telugu_backend(monkeypatch, backend)
    _set_twoway(monkeypatch, backend, True)
    assert await pf.preflight("agent_1", "te", mode="twoway") is None


@pytest.mark.parametrize("backend", _TELUGU_BACKENDS)
async def test_a_oneway_telugu_campaign_is_never_refused_by_this_guard(
    monkeypatch, backend
):
    """One-way is unaffected by the flag in either direction — it was always
    the proven, available path."""
    _configured(monkeypatch)
    _telugu_backend(monkeypatch, backend)
    for flag in (False, True):
        _set_twoway(monkeypatch, backend, flag)
        assert await pf.preflight("agent_1", "te", mode="oneway") is None


async def test_enabling_two_way_on_one_backend_does_not_enable_the_other(monkeypatch):
    """Each backend is a separate implementation proven by a separate live
    call. Sharing one flag would let a Sarvam live-call sign-off silently
    authorise two-way on a backend nobody tested."""
    _configured(monkeypatch)
    _telugu_backend(monkeypatch, pf.languages_module.OPENAI_REALTIME)
    monkeypatch.setattr(pf.languages_module, "SARVAM_TWOWAY_ENABLED", True)
    monkeypatch.setattr(pf.languages_module, "OPENAI_TWOWAY_ENABLED", False)
    assert await pf.preflight("agent_1", "te", mode="twoway") is not None


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
    monkeypatch.setattr(pf, "SARVAM_API_KEY", "sk-test")
    assert await pf.preflight("agent_1", "te") is None


# ── CONVERSATION_LLM_PROVIDER=sarvam cannot answer a two-way Sarvam call ─────
#
# Sarvam's chat models are reasoning models measured at 21.8s on a real
# conversational turn (conversation_llm.py's own docstring) and the reasoning
# cannot be turned off. Against CONVERSATION_LLM_TIMEOUT_S (12s by default)
# that is not a slow agent, it is TurnFailed on nearly every turn, swallowed
# by sarvam_bridge._reply's catch-all — the lead hears silence, and the call
# is still graded a clean exit. The danger was already documented in three
# places (config.py, conversation_llm.py, CLAUDE.md) and enforced in none.

async def test_a_twoway_sarvam_campaign_is_refused_when_the_llm_is_also_sarvam(
    monkeypatch,
):
    _configured(monkeypatch)
    _telugu_backend(monkeypatch, "sarvam")
    _set_twoway(monkeypatch, "sarvam", True)
    monkeypatch.setattr(pf, "CONVERSATION_LLM_PROVIDER", "sarvam")
    reason = await pf.preflight("agent_1", "te", mode="twoway")
    assert reason is not None
    assert "CONVERSATION_LLM_PROVIDER" in reason
    assert "openai" in reason


async def test_a_twoway_sarvam_campaign_dials_when_the_llm_is_openai(monkeypatch):
    _configured(monkeypatch)
    _telugu_backend(monkeypatch, "sarvam")
    _set_twoway(monkeypatch, "sarvam", True)
    monkeypatch.setattr(pf, "CONVERSATION_LLM_PROVIDER", "openai")
    assert await pf.preflight("agent_1", "te", mode="twoway") is None


async def test_a_oneway_sarvam_campaign_is_never_refused_by_this_guard(monkeypatch):
    """One-way never calls conversation_llm at all, so it must not gain a new
    way to not dial."""
    _configured(monkeypatch)
    _telugu_backend(monkeypatch, "sarvam")
    monkeypatch.setattr(pf, "CONVERSATION_LLM_PROVIDER", "sarvam")
    assert await pf.preflight("agent_1", "te", mode="oneway") is None


async def test_a_twoway_openai_realtime_campaign_is_unaffected_by_this_guard(
    monkeypatch,
):
    """OPENAI_REALTIME never calls conversation_llm either — it carries its
    own turn entirely on the Realtime session. This guard is Sarvam-specific."""
    _configured(monkeypatch)
    _telugu_backend(monkeypatch, "openai_realtime")
    _set_twoway(monkeypatch, "openai_realtime", True)
    monkeypatch.setattr(pf, "CONVERSATION_LLM_PROVIDER", "sarvam")
    assert await pf.preflight("agent_1", "te", mode="twoway") is None


# ── ElevenLabs account billing must be healthy before any ElevenLabs call ────
#
# A past_due account refuses the agent WebSocket handshake with a payment-issue
# error — verified live 2026-07. Checked here, not only on the readiness
# dashboard, so a dead account stops dialling instead of ringing every lead
# into instant silence.

async def test_unhealthy_elevenlabs_billing_refuses_to_dial(monkeypatch):
    _configured(monkeypatch)

    async def past_due(*, force_refresh=False):
        return {"status": "past_due", "character_count": 1, "character_limit": 2}

    monkeypatch.setattr(pf, "cached_subscription_status", past_due)

    reason = await pf.preflight("agent_1")
    assert reason is not None
    assert "past_due" in reason


async def test_healthy_elevenlabs_billing_does_not_block(monkeypatch):
    _configured(monkeypatch)
    assert await pf.preflight("agent_1") is None


async def test_an_unreachable_billing_check_refuses_rather_than_dials_blind(monkeypatch):
    """Same discipline as agent_exists() above: a check that cannot answer must
    not be read as "must be fine". A false refusal costs one paused campaign; a
    false pass costs every lead on it."""
    _configured(monkeypatch)

    async def boom(*, force_refresh=False):
        raise RuntimeError("ElevenLabs API down")

    monkeypatch.setattr(pf, "cached_subscription_status", boom)

    reason = await pf.preflight("agent_1")
    assert reason is not None
    assert "Could not check" in reason


async def test_billing_is_checked_even_for_a_plain_auto_campaign(monkeypatch):
    """THE ordering this check depends on. The 'auto'-with-no-voice-override
    early return exists to skip an unnecessary ElevenLabs API call — but
    billing applies to every ElevenLabs call regardless of override, so a
    billing check placed after it would be skipped by exactly the commonest
    campaign shape there is."""
    _configured(monkeypatch, ELEVENLABS_VOICE_ID=None)

    async def past_due(*, force_refresh=False):
        return {"status": "past_due", "character_count": 1, "character_limit": 2}

    monkeypatch.setattr(pf, "cached_subscription_status", past_due)

    # language=None and no forced voice: the early-return path.
    reason = await pf.preflight("agent_1")
    assert reason is not None, "a plain 'auto' campaign skipped the billing check"
    assert "past_due" in reason


async def test_billing_is_not_checked_for_a_sarvam_backed_call(monkeypatch):
    """One backend's outage must not ground the other's campaigns."""
    _configured(monkeypatch, SARVAM_API_KEY="sk-test")

    async def must_not_be_called(*, force_refresh=False):
        raise AssertionError("must not check ElevenLabs billing for a Sarvam call")

    monkeypatch.setattr(pf, "cached_subscription_status", must_not_be_called)
    assert await pf.preflight("agent_1", "te") is None


# ── the Sarvam circuit breaker must be clear before any Sarvam call ─────────
#
# Sarvam has no programmatic credits/balance API, so this is reactive: a call
# already failed with "insufficient credits" and tripped it, and every later
# Sarvam-backed dial must refuse until an admin clears it.

async def test_a_tripped_sarvam_breaker_refuses_to_dial(monkeypatch):
    _configured(monkeypatch, SARVAM_API_KEY="sk-test")

    async def tripped():
        return "Insufficient credits"

    monkeypatch.setattr(pf.sarvam_circuit_breaker, "tripped_reason", tripped)

    reason = await pf.preflight("agent_1", "te")
    assert reason is not None
    assert "Insufficient credits" in reason


async def test_the_refusal_says_how_to_clear_the_breaker(monkeypatch):
    """Manual-clear-only is the design, so the refusal has to carry the way
    out — otherwise the operator is told dialling stopped and not how to
    restart it."""
    _configured(monkeypatch, SARVAM_API_KEY="sk-test")

    async def tripped():
        return "Insufficient credits"

    monkeypatch.setattr(pf.sarvam_circuit_breaker, "tripped_reason", tripped)

    reason = await pf.preflight("agent_1", "te")
    assert "sarvam-circuit-breaker/clear" in reason
    assert "dashboard.sarvam.ai" in reason


async def test_a_clear_sarvam_breaker_does_not_block(monkeypatch):
    _configured(monkeypatch, SARVAM_API_KEY="sk-test")
    assert await pf.preflight("agent_1", "te") is None


async def test_the_sarvam_breaker_is_not_checked_for_an_elevenlabs_backed_call(monkeypatch):
    """The mirror of the billing test above: a Sarvam credits outage must not
    stop English and Hindi campaigns that never touch Sarvam."""
    _configured(monkeypatch)

    async def must_not_be_called():
        raise AssertionError("must not check the Sarvam breaker for an ElevenLabs call")

    monkeypatch.setattr(pf.sarvam_circuit_breaker, "tripped_reason", must_not_be_called)
    assert await pf.preflight("agent_1") is None
