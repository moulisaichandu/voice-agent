"""Unit tests for telephony/sarvam_llm.py — English script -> spoken Telugu.

No network: the httpx client is replaced with a fake, the same way
tests/unit/test_preflight.py does it. Redis is faked too, because the cache is
what keeps an answered call from opening with two seconds of silence.
"""

import pytest

from app.telephony import sarvam_llm

_SCRIPT = "This is an automated AI call. Our new Python course starts Monday."
_TELUGU = "ఇది కృత్రిమ మేధ ద్వారా చేసే ఆటోమేటెడ్ కాల్. కొత్త కోర్సు సోమవారం మొదలవుతుంది."


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self.text = text
        self._payload = payload if payload is not None else {
            "choices": [{"message": {"content": _TELUGU}}]
        }

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Records every POST so a test can assert the API was NOT called."""

    calls: list[dict] = []

    def __init__(self, *, status_code=200, payload=None, raise_error=None, text=""):
        self._status_code = status_code
        self._payload = payload
        self._raise_error = raise_error
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None):
        type(self).calls.append({"url": url, "json": json, "headers": headers})
        if self._raise_error:
            raise self._raise_error
        return _FakeResponse(self._status_code, self._payload, self._text)


class _FakeRedis:
    def __init__(self, *, broken=False):
        self.store: dict[str, str] = {}
        self._broken = broken

    async def get(self, key):
        if self._broken:
            raise ConnectionError("redis is down")
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        if self._broken:
            raise ConnectionError("redis is down")
        self.store[key] = value


@pytest.fixture
def llm(monkeypatch):
    """A configured renderer with a fake API and a fake Redis."""
    monkeypatch.setattr(sarvam_llm, "SARVAM_API_KEY", "sk-test")
    _FakeAsyncClient.calls = []

    def _install(*, redis=None, **client_kwargs):
        monkeypatch.setattr(
            sarvam_llm.httpx, "AsyncClient",
            lambda **kw: _FakeAsyncClient(**client_kwargs),
        )
        fake_redis = redis if redis is not None else _FakeRedis()
        monkeypatch.setattr(sarvam_llm, "get_redis", lambda: fake_redis)
        return fake_redis

    return _install


# ── rendering ────────────────────────────────────────────────────────────────

async def test_an_english_script_comes_back_as_telugu(llm):
    llm()
    assert await sarvam_llm.render(_SCRIPT, language_style=None) == _TELUGU


async def test_the_script_is_sent_to_the_configured_model(llm):
    llm()
    await sarvam_llm.render(_SCRIPT, language_style=None)

    sent = _FakeAsyncClient.calls[0]
    assert sent["json"]["model"] == sarvam_llm.SARVAM_LLM_MODEL
    assert _SCRIPT in str(sent["json"]["messages"])
    assert sent["headers"]["Authorization"] == "Bearer sk-test"


async def test_a_blank_script_never_calls_the_api(llm):
    """A one-way campaign cannot be created without a script, so a blank one
    means a caller bypassed that guard. Paying for a completion to render
    nothing, then speaking it, helps no one."""
    llm()
    assert await sarvam_llm.render("   ", language_style=None) == ""
    assert _FakeAsyncClient.calls == []


async def test_an_api_failure_raises_rather_than_speaking_english(llm):
    """The alternative — falling back to the raw English script — would call a
    Telugu-speaking lead and read English at them, which is the exact failure
    the prompt rules exist to prevent. Fail the call instead."""
    llm(status_code=500, payload={})
    with pytest.raises(sarvam_llm.SarvamRenderFailed):
        await sarvam_llm.render(_SCRIPT, language_style=None)


async def test_no_api_key_raises_before_calling_out(llm, monkeypatch):
    llm()
    monkeypatch.setattr(sarvam_llm, "SARVAM_API_KEY", None)
    with pytest.raises(sarvam_llm.SarvamNotConfigured):
        await sarvam_llm.render(_SCRIPT, language_style=None)
    assert _FakeAsyncClient.calls == []


# ── the cache ────────────────────────────────────────────────────────────────

async def test_the_second_call_for_a_script_does_not_hit_the_api(llm):
    """THE reason this cache exists. The script is per-CAMPAIGN, so without it
    every answered call would open with a second or two of silence while the
    lead waits for a completion that renders the identical text every time."""
    llm()
    first = await sarvam_llm.render(_SCRIPT, language_style=None)
    second = await sarvam_llm.render(_SCRIPT, language_style=None)

    assert first == second == _TELUGU
    assert len(_FakeAsyncClient.calls) == 1


async def test_a_different_script_is_rendered_again(llm):
    llm()
    await sarvam_llm.render(_SCRIPT, language_style=None)
    await sarvam_llm.render("This is an AI call. Fees dropped.", language_style=None)
    assert len(_FakeAsyncClient.calls) == 2


async def test_a_different_language_style_is_rendered_again(llm):
    """te and tinglish produce genuinely different speech from the same script.
    Sharing a cache entry would make whichever campaign dialled first decide
    the register for the other."""
    llm()
    await sarvam_llm.render(_SCRIPT, language_style="Speak only in Telugu.")
    await sarvam_llm.render(_SCRIPT, language_style="Mix in English words.")
    assert len(_FakeAsyncClient.calls) == 2


async def test_a_dead_redis_still_renders(llm):
    """CLAUDE.md: Redis is ephemeral and must never be able to block a call.
    A cache outage should cost latency, not the campaign."""
    llm(redis=_FakeRedis(broken=True))
    assert await sarvam_llm.render(_SCRIPT, language_style=None) == _TELUGU


# ── the disclosure the lead will actually hear ───────────────────────────────

async def test_a_rendered_script_that_dropped_the_disclosure_is_repaired(llm, caplog):
    """The one thing this backend can do that OpenAI Realtime could not.

    We hold the exact Telugu BEFORE it is spoken, so a dropped disclosure can
    be fixed rather than merely reported after the fact. call_routes'
    post-call check stays as the backstop; this stops the lead ever hearing a
    non-compliant opening in the first place."""
    llm(payload={"choices": [{"message": {"content": "కొత్త కోర్సు సోమవారం మొదలవుతుంది."}}]})

    with caplog.at_level("WARNING"):
        rendered = await sarvam_llm.render(_SCRIPT, language_style=None)

    from app.compliance.disclosure import has_ai_disclosure
    assert has_ai_disclosure(rendered)
    assert "కొత్త కోర్సు సోమవారం మొదలవుతుంది." in rendered
    assert "[compliance]" in caplog.text


async def test_a_compliant_render_is_left_exactly_as_it_is(llm, caplog):
    """No prefix, no duplicated disclosure, nothing logged — the normal path."""
    llm()
    with caplog.at_level("WARNING"):
        assert await sarvam_llm.render(_SCRIPT, language_style=None) == _TELUGU
    assert "[compliance]" not in caplog.text


async def test_the_repaired_text_is_what_gets_cached(llm):
    """Caching the raw model output would serve a non-compliant opening to
    every later call on that campaign, and the repair would run again each
    time — or not at all, if someone later trusted the cache."""
    redis = llm(payload={"choices": [{"message": {"content": "కోర్సు సోమవారం."}}]})
    rendered = await sarvam_llm.render(_SCRIPT, language_style=None)

    from app.compliance.disclosure import has_ai_disclosure
    assert list(redis.store.values()) == [rendered]
    assert has_ai_disclosure(next(iter(redis.store.values())))


# ── reasoning-model latency and empty content ────────────────────────────────
#
# Measured against the live API on 2026-07-27. sarvam-105b and sarvam-30b are
# REASONING models: every response spends 400-2000 completion tokens on
# `reasoning_content` before emitting ~100 characters of actual `content`.
# Reasoning cannot be switched off (reasoning_effort only accepts
# low/medium/high, and 'low' still reasoned for 17s; thinking.type=disabled and
# chat_template_kwargs.enable_thinking=False were accepted and ignored).
#
# Consequences this module has to survive:
#   * a render takes 14-22s, so a 20s timeout fails on a normal script;
#   * capping max_tokens truncates INSIDE the reasoning, so `content` comes
#     back null rather than short.

def test_the_client_allows_for_a_reasoning_model_s_latency():
    """REGRESSION. _TIMEOUT_S was 20s, chosen before anyone had run this
    against the real API. A render of an ordinary three-sentence script
    measured 14-22s, so the very first live call failed with ReadTimeout and
    the campaign would not have dialled at all."""
    assert sarvam_llm._TIMEOUT_S >= 45, (
        "a reasoning model needs headroom; 20s failed on the first real render"
    )


async def test_a_render_that_returns_only_reasoning_fails_loudly(llm):
    """Same shape on the render path, where it already had to fail — pinned so
    the null-content case stays distinguishable from a working empty script."""
    llm(payload={"choices": [{"message": {
        "content": None, "reasoning_content": "Let me think about"}}]})

    with pytest.raises(sarvam_llm.SarvamRenderFailed):
        await sarvam_llm.render(_SCRIPT, language_style=None)


async def test_the_render_asks_for_enough_tokens_to_finish_thinking(llm):
    """REGRESSION from a live failure: "Sarvam chat completions returned no
    text", intermittently, on a script that had rendered fine minutes earlier.

    Sarvam's default completion budget is 2048 tokens and the reasoning
    routinely eats all of it. Measured over four runs with no max_tokens: two
    returned finish_reason=length at exactly 2048 with content=0 characters.
    The same prompt with max_tokens=4000 succeeded three times out of three,
    using 1493-2234 tokens — so the ceiling has to sit well above the reasoning,
    not near it.

    A render that fails means the call does not dial at all, so this was a
    campaign that worked or did not depending on how long the model thought."""
    llm()
    await sarvam_llm.render(_SCRIPT, language_style=None)

    sent = _FakeAsyncClient.calls[0]["json"]
    assert sent.get("max_tokens", 0) >= 4000, (
        "Sarvam's 2048 default is below what its own reasoning consumes"
    )


# ── the third leg: rendering ─────────────────────────────────────────────────
#
# STT and TTS both trip the circuit breaker when Sarvam reports no credits.
# This module talks to the same account over HTTP, and until now it discarded
# the body and reported a bare "HTTP 402" — the one line that would have named
# the cause, thrown away, and the breaker left clear so preflight went on
# letting campaigns dial.

async def test_a_credits_failure_trips_the_circuit_breaker(llm, monkeypatch):
    tripped = {}

    async def fake_trip(reason):
        tripped["reason"] = reason

    monkeypatch.setattr(sarvam_llm.sarvam_circuit_breaker, "trip", fake_trip)
    llm(status_code=402, payload={},
        text='{"error":{"message":"No credits available."}}')

    with pytest.raises(sarvam_llm.SarvamRenderFailed):
        await sarvam_llm.render(_SCRIPT, language_style=None)

    assert "No credits available." in tripped["reason"]


async def test_an_ordinary_http_failure_does_not_trip_the_breaker(llm, monkeypatch):
    """A 500 is a blip; the next render may well succeed. Halting every Sarvam
    campaign over one would cost far more than the retry."""
    async def must_not_be_called(reason):
        raise AssertionError("an ordinary HTTP failure must not trip the breaker")

    monkeypatch.setattr(sarvam_llm.sarvam_circuit_breaker, "trip", must_not_be_called)
    llm(status_code=500, payload={}, text="internal server error")

    with pytest.raises(sarvam_llm.SarvamRenderFailed):
        await sarvam_llm.render(_SCRIPT, language_style=None)


async def test_the_failure_names_the_body_not_just_the_status(llm, monkeypatch):
    """"HTTP 402" is a fact; "No credits available" is the diagnosis. The
    operator reading this line is the one who has to fix the account."""
    async def fake_trip(reason):
        pass

    monkeypatch.setattr(sarvam_llm.sarvam_circuit_breaker, "trip", fake_trip)
    llm(status_code=402, payload={},
        text='{"error":{"message":"No credits available."}}')

    with pytest.raises(sarvam_llm.SarvamRenderFailed) as excinfo:
        await sarvam_llm.render(_SCRIPT, language_style=None)

    assert "402" in str(excinfo.value)
    assert "No credits available." in str(excinfo.value)


async def test_a_huge_error_body_is_truncated(llm, monkeypatch):
    """An HTML error page must not become the breaker's reason string, which
    an admin reads on the readiness dashboard."""
    async def fake_trip(reason):
        pass

    monkeypatch.setattr(sarvam_llm.sarvam_circuit_breaker, "trip", fake_trip)
    llm(status_code=502, payload={}, text="x" * 5000)

    with pytest.raises(sarvam_llm.SarvamRenderFailed) as excinfo:
        await sarvam_llm.render(_SCRIPT, language_style=None)

    assert len(str(excinfo.value)) < 600


async def test_a_response_with_no_body_attribute_does_not_crash(llm, monkeypatch):
    """Defensive: this runs on the failure path, where raising something
    unexpected would replace a diagnosed failure with an undiagnosed one."""
    async def must_not_be_called(reason):
        raise AssertionError("no body cannot be a credits failure")

    monkeypatch.setattr(sarvam_llm.sarvam_circuit_breaker, "trip", must_not_be_called)

    class _NoBodyResponse:
        status_code = 500

        def json(self):
            return {}

    class _NoBodyClient(_FakeAsyncClient):
        async def post(self, url, json=None, headers=None):
            return _NoBodyResponse()

    monkeypatch.setattr(sarvam_llm.httpx, "AsyncClient", lambda **kw: _NoBodyClient())
    monkeypatch.setattr(sarvam_llm, "get_redis", lambda: _FakeRedis())

    with pytest.raises(sarvam_llm.SarvamRenderFailed):
        await sarvam_llm.render(_SCRIPT, language_style=None)


# ── the reasoning budget outgrew its ceiling ────────────────────────────────
#
# Measured against the LIVE API on 2026-08-27, on the exact default two-way
# script, after two real calls were abandoned mid-dial:
#   max_tokens=4096                -> 51.3s, finish_reason=length, content=''
#                                     (14,447 chars of reasoning_content)
#   max_tokens=4096  effort=low    -> 50.2s, finish_reason=length, content=''
#   max_tokens=16384 effort=low    -> 93.1s, finish_reason=stop, real Telugu
#   max_tokens=16384 no effort     -> 139.9s, finish_reason=stop, real Telugu
# The 4096 ceiling was chosen when the reasoning spent 1493-2234 tokens; it
# now spends the whole budget before writing a word, so every render of a
# NEW campaign returned nothing and its first lead heard silence.

async def test_the_completion_budget_clears_the_models_reasoning(llm):
    llm()
    await sarvam_llm.render("Course starts Monday.")

    sent = _FakeAsyncClient.calls[-1]["json"]
    assert sent["max_tokens"] >= 16384, (
        "the reasoning spends the whole budget below this and returns no text"
    )


async def test_the_render_asks_for_the_least_reasoning_the_api_allows(llm):
    """effort=low does not stop the reasoning (it still overran 4096), but it
    measured 93s against 140s — the difference between a warm-up that lands
    inside its timeout and one that does not."""
    llm()
    await sarvam_llm.render("Course starts Monday.")

    assert _FakeAsyncClient.calls[-1]["json"]["reasoning_effort"] == "low"


def test_the_http_timeout_outlasts_a_real_render():
    """93.1s measured. A 60s timeout killed the warm-up before it could
    finish, so the cache never filled and every call paid the failure
    again — exactly what the log said would happen."""
    assert sarvam_llm._TIMEOUT_S >= 150.0
