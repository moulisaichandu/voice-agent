"""Unit tests for telephony/elevenlabs_client.py — the real SDK is mocked at
the app.telephony.elevenlabs_client.get_client() seam, so these run with zero
network calls. The shapes mocked here were verified by directly introspecting
the installed elevenlabs==2.58.0 SDK, not assumed."""

from types import SimpleNamespace

import pytest
from elevenlabs.core.api_error import ApiError
from elevenlabs.errors import NotFoundError

from app.telephony import elevenlabs_client as ec


class _FakeSipTrunk:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.last_call_kwargs = None

    def outbound_call(self, **kwargs):
        self.last_call_kwargs = kwargs
        if self._error:
            raise self._error
        return self._response


def _api_error(status_code: int) -> ApiError:
    """The shape the REAL API returns. Verified live: a missing agent id comes
    back as a bare ApiError with status_code=404, NOT the SDK's NotFoundError
    — which is why the original `except NotFoundError` never fired and a
    typo'd agent id crashed preflight instead of failing it cleanly."""
    return ApiError(status_code=status_code, headers={}, body={"detail": "boom"})


class _FakeAgents:
    def __init__(self, exists=True, error=None):
        self._exists = exists
        self._error = error

    def get(self, agent_id):
        if self._error:
            raise self._error
        if not self._exists:
            raise _api_error(404)
        return SimpleNamespace(agent_id=agent_id)


class _FakePhoneNumbers:
    def __init__(self, exists=True, error=None):
        self._exists = exists
        self._error = error

    def get(self, phone_number_id):
        if self._error:
            raise self._error
        if not self._exists:
            raise _api_error(404)
        return SimpleNamespace(phone_number_id=phone_number_id)


class _FakeWebhooks:
    def __init__(self, event=None, error=None):
        self._event = event
        self._error = error

    def construct_event(self, rawBody, sig_header, secret):
        if self._error:
            raise self._error
        return self._event


def _fake_client(*, sip_trunk=None, agents=None, phone_numbers=None, webhooks=None):
    return SimpleNamespace(conversational_ai=SimpleNamespace(
        sip_trunk=sip_trunk or _FakeSipTrunk(),
        agents=agents or _FakeAgents(),
        phone_numbers=phone_numbers or _FakePhoneNumbers(),
    ), webhooks=webhooks or _FakeWebhooks())


def test_place_outbound_call_maps_response_and_injects_dynamic_variables(monkeypatch):
    # SipTrunkOutboundCallResponse's call-id field is sip_call_id (not
    # Exotel/Twilio's call_sid) — mapped onto OutboundCallResult.call_sid,
    # the internal name every other module (worker.py, db/calls.py) uses.
    fake_response = SimpleNamespace(
        success=True, message="ok", conversation_id="conv_123", sip_call_id="call_abc",
    )
    sip_trunk = _FakeSipTrunk(response=fake_response)
    monkeypatch.setattr(ec, "get_client", lambda: _fake_client(sip_trunk=sip_trunk))

    result = ec.place_outbound_call(
        agent_id="agent_1", agent_phone_number_id="phone_1", to_number="+919876543210",
        dynamic_variables={"lead_id": "lead_xyz"},
    )

    assert result.success is True
    assert result.conversation_id == "conv_123"
    assert result.call_sid == "call_abc"
    # dynamic_variables actually reached the SDK call, not just accepted and dropped
    init_data = sip_trunk.last_call_kwargs["conversation_initiation_client_data"]
    assert init_data.dynamic_variables == {"lead_id": "lead_xyz"}
    assert sip_trunk.last_call_kwargs["agent_id"] == "agent_1"
    assert sip_trunk.last_call_kwargs["to_number"] == "+919876543210"


def test_agent_exists_true_and_false(monkeypatch):
    monkeypatch.setattr(ec, "get_client", lambda: _fake_client(agents=_FakeAgents(exists=True)))
    assert ec.agent_exists("agent_1") is True

    monkeypatch.setattr(ec, "get_client", lambda: _fake_client(agents=_FakeAgents(exists=False)))
    assert ec.agent_exists("agent_missing") is False


def test_phone_number_exists_true_and_false(monkeypatch):
    monkeypatch.setattr(
        ec, "get_client", lambda: _fake_client(phone_numbers=_FakePhoneNumbers(exists=True))
    )
    assert ec.phone_number_exists("phone_1") is True

    monkeypatch.setattr(
        ec, "get_client", lambda: _fake_client(phone_numbers=_FakePhoneNumbers(exists=False))
    )
    assert ec.phone_number_exists("phone_missing") is False


def test_missing_agent_is_detected_from_a_bare_ApiError_404(monkeypatch):
    """Regression: the live API raises ApiError(404), not NotFoundError. The
    original `except NotFoundError` never caught it, so preflight crashed
    instead of reporting a missing agent."""
    monkeypatch.setattr(ec, "get_client", lambda: _fake_client(agents=_FakeAgents(exists=False)))
    assert ec.agent_exists("agent_missing") is False

    monkeypatch.setattr(
        ec, "get_client", lambda: _fake_client(phone_numbers=_FakePhoneNumbers(exists=False))
    )
    assert ec.phone_number_exists("phone_missing") is False


def test_notfounderror_subclass_is_also_treated_as_missing(monkeypatch):
    """NotFoundError subclasses ApiError and carries status_code 404, so the
    status-code check must cover it too — some endpoints do raise it."""
    err = NotFoundError(body={"detail": "nope"})
    monkeypatch.setattr(
        ec, "get_client", lambda: _fake_client(agents=_FakeAgents(error=err))
    )
    assert ec.agent_exists("agent_missing") is False


@pytest.mark.parametrize("status", [401, 429, 500])
def test_non_404_api_errors_are_reraised_not_reported_as_missing(monkeypatch, status):
    """A revoked key or a rate limit must NOT be reported as "agent not
    found" — that sends whoever reads the preflight message hunting for a
    deleted agent that is actually fine."""
    monkeypatch.setattr(
        ec, "get_client",
        lambda: _fake_client(agents=_FakeAgents(error=_api_error(status))),
    )
    with pytest.raises(ApiError):
        ec.agent_exists("agent_1")


def test_construct_webhook_event_delegates_to_sdk(monkeypatch):
    webhooks = _FakeWebhooks(event={"type": "post_call_transcription", "data": {}})
    monkeypatch.setattr(ec, "get_client", lambda: _fake_client(webhooks=webhooks))
    event = ec.construct_webhook_event("raw", "t=1,v0=abc", "secret")
    assert event["type"] == "post_call_transcription"


def test_get_client_refuses_without_api_key(monkeypatch):
    monkeypatch.setattr(ec, "ELEVENLABS_API_KEY", None)
    monkeypatch.setattr(ec, "_client", None)
    with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
        ec.get_client()


# ── agent_language_support: what the agent can speak, and in which voice ──────
# Nested shapes verified against elevenlabs==2.58.0:
#   conversation_config.tts.{model_id,voice_id}
#   conversation_config.language_presets[iso].overrides.tts.voice_id
#   platform_settings.overrides.conversation_config_override.{agent.language,
#                                                             tts.voice_id}
# The last one is the Security-tab ALLOWLIST (booleans), not the payload.


def _configured_agent(*, language="en", voice_id="voice_base",
                      tts_model="eleven_flash_v2", language_presets=None,
                      language_override=False, voice_override=False):
    return SimpleNamespace(
        conversation_config=SimpleNamespace(
            agent=SimpleNamespace(language=language),
            tts=SimpleNamespace(model_id=tts_model, voice_id=voice_id),
            language_presets=language_presets or {},
        ),
        platform_settings=SimpleNamespace(
            overrides=SimpleNamespace(
                conversation_config_override=SimpleNamespace(
                    agent=SimpleNamespace(language=language_override),
                    tts=SimpleNamespace(voice_id=voice_override),
                ),
            ),
        ),
    )


class _FakeConfiguredAgents:
    def __init__(self, agent):
        self._agent = agent

    def get(self, agent_id):
        return self._agent


def test_agent_language_support_reports_the_configured_voice_and_override(monkeypatch):
    """The voice fields are what let preflight refuse a dial before ElevenLabs
    rejects the conversation, and what lets the operator see whether their
    dashboard change actually landed."""
    agent = _configured_agent(voice_id="voice_base", language_override=True,
                              voice_override=True)
    monkeypatch.setattr(ec, "get_client",
                        lambda: _fake_client(agents=_FakeConfiguredAgents(agent)))

    support = ec.agent_language_support("agent_1")

    assert support["voice_id"] == "voice_base"
    assert support["voice_override_allowed"] is True
    # The pre-existing keys must be untouched — preflight and campaigns.py read them.
    assert support["override_allowed"] is True
    assert support["languages"] == {"en"}
    assert support["tts_model"] == "eleven_flash_v2"


def test_agent_language_support_degrades_to_unknown_when_the_shape_is_missing(monkeypatch):
    """The SDK response shape is not a contract we control. A missing
    intermediate must degrade to "I don't know" rather than raise inside
    preflight — which never raises by design (worker.py would abort a
    reservation). This is the property the new voice check depends on."""
    monkeypatch.setattr(ec, "get_client", lambda: _fake_client(agents=_FakeAgents()))

    support = ec.agent_language_support("agent_1")  # bare SimpleNamespace(agent_id=...)

    assert support["voice_id"] is None
    assert support["voice_override_allowed"] is False
    assert support["preset_voice_ids"] == {}


def test_agent_language_support_reports_a_per_language_preset_voice(monkeypatch):
    """A per-language preset PINS a voice for that language, so changing the
    agent's base voice in the dashboard cannot affect it — the most likely
    reason a voice change appears not to take effect for one language."""
    presets = {"hi": SimpleNamespace(
        overrides=SimpleNamespace(tts=SimpleNamespace(voice_id="voice_preset")))}
    agent = _configured_agent(voice_id="voice_base", language_presets=presets)
    monkeypatch.setattr(ec, "get_client",
                        lambda: _fake_client(agents=_FakeConfiguredAgents(agent)))

    support = ec.agent_language_support("agent_1")

    assert support["preset_voice_ids"] == {"hi": "voice_preset"}
    assert support["voice_id"] == "voice_base"   # the base voice is unchanged
    assert "hi" in support["languages"]          # presets still count as languages


# ── cached_subscription_status: the shared 60s cache ─────────────────────────
# Shared by app/admin/system.py's readiness dashboard and
# app/telephony/preflight.py's dial-time gate — this is the function both
# call, so a dashboard poll and a preflight check within the same 60s window
# cost exactly one real ElevenLabs API call between them, not two.


class _FakeRedis:
    def __init__(self, store: dict | None = None):
        self.store: dict[str, str] = store or {}
        self.set_calls: list[tuple] = []

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value
        self.set_calls.append((key, value, ex))
        return True


class _BrokenRedis(_FakeRedis):
    async def get(self, key):
        raise ConnectionError("redis is down")

    async def set(self, key, value, ex=None):
        raise ConnectionError("redis is down")


async def test_cached_subscription_status_caches_across_calls(monkeypatch):
    calls = {"n": 0}

    def counted():
        calls["n"] += 1
        return {"status": "active", "character_count": 1, "character_limit": 2}

    monkeypatch.setattr(ec, "subscription_status", counted)
    # ONE shared instance: a lambda building a fresh _FakeRedis per call would
    # make the cache unable to persist, and this test unable to fail.
    redis = _FakeRedis()
    monkeypatch.setattr(ec.redis_client, "get_redis", lambda: redis)

    first = await ec.cached_subscription_status()
    second = await ec.cached_subscription_status()

    assert calls["n"] == 1
    assert first == second == {"status": "active", "character_count": 1,
                               "character_limit": 2}


async def test_cached_subscription_status_force_refresh_bypasses_the_cache(monkeypatch):
    """POST /admin/readiness/refresh means "I just paid the invoice, tell me
    now" — waiting out a 60s TTL there is the one case where the cache is
    exactly wrong."""
    calls = {"n": 0}

    def counted():
        calls["n"] += 1
        return {"status": "active", "character_count": calls["n"], "character_limit": 2}

    monkeypatch.setattr(ec, "subscription_status", counted)
    # Shared instance, so the first call genuinely warms the cache. With a
    # fresh fake per call this would pass whether or not force_refresh works.
    redis = _FakeRedis()
    monkeypatch.setattr(ec.redis_client, "get_redis", lambda: redis)

    await ec.cached_subscription_status()
    await ec.cached_subscription_status()          # served from cache
    await ec.cached_subscription_status(force_refresh=True)

    assert calls["n"] == 2, "cached call hit the API, or force_refresh did not"


async def test_cached_subscription_status_survives_a_malformed_cache_entry(monkeypatch):
    def fresh():
        return {"status": "active", "character_count": 1, "character_limit": 2}

    monkeypatch.setattr(ec, "subscription_status", fresh)
    redis = _FakeRedis({ec._SUBSCRIPTION_CACHE_KEY: "not-json"})
    monkeypatch.setattr(ec.redis_client, "get_redis", lambda: redis)

    result = await ec.cached_subscription_status()

    assert result == {"status": "active", "character_count": 1, "character_limit": 2}


async def test_a_redis_outage_still_answers_from_the_live_api(monkeypatch):
    """Redis is the ephemeral speed layer, never a dependency that can take
    the answer away. If the cache is unreachable this must still report the
    account's real state — preflight is about to refuse or allow a dial on
    the strength of it."""
    def fresh():
        return {"status": "active", "character_count": 1, "character_limit": 2}

    monkeypatch.setattr(ec, "subscription_status", fresh)
    broken = _BrokenRedis()
    monkeypatch.setattr(ec.redis_client, "get_redis", lambda: broken)

    assert await ec.cached_subscription_status() == {
        "status": "active", "character_count": 1, "character_limit": 2}


async def test_the_cache_entry_expires(monkeypatch):
    """A TTL, unlike the circuit breaker's deliberate lack of one. Billing can
    recover on its own the moment an invoice is paid, so a stale 'past_due'
    must not outlive the fix and keep refusing healthy dials."""
    monkeypatch.setattr(
        ec, "subscription_status",
        lambda: {"status": "active", "character_count": 1, "character_limit": 2})
    redis = _FakeRedis()
    monkeypatch.setattr(ec.redis_client, "get_redis", lambda: redis)

    await ec.cached_subscription_status()

    assert redis.set_calls, "the result should have been cached"
    assert redis.set_calls[0][2] == ec._SUBSCRIPTION_CACHE_TTL_S


async def test_an_api_failure_propagates_rather_than_being_cached(monkeypatch):
    """The caller decides what an unreachable ElevenLabs means — preflight
    refuses to dial, the dashboard shows 'Could not check'. Swallowing it into
    a cached fake-healthy value would let a dead account dial."""
    def boom():
        raise RuntimeError("elevenlabs unreachable")

    monkeypatch.setattr(ec, "subscription_status", boom)
    redis = _FakeRedis()
    monkeypatch.setattr(ec.redis_client, "get_redis", lambda: redis)

    with pytest.raises(RuntimeError):
        await ec.cached_subscription_status()
    assert redis.store == {}, "a failure must not be cached as a result"
