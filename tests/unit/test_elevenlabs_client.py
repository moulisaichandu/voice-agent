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
