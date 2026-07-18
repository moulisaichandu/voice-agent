"""Unit tests for telephony/elevenlabs_client.py — the real SDK is mocked at
the app.telephony.elevenlabs_client.get_client() seam, so these run with zero
network calls. The shapes mocked here were verified by directly introspecting
the installed elevenlabs==2.58.0 SDK, not assumed."""

from types import SimpleNamespace

import pytest
from elevenlabs.errors import NotFoundError

from app.telephony import elevenlabs_client as ec


class _FakeExotel:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.last_call_kwargs = None

    def outbound_call(self, **kwargs):
        self.last_call_kwargs = kwargs
        if self._error:
            raise self._error
        return self._response


class _FakeAgents:
    def __init__(self, exists=True):
        self._exists = exists

    def get(self, agent_id):
        if not self._exists:
            raise NotFoundError(body={"detail": "not found"})
        return SimpleNamespace(agent_id=agent_id)


class _FakePhoneNumbers:
    def __init__(self, exists=True):
        self._exists = exists

    def get(self, phone_number_id):
        if not self._exists:
            raise NotFoundError(body={"detail": "not found"})
        return SimpleNamespace(phone_number_id=phone_number_id)


class _FakeWebhooks:
    def __init__(self, event=None, error=None):
        self._event = event
        self._error = error

    def construct_event(self, rawBody, sig_header, secret):
        if self._error:
            raise self._error
        return self._event


def _fake_client(*, exotel=None, agents=None, phone_numbers=None, webhooks=None):
    return SimpleNamespace(conversational_ai=SimpleNamespace(
        exotel=exotel or _FakeExotel(),
        agents=agents or _FakeAgents(),
        phone_numbers=phone_numbers or _FakePhoneNumbers(),
    ), webhooks=webhooks or _FakeWebhooks())


def test_place_outbound_call_maps_response_and_injects_dynamic_variables(monkeypatch):
    fake_response = SimpleNamespace(
        success=True, message="ok", conversation_id="conv_123", call_sid="call_abc",
    )
    exotel = _FakeExotel(response=fake_response)
    monkeypatch.setattr(ec, "get_client", lambda: _fake_client(exotel=exotel))

    result = ec.place_outbound_call(
        agent_id="agent_1", agent_phone_number_id="phone_1", to_number="+919876543210",
        dynamic_variables={"lead_id": "lead_xyz"},
    )

    assert result.success is True
    assert result.conversation_id == "conv_123"
    assert result.call_sid == "call_abc"
    # dynamic_variables actually reached the SDK call, not just accepted and dropped
    init_data = exotel.last_call_kwargs["conversation_initiation_client_data"]
    assert init_data.dynamic_variables == {"lead_id": "lead_xyz"}
    assert exotel.last_call_kwargs["agent_id"] == "agent_1"
    assert exotel.last_call_kwargs["to_number"] == "+919876543210"


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
