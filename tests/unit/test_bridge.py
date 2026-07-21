"""Unit tests for telephony/bridge.py's pure parts — the answer XML.

The bridge loop itself needs two live WebSockets and is exercised end to end
against a real call; what is unit-testable here is the XML Plivo fetches on
answer, which is easy to get subtly wrong and fails in a way that is very hard
to diagnose from the outside (the call rings, connects, and is silent).
"""

import pytest

from app.telephony import bridge


@pytest.fixture(autouse=True)
def _public_url(monkeypatch):
    monkeypatch.setattr(bridge, "PUBLIC_BASE_URL", "https://example.trycloudflare.com")
    monkeypatch.setattr(bridge, "CALL_WEBHOOK_SECRET", "s3cret")


def test_answer_xml_points_at_the_websocket_over_wss():
    """https must become wss, not stay https — Plivo silently fails to open
    the stream otherwise, and the call connects to silence."""
    xml = bridge.answer_xml("lead-123")
    assert "wss://example.trycloudflare.com/calls/stream" in xml
    assert "https://" not in xml


def test_answer_xml_declares_mulaw_8000():
    """Plivo must be told the stream is mu-law 8 kHz. Any other framing and
    the caller hears noise, because the agent's audio really is mu-law."""
    xml = bridge.answer_xml("lead-123")
    assert 'contentType="audio/x-mulaw;rate=8000"' in xml


def test_answer_xml_is_bidirectional_and_keeps_the_call_alive():
    """Without keepCallAlive Plivo treats the rest of the (empty) document as
    the whole call and hangs up the moment the stream is established."""
    xml = bridge.answer_xml("lead-123")
    assert 'bidirectional="true"' in xml
    assert 'keepCallAlive="true"' in xml


def test_answer_xml_carries_the_lead_and_token():
    xml = bridge.answer_xml("lead-123")
    assert "lead=lead-123" in xml
    assert "token=s3cret" in xml


def test_answer_xml_escapes_the_url():
    """The URL goes inside an XML element, so & between query params must be
    escaped or Plivo's parser rejects the document."""
    xml = bridge.answer_xml("lead-123")
    assert "&amp;" in xml
    # A bare & would mean the ampersand was left unescaped.
    assert "&lead=" not in xml.replace("&amp;lead=", "")


def test_answer_xml_handles_a_http_base_url(monkeypatch):
    monkeypatch.setattr(bridge, "PUBLIC_BASE_URL", "http://localhost:8091")
    xml = bridge.answer_xml("lead-1")
    assert "ws://localhost:8091/calls/stream" in xml
