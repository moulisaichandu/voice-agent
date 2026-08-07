"""Unit tests for telephony/sarvam_tts.py — the Telugu voice.

Driven against a fake Sarvam TTS socket, the same hand-rolled style
tests/unit/test_openai_bridge.py uses: no network, no Docker, no mock library.

The behaviour that matters most here is the audio FORMAT. Sarvam's TTS defaults
to 24 kHz MP3, which is a perfectly good podcast and complete noise on a phone
call. Plivo is told every payload is mu-law 8 kHz, so anything else is misframed
and the lead hears static — with no error anywhere, because both sides think
they did their job.
"""

import asyncio
import base64
import json

import pytest

from app.telephony import sarvam_tts

# 160 bytes of mu-law is exactly 20 ms at 8 kHz, one byte per sample — the
# frame size Plivo itself streams in.
_FRAME = base64.b64encode(b"\x7f" * 160).decode()


def _audio(b64: str = _FRAME) -> str:
    return json.dumps({"type": "audio", "data": {"audio": b64,
                                                 "request_id": "req-1"}})


def _final() -> str:
    return json.dumps({"type": "event", "data": {"event_type": "final"}})


class _FakeSarvamWS:
    """Yields *messages*, then goes quiet — a socket that has said everything
    it is going to say but has not been closed by the far end."""

    def __init__(self, messages: list[str]):
        self._messages = list(messages)
        self.sent: list[str] = []
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def send(self, payload):
        self.sent.append(payload)

    async def close(self):
        self.closed = True

    def __aiter__(self):
        async def _gen():
            for m in self._messages:
                yield m
            await asyncio.Event().wait()
        return _gen()


@pytest.fixture
def connected(monkeypatch):
    """Install a fake socket in place of the real connect()."""
    monkeypatch.setattr(sarvam_tts, "SARVAM_API_KEY", "sk-test")

    def _install(ws):
        async def fake_connect(*a, **kw):
            return ws
        monkeypatch.setattr(sarvam_tts, "_connect", fake_connect)
        return ws

    return _install


def _config(ws) -> dict:
    """The config frame — the first thing sent after connecting."""
    return json.loads(ws.sent[0])


# ── the audio format contract ────────────────────────────────────────────────

async def test_the_config_pins_mulaw_at_8kHz(connected):
    """The single most important frame in this module. Sarvam defaults to
    24 kHz MP3; Plivo is told every payload is mu-law 8 kHz. Get this wrong and
    the lead hears static while every log line says the call succeeded."""
    ws = connected(_FakeSarvamWS([_audio(), _final()]))

    async with sarvam_tts.SarvamTTS(language="te-IN") as tts:
        await tts.collect("నమస్తే")

    cfg = _config(ws)["data"]
    assert cfg["output_audio_codec"] == "mulaw"
    assert cfg["speech_sample_rate"] == "8000"


async def test_the_config_carries_the_language_and_the_configured_voice(connected):
    """The speaker and model come from .env so the voice can be changed after a
    live listen without a deploy — assert they are forwarded, not what they
    currently happen to be."""
    ws = connected(_FakeSarvamWS([_audio(), _final()]))

    async with sarvam_tts.SarvamTTS(language="te-IN") as tts:
        await tts.collect("నమస్తే")

    cfg = _config(ws)["data"]
    assert cfg["target_language_code"] == "te-IN"
    assert cfg["speaker"] == sarvam_tts.SARVAM_TTS_SPEAKER
    assert cfg["model"] == sarvam_tts.SARVAM_TTS_MODEL


async def test_the_speaker_can_be_overridden_per_connection(connected):
    """Choosing a Telugu voice means hearing the same line in several of them
    at real phone quality. Without this the only way to switch voices is a
    process-wide config value, so a comparison tool would have to reach in and
    mutate module state — and would then be testing something the live call
    path does not do."""
    ws = connected(_FakeSarvamWS([_audio(), _final()]))

    async with sarvam_tts.SarvamTTS(language="te-IN", speaker="kavitha") as tts:
        await tts.collect("నమస్తే")

    assert _config(ws)["data"]["speaker"] == "kavitha"


async def test_the_configured_speaker_is_still_the_default(connected):
    """The override is for tools. A real call must keep taking its voice from
    .env, so the voice can be changed without a deploy."""
    ws = connected(_FakeSarvamWS([_audio(), _final()]))

    async with sarvam_tts.SarvamTTS(language="te-IN") as tts:
        await tts.collect("నమస్తే")

    assert _config(ws)["data"]["speaker"] == sarvam_tts.SARVAM_TTS_SPEAKER


# ── streaming behaviour ──────────────────────────────────────────────────────

async def test_text_is_sent_then_flushed(connected):
    """Sarvam buffers text until told to flush. Without the flush the last
    sentence of every call is generated but never spoken."""
    ws = connected(_FakeSarvamWS([_audio(), _final()]))

    async with sarvam_tts.SarvamTTS(language="te-IN") as tts:
        await tts.collect("ఇది ఒక పరీక్ష")

    kinds = [json.loads(s).get("type") for s in ws.sent]
    assert kinds == ["config", "text", "flush"]
    assert json.loads(ws.sent[1])["data"]["text"] == "ఇది ఒక పరీక్ష"


async def test_audio_chunks_arrive_with_their_duration(connected):
    """duration_ms is what Milestone B's barge-in ledger needs to work out how
    much of a sentence the lead actually heard. Mu-law 8 kHz is one byte per
    sample, so 160 bytes is exactly 20 ms."""
    connected(_FakeSarvamWS([_audio(), _audio(), _final()]))

    async with sarvam_tts.SarvamTTS(language="te-IN") as tts:
        chunks = [c async for c in tts.speak("రెండు")]

    assert [payload for payload, _ in chunks] == [_FRAME, _FRAME]
    assert [ms for _, ms in chunks] == [20, 20]


async def test_the_stream_ends_on_the_final_event(connected):
    """Sarvam does not close the socket after an utterance — it stays open for
    the next one. Without honouring 'final' the caller would await audio that
    is never coming and the call would hang until CALL_MAX_DURATION_S."""
    connected(_FakeSarvamWS([_audio(), _final()]))

    async with sarvam_tts.SarvamTTS(language="te-IN") as tts:
        chunks = await asyncio.wait_for(tts.collect("ఒకటి"), timeout=5)

    assert len(chunks) == 1


async def test_cancelled_synthesis_reconnects_before_the_next_turn(monkeypatch):
    """A barge-in must not leave stale audio/final frames on the reusable
    socket, or the next answer can finish immediately without speaking."""
    monkeypatch.setattr(sarvam_tts, "SARVAM_API_KEY", "sk-test")
    sockets = [
        _FakeSarvamWS([]),
        _FakeSarvamWS([_audio(), _final()]),
    ]
    connected_sockets = []

    async def fake_connect(*a, **kw):
        ws = sockets.pop(0)
        connected_sockets.append(ws)
        return ws

    monkeypatch.setattr(sarvam_tts, "_connect", fake_connect)

    async with sarvam_tts.SarvamTTS(language="te-IN") as tts:
        task = asyncio.create_task(tts.collect("first answer"))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        chunks = await tts.collect("second answer")

    assert len(connected_sockets) == 2
    assert chunks == [(_FRAME, 20)]
    assert connected_sockets[0].closed is True


async def test_an_error_frame_stops_the_stream_rather_than_hanging(connected):
    """A rejected config (bad speaker, expired key) arrives as an error frame
    and then silence. Treating it as end-of-utterance turns a 300-second billed
    silence into a call that ends and a line in the log."""
    connected(_FakeSarvamWS([json.dumps(
        {"type": "error", "data": {"message": "invalid speaker", "code": 400}}
    )]))

    async with sarvam_tts.SarvamTTS(language="te-IN") as tts:
        chunks = await asyncio.wait_for(tts.collect("ఒకటి"), timeout=5)

    assert chunks == []


async def test_blank_text_never_opens_a_turn(connected):
    """An empty render must not send an empty text frame and then wait for
    audio Sarvam has no reason to produce."""
    connected(_FakeSarvamWS([]))

    async with sarvam_tts.SarvamTTS(language="te-IN") as tts:
        chunks = await asyncio.wait_for(tts.collect("   "), timeout=5)

    assert chunks == []


# ── credentials ──────────────────────────────────────────────────────────────

async def test_no_api_key_raises_before_connecting(monkeypatch):
    """Fail where the caller can turn it into a failed outcome, not halfway
    through an answered call."""
    monkeypatch.setattr(sarvam_tts, "SARVAM_API_KEY", None)

    async def boom(*a, **kw):
        raise AssertionError("must not connect without a key")

    monkeypatch.setattr(sarvam_tts, "_connect", boom)

    with pytest.raises(sarvam_tts.SarvamNotConfigured):
        async with sarvam_tts.SarvamTTS(language="te-IN"):
            pass


async def test_the_completion_event_is_requested_explicitly(monkeypatch):
    """REGRESSION, found on the first real call against the live API.

    Sarvam only sends the {"event_type": "final"} frame if the connection asked
    for it. Without send_completion_event=true it sends the audio and then
    NOTHING — the socket simply sits open until the server kills it as idle
    with a 408, which took 25s+ in testing.

    speak() ends its stream on that event, so without this parameter every
    utterance blocks for the server's idle timeout. One-way survives it (the
    audio has already been played by then); two-way would stall on every
    single sentence."""
    monkeypatch.setattr(sarvam_tts, "SARVAM_API_KEY", "sk-test")
    seen: dict = {}

    async def fake_connect(url, headers):
        seen["url"] = url
        return _FakeSarvamWS([])

    monkeypatch.setattr(sarvam_tts, "_connect", fake_connect)

    async with sarvam_tts.SarvamTTS(language="te-IN"):
        pass

    assert "send_completion_event=true" in seen["url"]


async def test_the_key_is_sent_as_the_subscription_header(monkeypatch):
    """Sarvam authenticates with Api-Subscription-Key, not a Bearer token."""
    monkeypatch.setattr(sarvam_tts, "SARVAM_API_KEY", "sk-test")
    seen: dict = {}

    async def fake_connect(url, headers):
        seen["url"] = url
        seen["headers"] = headers
        return _FakeSarvamWS([])

    monkeypatch.setattr(sarvam_tts, "_connect", fake_connect)

    async with sarvam_tts.SarvamTTS(language="te-IN"):
        pass

    assert seen["headers"]["Api-Subscription-Key"] == "sk-test"
    assert seen["url"].startswith("wss://api.sarvam.ai/text-to-speech/ws")
