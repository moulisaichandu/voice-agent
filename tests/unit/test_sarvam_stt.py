"""Unit tests for telephony/sarvam_stt.py — hearing the lead.

Same fake-socket style as the other Sarvam tests: no network, no Docker.

Two things here are load-bearing beyond "does it parse JSON":

  * the language MUST be pinned. On 8 kHz phone audio, auto-detection was
    observed hearing Telugu as Croatian and Urdu — the lead is transcribed as
    nonsense and the agent answers the nonsense, fluently. That failure is
    recorded in app/config.py's OPENAI_REALTIME_STT_LANGUAGE comment and cost
    the sibling project real calls;

  * the audio must be DECODED. Sarvam's STT does not accept mu-law, so sending
    Plivo's bytes through untouched would be sending noise that transcribes to
    nothing, on every call, silently.
"""

import asyncio
import base64
import json

import pytest

from app.telephony import sarvam_stt, ulaw

_MULAW_FRAME = b"\x7f" * 160          # 20 ms at 8 kHz
_MULAW_B64 = base64.b64encode(_MULAW_FRAME).decode()


def _transcript(text: str) -> str:
    return json.dumps({"type": "data", "data": {"transcript": text,
                                                "request_id": "r1"}})


def _vad(signal: str) -> str:
    return json.dumps({"type": "events", "data": {"signal_type": signal,
                                                  "occured_at": 1.0}})


class _FakeSTTWS:
    def __init__(self, messages: list[str]):
        self._messages = list(messages)
        self.sent: list[str] = []

    async def send(self, payload):
        self.sent.append(payload)

    async def close(self):
        pass

    def __aiter__(self):
        async def _gen():
            for m in self._messages:
                yield m
            await asyncio.Event().wait()
        return _gen()


@pytest.fixture
def connected(monkeypatch):
    monkeypatch.setattr(sarvam_stt, "SARVAM_API_KEY", "sk-test")
    captured: dict = {}

    def _install(ws):
        async def fake_connect(url, headers):
            captured["url"] = url
            captured["headers"] = headers
            return ws
        monkeypatch.setattr(sarvam_stt, "_connect", fake_connect)
        return captured

    return _install


# ── the connection ───────────────────────────────────────────────────────────

async def test_the_language_is_pinned_on_the_connection(connected):
    """THE most load-bearing parameter in this module. Auto-detection on 8 kHz
    Telugu was observed guessing Croatian and Urdu; the lead is then
    transcribed as nonsense and the agent answers it."""
    captured = connected(_FakeSTTWS([]))

    async with sarvam_stt.SarvamSTT():
        pass

    assert f"language-code={sarvam_stt.SARVAM_STT_LANGUAGE}" in captured["url"]


async def test_the_connection_asks_for_telephony_audio_and_vad(connected):
    """8000 because that is what a phone line is, and vad_signals because
    START_SPEECH is the only barge-in signal this backend gets — without it
    the lead cannot interrupt the agent at all."""
    captured = connected(_FakeSTTWS([]))

    async with sarvam_stt.SarvamSTT():
        pass

    url = captured["url"]
    assert url.startswith("wss://api.sarvam.ai/speech-to-text/ws")
    assert "sample_rate=8000" in url
    assert "vad_signals=true" in url
    assert f"model={sarvam_stt.SARVAM_STT_MODEL}" in url
    assert captured["headers"]["Api-Subscription-Key"] == "sk-test"


async def test_no_api_key_raises_before_connecting(monkeypatch):
    monkeypatch.setattr(sarvam_stt, "SARVAM_API_KEY", None)

    async def boom(*a, **kw):
        raise AssertionError("must not connect without a key")

    monkeypatch.setattr(sarvam_stt, "_connect", boom)

    with pytest.raises(sarvam_stt.SarvamNotConfigured):
        async with sarvam_stt.SarvamSTT():
            pass


# ── sending the lead's audio ─────────────────────────────────────────────────

async def test_plivo_mulaw_is_decoded_to_pcm_before_being_sent(connected):
    """Sarvam's STT accepts wav/pcm only — mu-law is not in its codec list. Send
    Plivo's bytes through untouched and Sarvam transcribes noise as nothing,
    on every call, with no error from either side."""
    ws = _FakeSTTWS([])
    connected(ws)

    async with sarvam_stt.SarvamSTT() as stt:
        await stt.send_audio(_MULAW_B64)

    frame = json.loads(ws.sent[0])["audio"]
    # 'audio/wav', NOT 'audio/pcm_s16le'. Those are two different fields with
    # two different enums, and conflating them is how this was originally
    # written: the CONNECTION's input_audio_codec query param accepts
    # pcm_s16le, while the per-message encoding field accepts only audio/wav.
    # Sending pcm_s16le here made Sarvam reject the stream outright — verified
    # live: "audio.encoding: Input should be 'audio/wav'" — which on a real
    # call means the agent is completely deaf.
    #
    # No WAV container is needed; the bytes stay raw PCM16, as the codec param
    # already declared. Confirmed by transcribing real Telugu speech through
    # this exact frame shape.
    assert frame["encoding"] == "audio/wav"
    assert frame["sample_rate"] == "8000"
    assert base64.b64decode(frame["data"]) == ulaw.decode(_MULAW_FRAME)
    assert len(base64.b64decode(frame["data"])) == 2 * len(_MULAW_FRAME)


async def test_an_undecodable_frame_is_dropped_rather_than_killing_the_call(connected):
    """One corrupt frame should cost the lead 20 ms of audio, not the call."""
    ws = _FakeSTTWS([])
    connected(ws)

    async with sarvam_stt.SarvamSTT() as stt:
        await stt.send_audio("!!!not base64!!!")
        await stt.send_audio(_MULAW_B64)

    assert len(ws.sent) == 1, "the bad frame should have been dropped, the good one sent"


# ── what comes back ──────────────────────────────────────────────────────────

async def test_a_transcript_is_surfaced_as_a_transcript_event(connected):
    connected(_FakeSTTWS([_transcript("ఫీజు ఎంత?")]))

    async with sarvam_stt.SarvamSTT() as stt:
        events = []
        async for event in stt.events():
            events.append(event)
            break

    assert events[0].kind == "transcript"
    assert events[0].text == "ఫీజు ఎంత?"


async def test_speech_start_and_end_are_surfaced(connected):
    connected(_FakeSTTWS([_vad("START_SPEECH"), _vad("END_SPEECH")]))

    async with sarvam_stt.SarvamSTT() as stt:
        kinds = []
        async for event in stt.events():
            kinds.append(event.kind)
            if len(kinds) == 2:
                break

    assert kinds == ["speech_started", "speech_ended"]


async def test_a_blank_transcript_is_not_surfaced(connected):
    """Sarvam emits empty transcripts for non-speech. Treating one as a turn
    would have the agent answer a cough."""
    connected(_FakeSTTWS([_transcript("   "), _transcript("అవును")]))

    async with sarvam_stt.SarvamSTT() as stt:
        async for event in stt.events():
            first = event
            break

    assert first.text == "అవును"


async def test_an_error_frame_ends_the_stream_rather_than_hanging(connected):
    """Same reasoning as the TTS client: a rejected connection otherwise looks
    exactly like a lead who never speaks, and costs the full call duration."""
    connected(_FakeSTTWS([json.dumps(
        {"type": "error", "data": {"error": "bad language", "code": "400"}}
    )]))

    async with sarvam_stt.SarvamSTT() as stt:
        events = [e async for e in stt.events()]

    assert events == []


async def test_unparseable_frames_are_skipped(connected):
    connected(_FakeSTTWS(["not json at all", _transcript("సరే")]))

    async with sarvam_stt.SarvamSTT() as stt:
        async for event in stt.events():
            first = event
            break

    assert first.text == "సరే"
