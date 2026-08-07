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


# ── audio-health accumulation ────────────────────────────────────────────────

def test_sumsq_and_count_of_known_samples():
    """Two samples, 3 and 4 — sum of squares is 25, count is 2. Chosen so the
    RMS ends up being sqrt(12.5), an easy value to check by hand."""
    pcm = (3).to_bytes(2, "little", signed=True) + (4).to_bytes(2, "little", signed=True)
    assert sarvam_stt._sumsq_and_count(pcm) == (25, 2)


def test_sumsq_and_count_of_empty_pcm_is_zero():
    assert sarvam_stt._sumsq_and_count(b"") == (0, 0)


def test_sumsq_and_count_drops_a_trailing_odd_byte():
    """A stray half-sample must not be read as a sample — same reasoning as
    ulaw.encode()'s trailing-byte handling."""
    pcm = (5).to_bytes(2, "little", signed=True) + b"\x01"
    assert sarvam_stt._sumsq_and_count(pcm) == (25, 1)


def test_rms_of_the_3_4_example():
    assert sarvam_stt._rms(25, 2) == pytest.approx(12.5 ** 0.5)


def test_rms_with_no_samples_is_zero():
    """Zero samples measured, not a silent signal — the caller (send_audio's
    accumulator) never has zero samples for a frame it actually received, but
    the function must not divide by zero if it's ever called with none."""
    assert sarvam_stt._rms(0, 0) == 0.0


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


async def test_audio_health_is_logged_once_per_utterance(connected, caplog):
    """One INFO line at END_SPEECH, correlating this utterance's noise floor
    (before it started) against how loud the speech itself was — the whole
    point being to tell genuine background noise apart from a connection
    problem using real calls, not recorded audio (this project has none).
    See docs/superpowers/specs/2026-08-07-audio-noise-diagnostics-design.md.
    """
    caplog.set_level("INFO")
    connected(_FakeSTTWS([_vad("START_SPEECH"), _vad("END_SPEECH")]))

    # Nonzero, not silence: _rms(0, 0) ("nothing measured") and the RMS of
    # true zero-amplitude audio both render as 0.0, which would make the
    # silence_rms assertion below pass even if silence accumulation were
    # deleted entirely. A small nonzero amplitude keeps the assertion
    # actually discriminating.
    quiet = (200).to_bytes(2, "little", signed=True) * 80
    loud = (12000).to_bytes(2, "little", signed=True) * 80
    quiet_mulaw_b64 = base64.b64encode(ulaw.encode(quiet)).decode()
    loud_mulaw_b64 = base64.b64encode(ulaw.encode(loud)).decode()

    async with sarvam_stt.SarvamSTT(lead_id="lead-9") as stt:
        await stt.send_audio(quiet_mulaw_b64)  # before START_SPEECH: silence window
        events_iter = stt.events()
        started = await events_iter.__anext__()
        assert started.kind == "speech_started"
        await stt.send_audio(loud_mulaw_b64)   # after START_SPEECH: speech window
        ended = await events_iter.__anext__()
        assert ended.kind == "speech_ended"

    expected_silence = sarvam_stt._rms(*sarvam_stt._sumsq_and_count(ulaw.decode(
        base64.b64decode(quiet_mulaw_b64))))
    expected_speech = sarvam_stt._rms(*sarvam_stt._sumsq_and_count(ulaw.decode(
        base64.b64decode(loud_mulaw_b64))))

    lines = [r.message for r in caplog.records if "audio health" in r.message]
    assert lines, "no audio-health line was logged for the utterance"
    line = lines[0]
    assert "lead=lead-9" in line
    assert f"silence_rms={expected_silence:.1f}" in line
    assert f"speech_rms={expected_speech:.1f}" in line
    assert "frames=2" in line


async def test_audio_health_resets_between_consecutive_utterances(connected, caplog):
    """A second utterance's logged line must reflect only ITS OWN frames —
    not a running total across the whole call. Guards against a refactor
    that moves or drops one of the resets (e.g. the speech-window reset
    currently at START_SPEECH): if silence/speech/frame accumulators ever
    stopped resetting at the right VAD boundary, utterance 2 would silently
    report a blend of utterance 1's and utterance 2's audio, and every line
    after the first would be wrong in a call with real back-and-forth.
    """
    caplog.set_level("INFO")
    connected(_FakeSTTWS([
        _vad("START_SPEECH"), _vad("END_SPEECH"),
        _vad("START_SPEECH"), _vad("END_SPEECH"),
    ]))

    # Deliberately different amplitudes per utterance and per window, so a
    # carry-over bug (utterance 2's line still including utterance 1's
    # samples) produces a visibly wrong number rather than an accidental match.
    quiet_1 = (200).to_bytes(2, "little", signed=True) * 80
    loud_1 = (12000).to_bytes(2, "little", signed=True) * 80
    quiet_2 = (900).to_bytes(2, "little", signed=True) * 80
    loud_2 = (6000).to_bytes(2, "little", signed=True) * 80
    quiet_1_b64 = base64.b64encode(ulaw.encode(quiet_1)).decode()
    loud_1_b64 = base64.b64encode(ulaw.encode(loud_1)).decode()
    quiet_2_b64 = base64.b64encode(ulaw.encode(quiet_2)).decode()
    loud_2_b64 = base64.b64encode(ulaw.encode(loud_2)).decode()

    async with sarvam_stt.SarvamSTT(lead_id="lead-two") as stt:
        events_iter = stt.events()

        # utterance 1
        await stt.send_audio(quiet_1_b64)
        started_1 = await events_iter.__anext__()
        assert started_1.kind == "speech_started"
        await stt.send_audio(loud_1_b64)
        ended_1 = await events_iter.__anext__()
        assert ended_1.kind == "speech_ended"

        # utterance 2 — its own silence and speech windows
        await stt.send_audio(quiet_2_b64)
        started_2 = await events_iter.__anext__()
        assert started_2.kind == "speech_started"
        await stt.send_audio(loud_2_b64)
        ended_2 = await events_iter.__anext__()
        assert ended_2.kind == "speech_ended"

    expected_silence_2 = sarvam_stt._rms(*sarvam_stt._sumsq_and_count(ulaw.decode(
        base64.b64decode(quiet_2_b64))))
    expected_speech_2 = sarvam_stt._rms(*sarvam_stt._sumsq_and_count(ulaw.decode(
        base64.b64decode(loud_2_b64))))

    lines = [r.message for r in caplog.records if "audio health" in r.message]
    assert len(lines) == 2, f"expected exactly 2 audio-health lines, got {lines}"
    second_line = lines[1]
    assert "lead=lead-two" in second_line
    assert f"silence_rms={expected_silence_2:.1f}" in second_line
    assert f"speech_rms={expected_speech_2:.1f}" in second_line
    assert "frames=2" in second_line


async def test_lead_id_defaults_to_empty_string(connected, caplog):
    """The constructor's lead_id is optional so every other test in this
    file (and every existing call site) keeps constructing SarvamSTT() with
    no arguments."""
    caplog.set_level("INFO")
    connected(_FakeSTTWS([_vad("START_SPEECH"), _vad("END_SPEECH")]))

    async with sarvam_stt.SarvamSTT() as stt:
        async for event in stt.events():
            if event.kind == "speech_ended":
                break

    lines = [r.message for r in caplog.records if "audio health" in r.message]
    assert lines, "no audio-health line was logged"
    assert "lead= audio health" in lines[0], (
        f"expected an empty lead_id to render as 'lead= audio health', got: {lines[0]!r}"
    )


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
