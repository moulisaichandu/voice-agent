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


async def test_aexit_flushes_unlogged_audio_health_on_teardown(connected, caplog):
    """A call that ends without a clean final END_SPEECH — the lead hangs up
    mid-utterance, or no VAD boundary ever fires — must not silently discard
    whatever was accumulated. These are exactly the pathological calls most
    worth measuring, so __aexit__ flushes them rather than losing the data.
    """
    caplog.set_level("INFO")
    # Only a START_SPEECH ever arrives — the call ends mid-utterance, before
    # any END_SPEECH would normally trigger a log line.
    connected(_FakeSTTWS([_vad("START_SPEECH")]))

    quiet = (200).to_bytes(2, "little", signed=True) * 80
    loud = (12000).to_bytes(2, "little", signed=True) * 80
    quiet_b64 = base64.b64encode(ulaw.encode(quiet)).decode()
    loud_b64 = base64.b64encode(ulaw.encode(loud)).decode()

    async with sarvam_stt.SarvamSTT(lead_id="lead-flush") as stt:
        await stt.send_audio(quiet_b64)  # before START_SPEECH: silence window
        events_iter = stt.events()
        started = await events_iter.__anext__()
        assert started.kind == "speech_started"
        await stt.send_audio(loud_b64)   # after START_SPEECH: speech window
        # the lead hangs up here — no END_SPEECH ever arrives

    expected_silence = sarvam_stt._rms(*sarvam_stt._sumsq_and_count(ulaw.decode(
        base64.b64decode(quiet_b64))))
    expected_speech = sarvam_stt._rms(*sarvam_stt._sumsq_and_count(ulaw.decode(
        base64.b64decode(loud_b64))))

    lines = [r.message for r in caplog.records if "audio health" in r.message]
    assert lines, "no audio-health line was flushed on teardown despite tracked frames"
    line = lines[0]
    assert "lead=lead-flush" in line
    assert f"silence_rms={expected_silence:.1f}" in line
    assert f"speech_rms={expected_speech:.1f}" in line
    assert "frames=2" in line


async def test_aexit_logs_nothing_when_no_frames_were_ever_tracked(connected, caplog):
    """A call that enters and exits without a single tracked frame has
    nothing to report — flushing here would spam an empty line for every
    connection that never received audio (e.g. a call that failed before
    Plivo ever streamed anything)."""
    caplog.set_level("INFO")
    connected(_FakeSTTWS([]))

    async with sarvam_stt.SarvamSTT(lead_id="lead-empty"):
        pass

    lines = [r.message for r in caplog.records if "audio health" in r.message]
    assert not lines, f"expected no audio-health line for zero tracked frames, got: {lines}"


async def test_an_unpaired_end_speech_reports_zero_speech_rms_not_a_stale_value(
    connected, caplog,
):
    """END_SPEECH firing twice in a row with no intervening START_SPEECH is a
    real reachable scenario (see test_sarvam_bridge.py's
    test_the_lead_id_reaches_the_audio_health_log, which drives a bare
    END_SPEECH with no prior START_SPEECH). The second line must honestly
    report that no speech was measured for it (speech_rms=0.0, matching
    _rms's own "0 samples = nothing measured" convention) — not carry over
    the previous utterance's speech_rms.
    """
    caplog.set_level("INFO")
    connected(_FakeSTTWS([
        _vad("START_SPEECH"), _vad("END_SPEECH"),
        _vad("END_SPEECH"),
    ]))

    loud = (12000).to_bytes(2, "little", signed=True) * 80
    loud_b64 = base64.b64encode(ulaw.encode(loud)).decode()

    async with sarvam_stt.SarvamSTT(lead_id="lead-unpaired") as stt:
        events_iter = stt.events()

        # utterance 1 — real speech, so line 1 has a nonzero speech_rms
        started = await events_iter.__anext__()
        assert started.kind == "speech_started"
        await stt.send_audio(loud_b64)
        ended_1 = await events_iter.__anext__()
        assert ended_1.kind == "speech_ended"

        # a second, unpaired END_SPEECH — no START_SPEECH, no frames at all
        ended_2 = await events_iter.__anext__()
        assert ended_2.kind == "speech_ended"

    expected_speech_1 = sarvam_stt._rms(*sarvam_stt._sumsq_and_count(ulaw.decode(
        base64.b64decode(loud_b64))))

    lines = [r.message for r in caplog.records if "audio health" in r.message]
    assert len(lines) == 2, f"expected exactly 2 audio-health lines, got {lines}"
    assert f"speech_rms={expected_speech_1:.1f}" in lines[0]
    assert expected_speech_1 != pytest.approx(0.0), (
        "test setup bug: utterance 1's speech_rms must be nonzero for this "
        "test to actually discriminate a stale carry-over from a real reset"
    )
    assert "speech_rms=0.0" in lines[1], (
        f"expected the unpaired END_SPEECH to report speech_rms=0.0 (nothing "
        f"measured), got: {lines[1]!r}"
    )


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
    exactly like a lead who never speaks, and costs the full call duration.

    The fixture says "message", not "error", because that is the key Sarvam
    actually populates — confirmed live:
    {'type': 'error', 'data': {'message': 'Insufficient credits'}}."""
    connected(_FakeSTTWS([json.dumps(
        {"type": "error", "data": {"message": "bad language", "code": "400"}}
    )]))

    async with sarvam_stt.SarvamSTT() as stt:
        events = [e async for e in stt.events()]

    assert events == []


async def test_the_error_message_itself_is_logged_not_the_raw_frame(connected, caplog):
    """REGRESSION. The handler read data['error'], a key Sarvam never sends,
    so it fell through to dumping the whole event dict — seen live as
    "refused to transcribe: {'type': 'error', 'data': {...}} (code=None)".

    Cosmetic on its own, and NOT cosmetic here: the insufficient-credits
    substring match below runs on this same extracted message, so with the
    wrong key it could never have fired no matter what Sarvam sent."""
    connected(_FakeSTTWS([json.dumps(
        {"type": "error", "data": {"message": "bad language", "code": "400"}}
    )]))

    with caplog.at_level("ERROR"):
        async with sarvam_stt.SarvamSTT() as stt:
            [e async for e in stt.events()]

    assert "bad language" in caplog.text
    assert "'type': 'error'" not in caplog.text, "logged the raw frame, not the message"


async def test_an_insufficient_credits_error_trips_the_circuit_breaker(
    connected, monkeypatch
):
    """The exact signal observed live: Sarvam's STT reports the account is out
    of credits and the stream dies. Every later call on this backend fails the
    same way until an admin clears the breaker, so it must stop dialling."""
    tripped = {}

    async def fake_trip(reason):
        tripped["reason"] = reason

    monkeypatch.setattr(sarvam_stt.sarvam_circuit_breaker, "trip", fake_trip)
    connected(_FakeSTTWS([json.dumps(
        {"type": "error", "data": {"message": "Insufficient credits", "code": "402"}}
    )]))

    async with sarvam_stt.SarvamSTT() as stt:
        events = [e async for e in stt.events()]

    assert events == []
    assert tripped["reason"] == "Insufficient credits"


async def test_an_unrelated_error_does_not_trip_the_circuit_breaker(
    connected, monkeypatch
):
    """Scoped on purpose. Tripping on any error would halt every Sarvam
    campaign over an ordinary transient blip — a call that would have
    succeeded on the next attempt."""
    async def must_not_be_called(reason):
        raise AssertionError("an unrelated error must not trip the breaker")

    monkeypatch.setattr(sarvam_stt.sarvam_circuit_breaker, "trip", must_not_be_called)
    connected(_FakeSTTWS([json.dumps(
        {"type": "error", "data": {"message": "bad language", "code": "400"}}
    )]))

    async with sarvam_stt.SarvamSTT() as stt:
        events = [e async for e in stt.events()]

    assert events == []


async def test_an_error_frame_with_no_message_does_not_crash(connected, monkeypatch):
    """message falls back to the whole event dict when Sarvam sends no
    'message' key. Calling .lower() on a dict would raise inside the error
    handler — turning a reported failure into an unreported one."""
    async def fake_trip(reason):
        raise AssertionError("a frame with no message cannot be a credits failure")

    monkeypatch.setattr(sarvam_stt.sarvam_circuit_breaker, "trip", fake_trip)
    connected(_FakeSTTWS([json.dumps({"type": "error", "data": {}})]))

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


# ── the close frame: how Sarvam ACTUALLY reported the outage ─────────────────
#
# Everything above tests {"type": "error"} frames. On 2026-08-13, with the
# account genuinely out of credits, the STT leg sent no error frame at all — it
# closed the socket:
#
#     ConnectionClosedError: received 1003 (unsupported data)
#     Credits exhausted. Visit the API Dashboard to review and manage your
#     subscription.
#
# events() never saw a frame, so the breaker could not fire however well it
# matched the wording. A close frame is a first-class failure signal here, not
# a transport detail.

from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close

_CREDITS_CLOSE = ("Credits exhausted. Visit the API Dashboard to review "
                  "and manage your subscription.")


class _ClosingSTTWS(_FakeSTTWS):
    """A socket that yields its messages and then closes with *close*."""

    def __init__(self, messages: list[str], close: Close):
        super().__init__(messages)
        self._close = close

    def __aiter__(self):
        async def _gen():
            for m in self._messages:
                yield m
            raise ConnectionClosedError(self._close, None)
        return _gen()


async def test_a_credits_close_frame_trips_the_circuit_breaker(connected, monkeypatch):
    """THE bug this section exists for. No error frame is ever sent."""
    tripped = {}

    async def fake_trip(reason):
        tripped["reason"] = reason

    monkeypatch.setattr(sarvam_stt.sarvam_circuit_breaker, "trip", fake_trip)
    connected(_ClosingSTTWS([], Close(1003, _CREDITS_CLOSE)))

    with pytest.raises(ConnectionClosedError):
        async with sarvam_stt.SarvamSTT() as stt:
            [e async for e in stt.events()]

    assert tripped["reason"] == _CREDITS_CLOSE


async def test_a_credits_close_still_ends_the_call(connected, monkeypatch):
    """Tripping the breaker must not SWALLOW the close. Returning quietly
    would leave the bridge with a listener that has ended and a call that has
    not — silence until the watchdog, billed, and recorded as a clean exit."""
    async def fake_trip(reason):
        pass

    monkeypatch.setattr(sarvam_stt.sarvam_circuit_breaker, "trip", fake_trip)
    connected(_ClosingSTTWS([], Close(1003, _CREDITS_CLOSE)))

    with pytest.raises(ConnectionClosedError):
        async with sarvam_stt.SarvamSTT() as stt:
            [e async for e in stt.events()]


async def test_transcripts_before_a_close_are_still_delivered(connected, monkeypatch):
    """The close ends the stream; it does not retract what the lead already
    said. A turn dropped here is one the transcript loses forever."""
    async def fake_trip(reason):
        pass

    monkeypatch.setattr(sarvam_stt.sarvam_circuit_breaker, "trip", fake_trip)
    connected(_ClosingSTTWS(
        [json.dumps({"type": "data", "data": {"transcript": "అవును"}})],
        Close(1003, _CREDITS_CLOSE),
    ))

    heard = []
    with pytest.raises(ConnectionClosedError):
        async with sarvam_stt.SarvamSTT() as stt:
            async for e in stt.events():
                heard.append(e)

    assert [e.text for e in heard] == ["అవును"]


async def test_an_ordinary_close_does_not_trip_the_breaker(connected, monkeypatch):
    """A dropped connection reports code 1006 and an EMPTY reason — including
    when we are the side that closed. Tripping on that would ground every
    Sarvam campaign the first time a call ended untidily."""
    async def must_not_be_called(reason):
        raise AssertionError("an ordinary disconnect must not trip the breaker")

    monkeypatch.setattr(sarvam_stt.sarvam_circuit_breaker, "trip", must_not_be_called)
    connected(_ClosingSTTWS([], Close(1006, "")))

    with pytest.raises(ConnectionClosedError):
        async with sarvam_stt.SarvamSTT() as stt:
            [e async for e in stt.events()]


async def test_an_unrelated_close_reason_does_not_trip_the_breaker(
    connected, monkeypatch
):
    async def must_not_be_called(reason):
        raise AssertionError("an unrelated close must not trip the breaker")

    monkeypatch.setattr(sarvam_stt.sarvam_circuit_breaker, "trip", must_not_be_called)
    connected(_ClosingSTTWS([], Close(1011, "internal error")))

    with pytest.raises(ConnectionClosedError):
        async with sarvam_stt.SarvamSTT() as stt:
            [e async for e in stt.events()]


async def test_the_close_reason_is_logged(connected, monkeypatch, caplog):
    """The live symptom was a bare 'conversation failed: ConnectionClosedOK'
    in the bridge, which says nothing about WHY. The reason Sarvam sent is the
    whole diagnosis."""
    async def fake_trip(reason):
        pass

    monkeypatch.setattr(sarvam_stt.sarvam_circuit_breaker, "trip", fake_trip)
    connected(_ClosingSTTWS([], Close(1003, _CREDITS_CLOSE)))

    with caplog.at_level("ERROR"):
        with pytest.raises(ConnectionClosedError):
            async with sarvam_stt.SarvamSTT() as stt:
                [e async for e in stt.events()]

    assert "Credits exhausted" in caplog.text
    assert "1003" in caplog.text


async def test_a_normal_close_ends_the_stream_without_raising(connected, monkeypatch):
    """websockets swallows ConnectionClosedOK inside its own __aiter__, but
    this module must not turn a tidy end-of-call into an exception if it ever
    surfaces one."""
    async def must_not_be_called(reason):
        raise AssertionError("a normal close is not a credits failure")

    monkeypatch.setattr(sarvam_stt.sarvam_circuit_breaker, "trip", must_not_be_called)

    class _OKClosing(_FakeSTTWS):
        def __aiter__(self):
            async def _gen():
                yield json.dumps({"type": "data", "data": {"transcript": "సరే"}})
                raise ConnectionClosedOK(Close(1000, ""), None)
            return _gen()

    connected(_OKClosing([]))

    async with sarvam_stt.SarvamSTT() as stt:
        heard = [e async for e in stt.events()]

    assert [e.text for e in heard] == ["సరే"]
