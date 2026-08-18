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


async def test_an_insufficient_credits_error_trips_the_circuit_breaker(
    connected, monkeypatch
):
    """The same signal from the TTS leg — an account can run out of credits
    mid-synthesis just as easily as at STT."""
    tripped = {}

    async def fake_trip(reason):
        tripped["reason"] = reason

    monkeypatch.setattr(sarvam_tts.sarvam_circuit_breaker, "trip", fake_trip)
    connected(_FakeSarvamWS([json.dumps(
        {"type": "error", "data": {"message": "Insufficient credits", "code": 402}}
    )]))

    async with sarvam_tts.SarvamTTS(language="te-IN") as tts:
        chunks = await asyncio.wait_for(tts.collect("ఒకటి"), timeout=5)

    assert chunks == []
    assert tripped["reason"] == "Insufficient credits"


async def test_an_unrelated_error_does_not_trip_the_circuit_breaker(
    connected, monkeypatch
):
    async def must_not_be_called(reason):
        raise AssertionError("an unrelated error must not trip the breaker")

    monkeypatch.setattr(sarvam_tts.sarvam_circuit_breaker, "trip", must_not_be_called)
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


# ── the close frame ──────────────────────────────────────────────────────────
#
# The error frames above are only half of how Sarvam reports an outage. On
# 2026-08-13 the account ran out of credits and this leg sent
# {"message": "No credits available.", "code": 402} — and the socket then
# closed, which is all the bridge managed to log:
#
#     [sarvam] lead=30c72ac1 conversation failed: ConnectionClosedOK
#
# A close carrying a reason is a diagnosis; a close that ends up logged like
# that is not.

from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close

_CREDITS_CLOSE = "Credits exhausted. Visit the API Dashboard."


class _ClosingWS(_FakeSarvamWS):
    """Yields *messages*, then closes with *close* instead of going quiet."""

    def __init__(self, messages: list[str], close: Close, ok: bool = False):
        super().__init__(messages)
        self._close = close
        self._ok = ok

    def _closed(self):
        cls = ConnectionClosedOK if self._ok else ConnectionClosedError
        return cls(self._close, None)

    def __aiter__(self):
        async def _gen():
            for m in self._messages:
                yield m
            raise self._closed()
        return _gen()


async def test_a_credits_close_trips_the_circuit_breaker(connected, monkeypatch):
    tripped = {}

    async def fake_trip(reason):
        tripped["reason"] = reason

    monkeypatch.setattr(sarvam_tts.sarvam_circuit_breaker, "trip", fake_trip)
    connected(_ClosingWS([], Close(1003, _CREDITS_CLOSE)))

    with pytest.raises(ConnectionClosed):
        async with sarvam_tts.SarvamTTS(language="te-IN") as tts:
            await asyncio.wait_for(tts.collect("ఒకటి"), timeout=5)

    assert tripped["reason"] == _CREDITS_CLOSE


async def test_a_credits_close_trips_even_when_the_code_is_1000(
    connected, monkeypatch
):
    """The live bridge log said ConnectionClosedOK — a NORMAL closure code.
    'Tidy shutdown' and 'the account is empty' are not mutually exclusive, so
    the reason decides, not the code."""
    tripped = {}

    async def fake_trip(reason):
        tripped["reason"] = reason

    monkeypatch.setattr(sarvam_tts.sarvam_circuit_breaker, "trip", fake_trip)
    connected(_ClosingWS([], Close(1000, _CREDITS_CLOSE), ok=True))

    with pytest.raises(ConnectionClosed):
        async with sarvam_tts.SarvamTTS(language="te-IN") as tts:
            await asyncio.wait_for(tts.collect("ఒకటి"), timeout=5)

    assert tripped["reason"] == _CREDITS_CLOSE


async def test_an_ordinary_close_does_not_trip_the_breaker(connected, monkeypatch):
    """1006 with an empty reason is what a dropped connection looks like —
    and what OUR own close looks like from inside. Tripping on it would ground
    every Sarvam campaign the first time a call ended untidily."""
    async def must_not_be_called(reason):
        raise AssertionError("an ordinary disconnect must not trip the breaker")

    monkeypatch.setattr(sarvam_tts.sarvam_circuit_breaker, "trip", must_not_be_called)
    connected(_ClosingWS([], Close(1006, "")))

    with pytest.raises(ConnectionClosed):
        async with sarvam_tts.SarvamTTS(language="te-IN") as tts:
            await asyncio.wait_for(tts.collect("ఒకటి"), timeout=5)


async def test_a_normal_close_after_final_is_not_an_error(connected, monkeypatch):
    """final ends the stream before the close is ever reached, so a server
    tidying up after a completed utterance must not look like a failure."""
    async def must_not_be_called(reason):
        raise AssertionError("a completed utterance is not a credits failure")

    monkeypatch.setattr(sarvam_tts.sarvam_circuit_breaker, "trip", must_not_be_called)
    connected(_ClosingWS([_audio(), _final()], Close(1000, ""), ok=True))

    async with sarvam_tts.SarvamTTS(language="te-IN") as tts:
        chunks = await asyncio.wait_for(tts.collect("ఒకటి"), timeout=5)

    assert [c[0] for c in chunks] == [_FRAME]


async def test_audio_played_before_a_close_is_kept(connected, monkeypatch):
    """Frames already handed to Plivo were HEARD by the lead. The ledger sizes
    a barge-in from them, so losing them would misreport what was said."""
    async def fake_trip(reason):
        pass

    monkeypatch.setattr(sarvam_tts.sarvam_circuit_breaker, "trip", fake_trip)
    connected(_ClosingWS([_audio()], Close(1003, _CREDITS_CLOSE)))

    heard = []
    with pytest.raises(ConnectionClosed):
        async with sarvam_tts.SarvamTTS(language="te-IN") as tts:
            async for payload, _ms in tts.speak("ఒకటి"):
                heard.append(payload)

    assert heard == [_FRAME]


async def test_the_close_reason_is_logged(connected, monkeypatch, caplog):
    async def fake_trip(reason):
        pass

    monkeypatch.setattr(sarvam_tts.sarvam_circuit_breaker, "trip", fake_trip)
    connected(_ClosingWS([], Close(1003, _CREDITS_CLOSE)))

    with caplog.at_level("ERROR"):
        with pytest.raises(ConnectionClosed):
            async with sarvam_tts.SarvamTTS(language="te-IN") as tts:
                await asyncio.wait_for(tts.collect("ఒకటి"), timeout=5)

    assert "Credits exhausted" in caplog.text
    assert "1003" in caplog.text


async def test_a_closed_socket_is_not_reused_by_the_next_utterance(
    connected, monkeypatch
):
    """A socket the far end closed is dead. Keeping it would make every later
    turn of the call fail on send() rather than on the real cause — the same
    reasoning as _reset_connection() after a barge-in."""
    async def fake_trip(reason):
        pass

    monkeypatch.setattr(sarvam_tts.sarvam_circuit_breaker, "trip", fake_trip)
    connected(_ClosingWS([], Close(1006, "")))

    async with sarvam_tts.SarvamTTS(language="te-IN") as tts:
        with pytest.raises(ConnectionClosed):
            await asyncio.wait_for(tts.collect("ఒకటి"), timeout=5)
        assert tts._ws is None, "a closed socket must not be kept for the next turn"
