"""Unit tests for telephony/plivo_stream.py — the Plivo side of a call.

Two groups. The answer XML is pure and easy to get subtly wrong, and fails in
a way that is very hard to diagnose from the outside (the call rings,
connects, and is silent).

Then PlivoCall itself. Most of this used to be closures inside bridge(),
reachable only by driving a whole ElevenLabs conversation past them; now that
it has its own home the one-way watchdog can be tested directly, which is the
point of the extraction. Its timing is the most expensive thing in this
codebase to get wrong: before it existed every one-way call sat on
CALL_MAX_DURATION_S — 300 seconds of billed Plivo airtime and a 300-second
model conversation for a 20-second message, with the lead hearing silence for
the remainder.
"""

import asyncio
import base64
import json

import pytest
from fastapi import WebSocketDisconnect

from app.telephony import plivo_stream
from app.telephony.plivo_stream import PlivoCall


@pytest.fixture(autouse=True)
def _public_url(monkeypatch):
    monkeypatch.setattr(plivo_stream, "PUBLIC_BASE_URL",
                        "https://example.trycloudflare.com")
    monkeypatch.setattr(plivo_stream, "CALL_WEBHOOK_SECRET", "s3cret")


# ── the answer XML ───────────────────────────────────────────────────────────

def test_answer_xml_points_at_the_websocket_over_wss():
    """https must become wss, not stay https — Plivo silently fails to open
    the stream otherwise, and the call connects to silence."""
    xml = plivo_stream.answer_xml("lead-123")
    assert "wss://example.trycloudflare.com/calls/stream" in xml
    assert "https://" not in xml


def test_answer_xml_declares_mulaw_8000():
    """Plivo must be told the stream is mu-law 8 kHz. Any other framing and
    the caller hears noise, because the agent's audio really is mu-law."""
    xml = plivo_stream.answer_xml("lead-123")
    assert 'contentType="audio/x-mulaw;rate=8000"' in xml


def test_answer_xml_is_bidirectional_and_keeps_the_call_alive():
    """Without keepCallAlive Plivo treats the rest of the (empty) document as
    the whole call and hangs up the moment the stream is established."""
    xml = plivo_stream.answer_xml("lead-123")
    assert 'bidirectional="true"' in xml
    assert 'keepCallAlive="true"' in xml


def test_answer_xml_carries_the_lead_and_token():
    xml = plivo_stream.answer_xml("lead-123")
    assert "lead=lead-123" in xml
    assert "token=s3cret" in xml


def test_answer_xml_escapes_the_url():
    """The URL goes inside an XML element, so & between query params must be
    escaped or Plivo's parser rejects the document."""
    xml = plivo_stream.answer_xml("lead-123")
    assert "&amp;" in xml
    # A bare & would mean the ampersand was left unescaped.
    assert "&lead=" not in xml.replace("&amp;lead=", "")


def test_answer_xml_handles_a_http_base_url(monkeypatch):
    monkeypatch.setattr(plivo_stream, "PUBLIC_BASE_URL", "http://localhost:8091")
    xml = plivo_stream.answer_xml("lead-1")
    assert "ws://localhost:8091/calls/stream" in xml


# ── fakes ────────────────────────────────────────────────────────────────────

def _mulaw(seconds: float) -> str:
    """*seconds* of mu-law at 8 kHz, base64-encoded — what a backend hands to
    PlivoCall.play."""
    return base64.b64encode(b"\xff" * int(8000 * seconds)).decode()


_SHORT = _mulaw(0.08)   # a chunk that drains within one watchdog poll
_LONG = _mulaw(0.6)     # a chunk that takes several polls to drain


class _FakePlivoWS:
    """Records what was sent into the call, and never says anything back
    unless scripted to."""

    def __init__(self, frames: list[str] | None = None, *,
                 then: BaseException | None = None):
        self.sent: list[dict] = []
        self._frames = list(frames or [])
        self._then = then

    async def receive_text(self) -> str:
        if self._frames:
            return self._frames.pop(0)
        if self._then is not None:
            raise self._then
        await asyncio.Event().wait()  # a lead who has answered and is silent
        raise AssertionError("unreachable")

    async def send_text(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


def _frame(event: str, **rest) -> str:
    return json.dumps({"event": event, **rest})


async def _never_called(payload: str) -> None:
    raise AssertionError(f"the lead's audio was forwarded: {payload!r}")


# ── outbound audio ───────────────────────────────────────────────────────────

async def test_play_sends_mulaw_8000_and_notes_that_audio_was_seen():
    """The payload is passed through untouched: both legs are mu-law 8 kHz, so
    any transcoding here would be a bug, not a feature."""
    ws = _FakePlivoWS()
    call = PlivoCall(ws, lead_id="lead-1", one_way=True)

    await call.play(_SHORT)

    assert ws.sent == [{
        "event": "playAudio",
        "media": {"contentType": "audio/x-mulaw", "sampleRate": 8000,
                  "payload": _SHORT},
    }]
    assert call.audio_seen is True


async def test_play_end_accumulates_as_audio_is_queued_not_as_it_plays():
    """The agent emits audio far faster than real time, so three chunks handed
    over back to back must push play_end out by the sum of their durations. If
    play_end tracked wall-clock instead, the watchdog would hang up on the
    first chunk's worth of speech and cut the message off mid-sentence."""
    call = PlivoCall(_FakePlivoWS(), lead_id="lead-1", one_way=True)
    loop = asyncio.get_running_loop()

    # Measured from BEFORE the first chunk, not from "now" after the last one.
    # Against `loop.time()` this asserts that under 100ms elapsed while queueing,
    # which a stalled test runner can violate on a function whose whole point is
    # that it does not depend on elapsed time. From t0 the expected value is
    # exact at any machine speed, and still collapses to ~0.6 if play_end ever
    # started tracking wall-clock instead of accumulating.
    t0 = loop.time()
    for _ in range(3):
        await call.play(_LONG)

    assert call.play_end - t0 >= 3 * 0.6 - 0.01


async def test_clear_drops_buffered_audio_and_resets_play_end():
    ws = _FakePlivoWS()
    call = PlivoCall(ws, lead_id="lead-1", one_way=False)
    call.stream_id = "stream-9"
    await call.play(_LONG)

    await call.clear()

    assert ws.sent[-1] == {"event": "clearAudio", "streamId": "stream-9"}
    assert call.play_end == 0.0


async def test_clear_omits_the_stream_id_before_the_start_frame():
    """clearAudio can arrive before Plivo has told us the streamId; sending
    streamId=None would be a malformed frame."""
    ws = _FakePlivoWS()
    call = PlivoCall(ws, lead_id="lead-1", one_way=False)

    await call.clear()

    assert ws.sent[-1] == {"event": "clearAudio"}


async def test_interrupt_is_ignored_during_the_greeting_grace_window():
    """REGRESSION-adjacent. The lead's first sound on an outbound call is
    almost always "Hello?" — an acknowledgement. Clearing on it made the agent
    restart its greeting from the top."""
    ws = _FakePlivoWS()
    call = PlivoCall(ws, lead_id="lead-1", one_way=False)
    await call.play(_LONG)
    call.grace_until = asyncio.get_running_loop().time() + 30

    cleared = await call.interrupt()

    assert cleared is False
    assert not any(m["event"] == "clearAudio" for m in ws.sent)
    assert call.play_end > 0.0, "queued audio must keep playing through the grace window"


async def test_interrupt_clears_once_the_grace_window_has_passed():
    ws = _FakePlivoWS()
    call = PlivoCall(ws, lead_id="lead-1", one_way=False)
    await call.play(_LONG)
    call.grace_until = asyncio.get_running_loop().time() - 1

    cleared = await call.interrupt()

    assert cleared is True
    assert ws.sent[-1]["event"] == "clearAudio"
    assert call.play_end == 0.0


# ── played_ms: how much of the current response the lead actually heard ───────
# Feeds the barge-in truncate. Getting it wrong misinforms the model about what
# the lead heard, which is the whole bug this path exists to avoid.


async def test_played_ms_is_zero_before_anything_has_played():
    call = PlivoCall(_FakePlivoWS(), lead_id="lead-1", one_way=False)
    assert call.played_ms() == 0


async def test_played_ms_counts_only_what_has_actually_played_not_what_is_queued():
    """The agent queues a whole response in an instant, but the lead has only
    heard the part that has played out in real time. Truncating at the queued
    length would tell the model the lead heard words still sitting in the
    buffer."""
    call = PlivoCall(_FakePlivoWS(), lead_id="lead-1", one_way=False)
    await call.play(_LONG)   # 0.6s queued the moment it is handed over

    # Almost no wall-clock time has passed, so almost none of it has played.
    assert call.played_ms() < 100


async def test_played_ms_reports_audio_heard_before_a_barge_in_and_survives_clear():
    """The value the truncate needs, and the case the design turns on: it is
    read AFTER clear() has wiped play_end to 0.0. Reading play_end then would
    give 0 and tell the model the lead heard none of its reply; the snapshot
    taken at the interruption is what keeps it honest."""
    call = PlivoCall(_FakePlivoWS(), lead_id="lead-1", one_way=False)
    await call.play(_LONG)   # 0.6s queued, segment starts "now"
    # Pretend 0.3s of it has already played by backdating the segment start.
    call.play_start -= 0.3

    heard_live = call.played_ms()      # live, before the barge-in
    await call.clear()                 # barge-in drops the queue, play_end -> 0
    heard_after = call.played_ms()     # must still report what was heard

    assert 250 <= heard_live <= 400
    assert 250 <= heard_after <= 400
    assert call.play_end == 0.0


async def test_played_ms_is_zero_right_after_a_clear_with_nothing_played():
    """A barge-in during the very first instant of a response: essentially
    nothing reached the lead, so the truncate must say ~0 — not the queued
    length, and not stale state from an earlier turn."""
    call = PlivoCall(_FakePlivoWS(), lead_id="lead-1", one_way=False)
    await call.play(_LONG)   # queued, but interrupted immediately

    await call.clear()

    assert call.played_ms() < 100


async def test_played_ms_resets_for_a_fresh_response_after_a_clear():
    """After a barge-in the next response is a NEW segment: its played time is
    measured from its own start, not carried over from the interrupted one."""
    call = PlivoCall(_FakePlivoWS(), lead_id="lead-1", one_way=False)
    await call.play(_LONG)
    call.play_start -= 0.3
    await call.clear()       # snapshot ~300ms for the interrupted turn

    await call.play(_LONG)   # a brand-new response begins playing now
    assert call.played_ms() < 100, "the new response's clock must start at zero"


# ── exit bookkeeping ─────────────────────────────────────────────────────────

async def test_the_first_exit_reason_wins():
    """The first reason is the cause; everything after it is a consequence of
    the socket being torn down."""
    call = PlivoCall(_FakePlivoWS(), lead_id="lead-1")

    call.note_exit("oneway_complete")
    call.note_exit("error:ConnectionClosed")

    assert call.exit_reason == "oneway_complete"


@pytest.mark.parametrize("reason", ["plivo_stop", "plivo_disconnect",
                                    "max_duration", "oneway_complete"])
async def test_a_clean_exit_reason_means_the_call_ran_its_course(reason):
    """oneway_complete belongs in this set: without it every successful
    one-way call would be stored 'failed', mark its lead 'failed', and burn a
    retry attempt."""
    call = PlivoCall(_FakePlivoWS(), lead_id="lead-1")
    call.note_exit(reason)
    assert call.ran_its_course is True


@pytest.mark.parametrize("reason", ["unknown", "oneway_no_audio",
                                    "error:RuntimeError", "elevenlabs_closed:1002"])
async def test_a_failure_reason_does_not_count_as_a_completed_call(reason):
    call = PlivoCall(_FakePlivoWS(), lead_id="lead-1")
    call.note_exit(reason)
    assert call.ran_its_course is False


# ── the inbound reader ───────────────────────────────────────────────────────

async def test_read_events_captures_the_stream_id_from_the_start_frame():
    ws = _FakePlivoWS([_frame("start", streamId="stream-1"), _frame("stop")])
    call = PlivoCall(ws, lead_id="lead-1", one_way=True)

    await call.read_events(_never_called)

    assert call.stream_id == "stream-1"
    assert call.exit_reason == "plivo_stop"
    assert call.stop.is_set()


async def test_read_events_falls_back_to_the_nested_stream_id():
    """Plivo has been observed to put streamId inside the start object rather
    than at the top level; without the streamId, clearAudio is unaddressed."""
    ws = _FakePlivoWS([_frame("start", start={"streamId": "stream-2"}), _frame("stop")])
    call = PlivoCall(ws, lead_id="lead-1", one_way=True)

    await call.read_events(_never_called)

    assert call.stream_id == "stream-2"


async def test_the_greeting_grace_window_opens_on_a_two_way_call(monkeypatch):
    monkeypatch.setattr(plivo_stream, "PLIVO_GREETING_GRACE_MS", 2500)
    ws = _FakePlivoWS([_frame("start", streamId="s"), _frame("stop")])
    call = PlivoCall(ws, lead_id="lead-1", one_way=False)
    loop = asyncio.get_running_loop()

    await call.read_events(_never_called)

    assert call.grace_until - loop.time() > 2.0


async def test_a_one_way_call_opens_no_grace_window(monkeypatch):
    """Nothing can barge in on a call whose audio is never forwarded, so there
    is nothing to protect from it."""
    monkeypatch.setattr(plivo_stream, "PLIVO_GREETING_GRACE_MS", 2500)
    ws = _FakePlivoWS([_frame("start", streamId="s"), _frame("stop")])
    call = PlivoCall(ws, lead_id="lead-1", one_way=True)

    await call.read_events(_never_called)

    assert call.grace_until == 0.0


async def test_read_events_forwards_the_leads_audio_on_a_two_way_call():
    ws = _FakePlivoWS([
        _frame("media", media={"payload": "aaa"}),
        _frame("media", media={"payload": "bbb"}),
        _frame("stop"),
    ])
    call = PlivoCall(ws, lead_id="lead-1", one_way=False)
    heard: list[str] = []

    async def on_media(payload: str) -> None:
        heard.append(payload)

    await call.read_events(on_media)

    assert heard == ["aaa", "bbb"]


async def test_a_one_way_call_never_forwards_the_leads_audio():
    """The whole definition of one-way. Forwarding would give the backend a
    turn to answer, and the lead would get a conversation they were never
    told they could have."""
    ws = _FakePlivoWS([_frame("media", media={"payload": "aaa"}), _frame("stop")])
    call = PlivoCall(ws, lead_id="lead-1", one_way=True)

    await call.read_events(_never_called)

    assert call.exit_reason == "plivo_stop"


async def test_read_events_skips_frames_it_cannot_parse():
    """A malformed frame must not end a live call."""
    ws = _FakePlivoWS(["not json at all", _frame("media", media={"payload": "aaa"}),
                       _frame("stop")])
    call = PlivoCall(ws, lead_id="lead-1", one_way=False)
    heard: list[str] = []

    async def on_media(payload: str) -> None:
        heard.append(payload)

    await call.read_events(on_media)

    assert heard == ["aaa"]
    assert call.exit_reason == "plivo_stop"


async def test_a_lead_hanging_up_is_a_clean_exit():
    ws = _FakePlivoWS(then=WebSocketDisconnect(code=1000))
    call = PlivoCall(ws, lead_id="lead-1", one_way=False)

    await call.read_events(_never_called)

    assert call.exit_reason == "plivo_disconnect"
    assert call.ran_its_course is True


async def test_an_unexpected_reader_failure_is_recorded_and_still_stops_the_call():
    ws = _FakePlivoWS(then=RuntimeError("socket exploded"))
    call = PlivoCall(ws, lead_id="lead-1", one_way=False)

    await call.read_events(_never_called)

    assert call.exit_reason == "error:RuntimeError"
    assert call.ran_its_course is False
    assert call.stop.is_set(), "a dead reader must not leave the call running"


# ── the one-way watchdog ─────────────────────────────────────────────────────

@pytest.fixture
def fast_oneway(monkeypatch):
    """Real timings, scaled down. The watchdog polls every 250 ms, so these
    are still several polls apart."""
    monkeypatch.setattr(plivo_stream, "ONEWAY_SILENCE_TAIL_S", 0)
    monkeypatch.setattr(plivo_stream, "ONEWAY_MAX_SILENT_S", 5)
    return monkeypatch


async def test_the_watchdog_ends_the_call_once_the_message_has_drained(fast_oneway):
    """REGRESSION, and the reason this class exists. Nothing else ends a
    one-way call: the reader never forwards the lead's audio, so the backend
    gets no turn-end signal and never closes its socket, and read_events only
    returns if the lead hangs up first."""
    call = PlivoCall(_FakePlivoWS(), lead_id="lead-1", one_way=True)
    await call.play(_SHORT)

    await asyncio.wait_for(call.oneway_watchdog(), timeout=5)

    assert call.exit_reason == "oneway_complete"
    assert call.stop.is_set()
    assert call.ran_its_course is True


async def test_the_watchdog_waits_for_queued_audio_to_finish_playing(fast_oneway):
    """The give-up condition is play_end, not wall-clock: hanging up while
    audio is still queued cuts the message off mid-sentence, which is the
    failure the lead actually hears."""
    call = PlivoCall(_FakePlivoWS(), lead_id="lead-1", one_way=True)
    loop = asyncio.get_running_loop()
    await call.play(_LONG)
    await call.play(_LONG)  # 1.2s of speech queued in an instant

    started = loop.time()
    await asyncio.wait_for(call.oneway_watchdog(), timeout=5)
    elapsed = loop.time() - started

    assert call.exit_reason == "oneway_complete"
    assert elapsed >= 1.1, f"the message was cut off after {elapsed:.2f}s of 1.20s"


async def test_the_watchdog_honours_the_silence_tail(monkeypatch):
    """play_end is estimated from the base64 byte count, not reported by
    Plivo, so the tail is what stops the last syllable being clipped."""
    monkeypatch.setattr(plivo_stream, "ONEWAY_SILENCE_TAIL_S", 1)
    monkeypatch.setattr(plivo_stream, "ONEWAY_MAX_SILENT_S", 5)
    call = PlivoCall(_FakePlivoWS(), lead_id="lead-1", one_way=True)
    loop = asyncio.get_running_loop()
    await call.play(_SHORT)

    started = loop.time()
    await asyncio.wait_for(call.oneway_watchdog(), timeout=5)
    elapsed = loop.time() - started

    assert call.exit_reason == "oneway_complete"
    assert elapsed >= 1.0, f"the tail was skipped: ended after {elapsed:.2f}s"


async def test_a_silent_agent_gives_up_after_the_backstop(monkeypatch):
    """An agent that never emits any audio (a broken prompt, a dynamic
    variable it is waiting on) must cost ONEWAY_MAX_SILENT_S, not
    CALL_MAX_DURATION_S. audio_seen is tracked separately from play_end for
    exactly this: play_end starts at 0.0, which is indistinguishable from
    'the message already finished'."""
    monkeypatch.setattr(plivo_stream, "ONEWAY_SILENCE_TAIL_S", 0)
    monkeypatch.setattr(plivo_stream, "ONEWAY_MAX_SILENT_S", 0.4)
    call = PlivoCall(_FakePlivoWS(), lead_id="lead-1", one_way=True)
    loop = asyncio.get_running_loop()

    started = loop.time()
    await asyncio.wait_for(call.oneway_watchdog(), timeout=5)
    elapsed = loop.time() - started

    assert call.exit_reason == "oneway_no_audio"
    assert call.ran_its_course is False, "nothing was said, so nothing was delivered"
    assert elapsed < 2.0, f"gave up after {elapsed:.2f}s, not ~0.4s"


async def test_the_watchdog_returns_immediately_when_the_call_already_ended(fast_oneway):
    """Whoever ended the call first owns the reason; the watchdog must not
    overwrite a lead's hangup with 'oneway_complete'."""
    call = PlivoCall(_FakePlivoWS(), lead_id="lead-1", one_way=True)
    call.note_exit("plivo_stop")
    call.stop.set()

    await asyncio.wait_for(call.oneway_watchdog(), timeout=5)

    assert call.exit_reason == "plivo_stop"


# ── supervision ──────────────────────────────────────────────────────────────

async def test_run_adds_the_watchdog_on_a_one_way_call(fast_oneway):
    """No backend can forget it, because no backend passes it in."""
    call = PlivoCall(_FakePlivoWS(), lead_id="lead-1", one_way=True)
    await call.play(_SHORT)

    async def silent_reader() -> None:
        await asyncio.Event().wait()

    await asyncio.wait_for(call.run(silent_reader()), timeout=5)

    assert call.exit_reason == "oneway_complete"


async def test_run_does_not_add_the_watchdog_on_a_two_way_call(monkeypatch):
    """On a two-way call the silence after the agent's greeting is the lead
    thinking about their reply; hanging up on it would cut off every
    conversation at hello. Asserted on elapsed time as well as the reason:
    a wrongly-running watchdog would exit 'oneway_complete', which also
    counts as a completed call."""
    monkeypatch.setattr(plivo_stream, "ONEWAY_SILENCE_TAIL_S", 0)
    monkeypatch.setattr(plivo_stream, "ONEWAY_MAX_SILENT_S", 0)
    monkeypatch.setattr(plivo_stream, "CALL_MAX_DURATION_S", 1)
    call = PlivoCall(_FakePlivoWS(), lead_id="lead-1", one_way=False)
    loop = asyncio.get_running_loop()
    await call.play(_SHORT)

    started = loop.time()

    async def silent_reader() -> None:
        await asyncio.Event().wait()

    await asyncio.wait_for(call.run(silent_reader()), timeout=5)
    elapsed = loop.time() - started

    assert call.exit_reason == "max_duration"
    assert elapsed >= 0.9, f"the call was ended early, after {elapsed:.2f}s"


async def test_run_bounds_the_whole_call(monkeypatch):
    """A call that never ends would hold a concurrency slot forever."""
    monkeypatch.setattr(plivo_stream, "CALL_MAX_DURATION_S", 0.3)
    call = PlivoCall(_FakePlivoWS(), lead_id="lead-1", one_way=False)

    async def never_ends() -> None:
        await asyncio.Event().wait()

    await asyncio.wait_for(call.run(never_ends()), timeout=5)

    assert call.exit_reason == "max_duration"
    assert call.stop.is_set()


async def test_run_lets_queued_audio_play_out_before_the_line_drops(monkeypatch):
    """The lead should hear the agent's last words, not a click."""
    monkeypatch.setattr(plivo_stream, "CALL_MAX_DURATION_S", 30)
    call = PlivoCall(_FakePlivoWS(), lead_id="lead-1", one_way=False)
    loop = asyncio.get_running_loop()
    await call.play(_LONG)

    async def hang_up_now() -> None:
        call.note_exit("plivo_stop")
        call.stop.set()

    started = loop.time()
    await asyncio.wait_for(call.run(hang_up_now()), timeout=5)
    elapsed = loop.time() - started

    assert call.exit_reason == "plivo_stop"
    assert elapsed >= 0.6, f"cut the last 0.60s of speech off after {elapsed:.2f}s"


async def test_run_cancels_its_tasks_when_the_call_ends(monkeypatch):
    """A leaked reader would hold the Plivo socket open past the call."""
    monkeypatch.setattr(plivo_stream, "CALL_MAX_DURATION_S", 0.3)
    call = PlivoCall(_FakePlivoWS(), lead_id="lead-1", one_way=False)
    cancelled = asyncio.Event()

    async def never_ends() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    await asyncio.wait_for(call.run(never_ends()), timeout=5)

    assert cancelled.is_set()


async def test_run_survives_a_task_that_raises(monkeypatch):
    """gather(return_exceptions=True): one reader falling over must not take
    the outcome-collecting `finally` in bridge() down with it."""
    monkeypatch.setattr(plivo_stream, "CALL_MAX_DURATION_S", 5)
    call = PlivoCall(_FakePlivoWS(), lead_id="lead-1", one_way=False)

    async def explodes() -> None:
        call.note_exit("error:RuntimeError")
        call.stop.set()
        raise RuntimeError("boom")

    await asyncio.wait_for(call.run(explodes()), timeout=5)

    assert call.exit_reason == "error:RuntimeError"
