"""telephony/plivo_stream.py — the Plivo side of one answered call.

Everything here is about the CALL, not about whichever model is speaking on
it. Plivo Audio Streams carry 8 kHz G.711 mu-law both ways; every voice
backend we bridge to is configured for ulaw_8000 in both directions, so audio
is a direct base64 passthrough with no resampling or transcoding. That is what
CLAUDE.md's "keep TTS output format ulaw_8000 for the phone path" rule is for.

    Plivo call  --(media: base64 mu-law)-->  read_events(on_media=...)
    Plivo call  <--(playAudio: base64)-----  PlivoCall.play(...)

This module was split out of bridge.py so a second voice backend (OpenAI
Realtime, for Telugu — a language ElevenLabs cannot speak at all) can reuse
the call plumbing without copying it. The piece that must never exist in two
hand-maintained copies is `PlivoCall.oneway_watchdog`: read its docstring.

Observed Plivo client messages on the stream socket:

    start   {streamId, start: {streamId, ...}}
    media   {media: {payload: <base64 mu-law>}}
    stop    {}

Server messages we send back: playAudio and clearAudio.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from urllib.parse import quote
from xml.sax.saxutils import escape

from fastapi import WebSocket, WebSocketDisconnect

from app.config import (
    CALL_MAX_DURATION_S,
    CALL_WEBHOOK_SECRET,
    ONEWAY_MAX_SILENT_S,
    ONEWAY_SILENCE_TAIL_S,
    PLIVO_GREETING_GRACE_MS,
    PUBLIC_BASE_URL,
)

logger = logging.getLogger(__name__)

# mu-law at 8 kHz is 8000 bytes/sec, and base64 inflates by 4/3 — so decoded
# bytes = len(b64) * 3/4, and seconds = that / 8000. Used to estimate when
# audio already handed to Plivo will finish playing.
_ULAW_BYTES_PER_S = 8000.0

# How often the one-way watchdog re-checks whether the message has finished.
# Fine-grained enough that the poll adds no audible tail of its own.
_ONEWAY_POLL_S = 0.25

# Exit reasons that mean the call ran its course rather than fell over. Kept
# here because they are all decided on the Plivo side, and every backend has
# to grade its calls the same way. oneway_complete belongs in this set: it IS
# the natural end of a one-way call, and without it every successful one-way
# call would be stored 'failed', mark its lead 'failed', and burn a retry.
# elevenlabs_closed:1000 is the same story for a two-way call the AGENT ends
# (its End Call tool, e.g. after a caller says goodbye): code 1000 is a normal
# WebSocket closure, not a failure, and without it that lead would be marked
# 'failed' and re-dialled after a conversation that completed cleanly. Other
# close codes stay out of this set on purpose — 1002 is the billing-stop path
# bridge.py's docstring calls out, and must still grade as failed.
# openai_end_call is the OpenAI-backend equivalent: that platform has no
# built-in "End Call" tool, so openai_bridge.py gives the model its own
# end_call function and grades the resulting exit the same way — a
# conversation the model chose to end cleanly, not a dropped call.
_CLEAN_EXITS = ("plivo_stop", "plivo_disconnect", "max_duration", "oneway_complete",
                "elevenlabs_closed:1000", "openai_end_call")


def answer_xml(lead_id: str) -> str:
    """The XML Plivo fetches on answer: open a bidirectional mu-law stream back
    to this server.

    keepCallAlive keeps the call up while the stream runs; without it Plivo
    treats the (empty) rest of the XML document as the whole call and hangs up
    the moment the stream is established.
    """
    base = (PUBLIC_BASE_URL or "").rstrip("/")
    wss = base.replace("https://", "wss://").replace("http://", "ws://")
    url = (f"{wss}/calls/stream?token={quote(CALL_WEBHOOK_SECRET or '')}"
           f"&lead={quote(lead_id)}")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        '<Stream bidirectional="true" keepCallAlive="true" '
        f'contentType="audio/x-mulaw;rate=8000">{escape(url)}</Stream>'
        "</Response>"
    )


class PlivoCall:
    """One answered Plivo call's audio channel and lifecycle.

    Owns the per-call mutable state that the audio path and the lifecycle
    share, and the `stop` event that every task on the call watches. A voice
    backend drives it: `read_events` for the inbound leg, `play` / `interrupt`
    for the outbound one, and `run` to supervise the whole thing.
    """

    def __init__(self, ws: WebSocket, *, lead_id: str, one_way: bool = False) -> None:
        self.ws = ws
        self.lead_id = lead_id
        self.one_way = one_way
        self._loop = asyncio.get_running_loop()

        # Set by whichever task ends the call; every other task watches it.
        self.stop: asyncio.Event = asyncio.Event()

        self.stream_id: str | None = None
        # Loop-time at which audio already sent to Plivo finishes playing. The
        # agent can emit audio faster than real time, so a graceful hangup must
        # wait on this rather than a fixed sleep, or the last words are cut off.
        # 0.0 is the sentinel for "nothing queued" — set at init and reset by
        # clear(); a call that has played audio always holds a positive
        # loop-time here.
        self.play_end: float = 0.0
        # Loop-time at which the CURRENT contiguous stretch of playback began.
        # The agent streams a whole response's audio back-to-back faster than
        # real time, so those chunks share one segment; a gap (the lead's turn)
        # starts a new one. played_ms() measures from here to answer "how much
        # of this response has the lead actually heard" for barge-in truncation.
        self.play_start: float = 0.0
        # How many ms of the current response had reached the lead at the last
        # barge-in. clear() snapshots it BEFORE wiping play_end, because the
        # truncate that follows needs the amount heard, and play_end is gone by
        # then. See played_ms().
        self._interrupted_ms: int = 0
        # Set when the next audio chunk begins a NEW response item. The tool-call
        # flow queues a second response's audio back-to-back with the first, with
        # no lead turn (and so no play_end gap) between them: without this flag
        # play() would treat the two as one contiguous segment, and played_ms()
        # would measure across BOTH. A barge-in into the second response would
        # then report more audio than that item contains, so the truncate
        # overshoots the item it names and the API rejects it. See
        # mark_response_boundary() and played_ms().
        self._segment_boundary_pending: bool = False
        # The lead's first sound on an outbound call is almost always "Hello?" —
        # an acknowledgement, not an interruption. Cancelling the greeting on it
        # made the agent restart from the top. Barge-in is live after this.
        self.grace_until: float = 0.0
        # Whether ANY agent audio has arrived yet. Tracked separately from
        # play_end because that starts at 0.0, which the one-way watchdog
        # cannot tell apart from "the message already finished playing" — it
        # would hang up before the agent had said a word.
        self.audio_seen: bool = False
        self.exit_reason: str = "unknown"

    # ── lifecycle bookkeeping ────────────────────────────────────────────────

    def note_exit(self, reason: str) -> None:
        """Record why the call ended. First reason wins — it is the cause;
        anything after it is a consequence of the socket being torn down."""
        if self.exit_reason == "unknown":
            self.exit_reason = reason

    @property
    def ran_its_course(self) -> bool:
        """Whether this call ended normally, as opposed to falling over.

        A bridge that failed part-way must not look like a completed
        conversation in the calls table — see _CLEAN_EXITS.
        """
        return self.exit_reason in _CLEAN_EXITS

    # ── outbound audio ───────────────────────────────────────────────────────

    async def play(self, payload_b64: str) -> None:
        """Send agent audio into the call, and account for how long it will
        take to play out."""
        await self.ws.send_text(json.dumps({
            "event": "playAudio",
            "media": {"contentType": "audio/x-mulaw", "sampleRate": 8000,
                      "payload": payload_b64},
        }))
        now = self._loop.time()
        if self._segment_boundary_pending:
            # A new response item begins: its playback clock starts where the
            # previous item's queued audio ends (max(now, play_end)) — not
            # carried over from the previous item, and not "now" if that item's
            # audio is still draining ahead of it.
            self.play_start = max(now, self.play_end)
            self._segment_boundary_pending = False
        elif self.play_end <= now:
            # The queue had drained (or was cleared): this chunk begins a fresh
            # contiguous segment, so playback of it starts now, not after some
            # already-finished earlier audio.
            self.play_start = now
        dur = (len(payload_b64) * 3 / 4) / _ULAW_BYTES_PER_S
        self.play_end = max(self.play_end, now) + dur
        self.audio_seen = True

    def mark_response_boundary(self) -> None:
        """Tell the call that the next audio chunk starts a NEW response item.

        The tool-call flow queues a second response's audio back-to-back with
        the first; the voice backend calls this between them so the playback
        clock (play_start) restarts at the new item, and played_ms() measures
        only that item rather than spanning both. See the field's comment and
        played_ms()."""
        self._segment_boundary_pending = True

    def played_ms(self) -> int:
        """How many milliseconds of the CURRENT response actually reached the
        lead — the value a barge-in truncate needs so the model's belief about
        what it said matches what the lead heard.

        While audio is queued (`play_end` still a live loop-time), this is the
        real elapsed playback of the current segment, capped at the audio
        actually queued. After clear() has wiped `play_end` to 0.0 — which is
        exactly when the barge-in handler asks — the live figure is gone, so we
        return the value clear() snapshotted at the instant of the interruption.
        """
        if self.play_end > 0.0:
            audible_until = min(self._loop.time(), self.play_end)
            played = audible_until - self.play_start
            return int(played * 1000) if played > 0 else 0
        return self._interrupted_ms

    async def clear(self) -> None:
        """Drop audio already buffered on the call, so barge-in is immediate
        rather than waiting for the agent's queued speech to drain.

        Nothing queued is left to play, so play_end resets with it."""
        # Snapshot how much of the current response actually reached the lead
        # BEFORE dropping the queue: the barge-in path calls played_ms() right
        # after this, and resetting play_end below would otherwise make it read
        # 0 — telling the model the lead heard none of its reply.
        self._interrupted_ms = self.played_ms()
        msg: dict = {"event": "clearAudio"}
        if self.stream_id:
            msg["streamId"] = self.stream_id
        await self.ws.send_text(json.dumps(msg))
        self.play_end = 0.0

    async def interrupt(self) -> bool:
        """Honour barge-in, except during the greeting window. Returns whether
        the buffered audio was actually dropped."""
        if self._loop.time() < self.grace_until:
            return False
        await self.clear()
        return True

    # ── inbound audio ────────────────────────────────────────────────────────

    async def read_events(self, on_media: Callable[[str], Awaitable[None]]) -> None:
        """Read the Plivo stream socket until the call ends.

        *on_media* receives each base64 mu-law payload from the lead, and is
        the one backend-specific thing here. It is never called on a one-way
        call: those deliver a message and do not listen.
        """
        try:
            while not self.stop.is_set():
                raw = await self.ws.receive_text()
                try:
                    msg = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                ev = msg.get("event")
                if ev == "start":
                    st = msg.get("start") or {}
                    self.stream_id = msg.get("streamId") or st.get("streamId")
                    if not self.one_way and PLIVO_GREETING_GRACE_MS > 0:
                        self.grace_until = (
                            self._loop.time() + PLIVO_GREETING_GRACE_MS / 1000.0
                        )
                elif ev == "media":
                    if self.one_way:
                        continue  # never listen on a one-way call
                    payload = (msg.get("media") or {}).get("payload")
                    if payload:
                        await on_media(payload)
                elif ev == "stop":
                    self.note_exit("plivo_stop")
                    break
        except WebSocketDisconnect:
            self.note_exit("plivo_disconnect")
        except Exception as exc:
            self.note_exit(f"error:{type(exc).__name__}")
            logger.info(f"[plivo] Plivo reader stopped: {type(exc).__name__}: {exc}")
        finally:
            self.stop.set()

    # ── one-way: end the call when the message has been delivered ────────────

    async def oneway_watchdog(self) -> None:
        """Nothing else can end a one-way call.

        A one-way bridge never forwards the lead's audio (see read_events),
        so the voice backend receives no turn-end signal and never closes its
        socket; read_events only returns if the lead hangs up first. That
        left `stop` unset and the bridge parked on CALL_MAX_DURATION_S —
        300 seconds of billed Plivo airtime and a 300-second ElevenLabs
        conversation for a 20-second message, with the lead listening to
        silence for the remainder.

        So: wait for the agent's audio to drain, allow a short tail, then
        end the call ourselves. Bounded at both ends — an agent that never
        speaks at all costs ONEWAY_MAX_SILENT_S, not the full duration.
        """
        started = self._loop.time()
        while not self.stop.is_set():
            await asyncio.sleep(_ONEWAY_POLL_S)
            now = self._loop.time()
            if not self.audio_seen:
                if now - started >= ONEWAY_MAX_SILENT_S:
                    logger.warning(
                        f"[plivo] lead={self.lead_id} one-way agent produced no audio "
                        f"in {ONEWAY_MAX_SILENT_S}s — ending the call. Check the "
                        "agent's prompt and its dynamic variables."
                    )
                    self.note_exit("oneway_no_audio")
                    break
                continue
            if now >= self.play_end + ONEWAY_SILENCE_TAIL_S:
                self.note_exit("oneway_complete")
                break
        self.stop.set()

    # ── supervision ──────────────────────────────────────────────────────────

    async def run(self, *tasks: Awaitable[None]) -> None:
        """Run *tasks* until the call ends, then drain and cancel them.

        On a one-way call the watchdog is added here, so no backend can forget
        it. Returns once the line is safe to drop; the caller is responsible
        for whatever it collected along the way.
        """
        coros: list[Awaitable[None]] = list(tasks)
        if self.one_way:
            coros.append(self.oneway_watchdog())
        reader = asyncio.gather(*coros, return_exceptions=True)
        try:
            # A call that never ends would hold a concurrency slot forever, so
            # the whole bridge is bounded rather than trusting either side.
            await asyncio.wait_for(self.stop.wait(), timeout=CALL_MAX_DURATION_S)
        except asyncio.TimeoutError:
            self.note_exit("max_duration")
            self.stop.set()
        finally:
            # Let the agent's final words play out before the line drops.
            remaining = self.play_end - self._loop.time()
            if remaining > 0:
                await asyncio.sleep(min(remaining + 0.3, 15))
            reader.cancel()
            try:
                await reader
            except (asyncio.CancelledError, Exception):
                pass
