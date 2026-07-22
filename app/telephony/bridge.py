"""telephony/bridge.py — Bridge a live Plivo call to an ElevenLabs agent.

Plivo Audio Streams carry 8 kHz G.711 mu-law both ways, and the ElevenLabs
agents are configured with ulaw_8000 for BOTH directions — so this is a direct
base64 passthrough with no resampling or transcoding. That is what CLAUDE.md's
"keep TTS output format ulaw_8000 for the phone path" rule is for.

    Plivo call  --(media: base64 mu-law)-->  ElevenLabs  (user_audio_chunk)
    Plivo call  <--(playAudio: base64)-----  ElevenLabs  (audio_event)

The wire protocol below was captured from the live service, not taken from the
SDK's type stubs — those do not mirror it (the same assumption produced a real
bug in elevenlabs_client.agent_exists). Observed server messages:

    conversation_initiation_metadata  {conversation_id, agent_output_audio_format, ...}
    audio                             {audio_event: {audio_base_64, event_id, is_final}}
    agent_response                    {agent_response_event: {agent_response, event_id}}
    user_transcript                   {user_transcription_event: {user_transcript}}
    interruption                      {interruption_event: {event_id}}
    ping                              {ping_event: {event_id, ping_ms}}   <- must pong

Client messages: an initiation frame carrying dynamic_variables, then
{"user_audio_chunk": <base64>} per media frame, and {"type":"pong","event_id":N}.
"""

from __future__ import annotations

import asyncio
import json
import logging
from xml.sax.saxutils import escape

import websockets
from fastapi import WebSocket, WebSocketDisconnect

from app.config import (
    CALL_MAX_DURATION_S,
    CALL_WEBHOOK_SECRET,
    ELEVENLABS_API_KEY,
    ONEWAY_MAX_SILENT_S,
    ONEWAY_SILENCE_TAIL_S,
    PLIVO_GREETING_GRACE_MS,
    PUBLIC_BASE_URL,
)
from app.db.models import TranscriptTurn
from app.telephony.elevenlabs_client import get_client

logger = logging.getLogger(__name__)

# mu-law at 8 kHz is 8000 bytes/sec, and base64 inflates by 4/3 — so decoded
# bytes = len(b64) * 3/4, and seconds = that / 8000. Used to estimate when
# audio already handed to Plivo will finish playing.
_ULAW_BYTES_PER_S = 8000.0

# How often the one-way watchdog re-checks whether the message has finished.
# Fine-grained enough that the poll adds no audible tail of its own.
_ONEWAY_POLL_S = 0.25


def answer_xml(lead_id: str) -> str:
    """The XML Plivo fetches on answer: open a bidirectional mu-law stream back
    to this server.

    keepCallAlive keeps the call up while the stream runs; without it Plivo
    treats the (empty) rest of the XML document as the whole call and hangs up
    the moment the stream is established.
    """
    base = (PUBLIC_BASE_URL or "").rstrip("/")
    wss = base.replace("https://", "wss://").replace("http://", "ws://")
    from urllib.parse import quote
    url = (f"{wss}/calls/stream?token={quote(CALL_WEBHOOK_SECRET or '')}"
           f"&lead={quote(lead_id)}")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        '<Stream bidirectional="true" keepCallAlive="true" '
        f'contentType="audio/x-mulaw;rate=8000">{escape(url)}</Stream>'
        "</Response>"
    )


async def _play(plivo_ws: WebSocket, payload_b64: str) -> None:
    """Send agent audio into the call."""
    await plivo_ws.send_text(json.dumps({
        "event": "playAudio",
        "media": {"contentType": "audio/x-mulaw", "sampleRate": 8000,
                  "payload": payload_b64},
    }))


async def _clear(plivo_ws: WebSocket, stream_id: str | None) -> None:
    """Drop audio already buffered on the call, so barge-in is immediate rather
    than waiting for the agent's queued speech to drain."""
    msg: dict = {"event": "clearAudio"}
    if stream_id:
        msg["streamId"] = stream_id
    await plivo_ws.send_text(json.dumps(msg))


async def _signed_url(agent_id: str) -> str:
    """A short-lived signed WebSocket URL for *agent_id*.

    get_signed_url is a synchronous SDK call; off the event loop because this
    server also runs the scheduler, the call worker and any other live
    bridges — blocking here would stall all of them.
    """
    client = get_client()
    resp = await asyncio.to_thread(
        client.conversational_ai.conversations.get_signed_url, agent_id=agent_id
    )
    return resp.signed_url


async def bridge(plivo_ws: WebSocket, *, agent_id: str, lead_id: str,
                 dynamic_variables: dict | None = None,
                 language: str | None = None,
                 one_way: bool = False, outcome: dict | None = None) -> dict:
    """Bridge one answered call until either side ends.

    Returns {status, turns, transcript, conversation_id} — the same shape the
    transcript webhook used to produce, so app/db/calls.py's recording path is
    unchanged.

    one_way: deliver the message and hang up without listening. The caller's
    audio is never forwarded, so the agent cannot respond to it.

    language: an ISO code ('te', 'hi', 'en') for ElevenLabs to run the
    conversation in, or None to send no override at all. None is not the same
    as 'the default language': ElevenLabs RAISES if an override arrives for a
    field that is not enabled in the agent's Security tab, so a campaign that
    never asked for a language must send a frame with no override key in it.
    See app/languages.py.

    *outcome*: an optional dict for the caller to OWN, populated in place as
    the call progresses. It exists because this function's results used to be
    reachable only through its return value, so a mid-call exception threw
    them away: app/telephony/call_routes.py kept its own default dict, and a
    three-minute conversation that ended with a ConnectionClosed escaping the
    `async with` below was recorded as zero turns, no conversation_id, and a
    'failed' lead that then burned a retry. Callers that pass their own dict
    still see every turn collected before the failure. The same object is
    returned either way, so callers that ignore this keep working unchanged.
    """
    if outcome is None:
        outcome = {}
    outcome.update({"status": "failed", "turns": 0, "transcript": [],
                    "conversation_id": None})
    if not ELEVENLABS_API_KEY:
        outcome["status"] = "failed"
        return outcome

    loop = asyncio.get_running_loop()
    signed = await _signed_url(agent_id)

    stop = asyncio.Event()
    state = {
        "stream_id": None,
        # Loop-time at which audio already sent to Plivo finishes playing. The
        # agent can emit audio faster than real time, so a graceful hangup must
        # wait on this rather than a fixed sleep, or the last words are cut off.
        "play_end": 0.0,
        # The lead's first sound on an outbound call is almost always "Hello?" —
        # an acknowledgement, not an interruption. Cancelling the greeting on it
        # made the agent restart from the top. Barge-in is live after this.
        "grace_until": 0.0,
        # Whether ANY agent audio has arrived yet. Tracked separately from
        # play_end because that starts at 0.0, which the one-way watchdog
        # cannot tell apart from "the message already finished playing" — it
        # would hang up before the agent had said a word.
        "audio_seen": False,
        "exit_reason": "unknown",
    }
    turns: list[TranscriptTurn] = []

    def _exit(reason: str) -> None:
        if state["exit_reason"] == "unknown":
            state["exit_reason"] = reason

    try:
        async with websockets.connect(signed, max_size=16 * 1024 * 1024) as el_ws:
            init: dict = {
                "type": "conversation_initiation_client_data",
                "dynamic_variables": dynamic_variables or {},
            }
            if language:
                # Only when a language was actually chosen — see the docstring.
                init["conversation_config_override"] = {"agent": {"language": language}}
            await el_ws.send(json.dumps(init))

            # ── Plivo → ElevenLabs ───────────────────────────────────────────────
            async def from_plivo() -> None:
                try:
                    while not stop.is_set():
                        raw = await plivo_ws.receive_text()
                        try:
                            msg = json.loads(raw)
                        except (ValueError, TypeError):
                            continue
                        ev = msg.get("event")
                        if ev == "start":
                            st = msg.get("start") or {}
                            state["stream_id"] = msg.get("streamId") or st.get("streamId")
                            if not one_way and PLIVO_GREETING_GRACE_MS > 0:
                                state["grace_until"] = (
                                    loop.time() + PLIVO_GREETING_GRACE_MS / 1000.0
                                )
                        elif ev == "media":
                            if one_way:
                                continue  # never listen on a one-way call
                            payload = (msg.get("media") or {}).get("payload")
                            if payload:
                                await el_ws.send(json.dumps({"user_audio_chunk": payload}))
                        elif ev == "stop":
                            _exit("plivo_stop")
                            break
                except WebSocketDisconnect:
                    _exit("plivo_disconnect")
                except Exception as exc:
                    _exit(f"error:{type(exc).__name__}")
                    logger.info(f"[bridge] Plivo reader stopped: {type(exc).__name__}: {exc}")
                finally:
                    stop.set()

            # ── ElevenLabs → Plivo ───────────────────────────────────────────────
            async def from_elevenlabs() -> None:
                try:
                    async for raw in el_ws:
                        if stop.is_set():
                            break
                        try:
                            event = json.loads(raw)
                        except (ValueError, TypeError):
                            continue
                        etype = event.get("type", "")

                        if etype == "audio":
                            b64 = (event.get("audio_event") or {}).get("audio_base_64")
                            if b64:
                                await _play(plivo_ws, b64)
                                dur = (len(b64) * 3 / 4) / _ULAW_BYTES_PER_S
                                state["play_end"] = max(state["play_end"], loop.time()) + dur
                                state["audio_seen"] = True
                        elif etype == "agent_response":
                            text = (event.get("agent_response_event") or {}).get("agent_response")
                            if text:
                                turns.append(TranscriptTurn(role="agent", text=text.strip()))
                        elif etype == "user_transcript":
                            text = (event.get("user_transcription_event") or {}).get(
                                "user_transcript")
                            if text:
                                turns.append(TranscriptTurn(role="lead", text=text.strip()))
                        elif etype == "interruption":
                            # Honour barge-in, except during the greeting window.
                            if loop.time() >= state["grace_until"]:
                                await _clear(plivo_ws, state["stream_id"])
                                state["play_end"] = 0.0
                        elif etype == "ping":
                            # Unanswered pings cause the server to drop the socket.
                            ev = event.get("ping_event") or {}
                            await el_ws.send(json.dumps({"type": "pong",
                                                         "event_id": ev.get("event_id")}))
                        elif etype == "conversation_initiation_metadata":
                            meta = event.get("conversation_initiation_metadata_event") or {}
                            outcome["conversation_id"] = meta.get("conversation_id")
                            fmt = meta.get("agent_output_audio_format")
                            if fmt != "ulaw_8000":
                                # Would be silence or noise on the line: Plivo is told
                                # the stream is mu-law, so anything else is misframed.
                                logger.error(
                                    f"[bridge] agent {agent_id} output format is {fmt!r}, "
                                    "not 'ulaw_8000' — the caller will hear noise. Fix the "
                                    "agent's audio format in ElevenLabs."
                                )
                except websockets.exceptions.ConnectionClosed as exc:
                    _exit(f"elevenlabs_closed:{exc.code}")
                    # 1002 with a payment message is a billing stop, not a bug —
                    # surface it plainly rather than as a generic disconnect.
                    logger.warning(f"[bridge] ElevenLabs closed the socket: {exc}")
                except Exception as exc:
                    _exit(f"error:{type(exc).__name__}")
                    logger.error(f"[bridge] ElevenLabs reader failed: "
                                 f"{type(exc).__name__}: {exc}")
                finally:
                    stop.set()

            # ── One-way: end the call when the message has been delivered ────────
            async def oneway_watchdog() -> None:
                """Nothing else can end a one-way call.

                A one-way bridge never forwards the lead's audio (see from_plivo),
                so ElevenLabs receives no turn-end signal and never closes its
                socket; from_plivo only returns if the lead hangs up first. That
                left `stop` unset and the bridge parked on CALL_MAX_DURATION_S —
                300 seconds of billed Plivo airtime and a 300-second ElevenLabs
                conversation for a 20-second message, with the lead listening to
                silence for the remainder.

                So: wait for the agent's audio to drain, allow a short tail, then
                end the call ourselves. Bounded at both ends — an agent that never
                speaks at all costs ONEWAY_MAX_SILENT_S, not the full duration.
                """
                started = loop.time()
                while not stop.is_set():
                    await asyncio.sleep(_ONEWAY_POLL_S)
                    now = loop.time()
                    if not state["audio_seen"]:
                        if now - started >= ONEWAY_MAX_SILENT_S:
                            logger.warning(
                                f"[bridge] lead={lead_id} one-way agent produced no audio "
                                f"in {ONEWAY_MAX_SILENT_S}s — ending the call. Check the "
                                "agent's prompt and its dynamic variables."
                            )
                            _exit("oneway_no_audio")
                            break
                        continue
                    if now >= state["play_end"] + ONEWAY_SILENCE_TAIL_S:
                        _exit("oneway_complete")
                        break
                stop.set()

            tasks = [from_plivo(), from_elevenlabs()]
            if one_way:
                tasks.append(oneway_watchdog())
            reader = asyncio.gather(*tasks, return_exceptions=True)
            try:
                # A call that never ends would hold a concurrency slot forever, so
                # the whole bridge is bounded rather than trusting either side.
                await asyncio.wait_for(stop.wait(), timeout=CALL_MAX_DURATION_S)
            except asyncio.TimeoutError:
                _exit("max_duration")
                stop.set()
            finally:
                # Let the agent's final words play out before the line drops.
                remaining = state["play_end"] - loop.time()
                if remaining > 0:
                    await asyncio.sleep(min(remaining + 0.3, 15))
                reader.cancel()
                try:
                    await reader
                except (asyncio.CancelledError, Exception):
                    pass
    finally:
        # In a `finally` so a mid-call exception — a ConnectionClosed escaping
        # the `async with` above, say — still hands the caller everything
        # collected up to that point. These used to run only on the normal
        # path, so a three-minute conversation that ended badly was recorded
        # as zero turns with no conversation_id, and the lead was marked
        # 'failed' and re-dialled.
        outcome["turns"] = len(turns)
        outcome["transcript"] = turns
        # "done" only when the call ran its course; a bridge that fell over
        # should not look like a completed conversation in the calls table.
        # oneway_complete belongs here: it IS the natural end of a one-way
        # call, and without it every successful one-way call would be stored
        # 'failed', mark its lead 'failed', and burn a retry attempt.
        outcome["status"] = ("done" if state["exit_reason"] in
                             ("plivo_stop", "plivo_disconnect", "max_duration",
                              "oneway_complete")
                             else "failed")
        logger.info(f"[bridge] lead={lead_id} ended reason={state['exit_reason']} "
                    f"turns={len(turns)}")
    return outcome
