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
                 one_way: bool = False) -> dict:
    """Bridge one answered call until either side ends.

    Returns {status, turns, transcript, conversation_id} — the same shape the
    transcript webhook used to produce, so app/db/calls.py's recording path is
    unchanged.

    one_way: deliver the message and hang up without listening. The caller's
    audio is never forwarded, so the agent cannot respond to it.
    """
    outcome: dict = {"status": "failed", "turns": 0, "transcript": [],
                     "conversation_id": None}
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
        "exit_reason": "unknown",
    }
    turns: list[TranscriptTurn] = []

    def _exit(reason: str) -> None:
        if state["exit_reason"] == "unknown":
            state["exit_reason"] = reason

    async with websockets.connect(signed, max_size=16 * 1024 * 1024) as el_ws:
        await el_ws.send(json.dumps({
            "type": "conversation_initiation_client_data",
            "dynamic_variables": dynamic_variables or {},
        }))

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

        reader = asyncio.gather(from_plivo(), from_elevenlabs(),
                                return_exceptions=True)
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

    outcome["turns"] = len(turns)
    outcome["transcript"] = turns
    # "done" only when the call ran its course; a bridge that fell over should
    # not look like a completed conversation in the calls table.
    outcome["status"] = ("done" if state["exit_reason"] in
                         ("plivo_stop", "plivo_disconnect", "max_duration")
                         else "failed")
    logger.info(f"[bridge] lead={lead_id} ended reason={state['exit_reason']} "
                f"turns={len(turns)}")
    return outcome
