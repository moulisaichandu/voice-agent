"""telephony/bridge.py — Bridge a live Plivo call to an ElevenLabs agent.

Everything here is ElevenLabs-specific: the signed-URL handshake, that
vendor's WebSocket wire protocol, and the translation of its events into
audio on the line and transcript turns. The call itself — the answer XML,
sending and clearing audio, and the one-way watchdog that ends a call once
its message has been delivered — lives in app/telephony/plivo_stream.py, so
a second voice backend can reuse it verbatim.

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

import websockets
from fastapi import WebSocket

from app.config import ELEVENLABS_API_KEY
from app.db.models import TranscriptTurn
from app.telephony.elevenlabs_client import get_client
from app.telephony.plivo_stream import PlivoCall

logger = logging.getLogger(__name__)


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

    signed = await _signed_url(agent_id)

    call = PlivoCall(plivo_ws, lead_id=lead_id, one_way=one_way)
    turns: list[TranscriptTurn] = []

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
            async def to_elevenlabs(payload_b64: str) -> None:
                await el_ws.send(json.dumps({"user_audio_chunk": payload_b64}))

            # ── ElevenLabs → Plivo ───────────────────────────────────────────────
            async def from_elevenlabs() -> None:
                try:
                    async for raw in el_ws:
                        if call.stop.is_set():
                            break
                        try:
                            event = json.loads(raw)
                        except (ValueError, TypeError):
                            continue
                        etype = event.get("type", "")

                        if etype == "audio":
                            b64 = (event.get("audio_event") or {}).get("audio_base_64")
                            if b64:
                                await call.play(b64)
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
                            await call.interrupt()
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
                    call.note_exit(f"elevenlabs_closed:{exc.code}")
                    # 1002 with a payment message is a billing stop, not a bug —
                    # surface it plainly rather than as a generic disconnect.
                    logger.warning(f"[bridge] ElevenLabs closed the socket: {exc}")
                except Exception as exc:
                    call.note_exit(f"error:{type(exc).__name__}")
                    logger.error(f"[bridge] ElevenLabs reader failed: "
                                 f"{type(exc).__name__}: {exc}")
                finally:
                    call.stop.set()

            # PlivoCall.run adds the one-way watchdog itself, bounds the whole
            # call at CALL_MAX_DURATION_S, and lets queued audio drain before
            # the line drops.
            await call.run(call.read_events(to_elevenlabs), from_elevenlabs())
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
        # See PlivoCall.ran_its_course for which reasons count.
        outcome["status"] = "done" if call.ran_its_course else "failed"
        logger.info(f"[bridge] lead={lead_id} ended reason={call.exit_reason} "
                    f"turns={len(turns)}")
    return outcome
