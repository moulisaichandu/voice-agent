"""telephony/openai_bridge.py — Bridge a live Plivo call to OpenAI Realtime.

The Telugu backend. Exists because the ElevenLabs Agents platform does not
offer Telugu at all (see app/languages.py's ELEVENLABS_AGENT_LANGUAGES) and
Telugu is this business's primary market language.

Deliberately mirrors app/telephony/bridge.py's signature and `outcome`
contract exactly, so app/telephony/call_routes.py picks a backend rather than
branching into two different call shapes, and everything downstream — call
recording, Sheets write-back, slot release, retry accounting — never learns
which one ran.

All the Plivo-side behaviour (playing audio, barge-in, the one-way watchdog
that stops a message-only call billing 300 seconds of silence) lives in
app/telephony/plivo_stream.py and is shared with the ElevenLabs bridge. This
module owns only OpenAI's wire protocol.

ONE-WAY ONLY for now. Two-way — turn detection, input transcription, tools,
lead transcripts, barge-in — is Milestone B and is gated on a live one-way
call first; see bridge()'s warning for what happens if a two-way campaign is
routed here before then.

Wire protocol, captured from the sibling ../ai-voice-agent's working
implementation rather than assumed from documentation:

    connect  wss://api.openai.com/v1/realtime?model=<model>
             header Authorization: Bearer <OPENAI_API_KEY>
    ->       {"type": "session.update", "session": {...}}
    ->       {"type": "response.create"}                  (cue to speak)
    ->       {"type": "input_audio_buffer.append", "audio": <b64 mu-law>}
    <-       response.output_audio.delta            {delta: <b64 mu-law>}
    <-       response.output_audio_transcript.delta {delta: <text>}
    <-       response.done                          {response: {status, output}}
    <-       error

Both legs are mu-law 8 kHz (`audio/pcmu`), so audio is a passthrough with no
transcoding — the same property the ElevenLabs path has.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import websockets
from fastapi import WebSocket

from app.config import (
    OPENAI_API_KEY,
    OPENAI_REALTIME_MODEL,
    OPENAI_REALTIME_NOISE_REDUCTION,
    OPENAI_REALTIME_VOICE,
)
from app.db.models import TranscriptTurn

# The MODULE, not just PlivoCall: everything Plivo-side is deliberately in one
# place, and importing it this way keeps that visible (and keeps its tunables
# patchable in tests without reaching into another module's namespace).
from app.telephony import openai_prompts, plivo_stream

logger = logging.getLogger(__name__)

_REALTIME_URL = "wss://api.openai.com/v1/realtime?model={model}"


async def _connect(url: str, headers: dict[str, str]) -> Any:
    """Open the Realtime socket, tolerating both `websockets` header kwargs.

    websockets >= 14 renamed `extra_headers` to `additional_headers`; try the
    new name and fall back to the old, exactly as the sibling does — the pinned
    version is 15.x, but this module must not break on an older environment.
    """
    try:
        return await websockets.connect(url, additional_headers=headers, max_size=None)
    except TypeError:
        return await websockets.connect(url, extra_headers=headers, max_size=None)


def _session_update(*, lead_name: str | None, script: str | None,
                    language_style: str | None) -> dict:
    """The session frame sent immediately after the socket opens.

    No tools and no input transcription: a one-way call has nothing to look up
    and never forwards the lead's audio, so transcribing it would be paying to
    transcribe audio the model never receives.
    """
    output_cfg: dict = {"format": {"type": "audio/pcmu"}}
    if OPENAI_REALTIME_VOICE:
        # Blank means "whatever the API defaults to" — see config.py on why no
        # voice is Telugu-native and how the sibling picked theirs by ear.
        output_cfg["voice"] = OPENAI_REALTIME_VOICE

    input_cfg: dict = {"format": {"type": "audio/pcmu"}}
    if OPENAI_REALTIME_NOISE_REDUCTION:
        input_cfg["noise_reduction"] = {"type": OPENAI_REALTIME_NOISE_REDUCTION}
    # Unconditionally off: the model must never take a turn against a lead it
    # is not listening to. Milestone B's two-way path overrides this.
    input_cfg["turn_detection"] = None

    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": openai_prompts.one_way_instructions(
                lead_name, script, language_style=language_style,
            ),
            "output_modalities": ["audio"],
            "audio": {"input": input_cfg, "output": output_cfg},
        },
    }


async def bridge(plivo_ws: WebSocket, *, agent_id: str, lead_id: str,
                 dynamic_variables: dict | None = None,
                 language: str | None = None,
                 one_way: bool = False, outcome: dict | None = None) -> dict:
    """Bridge one answered call to OpenAI Realtime until either side ends.

    Same contract as app/telephony/bridge.py's bridge(), so call_routes can
    dispatch on backend without reshaping the call:

    *agent_id* is accepted and IGNORED — OpenAI Realtime has no agent objects.
    *language* is accepted and ignored too: there is no per-conversation
    language override on this backend, the persona is Telugu by construction
    (see openai_prompts), and the STT language is pinned in config. It stays in
    the signature so the two bridges remain drop-in substitutes.

    *outcome* is populated in place and returned, so a mid-call exception still
    hands the caller every turn collected before it. The ElevenLabs bridge's
    docstring explains why that matters: a three-minute conversation that ended
    badly used to be recorded as zero turns and a 'failed' lead that then
    burned a retry. `conversation_id` stays None — Realtime has no equivalent,
    and calls.el_conversation_id is nullable.
    """
    if outcome is None:
        outcome = {}
    outcome.update({"status": "failed", "turns": 0, "transcript": [],
                    "conversation_id": None})
    if not OPENAI_API_KEY:
        # Return rather than raise, so call_routes still records the call and
        # releases the slot instead of unwinding through its finally.
        logger.error("[openai] OPENAI_API_KEY is not set — cannot bridge the call.")
        return outcome
    if not one_way:
        # Task 4 must not route a two-way campaign here yet. If one arrives,
        # this says so out loud: turn detection is off and nothing cues a
        # response, so the lead would otherwise hear silence for
        # CALL_MAX_DURATION_S with no logged reason.
        logger.warning(f"[openai] lead={lead_id} two-way is not implemented on this "
                       "backend yet (Milestone B) — the agent will not speak.")

    variables = dynamic_variables or {}
    call = plivo_stream.PlivoCall(plivo_ws, lead_id=lead_id, one_way=one_way)
    turns: list[TranscriptTurn] = []
    agent_text: list[str] = []

    url = _REALTIME_URL.format(model=OPENAI_REALTIME_MODEL)
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    try:
        oa = await _connect(url, headers)
        async with oa:
            await oa.send(json.dumps(_session_update(
                lead_name=variables.get("lead_name"),
                script=variables.get("script"),
                # Put here by call_routes._call_language() — the exact style
                # text for te vs tinglish, so this backend's persona actually
                # differs by register instead of one hardcoded rule for both.
                language_style=variables.get("language_style"),
            )))
            # One-way: ask for the message immediately. Nothing else will —
            # with turn detection off the model waits for an explicit cue.
            # (The sibling sends this on Plivo's `start` event instead; here
            # the audio is queued on the same socket Plivo just opened, which
            # is what the ElevenLabs path already relies on.)
            if one_way:
                await oa.send(json.dumps({"type": "response.create"}))

            # ── Plivo → OpenAI ───────────────────────────────────────────────
            # Never called on a one-way call: PlivoCall.read_events drops the
            # lead's media itself. Wired now so Milestone B has one less thing
            # to add to the audio path.
            async def to_openai(payload_b64: str) -> None:
                await oa.send(json.dumps({
                    "type": "input_audio_buffer.append", "audio": payload_b64,
                }))

            # ── OpenAI → Plivo ───────────────────────────────────────────────
            async def from_openai() -> None:
                try:
                    async for raw in oa:
                        if call.stop.is_set():
                            break
                        try:
                            event = json.loads(raw)
                        except (ValueError, TypeError):
                            continue
                        etype = event.get("type", "")

                        if etype in ("response.output_audio.delta",
                                     "response.audio.delta"):
                            delta = event.get("delta")
                            if delta:
                                await call.play(delta)
                        elif etype in ("response.output_audio_transcript.delta",
                                       "response.audio_transcript.delta"):
                            piece = event.get("delta")
                            if piece:
                                agent_text.append(piece)
                        elif etype == "response.done":
                            resp = event.get("response") or {}
                            status = resp.get("status")
                            if status in ("failed", "incomplete"):
                                # NOT delivered as an `error` event, so without
                                # this the agent simply stops talking and the
                                # log says nothing. Quota exhaustion, rate
                                # limits and content filtering all land here.
                                details = resp.get("status_details") or {}
                                logger.error(f"[openai] lead={lead_id} response "
                                             f"{status}: {details.get('error') or details}")
                            # Transcript deltas arrive piecemeal; a completed
                            # response is the only place they form a turn.
                            said = "".join(agent_text).strip()
                            agent_text.clear()
                            if said:
                                turns.append(TranscriptTurn(role="agent", text=said))
                        elif etype == "error":
                            logger.error(f"[openai] lead={lead_id} error event: "
                                         f"{event.get('error') or event}")
                except websockets.exceptions.ConnectionClosed as exc:
                    call.note_exit(f"openai_closed:{exc.code}")
                    logger.warning(f"[openai] socket closed: {exc}")
                except Exception as exc:
                    call.note_exit(f"error:{type(exc).__name__}")
                    logger.error(f"[openai] reader failed: "
                                 f"{type(exc).__name__}: {exc}")
                finally:
                    call.stop.set()

            # PlivoCall.run adds the one-way watchdog itself, bounds the whole
            # call at CALL_MAX_DURATION_S, and lets queued audio drain before
            # the line drops.
            await call.run(call.read_events(to_openai), from_openai())
    except Exception as exc:
        # Swallowed for the same reason the no-key branch returns: the caller's
        # finally has to record the call, not re-raise past it.
        call.note_exit(f"error:{type(exc).__name__}")
        logger.exception(f"[openai] bridge failed for lead {lead_id}: "
                         f"{type(exc).__name__}: {exc}")
    finally:
        # In a finally for the same reason the ElevenLabs bridge is: a call
        # that really happened must not be recorded as nothing.
        trailing = "".join(agent_text).strip()
        if trailing:
            # The socket died mid-turn: keep what was said rather than dropping
            # a turn that never got its response.done.
            turns.append(TranscriptTurn(role="agent", text=trailing))
        outcome["turns"] = len(turns)
        outcome["transcript"] = turns
        # "done" only when the call ran its course — see PlivoCall.ran_its_course
        # for which exit reasons count (oneway_complete is one of them).
        outcome["status"] = "done" if call.ran_its_course else "failed"
        logger.info(f"[openai] lead={lead_id} ended reason={call.exit_reason} "
                    f"turns={len(turns)}")
    return outcome
