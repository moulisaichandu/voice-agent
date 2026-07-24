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

Both call shapes live here. One-way delivers a message and hangs up (turn
detection off, no tools, the lead's audio never forwarded). Two-way holds a
conversation: turn detection on, the STT language pinned, lead transcripts
captured, barge-in handled with conversation.item.truncate so the model's
belief about what it said matches what the lead actually heard, and course
questions answered via the search_course_material tool — which calls
search_relevant() ONLY (CLAUDE.md's hard rule; see _SEARCH_TOOL's comment).
Two-way campaigns are still refused at campaign creation and in preflight for
languages this backend serves, pending a live call verifying the conversation
end to end (see the plan's Milestone B gate) — not because the tool is
missing.

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
    OPENAI_REALTIME_SILENCE_MS,
    OPENAI_REALTIME_STT_LANGUAGE,
    OPENAI_REALTIME_STT_MODEL,
    OPENAI_REALTIME_VAD_THRESHOLD,
    OPENAI_REALTIME_VOICE,
)
from app.db.models import TranscriptTurn
from app.rag.search import search_relevant

# The MODULE, not just PlivoCall: everything Plivo-side is deliberately in one
# place, and importing it this way keeps that visible (and keeps its tunables
# patchable in tests without reaching into another module's namespace).
from app.telephony import openai_prompts, plivo_stream

logger = logging.getLogger(__name__)

_REALTIME_URL = "wss://api.openai.com/v1/realtime?model={model}"

# The live course-question tool, two-way only (see _session_update). Calls
# search_relevant() ONLY — CLAUDE.md's hard rule. search_relevant applies the
# relevance floor (RAG_MIN_SCORE) and always returns something speakable, so
# the agent never recites irrelevant course text as though it were an answer.
# search_permissive() must never be wired to this path.
_SEARCH_TOOL_NAME = "search_course_material"
_SEARCH_TOOL: dict = {
    "type": "function",
    "name": _SEARCH_TOOL_NAME,
    "description": (
        "Search Digital Brolly's course documents for material relevant to the "
        "lead's question. Call this for every question about courses, fees, "
        "timings, batches or placement — never answer those from memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The lead's question, as a concise search query in English.",
            },
        },
        "required": ["query"],
    },
}

# The only way a two-way call ends itself. Without this the model has no tool
# to hang up with, so once the conversation is naturally over it just keeps
# the line open — the lead sits there until they hang up themselves or the
# call runs all the way to CALL_MAX_DURATION_S. One-way needs nothing here:
# oneway_watchdog already ends those calls.
_END_CALL_TOOL_NAME = "end_call"
_END_CALL_TOOL: dict = {
    "type": "function",
    "name": _END_CALL_TOOL_NAME,
    "description": (
        "Hang up the call. Call this once the conversation has reached a "
        "natural close — goodbyes exchanged, or the lead has no more "
        "questions and confirms they're done."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

# Event types whose `item_id` names the ASSISTANT item currently being
# spoken — the only ones allowed to update `last_item_id` (see the comment
# at its use in `from_openai()` for why this must be a whitelist, not a
# blanket "any item_id wins").
_ASSISTANT_ITEM_EVENTS = frozenset({
    "response.created",
    "response.output_audio.delta", "response.audio.delta",
    "response.output_audio_transcript.delta", "response.audio_transcript.delta",
})


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
                    language_style: str | None, one_way: bool) -> dict:
    """The session frame sent immediately after the socket opens.

    One-way: no tools and no input transcription — a one-way call has nothing to
    look up and never forwards the lead's audio, so transcribing it would be
    paying to transcribe audio the model never receives, and turn detection is
    off so the model never takes a turn against a lead it is not listening to.

    Two-way: turn detection on so the model answers when the lead stops, plus a
    transcription config with the STT language PINNED — on 8 kHz phone audio
    auto-detection was observed hearing Telugu as Croatian/Urdu, so the lead
    would be transcribed as nonsense and answered with nonsense (see config.py's
    OPENAI_REALTIME_STT_LANGUAGE). This is the path where the lead actually
    speaks, so it is where the pin has to be.
    """
    output_cfg: dict = {"format": {"type": "audio/pcmu"}}
    if OPENAI_REALTIME_VOICE:
        # Blank means "whatever the API defaults to" — see config.py on why no
        # voice is Telugu-native and how the sibling picked theirs by ear.
        output_cfg["voice"] = OPENAI_REALTIME_VOICE

    input_cfg: dict = {"format": {"type": "audio/pcmu"}}
    if OPENAI_REALTIME_NOISE_REDUCTION:
        input_cfg["noise_reduction"] = {"type": OPENAI_REALTIME_NOISE_REDUCTION}
    if one_way:
        # The model must never take a turn against a lead it is not listening to.
        input_cfg["turn_detection"] = None
        instructions = openai_prompts.one_way_instructions(
            lead_name, script, language_style=language_style,
        )
    else:
        transcription: dict = {"model": OPENAI_REALTIME_STT_MODEL}
        if OPENAI_REALTIME_STT_LANGUAGE:
            transcription["language"] = OPENAI_REALTIME_STT_LANGUAGE
        input_cfg["transcription"] = transcription
        input_cfg["turn_detection"] = {
            "type": "server_vad",
            "threshold": OPENAI_REALTIME_VAD_THRESHOLD,
            "prefix_padding_ms": 300,
            # The sibling floors this at 600ms: below it the model cuts leads
            # off mid-sentence on a natural pause.
            "silence_duration_ms": max(OPENAI_REALTIME_SILENCE_MS, 600),
        }
        instructions = openai_prompts.two_way_instructions(
            lead_name, script, language_style=language_style,
        )

    session: dict = {
        "type": "realtime",
        "instructions": instructions,
        "output_modalities": ["audio"],
        "audio": {"input": input_cfg, "output": output_cfg},
    }
    if not one_way:
        # One-way gets no tools at all — there is nobody to answer, and a
        # tool the model can never get a reply from is worse than none.
        session["tools"] = [_SEARCH_TOOL, _END_CALL_TOOL]
        session["tool_choice"] = "auto"

    return {"type": "session.update", "session": session}


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
    variables = dynamic_variables or {}
    call = plivo_stream.PlivoCall(plivo_ws, lead_id=lead_id, one_way=one_way)
    turns: list[TranscriptTurn] = []
    agent_text: list[str] = []
    # The id of the response item currently being spoken, tracked from the
    # events that carry it. Barge-in truncation needs it to tell the model which
    # item to trim to what the lead actually heard.
    last_item_id: str | None = None
    # Whether an OpenAI response is currently in progress. A barge-in may only
    # send response.cancel while one is: cancelling after response.done has
    # arrived (its audio still draining on the Plivo side) is rejected as
    # response_cancel_not_active. Set on any assistant output event, cleared on
    # response.done.
    response_active = False
    # The assistant item whose audio is currently playing out. When a NEW item's
    # audio begins (the tool-call flow queues one back-to-back with the last),
    # the playback clock must restart or played_ms() spans both items and the
    # barge-in truncate overshoots the one it names — see PlivoCall.played_ms().
    audio_seg_item: str | None = None

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
                one_way=one_way,
            )))
            # The AI placed this call, so it has to speak first — and the
            # AI-disclosure compliance rule requires that first line to be the
            # disclosure. One-way needs this because turn detection is off
            # there and nothing else would ever cue a response; two-way needs
            # it exactly as much, because server_vad only responds to the
            # LEAD speaking — with no explicit cue, the model just waits for
            # the lead to talk first, which is backwards for an outbound call
            # and left leads sitting in silence. (The sibling ../ai-voice-agent
            # fires this on Plivo's `start` event instead; here the audio is
            # queued on the same socket Plivo just opened, which this
            # project's ElevenLabs path already relies on, so firing it right
            # after session.update works the same way for both call shapes.)
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
                nonlocal last_item_id, response_active, audio_seg_item
                try:
                    async for raw in oa:
                        if call.stop.is_set():
                            break
                        try:
                            event = json.loads(raw)
                        except (ValueError, TypeError):
                            continue
                        etype = event.get("type", "")

                        # Track the current item id ONLY from the AGENT's own
                        # output events — a WHITELIST (_ASSISTANT_ITEM_EVENTS),
                        # not the old "any event carrying item_id wins, most
                        # recent takes it" logic.
                        #
                        # Why the blanket version was a real bug, not just
                        # untidy: input_audio_buffer.speech_started ALSO
                        # carries a top-level item_id, but per the OpenAI
                        # Realtime API schema it is "the ID of the user
                        # message item that will be created when speech
                        # stops" — the LEAD's upcoming item, not the
                        # assistant's (conversation.item.input_audio_
                        # transcription.completed is the same: the lead's
                        # item). conversation.item.truncate below only
                        # operates on ASSISTANT audio items, so if a lead's
                        # item_id ever won here the API would reject the
                        # truncate outright and the model's belief about
                        # what it said would never be corrected — silently
                        # reintroducing the sibling's unresolved bug
                        # (ARCHITECTURE.md:268-276).
                        #
                        # A whitelist, not a blocklist, is deliberate: a
                        # future event type must be added here on purpose
                        # before it can touch last_item_id, so it cannot
                        # silently win by omission the way the old logic did.
                        # Do NOT "simplify" this back to a blanket update.
                        if etype in _ASSISTANT_ITEM_EVENTS:
                            # Any assistant output event means a response is in
                            # progress — see response_active's comment (Bug C).
                            response_active = True
                            item_id = event.get("item_id")
                            if item_id:
                                last_item_id = item_id

                        if etype in ("response.output_audio.delta",
                                     "response.audio.delta"):
                            delta = event.get("delta")
                            if delta:
                                item_id = event.get("item_id")
                                if (item_id and audio_seg_item is not None
                                        and item_id != audio_seg_item):
                                    # A new response's audio begins back-to-back
                                    # with the last (the tool-call flow): restart
                                    # the playback clock so a barge-in truncate
                                    # measures only this item (Bug B).
                                    call.mark_response_boundary()
                                if item_id:
                                    audio_seg_item = item_id
                                await call.play(delta)
                        elif etype in ("response.output_audio_transcript.delta",
                                       "response.audio_transcript.delta"):
                            piece = event.get("delta")
                            if piece:
                                agent_text.append(piece)
                        elif etype == "response.done":
                            # The response is finished; its audio may still be
                            # draining on the Plivo side, but there is no longer
                            # an active response to cancel (Bug C).
                            response_active = False
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
                            for item in resp.get("output") or []:
                                if item.get("type") != "function_call":
                                    continue
                                name = item.get("name")
                                if name == _END_CALL_TOOL_NAME:
                                    # PlivoCall.run() already waits for this
                                    # response's queued audio (the goodbye
                                    # line, sent as audio deltas earlier in
                                    # this same response) to finish playing
                                    # before hanging up — the same draining
                                    # oneway_complete relies on. No
                                    # function_call_output needed; the call
                                    # is ending, not continuing.
                                    call.note_exit("openai_end_call")
                                    call.stop.set()
                                    continue
                                if name != _SEARCH_TOOL_NAME:
                                    continue
                                try:
                                    args = json.loads(item.get("arguments") or "{}")
                                except (ValueError, TypeError):
                                    # Malformed tool-call JSON must not crash
                                    # the reader task mid-call.
                                    args = {}
                                # search_relevant, never search_permissive —
                                # CLAUDE.md's hard rule. It applies the
                                # relevance floor and always returns something
                                # speakable, so the agent is never left
                                # tool-calling with nothing to say.
                                answer = await search_relevant(str(args.get("query") or ""))
                                await oa.send(json.dumps({
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "function_call_output",
                                        "call_id": item.get("call_id"),
                                        "output": answer,
                                    },
                                }))
                                await oa.send(json.dumps({"type": "response.create"}))
                        elif (etype ==
                              "conversation.item.input_audio_transcription.completed"):
                            # What the lead said, transcribed. Only present on a
                            # two-way call — one-way never forwards their audio.
                            said = (event.get("transcript") or "").strip()
                            if said:
                                turns.append(TranscriptTurn(role="lead", text=said))
                        elif etype == "input_audio_buffer.speech_started":
                            # Barge-in. Honour the greeting grace window first
                            # (interrupt() returns False inside it), then drop
                            # the queued audio AND tell the model how much of its
                            # last message the lead actually heard — without the
                            # truncate the model believes it said everything it
                            # generated and every later turn builds on words the
                            # lead never heard. This is the sibling's unresolved
                            # bug (its ARCHITECTURE.md:268-276); we must not
                            # inherit it.
                            if await call.interrupt():
                                # Only cancel a response that is actually in
                                # progress — cancelling a finished one is
                                # rejected as response_cancel_not_active (Bug C).
                                if response_active:
                                    await oa.send(json.dumps(
                                        {"type": "response.cancel"}))
                                    response_active = False
                                # Only truncate a real assistant item. Before the
                                # model has produced one, last_item_id is None,
                                # and a null item_id is rejected outright (Bug A).
                                if last_item_id is not None:
                                    await oa.send(json.dumps({
                                        "type": "conversation.item.truncate",
                                        "item_id": last_item_id,
                                        "content_index": 0,
                                        "audio_end_ms": call.played_ms(),
                                    }))
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
