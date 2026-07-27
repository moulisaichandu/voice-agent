"""telephony/sarvam_bridge.py — Bridge a live Plivo call to Sarvam.

The Telugu backend. Exists because ElevenLabs Agents does not offer Telugu at
all (see app/languages.py's ELEVENLABS_AGENT_LANGUAGES) and because the backend
that filled that gap first, OpenAI Realtime, has no Telugu-native voice — every
Realtime voice is English-first, and no code change fixes that. Sarvam's are
recorded by Indian voice artists.

Deliberately mirrors app/telephony/bridge.py's and openai_bridge.py's signature
and `outcome` contract exactly, so app/telephony/call_routes.py picks a backend
rather than branching into three different call shapes, and everything
downstream — call recording, Sheets write-back, slot release, retry accounting
— never learns which one ran.

HOW THIS BACKEND DIFFERS. The other two are speech-to-speech: one socket, and
the platform does the turn-taking. Sarvam has no such API — STT, LLM and TTS
are three separate services. A one-way call therefore reads:

    script (English, from the campaign)
      -> sarvam_llm.render()   the Telugu to speak, cached per campaign
      -> sarvam_tts.speak()    mu-law 8 kHz frames
      -> PlivoCall.play()      straight down the wire, no transcoding

The outbound leg stays a pure base64 passthrough, exactly like the other two.
The INBOUND leg cannot: Sarvam's STT does not accept mu-law, so Milestone B's
two-way path has to decode it. One-way never listens, so nothing here does.

All the Plivo-side behaviour — playing audio, the one-way watchdog that stops a
message-only call billing 300 seconds of silence — lives in
app/telephony/plivo_stream.py and is shared with both other bridges. This
module owns only the ordering of Sarvam's three services.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re

from fastapi import WebSocket

from app.config import SARVAM_API_KEY, SARVAM_STT_LANGUAGE, SARVAM_TWOWAY_ENABLED
from app.db.models import TranscriptTurn
from app.rag.search import search_relevant
from app.telephony import (
    conversation_llm,
    plivo_stream,
    sarvam_llm,
    sarvam_prompts,
    sarvam_stt,
    sarvam_tts,
)

logger = logging.getLogger(__name__)

# A tool call that calls a tool that calls a tool... Bounded so a model that
# gets stuck looking things up cannot hold a lead on a silent line.
_MAX_TOOL_ROUNDS = 3

# Sentence enders for Telugu and English. Splitting the reply lets the ledger
# record what the lead heard at sentence granularity, which is the unit a
# conversation can actually be truncated at.
_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")

# How long a two-way call may sit with NEITHER side speaking before it ends
# itself.
#
# This exists because ending the call cannot be left to the model. Measured
# against the live API on 2026-07-27: told to say goodbye and call end_call,
# it said the goodbye and did not call the tool — 0 times out of 3, even when
# instructed to do it in that specific turn. The best prompt variant managed
# 2/3, and only by weakening the OUTPUT FORMAT rule that stops the synthesiser
# reading preambles aloud to the lead, which is not a trade worth making.
#
# So the call ends on silence instead, the same way PlivoCall.oneway_watchdog
# ends a message-only call. Without it a lead who says goodbye leaves the line
# open to CALL_MAX_DURATION_S: five minutes of billed airtime and silence.
#
# Deliberately driven from this bridge rather than PlivoCall.run(), because the
# ElevenLabs two-way path is in production and must not acquire a new way to
# hang up on someone.
TWOWAY_MAX_SILENT_S = 20.0
_SILENCE_POLL_S = 0.25


class SpokenLedger:
    """What the lead actually HEARD of the agent's current turn.

    The piece of two-way this backend cannot borrow from the other one. On
    OpenAI Realtime, a barge-in is answered with `conversation.item.truncate`
    and the platform repairs its own history. Here the history is ours, so if
    the lead interrupts two sentences into a five-sentence answer, WE have to
    forget the other three — otherwise every later turn is built on words
    nobody heard, and the agent starts referring back to things it never said.

    The previous plan for the OpenAI backend flagged exactly this as a bug
    inherited from the sibling project and worth not repeating.

    Pure: sentences in with their audio durations, text out for a given
    playback position. No sockets, no clock.
    """

    def __init__(self) -> None:
        # (text, cumulative_ms_at_end_of_this_text)
        self._entries: list[tuple[str, int]] = []
        self._total_ms = 0

    def add(self, text: str, duration_ms: int) -> None:
        self._total_ms += duration_ms
        self._entries.append((text, self._total_ms))

    def reset(self) -> None:
        """Start a new agent turn. Playback is measured from each turn's own
        start (PlivoCall.mark_response_boundary does the same on the audio
        side), so entries from the previous turn would be mis-attributed."""
        self._entries.clear()
        self._total_ms = 0

    def full_text(self) -> str:
        """Everything that was handed to the synthesiser. For logs and
        diagnostics — never for the conversation history, which must only ever
        contain what was heard."""
        return " ".join(text for text, _ in self._entries)

    def heard_within(self, played_ms: int) -> str:
        """The part of this turn that finished playing by *played_ms*.

        A sentence only PARTLY played is dropped rather than claimed. That
        direction is deliberate: keeping it risks the agent referring back to
        something the lead never received, while dropping it risks the agent
        repeating itself. Redundancy is a much cheaper failure than incoherence
        on a phone call.
        """
        return " ".join(text for text, ends_at in self._entries
                        if ends_at <= played_ms)


def _split_sentences(text: str) -> list[str]:
    """*text* as speakable sentences.

    Granularity matters here: the ledger can only forget what it recorded
    separately, so feeding TTS one sentence at a time is what makes a barge-in
    truncatable to something meaningful rather than all-or-nothing.
    """
    return [part.strip() for part in _SENTENCE_END.split(text.strip()) if part.strip()]


class _Conversation:
    """One two-way call's state: history, playback ledger, in-flight reply.

    Exists because the turn-taking on this backend is OURS. The other two
    backends are speech-to-speech and the platform owns the conversation; here
    STT, LLM and TTS are three separate services and something has to hold the
    thread between them.
    """

    def __init__(self, call: plivo_stream.PlivoCall, *, lead_id: str,
                 system_prompt: str, turns: list[TranscriptTurn]) -> None:
        self.call = call
        self.lead_id = lead_id
        self.turns = turns
        self.history: list[dict] = [{"role": "system", "content": system_prompt}]
        self.ledger = SpokenLedger()
        # Bumped on every barge-in. Audio synthesised for an older generation is
        # dropped rather than played, so a cancelled answer cannot leak a few
        # more sentences at the lead after they interrupted.
        self.generation = 0
        self.reply_task: asyncio.Task | None = None
        # Wall-clock of the last thing either side did. The silence watchdog
        # measures against this rather than against call start, so a lead who
        # is mid-question is never hung up on.
        self._loop = asyncio.get_event_loop()
        self.last_activity = self._loop.time()
        # Index into self.turns / self.history of the agent turn currently
        # being spoken, so a barge-in can rewrite exactly that one.
        self._agent_turn_index: int | None = None

    # ── speaking ─────────────────────────────────────────────────────────────

    async def say(self, tts: sarvam_tts.SarvamTTS, text: str) -> None:
        """Speak *text*, recording what actually reached the lead.

        Each sentence is synthesised, played, and logged to the ledger with its
        audio duration. If the generation changes mid-way the lead has
        interrupted, so the rest is abandoned unplayed.
        """
        sentences = _split_sentences(text)
        if not sentences:
            return

        generation = self.generation
        self.ledger.reset()
        # Restart the playback clock: played_ms() is relative to the current
        # response, and a barge-in truncates against it.
        self.call.mark_response_boundary()

        self._agent_turn_index = len(self.turns)
        self.turns.append(TranscriptTurn(role="agent", text=text))
        self.history.append({"role": "assistant", "content": text})

        for sentence in sentences:
            if self.generation != generation or self.call.stop.is_set():
                return
            spoken_ms = 0
            async for payload, duration_ms in tts.speak(sentence):
                if self.generation != generation or self.call.stop.is_set():
                    return
                await self.call.play(payload)
                spoken_ms += duration_ms
            self.ledger.add(sentence, spoken_ms)
            self.note_activity()

    def note_activity(self) -> None:
        """Somebody spoke. Resets the silence watchdog."""
        self.last_activity = self._loop.time()

    async def watch_for_silence(self) -> None:
        """End the call once neither side has spoken for TWOWAY_MAX_SILENT_S.

        Not a duration cap — it measures SILENCE. A conversation that is still
        going resets the clock on every event, so this only fires on a line
        that has genuinely finished.

        A reply still being generated counts as activity: the lead is waiting
        for an answer, not sitting in a dead call.

        So does audio that has been SENT but not yet HEARD. Sarvam's TTS
        delivers far faster than real time — an eight-second answer arrives in
        about three — so the send loop finishing means nothing about whether
        the lead has stopped listening. PlivoCall.play_end is when the queued
        audio actually runs out, which is the same accounting the one-way
        watchdog drains against. Measuring from the send loop instead hung up
        mid-sentence on a call that was working perfectly.
        """
        while not self.call.stop.is_set():
            await asyncio.sleep(_SILENCE_POLL_S)
            if self.reply_task is not None and not self.reply_task.done():
                self.note_activity()
                continue
            quiet_since = max(self.last_activity, self.call.play_end)
            if self._loop.time() - quiet_since >= TWOWAY_MAX_SILENT_S:
                logger.info(
                    f"[sarvam] lead={self.lead_id} ending a conversation that "
                    f"has been silent for {TWOWAY_MAX_SILENT_S:.0f}s."
                )
                self.call.note_exit("twoway_silence")
                break
        self.call.stop.set()

    def truncate_to_what_was_heard(self) -> None:
        """Rewrite the agent's current turn down to the part that played.

        The whole point of the ledger. Without this the model is told it said
        things the lead never heard, and every later turn is built on that.
        """
        if self._agent_turn_index is None:
            return
        heard = self.ledger.heard_within(self.call.played_ms())
        index = self._agent_turn_index
        self._agent_turn_index = None

        if heard:
            self.turns[index] = TranscriptTurn(role="agent", text=heard)
            for message in reversed(self.history):
                if message.get("role") == "assistant":
                    message["content"] = heard
                    break
            return

        # Nothing played at all: as far as the conversation is concerned the
        # agent did not speak. Drop the turn rather than leave an empty one,
        # which would read as the agent having said nothing on purpose.
        if index < len(self.turns):
            del self.turns[index]
        for position in range(len(self.history) - 1, 0, -1):
            if self.history[position].get("role") == "assistant":
                del self.history[position]
                break

    # ── listening ────────────────────────────────────────────────────────────

    async def on_barge_in(self) -> None:
        """The lead started talking over the agent."""
        if not await self.call.interrupt():
            # Inside the opening-disclosure guard or the greeting grace window.
            # PlivoCall said no, and it owns that decision.
            return
        self.generation += 1
        await self._cancel_reply()
        self.truncate_to_what_was_heard()

    async def _cancel_reply(self) -> None:
        task, self.reply_task = self.reply_task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def on_lead_said(self, tts: sarvam_tts.SarvamTTS, text: str) -> None:
        """A completed lead utterance: record it and answer it."""
        self.turns.append(TranscriptTurn(role="lead", text=text))
        self.history.append({"role": "user", "content": text})
        # One reply at a time. A lead who says two things quickly should get
        # one answer to both, not two answers talking over each other.
        await self._cancel_reply()
        self.reply_task = asyncio.create_task(self._reply(tts))

    # ── answering ────────────────────────────────────────────────────────────

    async def _reply(self, tts: sarvam_tts.SarvamTTS) -> None:
        try:
            for _round in range(_MAX_TOOL_ROUNDS):
                reply = await conversation_llm.turn(self.history)
                if not reply.tool_calls:
                    if reply.text:
                        await self.say(tts, reply.text)
                    return
                if reply.text:
                    await self.say(tts, reply.text)
                if await self._run_tools(reply):
                    return  # end_call: nothing further to say
            logger.warning(
                f"[sarvam] lead={self.lead_id} the model kept calling tools for "
                f"{_MAX_TOOL_ROUNDS} rounds — giving up on this turn rather "
                "than holding the lead on a silent line."
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad turn must not end the call
            logger.error(f"[sarvam] lead={self.lead_id} turn failed: "
                         f"{type(exc).__name__}: {exc}")

    async def _run_tools(self, reply: conversation_llm.LLMReply) -> bool:
        """Run the model's tool calls. Returns True if the call should end.

        The assistant message carrying tool_calls has to go into the history
        before the tool results, or the model sees answers to questions it has
        no record of asking.
        """
        self.history.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": tc.call_id, "type": "function",
                 "function": {"name": tc.name,
                              "arguments": json.dumps(tc.arguments)}}
                for tc in reply.tool_calls
            ],
        })

        for tool_call in reply.tool_calls:
            if tool_call.name == conversation_llm.END_CALL_TOOL_NAME:
                # A conversation the agent ended cleanly, not a dropped call.
                # 'sarvam_end_call' is in plivo_stream._CLEAN_EXITS for this.
                self.call.note_exit("sarvam_end_call")
                self.call.stop.set()
                return True

            if tool_call.name == conversation_llm.SEARCH_TOOL_NAME:
                # search_relevant, never search_permissive — CLAUDE.md's hard
                # rule. It applies the relevance floor and always returns
                # something speakable, so the agent is never left with nothing
                # to say to a lead who just asked a question.
                answer = await search_relevant(str(tool_call.arguments.get("query") or ""))
            else:
                logger.warning(f"[sarvam] lead={self.lead_id} unknown tool "
                               f"{tool_call.name!r} — telling the model so.")
                answer = "That tool does not exist."

            self.history.append({
                "role": "tool",
                "tool_call_id": tool_call.call_id,
                "content": answer,
            })
        return False


async def _run_two_way(call: plivo_stream.PlivoCall, *, lead_id: str,
                       opening: str, system_prompt: str,
                       turns: list[TranscriptTurn]) -> None:
    """Hold a conversation until either side ends it."""
    conversation = _Conversation(call, lead_id=lead_id,
                                 system_prompt=system_prompt, turns=turns)

    async with sarvam_tts.SarvamTTS(language=SARVAM_STT_LANGUAGE) as tts, \
            sarvam_stt.SarvamSTT() as stt:

        async def to_sarvam(payload_b64: str) -> None:
            await stt.send_audio(payload_b64)

        async def converse() -> None:
            try:
                # The AI placed this call, so it speaks first — the disclosure
                # cannot wait for the lead to say something.
                await conversation.say(tts, opening)
                call.mark_opening_delivered()

                async for event in stt.events():
                    if call.stop.is_set():
                        break
                    conversation.note_activity()
                    if event.kind == "speech_started":
                        await conversation.on_barge_in()
                    elif event.kind == "transcript":
                        await conversation.on_lead_said(tts, event.text)
            except Exception as exc:  # noqa: BLE001 - the call must still be graded
                call.note_exit(f"error:{type(exc).__name__}")
                logger.error(f"[sarvam] lead={lead_id} conversation failed: "
                             f"{type(exc).__name__}: {exc}")
            finally:
                await conversation._cancel_reply()
                call.stop.set()

        await call.run(call.read_events(to_sarvam), converse(),
                       conversation.watch_for_silence())


async def bridge(plivo_ws: WebSocket, *, agent_id: str, lead_id: str,
                 dynamic_variables: dict | None = None,
                 language: str | None = None,
                 one_way: bool = False, outcome: dict | None = None) -> dict:
    """Bridge one answered call to Sarvam until either side ends it.

    Same contract as app/telephony/bridge.py's bridge(). *agent_id* is accepted
    and ignored — Sarvam has no agent objects — so call_routes can dispatch on
    backend without reshaping the call. *language* is accepted and ignored too:
    the ISO code it carries ('te') is not the BCP-47 code Sarvam wants, and
    which one to send is a property of the backend, not of the call, so it
    comes from SARVAM_STT_LANGUAGE.

    *outcome* is populated in place and returned, so a mid-call exception still
    hands the caller every turn collected before it. The ElevenLabs bridge's
    docstring explains why that matters: a conversation that ended badly used
    to be recorded as zero turns and a 'failed' lead that then burned a retry.
    """
    if outcome is None:
        outcome = {}
    outcome.update({"status": "failed", "turns": 0, "transcript": [],
                    "conversation_id": None})

    if not SARVAM_API_KEY:
        logger.error("[sarvam] SARVAM_API_KEY is not set — cannot bridge the call.")
        return outcome

    if not one_way and not SARVAM_TWOWAY_ENABLED:
        # The third and cheapest of three guards; app/admin/campaigns.py
        # refuses this at creation and app/telephony/preflight.py refuses it
        # again at dial time. It is still worth having: without it the call
        # connects, never speaks, never listens, bills the full
        # CALL_MAX_DURATION_S of silence, and is then recorded as a SUCCESSFUL
        # zero-turn call, because max_duration counts as a clean exit.
        logger.error(
            f"[sarvam] lead={lead_id} refusing a TWO-WAY call: "
            "SARVAM_TWOWAY_ENABLED is not set, so a human has not yet judged "
            "this backend on a live conversation. This campaign should have "
            "been refused at creation and again at preflight — check both."
        )
        return outcome

    variables = dynamic_variables or {}
    call = plivo_stream.PlivoCall(plivo_ws, lead_id=lead_id, one_way=one_way,
                                  # Two-way protects the opening so the lead
                                  # cannot talk over the AI disclosure before
                                  # it has been said.
                                  protect_opening=not one_way)
    turns: list[TranscriptTurn] = []

    try:
        # Rendered before the socket opens: this is the one step that can take
        # a noticeable moment on a cache miss, and holding a TTS connection
        # open through it buys nothing.
        body = await sarvam_llm.render(
            variables.get("script") or "",
            language_style=variables.get("language_style"),
        )
        spoken = sarvam_prompts.compose_spoken(
            body, lead_name=variables.get("lead_name"),
        )
        if not spoken.strip():
            logger.error(
                f"[sarvam] lead={lead_id} nothing to say — the campaign script "
                "rendered empty. A one-way campaign cannot be created without "
                "a script, so check how this one was made."
            )
            return outcome

        if not one_way:
            await _run_two_way(
                call, lead_id=lead_id, opening=spoken,
                system_prompt=sarvam_prompts.two_way_instructions(
                    variables.get("lead_name"),
                    variables.get("script"),
                    language_style=variables.get("language_style"),
                ),
                turns=turns,
            )
            return outcome

        async def to_sarvam(_payload_b64: str) -> None:
            """Never called: PlivoCall drops the lead's media on a one-way
            call. Present so the reader has the shape it expects."""
            return None

        async def speak() -> None:
            try:
                async with sarvam_tts.SarvamTTS(language=SARVAM_STT_LANGUAGE) as tts:
                    async for payload, _duration_ms in tts.speak(spoken):
                        if call.stop.is_set():
                            break
                        await call.play(payload)
            except sarvam_tts.SarvamNotConfigured as exc:
                call.note_exit("sarvam_unconfigured")
                logger.error(f"[sarvam] lead={lead_id} {exc}")
            except Exception as exc:  # noqa: BLE001 - the call must still be graded
                call.note_exit(f"error:{type(exc).__name__}")
                logger.error(f"[sarvam] lead={lead_id} synthesis failed: "
                             f"{type(exc).__name__}: {exc}")
            finally:
                # The transcript is evidence, not intent. Only record the turn
                # if at least one frame actually reached the lead: a call whose
                # synthesis was refused outright would otherwise certify words
                # nobody heard — to _spoken_disclosure_ok(), which reads this
                # to check the AI disclosure, and to the operator, who reads it
                # in the Sheet as a delivered message.
                if call.audio_seen:
                    turns.append(TranscriptTurn(role="agent", text=spoken))
                # The watchdog ends the call once this audio has drained; it
                # needs the tail, so `stop` is deliberately NOT set here.
                call.mark_opening_delivered()

        await call.run(call.read_events(to_sarvam), speak())
    except (sarvam_llm.SarvamNotConfigured, sarvam_llm.SarvamRenderFailed) as exc:
        # Deliberately not falling back to the English script: that would ring
        # a Telugu-speaking lead and read English at them.
        call.note_exit("sarvam_render_failed")
        logger.error(f"[sarvam] lead={lead_id} not dialling: {exc}")
    except Exception as exc:  # noqa: BLE001 - a bridge must never raise into _finalise_call
        call.note_exit(f"error:{type(exc).__name__}")
        logger.exception(f"[sarvam] bridge failed for lead {lead_id}: "
                         f"{type(exc).__name__}: {exc}")
    finally:
        # In a finally for the same reason both other bridges are: a call that
        # really happened must not be recorded as nothing.
        outcome["turns"] = len(turns)
        outcome["transcript"] = turns
        outcome["status"] = "done" if call.ran_its_course else "failed"
        logger.info(f"[sarvam] lead={lead_id} ended turns={len(turns)} "
                    f"exit={call.exit_reason}")
    return outcome
