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

from app.config import (
    RAG_VOICE_DEADLINE_S,
    SARVAM_API_KEY,
    SARVAM_REPLY_MAX_WAIT_S,
    SARVAM_REPLY_SETTLE_S,
    SARVAM_STT_LANGUAGE,
    SARVAM_TWOWAY_ENABLED,
)
from app.db.models import TranscriptTurn
from app.rag.search import NO_MATERIAL_NOTE, search_relevant
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

# The lead's goodbye is a call-control signal, not a question for the model.
# Relying on the end_call tool alone leaves the line open when the model says a
# polite farewell but omits the tool (observed with the live Sarvam model).
_GOODBYE_RE = re.compile(
    r"(?:^|\s)(?:bye|goodbye|good\s+bye|bye\s+bye|alvida)(?:[.!?,\s]|$)",
    re.IGNORECASE,
)
_GOODBYE_PHRASES = (
    "that's all",
    "thats all",
    "no more questions",
    "i have no questions",
    "ఇక అంతే",
    "బై",
)

# Spoken when the model calls end_call with no farewell text at all — not
# suppressed (see _reply's ends_call handling), just never generated. Per
# CLAUDE.md, measured against the live API: told to say goodbye AND call
# end_call in the same turn, the model sometimes does the tool with no words
# attached (0/3 in one measurement). Ending in total silence either way is
# what CONFIRMED as the "the agent stopped answering" symptom on 2026-08-07 —
# a plain canned farewell is strictly better than nothing.
_FALLBACK_FAREWELL = "ధన్యవాదాలు, శుభదినం."


def _is_lead_goodbye(text: str) -> bool:
    """Whether a transcript is an explicit request to finish the call.

    Keep this deliberately narrow: thanks, okay, and other ordinary turn
    closers must still receive an answer. The LLM remains responsible for
    nuanced decisions; this is the reliable fast path for an unambiguous
    goodbye.
    """
    normalized = " ".join(text.strip().split()).casefold()
    return bool(_GOODBYE_RE.search(normalized)) or any(
        phrase in normalized for phrase in _GOODBYE_PHRASES
    )

# How long to wait for the lead to finish before answering.
#
# Not politeness — this is what stops a burst of utterances starving the agent
# completely. Every new transcript cancels the answer still being generated for
# the previous one, which is correct for a genuine barge-in and wrong for a
# lead who simply said two things in a row. On a real call the lead said
# "హలో", "ఓకే ఓకే", "హలో", "ఓకే"; the logs show SIX completions, all 200, and
# the transcript shows ONE agent turn. The agent answered every time and every
# answer was thrown away. The lead heard silence and hung up at 28 seconds.
#
# Sarvam's STT emits an utterance per pause, so this is the normal shape of
# speech, not an edge case. Waiting briefly collapses a burst into one answer
# that has heard all of it — and costs one completion instead of six.
#
# Originally a single fixed sleep (SARVAM_REPLY_MAX_WAIT_S, née
# _REPLY_DEBOUNCE_S) paid on EVERY turn regardless of whether Sarvam had
# already told us the lead stopped. sarvam_stt.py decodes Sarvam's own
# END_SPEECH signal — see _Conversation.on_lead_stopped below — so
# _reply_when_they_stop now waits only SARVAM_REPLY_SETTLE_S (to absorb
# END_SPEECH/transcript arriving in either order for the same utterance) and
# then answers immediately once END_SPEECH has actually landed.
# SARVAM_REPLY_MAX_WAIT_S survives as a CEILING, not a floor: if END_SPEECH
# never arrives for some reason, this is the fallback wait, unchanged from
# before — a turn can never be slower than it used to be.
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
        # True only while a conversation_llm.turn() HTTP call is actually in
        # flight for the current reply. Read by _cancel_reply(): a fragment
        # arriving mid-call used to cancel-and-restart unconditionally, and on
        # a lead who speaks in several quick bursts that restart kept winning
        # the race against the model ever finishing a single round — CONFIRMED
        # live on 2026-08-07 (two consecutive turns logged 0 rounds, nothing
        # spoken, on an ordinary mid-conversation question, not an ending).
        # Scoped to just the LLM call, not the whole reply, so cancellation
        # during a tool lookup keeps working exactly as before (see
        # _discard_incomplete_tool_turn) — that path is already tested and
        # intentional.
        self._reply_committed = False
        # Set when a fragment arrives while _reply_committed is True, so the
        # cancel that would normally have restarted the reply is skipped and
        # the fragment's text lands in history too late for the in-flight
        # round to see it. Checked once that round finishes so the fragment
        # still gets its own follow-up reply, rather than silently reading as
        # "already answered" once it is sitting behind the assistant's turn.
        self._missed_while_committed = False
        # Wall-clock of the last thing either side did. The silence watchdog
        # measures against this rather than against call start, so a lead who
        # is mid-question is never hung up on.
        self._loop = asyncio.get_event_loop()
        self.last_activity = self._loop.time()
        # Index into self.turns / self.history of the agent turn currently
        # being spoken, so a barge-in can rewrite exactly that one.
        self._agent_turn_index: int | None = None
        # Sarvam's own view of whether the lead is mid-utterance, from its
        # START_SPEECH / END_SPEECH VAD signals. Read by _reply_when_they_stop
        # so a reply need not blind-wait the full ceiling once Sarvam has
        # already said the lead stopped.
        self._lead_speaking = False
        # Per-turn latency accounting — see _log_turn_latency. Measured from
        # the first thing the lead said that nobody had answered yet (a burst
        # is ONE turn, so later fragments deliberately do not reset it) to the
        # first audio frame they hear back. Without this, "the agent replies
        # late" is unattributable: the gate, the model, the course lookup and
        # the synthesiser are four very different fixes and the logs could not
        # tell them apart.
        self._turn_opened_at: float | None = None
        self._turn_gate_done: float | None = None
        self._turn_say_started: float | None = None
        self._turn_llm_s = 0.0
        self._turn_tool_s = 0.0
        self._turn_llm_rounds = 0

    # ── measuring ────────────────────────────────────────────────────────────

    def _log_turn_latency(self, *, spoke: bool) -> None:
        """One INFO line per turn: where the lead's wait actually went.

        Emitted when the first audio frame of the answer reaches Plivo, which
        is the moment the lead stops waiting — not when the model returned, and
        not when the whole answer finished sending.

        Also emitted with spoke=False when a turn ends without saying anything
        (a cancelled reply, an end_call, or a turn that failed and was
        swallowed). That case is the "the agent never answered me" symptom, and
        it is worth exactly as much in the log as a slow one.
        """
        opened = self._turn_opened_at
        if opened is None:
            # The opening disclosure, or a turn already accounted for. Nobody
            # was waiting on an answer, so there is no latency to attribute.
            return
        now = self._loop.time()
        gate = (self._turn_gate_done - opened) if self._turn_gate_done else 0.0
        speak = (now - self._turn_say_started) if self._turn_say_started else 0.0
        logger.info(
            f"[sarvam] lead={self.lead_id} turn latency: "
            f"gate={gate:.2f}s llm={self._turn_llm_s:.2f}s"
            f"({self._turn_llm_rounds} round"
            f"{'' if self._turn_llm_rounds == 1 else 's'}) "
            f"tools={self._turn_tool_s:.2f}s tts={speak:.2f}s "
            f"total={now - opened:.2f}s"
            f"{'' if spoke else ' NOTHING SPOKEN'}"
        )
        self._reset_turn_metrics()

    def _reset_turn_metrics(self) -> None:
        self._turn_opened_at = None
        self._turn_gate_done = None
        self._turn_say_started = None
        self._turn_llm_s = 0.0
        self._turn_tool_s = 0.0
        self._turn_llm_rounds = 0

    def _log_playback_gaps(self, gap_total_ms: float, gap_max_ms: float,
                           sentences: int) -> None:
        """One INFO line per spoken turn: how much dead air, if any, the lead
        heard IN THE MIDDLE of this response.

        self.call.play_end is the wall-clock time Plivo's queued audio will
        finish playing, already maintained by PlivoCall.play() for every
        backend. If a later frame arrives after play_end has passed, the
        queue had already drained — dead air between whatever played last
        and this new chunk. say() accumulates that across the whole
        response and logs it here regardless of how the response ended, as
        long as at least one frame was actually played.

        A turn with zero gaps still logs (total=0.0ms) — the point is a
        dataset across many real calls, the same reasoning
        app/telephony/sarvam_stt.py's _log_audio_health uses for the
        inbound side. See
        docs/superpowers/specs/2026-08-08-playback-gap-diagnostics-design.md.
        """
        logger.info(
            f"[sarvam] lead={self.lead_id} playback gaps: "
            f"total={gap_total_ms:.1f}ms max={gap_max_ms:.1f}ms "
            f"sentences={sentences}"
        )

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

        self._turn_say_started = self._loop.time()
        spoke_a_frame = False
        # Dead air detected IN THE MIDDLE of this response — see
        # _log_playback_gaps for what these mean and why they're measured
        # this way.
        gap_total_ms = 0.0
        gap_max_ms = 0.0
        sentences_started = 0

        async def reset_tts_after_cancel() -> None:
            # The production SarvamTTS exposes this hook. Keep the bridge
            # tolerant of the tiny test/tool TTS adapters that only implement
            # speak().
            reset = getattr(tts, "reset_after_cancel", None)
            if reset is not None:
                await reset()

        try:
            for sentence in sentences:
                if self.generation != generation or self.call.stop.is_set():
                    await reset_tts_after_cancel()
                    return
                sentences_started += 1
                spoken_ms = 0
                async for payload, duration_ms in tts.speak(sentence):
                    if self.generation != generation or self.call.stop.is_set():
                        await reset_tts_after_cancel()
                        return
                    if spoke_a_frame:
                        # Only meaningful once THIS response has already
                        # started playing — the delay before the very first
                        # frame is normal startup latency, already captured
                        # by _log_turn_latency's tts= figure, not a gap.
                        now = self._loop.time()
                        if now > self.call.play_end:
                            gap_ms = (now - self.call.play_end) * 1000
                            gap_total_ms += gap_ms
                            gap_max_ms = max(gap_max_ms, gap_ms)
                    await self.call.play(payload)
                    if not spoke_a_frame:
                        # The lead's wait ends HERE, at the first frame — not
                        # when the whole answer has been sent.
                        spoke_a_frame = True
                        self._log_turn_latency(spoke=True)
                    spoken_ms += duration_ms
                self.ledger.add(sentence, spoken_ms)
                self.note_activity()
        except asyncio.CancelledError:
            # The cancellation can land while call.play() is awaiting Plivo,
            # after speak() already yielded a frame. Reset explicitly because
            # the async generator may not receive that cancellation itself.
            await reset_tts_after_cancel()
            raise
        finally:
            # Runs whether the response finished, was interrupted, or was
            # cancelled — as long as something was actually played, there is
            # something to report.
            if spoke_a_frame:
                self._log_playback_gaps(gap_total_ms, gap_max_ms, sentences_started)

    async def deliver_opening(self, tts: sarvam_tts.SarvamTTS,
                              opening: str) -> None:
        """Speak the opening, and keep it barge-in protected until it has been
        HEARD rather than merely sent.

        PlivoCall's protect_opening exists so a lead cannot talk over the AI
        disclosure. Lifting it when say() returns defeated that: say() returns
        once the audio is handed to Plivo, and Sarvam's TTS delivers far faster
        than real time, so the lead was typically still seconds into the
        disclosure. A "hello" then counted as a valid barge-in, the ledger
        truthfully rewrote the turn to the only sentence that had played — the
        bare name greeting — and the disclosure vanished from the call and from
        the record. Observed live as a [compliance] ERROR.

        play_end is when the queued audio actually runs out, the same clock the
        one-way watchdog drains against.
        """
        await self.say(tts, opening)
        while not self.call.stop.is_set():
            remaining = self.call.play_end - self._loop.time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(remaining, _SILENCE_POLL_S))
        self.call.mark_opening_delivered()

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
        """The lead started talking — over the agent, or into a silence."""
        # Set before the interrupt check below: the lead IS speaking either
        # way, whether or not there was anything to interrupt. This is what
        # _reply_when_they_stop polls, so a reply gated on it must not depend
        # on whether the agent happened to be talking at the same moment.
        self._lead_speaking = True
        if not await self.call.interrupt():
            # Inside the opening-disclosure guard or the greeting grace window.
            # PlivoCall said no, and it owns that decision.
            return
        self.generation += 1
        await self._cancel_reply()
        self.truncate_to_what_was_heard()

    def on_lead_stopped(self, tts: sarvam_tts.SarvamTTS) -> None:
        """Sarvam's END_SPEECH signal: the lead's current utterance ended.

        Two jobs.

        RELEASE: lets a reply already waiting in _reply_when_they_stop skip the
        rest of its ceiling instead of always paying SARVAM_REPLY_MAX_WAIT_S in
        full.

        REPAIR: on_barge_in() cancels the in-flight reply the instant the lead
        makes ANY sound — right when that sound turns into words, wrong when it
        doesn't. Sarvam drops empty transcripts (a cough, a line pop, a
        syllable under the transcription threshold), so START_SPEECH can fire
        with no transcript ever following it. Without this, the question the
        lead asked BEFORE that sound stays cancelled and nothing re-asks it —
        the lead sits in silence until the watchdog hangs up on them 20s later,
        which looks identical to "the agent never replied". END_SPEECH is the
        right place to notice something is still unanswered, because it means
        the interrupting sound is over and the line is genuinely quiet again.
        """
        self._lead_speaking = False
        if self.reply_task is not None and not self.reply_task.done():
            return  # already armed — the wait it's polling will now unblock
        if self._awaiting_answer():
            self.reply_task = asyncio.create_task(self._reply_when_they_stop(tts))

    def _awaiting_answer(self) -> bool:
        """Whether the lead has said something the agent has not yet answered.

        Walks history backward to the first thing that settles it: a user
        message means yes; an assistant message WITH WORDS means no. Tool
        bookkeeping — an assistant message carrying tool_calls with no content,
        or a tool result — is skipped on purpose: a lookup cancelled mid-flight
        leaves exactly that sitting on top of a question that is still
        unanswered.
        """
        for message in reversed(self.history):
            role = message.get("role")
            if role == "user":
                return True
            if role == "assistant" and message.get("content"):
                return False
        return False

    async def _cancel_reply(self, *, force: bool = False) -> None:
        """Cancel the in-flight reply, unless its model call cannot be undone.

        *force* skips that protection — used only where the call itself is
        ending (an explicit goodbye, or teardown), so there is no reason to
        let a stale model call keep running in the background.
        """
        task = self.reply_task
        if task is None or task.done():
            self.reply_task = None
            return
        if self._reply_committed and not force:
            # The model's HTTP round-trip is already in flight; see
            # _reply_committed's docstring. Let this round finish and speak —
            # a later fragment still gets answered via _missed_while_committed
            # rather than being lost to a restart that never lands.
            return
        self.reply_task = None
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        self._discard_incomplete_tool_turn()

    def _discard_incomplete_tool_turn(self) -> None:
        """Remove a tool-call turn interrupted by barge-in/cancellation.

        OpenAI requires every assistant ``tool_calls`` message to be followed
        by a tool result for each call ID. If a lead interrupts while RAG is in
        flight, cancellation used to leave that assistant message in history
        without its result; the next completion then returned HTTP 400 and the
        lead heard silence on every subsequent question.
        """
        for index in range(len(self.history) - 1, -1, -1):
            message = self.history[index]
            if message.get("role") != "assistant" or not message.get("tool_calls"):
                continue
            expected = {
                str(call.get("id") or "")
                for call in message["tool_calls"]
            }
            result_ids: set[str] = set()
            end = index + 1
            while end < len(self.history) and self.history[end].get("role") == "tool":
                result_ids.add(str(self.history[end].get("tool_call_id") or ""))
                end += 1
            if result_ids != expected:
                del self.history[index:end]
            return

    async def on_lead_said(self, tts: sarvam_tts.SarvamTTS, text: str) -> None:
        """A completed lead utterance: record it and answer it."""
        self.turns.append(TranscriptTurn(role="lead", text=text))
        self.history.append({"role": "user", "content": text})
        if _is_lead_goodbye(text):
            # Do not ask the model to decide whether an explicit goodbye ends
            # the call. Cancelling first prevents a reply already waiting on
            # the speech-settle gate from speaking after the lead said bye.
            # force=True: the call is ending regardless, so there is nothing
            # to protect an in-flight model call for.
            await self._cancel_reply(force=True)
            self.call.note_exit("sarvam_lead_goodbye")
            self.call.stop.set()
            return
        if self._turn_opened_at is None:
            # First unanswered thing the lead has said. A burst is ONE turn
            # from their point of view — they have been waiting since this
            # fragment, not since the last one — so later fragments must not
            # restart the clock.
            self._reset_turn_metrics()
            self._turn_opened_at = self._loop.time()
        # One reply at a time, and not until the lead has actually stopped.
        # Cancelling here is what starved the agent on a live call (see
        # SARVAM_REPLY_MAX_WAIT_S's comment above): the cancel is cheap only
        # because the replacement waits before calling the model, so a burst
        # collapses into one answer rather than N answers of which N-1 are
        # discarded.
        await self._cancel_reply()
        if self.reply_task is not None and not self.reply_task.done():
            # _cancel_reply() left it running — its model call is already
            # committed. This fragment's text is already in history (above)
            # but arrived too late for that round to see it; flag it so
            # _reply_when_they_stop answers it in a follow-up pass instead of
            # a second task racing the first.
            self._missed_while_committed = True
            return
        self.reply_task = asyncio.create_task(self._reply_when_they_stop(tts))

    # ── answering ────────────────────────────────────────────────────────────

    async def _reply_when_they_stop(self, tts: sarvam_tts.SarvamTTS) -> None:
        """Wait for the lead to actually be done, then answer everything said
        so far.

        Signal-driven, not a blind sleep: Sarvam's END_SPEECH tells us when
        the lead stopped (on_lead_stopped sets self._lead_speaking = False),
        but the transcript that triggered this coroutine and that signal can
        arrive in EITHER order for the same utterance, so this cannot simply
        check the flag once. SARVAM_REPLY_SETTLE_S is a short, always-paid
        pause that absorbs an END_SPEECH arriving just after the transcript;
        if it arrived first, self._lead_speaking is already False and the
        loop below never executes. SARVAM_REPLY_MAX_WAIT_S is a ceiling for
        the case END_SPEECH never arrives at all — the previous fixed wait,
        kept as a fallback rather than a floor, so a turn is never slower
        than it was before this became signal-driven.

        Cancelled by the next transcript or barge-in if one arrives inside the
        window, and cancelled here is free: no completion has been requested
        yet. That is the whole point — the model is called once, after the
        lead has finished, instead of once per fragment with every answer but
        the last thrown away.
        """
        deadline = self._loop.time() + SARVAM_REPLY_MAX_WAIT_S
        await asyncio.sleep(SARVAM_REPLY_SETTLE_S)
        while (self._lead_speaking and self._loop.time() < deadline
               and not self.call.stop.is_set()):
            await asyncio.sleep(_SILENCE_POLL_S)
        self._turn_gate_done = self._loop.time()
        await self._reply(tts)
        if self._missed_while_committed and not self.call.stop.is_set():
            # A fragment arrived while the round above was already committed
            # to the model (see _reply_committed) and could not be folded in.
            # It is in history now that the round is done, so answer it —
            # this is what keeps a multi-part question from losing its last
            # part to "already answered" once it lands behind an assistant
            # turn that never actually saw it.
            self._missed_while_committed = False
            if self._turn_opened_at is None:
                self._turn_opened_at = self._loop.time()
            self.reply_task = asyncio.create_task(self._reply_when_they_stop(tts))

    async def _reply(self, tts: sarvam_tts.SarvamTTS) -> None:
        try:
            for _round in range(_MAX_TOOL_ROUNDS):
                _llm_started = self._loop.time()
                self._reply_committed = True
                try:
                    reply = await conversation_llm.turn(self.history)
                finally:
                    self._reply_committed = False
                self._turn_llm_s += self._loop.time() - _llm_started
                self._turn_llm_rounds += 1
                if not reply.tool_calls:
                    if reply.text:
                        await self.say(tts, reply.text)
                    return
                # Text that arrives WITH a tool call is usually filler — "let
                # me look that up for you". On a real call the agent
                # announced its own lookup twice and then reported failure,
                # which is three synthesised turns to deliver nothing. A
                # search takes about a second; silence is shorter than saying
                # so. The answer that follows is what the lead wants.
                #
                # end_call is the one exception: the tool's own description
                # tells the model to send its farewell "together with"
                # end_call, and there is no next round for that farewell to
                # arrive in — this IS the only chance to say it. Treating it
                # as filler was confirmed live on 2026-08-07: every Sarvam
                # call ended by the model (as opposed to the silence
                # watchdog) hung up in total silence, right after the lead's
                # last word, which reads exactly like "the agent stopped
                # answering."
                ends_call = any(tc.name == conversation_llm.END_CALL_TOOL_NAME
                                for tc in reply.tool_calls)
                if ends_call:
                    # No next round for this text to arrive in, so speak
                    # whatever the model wrote — or, if it wrote nothing at
                    # all, a canned farewell rather than hanging up in
                    # silence. See _FALLBACK_FAREWELL.
                    await self.say(tts, reply.text or _FALLBACK_FAREWELL)
                elif reply.text:
                    logger.debug(f"[sarvam] lead={self.lead_id} not speaking "
                                 f"tool-call filler: {reply.text[:60]!r}")
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
        finally:
            # Still set means say() never reached its first frame: the turn
            # ended without the lead hearing anything. Log it rather than let
            # it silently roll into the next turn's total — "the agent never
            # answered me" is the symptom worth the loudest evidence.
            self._log_turn_latency(spoke=False)

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
                #
                # Bounded here, not inside search_relevant(): that function has
                # no deadline of its own (an embed retry alone is ~16s), and it
                # is also POST /rag/search, the ElevenLabs agent's live tool —
                # a path already in production that must not change. The
                # silence watchdog will not save the lead either; an in-flight
                # reply counts as activity. On a timeout, the model gets the
                # SAME note a genuine miss returns: the system prompt already
                # knows how to say that in the lead's language without
                # inventing a fact, whereas a bespoke "the search timed out"
                # string would be read out to the lead verbatim.
                query = str(tool_call.arguments.get("query") or "")
                _tool_started = self._loop.time()
                try:
                    answer = await asyncio.wait_for(
                        search_relevant(query), timeout=RAG_VOICE_DEADLINE_S)
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[sarvam] lead={self.lead_id} course lookup for "
                        f"{query!r} did not finish inside "
                        f"{RAG_VOICE_DEADLINE_S:.1f}s — telling the model "
                        "there is no material rather than holding the lead "
                        "on a silent line."
                    )
                    answer = NO_MATERIAL_NOTE
                except Exception as exc:  # noqa: BLE001 - degrade one lookup
                    # The live endpoint has the same contract, but this path
                    # calls search_relevant() directly to avoid an HTTP hop.
                    # Embedding, database, or translation failures must not
                    # escape into _reply: that would cancel the entire turn
                    # and leave the lead hearing silence after asking a
                    # question. Give the model a normal no-material result so
                    # its prompt can produce a short spoken fallback.
                    logger.error(
                        f"[sarvam] lead={self.lead_id} course lookup for "
                        f"{query!r} failed: {type(exc).__name__}: {exc}"
                    )
                    answer = NO_MATERIAL_NOTE
                finally:
                    self._turn_tool_s += self._loop.time() - _tool_started
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
            sarvam_stt.SarvamSTT(lead_id=lead_id) as stt:

        async def to_sarvam(payload_b64: str) -> None:
            await stt.send_audio(payload_b64)

        async def converse() -> None:
            try:
                # The AI placed this call, so it speaks first — the disclosure
                # cannot wait for the lead to say something, and cannot be
                # talked over until it has actually been heard.
                await conversation.deliver_opening(tts, opening)

                async for event in stt.events():
                    if call.stop.is_set():
                        break
                    conversation.note_activity()
                    if event.kind == "speech_started":
                        await conversation.on_barge_in()
                    elif event.kind == "speech_ended":
                        conversation.on_lead_stopped(tts)
                    elif event.kind == "transcript":
                        await conversation.on_lead_said(tts, event.text)
            except Exception as exc:  # noqa: BLE001 - the call must still be graded
                call.note_exit(f"error:{type(exc).__name__}")
                logger.error(f"[sarvam] lead={lead_id} conversation failed: "
                             f"{type(exc).__name__}: {exc}")
            finally:
                # force=True: the call is over either way, so there is no
                # reason to let an in-flight model call keep running.
                await conversation._cancel_reply(force=True)
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
        script = (variables.get("script") or "").strip()
        if not one_way and not script:
            # A two-way campaign needs no script, and call_routes only forwards
            # one for oneway — so this is the NORMAL two-way path, not an edge
            # case. Without a fallback the render is empty and the opening
            # collapses to the bare name greeting, which is exactly what a real
            # lead heard on 2026-07-27: a call with no AI disclosure at all.
            script = sarvam_prompts.DEFAULT_TWOWAY_SCRIPT

        body = await sarvam_llm.render(
            script, language_style=variables.get("language_style"),
        )
        if not body.strip():
            # Checked on the BODY, not on the composed line. compose_spoken()
            # prepends a name greeting, so a lead called "Mouli" turned an empty
            # render into a non-empty "నమస్తే Mouli గారు." and sailed past this
            # guard into a non-compliant call.
            logger.error(
                f"[sarvam] lead={lead_id} nothing to say — the campaign script "
                "rendered empty. A one-way campaign cannot be created without "
                "a script, so check how this one was made."
            )
            return outcome

        # Last line of defence before a real person hears this. render() already
        # repairs a dropped disclosure, but the greeting is spliced in AFTER
        # that, and only the composed text is what the lead actually hears.
        spoken = sarvam_llm.ensure_disclosure(sarvam_prompts.compose_spoken(
            body, lead_name=variables.get("lead_name"),
        ))

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
