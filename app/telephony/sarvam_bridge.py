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

from app import redis_client
from app.config import (
    ONEWAY_MAX_SILENT_S,
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
    lead_name,
    plivo_stream,
    sarvam_llm,
    sarvam_prompts,
    sarvam_stt,
    sarvam_tts,
    term_repair,
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

# How long the lead may hold the floor before watch_for_silence stops
# treating "still speaking" as activity. Generous — a long question is a
# normal thing to ask — but finite, because the flag is cleared by an
# END_SPEECH that can be lost, and a stuck flag must not disable the
# watchdog for the whole call.
_MAX_LEAD_SPEECH_S = 45.0

# How many times an end_call may be refused because the lead's own question
# is still unanswered. Finite: a model that only ever wants to end must
# eventually be allowed to, or a lead who asked something the agent cannot
# answer is held on a line that will not close. Two attempts is enough for
# the model to notice the tool result and answer instead.
_MAX_END_CALL_REFUSALS = 2

# How long the script render may take before the call is abandoned.
#
# render() runs AFTER Plivo has answered the leg but BEFORE PlivoCall.run()
# starts, so neither the one-way watchdog nor CALL_MAX_DURATION_S can see it —
# its only bound was sarvam_llm's own 60s HTTP timeout. A Sarvam latency spike
# with a cold render cache (a new campaign, an edited script, or a Redis flush,
# which CLAUDE.md documents as safe) therefore meant a real person answered the
# phone and heard up to a minute of complete silence, on every lead in the
# campaign. The credits breaker cannot help: it only trips on 'credit'.
#
# Bounded by the same figure the one-way watchdog uses for "this agent is not
# speaking" — past that the lead has already decided the line is dead, and
# hanging up beats billing more silence.
RENDER_DEADLINE_S = float(ONEWAY_MAX_SILENT_S)

# The lead's goodbye is a call-control signal, not a question for the model.
# Relying on the end_call tool alone leaves the line open when the model says a
# polite farewell but omits the tool (observed with the live Sarvam model).
# Boundaries are LOOKAROUNDS, not consumed characters: _is_lead_goodbye
# strips matches out and judges the remainder, and a consumed separator left
# the second half of "bye bye" unmatchable. Longest alternative first, so
# "bye bye" is one goodbye rather than one plus an unexplained word.
_GOODBYE_RE = re.compile(
    r"(?<![^\s])(?:good\s+bye|bye\s+bye|goodbye|bye|alvida)(?=[.!?,]|\s|$)",
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

# Telugu writes case endings onto the word, so "బై" (bye) is a SUBSTRING of
# మొబైల్ (mobile) and బైక్ (bike). Matching the phrase list with a bare `in`
# hung up on a lead saying "send it to my mobile number" — mid-sentence, agent
# silent, graded a clean exit (sarvam_lead_goodbye is in _CLEAN_EXITS), lead
# marked done and never called back. Phrases are matched as whole TOKENS
# instead: the same word-boundary discipline _GOODBYE_RE already applies to the
# Latin spellings, which is why the English path never had this bug.
_GOODBYE_STRIP = ".,!?;:'\"()[]{}…।॥"


def _goodbye_tokens(text: str) -> list[str]:
    """Whitespace tokens with edge punctuation removed, so "బై." still matches."""
    return [t for t in (w.strip(_GOODBYE_STRIP) for w in text.split()) if t]


_GOODBYE_PHRASE_TOKENS = tuple(
    _goodbye_tokens(phrase.casefold()) for phrase in _GOODBYE_PHRASES
)

# Pure agreement, and nothing else. An utterance made only of these is the
# lead saying "yes, go on" — never a request to end the call, whatever the
# model concludes. See _lead_only_agreed; a live call ended on "ఆ ఓకే".
_AFFIRMATION_ONLY = frozenset({
    "ఓకే", "సరే", "అవును", "ఊ", "ఆ", "ఉమ్", "హా", "అవునండి", "సరేనండి",
    "ok", "okay", "yes", "yeah", "yep", "hmm", "hm", "uh",
    "సార్", "గారు", "మేడమ్", "sir", "madam",
})

# Politeness that surrounds a goodbye without adding a request: "థ్యాంక్స్,
# బై" and "సరే, ఇక అంతే" are goodbyes, while the same phrases followed by
# anything substantive are not (see _is_lead_goodbye). Kept small on
# purpose — every addition widens what can hang up on a live lead.
_GOODBYE_FILLERS = frozenset({
    "thanks", "thank", "you", "ok", "okay", "alright", "right", "sir",
    "madam", "సరే", "ఓకే", "థ్యాంక్స్", "థాంక్స్", "ధన్యవాదాలు", "సార్",
    "మేడమ్",
})
# Deliberately NOT a filler: "అంతే". Adding it would let the garbled
# repetition this fix exists for — "ఇక అంతే అంతే అంతే" — consume itself
# entirely and hang up again.

# Spoken when the model calls end_call with no farewell text at all — not
# suppressed (see _reply's ends_call handling), just never generated. Per
# CLAUDE.md, measured against the live API: told to say goodbye AND call
# end_call in the same turn, the model sometimes does the tool with no words
# attached (0/3 in one measurement). Ending in total silence either way is
# what CONFIRMED as the "the agent stopped answering" symptom on 2026-08-07 —
# a plain canned farewell is strictly better than nothing.
_FALLBACK_FAREWELL = "ధన్యవాదాలు, శుభదినం."

# Spoken when a turn produces no answer at all — a TurnFailed, a NoAnswer, a
# swallowed exception, or the model looping on tool calls until the round limit.
# Every one of those used to end in a log line and TOTAL SILENCE.
#
# Confirmed on a real call (2026-08-12): the lead asked what courses there are,
# heard nothing, said "హలో", then "చెప్పండి", and only then got an answer. With
# CONVERSATION_LLM_TIMEOUT_S at 12s the silence is twelve seconds long, which on
# a phone line is indistinguishable from a dropped call — and the same symptom
# ("the agent stopped answering") that _FALLBACK_FAREWELL above already exists
# to prevent at the end of a call.
#
# Neither line may carry a course fact, price or date: they are spoken when the
# system does NOT know the answer, and inventing one is the failure CLAUDE.md's
# RAG rule exists to prevent. They ask the lead to repeat, or hand off.
_FALLBACK_REPROMPT = "క్షమించండి, మళ్ళీ ఒకసారి చెప్పగలరా?"

# Spoken once per turn, before a course-material lookup, so the lead hears an
# acknowledgment about 1.5s in instead of the tool turn's full silent span —
# measured 3.18-5.23s end to end on the 2026-08-27 live calls (round-1 LLM +
# lookup + round-2 LLM), during which the first lead of the morning said
# "హలో?" into the wait. Fixed and short on purpose: the model's own tool-call
# text stays suppressed (see _reply — it verbosely announced lookups that
# then failed), and its audio finishes playing at roughly the moment the
# answer's first frame is ready. Pure Telugu, no course facts, no promises.
_RAG_HOLDING_LINE = "ఒక్క క్షణం, చూసి చెప్తాను."
_FALLBACK_HANDOFF = (
    "క్షమించండి, ఇప్పుడు సాంకేతిక సమస్య వస్తోంది. "
    "మా టీమ్ మిమ్మల్ని త్వరలో సంప్రదిస్తుంది."
)
# Consecutive dead turns before the reprompt gives way to the handoff line.
# Asking someone to repeat themselves a third time is its own kind of broken:
# by then the fault is clearly ours, not their diction.
_MAX_FAILED_TURNS = 3


# Recorded announcements the CARRIER plays into the call — a hold notice, a
# call-waiting voice — which Sarvam transcribes as if the lead had said
# them. Live 2026-09-01 21:00: "the person you are speaking with has put your
# call on hold, please stay on the line" arrived as lead speech, the model
# read it as the person leaving, said goodbye and ended the call while the
# lead was still on hold. These are not the lead: they stay in the transcript
# (the operator should see the hold) but never reach history or the model,
# and nothing is answered. Deliberately narrow — hold phrases only, in the
# English and the Telugu-script spellings the STT produces. Voicemail
# prompts are NOT here on purpose: on a voicemail the right outcome IS to end
# the call, which the model already does.
_HOLD_ANNOUNCEMENT_RE = re.compile(
    r"(put\s+your\s+call\s+on\s+hold|call\s+on\s+hold|stay\s+on\s+the\s+line"
    r"|please\s+hold|కాల్\s+ఆన్\s+హోల్డ్|ఆన్\s+హోల్డ్|స్టే\s+ఆన్\s+ద\s+లైన్"
    r"|ప్లీజ్\s+హోల్డ్)",
    re.IGNORECASE,
)


def _is_hold_announcement(text: str) -> bool:
    return bool(_HOLD_ANNOUNCEMENT_RE.search(text))


def _is_lead_goodbye(text: str) -> bool:
    """Whether a transcript is an explicit request to finish the call.

    Keep this deliberately narrow: thanks, okay, and other ordinary turn
    closers must still receive an answer. The LLM remains responsible for
    nuanced decisions; this is the reliable fast path for an unambiguous
    goodbye.

    THE WHOLE utterance must be the goodbye. Matching a phrase anywhere
    inside a longer transcript hung up on leads who were still talking —
    reported live 2026-08-29 as "the call cuts in the middle of the
    conversation" — because this path fires the instant a transcript
    arrives, with no model turn to sanity-check it, and Sarvam's Telugu STT
    garbles constantly on 8 kHz audio ("ఇక అంతే అంతే అంతే" from a lead who
    was mid-question). Three other layers can still end a call, so a MISSED
    goodbye costs one extra turn while a FALSE one hangs up on a customer.
    """
    normalized = " ".join(text.strip().split()).casefold()
    if not normalized:
        return False
    # Cut the goodbyes out, then judge what is LEFT. Removing them first is
    # what keeps the multi-word Latin spellings ("good bye") working, which
    # a token-at-a-time walk cannot see.
    remainder, latin_hits = _GOODBYE_RE.subn(" ", normalized)
    tokens = _goodbye_tokens(remainder)
    phrase_hits = 0
    index = 0
    kept: list[str] = []
    while index < len(tokens):
        for phrase in _GOODBYE_PHRASE_TOKENS:
            if phrase and tokens[index:index + len(phrase)] == phrase:
                index += len(phrase)
                phrase_hits += 1
                break
        else:
            kept.append(tokens[index])
            index += 1
    if not (latin_hits or phrase_hits):
        return False
    # Whatever survives must be politeness. A single leftover word means the
    # lead was still saying something, and something is not a goodbye.
    return all(token in _GOODBYE_FILLERS for token in kept)


# Words a lead uses to PROMPT a reply rather than to ask anything: "hello?",
# "go on", "can you hear me". They carry no question — the lead is filling a
# silence they should not be sitting in.
#
# Measured on the 2026-08-27 live call: the lead asked for course details, the
# reply took 7.5s (llm=6.01s over two tool rounds), and at about the 5s mark
# they said "హలో". That landed after the second round's request was already
# sent, so the missed-fragment follow-up correctly fired for it — and the model,
# handed "హలో" with the freshly-spoken course list above it in history,
# answered by reciting the course list a second time. The lead heard the same
# answer twice.
#
# DELIBERATELY tiny, and every entry is a word that cannot be an answer to
# anything the agent asks. "అవును"/"ఓకే" are excluded on purpose: the agent
# really does ask yes/no questions ("shall I tell you the fee?"), and on the
# same day's second call the lead answered exactly that with "అవును".
_PROMPTING_TOKENS = frozenset({
    "హలో", "హెలో", "హల్లో",          # hello
    "చెప్పండి", "చెప్పు", "చెప్పండీ",   # go on / tell me
    "ఉన్నారా", "వినిపిస్తోందా",        # are you there / can you hear me
    "hello", "helo", "hallo",
})


def _is_only_prompting(text: str) -> bool:
    """Whether *text* is nothing but a nudge for the agent to speak.

    Whole-token matching, like the goodbye check and for the same reason:
    "చెప్పండి" is a prompt on its own but the tail of a real question in
    "కోర్స్ డీటెయిల్స్ చెప్పండి", and substring matching cannot tell those apart.
    """
    tokens = _goodbye_tokens(text.casefold())
    return bool(tokens) and all(t in _PROMPTING_TOKENS for t in tokens)

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
        # (text, cumulative_ms_at_end_of_this_text, this_text's_own_ms)
        self._entries: list[tuple[str, int, int]] = []
        self._total_ms = 0

    def add(self, text: str, duration_ms: int) -> None:
        self._total_ms += duration_ms
        self._entries.append((text, self._total_ms, duration_ms))

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
        return " ".join(text for text, *_ in self._entries)

    def heard_within(self, played_ms: int) -> str:
        """The part of this turn that finished playing by *played_ms*.

        A sentence only PARTLY played is dropped rather than claimed. That
        direction is deliberate: keeping it risks the agent referring back to
        something the lead never received, while dropping it risks the agent
        repeating itself. Redundancy is a much cheaper failure than incoherence
        on a phone call.
        """
        # `duration > 0` is not a nicety. Sarvam's TTS ends its stream on an
        # error frame with ZERO yields and no exception (a rejected speaker, a
        # config the server refuses, credits gone), so say() records the
        # sentence at 0 ms. Without this guard those entries satisfied
        # `ends_at <= played_ms` even at played_ms == 0, and a later barge-in
        # truncate kept text the lead never heard a syllable of. The guard is
        # the sentence's OWN duration, not the running total: an error frame
        # AFTER a played sentence inherits a positive ends_at, and an
        # ends_at > 0 check claimed that unplayed sentence too.
        return " ".join(text for text, ends_at, duration in self._entries
                        if ends_at <= played_ms and duration > 0)


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
        # Where that fragment landed in history, against what the last request
        # to the model actually carried — see _reply_when_they_stop's gate.
        self._missed_at = 0
        # EVERY fragment missed by the committed round, not just the last.
        # A single slot let a trailing "హలో" classify the whole round's
        # follow-up as a nudge and silently drop a real question that arrived
        # just before it (confirmed with an executed repro, 2026-08-27).
        self._missed_texts: list[str] = []
        self._last_request_len = 0
        # Wall-clock of the last thing either side did. The silence watchdog
        # measures against this rather than against call start, so a lead who
        # is mid-question is never hung up on.
        self._loop = asyncio.get_event_loop()
        self.last_activity = self._loop.time()
        # Index into self.turns / self.history of the agent turn currently
        # being spoken, so a barge-in can rewrite exactly that one.
        self._agent_turn_index: int | None = None
        # Whether the turn at _agent_turn_index also has a history entry.
        # False for an ASIDE (the RAG holding line): truncate must then leave
        # history entirely alone — the newest assistant message there is
        # somebody else's (typically the tool_calls bookkeeping, whose
        # deletion orphans its tool result and 400s every later completion).
        self._agent_turn_in_history: bool = True
        # Sarvam's own view of whether the lead is mid-utterance, from its
        # START_SPEECH / END_SPEECH VAD signals. Read by _reply_when_they_stop
        # so a reply need not blind-wait the full ceiling once Sarvam has
        # already said the lead stopped.
        self._lead_speaking = False
        self._speech_started_at: float | None = None
        self._end_call_refusals = 0
        # Per-turn latency accounting — see _log_turn_latency. Measured from
        # the first thing the lead said that nobody had answered yet (a burst
        # is ONE turn, so later fragments deliberately do not reset it) to the
        # first audio frame they hear back. Without this, "the agent replies
        # late" is unattributable: the gate, the model, the course lookup and
        # the synthesiser are four very different fixes and the logs could not
        # tell them apart.
        self._turn_opened_at: float | None = None
        self._turn_gate_done: float | None = None
        # t0 -> t1: END_SPEECH to final transcript. The recogniser's
        # own thinking time, which the lead waits through and which no
        # other number here covers — `total` starts at the transcript.
        self._speech_ended_at: float | None = None
        self._turn_stt_s = 0.0
        self._turn_say_started: float | None = None
        self._turn_llm_s = 0.0
        self._turn_tool_s = 0.0
        self._turn_llm_rounds = 0
        # Consecutive turns that produced no answer — see _say_fallback. Reset
        # by any turn that actually speaks, so one bad turn in an otherwise
        # healthy call does not push the next one straight to a handoff.
        self._failed_turns = 0
        # Whether any audio from THIS turn actually reached the lead. Set in
        # say() beside spoke_a_frame. _say_fallback reads it so a failure that
        # lands after something was already spoken — an end_call farewell, say,
        # followed by a tool error — cannot stack "sorry, say that again" on
        # top of words the lead just heard.
        self._heard_something_this_turn = False

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
            f"stt={self._turn_stt_s:.2f}s "
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
        self._turn_stt_s = 0.0
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

    async def say(self, tts: sarvam_tts.SarvamTTS, text: str, *,
                  aside: bool = False) -> None:
        """Speak *text*, recording what actually reached the lead.

        Each sentence is synthesised, played, and logged to the ledger with its
        audio duration. If the generation changes mid-way the lead has
        interrupted, so the rest is abandoned unplayed.

        *aside* is True only for the RAG holding line, and bundles three
        deliberate exclusions (2026-08-27, the second and third confirmed by
        an adversarial review after the first shipped alone):
          1. It does not set _heard_something_this_turn or fire the
             turn-latency log — that flag means "an ANSWER reached the lead"
             and gates the failed-turn counter reset, _say_fallback's early
             return and the nudge suppression. A holding line that set it
             would disarm all three: a call whose every answer failed would
             say "ఒక్క క్షణం" forever and grade as a clean success.
          2. It is NOT appended to the model history. It carries nothing the
             model needs, and as an assistant-with-content message it made
             _awaiting_answer report a cancelled lookup's question as
             answered — the cough-repair then never re-asked, the call went
             silent, and graded done.
          3. _agent_turn_in_history is set False so a barge-in's truncate
             leaves history alone — the newest assistant message there is
             the tool_calls bookkeeping, whose deletion orphans its tool
             result and 400s every later completion.
        The ledger/transcript recording is unconditional either way: the
        line WAS spoken, and the record says what the lead heard.
        """
        # This is the last boundary before Sarvam TTS. Keep every fallback,
        # tool farewell, opening, and normal reply safe even if it bypassed
        # the renderer or came from a cached/third-party model response.
        #
        # normalize_spoken_telugu drops words in scripts the synthesiser
        # cannot speak, which rescues the sentence for the lead but hides a
        # real quality signal — so it is reported here before it disappears.
        # Live 2026-09-01: the model slipped the Armenian "պահին" into a
        # Telugu sentence and TTS read it aloud; the same check catches the
        # Hindi drift CLAUDE.md documents.
        unspeakable = sarvam_prompts.foreign_script_tokens(text)
        if unspeakable:
            logger.warning(
                f"[sarvam] lead={self.lead_id} the model emitted words that "
                f"are not Telugu or English and cannot be spoken — dropping "
                f"{unspeakable!r}. Check the language rule in the prompt."
            )
        text = sarvam_prompts.normalize_spoken_telugu(text)
        sentences = _split_sentences(text)
        if not sentences:
            return

        generation = self.generation
        self.ledger.reset()
        # Restart the playback clock: played_ms() is relative to the current
        # response, and a barge-in truncates against it.
        self.call.mark_response_boundary()

        self._agent_turn_index = len(self.turns)
        self._agent_turn_in_history = not aside
        self.turns.append(TranscriptTurn(role="agent", text=text))
        if not aside:
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
                        # by _log_turn_latency's tts= figure, not a gap. The
                        # play_end > 0.0 guard is the sentinel PlivoCall uses
                        # for "nothing queued yet" (see its field comment) —
                        # never a real gap, so it must not read as one.
                        now = self._loop.time()
                        if self.call.play_end > 0.0 and now > self.call.play_end:
                            gap_ms = (now - self.call.play_end) * 1000
                            gap_total_ms += gap_ms
                            gap_max_ms = max(gap_max_ms, gap_ms)
                    await self.call.play(payload)
                    if not spoke_a_frame:
                        # The lead's wait ends HERE, at the first frame — not
                        # when the whole answer has been sent.
                        spoke_a_frame = True
                        if not aside:
                            self._heard_something_this_turn = True
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
            else:
                # NOT ONE FRAME reached the lead, so this turn did not happen.
                # say() appends to turns/history BEFORE synthesising, which is
                # what lets a barge-in truncate mid-sentence — but it also
                # means a speak() that raises, or one that ends on an error
                # frame with zero yields, leaves a turn nobody heard sitting in
                # the record. truncate_to_what_was_heard cannot clean that up
                # on the raise path: converse() catches the error and
                # _cancel_reply early-returns with no reply task in flight.
                #
                # It matters most on the OPENING. The stored transcript would
                # certify an AI disclosure that was never played, and
                # call_routes' post-call [compliance] check reads that same
                # transcript and passes on it.
                #
                # Reuses the ledger's own "nothing played" branch, which drops
                # the turn rather than leaving an empty one, and is idempotent
                # (it clears _agent_turn_index on first use).
                self.truncate_to_what_was_heard()

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
            # And so does the lead's own voice. Sarvam emits speech_started
            # ONCE and then nothing until the transcript, so a lead in the
            # middle of a long question looked identical to a dead line and
            # was hung up on mid-sentence — reported live 2026-08-29.
            # Bounded by _MAX_LEAD_SPEECH_S: a lost END_SPEECH must not
            # disable the watchdog for the rest of the call.
            if (self._lead_speaking and self._speech_started_at is not None
                    and self._loop.time() - self._speech_started_at
                    < _MAX_LEAD_SPEECH_S):
                self.note_activity()
                continue
            quiet_since = max(self.last_activity, self.call.play_end)
            if self._loop.time() - quiet_since >= TWOWAY_MAX_SILENT_S:
                logger.info(
                    f"[sarvam] lead={self.lead_id} ending a conversation that "
                    f"has been silent for {TWOWAY_MAX_SILENT_S:.0f}s."
                )
                # A conversation that ended in silence and one where the
                # agent was never audible AT ALL are different events. The
                # second is a TTS outage: grading it a clean finish marked the
                # lead done and retried nobody, so a campaign of totally
                # silent calls looked like a campaign of good ones. Mirrors
                # the one-way path's own no-audio distinction.
                # Three different endings wear the same silence:
                #   - a real conversation that finished        -> clean
                #   - the agent was never audible at all       -> TTS outage
                #   - the agent was audible but only apologised, and ended on
                #     the handoff line promising a callback nothing schedules
                #                                              -> brain outage
                # Grading the last two 'done' marked the lead done, retried
                # nobody, and showed the operator a campaign of successful
                # calls while every lead heard an apology and a promise no one
                # will keep.
                if not self.call.audio_seen:
                    reason = "twoway_no_audio"
                elif self._failed_turns >= _MAX_FAILED_TURNS:
                    logger.error(
                        f"[sarvam] lead={self.lead_id} ending a call that never "
                        "answered anything — the lead heard only apologies and "
                        "the handoff promise. Check the conversation provider."
                    )
                    reason = "twoway_never_answered"
                else:
                    reason = "twoway_silence"
                self.call.note_exit(reason)
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

        # An aside (the RAG holding line) has NO history entry, so both
        # branches below must leave history alone for it: the newest
        # assistant message in history is then someone else's — typically
        # the tool_calls bookkeeping, and rewriting or deleting THAT orphans
        # its tool result, which OpenAI rejects on every later completion.
        # The walks also skip tool_calls messages outright as defence in
        # depth: they are bookkeeping, never the spoken turn.
        in_history = self._agent_turn_in_history
        self._agent_turn_in_history = True

        if heard:
            self.turns[index] = TranscriptTurn(role="agent", text=heard)
            if in_history:
                for message in reversed(self.history):
                    if (message.get("role") == "assistant"
                            and not message.get("tool_calls")):
                        message["content"] = heard
                        break
            return

        # Nothing played at all: as far as the conversation is concerned the
        # agent did not speak. Drop the turn rather than leave an empty one,
        # which would read as the agent having said nothing on purpose.
        if index < len(self.turns):
            del self.turns[index]
        if in_history:
            for position in range(len(self.history) - 1, 0, -1):
                message = self.history[position]
                if (message.get("role") == "assistant"
                        and not message.get("tool_calls")):
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
        self._speech_started_at = self._loop.time()
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
        self._speech_started_at = None
        self._speech_ended_at = self._loop.time()
        if self.reply_task is not None and not self.reply_task.done():
            return  # already armed — the wait it's polling will now unblock
        if self._awaiting_answer():
            self.reply_task = asyncio.create_task(self._reply_when_they_stop(tts))

    def _lead_has_open_question(self) -> bool:
        """The lead's last utterance is a question nobody has answered yet.

        Walks back to whichever comes first: an assistant turn WITH WORDS
        (something was answered since, so nothing is outstanding) or the
        lead's own last message. Tool bookkeeping is skipped, exactly as in
        _awaiting_answer — a lookup in flight has answered nothing.
        """
        for message in reversed(self.history):
            role = message.get("role")
            if role == "assistant" and message.get("content"):
                return False
            if role == "user":
                text = (message.get("content") or "").strip()
                return bool(text) and text.endswith(("?", "？")) \
                    and not _is_lead_goodbye(text)
        return False

    def _lead_only_agreed(self) -> bool:
        """Whether the lead's last words were a bare "yes" and nothing else.

        The observed failure, 2026-09-01: a call ended on "ఆ ఓకే." An
        affirmation is the lead AGREEING with the agent — it is never a
        request to hang up, and the owner's rule is that the call ends when
        they say goodbye.

        Deliberately narrow. "సరే బాయ్" carries బాయ్, "Okay, thank you"
        carries thanks — both are wrap-ups the model should still be trusted
        to end on, and Sarvam's STT spells a spoken "bye" in ways no token
        list reliably catches (బాయ్ was transcribed on a live call and is
        not in _GOODBYE_PHRASES). So this refuses ONLY an utterance that is
        pure agreement, and leaves every richer ending to the model.
        """
        for message in reversed(self.history):
            if message.get("role") != "user":
                continue
            tokens = _goodbye_tokens((message.get("content") or "").casefold())
            return bool(tokens) and all(t in _AFFIRMATION_ONLY for t in tokens)
        return False

    def _should_refuse_end_call(self) -> bool:
        """Whether an end_call right now would hang up on a live lead.

        The rule the owner asked for on 2026-09-01, after a call ended on a
        bare "ఆ ఓకే": the call ends when the LEAD says goodbye, and not
        before. An earlier version refused only when the lead's last words
        were a QUESTION, which let an affirmation through — and an
        affirmation is the lead agreeing with the agent, never asking to
        leave.

        Refusing is safe: watch_for_silence still ends a call nobody is
        speaking on, so a lead who simply stops talking is never trapped,
        and the refusal count is capped besides.

        PURE — _reply consults it to decide whether to speak the farewell,
        and _run_tools consults it again to decide whether to end; both must
        get the same answer for one reply, so the counter is incremented by
        _run_tools alone.
        """
        if self._end_call_refusals >= _MAX_END_CALL_REFUSALS:
            return False
        return self._lead_only_agreed() or self._lead_has_open_question()

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
        # That cancel may have killed say() mid-utterance, so the same repair
        # on_barge_in performs is owed here too: without it the full answer
        # stays in history and in the stored transcript though only part of it
        # played, and every later turn is built on words nobody heard.
        # Idempotent — truncate_to_what_was_heard clears _agent_turn_index on
        # its first use, so on_barge_in's own call stays harmless.
        self.truncate_to_what_was_heard()
        self._discard_incomplete_tool_turn()
        # Fragments flagged for the round just cancelled are consumed by no
        # one — but they are already in history, and the REPLACEMENT round
        # re-reads the whole history, so they still get answered. Leaving
        # them flagged let a stale real question from a dead round defeat a
        # LATER round's lone-nudge suppression and re-recite an answer.
        self._missed_while_committed = False
        self._missed_texts = []

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
        # Repair known names BEFORE anything reads the text. Everything
        # downstream is built from this one string — the model's history, the
        # RAG query the model writes from it, and the transcript the operator
        # reads afterwards — so repairing it once here keeps all three
        # agreeing about what the lead said. A live call was lost to this: the
        # brand came back as "బ్రౌనీ" and the agent refused a question about
        # its own courses. See term_repair's docstring.
        text = sarvam_prompts.normalize_heard_telugu(text)
        text = term_repair.repair(text)
        self.turns.append(TranscriptTurn(role="lead", text=text))
        if _is_hold_announcement(text):
            # The line, not the lead. Recorded for the operator, kept from
            # the model, answered by nobody — see _HOLD_ANNOUNCEMENT_RE.
            logger.info(f"[sarvam] lead={self.lead_id} ignored a hold "
                        f"announcement from the line: {text!r}")
            return
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
        if self._speech_ended_at is not None:
            # Only when END_SPEECH actually preceded the transcript. Sarvam
            # delivers the two in either order, and a transcript that beat
            # the signal must report nothing rather than a negative wait.
            self._turn_stt_s = max(0.0, self._loop.time() - self._speech_ended_at)
            self._speech_ended_at = None
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
            self._missed_at = len(self.history)
            self._missed_texts.append(text)
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
            # ...but only if no request actually carried it. A tool round
            # re-serialises the WHOLE history, so round 2 of a tool turn
            # usually DID include the fragment and answered both parts; firing
            # regardless queried the model again with nothing new and the agent
            # spoke an extra, unsolicited turn at the lead.
            #
            # _awaiting_answer() cannot answer this: say() appends the
            # assistant message ON TOP of the fragment, so it reports
            # "answered" even for a round that never saw it. Comparing the last
            # request's history length against where the fragment landed is the
            # question actually being asked.
            #
            # ...and not when the fragments were only the lead nudging us to
            # speak. Answering "హలో" once the answer they were waiting for has
            # just played makes the model recite it again — see
            # _PROMPTING_TOKENS for the call this was measured on. The nudge is
            # already satisfied by the turn that just spoke, so the honest
            # response to it is nothing. Gated on that turn HAVING spoken:
            # if it produced no audio, the lead's "hello?" is still unanswered
            # and deserves the follow-up. EVERY missed fragment must be a
            # nudge for the suppression to apply — judging only the last one
            # let a trailing "హలో" silently cancel the follow-up owed to a
            # real question that arrived just before it in the same round.
            missed_texts = self._missed_texts
            self._missed_texts = []
            prompting = (self._heard_something_this_turn
                         and bool(missed_texts)
                         and all(_is_only_prompting(t) for t in missed_texts))
            if prompting:
                logger.info(
                    f"[sarvam] lead={self.lead_id} dropping follow-up for "
                    f"{missed_texts!r} — the turn it was nudging already spoke"
                )
            if self._last_request_len < self._missed_at and not prompting:
                if self._turn_opened_at is None:
                    self._turn_opened_at = self._loop.time()
                self.reply_task = asyncio.create_task(
                    self._reply_when_they_stop(tts))

    async def _say_fallback(self, tts: sarvam_tts.SarvamTTS,
                            generation: int) -> None:
        """Speak a recovery line for a turn that produced no answer.

        The alternative is silence, and silence after a question is the single
        most damaging thing this backend can do — see _FALLBACK_REPROMPT.

        *generation* is the value captured when the turn began. If it has moved
        the lead interrupted while the model was working: that turn is over, and
        speaking into it would talk over the person who just took the floor. A
        cancelled turn is also not counted as a failure — nothing was wrong with
        it except that the lead had more to say.
        """
        if self._heard_something_this_turn:
            # Something already reached the lead this turn; the failure came
            # afterwards. Apologising for a turn they actually heard is worse
            # than saying nothing more.
            return
        if self.generation != generation or self.call.stop.is_set():
            return
        if self._failed_turns >= _MAX_FAILED_TURNS:
            # The handoff has already been spoken. Repeating it on every
            # further dead turn is how a lead learns it is talking to a
            # recording — and by now the fault is loudly logged for us.
            logger.error(
                f"[sarvam] lead={self.lead_id} another dead turn after the "
                "handoff was already spoken — this call is not working. Check "
                "the conversation provider."
            )
            return
        self._failed_turns += 1
        line = (_FALLBACK_HANDOFF if self._failed_turns >= _MAX_FAILED_TURNS
                else _FALLBACK_REPROMPT)
        try:
            await self.say(tts, line)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - this IS the recovery path
            # Nothing left to fall back to; log loudly rather than raise into
            # the caller's own exception handler and lose the original failure.
            logger.error(f"[sarvam] lead={self.lead_id} could not even speak "
                         f"the fallback: {type(exc).__name__}: {exc}")

    async def _reply(self, tts: sarvam_tts.SarvamTTS) -> None:
        generation = self.generation
        self._heard_something_this_turn = False
        try:
            for _round in range(_MAX_TOOL_ROUNDS):
                _llm_started = self._loop.time()
                self._reply_committed = True
                # What this request actually carries, for the missed-fragment
                # follow-up: a fragment appended after this point cannot have
                # been seen by it.
                self._last_request_len = len(self.history)
                try:
                    reply = await conversation_llm.turn(self.history)
                finally:
                    self._reply_committed = False
                self._turn_llm_s += self._loop.time() - _llm_started
                self._turn_llm_rounds += 1
                if not reply.tool_calls:
                    if reply.text:
                        # Sarvam TTS reads Telugu literally. Keep the model's
                        # response in the history/transcript in the same form
                        # that is actually spoken, including the narrow repair
                        # for recurring missing-vowel spellings.
                        reply_text = sarvam_prompts.normalize_spoken_telugu(
                            reply.text)
                        await self.say(tts, reply_text)
                        # say() returning is not evidence the lead HEARD
                        # anything: an error frame from Sarvam ends the stream
                        # with zero yields and no exception. Resetting the
                        # counter there let a total TTS outage look like a
                        # healthy conversation, turn after turn, and the
                        # reprompt built for this symptom never fired.
                        if self._heard_something_this_turn:
                            self._failed_turns = 0
                        elif (self.generation == generation
                              and not self.call.stop.is_set()):
                            logger.error(
                                f"[sarvam] lead={self.lead_id} the synthesiser "
                                "produced no audio for this turn — the lead "
                                "heard nothing. Treating it as a failed turn."
                            )
                            await self._say_fallback(tts, generation)
                    return
                # Text that arrives WITH a tool call is usually filler — "let
                # me look that up for you". On a real call the agent
                # announced its own lookup twice and then reported failure,
                # which is three synthesised turns to deliver nothing — so
                # the MODEL's text stays suppressed. But "silence is shorter
                # than saying so" undercounted the silence: the lead's wait
                # is round-1 LLM + lookup + round-2 LLM, measured 3.18-5.23s
                # on the 2026-08-27 live calls, and the lead talked into it.
                # A FIXED holding line (below, after the ends_call branch)
                # covers that window instead: one short canned sentence, not
                # the model's variable announcement.
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
                if ends_call and self._should_refuse_end_call():
                    # The lead asked something and it has not been answered.
                    # Speaking the farewell here would say goodbye to a
                    # question — _run_tools refuses the ending itself and
                    # tells the model to answer instead.
                    ends_call = False
                if ends_call:
                    # No next round for this text to arrive in, so speak
                    # whatever the model wrote — or, if it wrote nothing at
                    # all, a canned farewell rather than hanging up in
                    # silence. See _FALLBACK_FAREWELL.
                    await self.say(
                        tts,
                        sarvam_prompts.normalize_spoken_telugu(
                            reply.text or _FALLBACK_FAREWELL
                        ),
                    )
                else:
                    if reply.text:
                        logger.debug(
                            f"[sarvam] lead={self.lead_id} not speaking "
                            f"tool-call filler: {reply.text[:60]!r}")
                    # Before _run_tools on purpose: its ~1.5s of audio plays
                    # while the lookup and the next round compute. Spoken as
                    # an ASIDE — see say(): it stays out of the model
                    # history entirely, so it can never displace the
                    # tool_calls/tool pairing or convince _awaiting_answer
                    # that a cancelled lookup's question was answered. First
                    # round only — a chained lookup already has it playing —
                    # and never as the call's first spoken agent turn:
                    # call_routes re-checks that turn for the AI disclosure
                    # at ERROR level, and if the opening never played, a
                    # holding line must not become the evidence.
                    searching = any(
                        tc.name == conversation_llm.SEARCH_TOOL_NAME
                        for tc in reply.tool_calls)
                    if (_round == 0 and searching
                            and any(t.role == "agent" for t in self.turns)):
                        await self.say(tts, _RAG_HOLDING_LINE, aside=True)
                if await self._run_tools(reply):
                    return  # end_call: nothing further to say
            logger.warning(
                f"[sarvam] lead={self.lead_id} the model kept calling tools for "
                f"{_MAX_TOOL_ROUNDS} rounds — giving up on this turn rather "
                "than holding the lead on a silent line."
            )
            await self._say_fallback(tts, generation)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad turn must not end the call
            logger.error(f"[sarvam] lead={self.lead_id} turn failed: "
                         f"{type(exc).__name__}: {exc}")
            await self._say_fallback(tts, generation)
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
                if self._should_refuse_end_call():
                    # Instructions alone did not hold: across three live
                    # calls the model ended on an affirmation, on garbled
                    # speech, and finally on a plain question ("will you
                    # tell me class timings?"). The rule is enforced here
                    # instead, where it cannot be talked out of.
                    self._end_call_refusals += 1
                    logger.warning(
                        f"[sarvam] lead={self.lead_id} refused end_call — the "
                        "lead has not said goodbye "
                        f"({self._end_call_refusals}/{_MAX_END_CALL_REFUSALS})."
                    )
                    self.history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.call_id,
                        "content": ("The call was NOT ended: the lead has "
                                    "not said goodbye. An 'okay' or a nod is "
                                    "not a farewell. Continue the "
                                    "conversation in Telugu — answer what "
                                    "they said, or ask what else they would "
                                    "like to know — and do not call end_call "
                                    "again until they actually say goodbye."),
                    })
                    continue
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


# The documents, read BEFORE the call — the owner's direction after the
# 2026-08-29 test round, where every common question paid a tool round-trip
# and a spoken waiting line, and the fee question STILL missed (top-k
# crowd-out needs no conjunction: "SEO course fees" drowns the fee chunk in
# SEO-syllabus matches with nothing for the query splitter to split). These
# fixed English queries sweep the topics every call asks about; the digest
# rides in the system prompt so the model answers them in ONE round, and
# the search tool remains for the long tail.
#
# search_relevant, never search_permissive: the digest feeds live calls, so
# the relevance floor applies to it exactly as to the tool (CLAUDE.md).
_FACTS_QUERIES = (
    "course fees and complete price structure for all programs",
    "digital marketing course programs and levels overview",
    "course duration batches and timings",
    "placement internship and job support",
)
_FACTS_CACHE_KEY = "sarvam:course_facts:v1"
_FACTS_TTL_S = 3600          # re-ingested documents propagate within the hour
# Paid on EVERY request, so it is a latency knob as much as a size one:
# measured 2026-08-29, a 6.9k digest cost ~0.6s per turn against no digest
# at all. 4500 is the smallest value that still carried every fact once the
# budget was shared per topic and the curriculum noise was filtered out —
# below ~3000 the fee table itself starts falling off. Re-measure with the
# cap sweep in this file's tests if the documents grow substantially.
_FACTS_MAX_CHARS = 4500
# One failed build arms this instead of every call re-paying the stall for
# the whole outage — an answered lead in silence each time, per the review.
_FACTS_COOLDOWN_KEY = "sarvam:course_facts:cooldown"
_FACTS_COOLDOWN_S = 120
# The fetch sits between Plivo answering and the opening being spoken, and a
# HANG (a slow-but-alive embeddings endpoint, a wedged pool) is not an
# exception — neither degrade layer fires on it. Deadline-bounded at the
# call site exactly like the render (RENDER_DEADLINE_S) and the in-call
# lookups (RAG_VOICE_DEADLINE_S): a cold cache may cost this much silence
# once, never more.
FACTS_DEADLINE_S = 3.0


# Lines the digest carries no benefit from, dropped before it is measured
# against the cap. The retrieval pulls whole document regions, so a query
# about programs drags in the numbered curriculum index behind it — measured
# 2026-08-29 as 38% of a 6.8k-char digest, paid on EVERY request (~0.6s per
# turn). None of it answers the fee/duration/placement questions the digest
# exists for, and the search tool still covers it for the long tail.
_DIGEST_NOISE = (
    re.compile(r"^\s*-?\s*\d+\.\s"),          # "- 113. What is SEO?"
    re.compile(r"^\s*www\.|@|\[URL"),          # contact/footer lines
)


def _useful_facts(block: str) -> str:
    """*block* with curriculum-index and contact noise removed."""
    kept = [line for line in block.splitlines()
            if line.strip() and not any(p.search(line) for p in _DIGEST_NOISE)]
    return "\n".join(kept).strip()


async def _course_facts() -> str:
    """The course-facts digest, cached per corpus. Never raises: the digest
    is an optimisation, and RAG being down must not stop a call — the tool
    path still covers every question, just slower."""
    r = None
    try:
        r = redis_client.get_redis()
        cached = await r.get(_FACTS_CACHE_KEY)
        if cached:
            return cached.decode() if isinstance(cached, bytes) else str(cached)
        if await r.get(_FACTS_COOLDOWN_KEY):
            return ""    # a recent build failed; don't stall this call too
    except Exception:  # noqa: BLE001 - cache is best-effort
        pass
    try:
        results = await asyncio.gather(
            *(search_relevant(q) for q in _FACTS_QUERIES))
    except Exception as exc:  # noqa: BLE001 - degrade to tool-only behaviour
        logger.warning(
            f"[sarvam] course-facts digest unavailable "
            f"({type(exc).__name__}: {exc}) — the search tool still covers "
            "questions, at a round-trip per answer."
        )
        if r is not None:
            with contextlib.suppress(Exception):
                await r.set(_FACTS_COOLDOWN_KEY, "1", ex=_FACTS_COOLDOWN_S)
        return ""
    # A share of the budget per TOPIC, not first-come-first-served. The fee
    # query returns the most verbose chunks, so greedily filling from it ate
    # the whole cap and the duration/mode facts fell off the end entirely —
    # measured 2026-08-29 at every cap below 7000, while the agent was
    # simultaneously being asked those exact questions on live calls.
    # Unspent budget rolls forward, so a quiet topic never wastes the room a
    # verbose one could have used.
    share = max(1, _FACTS_MAX_CHARS // max(1, len(results)))
    seen: set[str] = set()
    blocks: list[str] = []
    total = 0
    allowance = 0
    for result in results:
        allowance += share
        if not result or result == NO_MATERIAL_NOTE:
            continue
        for raw_block in result.split("\n\n---\n\n"):
            block = _useful_facts(raw_block)
            if not block or block in seen:
                continue
            # Whole blocks only. A raw character slice could land inside a
            # figure — "Total Fee: ₹1,5" — and the prompt presents this text
            # as fact to answer from directly, with no fresh lookup to
            # correct it. The straddling block is dropped, never halved.
            cost = len(block) + (2 if blocks else 0)
            if total + cost > min(allowance, _FACTS_MAX_CHARS):
                continue
            seen.add(block)
            blocks.append(block)
            total += cost
    digest = "\n\n".join(blocks)
    if digest and r is not None:
        with contextlib.suppress(Exception):
            await r.set(_FACTS_CACHE_KEY, digest, ex=_FACTS_TTL_S)
    return digest


# Strong references to rebuilds that outlived their caller's patience. Without
# these the task is only weakly referenced and may be garbage collected
# mid-flight, which is the difference between "the next call has the facts"
# and "every call rebuilds them".
_BACKGROUND_REBUILDS: set[asyncio.Task] = set()


async def _spoken_name(lead_id: str, raw_name: str | None) -> str:
    """*raw_name* in Telugu script, or as written if that cannot be had.

    Sarvam's Telugu TTS reads a Latin run with English phonetics — the owner
    heard "Mouli గారు" come out wrong on a live call. Bounded and degrading on
    purpose: a mispronounced name is a blemish, a delayed or missing greeting
    is a broken call.

    Shared by both modes. It lived inside _prepare_opening (two-way only) and
    every one-way transcript still shows "నమస్తే Mouli గారు" as late as
    2026-08-18 — same synthesiser, same mispronunciation, so the same fix.
    """
    if not raw_name:
        return ""
    try:
        return await asyncio.wait_for(
            lead_name.telugu_name(raw_name), FACTS_DEADLINE_S)
    except Exception as exc:  # noqa: BLE001 - incl. TimeoutError
        logger.warning(
            f"[sarvam] lead={lead_id} could not render the name "
            f"{raw_name!r} in Telugu ({type(exc).__name__}) — greeting "
            "with it as written."
        )
        return raw_name


async def _prepare_opening(lead_id: str, raw_name: str | None):
    """(spoken name, course-facts digest) — fetched concurrently, each on its
    OWN deadline.

    Everything a two-way call needs BEFORE the lead hears anything, so both
    are bounded and both degrade to something harmless: the name to its
    written form (a mispronounced name is a blemish, a missing greeting is a
    broken call), the digest to "" (the search tool still answers every
    question, one round-trip slower).

    SEPARATE deadlines, not one around both. Live 2026-08-31: the digest
    cache expired at the minute of a call, its rebuild blew a shared
    deadline, and the single wait_for cancelled the name lookup with it — so
    a lead heard their own name in English letters because something
    OPTIONAL was slow. Independent work gets independent bounds.

    A digest that misses its deadline is not abandoned, only stopped being
    waited on: the rebuild is shielded so it runs to completion and fills the
    cache, and an hourly expiry then costs ONE call its facts instead of
    every call until someone warms it by hand.
    """
    async def name() -> str:
        return await _spoken_name(lead_id, raw_name)

    async def facts() -> str:
        rebuild = asyncio.ensure_future(_course_facts())
        _BACKGROUND_REBUILDS.add(rebuild)
        rebuild.add_done_callback(_BACKGROUND_REBUILDS.discard)
        try:
            return await asyncio.wait_for(
                asyncio.shield(rebuild), FACTS_DEADLINE_S)
        except asyncio.TimeoutError:
            logger.warning(
                f"[sarvam] lead={lead_id} course-facts fetch exceeded "
                f"{FACTS_DEADLINE_S:.0f}s — answering from the search tool "
                "this call; the rebuild continues and will be cached."
            )
            return ""
        except Exception as exc:  # noqa: BLE001 - the tool still covers it
            logger.warning(
                f"[sarvam] lead={lead_id} course-facts fetch failed "
                f"({type(exc).__name__}) — continuing with the search tool."
            )
            return ""

    resolved, digest = await asyncio.gather(name(), facts())
    return (resolved or raw_name), digest


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
            #
            # Served from a CONSTANT, not the renderer: this script is itself a
            # constant, so its Telugu cannot differ between calls, and paying a
            # model for it cost two live calls on 2026-08-27 when Sarvam's
            # reasoning outgrew its token budget and returned nothing. See
            # sarvam_prompts.DEFAULT_TWOWAY_TELUGU. The normal two-way call now
            # has no dependency on the renderer at all. Chosen by register:
            # a Tinglish campaign used to open in pure Telugu because there
            # was only the one constant.
            body = sarvam_prompts.default_twoway_opening(
                variables.get("language_style"))
        else:
            try:
                body = await asyncio.wait_for(
                    sarvam_llm.render(
                        script, language_style=variables.get("language_style"),
                    ),
                    RENDER_DEADLINE_S,
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"[sarvam] lead={lead_id} script render did not finish inside "
                    f"{RENDER_DEADLINE_S:.0f}s — abandoning the call rather than "
                    "holding an answered lead in silence. Check Sarvam's status; "
                    "the render cache is per campaign, so the next lead pays this "
                    "again until it succeeds once."
                )
                outcome["note"] = "sarvam_render_timeout"
                return outcome
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
        # The lead's own name, in a script the synthesiser can pronounce.
        # It arrives from the console or the sheet in Latin letters, and
        # Sarvam's Telugu TTS reads a Latin run with English phonetics — the
        # owner heard "Mouli గారు" come out wrong on a live call. Bounded and
        # degrading exactly like the digest below it: a mispronounced name is
        # a blemish, a delayed or missing greeting is a broken call.
        raw_name = variables.get("lead_name")
        if one_way:
            # The name ALONE. The course-facts digest exists to answer
            # questions and a one-way call takes none, so fetching it would
            # spend FACTS_DEADLINE_S of the lead's silence on something
            # nothing in this mode can read.
            spoken_name, course_facts = await _spoken_name(lead_id, raw_name), ""
        else:
            # TOGETHER, under ONE deadline. Both finish before the lead hears
            # a word, and both are independent, so awaiting them in sequence
            # would stack two deadlines' worth of silence onto a cold start
            # (a Redis flush, or simply a name never dialled before).
            spoken_name, course_facts = await _prepare_opening(
                lead_id, raw_name)

        spoken = sarvam_llm.ensure_disclosure(
            sarvam_prompts.normalize_spoken_telugu(
                sarvam_prompts.compose_spoken(body, lead_name=spoken_name)
            )
        )

        if not one_way:
            await _run_two_way(
                call, lead_id=lead_id, opening=spoken,
                system_prompt=sarvam_prompts.two_way_instructions(
                    # The Telugu-script name the greeting speaks, so the
                    # model can say it too — given the English spelling it
                    # avoided the name and said a bare "గారు".
                    spoken_name or variables.get("lead_name"),
                    variables.get("script"),
                    language_style=variables.get("language_style"),
                    course_facts=course_facts,
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
