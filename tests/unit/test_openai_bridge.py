"""Unit tests for telephony/openai_bridge.py — the Telugu voice backend.

Driven against a fake OpenAI Realtime socket and the same fake Plivo socket
`PlivoCall` is tested with, so no network and no Docker.

The behaviour that matters most here is the same as the ElevenLabs bridge's:
a one-way call must end once its message has been delivered, and a mid-call
failure must still hand back the transcript of a conversation that really
happened. Both are inherited from PlivoCall, so these tests prove the wiring
is right, not that the watchdog works (tests/unit/test_plivo_stream.py owns
that).
"""

import asyncio
import base64
import json

import pytest

from app import languages
from app.telephony import openai_bridge, openai_prompts

_AUDIO_B64 = base64.b64encode(b"\xff" * 640).decode()


def _audio_delta(b64: str = _AUDIO_B64) -> str:
    return json.dumps({"type": "response.output_audio.delta", "delta": b64})


def _agent_transcript(text: str) -> str:
    return json.dumps({
        "type": "response.output_audio_transcript.delta", "delta": text,
    })


def _response_done() -> str:
    return json.dumps({"type": "response.done", "response": {"output": []}})


class _FakeOpenAIWS:
    """Yields *messages*, then goes quiet — matching a one-way call, where the
    model has no turn to end and never closes the socket itself."""

    def __init__(self, messages: list[str]):
        self._messages = list(messages)
        self.sent: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def send(self, payload):
        self.sent.append(payload)

    async def close(self):
        pass

    def __aiter__(self):
        async def _gen():
            for m in self._messages:
                yield m
            await asyncio.Event().wait()
        return _gen()


class _FakePlivoWS:
    def __init__(self):
        self.sent: list[dict] = []

    async def receive_text(self):
        await asyncio.Event().wait()

    async def send_text(self, raw):
        self.sent.append(json.loads(raw))


@pytest.fixture
def bridged(monkeypatch):
    monkeypatch.setattr(openai_bridge, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(openai_bridge.plivo_stream, "ONEWAY_SILENCE_TAIL_S", 0)
    monkeypatch.setattr(openai_bridge.plivo_stream, "ONEWAY_MAX_SILENT_S", 0)
    # A two-way call has no watchdog to end it, and the fake OpenAI socket goes
    # quiet rather than closing (matching a real one, which stays open until the
    # lead hangs up), so nothing sets `stop` on its own. Bound the call short so
    # the two-way tests fall through to max_duration instead of the 300s default.
    # One-way tests still end via the watchdog well before this.
    monkeypatch.setattr(openai_bridge.plivo_stream, "CALL_MAX_DURATION_S", 1)

    def _install(oa_ws):
        async def fake_connect(*a, **kw):
            return oa_ws
        monkeypatch.setattr(openai_bridge, "_connect", fake_connect)

    return _install


def _session(oa_ws) -> dict:
    """The session.update frame — the first thing sent after connecting."""
    return json.loads(oa_ws.sent[0])


async def test_a_one_way_call_delivers_its_message_and_ends(bridged):
    """The expensive one. Nothing but the watchdog ends a one-way call: the
    model gets no turn-end signal because we never forward the lead's audio."""
    oa = _FakeOpenAIWS([_agent_transcript("Namaste."), _audio_delta(), _response_done()])
    bridged(oa)
    plivo = _FakePlivoWS()

    outcome = await asyncio.wait_for(
        openai_bridge.bridge(plivo, agent_id="ignored", lead_id="lead-1",
                             language="te", one_way=True),
        timeout=5,
    )

    assert outcome["status"] == "done"
    assert [t.text for t in outcome["transcript"]] == ["Namaste."]
    assert any(m["event"] == "playAudio" for m in plivo.sent)


async def test_a_one_way_session_disables_turn_detection(bridged):
    """A one-way call must never auto-respond to the lead. The bridge also
    never forwards their audio, but belt and braces: with turn detection on,
    the model would try to take turns against silence."""
    oa = _FakeOpenAIWS([_audio_delta()])
    bridged(oa)

    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=True),
        timeout=5,
    )

    session = _session(oa)["session"]
    assert session["audio"]["input"]["turn_detection"] is None
    assert "tools" not in session, "a one-way call has nothing to call tools for"


async def test_the_session_pins_mulaw_both_directions(bridged):
    """Plivo is told the stream is mu-law 8kHz. Anything else is misframed and
    the lead hears noise — the same rule the ElevenLabs path follows."""
    oa = _FakeOpenAIWS([_audio_delta()])
    bridged(oa)

    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=True),
        timeout=5,
    )

    audio = _session(oa)["session"]["audio"]
    assert audio["input"]["format"]["type"] == "audio/pcmu"
    assert audio["output"]["format"]["type"] == "audio/pcmu"


async def test_the_script_is_given_as_content_not_words_to_recite(bridged):
    """A script written in English must be CONVEYED in Telugu, not read out in
    English. The sibling hit exactly this: leads got an English call from a
    Telugu-first agent because the prompt said 'say this'."""
    oa = _FakeOpenAIWS([_audio_delta()])
    bridged(oa)

    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=True,
                             dynamic_variables={"script": "Our new course starts Monday."}),
        timeout=5,
    )

    instructions = _session(oa)["session"]["instructions"]
    assert "Our new course starts Monday." in instructions
    assert "never read English text aloud" in instructions.lower() or \
           "convey its meaning" in instructions.lower()


async def test_the_lead_name_reaches_the_prompt(bridged):
    oa = _FakeOpenAIWS([_audio_delta()])
    bridged(oa)

    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=True,
                             dynamic_variables={"lead_name": "Asha"}),
        timeout=5,
    )

    assert "Asha" in _session(oa)["session"]["instructions"]


async def test_no_api_key_fails_cleanly_without_dialling(monkeypatch):
    """Same contract as the ElevenLabs bridge: return a failed outcome rather
    than raising into the caller's finally."""
    monkeypatch.setattr(openai_bridge, "OPENAI_API_KEY", None)
    outcome = await openai_bridge.bridge(
        _FakePlivoWS(), agent_id="x", lead_id="l", language="te", one_way=True
    )
    assert outcome["status"] == "failed"
    assert outcome["turns"] == 0


async def test_a_one_way_call_cues_the_model_to_speak(bridged):
    """With turn detection off, nothing else will: no lead audio is forwarded,
    so without an explicit response.create the model sits silent until the
    watchdog gives up and the lead hears a dead line."""
    oa = _FakeOpenAIWS([_audio_delta()])
    bridged(oa)

    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=True),
        timeout=5,
    )

    assert json.loads(oa.sent[1]) == {"type": "response.create"}


class _FailingOpenAIWS(_FakeOpenAIWS):
    """Speaks, completes a turn, then the socket dies mid-call."""

    def __aiter__(self):
        async def _gen():
            for m in self._messages:
                yield m
            raise RuntimeError("socket died mid-call")
        return _gen()


async def test_a_mid_call_failure_still_returns_the_turns_it_collected(bridged):
    """THE contract. A conversation that really happened must not be recorded
    as zero turns — that used to mark the lead 'failed' AND burn a retry on a
    lead who had already been spoken to. Populated in place, so the caller's
    own dict sees it even though the call fell over."""
    oa = _FailingOpenAIWS([_agent_transcript("Namaste."), _response_done(),
                           _audio_delta()])
    bridged(oa)
    caller_owned: dict = {"status": "failed", "turns": 0, "transcript": [],
                          "conversation_id": None}

    returned = await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=True, outcome=caller_owned),
        timeout=5,
    )

    assert returned is caller_owned
    assert caller_owned["turns"] == 1
    assert [t.text for t in caller_owned["transcript"]] == ["Namaste."]
    # It did NOT run its course, so it must not look like a completed call.
    assert caller_owned["status"] == "failed"
    assert caller_owned["conversation_id"] is None


async def test_speech_cut_off_before_response_done_is_still_transcribed(bridged):
    """Transcript deltas only become a turn at response.done. If the socket
    dies first, the words the lead actually heard would otherwise be thrown
    away — the same "a call that happened must not be recorded as nothing"
    rule, one level down."""
    oa = _FailingOpenAIWS([_audio_delta(), _agent_transcript("Namaste, "),
                           _agent_transcript("Digital Brolly nunchi.")])
    bridged(oa)

    outcome = await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=True),
        timeout=5,
    )

    assert [t.text for t in outcome["transcript"]] == ["Namaste, Digital Brolly nunchi."]


# ── the prompt's compliance rules ────────────────────────────────────────────
# CLAUDE.md requires the AI disclosure to be enforced by a test rather than by
# review. On this backend nothing utters a fixed string — the model speaks, and
# the instruction is the only thing making it disclose — so the instruction is
# what has to be pinned.


def test_the_prompt_demands_the_ai_disclosure_first():
    instructions = openai_prompts.one_way_instructions("Asha", "Course starts Monday.")
    lowered = instructions.lower()
    assert "ai" in lowered and "first sentence" in lowered
    assert "not optional" in lowered


def test_the_prompt_forbids_hindi():
    """The sibling's model drifted into Hindi mid-call to Telugu leads."""
    assert "Never use Hindi" in openai_prompts.one_way_instructions(None, None)


def test_the_prompt_survives_a_campaign_with_no_script_or_name():
    """A one-way campaign whose script is blank still has to produce a usable
    persona, not a prompt with a dangling 'MESSAGE TO DELIVER:' label."""
    instructions = openai_prompts.one_way_instructions(None, "   ")
    assert "MESSAGE TO DELIVER" not in instructions
    assert "You are calling" not in instructions


# ── the language register: 'te' vs 'tinglish' must actually differ ──────────
#
# app/languages.py defines two distinct registers with two distinct style
# texts. Both used to route through one hardcoded _LANGUAGE_RULE that
# happened to permit English mixing — so pure Telugu wrongly invited mixing,
# and Tinglish was indistinguishable from it. The operator picked two
# different things and got one.

def test_pure_telugu_does_not_invite_english_mixing():
    instructions = openai_prompts.one_way_instructions(
        None, None, language_style=languages.style("te")
    )
    lowered = instructions.lower()
    assert "do not switch to english" in lowered
    assert "may mix" not in lowered and "keep these" not in lowered


def test_tinglish_keeps_everyday_english_words_in_english():
    instructions = openai_prompts.one_way_instructions(
        None, None, language_style=languages.style("tinglish")
    )
    lowered = instructions.lower()
    assert "course" in lowered and "fees" in lowered and "certificate" in lowered


def test_pure_telugu_and_tinglish_produce_different_instructions():
    te = openai_prompts.one_way_instructions(None, None, language_style=languages.style("te"))
    tinglish = openai_prompts.one_way_instructions(
        None, None, language_style=languages.style("tinglish")
    )
    assert te != tinglish


def test_hindi_stays_forbidden_regardless_of_register():
    """REGRESSION guard for a real observed failure (the sibling's model
    drifted into Hindi mid-call) — must survive no matter which register the
    LANGUAGE rule is built from, including the no-style fallback."""
    for style in (None, languages.style("te"), languages.style("tinglish")):
        assert "Never use Hindi" in openai_prompts.one_way_instructions(
            None, None, language_style=style
        )


def test_disclosure_and_meaning_rules_survive_in_both_registers():
    """The other two real-observed-failure rules must not get lost when the
    LANGUAGE rule becomes an input instead of a constant."""
    for style in (languages.style("te"), languages.style("tinglish")):
        instructions = openai_prompts.one_way_instructions(
            "Asha", "Course starts Monday.", language_style=style
        )
        lowered = instructions.lower()
        assert "first sentence" in lowered and "ai" in lowered
        assert "meaning to convey" in lowered or "convey its meaning" in lowered


async def test_the_language_style_reaches_the_session_instructions(bridged):
    """Wiring proof: call_routes._call_language() puts the resolved register's
    style text in dynamic_variables['language_style'] — this backend must
    actually use it rather than a hardcoded rule. A unique marker, not real
    style prose, so this fails for the right reason if the style is ignored
    (the old hardcoded rule already contained real words like 'certificate')."""
    oa = _FakeOpenAIWS([_audio_delta()])
    bridged(oa)

    await asyncio.wait_for(
        openai_bridge.bridge(
            _FakePlivoWS(), agent_id="x", lead_id="l", language="te", one_way=True,
            dynamic_variables={"language_style": "ZZZ-MARKER-STYLE-ZZZ"},
        ),
        timeout=5,
    )

    instructions = _session(oa)["session"]["instructions"]
    assert "ZZZ-MARKER-STYLE-ZZZ" in instructions


# ── two-way: the conversation path ───────────────────────────────────────────
# Two-way adds turn detection, a pinned STT language, lead transcripts, and
# barge-in truncation — the last is a regression guard for a bug the sibling
# has and must not be inherited.


async def test_a_two_way_session_enables_turn_detection(bridged):
    oa = _FakeOpenAIWS([_audio_delta()])
    bridged(oa)
    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False),
        timeout=5,
    )
    td = _session(oa)["session"]["audio"]["input"]["turn_detection"]
    assert td["type"] == "server_vad"


async def test_a_two_way_session_pins_the_transcription_language(bridged):
    """On 8kHz phone audio, auto-detection was observed hearing Telugu as
    Croatian or Urdu. The lead is then transcribed as nonsense and the model
    answers nonsense. Pinning the language is the fix."""
    oa = _FakeOpenAIWS([_audio_delta()])
    bridged(oa)
    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False),
        timeout=5,
    )
    tx = _session(oa)["session"]["audio"]["input"]["transcription"]
    assert tx["language"] == "te"


async def test_the_leads_words_are_recorded_as_lead_turns(bridged):
    oa = _FakeOpenAIWS([
        json.dumps({"type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "ఫీజు ఎంత?"}),
        _agent_transcript("Fees are..."), _response_done(),
    ])
    bridged(oa)
    outcome = await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False),
        timeout=5,
    )
    roles = [(t.role, t.text) for t in outcome["transcript"]]
    assert ("lead", "ఫీజు ఎంత?") in roles


async def test_barge_in_truncates_the_models_belief_about_what_it_said(bridged):
    """REGRESSION for a bug the source material has and we must not inherit.
    Without conversation.item.truncate the model believes it said everything
    it generated, while the lead only heard what had played — every later
    turn then builds on something the lead never heard.

    The real API's speech_started event carries an `item_id` too — it is the
    id of the user item VAD is about to create, NOT the agent's item. Include
    it here (distinct from the agent's), matching the real wire format,
    because a fake event that omits it would hide the exact bug this guards.

    The barge-in is into the SECOND response (an answer): the opening response
    is delivered first (its response.done), because the opening disclosure is
    now protected from barge-in — see
    test_a_barge_in_during_the_opening_disclosure_is_ignored."""
    oa = _FakeOpenAIWS([
        json.dumps({"type": "response.output_audio.delta",
                    "delta": _AUDIO_B64, "item_id": "item-opening"}),
        _response_done(),  # the opening (disclosure) is delivered
        json.dumps({"type": "response.output_audio.delta",
                    "delta": _AUDIO_B64, "item_id": "item-agent-1"}),
        json.dumps({"type": "input_audio_buffer.speech_started",
                    "item_id": "item-user-1"}),
    ])
    bridged(oa)
    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False),
        timeout=5,
    )
    sent = [json.loads(s).get("type") for s in oa.sent]
    assert "conversation.item.truncate" in sent
    assert "response.cancel" in sent


async def test_barge_in_truncate_names_the_current_item(bridged):
    """The truncate has to identify WHICH item to trim, or the model trims the
    wrong turn (or none). The id comes from the audio deltas of the response
    being spoken — NOT from speech_started, whose own item_id names the
    user's upcoming item (confirmed against the OpenAI Realtime API schema:
    "The ID of the user message item that will be created when speech
    stops."). conversation.item.truncate only operates on assistant audio
    items, so naming the user item here would make the API reject the
    truncate outright, silently reintroducing the sibling's unresolved bug.

    The barge-in is into the second response, after the opening is delivered."""
    oa = _FakeOpenAIWS([
        json.dumps({"type": "response.output_audio.delta",
                    "delta": _AUDIO_B64, "item_id": "item-opening"}),
        _response_done(),  # the opening (disclosure) is delivered
        json.dumps({"type": "response.output_audio.delta",
                    "delta": _AUDIO_B64, "item_id": "item-77"}),
        json.dumps({"type": "input_audio_buffer.speech_started",
                    "item_id": "item-user-99"}),
    ])
    bridged(oa)
    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False),
        timeout=5,
    )
    truncate = next(json.loads(s) for s in oa.sent
                    if json.loads(s).get("type") == "conversation.item.truncate")
    assert truncate["item_id"] == "item-77"
    assert truncate["item_id"] != "item-user-99"
    assert truncate["content_index"] == 0
    assert isinstance(truncate["audio_end_ms"], int)


async def test_a_barge_in_during_the_opening_disclosure_is_ignored(bridged):
    """REGRESSION (compliance): the opening agent response carries the legally-
    required AI disclosure. A lead who says "Hello?" while it is still playing
    (before its response.done) must NOT truncate it — no cancel, no truncate —
    or the disclosure is cut off. On a live call this fragmented the transcript
    and fired a false [compliance] alert, because the recorded first agent turn
    was just the pre-disclosure fragment 'ఇది Digital Brolly నుండ'."""
    oa = _FakeOpenAIWS([
        json.dumps({"type": "response.output_audio.delta",
                    "delta": _AUDIO_B64, "item_id": "item-opening"}),
        # No response.done before the barge-in: the disclosure is still being
        # delivered, so the barge-in must be ignored.
        json.dumps({"type": "input_audio_buffer.speech_started",
                    "item_id": "item-user-1"}),
    ])
    bridged(oa)
    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False),
        timeout=5,
    )
    types = [json.loads(s).get("type") for s in oa.sent]
    assert "conversation.item.truncate" not in types, "the disclosure must not be truncated"
    assert "response.cancel" not in types, "the disclosure response must not be cancelled"


async def test_barge_in_after_a_response_with_no_audio_item_does_not_truncate_null(bridged):
    """REGRESSION (Bug A): a response can complete without producing an audio
    item (empty or text-only), leaving last_item_id None. A barge-in after the
    opening is delivered must then SKIP the truncate rather than send item_id=
    null, which the API rejects ('expected a string, but got null')."""
    oa = _FakeOpenAIWS([
        _response_done(),  # opening delivered, but produced no audio item
        json.dumps({"type": "input_audio_buffer.speech_started",
                    "item_id": "item-user-1"}),
    ])
    bridged(oa)
    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False),
        timeout=5,
    )
    truncates = [json.loads(s) for s in oa.sent
                 if json.loads(s).get("type") == "conversation.item.truncate"]
    assert truncates == [], "must not truncate when there is no assistant item"


async def test_barge_in_after_the_response_finished_does_not_cancel(bridged):
    """REGRESSION (Bug C): when the response has already completed (response.done)
    but its audio is still draining on the Plivo side, a barge-in has no active
    response to cancel. The old code sent response.cancel unconditionally, which
    the API rejects ('Cancellation failed: no active response found'). The
    truncate must still fire to correct the model's belief; only the cancel is
    skipped."""
    oa = _FakeOpenAIWS([
        json.dumps({"type": "response.output_audio.delta",
                    "delta": _AUDIO_B64, "item_id": "item-agent-1"}),
        _response_done(),
        json.dumps({"type": "input_audio_buffer.speech_started",
                    "item_id": "item-user-1"}),
    ])
    bridged(oa)
    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False),
        timeout=5,
    )
    types = [json.loads(s).get("type") for s in oa.sent]
    assert "response.cancel" not in types, "no active response to cancel after response.done"
    assert "conversation.item.truncate" in types, "the model's belief must still be corrected"


async def test_a_new_response_items_audio_marks_a_playback_boundary(bridged, monkeypatch):
    """REGRESSION (Bug B): the tool-call flow queues a second response's audio
    back-to-back with the first (no lead turn between). Both share one Plivo
    playback segment but are separate OpenAI items, so a barge-in into the second
    reported the elapsed time across BOTH — overshooting the item the truncate
    names ('Audio content of Nms is already shorter than Mms'). The bridge must
    tell PlivoCall a new item's audio has begun, so its playback clock restarts.

    One boundary is marked between two DISTINCT audio items — not between deltas
    of the same item."""
    boundaries = []
    real = openai_bridge.plivo_stream.PlivoCall.mark_response_boundary

    def spy(self):
        boundaries.append(True)
        return real(self)

    monkeypatch.setattr(openai_bridge.plivo_stream.PlivoCall,
                        "mark_response_boundary", spy)
    oa = _FakeOpenAIWS([
        json.dumps({"type": "response.output_audio.delta",
                    "delta": _AUDIO_B64, "item_id": "item-A"}),
        json.dumps({"type": "response.output_audio.delta",
                    "delta": _AUDIO_B64, "item_id": "item-A"}),   # same item
        json.dumps({"type": "response.output_audio.delta",
                    "delta": _AUDIO_B64, "item_id": "item-B"}),   # new item
    ])
    bridged(oa)
    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False),
        timeout=5,
    )
    assert len(boundaries) == 1, "one boundary between the two distinct items"


async def test_a_two_way_call_also_cues_the_model_to_speak_first(bridged):
    """REGRESSION: this used to rely on server_vad alone, so the model never
    spoke until the LEAD did — backwards for an outbound call, where the AI
    placed the call and the AI-disclosure rule requires the AI's first line to
    be the disclosure. A lead who (reasonably) also waits for the other side
    to speak first got dead air instead of a greeting."""
    oa = _FakeOpenAIWS([_audio_delta()])
    bridged(oa)
    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False),
        timeout=5,
    )
    assert json.loads(oa.sent[1]) == {"type": "response.create"}


# ── the two-way prompt's compliance rules ────────────────────────────────────


def test_two_way_prompt_demands_the_ai_disclosure_first():
    """The disclosure is legally required and is the FIRST spoken line on the
    two-way path exactly as on one-way — nothing softens it."""
    instructions = openai_prompts.two_way_instructions("Asha", "Ask about their goals.")
    lowered = instructions.lower()
    assert "first sentence" in lowered and "ai" in lowered
    assert "not optional" in lowered


def test_two_way_prompt_forbids_hindi_in_every_register():
    for style in (None, languages.style("te"), languages.style("tinglish")):
        assert "Never use Hindi" in openai_prompts.two_way_instructions(
            None, None, language_style=style
        )


def test_two_way_prompt_treats_the_script_as_meaning_not_words_to_recite():
    instructions = openai_prompts.two_way_instructions(
        None, "Our new course starts Monday."
    )
    lowered = instructions.lower()
    assert "our new course starts monday." in lowered
    assert "never read english text aloud" in lowered


def test_two_way_prompt_confines_course_facts_to_the_search_tool():
    """No inventing prices or dates — course facts come only from the tool."""
    instructions = openai_prompts.two_way_instructions(None, None).lower()
    assert "search tool" in instructions
    assert "never invent" in instructions or "do not have that information" in instructions


def test_two_way_prompt_refuses_off_topic_questions():
    """REGRESSION — found on this project's own first live two-way call:
    asked who Virat Kohli was, the model just answered. Confining COURSE
    FACTS to the search tool (the test above) says nothing about a question
    that isn't about the course at all, so nothing stopped the model
    reaching into its own training. This pins the rule that closes that gap."""
    instructions = openai_prompts.two_way_instructions(None, None).lower()
    assert "stay on topic" in instructions
    assert "do not answer" in instructions
    assert "certain of the answer" in instructions


def test_two_way_prompt_bans_outside_knowledge_with_no_named_exceptions():
    """The owner's own words after seeing the first fix: 'it shouldn't
    answer any outside knowledge at all' — not just the categories
    (sports/celebrities/news/politics) the first version of this rule named.
    A category list invites a model to read it as bounding the rule (fine if
    not on the list) or as an implicit 'unless it's simple/harmless'
    exception. This pins that the rule is now an unqualified blanket ban —
    no category list, no severity escape hatch — and would fail if either
    crept back in."""
    instructions = openai_prompts.two_way_instructions(None, None)
    lowered = instructions.lower()
    assert "no exceptions" in lowered
    assert "for any reason" in lowered
    assert "no matter" in lowered and "simple" in lowered
    # The old wording is gone, not just superseded — its presence would mean
    # both versions of the rule are in the prompt at once, muddying it.
    assert "sports" not in lowered
    assert "celebrities" not in lowered


def test_two_way_prompt_survives_a_campaign_with_no_script_or_name():
    instructions = openai_prompts.two_way_instructions(None, "   ")
    assert "GOAL OF THIS CALL" not in instructions
    assert "You are speaking with" not in instructions


def test_two_way_prompt_tells_the_model_to_end_the_call():
    """Without this the model has a tool but no instruction to use it, and
    just keeps the line open after the conversation is naturally over."""
    instructions = openai_prompts.two_way_instructions(None, None).lower()
    assert "end_call" in instructions
    assert "natural close" in instructions


# ── the search tool: course questions answered from RAG, not memory ─────────
# CLAUDE.md's hard rule: search_relevant() is the ONLY function the live tool
# may call. search_permissive() ignores the relevance floor and must never be
# reachable from here.

_SEARCH_TOOL_NAME = "search_course_material"


async def test_a_two_way_session_attaches_the_search_tool(bridged):
    oa = _FakeOpenAIWS([_audio_delta()])
    bridged(oa)
    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False),
        timeout=5,
    )
    names = [t["name"] for t in _session(oa)["session"].get("tools", [])]
    assert _SEARCH_TOOL_NAME in names


# ── end_call: the only way a two-way call hangs up on its own ───────────────
# Unlike the ElevenLabs agent platform, OpenAI Realtime has no built-in
# "End Call" tool — without one of our own, a two-way call just sits open
# after the conversation is over, until the lead hangs up or it runs all the
# way to CALL_MAX_DURATION_S (300s of dead air billed to the lead).

_END_CALL_TOOL_NAME = "end_call"


async def test_a_two_way_session_attaches_the_end_call_tool(bridged):
    oa = _FakeOpenAIWS([_audio_delta()])
    bridged(oa)
    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False),
        timeout=5,
    )
    names = [t["name"] for t in _session(oa)["session"].get("tools", [])]
    assert _END_CALL_TOOL_NAME in names


async def test_calling_end_call_ends_the_call_as_done_not_failed(bridged, monkeypatch):
    """The whole point of the tool: a conversation the model chose to end
    cleanly must be graded 'done', not fall through to max_duration or be
    misread as a failure."""
    monkeypatch.setattr(openai_bridge.plivo_stream, "CALL_MAX_DURATION_S", 60)
    oa = _FakeOpenAIWS([json.dumps({
        "type": "response.done",
        "response": {"output": [{
            "type": "function_call", "name": _END_CALL_TOOL_NAME,
            "call_id": "call_1", "arguments": "{}",
        }]},
    })])
    bridged(oa)

    # 60s of CALL_MAX_DURATION_S is far longer than this timeout — if
    # end_call didn't actually end the call, this would time out instead of
    # returning promptly.
    outcome = await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False),
        timeout=5,
    )

    assert outcome["status"] == "done"


async def test_a_tool_call_is_answered_from_search_relevant(bridged, monkeypatch):
    """The hard rule: the live tool calls search_relevant and nothing else.
    search_permissive ignores the relevance floor and would let the agent
    recite irrelevant course text to a lead as though it were an answer."""
    seen = {}

    async def fake_relevant(query, *a, **kw):
        seen["query"] = query
        return "The fee is 25,000 rupees."

    monkeypatch.setattr(openai_bridge, "search_relevant", fake_relevant)

    def forbidden(*a, **kw):
        raise AssertionError("search_permissive must never be reachable live")

    monkeypatch.setattr(openai_bridge, "search_permissive", forbidden, raising=False)

    oa = _FakeOpenAIWS([json.dumps({
        "type": "response.done",
        "response": {"output": [{
            "type": "function_call", "name": _SEARCH_TOOL_NAME,
            "call_id": "call_1", "arguments": json.dumps({"query": "fees?"}),
        }]},
    })])
    bridged(oa)
    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False),
        timeout=5,
    )

    assert seen["query"] == "fees?"
    outputs = [json.loads(s) for s in oa.sent]
    item = next(o for o in outputs
                if o.get("type") == "conversation.item.create")
    assert "25,000" in json.dumps(item)
    assert any(o.get("type") == "response.create" for o in outputs), (
        "the model must be prompted to speak the tool result"
    )


async def test_a_one_way_call_gets_no_tools(bridged):
    """Nothing to search when nobody can ask."""
    oa = _FakeOpenAIWS([_audio_delta()])
    bridged(oa)
    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=True),
        timeout=5,
    )
    assert "tools" not in _session(oa)["session"]


async def test_malformed_tool_arguments_do_not_crash_the_call(bridged, monkeypatch):
    """Defensive JSON parsing: a malformed arguments string must not blow up
    the reader task and silently kill the call."""
    async def fake_relevant(query, *a, **kw):
        return "fallback answer"

    monkeypatch.setattr(openai_bridge, "search_relevant", fake_relevant)

    oa = _FakeOpenAIWS([json.dumps({
        "type": "response.done",
        "response": {"output": [{
            "type": "function_call", "name": _SEARCH_TOOL_NAME,
            "call_id": "call_1", "arguments": "{not valid json",
        }]},
    })])
    bridged(oa)
    outcome = await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False),
        timeout=5,
    )
    # Must not raise, and the call still records a normal (if empty) outcome.
    assert outcome["status"] in ("done", "failed")
