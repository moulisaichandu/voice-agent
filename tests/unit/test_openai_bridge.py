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
