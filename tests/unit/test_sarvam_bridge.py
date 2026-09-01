"""Unit tests for telephony/sarvam_bridge.py — the Telugu voice backend.

Driven against a fake Sarvam TTS socket and the same fake Plivo socket
PlivoCall is tested with, so no network and no Docker.

What matters most here is what matters for every bridge in this project: the
`outcome` contract is identical across backends, so app/telephony/call_routes.py
dispatches to a function rather than branching, and everything downstream —
call recording, Sheets write-back, slot release, retry accounting — never
learns which backend ran. A bridge that gets that wrong does not fail loudly;
it quietly mis-grades real calls and burns real retries.
"""

import asyncio
import base64
import contextlib
import json
import re

import pytest

from app.telephony import sarvam_bridge

_FRAME = base64.b64encode(b"\x7f" * 160).decode()
_TELUGU = "ఇది కృత్రిమ మేధ ద్వారా చేసే ఆటోమేటెడ్ కాల్. కోర్సు సోమవారం మొదలవుతుంది."


class _FakeSarvamWS:
    def __init__(self, messages: list[str]):
        self._messages = list(messages)
        self.sent: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *e):
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


def _audio_then_final() -> list[str]:
    return [
        json.dumps({"type": "audio", "data": {"audio": _FRAME}}),
        json.dumps({"type": "event", "data": {"event_type": "final"}}),
    ]


@pytest.fixture
def bridged(monkeypatch):
    """A configured bridge with a fake TTS socket and a fake renderer.

    Timers are shortened on the plivo_stream MODULE, which is why the bridge
    imports it as a module rather than importing PlivoCall directly.
    """
    monkeypatch.setattr(sarvam_bridge, "SARVAM_API_KEY", "sk-test")
    monkeypatch.setattr(sarvam_bridge.sarvam_tts, "SARVAM_API_KEY", "sk-test")
    monkeypatch.setattr(sarvam_bridge.plivo_stream, "ONEWAY_SILENCE_TAIL_S", 0)
    monkeypatch.setattr(sarvam_bridge.plivo_stream, "ONEWAY_MAX_SILENT_S", 0)
    monkeypatch.setattr(sarvam_bridge.plivo_stream, "CALL_MAX_DURATION_S", 1)

    def _install(ws, *, rendered=_TELUGU, render_error=None):
        async def fake_connect(*a, **kw):
            return ws

        async def fake_render(script, *, language_style=None):
            if render_error:
                raise render_error
            return rendered

        monkeypatch.setattr(sarvam_bridge.sarvam_tts, "_connect", fake_connect)
        monkeypatch.setattr(sarvam_bridge.sarvam_llm, "render", fake_render)
        return ws

    return _install


async def _run(plivo, **kwargs):
    defaults = dict(agent_id="ignored", lead_id="lead-1", language="te",
                    one_way=True,
                    dynamic_variables={"script": "Our course starts Monday."})
    defaults.update(kwargs)
    return await asyncio.wait_for(sarvam_bridge.bridge(plivo, **defaults), timeout=5)




# ── the one-way call ─────────────────────────────────────────────────────────

async def test_a_one_way_call_delivers_its_message_and_ends(bridged):
    """The expensive one. Nothing but the watchdog ends a one-way call: the
    lead's audio is never forwarded, so no backend signal ever arrives."""
    bridged(_FakeSarvamWS(_audio_then_final()))
    plivo = _FakePlivoWS()

    outcome = await _run(plivo)

    assert outcome["status"] == "done"
    assert any(m["event"] == "playAudio" for m in plivo.sent)


async def test_the_transcript_is_the_words_that_were_spoken(bridged):
    """call_routes._spoken_disclosure_ok() reads this to check the AI
    disclosure, so it must be the text that actually went to the synthesiser,
    not the operator's English script."""
    bridged(_FakeSarvamWS(_audio_then_final()))

    outcome = await _run(_FakePlivoWS())

    assert [t.role for t in outcome["transcript"]] == ["agent"]
    assert _TELUGU in outcome["transcript"][0].text
    assert outcome["turns"] == 1


async def test_a_call_where_nothing_was_synthesised_records_no_agent_turn(bridged):
    """The transcript is evidence, not intent.

    call_routes._spoken_disclosure_ok() runs has_ai_disclosure() over the first
    agent turn and logs a compliance PASS by staying quiet. If a call whose
    synthesis never produced a single frame still recorded the text we meant to
    say, that check would be certifying words no lead ever heard — and the
    transcript written back to the operator's Sheet would show a message that
    was never delivered."""
    bridged(_FakeSarvamWS([json.dumps(
        {"type": "error", "data": {"message": "invalid speaker", "code": 400}}
    )]))
    plivo = _FakePlivoWS()

    outcome = await _run(plivo)

    assert not any(m.get("event") == "playAudio" for m in plivo.sent)
    assert outcome["transcript"] == []
    assert outcome["turns"] == 0


async def test_the_lead_name_is_greeted_without_a_second_completion(bridged):
    """The rendered body is cached per campaign, so the name is spliced in
    front from a template instead of being rendered per lead."""
    ws = bridged(_FakeSarvamWS(_audio_then_final()))

    outcome = await _run(_FakePlivoWS(), dynamic_variables={
        "script": "Our course starts Monday.", "lead_name": "Asha",
    })

    spoken = json.loads(ws.sent[1])["data"]["text"]
    assert "Asha" in spoken
    assert _TELUGU in spoken
    assert "Asha" in outcome["transcript"][0].text


async def test_a_one_way_call_greets_the_lead_in_telugu_script_too(
        bridged, monkeypatch):
    """Sarvam's Telugu TTS reads a Latin run with English phonetics, so
    "Mouli" came out wrong on a live call. The two-way path resolves the name
    through lead_name.telugu_name(); one-way did not, and every one-way
    transcript still shows "నమస్తే Mouli గారు" (2026-08-18). Same voice, same
    synthesiser, same mispronunciation — so the same fix."""
    ws = bridged(_FakeSarvamWS(_audio_then_final()))

    async def fake_telugu_name(name):
        assert name == "Mouli"
        return "మౌలి"

    monkeypatch.setattr(sarvam_bridge.lead_name, "telugu_name", fake_telugu_name)

    outcome = await _run(_FakePlivoWS(), dynamic_variables={
        "script": "Our course starts Monday.", "lead_name": "Mouli",
    })

    spoken = json.loads(ws.sent[1])["data"]["text"]
    assert "మౌలి" in spoken, "the one-way greeting was not transliterated"
    assert "Mouli" not in spoken, "the Latin spelling still reached the synthesiser"
    assert "మౌలి" in outcome["transcript"][0].text


async def test_a_one_way_greeting_survives_the_name_lookup_failing(
        bridged, monkeypatch):
    """A mispronounced name is a blemish; a missing greeting is a broken call.
    The two-way path already degrades to the written form — one-way must not
    acquire a new way to lose its opening."""
    ws = bridged(_FakeSarvamWS(_audio_then_final()))

    async def boom(name):
        raise RuntimeError("translation backend down")

    monkeypatch.setattr(sarvam_bridge.lead_name, "telugu_name", boom)

    await _run(_FakePlivoWS(), dynamic_variables={
        "script": "Our course starts Monday.", "lead_name": "Mouli",
    })

    spoken = json.loads(ws.sent[1])["data"]["text"]
    assert "Mouli" in spoken, "the greeting was dropped instead of degrading"


async def test_the_language_style_reaches_the_renderer(bridged, monkeypatch):
    """te and tinglish are different registers and must not share a render."""
    seen = {}

    async def capture(script, *, language_style=None):
        seen["style"] = language_style
        return _TELUGU

    bridged(_FakeSarvamWS(_audio_then_final()))
    monkeypatch.setattr(sarvam_bridge.sarvam_llm, "render", capture)

    await _run(_FakePlivoWS(), dynamic_variables={
        "script": "Our course starts Monday.",
        "language_style": "Mix in English words.",
    })

    assert seen["style"] == "Mix in English words."


# ── the outcome contract, shared with every other backend ────────────────────

async def test_a_render_failure_yields_a_failed_outcome_not_an_exception(bridged):
    """call_routes passes `outcome` IN and reads it in a finally. A bridge that
    raises instead of populating it hands the caller a call that happened and
    no record of it."""
    bridged(_FakeSarvamWS([]),
            render_error=sarvam_bridge.sarvam_llm.SarvamRenderFailed("boom"))

    outcome = await _run(_FakePlivoWS())

    assert outcome["status"] == "failed"
    assert outcome["turns"] == 0


async def test_the_caller_s_outcome_dict_is_populated_in_place(bridged):
    """_finalise_call reads the dict it passed in, never the return value, so
    a mid-call exception still yields whatever was collected."""
    bridged(_FakeSarvamWS(_audio_then_final()))
    outcome: dict = {}

    returned = await _run(_FakePlivoWS(), outcome=outcome)

    assert returned is outcome
    assert set(outcome) == {"status", "turns", "transcript", "conversation_id"}


async def test_conversation_id_is_none_like_the_other_non_elevenlabs_backend(bridged):
    """Sarvam has no conversation objects. calls.el_conversation_id is
    nullable, and _finalise_call already records without one."""
    bridged(_FakeSarvamWS(_audio_then_final()))
    outcome = await _run(_FakePlivoWS())
    assert outcome["conversation_id"] is None


async def test_no_api_key_fails_cleanly_without_dialling(bridged, monkeypatch):
    """Same contract as both other bridges: return a failed outcome rather
    than raising into the caller's finally."""
    bridged(_FakeSarvamWS([]))
    monkeypatch.setattr(sarvam_bridge, "SARVAM_API_KEY", None)

    outcome = await _run(_FakePlivoWS())

    assert outcome["status"] == "failed"
    assert outcome["turns"] == 0


# ── two-way is not built on this backend yet ─────────────────────────────────

async def test_a_two_way_call_is_refused_rather_than_sitting_in_silence(bridged):
    """Refused twice upstream already (campaign creation and preflight); this
    is the third and cheapest guard, for a campaign that slipped past both.

    Without it the call connects, never speaks, never listens, bills the full
    CALL_MAX_DURATION_S of silence, and is then recorded as a SUCCESSFUL
    zero-turn call because max_duration counts as a clean exit."""
    bridged(_FakeSarvamWS(_audio_then_final()))
    plivo = _FakePlivoWS()

    outcome = await _run(plivo, one_way=False)

    assert outcome["status"] == "failed"
    assert outcome["turns"] == 0
    assert not any(m.get("event") == "playAudio" for m in plivo.sent)


# ── SpokenLedger: what the lead actually heard ───────────────────────────────
#
# THE piece of two-way this backend cannot borrow. On OpenAI Realtime the model
# is told what the lead heard via conversation.item.truncate, and the platform
# fixes its own history. Here the history is ours, so if the lead interrupts
# after two sentences of a five-sentence answer, WE have to forget the other
# three — otherwise every later turn builds on words nobody heard, and the
# agent starts referring back to things it never actually said.

def _ledger(*pairs):
    led = sarvam_bridge.SpokenLedger()
    for text, ms in pairs:
        led.add(text, ms)
    return led


def test_a_turn_played_to_completion_is_remembered_whole():
    led = _ledger(("One.", 1000), ("Two.", 1000))
    assert led.heard_within(2000) == "One. Two."


def test_playing_past_the_end_still_returns_everything():
    """played_ms() is wall-clock against a playback estimate, so it can drift
    slightly past the true total. That must not drop the last sentence."""
    assert _ledger(("One.", 1000)).heard_within(99999) == "One."


def test_an_interruption_drops_the_sentences_that_never_played():
    led = _ledger(("One.", 1000), ("Two.", 1000), ("Three.", 1000))
    assert led.heard_within(2000) == "One. Two."


def test_a_sentence_only_half_played_is_dropped_not_claimed():
    """The conservative direction, deliberately. If we keep a half-heard
    sentence, the agent may refer back to something the lead never got. If we
    drop it, the worst case is the agent repeating itself — annoying, not
    confusing. Under-claiming is the safe error here."""
    led = _ledger(("One.", 1000), ("Two.", 1000))
    assert led.heard_within(1500) == "One."


def test_an_interruption_before_a_word_played_leaves_nothing():
    """The lead spoke over the very start of the answer. The agent said
    nothing, as far as the conversation is concerned."""
    assert _ledger(("One.", 1000), ("Two.", 1000)).heard_within(0) == ""


def test_the_full_text_is_available_regardless_of_playback():
    """What we MEANT to say, for logs and diagnostics — never for history."""
    led = _ledger(("One.", 1000), ("Two.", 1000))
    assert led.full_text() == "One. Two."
    assert led.heard_within(1000) == "One."


def test_a_sentence_that_never_played_is_not_claimed_even_after_a_played_one():
    """Post-fix sweep: the ends_at > 0 guard only covers the all-zero case.
    When sentence 1 plays (ends_at 2000) and sentence 2's synthesis dies on
    an error frame (0ms), sentence 2 inherits ends_at 2000 — above zero — so
    heard_within claimed words the lead never heard a frame of. The guard
    must be per-sentence duration, not the running total."""
    led = _ledger(("Played.", 2000), ("Silent.", 0))

    assert led.heard_within(10_000) == "Played."


def test_a_ledger_resets_between_turns():
    """Each agent turn measures playback from its own start (PlivoCall's
    mark_response_boundary does the same on the audio side). A ledger carrying
    the previous turn's entries would mis-attribute them to this one."""
    led = _ledger(("Old.", 1000))
    led.reset()
    led.add("New.", 500)
    assert led.heard_within(500) == "New."


# ── two-way ──────────────────────────────────────────────────────────────────

class _FakeSTTWS:
    """The lead's side. Yields *messages*, then goes quiet."""

    def __init__(self, messages):
        self._messages = list(messages)
        self.sent: list[str] = []

    async def send(self, payload):
        self.sent.append(payload)

    async def close(self):
        pass

    def __aiter__(self):
        async def _gen():
            for m in self._messages:
                yield m
                await asyncio.sleep(0)
            await asyncio.Event().wait()
        return _gen()


def _said(text):
    return json.dumps({"type": "data", "data": {"transcript": text}})


def _barge_in():
    return json.dumps({"type": "events", "data": {"signal_type": "START_SPEECH"}})


def _speech_ended():
    return json.dumps({"type": "events", "data": {"signal_type": "END_SPEECH"}})


@pytest.fixture
def two_way(monkeypatch):
    """A two-way bridge with both sockets and the LLM faked."""
    monkeypatch.setattr(sarvam_bridge, "SARVAM_API_KEY", "sk-test")
    monkeypatch.setattr(sarvam_bridge, "SARVAM_TWOWAY_ENABLED", True)
    monkeypatch.setattr(sarvam_bridge.sarvam_tts, "SARVAM_API_KEY", "sk-test")
    monkeypatch.setattr(sarvam_bridge.sarvam_stt, "SARVAM_API_KEY", "sk-test")
    monkeypatch.setattr(sarvam_bridge.plivo_stream, "CALL_MAX_DURATION_S", 1)
    monkeypatch.setattr(sarvam_bridge.plivo_stream, "PLIVO_GREETING_GRACE_MS", 0)
    seen: dict = {"histories": [], "queries": []}

    def _install(stt_messages, *, replies=None, tts_messages=None):
        tts_ws = _FakeSarvamWS(tts_messages if tts_messages is not None
                               else _audio_then_final() * 8)
        stt_ws = _FakeSTTWS(stt_messages)

        async def fake_tts_connect(*a, **kw):
            return tts_ws

        async def fake_stt_connect(*a, **kw):
            return stt_ws

        queued = list(replies or [])

        async def fake_turn(history):
            seen["histories"].append([dict(m) for m in history])
            if queued:
                return queued.pop(0)
            return sarvam_bridge.conversation_llm.LLMReply(text="Ok.", tool_calls=[])

        async def fake_render(script, *, language_style=None):
            return _TELUGU

        async def fake_search(query, *a, **kw):
            seen["queries"].append(query)
            return "The fee is 25,000 rupees."

        monkeypatch.setattr(sarvam_bridge.sarvam_tts, "_connect", fake_tts_connect)
        monkeypatch.setattr(sarvam_bridge.sarvam_stt, "_connect", fake_stt_connect)
        monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", fake_turn)
        monkeypatch.setattr(sarvam_bridge.sarvam_llm, "render", fake_render)
        monkeypatch.setattr(sarvam_bridge, "search_relevant", fake_search)

        async def no_facts():
            # Deterministic default: the digest fetch would otherwise hit the
            # real test Redis and record its fixed queries into
            # seen["queries"], contaminating every test that asserts on the
            # queries the MODEL sent. Digest tests override this explicitly.
            return ""

        monkeypatch.setattr(sarvam_bridge, "_course_facts", no_facts)
        return seen, tts_ws, stt_ws

    return _install


def _reply(text="", tools=()):
    return sarvam_bridge.conversation_llm.LLMReply(text=text, tool_calls=list(tools))


def _tool(name, **arguments):
    return sarvam_bridge.conversation_llm.ToolCall(
        call_id="c1", name=name, arguments=arguments)


@pytest.mark.parametrize("text", [
    "Bye.", "goodbye", "Good bye!", "Bye bye", "That's all, thank you.",
    "I have no questions.", "బై",
])
def test_explicit_lead_goodbyes_are_detected(text):
    assert sarvam_bridge._is_lead_goodbye(text)


@pytest.mark.parametrize("text", ["okay", "thanks", "by the way, fees?", "hi"])
def test_non_goodbye_transcripts_are_not_treated_as_hangup(text):
    assert not sarvam_bridge._is_lead_goodbye(text)


async def test_lead_goodbye_hangs_up_without_waiting_for_the_model(two_way):
    seen, _tts, _stt = two_way([_said("Bye")])

    outcome = await _run(_FakePlivoWS(), one_way=False)

    assert outcome["status"] == "done"
    assert not any(t.role == "agent" and t.text == "Ok."
                   for t in outcome["transcript"])
    assert not seen["histories"], "goodbye must not spend an LLM turn"


def test_lead_goodbye_is_a_clean_exit():
    from app.telephony import plivo_stream
    assert "sarvam_lead_goodbye" in plivo_stream._CLEAN_EXITS


async def test_the_agent_speaks_first_on_a_two_way_call(two_way):
    """The AI placed the call, so the disclosure cannot wait for the lead to
    say something — India telecom rules put it in the first line spoken."""
    two_way([])
    outcome = await _run(_FakePlivoWS(), one_way=False)

    assert outcome["transcript"][0].role == "agent"
    assert _TELUGU in outcome["transcript"][0].text


async def test_the_leads_words_are_recorded_as_lead_turns(two_way):
    two_way([_said("Fee enta?")])
    outcome = await _run(_FakePlivoWS(), one_way=False)

    roles = [(t.role, t.text) for t in outcome["transcript"]]
    assert ("lead", "Fee enta?") in roles


async def test_the_lead_gets_an_answer(two_way):
    two_way([_said("Fee enta?")], replies=[_reply(text="Iravai aidu vela.")])
    outcome = await _run(_FakePlivoWS(), one_way=False)

    assert any(t.role == "agent" and "Iravai aidu vela." in t.text
               for t in outcome["transcript"])


async def test_the_transcriber_is_opened_for_a_two_way_call(two_way):
    """One-way never forwards the lead's media; two-way must, or the agent is
    holding a conversation with someone it cannot hear."""
    _seen, _tts, stt_ws = two_way([])

    await _run(_FakePlivoWS(), one_way=False)

    assert stt_ws is not None


# ── RAG ──────────────────────────────────────────────────────────────────────

async def test_a_course_question_is_answered_from_the_documents(two_way):
    """CLAUDE.md's hard rule: search_relevant() is the ONLY function the live
    path may call. search_permissive ignores the relevance floor and would let
    the agent recite irrelevant course text to a lead as though it were an
    answer."""
    seen, _tts, _stt = two_way(
        [_said("Fee enta?")],
        replies=[_reply(tools=[_tool("search_course_material", query="fees")]),
                 _reply(text="Fee iravai aidu vela.")],
    )

    outcome = await _run(_FakePlivoWS(), one_way=False)

    assert seen["queries"] == ["fees"]
    assert any("iravai aidu vela" in t.text for t in outcome["transcript"])
    tool_messages = [m for h in seen["histories"] for m in h
                     if m.get("role") == "tool"]
    assert any("25,000" in m["content"] for m in tool_messages)


async def test_search_permissive_is_never_reachable(two_way, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("search_permissive must never be reachable live")

    two_way([_said("Fee enta?")],
            replies=[_reply(tools=[_tool("search_course_material", query="fees")]),
                     _reply(text="Ok.")])
    monkeypatch.setattr(sarvam_bridge, "search_permissive", forbidden,
                        raising=False)

    await _run(_FakePlivoWS(), one_way=False)


async def test_a_slow_course_lookup_does_not_hold_the_lead_in_silence(
    two_way, monkeypatch,
):
    """search_relevant() has no deadline of its own — an 8s embed timeout with
    one SDK retry is ~16s — and the two-way silence watchdog will not save the
    lead either, since an in-flight reply counts as activity. The voice call
    site bounds it instead, and on a timeout the model is handed the same note
    a genuine miss returns, so the agent still says something instead of
    holding the line."""
    monkeypatch.setattr(sarvam_bridge, "RAG_VOICE_DEADLINE_S", 0.05)
    seen, _tts, _stt = two_way(
        [_said("Fee enta?")],
        replies=[_reply(tools=[_tool("search_course_material", query="fees")]),
                 _reply(text="Aa vivaram naa daggara ledu.")],
    )

    async def glacial(query, *a, **kw):
        await asyncio.sleep(5)
        return "never arrives"

    monkeypatch.setattr(sarvam_bridge, "search_relevant", glacial)

    outcome = await _run(_FakePlivoWS(), one_way=False)

    tool_messages = [m for h in seen["histories"] for m in h
                     if m.get("role") == "tool"]
    assert tool_messages, "the model must still be handed a tool result"
    assert all(m["content"] == sarvam_bridge.NO_MATERIAL_NOTE
               for m in tool_messages), (
        "a bespoke 'the search timed out' string would be read out verbatim "
        "by the synthesiser instead of going through the prompt's own rule "
        "for a genuine miss"
    )
    assert any(t.role == "agent" and "naa daggara ledu" in t.text
               for t in outcome["transcript"]), (
        "the agent must still say something, not sit in silence"
    )


async def test_a_failed_course_lookup_still_allows_a_spoken_answer(
    two_way, monkeypatch,
):
    """A RAG outage must become a safe fallback, not a silent call.

    The direct Sarvam path used to let embedding/database exceptions escape
    from _run_tools. _reply caught them at the outer boundary, so the lead's
    question was recorded but the answer turn disappeared entirely.
    """
    seen, _tts, _stt = two_way(
        [_said("Fee enta?")],
        replies=[
            _reply(tools=[_tool("search_course_material", query="fees")]),
            _reply(text="Aa vivaram ippudu naa daggara ledu."),
        ],
    )

    async def broken_search(query, *args, **kwargs):
        raise RuntimeError("embedding service unavailable")

    monkeypatch.setattr(sarvam_bridge, "search_relevant", broken_search)

    outcome = await _run(_FakePlivoWS(), one_way=False)

    tool_messages = [m for h in seen["histories"] for m in h
                     if m.get("role") == "tool"]
    assert tool_messages
    assert tool_messages[-1]["content"] == sarvam_bridge.NO_MATERIAL_NOTE
    assert any(
        t.role == "agent" and "naa daggara ledu" in t.text
        for t in outcome["transcript"]
    )


# ── ending the call ──────────────────────────────────────────────────────────

async def test_the_agent_can_end_the_call_cleanly(two_way):
    """'sarvam_end_call' must be in plivo_stream._CLEAN_EXITS. Without it every
    conversation the agent ended properly grades as FAILED, marks the lead
    failed, and burns a retry on someone who already said goodbye."""
    two_way([_said("Thanks, bye.")],
            replies=[_reply(text="Dhanyavadalu.", tools=[_tool("end_call")])])

    outcome = await _run(_FakePlivoWS(), one_way=False)

    assert outcome["status"] == "done"


async def test_the_farewell_is_spoken_when_the_model_ends_the_call(two_way):
    """CONFIRMED on two live calls on 2026-08-07: every call the model ended
    itself (as opposed to the silence watchdog) hung up in total silence right
    after the lead's last word — reading exactly like "the agent stopped
    answering". end_call's own tool description tells the model to send its
    farewell "together with" the tool call, and unlike the search tool's
    filler there is no next round for that text to arrive in, so it must be
    spoken rather than treated as commentary to suppress.

    Said as "Okay, thank you." on purpose, not an explicit goodbye: this must
    reach the model rather than exit through _is_lead_goodbye's fast path, the
    same way the live calls did — Sarvam's STT transcribed a spoken "bye" as
    "బాయ్", a spelling _GOODBYE_PHRASES does not recognise, so it fell through
    to the model exactly like this."""
    two_way([_said("Okay, thank you.")],
            replies=[_reply(text="Dhanyavadalu, namaskaram.",
                            tools=[_tool("end_call")])])

    outcome = await _run(_FakePlivoWS(), one_way=False)

    spoken = [t.text for t in outcome["transcript"] if t.role == "agent"]
    assert any("Dhanyavadalu" in t for t in spoken), (
        f"the model's farewell was dropped as filler: {spoken}"
    )


async def test_a_canned_farewell_is_spoken_when_the_model_gives_none(two_way):
    """CLAUDE.md: measured against the live API, the model sometimes calls
    end_call with no farewell text attached at all (not suppressed — never
    generated). That must not read as the earlier "farewell dropped" bug —
    there is nothing to speak that the code failed to pass through — but the
    lead still deserves something other than dead air on disconnect."""
    two_way([_said("Okay, thank you.")],
            replies=[_reply(text="", tools=[_tool("end_call")])])

    outcome = await _run(_FakePlivoWS(), one_way=False)

    spoken = [t.text for t in outcome["transcript"] if t.role == "agent"]
    assert spoken and spoken[-1] == sarvam_bridge._FALLBACK_FAREWELL


# ── barge-in ─────────────────────────────────────────────────────────────────

async def test_barge_in_forgets_what_the_lead_never_heard():
    """THE bug the previous plan warned about inheriting.

    OpenAI Realtime is told what the lead heard via conversation.item.truncate
    and repairs its own history. Here the history is ours: if we keep the whole
    answer after an interruption, every later turn is built on words nobody
    heard, and the agent starts referring back to things it never said.

    Tested on an ordinary answer rather than the opening, because the opening
    is barge-in PROTECTED until it has played — see
    test_the_opening_stays_protected_until_it_has_actually_played."""
    from app.db.models import TranscriptTurn

    call = sarvam_bridge.plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="l", one_way=False, protect_opening=False)
    turns = []
    convo = sarvam_bridge._Conversation(
        call, lead_id="l", system_prompt="s", turns=turns)

    # An answer of two sentences, of which only the first ever played.
    convo._agent_turn_index = 0
    turns.append(TranscriptTurn(role="agent", text="One. Two."))
    convo.history.append({"role": "assistant", "content": "One. Two."})
    convo.ledger.add("One.", 1000)
    convo.ledger.add("Two.", 1000)

    await convo.on_barge_in()

    claimed = " ".join(m.get("content") or "" for m in convo.history
                       if m.get("role") == "assistant")
    assert "Two." not in claimed, (
        "the history still claims a sentence the lead never heard"
    )
    assert all("Two." not in t.text for t in turns)


# ── playback gaps: dead air in the middle of the agent's own reply ──────────
#
# say() synthesizes one sentence at a time, sequentially — each is a fresh
# WebSocket round trip to Sarvam. If a short sentence's synthesis takes
# longer than its predecessor took to actually play out on the phone line,
# Plivo's queue drains before the next chunk arrives: dead air mid-reply.
# PlivoCall already tracks play_end (when queued audio finishes playing);
# these tests prove say() actually reads it. See
# docs/superpowers/specs/2026-08-08-playback-gap-diagnostics-design.md.

async def test_a_playback_gap_mid_response_is_logged(caplog):
    """A frame arriving well after the previous one finished playing is a
    real gap the lead heard, and it must show up in the logged total."""
    caplog.set_level("INFO")
    call = sarvam_bridge.plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="gap-test", one_way=False, protect_opening=False)
    turns = []
    convo = sarvam_bridge._Conversation(
        call, lead_id="gap-test", system_prompt="s", turns=turns)

    class _GappyTTS:
        async def speak(self, text):
            yield _FRAME, 20
            await asyncio.sleep(0.1)  # far longer than _FRAME's ~20ms play-out
            yield _FRAME, 20

    # No sentence-ending punctuation, so _split_sentences keeps this as ONE
    # sentence — one call to speak(), isolating exactly the gap under test.
    await convo.say(_GappyTTS(), "hello there")

    lines = [r.message for r in caplog.records if "playback gaps" in r.message]
    assert lines, "no playback-gap line was logged"
    assert "lead=gap-test" in lines[0]
    assert "sentences=1" in lines[0]
    match = re.search(r"total=(\d+\.\d+)ms", lines[0])
    assert match, f"couldn't parse total from: {lines[0]}"
    assert float(match.group(1)) > 50, f"expected a gap of at least 50ms: {lines[0]}"


async def test_fast_delivery_logs_zero_gap(caplog):
    """Frames arriving well within the previous frame's play-out window —
    the common case tonight's real calls already showed — must not be
    reported as a gap."""
    caplog.set_level("INFO")
    call = sarvam_bridge.plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="fast-test", one_way=False, protect_opening=False)
    turns = []
    convo = sarvam_bridge._Conversation(
        call, lead_id="fast-test", system_prompt="s", turns=turns)

    class _FastTTS:
        async def speak(self, text):
            yield _FRAME, 20
            yield _FRAME, 20
            yield _FRAME, 20

    await convo.say(_FastTTS(), "hello there")

    lines = [r.message for r in caplog.records if "playback gaps" in r.message]
    assert lines, "no playback-gap line was logged"
    assert "total=0.0ms max=0.0ms" in lines[0], (
        f"fast delivery must not read as a gap: {lines[0]}"
    )
    assert "sentences=1" in lines[0]


async def test_gaps_accumulate_across_sentence_boundaries(caplog):
    """THE actual production hypothesis this feature exists to investigate:
    a gap occurring BETWEEN sentences, where each sentence is its own fresh
    synthesis round trip to Sarvam. Both tests above use text with no
    sentence-ending punctuation, so _split_sentences keeps it as ONE
    sentence and only one speak() call ever happens — neither exercises
    that at all.

    Pins the two things a manual read of the code confirmed but no test
    asserted: `total` accumulates across sentence boundaries, and `max`
    tracks the single largest gap rather than their sum.
    """
    caplog.set_level("INFO")
    call = sarvam_bridge.plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="multi-test", one_way=False, protect_opening=False)
    turns = []
    convo = sarvam_bridge._Conversation(
        call, lead_id="multi-test", system_prompt="s", turns=turns)

    class _StaggeredTTS:
        """One frame per sentence. The 1st call never stalls — the very
        first frame overall is never counted as a gap (see say()'s
        spoke_a_frame gate), so delaying it would prove nothing. The 2nd
        and 3rd calls stall by different amounts, so a gap can be pinned to
        a specific sentence rather than an aggregate."""

        def __init__(self, delays_s):
            self._delays = list(delays_s)
            self.calls = 0

        async def speak(self, text):
            self.calls += 1
            if self.calls > 1:
                await asyncio.sleep(self._delays[self.calls - 2])
            yield _FRAME, 20

    await convo.say(_StaggeredTTS([0.3, 0.1]), "One. Two. Three.")

    lines = [r.message for r in caplog.records if "playback gaps" in r.message]
    assert lines, "no playback-gap line was logged"
    assert "lead=multi-test" in lines[0]
    assert "sentences=3" in lines[0], "all three sentences must be attempted"

    total_match = re.search(r"total=(\d+\.\d+)ms", lines[0])
    max_match = re.search(r"max=(\d+\.\d+)ms", lines[0])
    assert total_match and max_match, f"couldn't parse gaps from: {lines[0]}"
    total_ms = float(total_match.group(1))
    max_ms = float(max_match.group(1))

    # The two stalls (300ms, 100ms) inject ~400ms of dead air between them.
    # Each measured gap runs a little under its stall (play_end was already
    # a little ahead when the stall began) and real scheduling adds some
    # jitter either way, so use thresholds with headroom rather than an
    # exact match.
    assert total_ms > 200, (
        f"total must accumulate BOTH gaps, not just the larger one: {lines[0]}"
    )
    # If max were computed as the sum instead of the largest single gap, it
    # would land near total_ms (~350-400ms). Pinning it well under that, and
    # comfortably above what the 100ms stall alone could produce (~80ms), is
    # what actually proves max() over +=.
    assert 150 < max_ms < 340, (
        f"max must track the single larger gap (~300ms), not the sum of "
        f"both (~400ms) nor the smaller gap alone (~80ms): {lines[0]}"
    )


async def test_no_gap_line_when_nothing_was_spoken(caplog):
    """A TTS response that produces no audio frames at all (e.g. Sarvam
    returned an error before any audio chunk) has nothing to report.

    A different guard from _log_turn_latency's, not the same one:
    _log_turn_latency still logs an unanswered turn (with " NOTHING SPOKEN"
    appended) — its early return is only for when nobody was waiting on an
    answer at all. _log_playback_gaps is called only when spoke_a_frame is
    True, i.e. something was actually played, which is what this test
    checks."""
    caplog.set_level("INFO")
    call = sarvam_bridge.plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="silent-test", one_way=False, protect_opening=False)
    turns = []
    convo = sarvam_bridge._Conversation(
        call, lead_id="silent-test", system_prompt="s", turns=turns)

    class _SilentTTS:
        async def speak(self, text):
            return
            yield  # pragma: no cover - makes this an async generator

    await convo.say(_SilentTTS(), "hello there")

    lines = [r.message for r in caplog.records if "playback gaps" in r.message]
    assert not lines, f"a gap line was logged for a turn that never spoke: {lines}"


# ── the gate ─────────────────────────────────────────────────────────────────

async def test_two_way_is_refused_while_its_flag_is_off(two_way, monkeypatch):
    """Refused upstream twice already; this is the third and cheapest guard."""
    two_way([])
    monkeypatch.setattr(sarvam_bridge, "SARVAM_TWOWAY_ENABLED", False)
    plivo = _FakePlivoWS()

    outcome = await _run(plivo, one_way=False)

    assert outcome["status"] == "failed"
    assert outcome["turns"] == 0
    assert not any(m.get("event") == "playAudio" for m in plivo.sent)


# ── the two-way silence watchdog ─────────────────────────────────────────────
#
# Measured against the live API: told to say goodbye AND call end_call, the
# model says the goodbye and does not call the tool — 0/3, even when instructed
# to do it explicitly in that turn. The best prompt variant reached 2/3, and
# only by weakening the OUTPUT FORMAT rule that stops the synthesiser reading
# preambles aloud to the lead. That is not a trade worth making.
#
# So ending the call does not depend on the model. If nobody has spoken for
# TWOWAY_MAX_SILENT_S, the call ends by itself — the same shape as
# PlivoCall.oneway_watchdog, and for the same reason: without it the line sits
# open to CALL_MAX_DURATION_S, five minutes of billed airtime at a lead who
# already said goodbye.
#
# Scoped to this bridge on purpose. The ElevenLabs two-way path is in
# production and must not acquire a new way to hang up on someone.

async def test_a_two_way_call_ends_itself_when_both_sides_go_quiet(two_way,
                                                                   monkeypatch):
    monkeypatch.setattr(sarvam_bridge, "TWOWAY_MAX_SILENT_S", 0.2)
    monkeypatch.setattr(sarvam_bridge.plivo_stream, "CALL_MAX_DURATION_S", 30)
    two_way([_said("థాంక్స్, బై.")], replies=[_reply(text="ధన్యవాదాలు.")])

    outcome = await asyncio.wait_for(
        sarvam_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="lead-1",
                             language="te", one_way=False,
                             dynamic_variables={"script": "Course starts Monday."}),
        timeout=10,
    )

    assert outcome["status"] == "done", (
        "a conversation that ran its course is not a failed call — grading it "
        "failed would mark the lead failed and burn a retry"
    )


async def test_the_call_is_not_cut_off_while_someone_is_still_talking(two_way,
                                                                     monkeypatch):
    """The watchdog must measure SILENCE, not elapsed time. A lead who is
    mid-question when it fires is hung up on."""
    monkeypatch.setattr(sarvam_bridge, "TWOWAY_MAX_SILENT_S", 60)
    monkeypatch.setattr(sarvam_bridge.plivo_stream, "CALL_MAX_DURATION_S", 1)
    two_way([_said("ఫీజు ఎంత?")], replies=[_reply(text="ఇరవై అయిదు వేలు.")])

    outcome = await asyncio.wait_for(
        sarvam_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="lead-1",
                             language="te", one_way=False,
                             dynamic_variables={"script": "Course starts Monday."}),
        timeout=10,
    )

    # Ended by max_duration, not by the watchdog — the conversation was active.
    assert any(t.role == "lead" for t in outcome["transcript"])


def test_a_conversation_that_went_quiet_counts_as_a_clean_exit():
    """'twoway_silence' must be in plivo_stream._CLEAN_EXITS. Without it every
    conversation that ended by both sides finishing would be stored 'failed',
    mark its lead failed, and burn a retry on someone who said goodbye."""
    from app.telephony import plivo_stream
    assert "twoway_silence" in plivo_stream._CLEAN_EXITS


async def test_the_watchdog_waits_for_audio_to_finish_PLAYING(two_way, monkeypatch):
    """REGRESSION, found by driving the real vendors end to end.

    TTS delivers audio far faster than real time — an eight-second answer
    arrives from Sarvam in about three. The watchdog measured silence from the
    last frame SENT, so it started counting while the lead still had five
    seconds of speech to hear, and with a short enough timeout it hung up
    mid-sentence on a call that was working perfectly.

    PlivoCall already knows when queued audio finishes playing (play_end, what
    the one-way watchdog drains against). Silence has to be measured from
    there, not from the send loop.
    """
    monkeypatch.setattr(sarvam_bridge, "TWOWAY_MAX_SILENT_S", 0.3)
    monkeypatch.setattr(sarvam_bridge.plivo_stream, "CALL_MAX_DURATION_S", 30)
    two_way([])

    call = sarvam_bridge.plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="l", one_way=False, protect_opening=True)
    convo = sarvam_bridge._Conversation(
        call, lead_id="l", system_prompt="s", turns=[])

    # Everything was sent a while ago, but there is still audio queued to play.
    # audio_seen mirrors what PlivoCall.play() sets for real: this scenario is a
    # call that IS working, which the watchdog must distinguish from one where
    # the synthesiser never produced a frame at all.
    loop = asyncio.get_event_loop()
    convo.last_activity = loop.time() - 60
    call.play_end = loop.time() + 0.6
    call.audio_seen = True

    watch = asyncio.create_task(convo.watch_for_silence())
    await asyncio.sleep(0.35)
    assert not call.stop.is_set(), (
        "hung up while the lead was still listening to queued audio"
    )

    await asyncio.wait_for(watch, timeout=5)
    assert call.stop.is_set(), "should end once the audio has drained"
    assert call.exit_reason == "twoway_silence"


# ── the opening must disclose, script or no script ───────────────────────────
#
# REGRESSION from a real call. app/telephony/call_routes.py only puts `script`
# into dynamic_variables when mode == "oneway", so every TWO-WAY call arrives
# here with no script at all. render("") correctly returns "", compose_spoken()
# then produced nothing but the greeting — "నమస్తే Mouli గారు." — and that is
# what a real lead heard: a call with NO AI disclosure, which India telecom
# rules require in the first line.
#
# The [compliance] check in call_routes caught it after the fact, exactly as
# designed. This is the preventive half.

async def test_a_two_way_call_with_no_script_still_discloses(two_way):
    """The failing case from the live call, pinned."""
    two_way([])

    outcome = await asyncio.wait_for(
        sarvam_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False,
                             dynamic_variables={"lead_name": "Mouli"}),
        timeout=10,
    )

    from app.compliance.disclosure import has_ai_disclosure
    first_agent = next(t for t in outcome["transcript"] if t.role == "agent")
    assert has_ai_disclosure(first_agent.text), (
        f"a lead would hear {first_agent.text!r} with no AI disclosure"
    )


async def test_a_greeting_alone_is_never_spoken_as_the_opening(two_way,
                                                               monkeypatch):
    """The exact shape that went out on a real call: the render produces
    nothing, so the only thing left is the name greeting.

    The contract is NOT "always open the call" — it is "never open it
    non-compliantly". With the renderer returning nothing even for the default
    script, something is badly wrong, and a failed call is strictly better than
    a real person hearing an undisclosed AI. So: either a compliant opening, or
    silence. Never "నమస్తే Mouli గారు." on its own."""
    two_way([])

    async def empty_render(script, *, language_style=None):
        return ""

    monkeypatch.setattr(sarvam_bridge.sarvam_llm, "render", empty_render)

    outcome = await asyncio.wait_for(
        sarvam_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False,
                             dynamic_variables={"lead_name": "Mouli"}),
        timeout=10,
    )

    from app.compliance.disclosure import has_ai_disclosure
    spoken = [t.text for t in outcome["transcript"] if t.role == "agent"]
    assert all(has_ai_disclosure(t) for t in spoken), (
        f"a lead would hear {spoken!r} with no AI disclosure"
    )
    if not spoken:
        assert outcome["status"] == "failed", (
            "a call that said nothing must be recorded as failed, not done"
        )


async def test_a_one_way_call_with_no_script_refuses_rather_than_greeting(bridged):
    """One-way has nothing to say without a script — a campaign cannot even be
    created without one. Speaking a bare greeting and hanging up would be a
    non-compliant call for no purpose."""
    bridged(_FakeSarvamWS(_audio_then_final()), rendered="")

    outcome = await _run(_FakePlivoWS(), dynamic_variables={"lead_name": "Mouli"})

    assert outcome["status"] == "failed"
    assert outcome["turns"] == 0


# ── answering from the documents, not narrating the lookup ───────────────────
#
# From a real call. Asked about the course, the agent said, out loud, twice:
#   "I will look in our documents for the right information."
#   "I am searching our course documents. It will take a moment."
# ...and then that it had found nothing.
#
# Two separate faults. The narration is this one: the model returns filler text
# ALONGSIDE its tool call, and the bridge spoke it. A lookup takes about a
# second — far less than the two extra synthesised turns needed to announce it.

async def test_filler_said_alongside_a_tool_call_is_not_spoken(two_way):
    """The lead should hear the ANSWER, not a commentary on the search."""
    seen, _tts, _stt = two_way(
        [_said("Fee enta?")],
        replies=[
            _reply(text="I will look in our documents. It will take a moment.",
                   tools=[_tool("search_course_material", query="course fees")]),
            _reply(text="Fee is 25,000 rupees."),
        ],
    )

    outcome = await _run(_FakePlivoWS(), one_way=False)

    spoken = [t.text for t in outcome["transcript"] if t.role == "agent"]
    assert not any("look in our documents" in t for t in spoken), (
        f"the agent narrated its own lookup instead of answering: {spoken}"
    )
    assert any("25,000" in t for t in spoken), "the answer must still be spoken"


async def test_the_tool_still_runs_when_its_filler_is_suppressed(two_way):
    """Dropping the words must not drop the lookup."""
    seen, _tts, _stt = two_way(
        [_said("Fee enta?")],
        replies=[
            _reply(text="Let me check.",
                   tools=[_tool("search_course_material", query="course fees")]),
            _reply(text="Twenty five thousand."),
        ],
    )

    await _run(_FakePlivoWS(), one_way=False)

    assert seen["queries"] == ["course fees"]


# ── a burst of short utterances must not starve the agent ────────────────────
#
# REGRESSION from a real call. The lead said "హలో", "ఓకే ఓకే", "హలో", "ఓకే" in
# quick succession. The logs show SIX OpenAI completions, all 200 — and the
# transcript shows ONE agent turn. Every new transcript cancelled the answer
# still being generated for the previous one, so the agent was answering every
# time and we were throwing the answers away and paying for them. The lead
# heard silence and hung up after 28 seconds.
#
# Cancelling is right for a genuine barge-in — the lead talking OVER the agent.
# It is wrong for a lead who simply said two things in a row, which on a phone
# is most of them, and which Sarvam's STT also produces naturally by emitting
# an utterance per pause.

async def test_a_burst_of_utterances_produces_one_answer_not_none(two_way):
    """The whole failure in one test: four transcripts, one reply."""
    seen, _tts, _stt = two_way(
        [_said("హలో."), _said("ఓకే ఓకే."), _said("హలో."), _said("ఓకే.")],
        replies=[_reply(text="Chెప్పండి.")],
    )

    outcome = await _run(_FakePlivoWS(), one_way=False)

    agent_turns = [t for t in outcome["transcript"]
                   if t.role == "agent" and "Chెప్పండి" in t.text]
    assert agent_turns, (
        "the agent answered and every answer was cancelled — the lead heard "
        "nothing at all"
    )


async def test_a_burst_costs_one_completion_not_one_each(two_way):
    """Six completions were paid for and one was used. Each cancelled reply is
    a wasted API call as well as a silent lead."""
    seen, _tts, _stt = two_way(
        [_said("హలో."), _said("ఓకే ఓకే."), _said("హలో."), _said("ఓకే.")],
        replies=[_reply(text="సరే.")],
    )

    await _run(_FakePlivoWS(), one_way=False)

    assert len(seen["histories"]) <= 2, (
        f"a burst of 4 utterances triggered {len(seen['histories'])} model "
        "calls; they should collapse into one"
    )


async def test_everything_the_lead_said_reaches_the_model(two_way):
    """Collapsing the burst must not lose what was said in it — the one reply
    has to answer all of it, not just the last fragment."""
    seen, _tts, _stt = two_way(
        [_said("ఫీజు ఎంత?"), _said("మరియు ఎన్ని నెలలు?")],
        replies=[_reply(text="సరే.")],
    )

    await _run(_FakePlivoWS(), one_way=False)

    assert seen["histories"], "the model should have been called"
    last = seen["histories"][-1]
    said = " ".join(m.get("content") or "" for m in last if m.get("role") == "user")
    assert "ఫీజు ఎంత?" in said and "మరియు ఎన్ని నెలలు?" in said


# ── a burst DURING the model call must not starve the agent either ───────────
#
# REGRESSION from a live call, 2026-08-07 — the sequel to the burst above.
# That fix collapses fragments arriving before the model is ever called. It
# does not cover fragments arriving AFTER: on a real call two consecutive
# turns logged 0 rounds, nothing spoken, on an ordinary mid-conversation
# question (not an ending). Cancel-and-restart on every fragment, applied
# unconditionally, means a lead who keeps talking for the length of one model
# round-trip can cancel that round right as it would have finished, forever —
# the burst-collapse fix and this bug are the same mechanism pointed at two
# different windows in the same turn.

async def test_a_fragment_mid_call_does_not_cancel_the_committed_round(monkeypatch):
    """Once conversation_llm.turn() is actually in flight, a new fragment must
    not cancel it — that round has to be allowed to finish and speak."""
    monkeypatch.setattr(sarvam_bridge, "SARVAM_REPLY_SETTLE_S", 0)
    monkeypatch.setattr(sarvam_bridge, "SARVAM_REPLY_MAX_WAIT_S", 0)

    call = sarvam_bridge.plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="l", one_way=False, protect_opening=False)
    turns = []
    convo = sarvam_bridge._Conversation(
        call, lead_id="l", system_prompt="s", turns=turns)

    calls: list[list[dict]] = []
    first_call_started = asyncio.Event()
    release_first_call = asyncio.Event()

    async def controlled_turn(history):
        calls.append([dict(m) for m in history])
        if len(calls) == 1:
            first_call_started.set()
            await release_first_call.wait()
        return sarvam_bridge.conversation_llm.LLMReply(
            text=f"Answer {len(calls)}.", tool_calls=[])

    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", controlled_turn)

    class _FakeTTS:
        async def speak(self, text):
            yield _FRAME, 500

    tts = _FakeTTS()

    await convo.on_lead_said(tts, "Course fee?")
    await asyncio.wait_for(first_call_started.wait(), timeout=1)
    assert len(calls) == 1

    # A second fragment lands while the round above is still in flight.
    await convo.on_lead_said(tts, "and the duration?")

    assert len(calls) == 1, (
        "a fragment that arrived mid-call cancelled the committed round — "
        "this is the live 'agent never answered' bug"
    )

    release_first_call.set()
    for _ in range(50):
        if any(t.role == "agent" for t in turns):
            break
        await asyncio.sleep(0)

    spoken = [t.text for t in turns if t.role == "agent"]
    assert spoken, "the committed round must still be spoken, not dropped"
    assert "Answer 1." in spoken[0]


async def test_a_fragment_missed_mid_call_still_gets_its_own_answer(monkeypatch):
    """The fragment that could not cancel the in-flight round (previous test)
    must not be silently dropped either — it gets answered in a follow-up
    pass once that round is done, via _missed_while_committed."""
    monkeypatch.setattr(sarvam_bridge, "SARVAM_REPLY_SETTLE_S", 0)
    monkeypatch.setattr(sarvam_bridge, "SARVAM_REPLY_MAX_WAIT_S", 0)

    call = sarvam_bridge.plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="l", one_way=False, protect_opening=False)
    turns = []
    convo = sarvam_bridge._Conversation(
        call, lead_id="l", system_prompt="s", turns=turns)

    calls: list[list[dict]] = []
    first_call_started = asyncio.Event()
    release_first_call = asyncio.Event()

    async def controlled_turn(history):
        calls.append([dict(m) for m in history])
        if len(calls) == 1:
            first_call_started.set()
            await release_first_call.wait()
        return sarvam_bridge.conversation_llm.LLMReply(
            text=f"Answer {len(calls)}.", tool_calls=[])

    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", controlled_turn)

    class _FakeTTS:
        async def speak(self, text):
            yield _FRAME, 500

    tts = _FakeTTS()

    await convo.on_lead_said(tts, "Course fee?")
    await asyncio.wait_for(first_call_started.wait(), timeout=1)
    await convo.on_lead_said(tts, "and the duration?")
    release_first_call.set()

    for _ in range(200):
        if len(calls) >= 2 and len([t for t in turns if t.role == "agent"]) >= 2:
            break
        await asyncio.sleep(0)

    spoken = [t.text for t in turns if t.role == "agent"]
    assert len(spoken) >= 2, (
        f"the fragment that arrived mid-call was never answered: {spoken}"
    )
    assert "Answer 2." in spoken[-1]


# ── the reply gate is driven by Sarvam's END_SPEECH, not a blind timer ───────
#
# sarvam_stt.py always decoded Sarvam's END_SPEECH VAD signal into a
# 'speech_ended' event; converse() had no branch for it and it fell through
# doing nothing, so the agent blind-waited SARVAM_REPLY_MAX_WAIT_S on every
# single turn even when the vendor had already said the line was quiet.
# Sarvam does not guarantee whether END_SPEECH arrives before or after the
# final transcript for the same utterance, so none of this may assume an
# order.

async def test_end_speech_before_the_transcript_answers_without_the_full_ceiling(
    two_way, monkeypatch,
):
    """END_SPEECH landing before the transcript must not still pay the old
    fixed wait — the ceiling is a fallback, not a floor.

    Proven by squeezing CALL_MAX_DURATION_S between the settle time and the
    ceiling: if the reply is still blind-waiting the 5s ceiling, the call ends
    before it ever gets spoken and the assertion below fails. A tight outer
    asyncio.wait_for can't prove this on its own — plivo_stream.PlivoCall.run
    always blocks for the full CALL_MAX_DURATION_S regardless of how fast the
    reply was, since nothing else ends a two-way call early.
    """
    monkeypatch.setattr(sarvam_bridge, "SARVAM_REPLY_SETTLE_S", 0.02)
    monkeypatch.setattr(sarvam_bridge, "SARVAM_REPLY_MAX_WAIT_S", 5.0)
    monkeypatch.setattr(sarvam_bridge.plivo_stream, "CALL_MAX_DURATION_S", 0.3)
    two_way(
        [_barge_in(), _speech_ended(), _said("Fee enta?")],
        replies=[_reply(text="Iravai vela.")],
    )

    outcome = await _run(_FakePlivoWS(), one_way=False)

    assert any(t.role == "agent" and "Iravai vela." in t.text
               for t in outcome["transcript"]), (
        "the reply did not land inside the 0.3s call — it is still "
        "blind-waiting the 5s ceiling regardless of END_SPEECH"
    )


async def test_end_speech_after_the_transcript_also_answers_promptly(
    two_way, monkeypatch,
):
    """The opposite order must also not block for the full ceiling — the poll
    loop has to pick up END_SPEECH once it lands. Same proof technique as
    above: a short CALL_MAX_DURATION_S that only a prompt reply can beat."""
    monkeypatch.setattr(sarvam_bridge, "SARVAM_REPLY_SETTLE_S", 0.02)
    monkeypatch.setattr(sarvam_bridge, "SARVAM_REPLY_MAX_WAIT_S", 5.0)
    monkeypatch.setattr(sarvam_bridge, "_SILENCE_POLL_S", 0.02)
    monkeypatch.setattr(sarvam_bridge.plivo_stream, "CALL_MAX_DURATION_S", 0.3)
    two_way(
        [_barge_in(), _said("Fee enta?"), _speech_ended()],
        replies=[_reply(text="Iravai vela.")],
    )

    outcome = await _run(_FakePlivoWS(), one_way=False)

    assert any(t.role == "agent" and "Iravai vela." in t.text
               for t in outcome["transcript"])


async def test_missing_end_speech_still_answers_on_the_ceiling(two_way, monkeypatch):
    """If END_SPEECH never arrives for an utterance — a vendor hiccup, a VAD
    that doesn't fire — the reply must still happen, bounded by
    SARVAM_REPLY_MAX_WAIT_S rather than stuck forever. CALL_MAX_DURATION_S is
    set comfortably above the ceiling here, the opposite of the two tests
    above, so the reply has time to land."""
    monkeypatch.setattr(sarvam_bridge, "SARVAM_REPLY_SETTLE_S", 0.02)
    monkeypatch.setattr(sarvam_bridge, "SARVAM_REPLY_MAX_WAIT_S", 0.1)
    monkeypatch.setattr(sarvam_bridge, "_SILENCE_POLL_S", 0.02)
    monkeypatch.setattr(sarvam_bridge.plivo_stream, "CALL_MAX_DURATION_S", 0.5)
    two_way(
        [_barge_in(), _said("Fee enta?")],
        replies=[_reply(text="Iravai vela.")],
    )

    outcome = await _run(_FakePlivoWS(), one_way=False)

    assert any(t.role == "agent" and "Iravai vela." in t.text
               for t in outcome["transcript"])


async def test_a_burst_still_costs_one_completion_with_vad_signals_interleaved(
    two_way, monkeypatch,
):
    """The regression the old debounce fixed, replayed with the VAD events a
    real call actually produces around each utterance — not just bare
    transcripts. Must still collapse to one answer, one completion."""
    monkeypatch.setattr(sarvam_bridge, "SARVAM_REPLY_SETTLE_S", 0.02)
    monkeypatch.setattr(sarvam_bridge, "SARVAM_REPLY_MAX_WAIT_S", 0.3)
    monkeypatch.setattr(sarvam_bridge, "_SILENCE_POLL_S", 0.02)
    seen, _tts, _stt = two_way(
        [_barge_in(), _said("హలో."), _speech_ended(),
         _barge_in(), _said("ఓకే ఓకే."), _speech_ended(),
         _barge_in(), _said("హలో."), _speech_ended(),
         _barge_in(), _said("ఓకే."), _speech_ended()],
        replies=[_reply(text="Cheppandi.")],
    )

    await _run(_FakePlivoWS(), one_way=False)

    assert len(seen["histories"]) <= 2, (
        f"a burst of 4 utterances triggered {len(seen['histories'])} model "
        "calls with VAD signals in play; they should still collapse into one"
    )


def test_the_bridge_knows_about_every_kind_of_thing_the_lead_can_do():
    """REGRESSION GUARD. speech_ended was decoded by sarvam_stt and thrown
    away by converse() for the whole of two-way's first life: it branched on
    speech_started and transcript and let END_SPEECH fall through to nothing,
    so the agent blind-waited a timer while the vendor was telling it the
    answer. If a fourth EventKind is ever added, this fails until someone
    decides what converse() does with it."""
    from typing import get_args

    assert set(get_args(sarvam_bridge.sarvam_stt.EventKind)) == {
        "transcript", "speech_started", "speech_ended",
    }


# ── one measured line per turn ───────────────────────────────────────────────
#
# "The agent replies late" was unattributable for the whole of two-way's first
# life: the only per-call logging was call start and call end, so the gate, the
# model, the course lookup and the synthesiser — four completely different
# fixes — could not be told apart from the logs of a real call.

async def test_a_turn_logs_where_the_lead_s_wait_actually_went(two_way, caplog):
    caplog.set_level("INFO")
    two_way([_said("Fee enta?")], replies=[_reply(text="Iravai vela.")])

    await _run(_FakePlivoWS(), one_way=False)

    lines = [r.message for r in caplog.records if "turn latency" in r.message]
    assert lines, "a turn that answered the lead logged no latency line"
    assert "gate=" in lines[0] and "llm=" in lines[0] and "total=" in lines[0]


async def test_the_lead_id_reaches_the_audio_health_log(two_way, caplog):
    """SarvamSTT logs its own [sarvam-stt] audio-health line (see
    app/telephony/sarvam_stt.py) — this only confirms the bridge actually
    threads the call's lead_id into it, so that line can be correlated
    against this call's other [sarvam] lines afterward."""
    caplog.set_level("INFO")
    two_way([_speech_ended()], replies=[])

    await _run(_FakePlivoWS(), one_way=False, lead_id="lead-audio-health")

    lines = [r.message for r in caplog.records if "audio health" in r.message]
    assert lines, "no audio-health line was logged"
    assert "lead=lead-audio-health" in lines[0]


async def test_the_latency_line_is_measured_from_the_first_unanswered_word(
    two_way, caplog, monkeypatch,
):
    """A burst is ONE turn. Measuring from the LAST fragment would report a
    fast reply to a lead who had been waiting since the first one."""
    caplog.set_level("INFO")
    monkeypatch.setattr(sarvam_bridge, "SARVAM_REPLY_SETTLE_S", 0.02)
    monkeypatch.setattr(sarvam_bridge, "SARVAM_REPLY_MAX_WAIT_S", 0.3)
    two_way([_said("ఫీజు ఎంత?"), _said("మరియు ఎన్ని నెలలు?")],
            replies=[_reply(text="సరే.")])

    await _run(_FakePlivoWS(), one_way=False)

    lines = [r.message for r in caplog.records if "turn latency" in r.message]
    assert len(lines) == 1, (
        f"a burst of 2 utterances logged {len(lines)} turns; it is one turn "
        "from the lead's point of view"
    )


async def test_a_failed_turn_is_answered_with_a_fallback_not_silence(
        two_way, caplog, monkeypatch):
    """The 'agent never answered me' symptom, on the path a real call takes.

    A turn swallowed by _reply's catch-all used to leave the lead with nothing;
    now it hears the reprompt, so the latency line reports a turn that SPOKE.
    """
    caplog.set_level("INFO")

    async def failing_turn(history):
        raise sarvam_bridge.conversation_llm.TurnFailed("boom")

    two_way([_said("Fee enta?")])
    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", failing_turn)

    outcome = await _run(_FakePlivoWS(), one_way=False)

    assert sarvam_bridge._FALLBACK_REPROMPT in [
        t.text for t in outcome["transcript"] if t.role == "agent"
    ], "the lead was left in silence after asking a question"
    lines = [r.message for r in caplog.records if "turn latency" in r.message]
    assert lines and "NOTHING SPOKEN" not in lines[0], (
        "the turn spoke a fallback, so it must not be logged as silent"
    )


async def test_a_turn_that_could_not_even_speak_is_still_logged(
        two_way, caplog, monkeypatch):
    """NOTHING SPOKEN must survive for the genuinely silent case — a turn
    whose fallback could not be synthesised either."""
    caplog.set_level("INFO")

    async def failing_turn(history):
        raise sarvam_bridge.conversation_llm.TurnFailed("boom")

    call = sarvam_bridge.plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="lead-1", one_way=False, protect_opening=False)
    convo = sarvam_bridge._Conversation(
        call, lead_id="lead-1", system_prompt="s", turns=[])
    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", failing_turn)
    convo._turn_opened_at = convo._loop.time()

    await convo._reply(_SilentTTS())

    lines = [r.message for r in caplog.records if "turn latency" in r.message]
    assert lines and "NOTHING SPOKEN" in lines[0], (
        "a turn that left the lead in silence must say so in the log"
    )


async def test_the_opening_is_not_counted_as_a_turn(two_way, caplog):
    """Nobody is waiting on an answer during the disclosure — counting it
    would report a huge false 'latency' on every single call."""
    caplog.set_level("INFO")
    two_way([])

    await _run(_FakePlivoWS(), one_way=False)

    assert not [r.message for r in caplog.records if "turn latency" in r.message]


# ── a sound with no words must not orphan the question before it ────────────
#
# REGRESSION. Sarvam drops empty transcripts (sarvam_stt.py) — a cough, a line
# pop, a syllable under the transcription threshold — so START_SPEECH can fire
# with no transcript ever following it. on_barge_in() unconditionally cancels
# whatever reply was in flight. Before on_lead_stopped repaired this, nothing
# then re-armed the answer to the question the lead asked BEFORE that sound:
# they would sit in silence until the watchdog hung up on them 20s later,
# which is indistinguishable from "the agent never replied".

class _SilentTTS:
    async def speak(self, text):
        return
        yield  # pragma: no cover - makes this an async generator

    async def reset_after_cancel(self):
        return None


async def test_a_sound_with_no_words_does_not_orphan_the_question(monkeypatch):
    monkeypatch.setattr(sarvam_bridge, "SARVAM_REPLY_SETTLE_S", 0.01)
    monkeypatch.setattr(sarvam_bridge, "SARVAM_REPLY_MAX_WAIT_S", 0.05)

    called = asyncio.Event()

    async def fake_turn(history):
        called.set()
        return sarvam_bridge.conversation_llm.LLMReply(text="Sare.", tool_calls=[])

    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", fake_turn)

    call = sarvam_bridge.plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="l", one_way=False, protect_opening=False)
    convo = sarvam_bridge._Conversation(
        call, lead_id="l", system_prompt="s", turns=[])
    tts = _SilentTTS()

    await convo.on_lead_said(tts, "Fee enta?")
    # A sound with no words behind it: Sarvam still fires START_SPEECH, which
    # cancels the reply just queued for the question above.
    await convo.on_barge_in()
    assert convo.reply_task is None, "the barge-in should have cancelled it"

    # The sound ends. Nothing else will ever re-ask this question.
    convo.on_lead_stopped(tts)

    assert convo.reply_task is not None, (
        "END_SPEECH must re-arm a reply when a question is still unanswered"
    )
    await asyncio.wait_for(called.wait(), timeout=1.0)


async def test_cancelled_tool_lookup_does_not_leave_invalid_history(monkeypatch):
    call = sarvam_bridge.plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="l", one_way=False, protect_opening=False)
    convo = sarvam_bridge._Conversation(
        call, lead_id="l", system_prompt="s", turns=[])

    convo.history.extend([
        {"role": "user", "content": "What is the fee?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "search_course_material",
                           "arguments": '{"query":"fees"}'}},
        ]},
    ])
    task = asyncio.create_task(asyncio.sleep(10))
    convo.reply_task = task
    await convo._cancel_reply()

    assert convo.history == [{"role": "system", "content": "s"},
                             {"role": "user", "content": "What is the fee?"}]


async def test_end_speech_does_not_re_arm_a_question_already_answered(monkeypatch):
    """The other side of the same guard: END_SPEECH must not conjure up a
    duplicate answer once the agent has actually replied."""
    called = asyncio.Event()

    async def fake_turn(history):
        called.set()
        return sarvam_bridge.conversation_llm.LLMReply(text="x", tool_calls=[])

    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", fake_turn)

    call = sarvam_bridge.plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="l", one_way=False, protect_opening=False)
    convo = sarvam_bridge._Conversation(
        call, lead_id="l", system_prompt="s", turns=[])
    convo.history.append({"role": "user", "content": "Fee enta?"})
    convo.history.append({"role": "assistant", "content": "Iravai vela."})

    convo.on_lead_stopped(_SilentTTS())

    assert convo.reply_task is None
    await asyncio.sleep(0.05)
    assert not called.is_set()


async def test_awaiting_answer_survives_a_cancelled_lookup():
    """A tool lookup cancelled mid-flight leaves an assistant message with
    tool_calls and no content, plus a tool result, sitting on top of a
    question that is still unanswered — neither counts as an answer."""
    call = sarvam_bridge.plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="l", one_way=False, protect_opening=False)
    convo = sarvam_bridge._Conversation(
        call, lead_id="l", system_prompt="s", turns=[])
    convo.history.extend([
        {"role": "user", "content": "Fee enta?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "search_course_material",
                          "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "..."},
    ])

    assert convo._awaiting_answer() is True


async def test_awaiting_answer_is_false_once_the_agent_has_spoken():
    call = sarvam_bridge.plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="l", one_way=False, protect_opening=False)
    convo = sarvam_bridge._Conversation(
        call, lead_id="l", system_prompt="s", turns=[])
    convo.history.append({"role": "user", "content": "Fee enta?"})
    convo.history.append({"role": "assistant", "content": "Iravai vela."})

    assert convo._awaiting_answer() is False


async def test_awaiting_answer_is_false_with_nothing_said_yet():
    call = sarvam_bridge.plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="l", one_way=False, protect_opening=False)
    convo = sarvam_bridge._Conversation(
        call, lead_id="l", system_prompt="s", turns=[])

    assert convo._awaiting_answer() is False


# ── the disclosure must be HEARD before the shield comes down ────────────────

async def test_the_opening_stays_protected_until_it_has_actually_played(two_way):
    """REGRESSION from a live call that logged a [compliance] ERROR.

    protect_opening=True stops a lead talking over the AI disclosure. But the
    shield was dropped the moment say() returned — and say() returns when the
    audio has been SENT, not heard. Sarvam's TTS delivers far faster than real
    time, so the lead was still two seconds into an eleven-second opening.
    Saying "హలో" then counted as a valid barge-in, the SpokenLedger truthfully
    rewrote the turn to the only sentence that had played — the bare name
    greeting — and the disclosure was gone from both the call and the record.

    Same send-versus-play distinction as the silence watchdog, with a
    compliance consequence instead of a billing one."""
    two_way([])
    call = sarvam_bridge.plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="l", one_way=False, protect_opening=True)
    convo = sarvam_bridge._Conversation(
        call, lead_id="l", system_prompt="s", turns=[])

    one_second = base64.b64encode(b"" * 8000).decode()  # 8000 B = 1s

    class _InstantTTS:
        async def speak(self, text):
            # A whole second of audio, handed over instantly — exactly what the
            # real client does when Sarvam answers faster than playback.
            # PlivoCall.play() derives play_end from the PAYLOAD, so this has
            # to be a genuine second of bytes, not a short frame labelled 1000.
            yield one_second, 1000

    task = asyncio.create_task(convo.deliver_opening(_InstantTTS(), "ఒకటి. రెండు."))
    await asyncio.sleep(0.05)

    assert not await call.interrupt(), (
        "the lead was able to talk over the AI disclosure while it was still "
        "playing"
    )
    await asyncio.wait_for(task, timeout=10)
    assert await call.interrupt(), "the shield must come down once it has played"


# ── the brand name reaches the model spelled correctly ──────────────────────
#
# app/telephony/term_repair.py fixes what STT could not know; this is the wiring
# test that it happens on the path a real call takes, not merely in isolation.
# The sentence is verbatim from the call where the agent refused a customer
# asking about its own courses.

async def test_a_mangled_brand_name_is_repaired_before_the_model_sees_it(monkeypatch):
    """The model, the RAG query it writes, and the stored transcript are all
    built from this one string, so the repair has to land before any of them."""
    call = sarvam_bridge.plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="l", one_way=False, protect_opening=False)
    turns = []
    convo = sarvam_bridge._Conversation(
        call, lead_id="l", system_prompt="s", turns=turns)

    async def never_reply(*a, **kw):
        raise AssertionError("this test is about the transcript, not the reply")

    monkeypatch.setattr(convo, "_reply_when_they_stop", never_reply)

    class _FakeTTS:
        async def speak(self, text):
            yield _FRAME, 500

    await convo.on_lead_said(_FakeTTS(), "డిజిటల్ బ్రౌనీలో ఉన్న కోర్సెస్ చెప్తారా?")
    # Let the reply task the call schedules die quietly.
    await convo._cancel_reply(force=True)

    assert "బ్రోలీలో" in turns[0].text, "the stored transcript still says Brownie"
    assert "బ్రోలీలో" in (convo.history[-1]["content"] or ""), \
        "the model was still handed a company that does not exist"


# ── a turn that fails still says something ──────────────────────────────────
#
# _reply's catch-all used to log the failure and return, which is TOTAL SILENCE
# for that turn. Confirmed on a real call (2026-08-12): the lead asked about
# the courses, heard nothing, said "హలో", then "చెప్పండి", and only then got an
# answer. With CONVERSATION_LLM_TIMEOUT_S at 12s that silence is twelve seconds
# long, and it reads exactly like a dropped line. _FALLBACK_FAREWELL already
# established the principle for the end_call case: a canned line beats nothing.


def _failing_turn_convo():
    call = sarvam_bridge.plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="l", one_way=False, protect_opening=False)
    turns = []
    convo = sarvam_bridge._Conversation(
        call, lead_id="l", system_prompt="s", turns=turns)
    return call, convo, turns


class _CountingTTS:
    def __init__(self):
        self.spoken = []

    async def speak(self, text):
        self.spoken.append(text)
        yield _FRAME, 500


def _agent_lines(turns):
    return [t.text for t in turns if t.role == "agent"]


async def test_a_failed_turn_speaks_a_fallback_rather_than_silence(monkeypatch):
    """A TurnFailed left the lead with nothing at all for the whole turn."""
    _call, convo, turns = _failing_turn_convo()

    async def boom(history):
        raise sarvam_bridge.conversation_llm.TurnFailed("upstream 500")

    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", boom)

    await convo._reply(_CountingTTS())

    assert sarvam_bridge._FALLBACK_REPROMPT in _agent_lines(turns), \
        "the lead heard nothing at all after asking a question"


async def test_a_model_that_answers_with_nothing_still_gets_a_spoken_fallback(
        monkeypatch):
    """NoAnswer is raised rather than returned so nothing can speak it by
    accident — but the lead still has to hear something."""
    _call, convo, turns = _failing_turn_convo()

    async def empty(history):
        raise sarvam_bridge.conversation_llm.NoAnswer("no text, no tool calls")

    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", empty)

    await convo._reply(_CountingTTS())

    assert sarvam_bridge._FALLBACK_REPROMPT in _agent_lines(turns)


async def test_a_barge_in_cancellation_is_never_answered_with_a_fallback(
        monkeypatch):
    """CancelledError means the lead interrupted. Speaking a fallback then
    would talk over the person who just took the floor."""
    _call, convo, turns = _failing_turn_convo()

    async def cancelled(history):
        raise asyncio.CancelledError()

    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", cancelled)
    tts = _CountingTTS()

    with pytest.raises(asyncio.CancelledError):
        await convo._reply(tts)

    assert not tts.spoken, "the agent talked over a lead who had interrupted"


async def test_a_fallback_is_not_spoken_into_a_stale_generation(monkeypatch):
    """A barge-in that lands while the model call is in flight bumps the
    generation. The turn is over; nothing from it may still be said."""
    _call, convo, turns = _failing_turn_convo()

    async def boom_after_barge_in(history):
        convo.generation += 1          # what on_barge_in() does
        raise sarvam_bridge.conversation_llm.TurnFailed("upstream 500")

    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn",
                        boom_after_barge_in)
    tts = _CountingTTS()

    await convo._reply(tts)

    assert not tts.spoken, "a fallback was spoken after the lead interrupted"


async def test_repeated_failures_escalate_to_a_handoff(monkeypatch):
    """Repeating the same 'say that again' line forever is its own failure —
    after three dead turns the lead is told a human will follow up."""
    _call, convo, turns = _failing_turn_convo()

    async def boom(history):
        raise sarvam_bridge.conversation_llm.TurnFailed("upstream 500")

    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", boom)
    tts = _CountingTTS()

    for _ in range(3):
        await convo._reply(tts)

    assert _agent_lines(turns) == [
        sarvam_bridge._FALLBACK_REPROMPT,
        sarvam_bridge._FALLBACK_REPROMPT,
        sarvam_bridge._FALLBACK_HANDOFF,
    ]


async def test_a_successful_turn_resets_the_failure_count(monkeypatch):
    """One bad turn in an otherwise healthy call must not push the next
    failure straight to a handoff."""
    _call, convo, turns = _failing_turn_convo()
    outcomes = ["fail", "ok", "fail"]

    async def flaky(history):
        which = outcomes.pop(0)
        if which == "fail":
            raise sarvam_bridge.conversation_llm.TurnFailed("upstream 500")
        return _reply(text="Iravai aidu vela.")

    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", flaky)
    tts = _CountingTTS()

    for _ in range(3):
        await convo._reply(tts)

    assert _agent_lines(turns)[-1] == sarvam_bridge._FALLBACK_REPROMPT, \
        "a healthy turn in between did not clear the failure count"


async def test_a_turn_that_already_spoke_is_not_given_a_fallback_on_top(
        monkeypatch):
    """A farewell, or any answer, that already reached the lead must not be
    followed by 'sorry, say that again' because something failed afterwards."""
    _call, convo, turns = _failing_turn_convo()

    async def speaks_then_fails(history):
        return _reply(text="ధన్యవాదాలు.", tools=[_tool("end_call")])

    async def boom(reply):
        raise RuntimeError("tool execution blew up")

    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", speaks_then_fails)
    monkeypatch.setattr(convo, "_run_tools", boom)

    await convo._reply(_CountingTTS())

    lines = _agent_lines(turns)
    assert "ధన్యవాదాలు." in lines, "the farewell should still have been spoken"
    assert sarvam_bridge._FALLBACK_REPROMPT not in lines, \
        "a fallback was stacked on top of audio the lead had already heard"


async def test_the_handoff_line_is_never_repeated(monkeypatch):
    """Once the lead has been told a human will call back, saying it again on
    every further dead turn just teaches them it is a recording."""
    _call, convo, turns = _failing_turn_convo()

    async def boom(history):
        raise sarvam_bridge.conversation_llm.TurnFailed("upstream 500")

    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", boom)

    for _ in range(5):
        await convo._reply(_CountingTTS())

    assert _agent_lines(turns).count(sarvam_bridge._FALLBACK_HANDOFF) == 1
    # reprompt, reprompt, handoff — then nothing further, however long the
    # provider stays broken.
    assert len(_agent_lines(turns)) == 3


async def test_tool_round_exhaustion_speaks_a_fallback(monkeypatch):
    """The model looping on tool calls used to end in a logger.warning and
    silence — the same symptom by a different route."""
    _call, convo, turns = _failing_turn_convo()

    async def always_searching(history):
        return _reply(tools=[_tool("search_course_material", query="fees")])

    async def fake_search(query, *a, **kw):
        return "The fee is 25,000 rupees."

    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", always_searching)
    monkeypatch.setattr(sarvam_bridge, "search_relevant", fake_search)
    tts = _CountingTTS()

    await convo._reply(tts)

    assert sarvam_bridge._FALLBACK_REPROMPT in tts.spoken


# ── the RAG holding line ────────────────────────────────────────────────────
#
# A tool turn is silent for its whole span: round-1 LLM decides to search
# (~1.5-2.5s already gone), the lookup runs, round 2 writes the answer —
# measured 3.18-5.23s end to end on the 2026-08-27 live calls, and the lead
# on the first morning call said "హలో?" into exactly that wait. The model's
# OWN tool-call text stays suppressed (the recorded lesson in _reply: it
# verbosely announced lookups that then failed); instead a short FIXED line
# is spoken once, before the lookup, so its audio plays while the tool and
# the second round compute.

def _tool_turn_convo(monkeypatch, replies, *, opening_spoken=True):
    call, convo, turns = _failing_turn_convo()
    if opening_spoken:
        turns.append(sarvam_bridge.TranscriptTurn(role="agent", text="opening"))
    queue = list(replies)

    async def fake_turn(history):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def fake_search(query, *a, **kw):
        return "The fee is 25,000 rupees."

    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", fake_turn)
    monkeypatch.setattr(sarvam_bridge, "search_relevant", fake_search)
    return call, convo, turns


async def test_a_rag_turn_speaks_a_holding_line_before_the_lookup(monkeypatch):
    _call, convo, _turns = _tool_turn_convo(monkeypatch, [
        _reply(tools=[_tool("search_course_material", query="fees")]),
        _reply(text="Fee is 25,000."),
    ])
    tts = _CountingTTS()

    await convo._reply(tts)

    assert tts.spoken[0] == sarvam_bridge._RAG_HOLDING_LINE, (
        "the lead should hear the holding line first, not silence"
    )
    assert any("25,000" in s for s in tts.spoken[1:]), "the answer must follow"


async def test_the_holding_line_never_enters_the_model_history(monkeypatch):
    """The 2026-08-27 adversarial review confirmed two HIGH defects from the
    holding line living in history as an assistant-with-content message: a
    barge-in's truncate could delete the tool_calls message instead
    (orphaning its tool result — OpenAI 400s every later turn), and
    _awaiting_answer reported a cancelled lookup's question as already
    answered (the cough-repair never re-asked; the call went silent and
    graded done). The line carries no information the model needs, so it is
    an ASIDE: spoken, in the transcript, and absent from history entirely."""
    _call, convo, turns = _tool_turn_convo(monkeypatch, [
        _reply(tools=[_tool("search_course_material", query="fees")]),
        _reply(text="Fee is 25,000."),
    ])

    await convo._reply(_CountingTTS())

    assert all(m.get("content") != sarvam_bridge._RAG_HOLDING_LINE
               for m in convo.history), "the aside leaked into model history"
    assert sarvam_bridge._RAG_HOLDING_LINE in _agent_lines(turns), (
        "the transcript must still record what the lead heard"
    )
    toolmsg_idx = next(
        i for i, m in enumerate(convo.history) if m.get("tool_calls"))
    assert convo.history[toolmsg_idx + 1].get("role") == "tool", (
        "the tool result must immediately follow its tool_calls message"
    )


async def test_a_barge_in_on_a_partly_played_holding_line_keeps_the_tool_pair(monkeypatch):
    """The HIGH from the review: mid round 2, _agent_turn_index still points
    at the holding line while the newest assistant message in history is the
    tool_calls bookkeeping. A barge-in before the ~2s of holding audio
    finished (heard == "") took truncate's drop branch, which deleted that
    tool_calls message — orphaning its tool result and 400-ing every later
    completion. An aside has no history entry, so truncate must leave
    history entirely alone for it."""
    _call, convo, turns = _failing_turn_convo()
    convo.history.append({"role": "user", "content": "ఫీజు ఎంత?"})
    convo.history.append({"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "search_course_material", "arguments": "{}"}}]})
    convo.history.append({"role": "tool", "tool_call_id": "c1", "content": "chunk"})
    before = [dict(m) for m in convo.history]
    turns.append(sarvam_bridge.TranscriptTurn(role="agent", text="opening"))
    turns.append(sarvam_bridge.TranscriptTurn(
        role="agent", text=sarvam_bridge._RAG_HOLDING_LINE))
    convo._agent_turn_index = 1
    convo._agent_turn_in_history = False   # an aside never has a history entry
    convo.ledger.reset()
    convo.ledger.add(sarvam_bridge._RAG_HOLDING_LINE, 1500)
    # played_ms() is 0 — the barge-in landed before the line finished playing.

    convo.truncate_to_what_was_heard()

    assert _agent_lines(turns) == ["opening"], (
        "the unheard holding line must drop from the transcript"
    )
    assert convo.history == before, (
        "truncate touched history for a turn that has no history entry — "
        "this is the orphaned-tool-result 400"
    )


async def test_the_holding_line_does_not_mark_the_question_answered():
    """The other HIGH: a cough during the lookup cancels the reply; Sarvam
    sends no transcript for it, so on_lead_stopped's repair is the only
    thing that re-asks — and it consults _awaiting_answer, which used to see
    the holding line as an assistant-with-content and give up. As an aside,
    the holding line must leave the question visibly unanswered."""
    _call, convo, _turns = _failing_turn_convo()
    convo.turns.append(sarvam_bridge.TranscriptTurn(role="agent", text="opening"))
    convo.history.append({"role": "user", "content": "ఫీజు ఎంత?"})

    await convo.say(_CountingTTS(), sarvam_bridge._RAG_HOLDING_LINE, aside=True)

    assert convo._awaiting_answer() is True, (
        "the holding line convinced the repair the question was answered — "
        "the lead hears 'ఒక్క క్షణం', then silence, and the call grades done"
    )


async def test_a_question_missed_with_a_trailing_nudge_still_gets_answered(monkeypatch):
    """CONFIRMED by the review with an executed repro: _missed_text kept only
    the LAST fragment, so a real question followed by 'హలో' inside one
    committed model round was classified as a nudge and the whole follow-up
    suppressed — the question was never sent to the model at all. The
    suppression must require EVERY missed fragment to be prompting-only."""
    monkeypatch.setattr(sarvam_bridge, "SARVAM_REPLY_SETTLE_S", 0.0)
    monkeypatch.setattr(sarvam_bridge, "SARVAM_REPLY_MAX_WAIT_S", 0.05)
    _call, convo, _turns = _failing_turn_convo()
    convo.turns.append(sarvam_bridge.TranscriptTurn(role="agent", text="opening"))

    entered = asyncio.Event()
    release = asyncio.Event()
    llm_histories = []

    async def slow_turn(history):
        llm_histories.append([dict(m) for m in history])
        if len(llm_histories) == 1:
            entered.set()
            await release.wait()
        return _reply(text=f"Answer {len(llm_histories)}.")

    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", slow_turn)
    tts = _CountingTTS()

    await convo.on_lead_said(tts, "కోర్సు వివరాలు చెప్పండి")
    await asyncio.wait_for(entered.wait(), timeout=2)
    # Two fragments land while round 1's HTTP call is committed:
    await convo.on_lead_said(tts, "ప్లేస్‌మెంట్ ఉంటుందా?")   # a real question
    await convo.on_lead_said(tts, "హలో")                      # then a nudge
    release.set()
    for _ in range(200):
        await asyncio.sleep(0.01)
        if convo.reply_task is None or convo.reply_task.done():
            if len(llm_histories) >= 2 or convo.reply_task is None:
                break
    if convo.reply_task is not None:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(convo.reply_task, timeout=2)

    assert len(llm_histories) >= 2, (
        "the trailing 'హలో' cancelled the follow-up owed to the real question"
    )
    carried = [m.get("content") for m in llm_histories[-1]]
    assert "ప్లేస్‌మెంట్ ఉంటుందా?" in carried, (
        "the follow-up ran but never carried the missed question"
    )


async def test_the_holding_line_does_not_count_as_an_answer(monkeypatch):
    """The dangerous interaction: _heard_something_this_turn gates the
    failed-turn counter reset, _say_fallback's early return, AND the nudge
    suppression. If the holding line sets it, a call where every RAG answer
    dies says 'one moment' forever, never reprompts, never trips
    twoway_never_answered, and is recorded as a clean successful call — the
    exact billed-silence-scored-as-success failure the two-way flags gate."""
    _call, convo, _turns = _tool_turn_convo(monkeypatch, [
        _reply(tools=[_tool("search_course_material", query="fees")]),
        sarvam_bridge.conversation_llm.TurnFailed("upstream 500"),
    ])
    tts = _CountingTTS()

    await convo._reply(tts)

    # The reprompt fires ONLY when _heard_something_this_turn was still False
    # when the answer died (_say_fallback early-returns otherwise), and the
    # reprompt itself then legitimately sets the flag — so the flag's state
    # at the decision point is asserted through the reprompt having spoken,
    # not by inspecting the post-call residue.
    assert sarvam_bridge._FALLBACK_REPROMPT in tts.spoken, (
        "the failed answer no longer reprompts — the watchdog chain is disarmed"
    )
    assert convo._failed_turns == 1


async def test_the_holding_line_is_never_the_first_spoken_agent_turn(monkeypatch):
    """call_routes re-checks the FIRST SPOKEN agent turn for the AI
    disclosure and logs [compliance] at ERROR — a stop-dialling event. If the
    opening never spoke, the holding line must not become that first turn."""
    _call, convo, _turns = _tool_turn_convo(monkeypatch, [
        _reply(tools=[_tool("search_course_material", query="fees")]),
        _reply(text="Fee is 25,000."),
    ], opening_spoken=False)
    tts = _CountingTTS()

    await convo._reply(tts)

    assert sarvam_bridge._RAG_HOLDING_LINE not in tts.spoken
    assert any("25,000" in s for s in tts.spoken)


async def test_the_holding_line_is_spoken_once_across_chained_lookups(monkeypatch):
    _call, convo, _turns = _tool_turn_convo(monkeypatch, [
        _reply(tools=[_tool("search_course_material", query="fees")]),
        _reply(tools=[_tool("search_course_material", query="duration")]),
        _reply(text="Fee is 25,000, duration 3 months."),
    ])
    tts = _CountingTTS()

    await convo._reply(tts)

    assert tts.spoken.count(sarvam_bridge._RAG_HOLDING_LINE) == 1


# ── goodbye detection must not fire inside an ordinary word ─────────────────
#
# "బై" (bye) is two codepoints and sits INSIDE common Telugu words: మొబైల్
# (mobile) and బైక్ (bike) both contain it. A raw substring test therefore
# hangs up on a lead who says "send it to my mobile number" — mid-sentence,
# with the agent saying nothing, graded a clean exit, and the lead marked done
# so they are never called back. The Latin path already avoids this with an
# anchored regex; the phrase list did not.

@pytest.mark.parametrize("text", [
    "నా మొబైల్ నంబర్‌కి పంపండి",   # "send it to my mobile number"
    "మొబైల్ మార్కెటింగ్ ఉందా?",      # "is there mobile marketing?"
    "బైక్ కోర్సు ఉందా?",             # "is there a bike course?"
])
def test_a_telugu_word_that_merely_contains_bye_is_not_a_goodbye(text):
    assert not sarvam_bridge._is_lead_goodbye(text), (
        f"{text!r} hung up on a lead who was still asking questions"
    )


@pytest.mark.parametrize("text", ["బై", "థాంక్స్, బై.", "ఇక అంతే", "సరే, ఇక అంతే."])
def test_a_real_telugu_goodbye_is_still_detected(text):
    assert sarvam_bridge._is_lead_goodbye(text)


async def test_a_cancelled_reply_also_forgets_what_the_lead_never_heard():
    """The same repair, on the other path that kills a half-spoken answer.

    truncate_to_what_was_heard was reachable only from on_barge_in. But
    _cancel_reply is called from on_lead_said too, and it cancels say() mid-
    utterance — so the full assistant text stayed in history and in the stored
    transcript even though only part of it played, and the agent went on to
    refer back to sentences the lead never received. That is precisely the
    incoherence the ledger exists to prevent.
    """
    from app.db.models import TranscriptTurn

    call = sarvam_bridge.plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="l", one_way=False, protect_opening=False)
    turns = []
    convo = sarvam_bridge._Conversation(
        call, lead_id="l", system_prompt="s", turns=turns)

    # An answer of two sentences, of which only the first ever played.
    convo._agent_turn_index = 0
    turns.append(TranscriptTurn(role="agent", text="One. Two."))
    convo.history.append({"role": "assistant", "content": "One. Two."})
    convo.ledger.add("One.", 1000)
    convo.ledger.add("Two.", 1000)

    async def still_speaking():
        await asyncio.sleep(3600)

    convo.reply_task = asyncio.create_task(still_speaking())
    await asyncio.sleep(0)

    await convo._cancel_reply()

    claimed = " ".join(m.get("content") or "" for m in convo.history
                       if m.get("role") == "assistant")
    assert "Two." not in claimed, (
        "the model was told it said a sentence the lead never heard"
    )
    assert "Two." not in " ".join(t.text for t in turns), (
        "the stored transcript claims words that were never played"
    )


async def test_a_committed_round_is_still_not_truncated_on_cancel():
    """_cancel_reply returns early for a committed round — it does not cancel
    say(), so there is nothing half-spoken to repair and the turn must be left
    exactly as it is."""
    from app.db.models import TranscriptTurn

    call = sarvam_bridge.plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="l", one_way=False, protect_opening=False)
    turns = []
    convo = sarvam_bridge._Conversation(
        call, lead_id="l", system_prompt="s", turns=turns)
    convo._agent_turn_index = 0
    turns.append(TranscriptTurn(role="agent", text="One. Two."))
    convo.history.append({"role": "assistant", "content": "One. Two."})
    convo.ledger.add("One.", 1000)
    convo.ledger.add("Two.", 1000)

    async def in_flight():
        await asyncio.sleep(3600)

    convo.reply_task = asyncio.create_task(in_flight())
    convo._reply_committed = True
    await asyncio.sleep(0)

    await convo._cancel_reply()

    assert turns[0].text == "One. Two.", "a committed round was wrongly rewritten"
    convo.reply_task.cancel()


async def test_a_fragment_already_folded_into_a_later_round_is_not_answered_twice(
        monkeypatch):
    """The follow-up for a missed fragment must ask "is anything unanswered?".

    A fragment that arrives while the model round is committed is recorded in
    _missed_while_committed. But a tool round re-serialises the whole history,
    so round 2 usually DOES see that fragment and answers both parts. Firing
    the follow-up unconditionally then queried the model again with nothing new
    and the agent spoke an extra, unsolicited turn at the lead.

    _awaiting_answer already answers exactly this question, and on_lead_stopped
    already gates on it.
    """
    _call, convo, turns = _failing_turn_convo()
    replies = []

    async def answered_everything(history):
        replies.append(len(history))
        return _reply(text="Both parts answered.")

    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", answered_everything)
    convo._lead_speaking = False
    convo._missed_while_committed = True   # a fragment landed mid-round
    convo.history.append({"role": "user", "content": "and the fee?"})
    convo._missed_at = len(convo.history)  # ...and the next round will carry it

    await convo._reply_when_they_stop(_CountingTTS())
    for _ in range(4):
        await asyncio.sleep(0)

    assert len(replies) == 1, (
        f"the model was queried {len(replies)} times for one question — the "
        "agent speaks an unsolicited extra turn at the lead"
    )
    assert convo.reply_task is None or convo.reply_task.done()
    if convo.reply_task:
        convo.reply_task.cancel()


async def test_a_hung_script_render_does_not_hold_the_lead_in_silence(
        two_way, monkeypatch):
    """render() runs AFTER Plivo answers the leg but BEFORE PlivoCall.run(),
    so neither the one-way watchdog nor CALL_MAX_DURATION_S bounds it — only
    sarvam_llm's own 60s HTTP timeout. A Sarvam latency spike therefore meant a
    real person answered the phone and heard up to a minute of complete
    silence, on every lead in the campaign, with the credits breaker unable to
    help because it only trips on 'credit'.
    """
    two_way([])

    async def never_returns(script, *, language_style=None):
        await asyncio.sleep(3600)

    monkeypatch.setattr(sarvam_bridge.sarvam_llm, "render", never_returns)
    monkeypatch.setattr(sarvam_bridge, "RENDER_DEADLINE_S", 0.05)

    outcome = await _run(_FakePlivoWS(), one_way=True)

    assert outcome["status"] == "failed"
    assert "render" in (outcome.get("note") or "").lower() or outcome["turns"] == 0


# ── nobody noticed when TTS produced no audio at all ───────────────────────
#
# sarvam_tts.speak() ends its stream on an error frame with ZERO yields and no
# exception (a rejected speaker, a config the server refuses, credits gone).
# say() had already appended the full reply to turns/history before synthesis,
# the ledger recorded each sentence at 0 ms, and _reply then reset
# _failed_turns because say() "returned normally". So the lead heard silence,
# _FALLBACK_REPROMPT — the line built for exactly this symptom — never fired,
# and the transcript certified words nobody heard. If it happened on the
# OPENING, the stored transcript certifies an AI disclosure that was never
# played, and the post-call [compliance] check passes on it.

class _SilentTTS:
    """Sarvam's error-frame shape: the stream simply ends, yielding nothing."""

    async def speak(self, text):
        return
        yield  # pragma: no cover - makes this an async generator


async def test_a_turn_that_produced_no_audio_is_not_counted_as_spoken():
    _call, convo, turns = _failing_turn_convo()

    await convo.say(_SilentTTS(), "ఫీజు ఇరవై ఐదు వేల రూపాయలు.")

    assert convo._heard_something_this_turn is False, (
        "a turn where zero frames reached the lead was recorded as heard"
    )


async def test_a_silent_reply_triggers_the_spoken_fallback(monkeypatch):
    """The recovery line exists for exactly this symptom; it must fire."""
    _call, convo, turns = _failing_turn_convo()

    async def answers(history):
        return _reply(text="ఫీజు ఇరవై ఐదు వేల రూపాయలు.")

    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", answers)
    convo._failed_turns = 0

    await convo._reply(_SilentTTS())

    assert convo._failed_turns >= 1, (
        "a turn that played nothing reset the failure counter instead of "
        "counting as a failure"
    )


async def test_the_ledger_does_not_claim_zero_duration_sentences():
    """heard_within kept 0-ms entries even at played_ms==0, so a later
    barge-in truncate could not strip text that was never audible."""
    ledger = sarvam_bridge.SpokenLedger()
    ledger.add("మొదటి వాక్యం.", 0)
    ledger.add("రెండవ వాక్యం.", 0)

    assert ledger.heard_within(0) == "", (
        "the ledger claimed sentences that produced no audio at all"
    )


async def test_a_call_with_no_audio_at_all_is_not_a_clean_exit(monkeypatch):
    """A Sarvam TTS outage made every call end on twoway_silence, which is in
    _CLEAN_EXITS, so a campaign of totally silent calls was graded 'done' and
    the leads were never retried."""
    from app.telephony import plivo_stream

    call = plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="l", one_way=False, protect_opening=False)
    convo = sarvam_bridge._Conversation(
        call, lead_id="l", system_prompt="s", turns=[])
    monkeypatch.setattr(sarvam_bridge, "TWOWAY_MAX_SILENT_S", 0.05)
    assert call.audio_seen is False        # TTS never produced a frame

    await asyncio.wait_for(convo.watch_for_silence(), timeout=3)

    assert call.exit_reason != "twoway_silence", (
        "a call where the agent was never audible was graded as a normal "
        f"finished conversation ({call.exit_reason!r})"
    )
    assert call.exit_reason not in plivo_stream._CLEAN_EXITS


# ── a call that only ever apologised is not a finished conversation ────────
#
# When the conversation LLM is down, every turn fails into _FALLBACK_REPROMPT,
# then the handoff line promises "మా టీమ్ మిమ్మల్ని త్వరలో సంప్రదిస్తుంది" —
# a callback nothing in this system schedules. twoway_silence then fires and,
# because it is in _CLEAN_EXITS, the call is graded 'done': the lead is marked
# done and never retried, and the operator sees a campaign of successful calls
# while every one of them was an apology and a promise nobody will keep.

async def test_a_call_that_only_apologised_is_not_graded_as_a_finished_call(
        monkeypatch):
    from app.telephony import plivo_stream

    call = plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="l", one_way=False, protect_opening=False)
    convo = sarvam_bridge._Conversation(
        call, lead_id="l", system_prompt="s", turns=[])
    monkeypatch.setattr(sarvam_bridge, "TWOWAY_MAX_SILENT_S", 0.05)

    # the agent WAS audible — it spoke the apologies — but never answered
    call.audio_seen = True
    convo._failed_turns = sarvam_bridge._MAX_FAILED_TURNS

    await asyncio.wait_for(convo.watch_for_silence(), timeout=3)

    assert call.exit_reason not in plivo_stream._CLEAN_EXITS, (
        f"a call that never answered anything was graded a clean finish "
        f"({call.exit_reason!r}) — the lead is marked done and never retried"
    )


async def test_a_normal_conversation_is_still_a_clean_exit(monkeypatch):
    """The guard must not turn ordinary calls into failures: one recovered
    hiccup in an otherwise healthy call is not a broken call."""
    from app.telephony import plivo_stream

    call = plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="l", one_way=False, protect_opening=False)
    convo = sarvam_bridge._Conversation(
        call, lead_id="l", system_prompt="s", turns=[])
    monkeypatch.setattr(sarvam_bridge, "TWOWAY_MAX_SILENT_S", 0.05)
    call.audio_seen = True
    convo._failed_turns = 0

    await asyncio.wait_for(convo.watch_for_silence(), timeout=3)

    assert call.exit_reason == "twoway_silence"
    assert call.exit_reason in plivo_stream._CLEAN_EXITS


class _RaisingTTS:
    """The socket dies mid-utterance, before a single frame is played."""

    async def speak(self, text):
        raise RuntimeError("sarvam socket died mid-utterance")
        yield  # pragma: no cover - makes this an async generator


async def test_a_turn_whose_synthesis_raised_is_not_left_in_the_record():
    """say() appends the turn BEFORE synthesising it, so a speak() that raises
    with zero frames left a phantom turn in turns AND history that nobody
    heard. truncate_to_what_was_heard never runs on that path: converse()
    catches the error and _cancel_reply early-returns because no reply task is
    in flight. When it is the OPENING that dies, the stored transcript
    certifies an AI disclosure that was never played — and call_routes' post-
    call [compliance] check reads that same transcript and passes."""
    _call, convo, turns = _failing_turn_convo()

    with pytest.raises(RuntimeError):
        await convo.say(_RaisingTTS(),
                        "ఇది డిజిటల్ బ్రోలీ నుండి ఆటోమేటెడ్ AI కాల్.")

    assert _agent_lines(turns) == [], f"phantom turn survived: {turns}"
    assert not [m for m in convo.history if m.get("role") == "assistant"], (
        "the model will be told it said something the lead never heard"
    )


async def test_a_turn_that_yielded_nothing_is_not_left_in_the_record():
    """The same hole via the other shape: an error FRAME ends the stream with
    zero yields and no exception at all."""
    _call, convo, turns = _failing_turn_convo()

    await convo.say(_SilentTTS(), "ఫీజు ఇరవై ఐదు వేల రూపాయలు.")

    assert _agent_lines(turns) == [], f"phantom turn survived: {turns}"
    assert not [m for m in convo.history if m.get("role") == "assistant"]


async def test_a_turn_that_did_play_is_still_recorded():
    """The guard must not delete real turns — this is the normal path."""
    _call, convo, turns = _failing_turn_convo()

    await convo.say(_CountingTTS(), "ఫీజు ఇరవై ఐదు వేల రూపాయలు.")

    assert _agent_lines(turns) == ["ఫీజు ఇరవై ఐదు వేల రూపాయలు."]
    assert [m for m in convo.history if m.get("role") == "assistant"]


# ── nudge-only follow-ups ────────────────────────────────────────────────────

def test_a_bare_hello_is_recognised_as_only_a_nudge():
    assert sarvam_bridge._is_only_prompting("హలో")
    assert sarvam_bridge._is_only_prompting("హలో.")
    assert sarvam_bridge._is_only_prompting("హలో హలో")
    assert sarvam_bridge._is_only_prompting("Hello?")
    assert sarvam_bridge._is_only_prompting("చెప్పండి")


def test_a_real_question_ending_in_the_same_word_is_not_a_nudge():
    """The exact false positive substring matching would produce: the live
    call's opening question ends in 'చెప్పండి' and is a genuine request."""
    assert not sarvam_bridge._is_only_prompting(
        "డిజిటల్ మార్కెటింగ్ కోర్స్ డీటెయిల్స్ చెప్పండి.")
    assert not sarvam_bridge._is_only_prompting(
        "ఎస్ఈఓ గురించి చెప్పండి అండ్ ఫీ స్ట్రక్చర్ గురించి కూడా చెప్పండి.")


def test_yes_and_okay_are_never_treated_as_nudges():
    """The agent asks yes/no questions, and on the 2026-08-27 call the lead
    answered one with 'అవును'. Dropping that would strand the conversation."""
    for answer in ("అవును", "ఓకే", "ఆన్లైన్", "సరే"):
        assert not sarvam_bridge._is_only_prompting(answer), answer


def test_empty_text_is_not_a_nudge():
    assert not sarvam_bridge._is_only_prompting("")
    assert not sarvam_bridge._is_only_prompting("   ")


async def _run_nudge_call(monkeypatch, *, tts, nudge="హలో"):
    """Drive the measured 2026-08-27 sequence: the lead asks a question, the
    reply is slow, the lead says *nudge* while that reply is committed, and the
    reply then completes. Returns (llm_call_count, agent_turn_texts)."""
    monkeypatch.setattr(sarvam_bridge, "SARVAM_REPLY_SETTLE_S", 0)
    monkeypatch.setattr(sarvam_bridge, "SARVAM_REPLY_MAX_WAIT_S", 0)

    call = sarvam_bridge.plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="l", one_way=False, protect_opening=False)
    turns = []
    convo = sarvam_bridge._Conversation(
        call, lead_id="l", system_prompt="s", turns=turns)

    calls: list[list[dict]] = []
    first_call_started = asyncio.Event()
    release_first_call = asyncio.Event()

    async def controlled_turn(history):
        calls.append([dict(m) for m in history])
        if len(calls) == 1:
            first_call_started.set()
            await release_first_call.wait()
        return sarvam_bridge.conversation_llm.LLMReply(
            text=f"Answer {len(calls)}.", tool_calls=[])

    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", controlled_turn)

    await convo.on_lead_said(tts, "కోర్స్ డీటెయిల్స్ చెప్పండి.")
    await asyncio.wait_for(first_call_started.wait(), timeout=1)
    await convo.on_lead_said(tts, nudge)
    release_first_call.set()

    # Let any follow-up it decided to make actually run.
    for _ in range(300):
        await asyncio.sleep(0)

    return len(calls), [t.text for t in turns if t.role == "agent"]


async def test_a_hello_during_a_slow_answer_does_not_get_the_answer_repeated(
    monkeypatch,
):
    """The defect measured on the live call of 2026-08-27.

    The lead asked for course details. The reply took 7.5s (llm=6.01s across
    two tool rounds), and partway through the lead said "హలో" to check the line
    was alive. That landed after the second round's request was already sent,
    so the missed-fragment follow-up fired for it — and the model, given "హలో"
    with the just-spoken course list above it, recited the course list again.
    The lead heard the same answer twice.

    The nudge is satisfied by the answer that just played. Nothing is owed.
    """
    class _FakeTTS:
        async def speak(self, text):
            yield _FRAME, 500

    llm_calls, spoken = await _run_nudge_call(monkeypatch, tts=_FakeTTS())

    assert len(spoken) == 1, f"the answer was repeated at the lead: {spoken}"
    assert llm_calls == 1, (
        f"the model was queried again for a bare nudge ({llm_calls} calls)"
    )


async def test_a_hello_is_still_answered_when_the_turn_it_nudged_said_nothing(
    monkeypatch,
):
    """The gate must not swallow a genuine plea into silence.

    If the turn the lead was nudging produced no audio at all, then from the
    lead's side nothing has happened since they asked — their "హలో" is still
    unanswered and suppressing the follow-up would leave them talking to a dead
    line, which is the very failure this backend is most punished for.
    """
    class _SilentTTS:
        async def speak(self, text):
            return
            yield   # pragma: no cover - makes this an async generator

    llm_calls, spoken = await _run_nudge_call(monkeypatch, tts=_SilentTTS())

    assert llm_calls >= 2, (
        "the turn played nothing, so the lead's nudge was still owed an answer "
        f"— but the model was only queried {llm_calls} time(s)"
    )


async def test_cancelled_round_fragments_do_not_poison_later_suppression():
    """Post-fix sweep (low): fragments flagged for a round that then got
    CANCELLED were never drained — a stale real question from a dead round
    later defeated a lone-nudge suppression and the agent re-recited its
    answer. The replacement round re-reads full history (the fragments are
    already in it), so clearing the flags on the cancel path loses nothing."""
    _call, convo, _turns = _failing_turn_convo()
    convo._missed_while_committed = True
    convo._missed_texts = ["ఫీజు ఎంత?"]
    convo.reply_task = asyncio.create_task(asyncio.sleep(10))
    await asyncio.sleep(0)

    await convo._cancel_reply()

    assert convo._missed_texts == []
    assert convo._missed_while_committed is False


# ── a scriptless two-way call must not depend on the renderer ───────────────
#
# Live incident 2026-08-27: a newly created two-way campaign (no script) was
# dialled, Sarvam's render returned nothing after 51s, and both leads
# answered to silence and were hung up on at RENDER_DEADLINE_S. The script
# being rendered was DEFAULT_TWOWAY_SCRIPT — our own fixed English constant —
# so the model was being paid to translate the same sentence every time, and
# the campaign could not dial at all when it failed.

async def test_a_two_way_call_with_no_script_never_calls_the_renderer(two_way, monkeypatch):
    called = {"n": 0}

    async def exploding_render(script, *, language_style=None):
        called["n"] += 1
        raise AssertionError("the default opening must not need the model")

    two_way([])
    monkeypatch.setattr(sarvam_bridge.sarvam_llm, "render", exploding_render)

    outcome = await asyncio.wait_for(
        sarvam_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False,
                             dynamic_variables={"lead_name": "Mouli"}),
        timeout=10,
    )

    assert called["n"] == 0
    agent_lines = [t.text for t in outcome["transcript"] if t.role == "agent"]
    assert agent_lines, "the lead heard nothing at all"
    assert sarvam_bridge.sarvam_prompts.DEFAULT_TWOWAY_TELUGU.split(".")[0] in agent_lines[0]


async def test_a_scriptless_tinglish_call_opens_in_tinglish(two_way):
    """The scriptless opening was ONE constant for both registers, so a
    Tinglish campaign's first words were pure Telugu — and the test guarding
    that constant forbids Latin letters in it, so the register could not be
    fixed in place. The opening is now chosen by the style that arrives,
    exactly as the system prompt already was."""
    from app import languages

    two_way([])

    outcome = await asyncio.wait_for(
        sarvam_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False,
                             dynamic_variables={
                                 "lead_name": "Mouli",
                                 "language_style": languages.style("tinglish"),
                             }),
        timeout=10,
    )

    agent_lines = [t.text for t in outcome["transcript"] if t.role == "agent"]
    assert agent_lines, "the lead heard nothing at all"
    assert sarvam_bridge.sarvam_prompts.DEFAULT_TWOWAY_TINGLISH in agent_lines[0]


async def test_a_campaign_with_its_own_script_still_renders(two_way, monkeypatch):
    """The canned opening covers the scriptless case ONLY — an operator's own
    script still has to be translated."""
    rendered = {"n": 0}

    async def fake_render(script, *, language_style=None):
        rendered["n"] += 1
        return "ఇది కృత్రిమ మేధ ద్వారా చేసే ఆటోమేటెడ్ కాల్. కోర్సు సోమవారం."

    two_way([])
    monkeypatch.setattr(sarvam_bridge.sarvam_llm, "render", fake_render)

    await asyncio.wait_for(
        sarvam_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False,
                             dynamic_variables={"script": "Course starts Monday.",
                                                "lead_name": "Mouli"}),
        timeout=10,
    )

    assert rendered["n"] == 1


# ── the documents are read before the call ──────────────────────────────────
#
# Owner's direction after the 2026-08-29 test round: the agent kept saying
# the waiting line and still missed the fee question, because every fact
# needed a live lookup. A digest of the corpus's core facts (fees, programs,
# duration, placement) is now fetched at call start — cached, built from the
# same search_relevant the tool uses — and rides in the system prompt.

async def test_the_system_prompt_carries_the_course_facts(two_way, monkeypatch):
    seen, _tts, _stt = two_way([_said("ఫీజు ఎంత?")])

    async def fake_facts():
        return "BDLP fee: 25,000 rupees. Duration: 3 months."

    monkeypatch.setattr(sarvam_bridge, "_course_facts", fake_facts)

    await asyncio.wait_for(
        sarvam_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False,
                             dynamic_variables={"lead_name": "Mouli"}),
        timeout=10,
    )

    system = seen["histories"][0][0]
    assert system["role"] == "system"
    assert "BDLP fee: 25,000 rupees." in system["content"], (
        "the pre-read facts never reached the model"
    )
    assert "KNOWN COURSE FACTS" in system["content"]


async def test_a_facts_failure_never_blocks_the_call(two_way, monkeypatch):
    """The digest is an optimisation; the tool remains. RAG being down must
    not stop the call from happening at all."""
    seen, _tts, _stt = two_way([_said("హలో")])

    async def broken_facts():
        raise RuntimeError("embeddings are down")

    monkeypatch.setattr(sarvam_bridge, "_course_facts", broken_facts)

    outcome = await asyncio.wait_for(
        sarvam_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False,
                             dynamic_variables={"lead_name": "Mouli"}),
        timeout=10,
    )

    assert outcome["turns"] > 0, "a digest failure killed the whole call"


async def test_course_facts_builds_from_the_relevance_filtered_search(monkeypatch):
    """search_relevant, never search_permissive — the digest feeds live
    calls, so CLAUDE.md's relevance floor applies to it exactly as to the
    tool. Duplicated chunks across the fixed queries collapse to one."""
    queries = []

    async def fake_search(query, *a, **kw):
        queries.append(query)
        if "fee" in query:
            return "[From fees]\nTotal Fee: 25,000."
        return "[From fees]\nTotal Fee: 25,000."  # duplicate on purpose

    class _FakeRedis:
        def __init__(self):
            self.store = {}

        async def get(self, key):
            return self.store.get(key)

        async def set(self, key, value, ex=None):
            self.store[key] = value

    fake_redis = _FakeRedis()
    monkeypatch.setattr(sarvam_bridge, "search_relevant", fake_search)
    monkeypatch.setattr(sarvam_bridge.redis_client, "get_redis", lambda: fake_redis)

    digest = await sarvam_bridge._course_facts()

    assert len(queries) >= 3, "the digest must sweep the core topics"
    assert digest.count("Total Fee: 25,000.") == 1, "duplicates must collapse"
    assert fake_redis.store, "the digest must cache — it is per corpus, not per call"

    queries.clear()
    again = await sarvam_bridge._course_facts()
    assert queries == [], "a cache hit must not re-embed"
    assert again == digest


async def test_a_hung_digest_fetch_cannot_hold_an_answered_lead_in_silence(two_way, monkeypatch):
    """Adversarial review, 2026-08-29: the digest fetch sits between Plivo
    answering and the opening being spoken, and a HANG is not an exception —
    neither degrade layer fires. The render above it is deadline-bounded and
    the in-call lookups are deadline-bounded; this must be too."""
    seen, _tts, _stt = two_way([_said("హలో")])

    async def hung_facts():
        await asyncio.sleep(60)
        return "too late"

    monkeypatch.setattr(sarvam_bridge, "_course_facts", hung_facts)
    monkeypatch.setattr(sarvam_bridge, "FACTS_DEADLINE_S", 0.1)

    outcome = await asyncio.wait_for(
        sarvam_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False,
                             dynamic_variables={"lead_name": "Mouli"}),
        timeout=10,
    )

    assert outcome["turns"] > 0, "the hung digest held the call hostage"


async def test_the_digest_cap_never_slices_a_fact_mid_number(monkeypatch):
    """The cap was a raw character slice, so char 7000 could land inside
    '₹1,50,000' and the prompt then presents 'Total Fee: ₹1,5' as a fact the
    model must quote directly with no fresh lookup to correct it. Whole
    blocks only: the straddling block is dropped, not halved."""
    async def fake_search(query, *a, **kw):
        return "[From fees]\nTotal Fee: 25,000 rupees for the base program."

    class _FakeRedis:
        async def get(self, key):
            return None

        async def set(self, key, value, ex=None):
            pass

    monkeypatch.setattr(sarvam_bridge, "search_relevant", fake_search)
    monkeypatch.setattr(sarvam_bridge.redis_client, "get_redis", lambda: _FakeRedis())
    monkeypatch.setattr(sarvam_bridge, "_FACTS_MAX_CHARS", 30)

    digest = await sarvam_bridge._course_facts()

    assert digest == "", (
        f"a block was sliced mid-fact instead of dropped: {digest!r}"
    )


async def test_a_failed_digest_build_backs_off_instead_of_stalling_every_call(monkeypatch):
    """During a RAG brownout an empty digest was never cached, so EVERY
    two-way call re-paid the full stall for the whole outage — an answered
    lead in silence each time. One failure now arms a short cooldown."""
    calls = {"n": 0}

    async def failing_search(query, *a, **kw):
        calls["n"] += 1
        raise RuntimeError("embeddings down")

    class _FakeRedis:
        def __init__(self):
            self.store = {}

        async def get(self, key):
            return self.store.get(key)

        async def set(self, key, value, ex=None):
            self.store[key] = value

    fake = _FakeRedis()
    monkeypatch.setattr(sarvam_bridge, "search_relevant", failing_search)
    monkeypatch.setattr(sarvam_bridge.redis_client, "get_redis", lambda: fake)

    assert await sarvam_bridge._course_facts() == ""
    first_round = calls["n"]
    assert await sarvam_bridge._course_facts() == ""

    assert calls["n"] == first_round, (
        "the second call re-ran the failing searches inside the cooldown"
    )


# ── a goodbye must be the WHOLE utterance, not a fragment inside one ────────
#
# Reported live 2026-08-29: "calls cutting in the middle of the conversation
# while the lead is still speaking". _is_lead_goodbye runs on the raw STT
# text and hangs up IMMEDIATELY — no model turn, no confirmation — so any
# garbled transcript that happens to contain a goodbye phrase ends the call
# on someone who was still talking. Sarvam's Telugu STT garbles constantly
# on 8 kHz audio (real examples below), and CLAUDE.md's standard for acting
# on a lead's words is zero false positives.
#
# Three other layers can still end a call (the model's end_call, the silence
# watchdog, CALL_MAX_DURATION_S), so a MISSED goodbye costs one extra turn
# while a FALSE one hangs up on a live customer.

@pytest.mark.parametrize("text", [
    "ఇక అంతే అంతే అంతే.",            # garbled repetition, seen live
    "ఇక అంతే చెప్పండి కోర్సు గురించి",  # "that's all, tell me about the course"
    "బై కాదు, ఇంకా చెప్పండి",          # "not bye, tell me more"
    "that's all the details you sent me, now the fees please",
])
def test_a_goodbye_buried_in_a_longer_utterance_does_not_cut_the_call(text):
    assert not sarvam_bridge._is_lead_goodbye(text), (
        f"{text!r} hung up on a lead who was still talking"
    )


@pytest.mark.parametrize("text", [
    "బై", "థ్యాంక్స్, బై.", "ఇక అంతే", "సరే, ఇక అంతే.",
    "Bye.", "goodbye", "Bye bye", "That's all, thank you.",
    "I have no questions.",
])
def test_a_real_goodbye_is_still_detected_by_code(text):
    assert sarvam_bridge._is_lead_goodbye(text)


async def test_the_watchdog_does_not_hang_up_while_the_lead_is_talking(monkeypatch):
    """The other mid-speech cut: watch_for_silence treats an in-flight REPLY
    as activity but not the lead's own speech, so a lead still mid-utterance
    when the 20s window elapses (Sarvam emits speech_started once, then
    nothing until the transcript) is hung up on mid-sentence."""
    monkeypatch.setattr(sarvam_bridge, "TWOWAY_MAX_SILENT_S", 0.2)
    call = sarvam_bridge.plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="l", one_way=False, protect_opening=False)
    convo = sarvam_bridge._Conversation(
        call, lead_id="l", system_prompt="s", turns=[])
    call.audio_seen = True
    convo.last_activity = convo._loop.time() - 60      # long since anyone spoke
    # exactly what on_barge_in sets when START_SPEECH arrives
    convo._lead_speaking = True                        # ...because they ARE talking
    convo._speech_started_at = convo._loop.time()

    watch = asyncio.create_task(convo.watch_for_silence())
    await asyncio.sleep(0.5)
    still_up = not call.stop.is_set()

    convo._lead_speaking = False                       # they finish
    await asyncio.wait_for(watch, timeout=5)

    assert still_up, "hung up on a lead who was mid-sentence"
    assert call.stop.is_set(), "must still end once they actually stop"


async def test_a_stuck_speaking_flag_cannot_hold_a_call_open_forever(monkeypatch):
    """The flag is set by speech_started and cleared by speech_ended; a lost
    END_SPEECH must not disable the watchdog for the whole call."""
    monkeypatch.setattr(sarvam_bridge, "TWOWAY_MAX_SILENT_S", 0.2)
    monkeypatch.setattr(sarvam_bridge, "_MAX_LEAD_SPEECH_S", 0.3)
    call = sarvam_bridge.plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="l", one_way=False, protect_opening=False)
    convo = sarvam_bridge._Conversation(
        call, lead_id="l", system_prompt="s", turns=[])
    call.audio_seen = True
    convo.last_activity = convo._loop.time() - 60
    convo._lead_speaking = True                        # and never cleared
    convo._speech_started_at = convo._loop.time()

    await asyncio.wait_for(convo.watch_for_silence(), timeout=5)

    assert call.stop.is_set(), "a stuck flag disabled the watchdog entirely"


async def test_the_digest_keeps_facts_and_drops_syllabus_noise(monkeypatch):
    """The digest rides in EVERY request, so its size is paid on every turn —
    measured 2026-08-29 at ~0.6s per turn for 6.8k chars. Most of that was
    numbered curriculum items, tool lists and contact URLs pulled in by the
    retrieval, none of which answer the fee/duration/placement questions the
    digest exists for; the search tool still covers them for the long tail."""
    chunk = "\n".join([
        "[From course material]",
        "- BDLP - Brolly Digital Marketing Launchpad. Total Fee: 25,000.",
        "- BDLP: Course Duration is 3 Months. Includes a 1 Month Internship.",
        "## Module 8: Introduction to {SEO - 2025} - AI POWERED",
        "- 111. What is SEO?",
        "- 112. What are the advantages of SEO?",
        "- 126. What are Keywords?",
        "www.digitalbrolly.com +91 96 96 96 3446 digitalbrolly@gmail.com",
        "- Placement Guarantee and internship support.",
    ])

    async def fake_search(query, *a, **kw):
        return chunk

    class _FakeRedis:
        async def get(self, key):
            return None

        async def set(self, key, value, ex=None):
            pass

    monkeypatch.setattr(sarvam_bridge, "search_relevant", fake_search)
    monkeypatch.setattr(sarvam_bridge.redis_client, "get_redis", lambda: _FakeRedis())

    digest = await sarvam_bridge._course_facts()

    assert "Total Fee: 25,000." in digest
    assert "3 Months" in digest and "1 Month Internship" in digest
    assert "Placement Guarantee" in digest
    assert "111. What is SEO?" not in digest, "numbered syllabus items survived"
    assert "126. What are Keywords?" not in digest
    assert "digitalbrolly@gmail.com" not in digest, "contact noise survived"


async def test_every_topic_gets_a_share_of_the_digest_budget(monkeypatch):
    """First-come-first-served let the FEE query's verbose chunks eat the
    whole cap, so duration/mode/placement facts fell off the end — measured
    2026-08-29: at any cap below 7000 the agent lost 'Course Duration'
    entirely while fee text was still being padded in. Each topic now gets
    its own share, so a verbose topic can never starve a quiet one."""
    # Many medium blocks, exactly how retrieval returns a verbose topic —
    # one oversized block would simply be skipped and starve nobody.
    # Retrieval returns whole document REGIONS — roughly a kilobyte each —
    # so a starved topic's chunk cannot squeeze into leftover budget the way
    # a toy 30-character string would.
    _SEP = "\n\n---\n\n"
    long_fee = _SEP.join(
        f"Fee detail block {i}: " + ("x" * 900) for i in range(8))

    async def fake_search(query, *a, **kw):
        if "fee" in query:
            return long_fee
        if "duration" in query:
            return "BDLP: Course Duration is 3 Months. " + ("y" * 800)
        if "placement" in query:
            return "Placement Guarantee included. " + ("z" * 800)
        return "Programs: BDLP, BDCP. " + ("w" * 800)

    class _FakeRedis:
        async def get(self, key):
            return None

        async def set(self, key, value, ex=None):
            pass

    monkeypatch.setattr(sarvam_bridge, "search_relevant", fake_search)
    monkeypatch.setattr(sarvam_bridge.redis_client, "get_redis", lambda: _FakeRedis())
    monkeypatch.setattr(sarvam_bridge, "_FACTS_MAX_CHARS", 4000)

    digest = await sarvam_bridge._course_facts()

    assert "Course Duration is 3 Months" in digest, "the quiet topic was starved"
    assert "Placement Guarantee" in digest
    assert "Programs: BDLP, BDCP." in digest
    assert len(digest) <= 4000


# ── the model may not hang up on an unanswered question ─────────────────────
#
# Live 2026-08-31, after the prompt-level guards were already in place: the
# lead asked "క్లాసెస్ ... టైమ్ చెప్తారా?" (will you tell me class timings?)
# and the model called end_call instead of answering. Instructions have now
# failed to prevent this three times in different shapes, so the rule moves
# into code: an end_call arriving while the lead's own question is still
# unanswered is REFUSED, and the model is told to answer instead. The silence
# watchdog still ends a call nobody is speaking on, so nothing can hang.

async def test_end_call_is_refused_while_the_leads_question_is_unanswered(monkeypatch):
    _call, convo, turns = _failing_turn_convo()
    turns.append(sarvam_bridge.TranscriptTurn(role="agent", text="opening"))
    convo.history.append({"role": "user", "content": "క్లాసెస్ టైమ్ చెప్తారా?"})
    rounds = {"n": 0}

    async def answers(history):
        rounds["n"] += 1
        if rounds["n"] == 1:
            return _reply(tools=[_tool("end_call", reason="no_more_questions")])
        return _reply(text="క్లాసులు ఉదయం పది గంటలకు.")

    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", answers)
    tts = _CountingTTS()

    await convo._reply(tts)

    assert not _call.stop.is_set(), "hung up on an unanswered question"
    assert _call.exit_reason != "sarvam_end_call"
    assert "క్లాసులు ఉదయం పది గంటలకు." in tts.spoken, "it never answered"
    assert sarvam_bridge._FALLBACK_FAREWELL not in tts.spoken, (
        "the farewell was spoken for an end_call that was refused"
    )


async def test_end_call_after_a_real_goodbye_still_ends_the_call(monkeypatch):
    """The guard must not trap a lead who genuinely wants to go."""
    _call, convo, turns = _failing_turn_convo()
    turns.append(sarvam_bridge.TranscriptTurn(role="agent", text="opening"))
    convo.history.append({"role": "user", "content": "సరే బాయ్"})

    async def farewell(history):
        return _reply(text="ధన్యవాదాలు.", tools=[_tool("end_call", reason="said_goodbye")])

    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", farewell)

    await convo._reply(_CountingTTS())

    assert _call.stop.is_set()
    assert _call.exit_reason == "sarvam_end_call"


async def test_end_call_is_honoured_once_the_question_has_been_answered(monkeypatch):
    _call, convo, turns = _failing_turn_convo()
    turns.append(sarvam_bridge.TranscriptTurn(role="agent", text="opening"))
    convo.history.append({"role": "user", "content": "ఫీజు ఎంత?"})
    convo.history.append({"role": "assistant", "content": "ఇరవై ఐదు వేల రూపాయలు."})

    async def farewell(history):
        return _reply(text="ధన్యవాదాలు.", tools=[_tool("end_call", reason="no_more_questions")])

    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", farewell)

    await convo._reply(_CountingTTS())

    assert _call.stop.is_set(), "refused an end_call after the answer was given"


async def test_the_refusal_cannot_trap_the_lead_forever(monkeypatch):
    """A model that only ever wants to end must eventually be allowed to —
    otherwise a lead who asked a question the agent cannot answer is held on
    a line that will not close."""
    _call, convo, turns = _failing_turn_convo()
    turns.append(sarvam_bridge.TranscriptTurn(role="agent", text="opening"))
    convo.history.append({"role": "user", "content": "అది చెప్తారా?"})

    async def always_ending(history):
        return _reply(tools=[_tool("end_call", reason="no_more_questions")])

    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", always_ending)

    for _ in range(sarvam_bridge._MAX_END_CALL_REFUSALS + 2):
        if _call.stop.is_set():
            break
        await convo._reply(_CountingTTS())

    assert _call.stop.is_set(), "the lead can never get off this call"


async def test_the_greeting_says_the_leads_name_in_telugu(two_way, monkeypatch):
    """Owner report after a live call 2026-08-31: "Mouli గారు" was not
    pronounced properly, because the name reaches the greeting in LATIN
    letters and Sarvam's Telugu TTS reads a Latin run with English phonetics.
    Same defect as the Latin "AI" fixed in the opening the same day."""
    two_way([])

    async def fake_name(name):
        assert name == "Mouli"
        return "మౌళి"

    monkeypatch.setattr(sarvam_bridge.lead_name, "telugu_name", fake_name)

    outcome = await asyncio.wait_for(
        sarvam_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False,
                             dynamic_variables={"lead_name": "Mouli"}),
        timeout=10,
    )

    opening = [t.text for t in outcome["transcript"] if t.role == "agent"][0]
    assert "మౌళి" in opening, opening
    assert "Mouli" not in opening, "the Latin spelling was still spoken"


async def test_a_slow_name_lookup_cannot_delay_the_opening(two_way, monkeypatch):
    """It runs before the lead has heard anything, so it is deadline-bounded
    exactly like the course-facts digest beside it."""
    two_way([])

    async def hung_name(name):
        await asyncio.sleep(60)
        return "too late"

    monkeypatch.setattr(sarvam_bridge.lead_name, "telugu_name", hung_name)
    monkeypatch.setattr(sarvam_bridge, "FACTS_DEADLINE_S", 0.1)

    outcome = await asyncio.wait_for(
        sarvam_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False,
                             dynamic_variables={"lead_name": "Mouli"}),
        timeout=10,
    )

    opening = [t.text for t in outcome["transcript"] if t.role == "agent"][0]
    assert "Mouli" in opening, "the greeting lost the lead's name entirely"


async def test_a_failed_name_lookup_still_greets_the_lead(two_way, monkeypatch):
    two_way([])

    async def broken_name(name):
        raise RuntimeError("transliteration down")

    monkeypatch.setattr(sarvam_bridge.lead_name, "telugu_name", broken_name)

    outcome = await asyncio.wait_for(
        sarvam_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False,
                             dynamic_variables={"lead_name": "Mouli"}),
        timeout=10,
    )

    agent_turns = [t.text for t in outcome["transcript"] if t.role == "agent"]
    assert agent_turns and "Mouli" in agent_turns[0]


async def test_the_name_and_the_digest_are_fetched_concurrently(two_way, monkeypatch):
    """Both happen before the lead hears a word, so running them one after
    the other stacks two deadlines' worth of silence onto a cold start. They
    are independent — fetch them together."""
    two_way([])
    order = []

    async def slow_name(name):
        order.append("name-start")
        await asyncio.sleep(0.30)
        order.append("name-end")
        return "మౌళి"

    async def slow_facts():
        order.append("facts-start")
        await asyncio.sleep(0.30)
        order.append("facts-end")
        return "FEE: 25,000"

    monkeypatch.setattr(sarvam_bridge.lead_name, "telugu_name", slow_name)
    monkeypatch.setattr(sarvam_bridge, "_course_facts", slow_facts)

    await asyncio.wait_for(
        sarvam_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False,
                             dynamic_variables={"lead_name": "Mouli"}),
        timeout=10,
    )

    # BOTH starts precede BOTH ends — that is what concurrent means, and it
    # is immune to the unrelated socket setup the rest of bridge() does.
    assert order.index("name-start") < order.index("facts-end")
    assert order.index("facts-start") < order.index("name-end")


async def test_a_slow_digest_does_not_cost_the_greeting_its_telugu_name(monkeypatch):
    """Live 2026-08-31: the greeting said "నమస్తే Mouli గారు" in Latin even
    though the name was cached and resolved in 0.01s. The digest cache had
    expired that minute, its rebuild blew the shared deadline, and one
    asyncio.wait_for around the gather cancelled BOTH — so a slow fetch of
    something optional cost the lead the pronunciation of their own name.
    Independent lookups need independent deadlines."""
    monkeypatch.setattr(sarvam_bridge, "FACTS_DEADLINE_S", 0.2)

    async def fast_name(name):
        return "మౌలి"

    async def slow_facts():
        await asyncio.sleep(5)
        return "FEE: 25,000"

    monkeypatch.setattr(sarvam_bridge.lead_name, "telugu_name", fast_name)
    monkeypatch.setattr(sarvam_bridge, "_course_facts", slow_facts)

    name, facts = await sarvam_bridge._prepare_opening("l", "Mouli")

    assert name == "మౌలి", "the cached name was cancelled by the slow digest"
    assert facts == "", "the slow digest should have degraded to the tool path"


async def test_a_slow_name_does_not_cost_the_call_its_course_facts(monkeypatch):
    """The same independence, the other way round."""
    monkeypatch.setattr(sarvam_bridge, "FACTS_DEADLINE_S", 0.2)

    async def slow_name(name):
        await asyncio.sleep(5)
        return "మౌలి"

    async def fast_facts():
        return "FEE: 25,000"

    monkeypatch.setattr(sarvam_bridge.lead_name, "telugu_name", slow_name)
    monkeypatch.setattr(sarvam_bridge, "_course_facts", fast_facts)

    name, facts = await sarvam_bridge._prepare_opening("l", "Mouli")

    assert name == "Mouli", "should fall back to the written name"
    assert facts == "FEE: 25,000", "the ready facts were thrown away"


async def test_a_timed_out_digest_keeps_building_so_the_next_call_has_it(monkeypatch):
    """The rebuild is most of a second's work that the NEXT call would
    otherwise repeat. Abandoning the wait must not abandon the work — the
    task runs on and fills the cache, so an expiry costs one call its facts
    rather than every call until someone warms it by hand."""
    monkeypatch.setattr(sarvam_bridge, "FACTS_DEADLINE_S", 0.1)
    finished = asyncio.Event()

    async def slow_facts():
        await asyncio.sleep(0.3)
        finished.set()
        return "FEE: 25,000"

    async def fast_name(name):
        return "మౌలి"

    monkeypatch.setattr(sarvam_bridge.lead_name, "telugu_name", fast_name)
    monkeypatch.setattr(sarvam_bridge, "_course_facts", slow_facts)

    _name, facts = await sarvam_bridge._prepare_opening("l", "Mouli")
    assert facts == ""

    await asyncio.wait_for(finished.wait(), timeout=3)


# ── the one boundary the turn-latency line never measured ───────────────────
#
# t0 -> t1 in the standard voice-agent breakdown: the lead stops speaking,
# and some time later the recogniser decides they are finished and emits the
# final transcript. Everything AFTER that has been instrumented since the
# line was added (gate, llm, tools, tts), and `total` is measured from the
# transcript — so an endpointing delay is time the lead waits that appears in
# no number anywhere, and "the agent is slow to answer" cannot be attributed
# without it.

async def test_the_turn_latency_line_reports_the_recogniser_wait(caplog):
    _call, convo, turns = _failing_turn_convo()
    turns.append(sarvam_bridge.TranscriptTurn(role="agent", text="opening"))

    convo.on_lead_stopped(_CountingTTS())          # END_SPEECH: lead finished
    convo._speech_ended_at = convo._loop.time() - 0.9   # ...0.9s ago
    convo._turn_opened_at = convo._loop.time()
    convo._turn_stt_s = 0.9

    with caplog.at_level("INFO"):
        convo._log_turn_latency(spoke=True)

    assert "stt=0.9" in caplog.text, caplog.text


async def test_the_recogniser_wait_is_measured_from_end_of_speech(monkeypatch):
    """Sarvam can deliver END_SPEECH and the transcript in either order (the
    reply gate exists because of it), so this must never report a negative or
    nonsense wait when the transcript arrives first."""
    _call, convo, _turns = _failing_turn_convo()
    monkeypatch.setattr(sarvam_bridge, "SARVAM_REPLY_SETTLE_S", 0.0)
    monkeypatch.setattr(sarvam_bridge, "SARVAM_REPLY_MAX_WAIT_S", 0.01)

    async def quiet(history):
        return _reply(text="సరే.")

    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", quiet)

    # transcript FIRST, END_SPEECH never seen for it
    convo._speech_ended_at = None
    await convo.on_lead_said(_CountingTTS(), "ఫీజు ఎంత?")

    assert convo._turn_stt_s == 0.0, (
        f"invented a recogniser wait of {convo._turn_stt_s}s with no END_SPEECH"
    )


async def test_a_foreign_script_slip_is_logged_for_the_operator(caplog):
    """The sentence is rescued silently by normalize_spoken_telugu, which is
    right for the lead — but a model emitting Armenian mid-Telugu is a
    quality signal nobody should have to find by reading transcripts."""
    _call, convo, _turns = _failing_turn_convo()

    with caplog.at_level("WARNING"):
        await convo.say(_CountingTTS(), "బ్యాచ్ వివరాలు నాకు ఈ պահին లేవు.")

    assert "պահին" in caplog.text
    assert "not telugu" in caplog.text.lower() or "foreign" in caplog.text.lower()


# ── the call ends when the LEAD says goodbye, and not before ────────────────
#
# Owner's instruction after the 2026-09-01 call: "it should cut the call when
# the lead says bye or thank you bye" — nothing else. That call ended on a
# bare "ఆ ఓకే." The earlier guard only refused an end_call when the lead's
# last words were a QUESTION, and "ఓకే" is not a question, so the model was
# allowed to hang up on someone who had just agreed with it.
#
# Refusing is safe: watch_for_silence still ends a call nobody is speaking
# on, so a lead who simply stops talking is never trapped.

async def test_the_call_is_not_ended_on_a_bare_okay(monkeypatch):
    _call, convo, turns = _failing_turn_convo()
    turns.append(sarvam_bridge.TranscriptTurn(role="agent", text="opening"))
    convo.history.append({"role": "user", "content": "ఆ ఓకే."})
    rounds = {"n": 0}

    async def answers(history):
        rounds["n"] += 1
        if rounds["n"] == 1:
            return _reply(tools=[_tool("end_call", reason="no_more_questions")])
        return _reply(text="ఇంకేమైనా తెలుసుకోవాలా?")

    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", answers)
    tts = _CountingTTS()

    await convo._reply(tts)

    assert not _call.stop.is_set(), "hung up on a lead who only said okay"
    assert sarvam_bridge._FALLBACK_FAREWELL not in tts.spoken


@pytest.mark.parametrize("farewell", ["బై", "సరే బాయ్", "థ్యాంక్స్, బై.",
                                      "thank you bye", "ఇక అంతే"])
async def test_a_real_goodbye_ends_the_call_immediately(monkeypatch, farewell):
    """What the owner asked FOR — these must still hang up at once."""
    _call, convo, turns = _failing_turn_convo()
    turns.append(sarvam_bridge.TranscriptTurn(role="agent", text="opening"))
    convo.history.append({"role": "user", "content": farewell})

    async def bye(history):
        return _reply(text="ధన్యవాదాలు.",
                      tools=[_tool("end_call", reason="said_goodbye")])

    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", bye)

    await convo._reply(_CountingTTS())

    assert _call.stop.is_set(), f"{farewell!r} should have ended the call"
    assert _call.exit_reason == "sarvam_end_call"
