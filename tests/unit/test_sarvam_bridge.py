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
import json

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
    loop = asyncio.get_event_loop()
    convo.last_activity = loop.time() - 60
    call.play_end = loop.time() + 0.6

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


async def test_a_turn_that_says_nothing_is_still_logged(two_way, caplog, monkeypatch):
    """The 'agent never answered me' symptom. A turn swallowed by _reply's
    catch-all used to leave no trace of how long the lead waited for it."""
    caplog.set_level("INFO")

    async def failing_turn(history):
        raise sarvam_bridge.conversation_llm.TurnFailed("boom")

    two_way([_said("Fee enta?")])
    monkeypatch.setattr(sarvam_bridge.conversation_llm, "turn", failing_turn)

    await _run(_FakePlivoWS(), one_way=False)

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
