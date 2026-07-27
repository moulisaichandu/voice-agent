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


# ── ending the call ──────────────────────────────────────────────────────────

async def test_the_agent_can_end_the_call_cleanly(two_way):
    """'sarvam_end_call' must be in plivo_stream._CLEAN_EXITS. Without it every
    conversation the agent ended properly grades as FAILED, marks the lead
    failed, and burns a retry on someone who already said goodbye."""
    two_way([_said("Thanks, bye.")],
            replies=[_reply(text="Dhanyavadalu.", tools=[_tool("end_call")])])

    outcome = await _run(_FakePlivoWS(), one_way=False)

    assert outcome["status"] == "done"


# ── barge-in ─────────────────────────────────────────────────────────────────

async def test_barge_in_forgets_what_the_lead_never_heard(two_way):
    """THE bug the previous plan warned about inheriting.

    OpenAI Realtime is told what the lead actually heard via
    conversation.item.truncate and repairs its own history. Here the history is
    ours: if we keep the whole answer after an interruption, every later turn
    is built on words nobody heard, and the agent starts referring back to
    things it never said.

    Asserted on the history handed to the NEXT turn, which is where the damage
    would actually show up."""
    seen, _tts, _stt = two_way(
        [_barge_in(), _said("Aagandi.")],
        replies=[_reply(text="Ok.")],
    )

    await _run(_FakePlivoWS(), one_way=False)

    assert seen["histories"], "the lead's turn should have reached the model"
    last = seen["histories"][-1]
    assistant_text = " ".join(m.get("content") or "" for m in last
                              if m.get("role") == "assistant")
    assert _TELUGU not in assistant_text, (
        "the history still claims the agent said an opening the lead cut off"
    )


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
