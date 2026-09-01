"""app/telephony/call_summary.py — the two-line note a sales team reads.

Until 2026-09-01 every Sarvam call had summary=None and the Sheet's Notes
said "N turns". These tests fake the single model seam (conversation_llm.turn)
and pin the shape of the note, the cases that must not call a model at all,
and the rule that a failed or slow summary never raises.
"""

import asyncio

import pytest

from app.db.models import TranscriptTurn
from app.telephony import call_summary, conversation_llm


@pytest.fixture
def enabled(monkeypatch):
    """conftest pins CALL_SUMMARY_ENABLED off for the rest of the suite so
    the call-finalisation tests never reach for a model or the database."""
    monkeypatch.setattr(call_summary, "CALL_SUMMARY_ENABLED", True)
    monkeypatch.setattr(call_summary, "CALL_SUMMARY_TIMEOUT_S", 2.0)


def _two_way():
    return [
        TranscriptTurn(role="agent", text="నమస్తే మౌలి గారు. ఇది ఏఐ ద్వారా చేసే ఆటోమేటెడ్ కాల్."),
        TranscriptTurn(role="lead", text="కోర్సు ఫీజు ఎంత?"),
        TranscriptTurn(role="agent", text="యాభై వేల రూపాయలు."),
        TranscriptTurn(role="lead", text="సరే, రేపు కాల్ చేయండి."),
    ]


def _model_says(monkeypatch, text, *, seen=None):
    async def fake_turn(messages, **kwargs):
        if seen is not None:
            seen["messages"] = messages
            seen["kwargs"] = kwargs
        return conversation_llm.LLMReply(text=text, tool_calls=[])

    monkeypatch.setattr(call_summary.conversation_llm, "turn", fake_turn)


async def test_the_note_is_a_tag_and_what_the_lead_said(enabled, monkeypatch):
    seen = {}
    _model_says(monkeypatch, "Callback\nAsked the course fee (₹50,000) and "
                             "wants a call back tomorrow.", seen=seen)

    note = await call_summary.summarize(_two_way(), one_way=False)

    assert note == ("Callback — Asked the course fee (₹50,000) and wants a "
                    "call back tomorrow.")
    # The whole conversation reaches the model, roles named, and it is asked
    # for a summary — not a conversational turn with tools to call.
    assert seen["kwargs"] == {"tools": [], "temperature": 0.2}
    user = seen["messages"][-1]["content"]
    assert "lead: కోర్సు ఫీజు ఎంత?" in user and "agent: యాభై వేల రూపాయలు." in user
    assert "Never invent" in seen["messages"][0]["content"]


async def test_a_reply_without_the_tag_line_is_marked_unclear_not_trusted(
        enabled, monkeypatch):
    _model_says(monkeypatch, "The lead asked about fees and said okay.")

    note = await call_summary.summarize(_two_way(), one_way=False)

    assert note == "Unclear — The lead asked about fees and said okay."


@pytest.mark.parametrize("first_line", ["Interested.", "interested", "INTERESTED —"])
async def test_the_tag_survives_case_and_trailing_punctuation(
        enabled, monkeypatch, first_line):
    _model_says(monkeypatch, f"{first_line}\nWants the BDCP batch timings.")

    note = await call_summary.summarize(_two_way(), one_way=False)

    assert note == "Interested — Wants the BDCP batch timings."


async def test_a_one_way_call_gets_a_fixed_note_and_no_model_call(enabled, monkeypatch):
    async def never(*a, **kw):
        raise AssertionError("a one-way call has nothing to summarise")

    monkeypatch.setattr(call_summary.conversation_llm, "turn", never)

    note = await call_summary.summarize(
        [TranscriptTurn(role="agent", text="నమస్తే.")], one_way=True)

    assert note == call_summary.ONE_WAY_NOTE


async def test_a_two_way_call_where_the_lead_never_spoke_is_no_answer(
        enabled, monkeypatch):
    async def never(*a, **kw):
        raise AssertionError("nothing to summarise")

    monkeypatch.setattr(call_summary.conversation_llm, "turn", never)

    note = await call_summary.summarize(
        [TranscriptTurn(role="agent", text="నమస్తే."),
         TranscriptTurn(role="agent", text="ధన్యవాదాలు.")], one_way=False)

    assert note == call_summary.NO_SPEECH_NOTE


async def test_an_unknown_mode_with_no_lead_speech_gets_no_note_rather_than_a_wrong_one(
        enabled, monkeypatch):
    """"No answer" on a delivered one-way message would be false; when the
    caller cannot say which it was, say nothing."""
    note = await call_summary.summarize(
        [TranscriptTurn(role="agent", text="నమస్తే.")], one_way=None)

    assert note is None


async def test_a_failing_model_costs_a_blank_note_and_nothing_else(
        enabled, monkeypatch, caplog):
    async def boom(*a, **kw):
        raise conversation_llm.TurnFailed("openai returned HTTP 500")

    monkeypatch.setattr(call_summary.conversation_llm, "turn", boom)

    note = await call_summary.summarize(_two_way(), one_way=False)

    assert note is None
    assert "the note stays blank" in caplog.text


async def test_a_slow_model_is_cut_off_at_the_timeout(enabled, monkeypatch):
    monkeypatch.setattr(call_summary, "CALL_SUMMARY_TIMEOUT_S", 0.02)

    async def slow(*a, **kw):
        await asyncio.sleep(0.5)
        return conversation_llm.LLMReply(text="Interested\nlate", tool_calls=[])

    monkeypatch.setattr(call_summary.conversation_llm, "turn", slow)

    assert await call_summary.summarize(_two_way(), one_way=False) is None


async def test_disabled_means_no_note_and_no_model(monkeypatch):
    monkeypatch.setattr(call_summary, "CALL_SUMMARY_ENABLED", False)

    async def never(*a, **kw):
        raise AssertionError("disabled must not call the model")

    monkeypatch.setattr(call_summary.conversation_llm, "turn", never)

    assert await call_summary.summarize(_two_way(), one_way=False) is None


def test_a_runaway_note_is_cut_to_cell_size():
    note = call_summary.format_note("Interested\n" + "x" * 1000)
    assert len(note) == 300
    assert note.startswith("Interested — ")


def test_an_empty_reply_is_no_note():
    assert call_summary.format_note("") is None
    assert call_summary.format_note("   \n  ") is None
