"""telephony/call_summary.py — a two-line English note per call.

What a sales team actually reads. Until 2026-09-01 the console's Summary
column and the Sheet's Notes column said nothing useful for a Sarvam call —
`summary` was always None and Notes was "N turns" — because the only summary
the system had ever produced came from ElevenLabs' own post-call analysis,
which the Sarvam and OpenAI bridges never see. The team lead's first question
about any call is "was the lead interested, and what did they ask?"; the
transcript answers it, in Telugu, six clicks away.

Shape of the note:  ``<Tag> — <one or two sentences>``  where the tag is one
of Interested / Not interested / Callback / No answer / Unclear, and the rest
is what the lead asked or said plus any next step they asked for. English,
because that is the language of the Sheet's Status and Notes columns and of
the team that works from them.

Three rules, all learned elsewhere in this codebase:
- It runs AFTER the outcome row is written and is never awaited by the call:
  a slow or failed summary costs a blank note, nothing else (CLAUDE.md's
  "never block a call on the Sheets mirror", applied one step earlier).
- It uses only the transcript. The model is told not to invent a fact, a
  price or a promise — the same failure the RAG rule exists to prevent.
- A one-way call, or a two-way call where the lead never spoke, gets a fixed
  note without a model call. There is nothing to summarise, and a model asked
  to summarise nothing produces something.
"""

from __future__ import annotations

import asyncio
import logging

from app.config import CALL_SUMMARY_ENABLED, CALL_SUMMARY_TIMEOUT_S
from app.telephony import conversation_llm

logger = logging.getLogger(__name__)

TAGS = ("Interested", "Not interested", "Callback", "No answer", "Unclear")

ONE_WAY_NOTE = "One-way message delivered."
NO_SPEECH_NOTE = "No answer — the lead did not say anything."

# Longer than any note needs; shorter than a spreadsheet cell wants.
_MAX_NOTE_CHARS = 300
# A five-minute two-way call is well under this. Past it, the tail matters
# most (the lead's decision is at the end), so the head is cut, not the tail.
_MAX_TRANSCRIPT_CHARS = 8000

_SYSTEM = (
    "You write a short note for the sales team of Digital Brolly, a "
    "digital-marketing training institute in Hyderabad, about a phone call "
    "its AI agent just made. The transcript may be in Telugu, Tinglish "
    "(Telugu mixed with English) or English; write the note in plain "
    "English.\n"
    "Reply with exactly two lines.\n"
    "Line 1: one of these tags and nothing else — Interested, Not "
    "interested, Callback, No answer, Unclear.\n"
    "Line 2: one or two short sentences: what the lead asked or said, and "
    "any next step they asked for.\n"
    "Use only what is in the transcript. Never invent a fact, a price or a "
    "promise. If the lead said almost nothing, say so."
)


def _role(turn) -> str:
    role = turn.get("role") if isinstance(turn, dict) else getattr(turn, "role", "")
    return "agent" if (role or "") in ("agent", "assistant") else "lead"


def _text(turn) -> str:
    text = turn.get("text") if isinstance(turn, dict) else getattr(turn, "text", "")
    return (text or "").strip()


def format_note(text: str | None) -> str | None:
    """The model's reply as a one-line note, or None if there is nothing in it.

    Tolerant of the two ways a model departs from the two-line format: a
    tag with trailing punctuation, and no tag line at all (then the whole
    reply is the note and it is tagged Unclear rather than trusted).
    """
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return None
    first = lines[0].rstrip(".:—- ").strip()
    tag = next((t for t in TAGS if t.lower() == first.lower()), None)
    if tag is None:
        tag, rest = "Unclear", " ".join(lines)
    else:
        rest = " ".join(lines[1:])
    note = f"{tag} — {rest}" if rest else tag
    return note[:_MAX_NOTE_CHARS]


async def summarize(transcript, *, one_way: bool | None) -> str | None:
    """The note for one finished call, or None when there is no note to give.

    *one_way* is None when the caller does not know the campaign's mode; a
    transcript with no lead speech is then left without a note rather than
    guessed at, because "No answer" would be wrong for a delivered message.
    Never raises.
    """
    if not CALL_SUMMARY_ENABLED:
        return None
    if one_way:
        return ONE_WAY_NOTE
    turns = [t for t in (transcript or []) if _text(t)]
    if not any(_role(t) == "lead" for t in turns):
        return NO_SPEECH_NOTE if one_way is False else None

    lines = "\n".join(f"{_role(t)}: {_text(t)}" for t in turns)
    if len(lines) > _MAX_TRANSCRIPT_CHARS:
        lines = "[earlier part omitted]\n" + lines[-_MAX_TRANSCRIPT_CHARS:]
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Transcript:\n{lines}"},
    ]
    try:
        reply = await asyncio.wait_for(
            conversation_llm.turn(messages, tools=[], temperature=0.2),
            CALL_SUMMARY_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 - incl. TimeoutError; a note must never cost the call
        logger.warning(
            f"[summary] could not summarise the call ({type(exc).__name__}: "
            f"{exc}) — the note stays blank"
        )
        return None
    return format_note(reply.text)
