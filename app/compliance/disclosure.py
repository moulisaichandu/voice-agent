"""compliance/disclosure.py — AI-disclosure-as-first-line enforcement.

CLAUDE.md hard rule: "AI disclosure as the FIRST line of every script
(enforced by a test, not just review)." This is that enforcement point —
app/db/campaigns.py's create_campaign() calls has_ai_disclosure() and refuses
to create a campaign whose script doesn't pass, so a missing disclosure can
never reach a real dial, not just get flagged in review.

Pure logic, no I/O. Detection is keyword-based, not semantic — TCCCPR doesn't
mandate exact wording, so this checks for a recognisable disclosure marker
(English "AI"/"artificial intelligence", or Telugu "కృత్రిమ మేధ", the
standard Telugu rendering of "artificial intelligence") in the FIRST sentence
only — a script that discloses AI involvement later, or not at all, fails.
"""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"[.!?\n]")
_AI_WORD = re.compile(r"\bai\b", re.IGNORECASE)
# Substring markers beyond bare "AI" — checked case-insensitively, so no word
# boundary needed (none of these are common English substrings of other words).
_DISCLOSURE_MARKERS = ("artificial intelligence", "automated voice", "voice assistant",
                       "కృత్రిమ మేధ")


def _first_sentence(script: str) -> str:
    match = _SENTENCE_END.search(script)
    return script[: match.start()] if match else script


def has_ai_disclosure(script: str | None) -> bool:
    """True only if *script*'s first sentence contains a recognisable AI
    disclosure marker. None/blank scripts fail outright — no script means no
    disclosure was ever spoken."""
    if not script or not script.strip():
        return False
    first = _first_sentence(script)
    if _AI_WORD.search(first):
        return True
    lowered = first.lower()
    return any(marker in lowered for marker in _DISCLOSURE_MARKERS)
