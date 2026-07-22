"""compliance/disclosure.py — AI-disclosure-as-first-line enforcement.

CLAUDE.md hard rule: "AI disclosure as the FIRST line of every script
(enforced by a test, not just review)." This is that enforcement point —
app/db/campaigns.py's create_campaign() calls has_ai_disclosure() and refuses
to create a campaign whose script doesn't pass, so a missing disclosure can
never reach a real dial, not just get flagged in review.

Pure logic, no I/O. Detection is keyword-based, not semantic — TCCCPR doesn't
mandate exact wording, so this looks for a recognisable marker that the CALLER
IS A MACHINE, in the script's opening only.

Two things this deliberately gets right, both of which a naive `"ai" in first
sentence` check gets wrong:

  1. The brand name is not a disclosure. This business trades as "AI Skills"
     (aiskills.in), so a bare \\bai\\b match accepted
     "Hello from AI Skills, we have a new course." — a script that discloses
     nothing whatsoever — as compliant. Brand names are stripped BEFORE
     looking for a marker, so the company's own name can never be mistaken
     for telling someone they're talking to a bot.

  2. A BARE greeting before the disclosure is still a disclosure. Splitting on
     the first [.!?] rejected "Namaste! This is an AI voice assistant..."
     because the "first sentence" was just "Namaste" — a false negative that
     blocks a perfectly compliant script and sends the operator round in
     circles. So a first sentence that is only a greeting rolls the window
     forward to the next one.

     This does NOT weaken the "first line" rule, because the roll-forward is
     conditional on the first sentence saying nothing substantive.
     "Namaste, this is Digital Brolly. By the way, this is an AI call." still
     fails: its first sentence identifies the caller WITHOUT disclosing, which
     is precisely the non-compliant pattern of sounding human up front and
     admitting the machine later.
"""

from __future__ import annotations

import re
import unicodedata

_SENTENCE_END = re.compile(r"[.!?\n]")

# A first sentence of at most this many words counts as a BARE greeting
# ("Namaste!", "Hello there!") and rolls the window forward to the next
# sentence. Anything longer is already making a substantive claim, so the
# disclosure has to be in it — see the module docstring, point 2.
_GREETING_MAX_WORDS = 3
_MAX_OPENING_CHARS = 200

# Stripped before marker matching so the company's OWN name can't satisfy the
# disclosure. Order matters only in that longer forms should precede shorter
# ones. Matched case-insensitively as whole words.
_BRAND_NAMES = ("ai skills", "aiskills", "digital brolly")

_AI_WORD = re.compile(r"\bai\b", re.IGNORECASE)
# Substring markers beyond bare "AI" — checked case-insensitively, so no word
# boundary needed (none of these are common English substrings of other words).
# Each must convey MACHINE-ness on its own; a marker that merely names the
# company or the product does not belong here.
_DISCLOSURE_MARKERS = (
    "artificial intelligence",
    "automated voice",
    "automated call",
    "voice assistant",
    "voice bot",
    "virtual assistant",
    "కృత్రిమ మేధ",  # Telugu: "artificial intelligence"
    "ఆటోమేటెడ్ కాల్",  # Telugu: "automated call"
    "ఆటోమేటెడ్ వాయిస్",  # Telugu: "automated voice"
    "एआई",  # Hindi: "AI" spelled out in Devanagari
    "कृत्रिम बुद्धिमत्ता",  # Hindi: "artificial intelligence"
    "स्वचालित कॉल",  # Hindi: "automated call"
    "स्वचालित आवाज़",  # Hindi: "automated voice"
    "वॉइस असिस्टेंट",  # Hindi: "voice assistant" (transliterated, इ spelling)
    "वॉयस असिस्टेंट",  # Hindi: "voice assistant" (transliterated, य spelling)
)

# Devanagari nukta letters (e.g. ज़) have two encodings that render identically
# — a precomposed codepoint, or a base letter + combining nukta (U+093C) — and
# ordinary Indian-language keyboards produce both interchangeably. NFC is the
# right normalization even though a nukta letter's composition is a Unicode
# "script-specific exclusion": NFC's decompose-then-recompose pipeline still
# canonically DEcomposes both forms and then skips recomposing the excluded
# character, so both encodings land on the same decomposed result. Applying it
# to both the script under test and these marker literals means a marker
# written in one encoding still matches a script typed in the other — see
# tests/unit/test_disclosure.py's nukta tests.
_DISCLOSURE_MARKERS = tuple(unicodedata.normalize("NFC", marker) for marker in _DISCLOSURE_MARKERS)


def _strip_brand_names(text: str) -> str:
    """Remove the company's own trading names so they can't be misread as a
    disclosure. See this module's docstring, point 1."""
    out = text
    for brand in _BRAND_NAMES:
        out = re.sub(rf"\b{re.escape(brand)}\b", " ", out, flags=re.IGNORECASE)
    return out


def _split_first_sentence(text: str) -> tuple[str, str]:
    match = _SENTENCE_END.search(text)
    if match is None:
        return text, ""
    return text[: match.start()], text[match.end() :]


def _is_bare_greeting(sentence: str) -> bool:
    """True for an opener that asserts nothing — "Namaste!", "Hello there!".
    Once a sentence names the caller it is no longer bare, and the disclosure
    belongs in it."""
    return len(sentence.split()) <= _GREETING_MAX_WORDS


def _opening(script: str) -> str:
    """The part of *script* a disclosure has to appear in — see the module
    docstring, point 2."""
    first, remainder = _split_first_sentence(script)
    if _is_bare_greeting(first):
        second, _ = _split_first_sentence(remainder)
        return f"{first} {second}"[:_MAX_OPENING_CHARS]
    return first[:_MAX_OPENING_CHARS]


def has_ai_disclosure(script: str | None) -> bool:
    """True only if *script*'s opening contains a recognisable marker that the
    caller is a machine. None/blank scripts fail outright — no script means no
    disclosure was ever spoken."""
    if not script or not script.strip():
        return False
    opening = _strip_brand_names(_opening(script))
    if _AI_WORD.search(opening):
        return True
    # NFC both sides: the script under test may use either encoding of a
    # nukta letter, and _DISCLOSURE_MARKERS was normalized once at import
    # time above — see the comment there.
    lowered = unicodedata.normalize("NFC", opening).lower()
    return any(marker in lowered for marker in _DISCLOSURE_MARKERS)
