"""telephony/term_repair.py — repairing the words STT cannot know.

One job: fix the handful of names that Sarvam's speech-to-text cannot get
right, in the lead's transcript, before that text is used to think with.

WHY THIS EXISTS. On a live call a lead asked, twice, what courses Digital
Brolly offers. The recogniser returned "డిజిటల్ బ్రానీ" and then "డిజిటల్
బ్రౌనీ" — Brany, then Brownie — and the agent answered "నేను ఇతర విషయాలపై
సహాయం చేయలేను" ("I can't help with other topics") to a question squarely about
its own business. The company name appears in nearly every question a lead
asks, so getting it wrong is not a cosmetic transcript blemish: it derails the
answer AND poisons the RAG query built from the same words, which is what
turned an interested customer into a refusal.

WHY NOT FIX IT AT THE RECOGNISER. Sarvam's STT API offers no custom
vocabulary, phrase hints or keyword boosting — checked against the API
reference on 2026-08-13. There is no way to tell it the brand exists. So the
repair belongs here, between hearing and understanding.

WHY FUZZY MATCHING RATHER THAN A LIST OF KNOWN MIS-HEARINGS. Two calls
produced two different manglings of the same word. A lookup table would fix
exactly the spellings already seen and be silently useless against the third.

WHY THE ANCHOR RULE IS THE INTERESTING PART. Measured over every lead
utterance ever recorded, four distinct words sit within edit distance 2 of
"బ్రోలీ", and only two are the brand:

    బ్రానీలో   d=2   "డిజిటల్ బ్రానీలో ..."       <- the brand
    బ్రౌనీలో   d=2   "డిజిటల్ బ్రౌనీలో ..."       <- the brand
    బ్రీఫ్గా   d=2   "కొంచెం బ్రీఫ్గా ..."         <- "briefly"
    బ్రో       d=2   "... వాయి బ్రో."             <- "bro"

Distance cannot separate them; they are all the same distance away. What
separates them is context: the brand is said as DIGITAL Brolly, and "డిజిటల్"
comes back spelled identically on every call. So the weaker the phonetic
match, the more context is demanded of it — a near-exact match stands alone, a
distance-2 match must be vouched for by the word in front of it.

This module rewrites what a human being said, so it errs toward leaving speech
alone: a missed repair costs one clumsy answer, an over-eager one puts words
in the lead's mouth and is far harder to notice.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Words, in either script. Telugu has no inter-word spelling of its own here —
# the class covers the Telugu block, and Latin runs are matched separately so
# a Tinglish sentence gets both.
_WORD = re.compile(r"[ఀ-౿]+|[A-Za-z]+")

# A match this close is acted on alone; anything weaker needs the anchor.
_STANDALONE_MAX = 1
# Beyond this, no amount of context makes it the brand — "పోలీస్" (police) and
# "ప్రోసెస్" (process) both sit at 3.
_ANCHORED_MAX = 2
# Guards against a fragment matching a much longer canonical form: "బ్రో" is
# distance 2 from "బ్రోలీ" but is the word "bro".
_MIN_STEM_RATIO = 0.75


@dataclass(frozen=True)
class Term:
    """A name worth repairing.

    *match* is the canonical spelling the recogniser is compared against and
    *replacement* is what gets written; they differ only where the written
    form carries capitalisation the match should ignore. *anchor* is the word
    that must immediately precede a weak match for it to be trusted — None
    means the term is distinctive enough not to need one.

    *aliases* are exact spellings to accept in addition to anything the fuzzy
    match reaches, and they exist because edit distance is not comparable
    across scripts. "బ్రౌనీ" is 2 edits from "బ్రోలీ", but the same confusion
    written in Latin — "brownie" against "brolly" — is 4, because the Telugu
    packs the same sounds into fewer codepoints. Loosening the threshold to
    4 to catch it would also swallow half the dictionary, so the Latin side
    names its variants instead. They are still anchor-checked: "brownie" on
    its own is a cake.
    """

    match: str
    replacement: str
    anchor: str | None = None
    aliases: tuple[str, ...] = ()


# Deliberately short. Every entry is a licence to rewrite a lead's words, so a
# name earns its place by being (a) frequent enough that mis-hearing it breaks
# calls and (b) distinctive enough not to collide with ordinary speech. The
# brand qualifies on both counts: 61 mentions across the course documents, and
# no Telugu word sounds like it.
TERMS: tuple[Term, ...] = (
    Term(match="బ్రోలీ", replacement="బ్రోలీ", anchor="డిజిటల్"),
    # No Latin mis-hearing has been recorded yet — every observed one came
    # back in Telugu script. These are the same confusions spelled the way a
    # Tinglish call would produce them, listed in advance rather than after
    # the next broken call.
    Term(match="brolly", replacement="Brolly", anchor="digital",
         aliases=("brownie", "brony", "brawly", "broly", "brolley")),
)


def _prefix_match(word: str, canon: str) -> tuple[int, int]:
    """Best edit distance between *canon* and any PREFIX of *word*.

    Returns (distance, prefix_length). Leaving the suffix free is what makes
    this work on Telugu at all: the language agglutinates, so the brand
    arrives with its case ending welded on — బ్రౌనీ+లో, "in Brolly" — and
    comparing whole words would put every real mis-hearing out of reach. The
    first scan for variants did exactly that and found none of them.
    """
    prev = list(range(len(canon) + 1))
    best_distance, best_len = prev[-1], 0
    for i, ca in enumerate(word, 1):
        cur = [prev[0] + 1]
        for j, cb in enumerate(canon, 1):
            cur.append(min(prev[j] + 1,              # delete from word
                           cur[j - 1] + 1,           # insert into word
                           prev[j - 1] + (ca != cb)))  # substitute
        prev = cur
        if prev[-1] < best_distance:
            best_distance, best_len = prev[-1], i
    return best_distance, best_len


def _anchored(term: Term, previous: str | None) -> bool:
    """Whether the word before this one vouches for a weak match.

    A term with no anchor is one distinctive enough to stand on its own; one
    with an anchor and nothing in front of it is not trusted.
    """
    if term.anchor is None:
        return True
    if previous is None:
        return False
    prior = previous.lower() if term.anchor.isascii() else previous
    return prior == term.anchor


def _repair_word(word: str, previous: str | None) -> str | None:
    """The repaired *word*, or None to leave it exactly as heard."""
    for term in TERMS:
        latin = term.match.isascii()
        candidate = word.lower() if latin else word
        canon = term.match.lower() if latin else term.match

        if candidate in term.aliases:
            # A named variant still has to be vouched for by the anchor.
            if _anchored(term, previous):
                return term.replacement
            continue

        distance, stem_len = _prefix_match(candidate, canon)
        if distance == 0:
            # Already correct — and this is the common case, so it also keeps
            # repair() idempotent over a transcript that gets re-read.
            return None
        if distance > _ANCHORED_MAX or stem_len < len(canon) * _MIN_STEM_RATIO:
            continue
        if distance > _STANDALONE_MAX and not _anchored(term, previous):
            continue
        # Only the stem is replaced; whatever the language welded onto the end
        # of it stays attached.
        return term.replacement + word[stem_len:]
    return None


def repair(text: str | None) -> str:
    """Return *text* with known names restored to their real spelling.

    Runs on the call path for every lead utterance, so it never raises: a
    non-string degrades to "" rather than ending a live conversation.
    """
    if not isinstance(text, str) or not text.strip():
        return text if isinstance(text, str) else ""

    out: list[str] = []
    last_end = 0
    previous: str | None = None
    for m in _WORD.finditer(text):
        word = m.group(0)
        repaired = _repair_word(word, previous)
        if repaired is not None and repaired != word:
            logger.info(f'[term-repair] heard "{word}", read it as "{repaired}"')
            out.append(text[last_end:m.start()])
            out.append(repaired)
            last_end = m.end()
        previous = word
    if last_end == 0:
        return text
    out.append(text[last_end:])
    return "".join(out)
