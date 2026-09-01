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

# What separates two sentences, for the anchor rule below. Sentence-ENDING
# marks only: a comma between an anchor and its term is still one thought.
_SENTENCE_BREAK = re.compile(r"[.!?।॥\n]")

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
    that must immediately precede a weak match for it to be trusted, and
    *anchor_after* the word that must immediately follow — either vouches on
    its own. Both None means the term is distinctive enough to need neither.
    anchor_after exists because the recogniser garbles the FIRST word of a
    collocation as readily as the second: "క్రిస్టల్ మార్కెటింగ్" (heard
    live 2026-08-27) has nothing in front of the mis-hearing to anchor on —
    the vouching word is behind it.

    *aliases* are exact spellings to accept in addition to anything the fuzzy
    match reaches, and they exist because edit distance is not comparable
    across scripts. "బ్రౌనీ" is 2 edits from "బ్రోలీ", but the same confusion
    written in Latin — "brownie" against "brolly" — is 4, because the Telugu
    packs the same sounds into fewer codepoints. Loosening the threshold to
    4 to catch it would also swallow half the dictionary, so the Latin side
    names its variants instead. They are still anchor-checked: "brownie" on
    its own is a cake.

    *fuzzy* False restricts a term to its exact aliases. The distance
    thresholds below were calibrated for బ్రోలీ, which no real Telugu word
    sits near; a canon with a populated neighbourhood needs the fuzzy path
    switched off instead of the thresholds re-tuned, because one number
    cannot serve both. See the డిజిటల్ term for what that neighbourhood
    costs when it is left on.
    """

    match: str
    replacement: str
    anchor: str | None = None
    anchor_after: str | None = None
    aliases: tuple[str, ...] = ()
    fuzzy: bool = True


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
    # The OTHER half of the brand collocation. Heard live 2026-08-27 as
    # "క్రిస్టల్ మార్కెటింగ్ కోర్సెస్..." — the mis-hearing is the FIRST
    # word, so only the word behind it can vouch.
    #
    # fuzzy=False, and that is the whole design of this entry. బ్రోలీ has no
    # real Telugu word within reach; డిజిటల్ heads a loanword family, and an
    # adversarial review measured what the fuzzy path did with it: డిజిట్
    # (digit) rewritten to డిజిటల్ with no anchor at all, డిజిటలైజేషన్
    # spliced into డిజిటల్ైజేషన్ (a virama followed by a dependent vowel
    # sign — unwritable Telugu, read aloud by the synthesiser), and — worst
    # — ఫిజికల్ and డీజిల్ before మార్కెటింగ్ rewritten too, so "is digital
    # marketing better than PHYSICAL marketing?", a question this business's
    # leads really ask, became a comparison of digital marketing with
    # itself. The observed mis-hearing is 5 edits away and only ever
    # reachable through the exact alias, so the fuzzy path bought this term
    # nothing and cost it all of that.
    Term(match="డిజిటల్", replacement="డిజిటల్", anchor_after="మార్కెటింగ్",
         aliases=("క్రిస్టల్",), fuzzy=False),
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


def _anchored(term: Term, previous: str | None,
              following: str | None) -> bool:
    """Whether a neighbouring word vouches for a weak match.

    A term with neither anchor is one distinctive enough to stand on its
    own; a term with anchors and no matching neighbour is not trusted.
    Either side vouches independently — the recogniser garbles whichever
    word it likes, and the intact one is the evidence.
    """
    if term.anchor is None and term.anchor_after is None:
        return True
    if term.anchor is not None and previous is not None:
        prior = previous.lower() if term.anchor.isascii() else previous
        if prior == term.anchor:
            return True
    if term.anchor_after is not None and following is not None:
        nxt = (following.lower() if term.anchor_after.isascii()
               else following)
        if nxt == term.anchor_after:
            return True
    return False


def _repair_word(word: str, previous: str | None,
                 following: str | None) -> str | None:
    """The repaired *word*, or None to leave it exactly as heard."""
    for term in TERMS:
        latin = term.match.isascii()
        candidate = word.lower() if latin else word
        canon = term.match.lower() if latin else term.match

        if candidate in term.aliases:
            # A named variant still has to be vouched for by an anchor.
            if _anchored(term, previous, following):
                return term.replacement
            continue

        if not term.fuzzy:
            continue

        distance, stem_len = _prefix_match(candidate, canon)
        if distance == 0:
            # Already correct — and this is the common case, so it also keeps
            # repair() idempotent over a transcript that gets re-read.
            return None
        if distance > _ANCHORED_MAX or stem_len < len(canon) * _MIN_STEM_RATIO:
            continue
        if distance > _STANDALONE_MAX and not _anchored(term, previous, following):
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
    matches = list(_WORD.finditer(text))
    for i, m in enumerate(matches):
        word = m.group(0)
        # A neighbour only vouches from INSIDE the same sentence. The word
        # scan skips punctuation, so a full stop between two words used to
        # be invisible: a lead's sentence-final "క్రిస్టల్" was rewritten
        # because the NEXT sentence happened to open with మార్కెటింగ్.
        previous = (matches[i - 1].group(0)
                    if i and not _SENTENCE_BREAK.search(
                        text[matches[i - 1].end():m.start()])
                    else None)
        following = (matches[i + 1].group(0)
                     if i + 1 < len(matches) and not _SENTENCE_BREAK.search(
                         text[m.end():matches[i + 1].start()])
                     else None)
        repaired = _repair_word(word, previous, following)
        if repaired is not None and repaired != word:
            logger.info(f'[term-repair] heard "{word}", read it as "{repaired}"')
            out.append(text[last_end:m.start()])
            out.append(repaired)
            last_end = m.end()
    if last_end == 0:
        return text
    out.append(text[last_end:])
    return "".join(out)
