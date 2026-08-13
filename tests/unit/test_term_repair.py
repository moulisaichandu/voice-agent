"""Unit tests for telephony/term_repair.py — un-mangling the brand name.

EVERY example here is real. The sentences below were said by real leads on
real calls and are quoted verbatim out of the `calls` table; the mis-hearings
are what Sarvam's STT actually returned, not invented variants.

The problem this module solves, from one live call:

    lead : డిజిటల్ బ్రానీలో ఏమేమి పోస్టులు ఉన్నాయి?      ("Brany")
    agent: నేను ఇతర విషయాలపై సహాయం చేయలేను               (I can't help with that)
    lead : డిజిటల్ బ్రౌనీలో ఉన్న కోర్సెస్ లిస్ట్ చెప్తారా?  ("Brownie")

A customer asked, twice, what courses Digital Brolly offers, and was refused —
because the agent was handed a company name that does not exist. The same
garbled name also goes into the RAG query, so the documents cannot rescue it.

WHY THIS CANNOT BE FIXED AT THE RECOGNISER. Sarvam's speech-to-text API has no
custom vocabulary, phrase hints or keyword boosting (checked against the API
reference on 2026-08-13) — there is no way to tell it the brand exists. So the
repair has to happen on our side, between hearing and understanding.

THE HARD PART IS PRECISION. Measured over every lead utterance ever recorded,
four distinct words sit within edit distance 2 of "బ్రోలీ" — and only two of
them are the brand:

    బ్రానీలో   d=2   డిజిటల్ బ్రానీలో ...        <- the brand
    బ్రౌనీలో   d=2   డిజిటల్ బ్రౌనీలో ...        <- the brand
    బ్రీఫ్గా   d=2   కొంచెం బ్రీఫ్గా ...          <- "briefly"
    బ్రో       d=2   ... వాయి బ్రో.              <- "bro"

Distance alone cannot separate them, because all four are the same distance
away. What separates them is that the brand is said as DIGITAL Brolly, and
"డిజిటల్" comes back spelled identically every time. Hence the anchor rule:
a weak phonetic match must be vouched for by the word in front of it.
"""

from app.telephony import term_repair

# Verbatim from the `calls` table.
_REAL_BRANY = "డిజిటల్ బ్రానీలో ఏమేమి పోస్టులు ఉన్నాయి?"
_REAL_BROWNIE = "డిజిటల్ బ్రౌనీలో ఉన్న కోర్సెస్ లిస్ట్ చెప్తారా?"
_REAL_BRIEFLY = "అవును, ఒక కోర్స్ గురించి కొంచెం బ్రీఫ్గా ఎక్స్టెండ్ చేస్తారా?"
_REAL_BRO = "ఒక ఏకా అచ్చెను వాయి బ్రో."

_BROLLY = "బ్రోలీ"


# ── the mis-hearings that actually happened ──────────────────────────────────

def test_brany_becomes_brolly():
    assert _BROLLY in term_repair.repair(_REAL_BRANY)


def test_brownie_becomes_brolly():
    assert _BROLLY in term_repair.repair(_REAL_BROWNIE)


def test_the_telugu_case_ending_survives_the_repair():
    """Telugu agglutinates: the brand carries its case ending, so the lead says
    బ్రౌనీ+లో ("in Brolly"), not the bare noun. Replacing the whole word would
    produce 'డిజిటల్ బ్రోలీ ఉన్న...' — grammatically broken Telugu handed to
    the model. Only the stem is replaced; the ending stays attached."""
    assert "బ్రోలీలో" in term_repair.repair(_REAL_BROWNIE)


def test_the_rest_of_the_sentence_is_untouched():
    """The repair must be surgical. Everything the lead said other than the
    mangled name has to reach the model exactly as heard."""
    repaired = term_repair.repair(_REAL_BROWNIE)
    assert "ఉన్న కోర్సెస్ లిస్ట్ చెప్తారా?" in repaired
    assert repaired.startswith("డిజిటల్ ")


# ── precision: the words that must survive untouched ─────────────────────────

def test_briefly_is_not_the_brand():
    """"బ్రీఫ్గా" is 'briefly', and sits at exactly the same edit distance as
    the two real mis-hearings. Rewriting it would put a company name in the
    middle of a sentence about explaining a course."""
    assert term_repair.repair(_REAL_BRIEFLY) == _REAL_BRIEFLY


def test_bro_is_not_the_brand():
    assert term_repair.repair(_REAL_BRO) == _REAL_BRO


def test_words_that_merely_rhyme_are_left_alone():
    """Real Telugu words from the same corpus one step further out. Nothing
    here is a brand; a repair that reached this far would be corrupting
    ordinary speech."""
    for word in ("పోలీస్", "ప్రోసెస్", "ప్లీజ్", "ఫ్రోట్స్", "బోల్చు", "డ్రోస్"):
        sentence = f"నాకు {word} కావాలి"
        assert term_repair.repair(sentence) == sentence, word


# ── the anchor rule ──────────────────────────────────────────────────────────

def test_a_weak_match_needs_the_anchor_in_front_of_it():
    """Same mis-hearing, no 'డిజిటల్' before it. At distance 2 the phonetic
    evidence is too thin to act on alone — this is the rule that keeps
    'briefly' and 'bro' intact."""
    assert term_repair.repair("బ్రౌనీలో ఏమి ఉంది?") == "బ్రౌనీలో ఏమి ఉంది?"


def test_a_near_exact_match_needs_no_anchor():
    """Distance 1 is strong enough on its own. A lead who says just the brand,
    slightly mis-heard, must still be understood."""
    assert term_repair.repair("బ్రోలి గురించి చెప్పండి").startswith(_BROLLY)


def test_the_anchor_itself_is_left_alone():
    assert term_repair.repair(_REAL_BRANY).startswith("డిజిటల్ ")


# ── idempotence and safety ───────────────────────────────────────────────────

def test_correctly_heard_text_is_returned_unchanged():
    """The common case by far. Repairing a name that is already right must not
    perturb it — and repeated repair must converge, since a repaired turn can
    be re-read from the stored transcript."""
    good = "డిజిటల్ బ్రోలీలో కోర్సు ఎంత?"
    assert term_repair.repair(good) == good
    assert term_repair.repair(term_repair.repair(good)) == good


def test_empty_and_blank_text_is_safe():
    assert term_repair.repair("") == ""
    assert term_repair.repair("   ") == "   "


def test_text_with_no_telugu_at_all_is_unchanged():
    assert term_repair.repair("hello, what is the fee?") == "hello, what is the fee?"


def test_a_non_string_does_not_raise():
    """This runs on every lead utterance on the call path; it must degrade
    rather than kill a live conversation."""
    assert term_repair.repair(None) == ""


# ── the Latin spelling, for Tinglish ─────────────────────────────────────────

def test_the_latin_mis_hearing_is_repaired_too():
    """Tinglish leads produce Latin script, where the same confusion appears
    as an English word."""
    assert "Brolly" in term_repair.repair("what courses are in digital brownie?")


def test_a_latin_word_without_the_anchor_is_left_alone():
    """"I ate a brownie" must survive contact with this module."""
    assert term_repair.repair("i ate a brownie") == "i ate a brownie"


def test_the_repair_is_reported(caplog):
    """Every rewrite of what a human said is logged. A silent rewrite of a
    lead's words is not something anyone should have to discover by reading
    transcripts and wondering."""
    with caplog.at_level("INFO"):
        term_repair.repair(_REAL_BROWNIE)
    assert "బ్రౌనీలో" in caplog.text and "బ్రోలీలో" in caplog.text
