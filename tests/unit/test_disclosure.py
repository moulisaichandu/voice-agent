from app.compliance.disclosure import has_ai_disclosure


def test_english_ai_disclosure_in_first_sentence_passes():
    assert has_ai_disclosure(
        "This is an AI voice assistant calling on behalf of Digital Brolly. "
        "We wanted to let you know about our new course."
    ) is True


def test_telugu_disclosure_marker_passes():
    assert has_ai_disclosure("ఇది ఒక AI వాయిస్ కాల్. డిజిటల్ బ్రోలీ నుండి కాల్ చేస్తున్నాము.") is True


def test_artificial_intelligence_phrase_passes():
    assert has_ai_disclosure(
        "This call uses artificial intelligence. Here is some information."
    ) is True


def test_missing_disclosure_fails():
    assert has_ai_disclosure("Namaste! We're calling about the course.") is False


def test_disclosure_only_in_second_sentence_fails():
    """The rule is FIRST line/sentence only — a disclosure buried later
    doesn't count, matching CLAUDE.md's "as the FIRST line" wording."""
    assert has_ai_disclosure(
        "Namaste, this is Digital Brolly. By the way, this is an AI call."
    ) is False


def test_ai_as_a_word_inside_another_word_does_not_false_positive():
    """'again', 'main', 'certain' etc. all contain the substring 'ai' — the
    check must require a whole-word match, not a bare substring."""
    assert has_ai_disclosure("We are calling again about your course.") is False


def test_company_trading_name_is_not_a_disclosure():
    """REGRESSION. This business trades as "AI Skills" (aiskills.in), so a bare
    \\bai\\b match accepted a script that discloses nothing at all as compliant —
    the worst possible failure for the one rule CLAUDE.md says must be enforced
    by a test. The brand name is stripped before the marker search."""
    assert has_ai_disclosure("Hello from AI Skills, we have a new course.") is False
    assert has_ai_disclosure(
        "Namaste, this is Ravi from AI Skills Digital Brolly about your enquiry."
    ) is False


def test_bare_greeting_before_the_disclosure_still_passes():
    """REGRESSION (false negative). Splitting on the first [.!?] made the
    "first sentence" of this script just "Namaste", so a fully compliant
    script was rejected and the operator had no way to tell why."""
    assert has_ai_disclosure(
        "Namaste! This is an AI voice assistant calling from Digital Brolly."
    ) is True


def test_a_substantive_first_sentence_must_carry_the_disclosure_itself():
    """The greeting roll-forward must not become a general second-sentence
    allowance — sounding human up front and admitting the machine afterwards
    is exactly the non-compliant pattern. Guards the fix above."""
    assert has_ai_disclosure(
        "Namaste, this is Digital Brolly. By the way, this is an AI call."
    ) is False


def test_none_script_fails():
    assert has_ai_disclosure(None) is False


def test_blank_script_fails():
    assert has_ai_disclosure("   ") is False


def test_a_hindi_disclosure_in_devanagari_is_accepted():
    """Offering Hindi means accepting a Hindi disclosure. Without a Devanagari
    marker, a correctly-disclosing Hindi script is refused at campaign
    creation and the operator has no way to comply."""
    assert has_ai_disclosure(
        "नमस्ते! यह एक स्वचालित एआई वॉइस असिस्टेंट है, डिजिटल ब्रॉली की ओर से।"
    )


def test_a_hindi_script_without_a_disclosure_still_fails():
    """The marker list must not become a rubber stamp: ordinary Hindi with no
    disclosure has to keep failing."""
    assert not has_ai_disclosure(
        "नमस्ते! हम डिजिटल ब्रॉली से बात कर रहे हैं। हमारा नया कोर्स शुरू हो रहा है।"
    )


def test_hindi_artificial_intelligence_spelled_out_is_accepted():
    assert has_ai_disclosure("यह कॉल कृत्रिम बुद्धिमत्ता द्वारा की जा रही है।")
