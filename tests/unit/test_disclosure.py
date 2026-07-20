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


def test_none_script_fails():
    assert has_ai_disclosure(None) is False


def test_blank_script_fails():
    assert has_ai_disclosure("   ") is False
