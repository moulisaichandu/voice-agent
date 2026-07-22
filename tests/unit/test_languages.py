"""Unit tests for app/languages.py.

The catalogue is pure, but it decides what language a real person is spoken to
in, and every failure mode here is silent: a mis-normalized spreadsheet cell
doesn't raise, it just calls someone in the wrong language.
"""

from pathlib import Path
from typing import get_args

import pytest

from app import languages
from app.db.models import CampaignLanguage


def test_tokens_match_the_campaign_language_literal():
    """The six tokens are duplicated on purpose — SQL and TypeScript can't
    import from Python, so app/languages.py's LANGUAGES dict,
    app/db/models.py's CampaignLanguage Literal, migrations/0003's two CHECK
    constraints, and frontend/lib/api.ts's CampaignLanguage union all repeat
    the same six strings by hand. Nothing enforces that they agree except this
    test: add a language to the catalogue and forget the Literal, and a valid
    campaign gets a silent 422 at the API boundary instead of dialling."""
    assert set(languages.TOKENS) == set(get_args(CampaignLanguage))


def test_tokens_all_appear_in_the_migration_check_constraints():
    """migrations/0003_campaign_language.sql hand-writes the same six tokens
    into two CHECK constraints (campaigns.language and leads.language_pref).
    A token present in app/languages.py but missing from the migration would
    pass every Python-side check and then blow up as a Postgres constraint
    violation (a 500) the first time a campaign or lead actually used it —
    a failure mode invisible until it reaches a real database."""
    migration_path = (
        Path(__file__).resolve().parents[2] / "migrations" / "0003_campaign_language.sql"
    )
    sql = migration_path.read_text(encoding="utf-8")
    for token in languages.TOKENS:
        assert token in sql, f"{token!r} is missing from {migration_path.name}"


def test_auto_sends_no_override():
    """The whole safety property of this feature. Every pre-existing campaign
    is 'auto', and 'auto' must produce no conversation_config_override at all
    — ElevenLabs raises if an override arrives for a field that isn't enabled
    in the agent's Security tab, so a wrong answer here breaks every call."""
    assert languages.iso_code("auto") is None
    assert languages.style("auto") is None


@pytest.mark.parametrize("raw", ["Telugu", "telugu", "  TELUGU  ", "te", "TE", "telegu", "tel"])
def test_telugu_spellings_normalize(raw):
    """Real spreadsheets contain all of these, including the common
    'telegu' misspelling."""
    assert languages.normalize(raw) == "te"


@pytest.mark.parametrize("raw", ["Hindi", "hi", "HIN", " hindi "])
def test_hindi_spellings_normalize(raw):
    assert languages.normalize(raw) == "hi"


@pytest.mark.parametrize("raw", ["Tinglish", "tenglish", "TINGLISH"])
def test_tinglish_spellings_normalize(raw):
    assert languages.normalize(raw) == "tinglish"


@pytest.mark.parametrize("raw", [None, "", "   ", "Klingon", "n/a", "-"])
def test_unknown_and_blank_values_fall_back_to_auto(raw):
    """A typo in ONE cell of a 5,000-row file must not fail the import or
    silently pick a language for that lead. 'auto' means 'whatever the agent
    is already configured with', which is the safe answer."""
    assert languages.normalize(raw) == "auto"


def test_code_mixed_registers_map_to_their_base_language():
    """Tinglish is Telugu with English words in it, so speech recognition must
    run as Telugu. Mapping it to 'en' would pin ASR to English and mis-hear a
    lead who answers mostly in Telugu — silent, and it lands on the lead."""
    assert languages.iso_code("tinglish") == "te"
    assert languages.iso_code("hinglish") == "hi"


def test_a_lead_language_beats_the_campaign_default():
    assert languages.resolve("hindi", "te") == "hi"


def test_the_campaign_default_applies_when_the_lead_has_none():
    assert languages.resolve("auto", "te") == "te"
    assert languages.resolve(None, "tinglish") == "tinglish"
    assert languages.resolve("", "hi") == "hi"


def test_auto_everywhere_stays_auto():
    assert languages.resolve(None, None) == "auto"
    assert languages.resolve("auto", "auto") == "auto"


def test_an_unknown_lead_value_does_not_override_the_campaign():
    """Junk in a spreadsheet cell must not silently downgrade a Telugu
    campaign to the agent's default language."""
    assert languages.resolve("Klingon", "te") == "te"


def test_every_token_is_complete():
    """Guards against a language being half-added — in the catalogue but with
    no ISO code, so it silently behaves like 'auto'."""
    for token in languages.TOKENS:
        assert languages.display(token), f"{token} has no display name"
        if token == languages.AUTO:
            continue
        assert languages.iso_code(token), f"{token} has no ISO code"
        assert languages.style(token), f"{token} has no style instruction"


def test_telugu_is_flagged_as_needing_a_v3_model():
    """Flash/Turbo v2.5's 32 languages do not include Telugu. This set is what
    preflight uses to refuse a Telugu campaign pointed at a v2.5 agent — the
    exact misconfiguration that produced garbled Telugu audio in testing."""
    assert "te" in languages.V3_ONLY_ISO
    assert "hi" not in languages.V3_ONLY_ISO
