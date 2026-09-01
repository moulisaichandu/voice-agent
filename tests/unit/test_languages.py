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


# ── for_call: the single shared resolution used by both dial-path call sites ─
#
# app/telephony/call_routes.py's _call_language() and app/telephony/worker.py
# both need "the ISO code and prompt variables for one call", computed from
# the same two inputs (a lead's language_pref and a campaign's language).
# Before this, each call site resolved it inline, and nothing pinned them
# together — a future "simplification" of one that passed the raw token
# (e.g. campaign.language directly) instead of running it through resolve()
# and iso_code() would silently send preflight the string "tinglish", which
# is not in any agent's ISO language set, and preflight would refuse EVERY
# dial for every code-mixed campaign. Hoisting the resolution into one
# function both call sites use is what makes that impossible instead of
# merely untested.

def test_for_call_returns_the_campaign_default_when_the_lead_has_none():
    lang, dyn = languages.for_call(None, "te")
    assert lang == "te"
    assert dyn["language"] == "Telugu"
    assert "Telugu" in dyn["language_style"]


def test_for_call_lets_the_lead_override_the_campaign():
    lang, dyn = languages.for_call("hi", "te")
    assert lang == "hi"
    assert dyn["language"] == "Hindi"


def test_for_call_maps_tinglish_to_the_telugu_iso_code():
    """The exact regression this function exists to prevent: 'tinglish' is a
    catalogue token, not an ISO code, and preflight only understands ISO
    codes — it must never receive the raw token."""
    lang, dyn = languages.for_call(None, "tinglish")
    assert lang == "te"
    assert "English" in dyn["language_style"]


def test_for_call_sends_nothing_for_auto():
    lang, dyn = languages.for_call(None, None)
    assert lang is None
    assert dyn == {}


# ── what ElevenLabs Agents can actually speak ────────────────────────────────

def test_telugu_is_not_an_elevenlabs_agent_language():
    """The whole reason Telugu needs a different voice backend. Verified
    against the live API, which rejects 'te' outright — this is not a plan
    tier or a model setting, the platform does not offer it."""
    assert not languages.elevenlabs_can_speak("te")


def test_hindi_and_tamil_are_elevenlabs_agent_languages():
    """Guards against over-correcting: the platform DOES offer these, so a
    campaign in Hindi must not be told its language is impossible."""
    assert languages.elevenlabs_can_speak("hi")
    assert languages.elevenlabs_can_speak("ta")
    assert languages.elevenlabs_can_speak("en")


def test_an_absent_language_is_not_claimed_as_speakable():
    assert not languages.elevenlabs_can_speak(None)
    assert not languages.elevenlabs_can_speak("")


def test_every_catalogue_iso_is_either_speakable_or_knowingly_not():
    """Every language we OFFER must resolve to a definite answer, so the
    operator is never told something vague about a language they picked."""
    for token in languages.TOKENS:
        iso = languages.iso_code(token)
        if iso is None:
            continue
        assert isinstance(languages.elevenlabs_can_speak(iso), bool)


# ── which backend carries which language ─────────────────────────────────────

def test_telugu_and_tinglish_route_to_sarvam():
    """ElevenLabs cannot speak Telugu at all — see ELEVENLABS_AGENT_LANGUAGES.
    Sarvam is where they go by default: its voices are Telugu-native, which is
    the one thing the OpenAI Realtime backend could not offer at any price."""
    assert languages.backend_for("te") == languages.SARVAM
    assert languages.backend_for("tinglish") == languages.SARVAM


def test_telugu_falls_back_to_openai_when_the_rollback_is_set(monkeypatch):
    """TELUGU_BACKEND is the rollback switch. Replacing a live voice backend is
    only a safe decision if reverting it is one .env line and a restart, so
    this path has to keep working — openai_bridge.py is still on disk for it."""
    monkeypatch.setattr(languages, "TELUGU_BACKEND", languages.OPENAI_REALTIME)
    assert languages.backend_for("te") == languages.OPENAI_REALTIME
    assert languages.backend_for("tinglish") == languages.OPENAI_REALTIME


def test_the_rollback_never_moves_a_language_off_elevenlabs(monkeypatch):
    """The switch is scoped to Telugu. English and Hindi campaigns are dialling
    in production and must not move backend because of a Telugu rollback."""
    monkeypatch.setattr(languages, "TELUGU_BACKEND", languages.OPENAI_REALTIME)
    for token in ("auto", "en", "hi", "hinglish"):
        assert languages.backend_for(token) == languages.ELEVENLABS


def test_english_hindi_and_auto_stay_on_elevenlabs():
    """The working path. Every campaign in production today is one of these,
    and none of them may move backend as a side effect of this feature."""
    for token in ("auto", "en", "hi", "hinglish"):
        assert languages.backend_for(token) == languages.ELEVENLABS


def test_every_catalogue_token_has_a_backend():
    """A language in the dropdown with no backend would fail at dial time with
    a KeyError rather than a message anyone can act on."""
    for token in languages.TOKENS:
        assert languages.backend_for(token) in (
            languages.ELEVENLABS, languages.OPENAI_REALTIME, languages.SARVAM
        )


def test_an_unknown_token_falls_back_to_the_working_backend():
    """normalize() sends junk to 'auto', so this can only happen if a caller
    hand-builds a token. Degrade to the backend that works, not to a crash."""
    assert languages.backend_for("klingon") == languages.ELEVENLABS


def test_no_language_routes_to_elevenlabs_for_something_it_cannot_speak():
    """The consistency guard between the two tables: if a language is routed
    to ElevenLabs, ElevenLabs must actually offer it. Adding Tamil later and
    forgetting to route it would otherwise dial into a wall."""
    for token in languages.TOKENS:
        iso = languages.iso_code(token)
        if iso and languages.backend_for(token) == languages.ELEVENLABS:
            assert languages.elevenlabs_can_speak(iso), (
                f"{token} routes to ElevenLabs but ElevenLabs cannot speak {iso}"
            )


# ── backend_for_iso: the same table, for callers that only have an ISO code ──
#
# app/telephony/preflight.py receives an ISO code (it round-trips through
# app/languages.py's for_call()/iso_code(), never the raw catalogue token), so
# it cannot call backend_for() directly. This must never disagree with it —
# two lookups into the same fact, computed two different ways, is exactly the
# kind of pair that drifts apart silently if only one of them is tested.

def test_backend_for_iso_agrees_with_backend_for_token():
    """The two lookups must never disagree — preflight uses the ISO one and
    the dial path uses the token one, on the same call."""
    for token in languages.TOKENS:
        iso = languages.iso_code(token)
        if iso:
            assert languages.backend_for_iso(iso) == languages.backend_for(token)


def test_backend_for_iso_routes_telugu_to_sarvam():
    assert languages.backend_for_iso("te") == languages.SARVAM


def test_backend_for_iso_follows_the_rollback_switch(monkeypatch):
    """preflight() only ever holds the ISO code, so if this lookup ignored the
    switch it would check Sarvam's prerequisites for a call OpenAI was about to
    place — refusing every rolled-back Telugu campaign for a missing key that
    the backend actually running does not need."""
    monkeypatch.setattr(languages, "TELUGU_BACKEND", languages.OPENAI_REALTIME)
    assert languages.backend_for_iso("te") == languages.OPENAI_REALTIME


def test_backend_for_iso_routes_hindi_and_none_to_elevenlabs():
    assert languages.backend_for_iso("hi") == languages.ELEVENLABS


@pytest.mark.parametrize("telugu_backend", ["sarvam", "openai_realtime"])
def test_tokens_sharing_an_iso_code_never_disagree_about_the_backend(
    monkeypatch, telugu_backend
):
    """THE invariant backend_for_iso() rests on, pinned under both settings of
    the rollback switch.

    backend_for_iso() resolves an ISO code by finding a token that maps to it,
    and 'te' and 'tinglish' both map to 'te'. If those two tokens could ever
    name different backends, which one won would depend on catalogue order —
    so preflight would silently validate one backend's prerequisites for a call
    the other was about to place. Scoping TELUGU_BACKEND to both tokens at once
    is what keeps that impossible; this test is what stops someone splitting
    them later without noticing."""
    monkeypatch.setattr(languages, "TELUGU_BACKEND", telugu_backend)
    by_iso: dict[str, set[str]] = {}
    for token in languages.TOKENS:
        iso = languages.iso_code(token)
        if iso:
            by_iso.setdefault(iso, set()).add(languages.backend_for(token))
    for iso, backends in by_iso.items():
        assert len(backends) == 1, (
            f"tokens for ISO {iso!r} disagree about the backend: {backends}"
        )


# ── two-way capability and human-readable names, per backend ─────────────────
#
# app/telephony/preflight.py and app/admin/campaigns.py both refuse a two-way
# campaign on a backend that has not been proven on a live call. They used to
# each hardcode OPENAI_TWOWAY_ENABLED and the words "OpenAI Realtime", which
# silently became wrong for any third backend. Both now ask here.

def test_elevenlabs_two_way_needs_no_flag():
    """Two-way English/Hindi is what production runs today. A flag it could be
    turned off by would be a way to break the working path by accident."""
    assert languages.twoway_enabled(languages.ELEVENLABS) is True


def test_sarvam_two_way_is_refused_until_a_live_call_proves_it(monkeypatch):
    """A two-way call on an unproven backend connects, never speaks, never
    listens, bills the full CALL_MAX_DURATION_S of silence, and is then
    recorded as a SUCCESSFUL zero-turn call because max_duration counts as a
    clean exit. That is why the flag defaults off."""
    monkeypatch.setattr(languages, "SARVAM_TWOWAY_ENABLED", False)
    assert languages.twoway_enabled(languages.SARVAM) is False
    monkeypatch.setattr(languages, "SARVAM_TWOWAY_ENABLED", True)
    assert languages.twoway_enabled(languages.SARVAM) is True


def test_each_backend_reads_its_own_two_way_flag(monkeypatch):
    """Enabling two-way on one backend must not enable it on the other. They
    are separate implementations proven by separate live calls."""
    monkeypatch.setattr(languages, "SARVAM_TWOWAY_ENABLED", True)
    monkeypatch.setattr(languages, "OPENAI_TWOWAY_ENABLED", False)
    assert languages.twoway_enabled(languages.SARVAM) is True
    assert languages.twoway_enabled(languages.OPENAI_REALTIME) is False


def test_an_unknown_backend_is_not_two_way_capable():
    """Degrade to refusing, not to dialling. An unrecognised backend here means
    a bug, and a lead should not take the consequences of it."""
    assert languages.twoway_enabled("klingon_voice") is False


def test_every_backend_has_a_name_an_operator_can_act_on():
    """These strings land in the refusal message an operator reads in the
    dashboard. 'sarvam' is a code constant; 'Sarvam' is an answer."""
    for backend in (languages.ELEVENLABS, languages.OPENAI_REALTIME,
                    languages.SARVAM):
        assert languages.backend_display(backend) == {
            languages.ELEVENLABS: "ElevenLabs",
            languages.OPENAI_REALTIME: "OpenAI Realtime",
            languages.SARVAM: "Sarvam",
        }[backend]
    assert languages.backend_for_iso(None) == languages.ELEVENLABS


def test_code_mixed_registers_are_recognised_from_their_own_style_text():
    """The style text is the only register signal that crosses the
    dynamic_variables boundary into a backend, so the backend has to be
    able to read the register back off it. Matched exactly against the
    catalogue: a substring test would misfire in both directions ("English"
    appears in the pure-Telugu style too), and a placeholder string that is
    not a catalogue style is pure by default — a caller that bypasses
    resolution must never be handed permission to mix in English."""
    assert languages.is_code_mixed(languages.style("tinglish"))
    assert languages.is_code_mixed(languages.style("hinglish"))
    assert not languages.is_code_mixed(languages.style("te"))
    assert not languages.is_code_mixed(languages.style("hi"))
    assert not languages.is_code_mixed(languages.style("en"))
    assert not languages.is_code_mixed(None)
    assert not languages.is_code_mixed("Mix in English words.")
