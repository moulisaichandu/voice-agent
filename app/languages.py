"""languages.py — The catalogue of languages this product dials in.

Pure logic, no I/O — the same shape as app/compliance/disclosure.py, and for
the same reason: what language a real person is spoken to in must be decidable
and testable without a database or a network.

Two things are deliberately kept apart here:

  * the ISO code, which drives ElevenLabs' speech RECOGNITION and SYNTHESIS
    and is the only part of this the platform understands;
  * the style, which is prose for the agent's prompt.

That split is what lets "Tinglish" exist at all. Tinglish and Hinglish are
code-mixed registers, not ISO languages, so they map to their BASE language
(te / hi) and express the mixing through the style text. Mapping them to 'en'
instead would pin speech recognition to English and mis-hear a lead who
answers mostly in Telugu — a silent failure that lands on the lead, not on us.

'auto' means "send no override at all", i.e. whatever the ElevenLabs agent is
already configured with. It is the default for every campaign, so a campaign
created before this feature existed dials exactly as it did before. That is
not a nicety: ElevenLabs RAISES if a conversation sends an override for a
field that is not enabled in the agent's Security tab, so 'auto' sending
nothing is what keeps the working one-way and two-way flows working.
"""

from __future__ import annotations

from dataclasses import dataclass

AUTO = "auto"

# Everyday English words an Indian lead uses untranslated in ordinary speech.
# Naming them explicitly beats "mix in some English": an unanchored instruction
# produced either near-pure Telugu or near-pure English depending on the turn.
_KEEP_IN_ENGLISH = "course, fees, batch, timings, online, demo, certificate, placement"


@dataclass(frozen=True)
class Language:
    """One offered language. *iso* is None only for AUTO, which is the signal
    to send no override; see the module docstring."""
    token: str
    display: str
    iso: str | None
    style: str | None


LANGUAGES: dict[str, Language] = {
    AUTO: Language(
        token=AUTO,
        display="Agent default",
        iso=None,
        style=None,
    ),
    "en": Language(
        token="en",
        display="English",
        iso="en",
        style="Speak in clear, simple English throughout.",
    ),
    "te": Language(
        token="te",
        display="Telugu",
        iso="te",
        style="Speak only in Telugu (తెలుగు) throughout. Do not switch to English.",
    ),
    "tinglish": Language(
        token="tinglish",
        display="Tinglish (Telugu + English)",
        iso="te",
        style=(
            "Speak in Telugu, in the natural mixed way people in Hyderabad "
            f"speak. Keep these everyday English words in English rather than "
            f"translating them: {_KEEP_IN_ENGLISH}."
        ),
    ),
    "hi": Language(
        token="hi",
        display="Hindi",
        iso="hi",
        style="Speak only in Hindi (हिन्दी) throughout. Do not switch to English.",
    ),
    "hinglish": Language(
        token="hinglish",
        display="Hinglish (Hindi + English)",
        iso="hi",
        style=(
            "Speak in Hindi, in the natural mixed way people speak in Indian "
            f"cities. Keep these everyday English words in English rather than "
            f"translating them: {_KEEP_IN_ENGLISH}."
        ),
    ),
}

TOKENS: tuple[str, ...] = tuple(LANGUAGES)

# ISO codes absent from ElevenLabs Flash/Turbo v2.5's 32-language list
# (en ja zh de hi fr ko pt it es id nl tr fil pl sv bg ro ar cs el fi hr ms sk
#  da ta uk ru hu no vi) and available only on Eleven v3 / v3 Conversational.
#
# This is not trivia. An agent left on a v2.5-class model cannot pronounce
# Telugu at all: asked to, it produces garbled audio and drifts back to
# English. That was the cause of this project's failed Telugu two-way test,
# and it is invisible from the outside — which is why preflight checks it.
V3_ONLY_ISO = frozenset({"te"})

# Free-text spellings seen in real uploaded files, mapped to canonical tokens.
# Lower-cased and stripped before lookup.
_ALIASES: dict[str, str] = {
    "auto": AUTO, "default": AUTO, "any": AUTO,
    "en": "en", "eng": "en", "english": "en",
    "te": "te", "tel": "te", "telugu": "te", "telegu": "te", "తెలుగు": "te",
    "tinglish": "tinglish", "tenglish": "tinglish", "te-en": "tinglish",
    "telugish": "tinglish",
    "hi": "hi", "hin": "hi", "hindi": "hi", "हिंदी": "hi", "हिन्दी": "hi",
    "hinglish": "hinglish", "hi-en": "hinglish",
}


def normalize(value: str | None) -> str:
    """A free-text language value → a canonical token, defaulting to AUTO.

    Never raises. A spreadsheet's "Language" column is typed by hand, and one
    unrecognised cell in a 5,000-row import must not fail the import or pick a
    language on the lead's behalf — falling back to AUTO means "use whatever
    the agent is configured with", which is the only safe unknown answer.
    """
    if value is None:
        return AUTO
    key = str(value).strip().lower()
    if not key:
        return AUTO
    if key in LANGUAGES:
        return key
    return _ALIASES.get(key, AUTO)


def resolve(lead_pref: str | None, campaign_language: str | None) -> str:
    """The language THIS call runs in.

    The lead's own value wins when it is set to something real; otherwise the
    campaign's. This mirrors how consent already resolves in
    app/admin/leads.py's upload path — the row's own value beats the
    operator-supplied default — so a mixed-language file works without
    splitting it into separate campaigns.
    """
    lead = normalize(lead_pref)
    if lead != AUTO:
        return lead
    return normalize(campaign_language)


def iso_code(token: str) -> str | None:
    """The ElevenLabs `language` value, or None for AUTO meaning "send no
    override". Callers MUST treat None as "omit the key entirely"."""
    return LANGUAGES[normalize(token)].iso


def style(token: str) -> str | None:
    """The prompt instruction for this language, or None for AUTO."""
    return LANGUAGES[normalize(token)].style


def display(token: str) -> str | None:
    """The human-readable name — shown in the dashboard and passed to the
    agent as the {{language}} dynamic variable."""
    return LANGUAGES[normalize(token)].display


def for_call(lead_pref: str | None, campaign_language: str | None) -> tuple[str | None, dict]:
    """The ISO code and prompt variables for one call, resolved from a lead's
    own preference and its campaign's default.

    This is the ONE place both dial-path call sites compute this —
    app/telephony/call_routes.py's _call_language() and app/telephony/
    worker.py both need it, and before this existed each resolved it inline.
    Nothing pinned the two together: a future edit that "simplified" either
    one to pass a raw catalogue token (e.g. campaign_language directly)
    instead of running it through resolve() + iso_code() would silently send
    preflight the string "tinglish" — not an ISO code, not in any agent's
    language set — and preflight would refuse EVERY dial for every code-mixed
    campaign, with no failing test to catch it. One function, one place that
    can go wrong.

    Returns (None, {}) for 'auto' — which is not "the default language" but
    "send nothing at all"; see the module docstring on why that matters.
    """
    token = resolve(lead_pref, campaign_language)
    if token == AUTO:
        return None, {}
    return iso_code(token), {
        "language": display(token),
        "language_style": style(token),
    }
