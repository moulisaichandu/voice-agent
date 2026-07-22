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
from typing import Literal

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

# Every language the ElevenLabs AGENTS platform will accept, verified against
# the live API on 2026-07-22 by asking it to add Telugu and reading the
# validation error it returned:
#
#   "Preset languages must be one of en, zh, es, hi, pt, fr, de, ja, ar, ko,
#    id, it, nl, tr, pl, ru, sv, tl, ms, ro, uk, el, cs, da, fi, bg, hr, sk,
#    ta, vi, no, hu, pt-br, fil but got te"
#
# Tamil is there. TELUGU IS NOT — and no plan tier or model setting changes
# that, because the platform simply does not offer it as an agent language.
# That is the real reason this project's Telugu calls failed, and it is why
# Telugu needs a different voice backend entirely rather than more ElevenLabs
# configuration.
#
# Used ONLY to word a failure honestly. The authoritative check is always what
# the agent itself reports as configured — so if ElevenLabs adds Telugu later
# and an operator enables it, that succeeds before this list is ever consulted
# and no code change is needed here.
ELEVENLABS_AGENT_LANGUAGES = frozenset({
    "en", "zh", "es", "hi", "pt", "fr", "de", "ja", "ar", "ko", "id", "it",
    "nl", "tr", "pl", "ru", "sv", "tl", "ms", "ro", "uk", "el", "cs", "da",
    "fi", "bg", "hr", "sk", "ta", "vi", "no", "hu", "pt-br", "fil",
})


def elevenlabs_can_speak(iso: str | None) -> bool:
    """Whether the ElevenLabs Agents platform offers *iso* at all.

    False means no amount of dashboard configuration will help — the operator
    should be told that, not sent to look for a setting that doesn't exist.
    """
    return bool(iso) and iso in ELEVENLABS_AGENT_LANGUAGES


# ── which backend carries which language ─────────────────────────────────────
# Two voice backends exist because no single one covers this business's
# languages. ElevenLabs Agents sounds better and is already in production, but
# does not offer Telugu at any price (see ELEVENLABS_AGENT_LANGUAGES). OpenAI
# Realtime does, and the sibling ../ai-voice-agent proves it on real 8 kHz
# phone calls.
#
# The backend is DERIVED from the language, never chosen per call and never
# inferred by a model — campaigns.language is admin-set at creation, so the
# backend is fully determined before a call is placed. That is the same
# discipline campaigns.mode follows, for the same reason.
ELEVENLABS = "elevenlabs"
OPENAI_REALTIME = "openai_realtime"

VoiceBackend = Literal["elevenlabs", "openai_realtime"]

_BACKEND_BY_TOKEN: dict[str, str] = {
    AUTO: ELEVENLABS,
    "en": ELEVENLABS,
    "hi": ELEVENLABS,
    "hinglish": ELEVENLABS,
    "te": OPENAI_REALTIME,
    "tinglish": OPENAI_REALTIME,
}


def backend_for(token: str) -> str:
    """The voice backend that can carry *token*.

    Falls back to ELEVENLABS for anything unrecognised: normalize() already
    routes junk to AUTO, so reaching this means a caller hand-built a token,
    and degrading to the backend that is known to work beats raising mid-dial.
    """
    return _BACKEND_BY_TOKEN.get(normalize(token), ELEVENLABS)


def backend_for_iso(iso: str | None) -> str:
    """The backend for an ISO code, for callers that only have the code.

    app/telephony/preflight.py is one such caller: it receives the ISO code
    that app/languages.py's own for_call()/iso_code() produced, never the raw
    catalogue token, so it cannot call backend_for() directly.

    Several tokens can share an ISO code (te and tinglish are both 'te'), but
    they never disagree about the backend — a language is carried by exactly
    one backend — so resolving through the first token that matches is safe.
    Falls back to ELEVENLABS for an unrecognised or absent code, for the same
    reason backend_for() does: degrading to the backend that is known to work
    beats raising mid-dial.
    """
    for token, backend in _BACKEND_BY_TOKEN.items():
        if iso_code(token) == iso:
            return backend
    return ELEVENLABS


# ISO codes ElevenLabs' Flash/Turbo v2.5 models cannot pronounce, kept as a
# backstop on the model rather than the language. Largely superseded by
# ELEVENLABS_AGENT_LANGUAGES above: 'te' is refused as an agent language before
# any model question arises, so this gate is unreachable for Telugu today. It
# stays because the model is derived from the language by ElevenLabs, not set
# by us, and a future language they add could reintroduce the mismatch.
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
