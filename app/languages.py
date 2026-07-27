"""languages.py — The catalogue of languages this product dials in.

Pure logic, no I/O — the same shape as app/compliance/disclosure.py, and for
the same reason: what language a real person is spoken to in must be decidable
and testable without a database or a network. The three values imported from
app.config are the one concession: which backend carries Telugu, and whether
each backend may run two-way, are operator decisions rather than facts about a
language. They are plain module-level constants read once at import, so every
function here stays synchronous, side-effect-free and testable by
monkeypatching this module's own namespace.

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

from app.config import (
    OPENAI_TWOWAY_ENABLED,
    SARVAM_TWOWAY_ENABLED,
    TELUGU_BACKEND,
)

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
# Three voice backends exist because no single one covers this business's
# languages. ElevenLabs Agents sounds better and is already in production, but
# does not offer Telugu at any price (see ELEVENLABS_AGENT_LANGUAGES).
#
# Telugu therefore needs its own backend, and has had two. OpenAI Realtime came
# first and works, but every Realtime voice is English-first — there is no
# Telugu-native voice to choose, and no code change fixes that. Sarvam's TTS
# voices ARE Telugu-native and its STT is trained on Indian telephony audio,
# which is why Telugu moved there. TELUGU_BACKEND switches between the two.
#
# The backend is DERIVED from the language, never chosen per call and never
# inferred by a model — campaigns.language is admin-set at creation, so the
# backend is fully determined before a call is placed. That is the same
# discipline campaigns.mode follows, for the same reason. TELUGU_BACKEND does
# not weaken that: it is read from .env at import, identically for every call.
ELEVENLABS = "elevenlabs"
OPENAI_REALTIME = "openai_realtime"
SARVAM = "sarvam"

VoiceBackend = Literal["elevenlabs", "openai_realtime", "sarvam"]

_BACKEND_DISPLAY: dict[str, str] = {
    ELEVENLABS: "ElevenLabs",
    OPENAI_REALTIME: "OpenAI Realtime",
    SARVAM: "Sarvam",
}

_BACKEND_BY_TOKEN: dict[str, str] = {
    AUTO: ELEVENLABS,
    "en": ELEVENLABS,
    "hi": ELEVENLABS,
    "hinglish": ELEVENLABS,
}

# Both Telugu tokens are resolved through TELUGU_BACKEND rather than listed in
# the table above, and they are resolved through the SAME switch on purpose.
#
# 'te' and 'tinglish' share the ISO code 'te'. backend_for_iso() resolves an
# ISO code by finding a token that maps to it, so if these two tokens could
# ever name different backends, which one won would depend on catalogue order.
# preflight would then validate one backend's prerequisites for a call the
# other was about to place — an ElevenLabs-style silent mismatch with no
# symptom until a live call. One switch for both tokens makes that impossible.
_TELUGU_TOKENS = ("te", "tinglish")


def backend_for(token: str) -> str:
    """The voice backend that can carry *token*.

    Falls back to ELEVENLABS for anything unrecognised: normalize() already
    routes junk to AUTO, so reaching this means a caller hand-built a token,
    and degrading to the backend that is known to work beats raising mid-dial.
    """
    normalised = normalize(token)
    if normalised in _TELUGU_TOKENS:
        # Read at call time, not baked into the table at import, so a test can
        # exercise the rollback without reloading the module.
        return TELUGU_BACKEND
    return _BACKEND_BY_TOKEN.get(normalised, ELEVENLABS)


def backend_for_iso(iso: str | None) -> str:
    """The backend for an ISO code, for callers that only have the code.

    app/telephony/preflight.py is one such caller: it receives the ISO code
    that app/languages.py's own for_call()/iso_code() produced, never the raw
    catalogue token, so it cannot call backend_for() directly.

    Several tokens can share an ISO code (te and tinglish are both 'te'), but
    they never disagree about the backend — see _TELUGU_TOKENS above for what
    guarantees that — so resolving through the first token that matches is
    safe. Falls back to ELEVENLABS for an unrecognised or absent code, for the
    same reason backend_for() does: degrading to the backend that is known to
    work beats raising mid-dial.
    """
    for token in TOKENS:
        if iso_code(token) == iso:
            return backend_for(token)
    return ELEVENLABS


def twoway_enabled(backend: str) -> bool:
    """Whether *backend* may carry a TWO-WAY (conversational) campaign.

    Two-way on a backend that has not been proven on a live call is not a
    degraded call, it is a silent one: it connects, never speaks, never
    listens, bills the full CALL_MAX_DURATION_S of silence, and is then
    recorded as a SUCCESSFUL zero-turn call because max_duration counts as a
    clean exit. Hence a per-backend flag an operator flips only after taking
    the call themselves.

    ElevenLabs needs no flag — two-way English/Hindi is what production runs.
    An unrecognised backend is refused rather than dialled: reaching here with
    one means a bug, and a lead should not carry the cost of it.
    """
    if backend == ELEVENLABS:
        return True
    if backend == SARVAM:
        return SARVAM_TWOWAY_ENABLED
    if backend == OPENAI_REALTIME:
        return OPENAI_TWOWAY_ENABLED
    return False


def backend_display(backend: str) -> str:
    """*backend* as a name an operator can act on.

    These strings land in the refusal an operator reads in the dashboard.
    preflight and admin/campaigns used to hardcode "OpenAI Realtime" in that
    message, which quietly became a lie the moment Telugu moved to Sarvam.
    """
    return _BACKEND_DISPLAY.get(backend, backend)


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
