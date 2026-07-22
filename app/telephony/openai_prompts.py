"""telephony/openai_prompts.py — Telugu call personas for the Realtime backend.

Adapted from the sibling ../ai-voice-agent's backend/prompts.py, which earned
these rules on real 8 kHz phone calls to real Telugu-speaking leads. Two of
them exist because of specific observed failures and must not be softened:

  1. The script is CONTENT, not words to recite. The sibling's reminder prompt
     originally said "say this", and the model dutifully read English script
     text aloud to Telugu speakers. Both labels below say the script is meaning
     to convey.

  2. Telugu and English only, never Hindi. Without it the model drifted into
     Hindi mid-call, which a Telugu-speaking lead in Hyderabad experiences as
     being called by a stranger who does not know them.

The disclosure rule is not a preference either: India's TCCCPR requires the
caller to say it is automated, CLAUDE.md makes it the first line of every
script, and here it is the model — not a fixed string — that speaks it, so the
instruction is the only thing enforcing it on this backend. Weakening it is a
compliance failure, and tests/unit/test_openai_bridge.py pins it.

Pure strings, no I/O — testable without a socket.
"""

from __future__ import annotations

_NO_HINDI_RULE = "Never use Hindi or any other language under any circumstances."

# Used only when no language_style arrives at all — every call that reaches
# this backend today resolves to 'te' or 'tinglish' (see app/languages.py's
# backend_for()), and both always carry a style, so this guards a caller that
# bypasses call_routes' resolution path entirely, not a real dial-time case.
# Pure Telugu, deliberately: a caller skipping resolution must not accidentally
# get permission to mix in English it never asked for.
_FALLBACK_STYLE = "Speak only in Telugu (తెలుగు) throughout. Do not switch to English."

_DISCLOSURE_RULE = (
    "FIRST SENTENCE: Your very first sentence must state plainly, in Telugu, "
    "that this is an automated AI call from Digital Brolly. This is a legal "
    "requirement in India and is not optional. Say it before anything else — "
    "before greeting them, before your name, before the reason for the call."
)


def _language_rule(language_style: str | None) -> str:
    """The LANGUAGE instruction for this call's register.

    *language_style* is app/languages.py's style() text for the resolved
    token — 'te' forbids English mixing, 'tinglish' explicitly permits
    certain everyday words to stay in English. Building the rule from that,
    rather than one constant shared by both, is the whole point of this
    function: before it existed, both registers got the same hardcoded rule
    (one that happened to permit mixing), so pure Telugu wrongly invited
    English and Tinglish was indistinguishable from it.

    The anti-Hindi rule is appended unconditionally, independent of register —
    it comes from a different observed failure (the sibling's model drifting
    into Hindi mid-call) that has nothing to do with whether English mixing
    is allowed, so it must survive regardless of which style text arrives.
    """
    style = language_style or _FALLBACK_STYLE
    return f"LANGUAGE: {style} {_NO_HINDI_RULE}"


def one_way_instructions(lead_name: str | None, script: str | None,
                         language_style: str | None = None) -> str:
    """The persona for a call that delivers a message and hangs up.

    language_style: the exact style text app/languages.py's style() built for
    this call's resolved language token (te vs tinglish) — put in
    dynamic_variables['language_style'] by
    app/telephony/call_routes._call_language() and passed through unchanged by
    openai_bridge.bridge(). None falls back to a pure-Telugu default; see
    _language_rule().
    """
    who = f"You are calling {lead_name}. " if lead_name else ""
    parts = [
        f"You are a voice assistant calling on behalf of Digital Brolly, an "
        f"education company in Hyderabad. {who}"
        "Deliver one short message, then say goodbye and stop. Do not ask "
        "questions and do not wait for a reply — this call does not listen.",
        _DISCLOSURE_RULE,
        _language_rule(language_style),
    ]
    if script and script.strip():
        parts.append(
            "MESSAGE TO DELIVER — this is the MEANING to convey in natural "
            "spoken Telugu, not words to recite. It may be written in English; "
            "if so, convey its meaning in Telugu and never read English text "
            f"aloud: {script.strip()}"
        )
    return "\n\n".join(parts)
