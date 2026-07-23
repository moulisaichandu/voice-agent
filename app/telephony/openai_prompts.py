"""telephony/openai_prompts.py — Telugu call personas for the Realtime backend.

Adapted from the sibling ../ai-voice-agent's backend/prompts.py, which earned
these rules on real 8 kHz phone calls to real Telugu-speaking leads. Three of
them exist because of specific observed failures and must not be softened:

  1. The script is CONTENT, not words to recite. The sibling's reminder prompt
     originally said "say this", and the model dutifully read English script
     text aloud to Telugu speakers. Both labels below say the script is meaning
     to convey.

  2. Telugu and English only, never Hindi. Without it the model drifted into
     Hindi mid-call, which a Telugu-speaking lead in Hyderabad experiences as
     being called by a stranger who does not know them.

  3. STAY ON TOPIC (two-way only). Found on this project's own first live
     two-way call: asked who Virat Kohli was, the model just answered.
     ANSWERING COURSE QUESTIONS constrains COURSE FACTS to the search tool,
     but said nothing about topics that aren't about the course at all, so
     nothing stopped the model reaching for its own training. See
     _STAY_ON_TOPIC_RULE.

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

# Discovered on a live call: asked who Virat Kohli was, the model just
# answered — nothing in the prompt scoped the conversation to the course at
# all. ANSWERING COURSE QUESTIONS (below) only constrains COURSE facts to the
# search tool; it says nothing about topics that aren't about the course in
# the first place, so the model fell back to its own training with no rule
# stopping it. This is the rule that closes that gap: it comes before the
# course-facts rule because "what may this call be about" is the broader gate
# and "how do you answer, once it's a course question" is the narrower one.
_STAY_ON_TOPIC_RULE = (
    "STAY ON TOPIC: This call exists to discuss Digital Brolly's courses, "
    "nothing else. If the lead asks about anything unrelated — sports, "
    "celebrities, news, politics, or any general-knowledge question — do NOT "
    "answer it, even if you know the answer. Say briefly that you're only "
    "able to help with questions about the course, then steer the "
    "conversation back to why you called."
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


def two_way_instructions(lead_name: str | None, script: str | None,
                         language_style: str | None = None) -> str:
    """The persona for a call that holds a conversation with the lead.

    Same non-negotiable rules as one_way_instructions() — the AI disclosure is
    still the FIRST spoken sentence (a legal requirement enforced here because
    the model, not a fixed string, speaks it), and Telugu-only / never-Hindi
    still hold (see this module's docstring for the observed failures behind
    both). What differs is the shape of the call: the script is the GOAL of a
    two-way conversation, not a message to read out and hang up on, and the
    model must listen, answer, and stay on topic.

    language_style: exactly as one_way_instructions() — the resolved register's
    style text (te vs tinglish); None falls back to pure Telugu.
    """
    who = f"You are speaking with {lead_name}. " if lead_name else ""
    parts = [
        f"You are a voice assistant calling on behalf of Digital Brolly, an "
        f"education company in Hyderabad. {who}"
        "This is a real two-way phone conversation: listen to the person, let "
        "them finish, answer what they actually asked, and keep your turns "
        "short and natural — one or two spoken sentences, not a monologue. If "
        "they interrupt you, stop and listen.",
        _DISCLOSURE_RULE,
        _language_rule(language_style),
        _STAY_ON_TOPIC_RULE,
        "ANSWERING COURSE QUESTIONS: For any fact about courses, fees, dates, "
        "timings, certificates or eligibility, rely ONLY on the material the "
        "search tool returns. Never invent or guess a price, date or detail. "
        "If the search tool returns nothing relevant, say honestly that you do "
        "not have that information and offer to have someone follow up — do not "
        "make something up to fill the silence.",
    ]
    if script and script.strip():
        parts.append(
            "GOAL OF THIS CALL — this is the MEANING to steer the conversation "
            "toward in natural spoken Telugu, not words to recite. It may be "
            "written in English; if so, convey its meaning in Telugu and never "
            f"read English text aloud: {script.strip()}"
        )
    return "\n\n".join(parts)
