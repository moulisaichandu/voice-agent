"""telephony/sarvam_prompts.py — what the Sarvam backend is told to say.

Pure strings, no I/O — testable without a socket, the same shape as
app/telephony/openai_prompts.py.

Most of the rules here are carried over from that module, because they were
earned on real 8 kHz calls to real Telugu-speaking leads and none of them
became less true when the vendor changed:

  1. The script is CONTENT, not words to recite. The sibling
     ../ai-voice-agent's prompt originally said "say this", and the model
     dutifully read English script text aloud to Telugu speakers.
  2. Telugu and English only, never Hindi. Without it the model drifted into
     Hindi mid-call, which a Telugu-speaking lead in Hyderabad experiences as
     being called by a stranger who does not know them.
  3. The AI disclosure is the first sentence. Legal requirement in India.

One rule is NEW here, and is the reason this module exists rather than reusing
openai_prompts. On the OpenAI backend the model speaks its own output, so a
stray "Sure, here you go:" is merely conversational. On this backend the LLM
writes TEXT that Sarvam's TTS then reads out verbatim — so a preamble, a note
about the translation, or a pair of quotation marks all become sounds a real
lead hears. The model must emit the spoken words and nothing else.
"""

from __future__ import annotations

_NO_HINDI_RULE = "Never use Hindi or any other language under any circumstances."

# Used only when no language_style arrives — every call routed here resolves to
# 'te' or 'tinglish' and both always carry a style, so this guards a caller
# that bypasses call_routes' resolution entirely. Pure Telugu deliberately:
# such a caller must not accidentally get permission to mix in English.
_FALLBACK_STYLE = "Speak only in Telugu (తెలుగు) throughout. Do not switch to English."

_DISCLOSURE_RULE = (
    "FIRST SENTENCE: The very first sentence must state plainly, in Telugu, "
    "that this is an automated AI call from Digital Brolly. This is a legal "
    "requirement in India and is not optional. It comes before anything else — "
    "before any greeting, before the company's name, before the reason for "
    "the call."
)

# The rule that only this backend needs. Deliberately concrete about the
# failure modes rather than a general "be concise": a model told to "just give
# the translation" still routinely wraps it in quotation marks, and quotation
# marks are read aloud by a TTS as a change in tone at best and as the words
# "quote unquote" at worst.
_OUTPUT_ONLY_RULE = (
    "OUTPUT FORMAT: Reply with ONLY the words to be spoken aloud, in Telugu "
    "script. Your entire reply is fed directly to a speech synthesiser and "
    "read aloud to the person on the phone, exactly as you write it. Do not "
    "add a preamble, an explanation, a note about the translation, a label, "
    "or quotation marks around the message — every character you write is "
    "heard by a real person."
)

# A short, name-only opener. It must stay a BARE GREETING as
# app/compliance/disclosure.py defines one — few words, and naming no caller —
# because that is what lets has_ai_disclosure() keep looking at the sentence
# AFTER it for the disclosure. Add the company name here, or a second clause,
# and every Telugu campaign starts failing its own compliance check.
_GREETING_TEMPLATE = "నమస్తే {name} గారు."


def _language_rule(language_style: str | None) -> str:
    """The LANGUAGE instruction for this call's register.

    *language_style* is app/languages.py's style() text for the resolved token
    — 'te' forbids English mixing, 'tinglish' explicitly permits certain
    everyday words to stay in English. The anti-Hindi rule is appended
    unconditionally: it comes from a different observed failure and must
    survive whichever register arrives.
    """
    return f"LANGUAGE: {language_style or _FALLBACK_STYLE} {_NO_HINDI_RULE}"


def render_instructions(script: str, *, language_style: str | None = None) -> str:
    """The system prompt that turns an operator's script into spoken Telugu.

    The operator may write in English; what the lead hears must be Telugu. The
    script is handed over as MEANING to convey, never as words to recite.
    """
    return "\n\n".join([
        "You write the words a voice assistant will speak on an outbound phone "
        "call for Digital Brolly, an education company in Hyderabad. The call "
        "delivers one short message and then ends — it does not ask questions "
        "and does not wait for a reply.",
        _DISCLOSURE_RULE,
        _language_rule(language_style),
        _OUTPUT_ONLY_RULE,
        "MESSAGE TO CONVEY — this is the MEANING to express in natural spoken "
        "Telugu, not words to recite. It may be written in English; if so, "
        "convey its meaning in Telugu and never read English text aloud: "
        f"{script.strip()}",
    ])


def greeting_line(lead_name: str) -> str:
    """A short spoken greeting naming *lead_name*.

    Kept out of the rendered body so the body can be cached per campaign — an
    Indian name needs no translation, so splicing it in costs nothing and no
    LLM call. See _GREETING_TEMPLATE for why it must stay this short.
    """
    return _GREETING_TEMPLATE.format(name=lead_name.strip())


def compose_spoken(body: str, *, lead_name: str | None) -> str:
    """The exact words the lead will hear: greeting, if any, then *body*.

    A lead with no name gets the body alone rather than a greeting addressed
    to nobody — most uploaded leads are phone numbers only.
    """
    if not lead_name or not lead_name.strip():
        return body
    return f"{greeting_line(lead_name)} {body}"


# Earned on a live call on the OpenAI backend: asked who Virat Kohli was, the
# model simply answered — nothing in the prompt had scoped the conversation, so
# it fell back on its training with no rule to stop it. The first version of
# this rule listed example topics to refuse; naming categories invites exactly
# the failure it exists to prevent, because a model reads a list as bounding the
# rule. This version is a blanket ban with no list and no severity qualifier.
_STAY_ON_TOPIC_RULE = (
    "STAY ON TOPIC — ALWAYS, NO EXCEPTIONS: You may ONLY discuss Digital "
    "Brolly's courses and the reason for this call. Do not answer ANY question "
    "outside that scope, for ANY reason, no matter how simple, harmless or "
    "unrelated it seems — even if you are completely certain of the answer. "
    "Not one fact, opinion, definition or piece of outside knowledge, ever. If "
    "the lead asks about anything else, say briefly that you can only help "
    "with questions about the course, then return to why you called."
)

# CLAUDE.md's hard rule, in the model's own terms. The danger is not that the
# model refuses to answer — it is that it answers a fee or a date from memory
# and sounds exactly as confident as it does when quoting a real document.
_COURSE_FACTS_RULE = (
    "COURSE FACTS: Every question about courses, fees, timings, batches, "
    "dates or placement must be answered from the search_course_material "
    "tool, and ONLY from what it returns. Never answer them from memory and "
    "never invent a number or a date. If the tool returns nothing relevant, "
    "say plainly that you do not have that information and offer to have a "
    "person call back."
)

# A phone call, not a chat window. The lead cannot skim, and every extra
# sentence is one more they have to interrupt to get a word in.
_BREVITY_RULE = (
    "LENGTH: Keep every reply short — one or two sentences, the way people "
    "actually speak on the phone. Ask one question at a time and then stop "
    "talking and listen. Never deliver a paragraph."
)


def two_way_instructions(lead_name: str | None, script: str | None,
                         language_style: str | None = None) -> str:
    """The persona for a call that holds a conversation.

    Differs from one_way_instructions in one structural way: the script is the
    GOAL of the call rather than a message to deliver. A model told to
    "deliver" a script in a conversation will talk over the lead's questions to
    finish it, which is precisely the experience two-way exists to avoid.

    Everything else is deliberately identical. Nothing about a conversation
    makes the disclosure, the anti-Hindi rule or the verbatim-output rule less
    true — they are only easier to forget when writing a second prompt.
    """
    who = f"You are speaking with {lead_name}. " if lead_name else ""
    parts = [
        "You are a voice assistant on an outbound phone call for Digital "
        "Brolly, an education company in Hyderabad. "
        f"{who}"
        "You called them, so you speak first. Have a natural conversation: "
        "listen to what they say, answer it, and stop.",
        _DISCLOSURE_RULE,
        _language_rule(language_style),
        _OUTPUT_ONLY_RULE,
        _BREVITY_RULE,
        _COURSE_FACTS_RULE,
        _STAY_ON_TOPIC_RULE,
        # Observed on the live API: told to "say a short goodbye and call the
        # end_call tool", the model said the goodbye and did not call the tool.
        # Nothing then ends the call, so it runs to CALL_MAX_DURATION_S — five
        # minutes of billed airtime and silence at a lead who has already left.
        # Saying goodbye and ending the call have to be ONE instruction, not a
        # sentence containing both.
        "ENDING THE CALL: Saying goodbye and calling the end_call tool are the "
        "SAME action — never do one without the other. The moment the "
        "conversation is finished (they say goodbye, ask not to be called "
        "again, or have nothing further to ask), your reply must be a short "
        "farewell AND a call to end_call in that same turn. A goodbye without "
        "end_call leaves the lead holding a silent line.",
    ]
    if script and script.strip():
        parts.append(
            "GOAL OF THIS CALL — this is what you are trying to achieve, not a "
            "speech to read out. Work it into the conversation naturally, in "
            "your own spoken Telugu, and never at the cost of ignoring what "
            f"the lead just said: {script.strip()}"
        )
    return "\n\n".join(parts)
