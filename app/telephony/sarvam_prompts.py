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

import re

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

# Telugu-capable chat models sometimes emit an ASCII-like Telugu skeleton:
# consonants are present but dependent vowel signs are dropped (for example
# "కరస" instead of "కోర్సు" and "ఫజ" instead of "ఫీజు"). Sarvam TTS then
# faithfully pronounces that malformed text, so this must be prevented at the
# text-generation boundary. Keep the examples in Telugu script: they are both
# a spelling reference and a strong signal that the answer must contain real
# Telugu orthography, not a consonant-only approximation.
_TELUGU_SPELLING_RULE = (
    "TELUGU SPELLING: Use complete, natural Telugu orthography with all vowel "
    "signs and case endings. Never drop vowel signs or shorten words into a "
    "consonant-only form. Always spell these terms exactly as shown when they "
    "apply: నమస్తే, గారు, ఇది, డిజిటల్ బ్రోలీ, ఆటోమేటెడ్, కాల్, డిజిటల్ "
    "మార్కెటింగ్, కోర్సు, కోర్సులో, కోర్సు వ్యవధి, ఫీజు, ఫీజు నిర్మాణం, "
    "నెలలు, నెల, సంవత్సరాలు, రూపాయలు, ఇంటర్న్‌షిప్, ట్రైనింగ్, "
    "అసైన్‌మెంట్లు, ప్రాజెక్ట్‌లు, ప్లేస్‌మెంట్ సపోర్ట్, ధన్యవాదాలు, "
    "శుభదినం. Write rupee amounts clearly with the ₹ symbol and comma "
    "grouping, for example ₹1,50,000; do not spell a rupee amount as a "
    "garbled Telugu word. Before replying, silently proofread every Telugu "
    "word for missing vowel signs."
)

# Phone TTS is more reliable when Indian currency is written as words rather
# than a rupee symbol followed by comma-grouped digits.
_RUPEE_SPEECH_RULE = (
    "Speak rupee amounts as natural Telugu words followed by రూపాయలు. Do not "
    "send the ₹ symbol or comma-formatted digits to the speech synthesiser; "
    "for example, say ఒక లక్షా యాభై వేల రూపాయలు for ₹1,50,000."
)
_TELUGU_SPELLING_RULE = _TELUGU_SPELLING_RULE.replace(
    "Write rupee amounts clearly with the \u20b9 symbol and comma grouping, "
    "for example \u20b91,50,000; do not spell a rupee amount as a garbled "
    "Telugu word.",
    _RUPEE_SPEECH_RULE,
)

# A small defensive repair for the recurring forms observed in production
# transcripts. This is intentionally a whitelist, not a Telugu spellchecker:
# guessing at arbitrary lead text could change its meaning. It is applied only
# to agent replies, never to the lead's words or to RAG source material.
_COMMON_MANGLED_TELUGU = {
    "నమసత": "నమస్తే",
    "గర": "గారు",
    "ఇద": "ఇది",
    "నడ": "నుండి",
    "డజటల": "డిజిటల్",
    "బరల": "బ్రోలీ",
    "ఆటమటడ": "ఆటోమేటెడ్",
    "కల": "కాల్",
    "మ": "మా",
    "మరకటగ": "మార్కెటింగ్",
    "కరస": "కోర్సు",
    "కరసల": "కోర్సుల",
    "గరచ": "గురించి",
    "మక": "మీకు",
    "వవరగ": "వివరంగా",
    "చపతర": "చెప్తారా",
    "గగల": "గూగుల్",
    "అడస": "యాడ్స్",
    "మట": "మెటా",
    "సషల": "సోషల్",
    "మడయ": "మీడియా",
    "కటట": "కంటెంట్",
    "రటగ": "రైటింగ్",
    "అనలటకస": "అనలిటిక్స్",
    "యటయబ": "యూట్యూబ్",
    "వటసప": "వాట్సాప్",
    "చటజపట": "చాట్‌జీపీటీ",
    "టలస": "టూల్స్",
    "ఉననయ": "ఉన్నాయి",
    "ఇదల": "ఇందులో",
    "నల": "నెల",
    "ఉటద": "ఉంటుంది",
    "ఎబఏ": "MBA",
    "డటయలస": "డిటైల్స్",
    "ఫజ": "ఫీజు",
    "టరనగ": "ట్రైనింగ్",
    "డపలమ": "డిప్లొమా",
    "ఆన-జబ": "ఆన్-జాబ్",
    "ఫ": "ఫీజు",
    "సటరకచర": "స్ట్రక్చర్",
    "మతత": "మొత్తం",
    "అదల": "అదనపు",
    "నలల": "నెలలు",
    "సవతసరల": "సంవత్సరాలు",
    "ఇటరనషప": "ఇంటర్న్‌షిప్",
    "పలసమట": "ప్లేస్‌మెంట్",
    "సపరట": "సపోర్ట్",
    "ధనయవదల": "ధన్యవాదాలు",
    "శభదన": "శుభదినం",
    "థయక": "థ్యాంక్",
    "య": "యూ",
}

# Incoming STT text is evidence of what the lead said, so it must be repaired
# more conservatively than agent text. These are course-domain words that
# Sarvam has repeatedly returned without dependent vowel signs. Do not include
# ambiguous one-letter entries such as "మ" or "య" here: changing a lead's
# short answer can change its meaning.
# Entries that must never run against a LEAD's speech.
#
# CLAUDE.md's rule for term_repair governs any rewrite of what a lead said:
# "Adding a term is a licence to put words in a lead's mouth ... keep the
# false-positive count at zero." This map runs BEFORE term_repair and its
# output feeds the LLM history, the RAG query and the stored transcript, so it
# is held to the same standard.
#
#   మ / ఫ / య  — one letter; changing a short answer changes its meaning.
#   కల        — "dream". Was rewritten to కాల్ ("call").
#   నల        — "black". Was rewritten to నెల ("month").
#   మట        — collides with మట్టి ("soil") and similar.
#   బరల       — the BRAND. term_repair owns this, and only trusts a weak match
#               when డిజిటల్ precedes it, because four words sit within edit
#               distance 2 of the brand and only two of them are the brand.
#               An unconditional entry here defeats that anchor entirely.
_AMBIGUOUS_FOR_HEARD = {"మ", "ఫ", "య", "కల", "నల", "మట", "బరల"}

_HEARD_MANGLED_TELUGU = {
    key: value for key, value in _COMMON_MANGLED_TELUGU.items()
    if key not in _AMBIGUOUS_FOR_HEARD
}
_HEARD_MANGLED_TELUGU.update({
    "ఓక": "ఓకే",
    "ఎనన": "ఎన్ని",
    "డస": "రోజులు",
    "డయరషన": "డ్యూరేషన్",
    "మరయ": "మరియు",
})

_MANGLED_PHRASES = {
    "ఏమన తలసకవలన ఉట, ననన అడగడ": "ఏమైనా తెలుసుకోవాలనుకుంటే నన్ను అడగండి",
    "SEO, గగల అడస, మట అడస": "SEO, గూగుల్ యాడ్స్, మెటా యాడ్స్",
    "సషల మడయ మరకటగ": "సోషల్ మీడియా మార్కెటింగ్",
    "కటట రటగ": "కంటెంట్ రైటింగ్",
    "యటయబ మరకటగ": "యూట్యూబ్ మార్కెటింగ్",
    "వటసప మరకటగ": "వాట్సాప్ మార్కెటింగ్",
    "చటజపట మరయ ఏఐ టలస": "చాట్‌జీపీటీ మరియు AI టూల్స్",
    "పరకటకల అసనమటల": "ప్రాక్టికల్ అసైన్‌మెంట్లు",
    "లవ పరజకటల": "లైవ్ ప్రాజెక్ట్‌లు",
    "ఇటరనషప ఎకసపజర": "ఇంటర్న్‌షిప్ ఎక్స్‌పోజర్",
    "ఇటరవయ పరపరషన": "ఇంటర్వ్యూ ప్రిపరేషన్",
    "పలసమట సపరట కడ ఉటద": "ప్లేస్‌మెంట్ సపోర్ట్ కూడా ఉంటుంది",
    "కరస డటయలస": "కోర్సు డీటెయిల్స్",
    "ఎనన డస ఉటద": "ఎన్ని రోజులు ఉంటుంది",
    "డయరషన కరస డయరషన": "డ్యూరేషన్, కోర్సు డ్యూరేషన్",
    "ఫజ సటరకచరగ": "ఫీజు స్ట్రక్చర్‌గా",
    "అదల ఎబఏ ఫ": "అదనపు MBA ఫీజు",
    "పలసమట ఫ": "ప్లేస్‌మెంట్ ఫీజు",
    "టరనగ/డపలమ": "ట్రైనింగ్/డిప్లొమా",
    "ఆన-జబ టరనగ": "ఆన్-జాబ్ ట్రైనింగ్",
    "నడ ఆటమటడ": "నుండి ఆటోమేటెడ్",
    "కరస ఎనన డస ఉటద": "కోర్సు ఎన్ని రోజులు ఉంటుంది",
}


def _repair_tokens(text: str, mapping: dict[str, str]) -> str:
    """Apply a whole-token spelling map without disturbing punctuation."""
    # Line by line. Joining the whole text on " " flattened paragraphs and
    # lists — on the composed one-way script, on every say() payload and on the
    # cached render — before TTS spoke it and before it was stored.
    out_lines: list[str] = []
    for line in text.split("\n"):
        repaired: list[str] = []
        for word in line.split():
            leading = word[:len(word) - len(word.lstrip("([{\"'"))]
            trailing = word[len(word.rstrip(".,!?;:)]}\"'")):]
            core = word[len(leading):len(word) - len(trailing) if trailing else None]
            repaired.append(f"{leading}{mapping.get(core, core)}{trailing}")
        out_lines.append(" ".join(repaired))
    return "\n".join(out_lines)


# [0-9][0-9,]* was greedy over "," and ate the SENTENCE comma after an
# amount ("₹50,000, and ...") — and with it the pause TTS gives it.
# A comma is only part of the number when a digit follows it.
_RUPEE_RE = re.compile(r"₹\s*([0-9](?:,?[0-9])*)")
_TELUGU_UNDER_20 = {
    0: "సున్నా", 1: "ఒకటి", 2: "రెండు", 3: "మూడు", 4: "నాలుగు",
    5: "ఐదు", 6: "ఆరు", 7: "ఏడు", 8: "ఎనిమిది", 9: "తొమ్మిది",
    10: "పది", 11: "పదకొండు", 12: "పన్నెండు", 13: "పదమూడు",
    14: "పద్నాలుగు", 15: "పదిహేను", 16: "పదహారు", 17: "పదిహేడు",
    18: "పద్దెనిమిది", 19: "పంతొమ్మిది",
}
_TELUGU_TENS = {
    20: "ఇరవై", 30: "ముప్పై", 40: "నలభై", 50: "యాభై",
    60: "అరవై", 70: "డెబ్బై", 80: "ఎనభై", 90: "తొంభై",
}
_COMMON_RUPEE_WORDS = {
    "150000": "ఒక లక్షా యాభై వేల",
    "50000": "యాభై వేల",
    "100000": "ఒక లక్ష",
    "165000": "ఒక లక్షా అరవై ఐదు వేల",
    "120000": "ఒక లక్షా ఇరవై వేల",
    "45000": "నలభై ఐదు వేల",
}


def _under_100(number: int) -> str:
    if number < 20:
        return _TELUGU_UNDER_20[number]
    tens = number - (number % 10)
    return _TELUGU_TENS[tens] + (f" {_TELUGU_UNDER_20[number % 10]}"
                                 if number % 10 else "")


def _generic_rupee_words(number: int) -> str:
    """Spell an Indian integer amount well enough for phone TTS."""
    if number < 100:
        return _under_100(number)
    # _under_100 only knows 0-99, so a crore count of 100 or more raised
    # KeyError — which escapes re.sub, normalize_spoken_telugu and say(),
    # ending the turn. A hallucinated or mistyped figure is enough to hit it.
    # Digits read aloud are a poor answer; a dead turn is a worse one.
    if number >= 100 * 10_000_000:
        return str(number)

    parts: list[str] = []
    crore, number = divmod(number, 10_000_000)
    if crore:
        parts.append(f"{_under_100(crore)} కోట్లు")
    lakh, number = divmod(number, 100_000)
    if lakh:
        parts.append(f"{_under_100(lakh)} లక్షలు")
    thousand, number = divmod(number, 1_000)
    if thousand:
        parts.append("వెయ్యి" if thousand == 1
                     else f"{_under_100(thousand)} వేల")
    hundred, number = divmod(number, 100)
    if hundred:
        parts.append("వంద" if hundred == 1
                     else f"{_TELUGU_UNDER_20[hundred]} వందల")
    if number:
        parts.append(_under_100(number))
    return " ".join(parts)


def _spoken_rupees(match: re.Match[str]) -> str:
    digits = match.group(1).replace(",", "")
    try:
        number = int(digits)
    except ValueError:
        return f"{match.group(1)} రూపాయలు"
    words = _COMMON_RUPEE_WORDS.get(digits) or _generic_rupee_words(number)
    return f"{words} రూపాయలు"


def _replace_rupee_amounts(text: str) -> str:
    """Make the common Indian fee amounts unambiguous for Sarvam TTS."""
    return _RUPEE_RE.sub(_spoken_rupees, text)


def normalize_spoken_telugu(text: str) -> str:
    """Repair known agent spellings and make rupees speakable."""
    for bad, good in _MANGLED_PHRASES.items():
        text = text.replace(bad, good)
    return _replace_rupee_amounts(_repair_tokens(text, _COMMON_MANGLED_TELUGU))


def normalize_heard_telugu(text: str) -> str:
    """Repair recurring vowel-dropped STT domain words before the LLM sees them.

    This is intentionally separate from ``normalize_spoken_telugu``. Agent
    output can use a few known fallback repairs; lead speech must never be
    aggressively rewritten just because a token is short or ambiguous.
    """
    for bad, good in _MANGLED_PHRASES.items():
        text = text.replace(bad, good)
    return _repair_tokens(text, _HEARD_MANGLED_TELUGU)

# A short, name-only opener. It must stay a BARE GREETING as
# app/compliance/disclosure.py defines one — few words, and naming no caller —
# because that is what lets has_ai_disclosure() keep looking at the sentence
# AFTER it for the disclosure. Add the company name here, or a second clause,
# and every Telugu campaign starts failing its own compliance check.
_GREETING_TEMPLATE = "నమస్తే {name} గారు."

# What a two-way call opens with when its campaign has no script.
#
# REQUIRED, not a nicety. app/telephony/call_routes.py only passes `script`
# into a call's dynamic_variables when mode == "oneway", so EVERY two-way call
# arrives with none. Without this the rendered body is empty and the opening
# collapses to the bare name greeting — which is what a real lead heard on
# 2026-07-27: a call with no AI disclosure at all.
#
# Written in English like any operator script, so it renders into whichever
# register the campaign uses and picks up the same disclosure repair.
DEFAULT_TWOWAY_SCRIPT = (
    "This is an automated AI call from Digital Brolly. I am calling about our "
    "digital marketing courses. Ask me anything you would like to know."
)


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
        _TELUGU_SPELLING_RULE,
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
        _TELUGU_SPELLING_RULE,
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
