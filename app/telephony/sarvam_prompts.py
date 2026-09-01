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
import unicodedata

from app import languages

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

# The same rule for a code-mixed register. "In Telugu script" was ordering the
# model to transliterate the very words the LANGUAGE rule told it to keep in
# English — one prompt, two instructions, and the model picked one at random
# per turn. Sarvam's TTS reads a Latin run as an English word, which for
# "course" or "fees" is exactly right (it is only a NAME like "Mouli" that
# it gets wrong, see lead_name.py).
_OUTPUT_ONLY_RULE_MIXED = (
    "OUTPUT FORMAT: Reply with ONLY the words to be spoken aloud. Telugu is "
    "written in Telugu script; the everyday English words the LANGUAGE rule "
    "keeps in English are written in ordinary English letters, spelled the "
    "ordinary English way. Your entire reply is fed directly to a speech "
    "synthesiser and read aloud to the person on the phone, exactly as you "
    "write it. Do not add a preamble, an explanation, a note about the "
    "translation, a label, or quotation marks around the message — every "
    "character you write is heard by a real person."
)


def _output_only_rule(mixed: bool) -> str:
    return _OUTPUT_ONLY_RULE_MIXED if mixed else _OUTPUT_ONLY_RULE

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

# The vowel-sign rule is about Telugu orthography and holds in every register.
# What changes for a code-mixed register is the mandated-spelling list: the
# pure-Telugu one mandates కోర్సు, ఫీజు and ప్లేస్‌మెంట్ — three of the eight
# words languages.py's tinglish style keeps in English. Mandating a Telugu
# spelling for a word the same prompt says to leave in English was the
# contradiction the project's design notes warned about; here the list holds
# only words the register actually says in Telugu.
_TELUGU_SPELLING_RULE_MIXED = (
    "TELUGU SPELLING: Every Telugu word must use complete, natural Telugu "
    "orthography with all vowel signs and case endings. Never drop vowel "
    "signs or shorten words into a consonant-only form. Always spell these "
    "Telugu terms exactly as shown when they apply: నమస్తే, గారు, ఇది, "
    "డిజిటల్ బ్రోలీ, ఆటోమేటెడ్, కాల్, నెలలు, నెల, సంవత్సరాలు, రూపాయలు, "
    "ఇంటర్న్‌షిప్, ట్రైనింగ్, అసైన్‌మెంట్లు, ప్రాజెక్ట్‌లు, ధన్యవాదాలు, "
    "శుభదినం. The English words you keep in English are never transliterated "
    "into Telugu script. "
    + _RUPEE_SPEECH_RULE
    + " Before replying, silently proofread every Telugu word for missing "
    "vowel signs."
)


def _spelling_rule(mixed: bool) -> str:
    return _TELUGU_SPELLING_RULE_MIXED if mixed else _TELUGU_SPELLING_RULE

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
    # Not a skeleton: the conversation LLM writes the fully-voweled "అడ్స్"
    # when answering from RAG text that says "Ads" (heard live 2026-08-27 as
    # "aḍs"), while the campaign renders spell it "యాడ్స్" — one call could
    # contain both spellings of the same product name. Same-word orthography,
    # so it is safe on the heard path too.
    "అడ్స్": "యాడ్స్",
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
    # డీటెయిల్స్, matching the phrase map's spelling — the two maps used to
    # disagree (డిటైల్స్ here), so which rendering the lead heard depended on
    # whether కోర్సు happened to precede the word.
    "డటయలస": "డీటెయిల్స్",
    # Spoken orthography for the acronym: today's live calls said Latin "SEO"
    # in one turn and ఎస్ఈఓ in the next, and TTS reads the two differently.
    # Heard text is EXCLUDED below — it feeds the RAG query, where English
    # embeds far better against the English-only corpus.
    "SEO": "ఎస్ఈఓ",
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
#   మక        — the skeleton of BOTH మీకు ("to you") and మాకు ("to us").
#               Resolving it flips the person of the lead's sentence.
#   ఉటద / ఉననయ / చపతర — dropping dependent signs erases exactly the
#               interrogative -ా, so each skeleton is both the statement and
#               the question (ఉంటుంది/ఉంటుందా, ఉన్నాయి/ఉన్నాయా,
#               చెప్తారు/చెప్తారా). Resolving to either form rewrites the
#               lead's mood — a question becomes an assertion or vice versa.
#   SEO       — not ambiguity: heard text becomes the RAG query, and the
#               English-only course docs embed English far better (Telugu-
#               script queries measured 0.10-0.19 against them vs 0.43+ for
#               English), so converting the acronym in a lead's question
#               would sabotage the very lookup it triggers.
_AMBIGUOUS_FOR_HEARD = {"మ", "ఫ", "య", "కల", "నల", "మట", "బరల",
                        "మక", "ఉటద", "ఉననయ", "చపతర", "SEO"}

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
    # No keys ending in a bare consonant ("...ఫ"): such a key is a prefix of
    # its own repaired form, so it matched INSIDE correct text ("పలసమట ఫీజు")
    # and spliced a replacement mid-word. The token map's whole-token
    # discipline repairs those inputs correctly on its own.
    "టరనగ/డపలమ": "ట్రైనింగ్/డిప్లొమా",
    "ఆన-జబ టరనగ": "ఆన్-జాబ్ ట్రైనింగ్",
    "నడ ఆటమటడ": "నుండి ఆటోమేటెడ్",
    "కరస ఎనన డస ఉటద": "కోర్సు ఎన్ని రోజులు ఉంటుంది",
}

# Phrase keys are matched as whole phrases, never inside a longer word. Bare
# str.replace() spliced a key that was a PREFIX of the actual text mid-word
# ("పలసమట ఫ" inside "పలసమట ఫీజు" → "ప్లేస్‌మెంట్ ఫీజుీజు" — a dangling
# dependent vowel sign, read aloud by TTS in the sentence quoting the price).
# \b is useless here: Python's \w excludes combining marks, so every dependent
# vowel sign would BE a boundary. A word-character for this purpose is a
# letter, a digit, or anything in the Telugu block (which includes the
# dependent signs). Longest key first, so a key that is a suffix of another
# ("ఎనన డస ఉటద" / "కరస ఎనన డస ఉటద") can never shadow the longer match.
_PHRASE_BOUNDARY = "0-9A-Za-zఀ-౿"  # ASCII alphanumerics + Telugu block
_MANGLED_PHRASE_RE = re.compile(
    f"(?<![{_PHRASE_BOUNDARY}])"
    + "(?:" + "|".join(re.escape(key) for key in
                       sorted(_MANGLED_PHRASES, key=len, reverse=True)) + ")"
    + f"(?![{_PHRASE_BOUNDARY}])"
)


def _repair_phrases(text: str) -> str:
    return _MANGLED_PHRASE_RE.sub(lambda m: _MANGLED_PHRASES[m.group(0)], text)


def _repair_tokens(text: str, mapping: dict[str, str]) -> str:
    """Apply a whole-token spelling map without disturbing punctuation."""
    # Line by line. Joining the whole text on " " flattened paragraphs and
    # lists — on the composed one-way script, on every say() payload and on the
    # cached render — before TTS spoke it and before it was stored.
    out_lines: list[str] = []
    for line in text.split("\n"):
        repaired: list[str] = []
        for word in line.split():
            # Trailing is sliced from what remains AFTER leading, never from
            # the whole word — a token made only of quote characters (present
            # in both strip sets) was otherwise claimed by BOTH slices and
            # emitted twice, and the render cache's repair-on-hit loop then
            # doubled it again on every later call: 2^N growth.
            leading = word[:len(word) - len(word.lstrip("([{\"'"))]
            rest = word[len(leading):]
            trailing = rest[len(rest.rstrip(".,!?;:)]}\"'")):]
            core = rest[:len(rest) - len(trailing)]
            fixed = mapping.get(core)
            if fixed is None and "‌" in core:
                # The conversation model welds case endings onto loanwords
                # with ZWNJ (అడ్స్‌లో, మార్కెటింగ్‌లో — its own live output),
                # so a whole-token miss is retried on the pre-ZWNJ head with
                # the suffix carried over. Map-driven, so the heard path's
                # exclusions still apply to the head.
                head, sep, tail = core.partition("‌")
                if head in mapping:
                    fixed = mapping[head] + sep + tail
            repaired.append(f"{leading}{fixed if fixed is not None else core}{trailing}")
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
    # Lakh and crore need the same singular special-casing thousand and
    # hundred always had, and the OBLIQUE plural (లక్షల/కోట్ల), never the
    # nominative లక్షలు/కోట్లు — the amount is always followed by another
    # word (more parts, or రూపాయలు). "ఒక లక్షా" is the connective form used
    # when more parts follow, exactly as _COMMON_RUPEE_WORDS hand-writes it.
    crore, number = divmod(number, 10_000_000)
    if crore:
        parts.append("ఒక కోటి" if crore == 1 else f"{_under_100(crore)} కోట్ల")
    lakh, number = divmod(number, 100_000)
    if lakh == 1:
        parts.append("ఒక లక్షా" if number else "ఒక లక్ష")
    elif lakh:
        parts.append(f"{_under_100(lakh)} లక్షల")
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


# Everything the synthesiser is allowed to be handed. Telugu is the language;
# ASCII covers English loanwords, course codes and numerals, both of which
# Sarvam TTS reads correctly. Anything else is a model slip, not a message.
#
# Earned on a live call 2026-09-01: the agent said "బ్యాచ్ వివరాలు నాకు ఈ
# պահին లేవు" — "պահին" is ARMENIAN — and the synthesiser read it out at the
# lead. The prompt already forbade it; nothing enforced it. The same rule
# catches the older, documented failure of drifting into HINDI mid-call
# (CLAUDE.md), because Devanagari is no more Telugu than Armenian is.
#
# Judged on a word's LETTERS ONLY. The first cut of this guard tested whole
# tokens, so "ఉన్నాయి।" — a perfectly good Telugu word wearing the danda that
# _SENTENCE_END itself splits sentences on — was called foreign and DELETED,
# and a lead heard words vanish mid-sentence. Punctuation says nothing about
# what language a word is in.
#
# Whole tokens go, not individual characters: half a foreign word is not an
# improvement on all of it, and the surrounding Telugu still carries the
# sentence.
def _is_speakable(word: str) -> bool:
    for ch in word:
        category = unicodedata.category(ch)
        if not (category.startswith("L") or category.startswith("M")):
            continue          # punctuation, digits, symbols, spacing
        if ch.isascii():
            continue          # English loanwords, course codes
        if "ఀ" <= ch <= "౿":
            continue          # Telugu, including its combining vowel signs
        if ch in "‌‍":
            continue          # ZWNJ / ZWJ, used inside Telugu loanwords
        return False
    return True


def foreign_script_tokens(text: str) -> list[str]:
    """Words containing a script the synthesiser must never be given.

    Pure — callers log these; see say() in app/telephony/sarvam_bridge.py. A
    model emitting Armenian mid-sentence is worth an operator's attention
    even though the sentence around it is rescued.
    """
    return [w for w in text.split() if w and not _is_speakable(w)]


def normalize_spoken_telugu(text: str) -> str:
    """Repair known agent spellings, drop unspeakable scripts, and make
    rupees speakable."""
    foreign = set(foreign_script_tokens(text))
    if foreign:
        text = " ".join(w for w in text.split() if w not in foreign)
    return _replace_rupee_amounts(
        _repair_tokens(_repair_phrases(text), _COMMON_MANGLED_TELUGU))


def normalize_heard_telugu(text: str) -> str:
    """Repair recurring vowel-dropped STT domain words before the LLM sees them.

    This is intentionally separate from ``normalize_spoken_telugu``. Agent
    output can use a few known fallback repairs; lead speech must never be
    aggressively rewritten just because a token is short or ambiguous.
    """
    return _repair_tokens(_repair_phrases(text), _HEARD_MANGLED_TELUGU)

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

# ...and its Telugu, held as a constant rather than rendered.
#
# Translating a CONSTANT is a constant. Paying a model for it cost two live
# calls on 2026-08-27: Sarvam's reasoning had outgrown its token budget, the
# render returned nothing, and both leads answered the phone to silence and
# were hung up on 30 seconds later. Every scriptless two-way campaign — the
# normal case, since call_routes only forwards a script for one-way — was
# rendering this same fixed English through a 93-second reasoning model on
# every cold cache, for an answer that could never differ.
#
# NOT hand-written: this is what the model itself produced for this script,
# spoken to real leads on the calls of 2026-08-27 and passed by the
# post-call [compliance] disclosure re-check. A test pins the disclosure, and
# another pins that it survives normalize_spoken_telugu unchanged — it is
# spoken verbatim, so the repairs must have nothing left to do to it.
#
# "ఏఐ ద్వారా చేసే ఆటోమేటెడ్ కాల్", not "ఆటోమేటెడ్ AI కాల్": an audit of every
# agent turn ever spoken found "AI" to be the ONLY Latin token the agent
# says, and it is in every opening. Sarvam's Telugu TTS reads a Latin run as
# an English word rather than the letter names, so the lead heard something
# that was not "ఏ-ఐ" in the one sentence India's rules require them to
# understand. The sentence is RESTRUCTURED rather than substituted: the
# compliance check was matching that Latin \bai\b token, so swapping in ఏఐ
# alone made the opening fail has_ai_disclosure — it now carries
# "ఆటోమేటెడ్ కాల్" adjacently, which is a recognised marker in its own right.
DEFAULT_TWOWAY_TELUGU = (
    "ఇది డిజిటల్ బ్రోలీ నుండి ఏఐ ద్వారా చేసే ఆటోమేటెడ్ కాల్. "
    "నేను మా డిజిటల్ మార్కెటింగ్ కోర్సుల గురించి కాల్ చేస్తున్నాను. "
    "మీకు ఏమైనా తెలుసుకోవాలని ఉంటే అడగండి."
)

# The same opening in the Tinglish register. Until 2026-09-01 there was only
# the Telugu one, so a Tinglish campaign's first words were pure Telugu and
# the only thing marking the register was the system prompt behind them.
# Same shape as the Telugu: the disclosure sentence is IDENTICAL (it is the
# one sentence India's rules require the lead to understand, and it carries
# the recognised "ఆటోమేటెడ్ కాల్" marker rather than a Latin "AI" the
# synthesiser would read as an English word); the English that follows is
# limited to the everyday words languages.py's tinglish style keeps in
# English, which Sarvam's TTS reads correctly as English words.
DEFAULT_TWOWAY_TINGLISH = (
    "ఇది డిజిటల్ బ్రోలీ నుండి ఏఐ ద్వారా చేసే ఆటోమేటెడ్ కాల్. "
    "మా digital marketing courses గురించి కాల్ చేస్తున్నాను. "
    "Course fees, batch timings, placement, ఏమైనా తెలుసుకోవాలంటే అడగండి."
)


def default_twoway_opening(language_style: str | None) -> str:
    """The canned opening for a scriptless two-way call, in this call's
    register. No style at all means pure Telugu — the same default every
    other register-dependent rule in this module falls back to."""
    if languages.is_code_mixed(language_style):
        return DEFAULT_TWOWAY_TINGLISH
    return DEFAULT_TWOWAY_TELUGU


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
    mixed = languages.is_code_mixed(language_style)
    if mixed:
        framing = (
            "MESSAGE TO CONVEY — this is the MEANING to express in natural "
            "spoken Tinglish, Telugu mixed with everyday English the way "
            "people in Hyderabad speak, not words to recite. It may be written "
            "in English; if so, do not read it out as English sentences — say "
            "it the way a Hyderabad Telugu speaker would, keeping only the "
            "everyday English words the LANGUAGE rule names: "
        )
    else:
        framing = (
            "MESSAGE TO CONVEY — this is the MEANING to express in natural "
            "spoken Telugu, not words to recite. It may be written in English; "
            "if so, convey its meaning in Telugu and never read English text "
            "aloud: "
        )
    return "\n\n".join([
        "You write the words a voice assistant will speak on an outbound phone "
        "call for Digital Brolly, an education company in Hyderabad. The call "
        "delivers one short message and then ends — it does not ask questions "
        "and does not wait for a reply.",
        _DISCLOSURE_RULE,
        _language_rule(language_style),
        _output_only_rule(mixed),
        _spelling_rule(mixed),
        f"{framing}{script.strip()}",
    ])


def _greeting_name(lead_name: str) -> str | None:
    """The single token the greeting may address the lead by.

    The greeting must stay a BARE greeting as disclosure.py defines one — at
    most 3 words — and "నమస్తే {name} గారు" only holds that budget when
    {name} is ONE word. A 'First Last' name from the sheet made it 4 words,
    so has_ai_disclosure refused the roll-forward, ensure_disclosure prepended
    a SECOND disclosure the lead then heard twice, and a spurious
    [compliance] WARNING fired on every such call. A dot in the name
    ("B. మౌళి") is worse still: it ends the greeting 'sentence' early.

    So: first token, punctuation stripped, skipping bare Latin initials —
    "రామ కృష్ణ" greets రామ, "B. మౌళి" greets మౌళి. Dots split exactly like
    whitespace: "B.మౌళి" typed WITHOUT the space is common, and a kept
    internal dot still ends the greeting 'sentence' early — the same double
    disclosure by a glued spelling. None when nothing greetable remains;
    the caller then skips the greeting entirely.
    """
    for token in re.split(r"[\s.]+", lead_name):
        cleaned = token.strip(",!?;:'\"()")
        if not cleaned:
            continue
        if len(cleaned) == 1 and cleaned.isascii():
            continue  # an initial, not a name to address someone by
        return cleaned
    return None


def greeting_line(lead_name: str) -> str:
    """A short spoken greeting naming *lead_name* — "" when the name has no
    greetable token.

    Kept out of the rendered body so the body can be cached per campaign — an
    Indian name needs no translation, so splicing it in costs nothing and no
    LLM call. See _GREETING_TEMPLATE and _greeting_name for why it must stay
    this short.
    """
    name = _greeting_name(lead_name)
    if name is None:
        return ""
    return _GREETING_TEMPLATE.format(name=name)


def compose_spoken(body: str, *, lead_name: str | None) -> str:
    """The exact words the lead will hear: greeting, if any, then *body*.

    A lead with no name gets the body alone rather than a greeting addressed
    to nobody — most uploaded leads are phone numbers only.
    """
    if not lead_name or not lead_name.strip():
        return body
    line = greeting_line(lead_name)
    if not line:
        return body
    return f"{line} {body}"


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
    "with questions about the course, then return to why you called. "
    "But a SHORT REPLY TO YOUR OWN QUESTION is never off-topic: if you just "
    "asked which course or which fee they want and they answer with a bare "
    "number like 25,000, a course code like BDLP, or a fragment, they are "
    "ANSWERING you — treat it as naming that course and continue. "
    "And garbled or mis-heard speech that still mentions courses, marketing, "
    "fees, batches, classes or placement is ON topic — the speech-to-text on "
    "this line is imperfect; never call such a question off-topic, answer the "
    "closest sensible one or ask which they meant."
)

# Speech that reaches the model but was never said TO it. Live 2026-09-01:
# a mid-call "హలో" was answered with a fresh greeting, and other people
# talking near the phone were treated as questions. Carrier hold
# announcements are filtered in code before the model sees them
# (sarvam_bridge._is_hold_announcement); this covers what code cannot.
_NOT_THE_LEAD_RULE = (
    "Speech that is clearly not the lead talking to you — other people "
    "talking in the background, a television, a recorded announcement — is "
    "not a question: do not answer it and do not end the call because of it. "
    "Reply only with a short check, addressing them by name, such as asking "
    "whether they are still on the line, and wait."
)

# Sarvam's Telugu STT garbles heavily on 8 kHz phone audio: one live call
# produced "వాక్య తింగ ఏమేం కోషమ్స్", "స్కూట్స్ ప్లాంట్స్" and a bare "కో."
# from a lead who was engaged throughout. Unclear input is therefore the
# NORMAL case here, and on 2026-08-29 the model answered it in the two worst
# possible ways — a scope refusal, and hanging up. Asking again is the only
# honest response, and it costs one turn.
_UNCLEAR_INPUT_RULE = (
    "UNCLEAR SPEECH: The speech-to-text on this phone line is imperfect and "
    "often returns garbled or truncated Telugu. When what you receive does "
    "not make sense, is a stray fragment, or could mean several different "
    "things, ask them politely to say it again — in Telugu, in one short "
    "sentence. NEVER treat unclear text as a question about something "
    "off-topic, and NEVER end the call because of it. Only unmistakable "
    "words end a call. "
    + _NOT_THE_LEAD_RULE
)
_UNCLEAR_INPUT_RULE_MIXED = (
    "UNCLEAR SPEECH: The speech-to-text on this phone line is imperfect and "
    "often returns garbled or truncated text. When what you receive does "
    "not make sense, is a stray fragment, or could mean several different "
    "things, ask them politely to say it again — in the same natural mix of "
    "Telugu and everyday English, in one short sentence. NEVER treat unclear "
    "text as a question about something off-topic, and NEVER end the call "
    "because of it. Only unmistakable words end a call. "
    + _NOT_THE_LEAD_RULE
)


def _unclear_input_rule(mixed: bool) -> str:
    return _UNCLEAR_INPUT_RULE_MIXED if mixed else _UNCLEAR_INPUT_RULE

# CLAUDE.md's hard rule, in the model's own terms. The danger is not that the
# model refuses to answer — it is that it answers a fee or a date from memory
# and sounds exactly as confident as it does when quoting a real document.
# Two variants: with a pre-read facts digest in the prompt (the normal case
# since 2026-08-29 — the owner's direction was "read and store my documents
# BEFORE the call", after every common question paid a tool round-trip, a
# spoken waiting line, and the fee question still missed to top-k crowd-out),
# and without one, when the digest could not be built and the tool is all
# there is. Neither variant may promise a callback: the model said "మా టీమ్
# నుంచి ఎవరో తిరిగి కాల్ చేస్తారు" on two live calls, nothing schedules such
# a call, and the source of that promise was THIS RULE's own old wording.
_COURSE_FACTS_RULE_WITH_DIGEST = (
    "COURSE FACTS: The KNOWN COURSE FACTS section below was read from "
    "Digital Brolly's own course documents before this call. It covers the "
    "programs, fees, durations, online/offline modes, the curriculum's "
    "modules and topics (SEO, Google Ads, social media, AI tools and the "
    "rest), placement and internship, who can join, certificates and the "
    "institute's location. Answer from it DIRECTLY and IMMEDIATELY — no tool "
    "call, no waiting line — including 'what do you teach' and 'tell me "
    "about SEO' style questions. Use the search_course_material tool ONLY "
    "when the question is about something those facts say nothing about at "
    "all, and then answer ONLY from what it returns. Never answer from "
    "memory and never invent a number or a date. If neither the facts nor "
    "the tool has it, say plainly that you do not have that information and "
    "suggest they contact Digital Brolly directly — NEVER promise that "
    "someone will call them back; nothing schedules such a call, so the "
    "promise would be false."
)
_COURSE_FACTS_RULE = (
    "COURSE FACTS: Every question about courses, fees, timings, batches, "
    "dates or placement must be answered from the search_course_material "
    "tool, and ONLY from what it returns. Never answer them from memory and "
    "never invent a number or a date. If the tool returns nothing relevant, "
    "say plainly that you do not have that information and suggest they "
    "contact Digital Brolly directly — NEVER promise that someone will call "
    "them back; nothing schedules such a call, so the promise would be "
    "false."
)

# Live 2026-08-29: the agent asked "మీకు కోర్సు వ్యవధి చెప్పాలా?", the lead
# said "ఓకే సార్" — accepting the offer — and the model called end_call and
# hung up on her mid-conversation. An affirmation after an offer is consent
# to CONTINUE; the same guard also lives in end_call's own tool description.
_AFFIRMATION_RULE = (
    "OFFERS AND AFFIRMATIONS: When you offer information — for example "
    "'shall I tell you the fees?' — and the lead replies with an "
    "affirmation such as 'ఓకే', 'సరే', 'అవును', 'ఊ' or 'okay', they are "
    "ACCEPTING the offer: give that information in your next turn. An "
    "affirmation is never a goodbye."
)

# A phone call, not a chat window. The lead cannot skim, and every extra
# sentence is one more they have to interrupt to get a word in.
_BREVITY_RULE = (
    "LENGTH: Keep every reply short — one or two sentences, the way people "
    "actually speak on the phone. Ask one question at a time and then stop "
    "talking and listen. Never deliver a paragraph."
)


def _addressing(lead_name: str | None) -> str:
    """How the model refers to the lead.

    Live 2026-09-01, the team lead's test call: the prompt carried the name
    in English letters, which the synthesiser cannot say, so the model
    avoided it and produced a bare "అవును గారు" / "నమస్తే గారు" — polite, but
    faceless, and the first thing a listener noticed. The bridge now passes
    the name in Telugu script (the same one the greeting speaks), and the
    rule asks for it by name: "మౌలి గారు", "గాయత్రీ గారు".

    The mid-call greeting ban comes from the same call: the lead said "హలో"
    to check the line and the agent started over with "నమస్తే గారు".
    """
    name = (lead_name or "").strip()
    if not name:
        return (
            "You do not know the lead's name. Address them as గారు when you "
            "need to, never as సార్ or మేడమ్. The greeting has already been "
            "said; never greet again with నమస్తే or హలో mid-call, even if "
            "they say హలో to check you are there — just carry on. "
        )
    return (
        f"You are speaking with {name}. ADDRESSING: call them \"{name} గారు\" "
        "— when you acknowledge what they said and when you ask them "
        "something. Never a bare \"గారు\", \"సార్\" or \"మేడమ్\". Use the "
        "name naturally, not in every sentence. The greeting has already been "
        "said; never greet again with నమస్తే or హలో mid-call, even if they "
        "say హలో to check you are there — just carry on. "
    )


def two_way_instructions(lead_name: str | None, script: str | None,
                         language_style: str | None = None,
                         course_facts: str | None = None) -> str:
    """The persona for a call that holds a conversation.

    Differs from one_way_instructions in one structural way: the script is the
    GOAL of the call rather than a message to deliver. A model told to
    "deliver" a script in a conversation will talk over the lead's questions to
    finish it, which is precisely the experience two-way exists to avoid.

    Everything else is deliberately identical. Nothing about a conversation
    makes the disclosure, the anti-Hindi rule or the verbatim-output rule less
    true — they are only easier to forget when writing a second prompt.
    """
    who = _addressing(lead_name)
    mixed = languages.is_code_mixed(language_style)
    parts = [
        "You are a voice assistant on an outbound phone call for Digital "
        "Brolly, an education company in Hyderabad. "
        f"{who}"
        "You called them, so you speak first. Have a natural conversation: "
        "listen to what they say, answer it, and stop.",
        _DISCLOSURE_RULE,
        _language_rule(language_style),
        _output_only_rule(mixed),
        _spelling_rule(mixed),
        _BREVITY_RULE,
        (_COURSE_FACTS_RULE_WITH_DIGEST
         if course_facts and course_facts.strip() else _COURSE_FACTS_RULE),
        _STAY_ON_TOPIC_RULE,
        _unclear_input_rule(mixed),
        # Observed on the live API: told to "say a short goodbye and call the
        # end_call tool", the model said the goodbye and did not call the tool.
        # Nothing then ends the call, so it runs to CALL_MAX_DURATION_S — five
        # minutes of billed airtime and silence at a lead who has already left.
        # Saying goodbye and ending the call have to be ONE instruction, not a
        # sentence containing both.
        "ENDING THE CALL: Saying goodbye and calling the end_call tool are the "
        "SAME action — never do one without the other. The moment the "
        "conversation is EXPLICITLY finished (they say goodbye, ask not to be "
        "called again, or clearly say they have nothing further to ask), your "
        "reply must be a short farewell AND a call to end_call in that same "
        "turn. A goodbye without end_call leaves the lead holding a silent "
        "line — and an 'ఓకే' or 'సరే' after you offered information is "
        "acceptance, never an ending.",
        _AFFIRMATION_RULE,
    ]
    if course_facts and course_facts.strip():
        parts.append(
            "KNOWN COURSE FACTS — read from Digital Brolly's course "
            "documents before this call; answer from these directly: "
            + course_facts.strip()
        )
    if script and script.strip():
        register = ("your own natural spoken Tinglish" if mixed
                    else "your own spoken Telugu")
        parts.append(
            "GOAL OF THIS CALL — this is what you are trying to achieve, not a "
            "speech to read out. Work it into the conversation naturally, in "
            f"{register}, and never at the cost of ignoring what "
            f"the lead just said: {script.strip()}"
        )
    return "\n\n".join(parts)
