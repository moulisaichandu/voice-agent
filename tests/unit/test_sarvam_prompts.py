"""Unit tests for telephony/sarvam_prompts.py — pure strings, no socket.

The rules under test here are not stylistic. Each one is a live-call failure
the sibling ../ai-voice-agent already paid for, carried forward from
app/telephony/openai_prompts.py, plus one that is new to this backend: Sarvam's
LLM writes text that a separate TTS then reads out VERBATIM, so anything the
model says that isn't the message — a preamble, a translation note, quotation
marks — is spoken aloud to a real person.
"""

import pytest

from app import languages
from app.compliance.disclosure import has_ai_disclosure
from app.telephony import sarvam_prompts

_SCRIPT = "This is an automated AI call. Our new Python course starts Monday."


def test_the_script_is_given_as_meaning_not_words_to_recite():
    """The sibling shipped 'say this' and the model read English script text
    aloud to Telugu speakers. Both labels must say the script is meaning to
    convey."""
    instructions = sarvam_prompts.render_instructions(_SCRIPT, language_style=None)
    assert _SCRIPT in instructions
    lowered = instructions.lower()
    assert "meaning" in lowered
    assert "never read english text aloud" in lowered or "not words to recite" in lowered


def test_hindi_is_forbidden():
    """Without this the model drifted into Hindi mid-call, which a
    Telugu-speaking lead in Hyderabad experiences as being called by a stranger
    who does not know them."""
    instructions = sarvam_prompts.render_instructions(_SCRIPT, language_style=None)
    assert "Hindi" in instructions


def test_the_disclosure_must_be_the_first_sentence():
    instructions = sarvam_prompts.render_instructions(_SCRIPT, language_style=None)
    assert "FIRST SENTENCE" in instructions


def test_the_model_is_told_to_emit_only_the_spoken_words():
    """NEW TO THIS BACKEND, and the reason it needs its own prompt module.

    OpenAI Realtime speaks its own output, so a stray 'Sure, here you go:' is
    just conversational. Here the output string goes straight into Sarvam's TTS
    and is read out verbatim — a preamble, a note about the translation, or a
    pair of quotation marks all become sounds a real lead hears."""
    instructions = sarvam_prompts.render_instructions(_SCRIPT, language_style=None)
    lowered = instructions.lower()
    assert "only" in lowered
    assert "spoken aloud" in lowered or "read aloud" in lowered
    assert "quotation marks" in lowered or "quotes" in lowered


def test_the_language_style_reaches_the_prompt():
    """te and tinglish are different registers. Before openai_prompts had this,
    both got one hardcoded rule that permitted English mixing, so pure Telugu
    wrongly invited English."""
    style = "Speak only in Telugu (తెలుగు) throughout."
    assert style in sarvam_prompts.render_instructions(_SCRIPT, language_style=style)


def test_known_sarvam_telugu_skeleton_words_are_repaired_before_tts():
    malformed = (
        "నమసత Mouli గర. ఇద డజటల బరల నడ ఆటమటడ AI కల. "
        "కరస ఫజ సటరకచర ₹1,50,000 అదల టరనగ/డపలమ."
    )
    repaired = sarvam_prompts.normalize_spoken_telugu(malformed)
    assert "నమస్తే Mouli గారు" in repaired
    assert "ఇది డిజిటల్ బ్రోలీ నుండి ఆటోమేటెడ్ AI కాల్" in repaired
    assert "కోర్సు ఫీజు స్ట్రక్చర్ ఒక లక్షా యాభై వేల రూపాయలు అదనపు ట్రైనింగ్/డిప్లొమా" in repaired


def test_rupee_amounts_are_spoken_as_telugu_words():
    repaired = sarvam_prompts.normalize_spoken_telugu(
        "ఫీజు ₹1,50,000. అదనపు ఫీజు ₹50,000."
    )
    assert "ఒక లక్షా యాభై వేల రూపాయలు" in repaired
    assert "యాభై వేల రూపాయలు" in repaired
    assert "ఇరవై ఐదు వేల రూపాయలు" in sarvam_prompts.normalize_spoken_telugu(
        "ఫీజు ₹25,000."
    )
    assert "₹" not in repaired


def test_known_stt_course_words_are_repaired_before_the_llm():
    heard = sarvam_prompts.normalize_heard_telugu(
        "ఓక కరస డటయలస కరస ఎనన డస ఉటద డయరషన"
    )
    assert heard == "ఓకే కోర్సు డీటెయిల్స్ కోర్సు ఎన్ని రోజులు ఉంటుంది డ్యూరేషన్"


def test_the_course_transcript_vocabulary_is_normalized_as_spoken_telugu():
    malformed = (
        "డజటల బరల డజటల మరకటగ కరసల SEO, గగల అడస, మట అడస, "
        "సషల మడయ మరకటగ, కటట రటగ, అనలటకస, యటయబ మరకటగ."
    )
    repaired = sarvam_prompts.normalize_spoken_telugu(malformed)
    assert "డిజిటల్ బ్రోలీ డిజిటల్ మార్కెటింగ్ కోర్సుల" in repaired
    # ఎస్ఈఓ, not Latin "SEO": the spoken path pins one orthography so TTS
    # reads the acronym the same way in every turn of a call.
    assert "ఎస్ఈఓ, గూగుల్ యాడ్స్, మెటా యాడ్స్" in repaired
    assert "సోషల్ మీడియా మార్కెటింగ్, కంటెంట్ రైటింగ్, అనలిటిక్స్" in repaired


# ── the greeting that carries the lead's name ────────────────────────────────
#
# The rendered body is cached per campaign, so the lead's name cannot be inside
# it. It is spliced in front instead, from a template — an Indian name needs no
# translation, so this costs nothing and no LLM call.

def test_a_named_greeting_stays_out_of_the_way_of_the_disclosure():
    """THE constraint on this template. has_ai_disclosure() looks at the first
    sentence, and at the second as well only when the first is a BARE greeting
    — short, and naming no caller. A greeting that grew a brand name or an
    extra clause would push the disclosure out of the checked opening and make
    every Telugu campaign fail its own compliance check."""
    body = "ఇది కృత్రిమ మేధ ద్వారా చేసే ఆటోమేటెడ్ కాల్. కోర్సు సోమవారం మొదలవుతుంది."
    spoken = sarvam_prompts.compose_spoken(body, lead_name="Asha")

    assert spoken.startswith(sarvam_prompts.greeting_line("Asha"))
    assert body in spoken
    assert has_ai_disclosure(spoken), (
        "the greeting pushed the disclosure out of the checked opening"
    )


def test_a_multi_word_name_does_not_push_the_disclosure_out():
    """disclosure.py's bare-greeting roll-forward tolerates at most 3 words.
    Greeting with the WHOLE lead name made "నమస్తే రామ కృష్ణ గారు" 4 words for
    any 'First Last' name, so has_ai_disclosure failed on a compliant opening
    — the lead then heard the disclosure TWICE (ensure_disclosure prepends a
    second one) and a spurious [compliance] WARNING fired on every such call,
    burying the signal CLAUDE.md treats as stop-dialling."""
    body = "ఇది కృత్రిమ మేధ ద్వారా చేసే ఆటోమేటెడ్ కాల్. కోర్సు సోమవారం మొదలవుతుంది."
    spoken = sarvam_prompts.compose_spoken(body, lead_name="రామ కృష్ణ")

    assert has_ai_disclosure(spoken), (
        "the multi-word greeting pushed the disclosure out of the checked opening"
    )
    assert "రామ" in spoken, "the lead should still be greeted by name"


def test_an_initialed_name_does_not_split_the_greeting_sentence():
    """A name like "B. మౌళి" carries a dot, which ends the 'sentence' early
    and breaks the bare-greeting roll-forward a different way."""
    body = "ఇది కృత్రిమ మేధ ద్వారా చేసే ఆటోమేటెడ్ కాల్."
    spoken = sarvam_prompts.compose_spoken(body, lead_name="B. మౌళి")

    assert has_ai_disclosure(spoken)
    assert "మౌళి" in spoken


def test_a_glued_dotted_name_does_not_split_the_greeting_sentence():
    """Found by the post-fix adversarial sweep: "B.మౌళి" typed WITHOUT the
    space is one whitespace token, so the first-token rule kept the internal
    dot — which still ends the greeting 'sentence' early and revives the
    double disclosure + spurious [compliance] WARNING the fix exists to
    prevent. Dots split names exactly like whitespace does."""
    body = "ఇది కృత్రిమ మేధ ద్వారా చేసే ఆటోమేటెడ్ కాల్."
    spoken = sarvam_prompts.compose_spoken(body, lead_name="B.మౌళి")

    assert has_ai_disclosure(spoken)
    assert "మౌళి" in spoken


def test_a_name_of_only_initials_falls_back_to_no_greeting():
    """Better no greeting than an opening that fails its own compliance check."""
    body = "ఇది కృత్రిమ మేధ ద్వారా చేసే ఆటోమేటెడ్ కాల్."

    assert sarvam_prompts.compose_spoken(body, lead_name="B. K.") == body


def test_no_name_means_no_greeting():
    """Most uploaded leads have no name. An empty greeting must not leave a
    stray space or an address to nobody in front of the message."""
    body = "ఇది కృత్రిమ మేధ ద్వారా చేసే ఆటోమేటెడ్ కాల్."
    assert sarvam_prompts.compose_spoken(body, lead_name=None) == body
    assert sarvam_prompts.compose_spoken(body, lead_name="  ") == body


def test_the_greeting_names_no_caller():
    """_is_bare_greeting() stops being true the moment a greeting names the
    company, however short it is."""
    assert "Digital Brolly" not in sarvam_prompts.greeting_line("Asha")
    assert "Asha" in sarvam_prompts.greeting_line("Asha")


# ── the two-way persona ──────────────────────────────────────────────────────

def test_two_way_keeps_every_rule_the_one_way_persona_has():
    """Nothing about a conversation makes the disclosure, the language rule or
    the verbatim-output rule less true. They are only easier to forget when
    writing a second prompt."""
    instructions = sarvam_prompts.two_way_instructions(
        "Asha", _SCRIPT, language_style=None)
    assert "FIRST SENTENCE" in instructions
    assert "Hindi" in instructions
    lowered = instructions.lower()
    assert "quotation marks" in lowered or "quotes" in lowered


def test_two_way_frames_the_script_as_a_goal_not_a_speech():
    """The difference from one-way. In a conversation the script is what the
    call is FOR, not a monologue to get through — a model told to 'deliver'
    it will talk over the lead's questions to finish it."""
    instructions = sarvam_prompts.two_way_instructions(
        None, _SCRIPT, language_style=None)
    assert _SCRIPT in instructions
    assert "GOAL" in instructions


def test_course_facts_must_come_from_the_search_tool():
    """CLAUDE.md: never invent course facts, prices or dates. The tool is the
    only source, and a model that answers fees from memory sounds exactly as
    confident as one that looked them up."""
    instructions = sarvam_prompts.two_way_instructions(
        None, _SCRIPT, language_style=None)
    assert "search_course_material" in instructions
    lowered = instructions.lower()
    assert "never" in lowered and ("invent" in lowered or "memory" in lowered)


def test_the_agent_is_confined_to_the_course():
    """Earned on a live call on the other backend: asked who Virat Kohli was,
    the model simply answered. Nothing had scoped the conversation, so it fell
    back to its training with no rule to stop it."""
    instructions = sarvam_prompts.two_way_instructions(
        None, _SCRIPT, language_style=None)
    assert "STAY ON TOPIC" in instructions


def test_replies_are_told_to_be_short():
    """A paragraph is fine on a screen and unbearable on a phone — the lead
    cannot skim it, and every extra sentence is another one they have to
    interrupt to get a word in."""
    lowered = sarvam_prompts.two_way_instructions(
        None, _SCRIPT, language_style=None).lower()
    assert "short" in lowered or "brief" in lowered


def test_the_lead_name_reaches_the_two_way_prompt():
    assert "Asha" in sarvam_prompts.two_way_instructions(
        "Asha", _SCRIPT, language_style=None)


# ── the STT repair map must not put words in the lead's mouth ───────────────
#
# CLAUDE.md's rule for term_repair governs any rewrite of a lead's speech:
# "Adding a term is a licence to put words in a lead's mouth — replay TERMS
# over the stored transcripts before adding one, and keep the false-positive
# count at zero." normalize_heard_telugu runs BEFORE term_repair and its output
# feeds the LLM history, the RAG query, and the stored transcript, so the same
# standard applies. These entries collide with ordinary Telugu words.

def test_ordinary_telugu_words_are_not_rewritten():
    cases = {
        "అవును కల వచ్చింది": "కాల్",   # కల = "dream", must not become "call"
        "నల రంగు": "నెల",              # నల- = "black", must not become "month"
    }
    for text, must_not_appear in cases.items():
        out = sarvam_prompts.normalize_heard_telugu(text)
        assert must_not_appear not in out, (
            f"{text!r} was rewritten to {out!r} — that is the lead's own words"
        )


def test_the_brand_is_left_to_term_repair_and_its_anchor_rule():
    """term_repair only trusts a weak brand match when 'డిజిటల్' precedes it,
    because four words sit within edit distance 2 of the brand and only two of
    them ARE the brand. An unconditional map entry defeats that entirely."""
    out = sarvam_prompts.normalize_heard_telugu("బరల")

    assert "బ్రోలీ" not in out, (
        "the brand was substituted with no anchor, bypassing term_repair's rule"
    )


def test_genuine_domain_repairs_still_happen():
    """The map exists for a reason — vowel-dropped course words must still be
    repaired, or this fix has simply disabled the feature."""
    out = sarvam_prompts.normalize_heard_telugu("ఇటరనషప పలసమట")

    assert "ఇంటర్న్‌షిప్" in out and "ప్లేస్‌మెంట్" in out


# ── repairs must not flatten the text they repair ──────────────────────────

def test_line_structure_survives_repair():
    """_repair_tokens joined on ' ', so paragraph and list structure was
    destroyed on the composed script, on every say() payload, and on the
    cached render — before TTS and before it was stored."""
    text = "మొదటి పంక్తి.\n\nరెండవ పంక్తి.\n- ఒకటి\n- రెండు"

    out = sarvam_prompts.normalize_spoken_telugu(text)

    assert "\n" in out, "all line breaks were flattened into spaces"
    assert out.count("\n") >= 3


# ── rupee conversion must not kill the turn ────────────────────────────────

def test_a_huge_amount_does_not_raise():
    """_under_100(crore) was called with crore unbounded, so ₹100 crore and up
    raised KeyError — which escapes re.sub, normalize_spoken_telugu and say(),
    ending the turn. A hallucinated or mistyped figure is enough."""
    out = sarvam_prompts.normalize_spoken_telugu("ఫీజు ₹1,50,00,00,000 మాత్రమే.")

    assert out, "conversion raised instead of degrading"


def test_the_comma_after_an_amount_survives():
    """_RUPEE_RE was greedy over ',', so it swallowed the sentence comma after
    the number and with it the TTS pause."""
    out = sarvam_prompts.normalize_spoken_telugu("₹50,000, తరువాత ₹25,000.")

    assert "," in out, "the sentence comma was eaten by the amount pattern"


# ── phrase repairs must not fire inside longer words ───────────────────────
#
# _MANGLED_PHRASES was applied with bare str.replace(), so a key that is a
# prefix of the actual text spliced its replacement mid-word and left orphaned
# trailing characters — including dangling dependent vowel signs, which Sarvam
# TTS reads aloud. Observed: "పలసమట ఫీజు" (mangled 'placement' + CORRECT
# 'ఫీజు') contained the key "పలసమట ఫ", yielding "ప్లేస్‌మెంట్ ఫీజుీజు" —
# 'feejuīju', a non-word, in the sentence quoting the price.

def test_a_phrase_key_never_splices_inside_a_correct_word():
    out = sarvam_prompts.normalize_spoken_telugu("పలసమట ఫీజు 45000 రూపాయలు")

    assert "ఫీజుీజు" not in out, "phrase key spliced mid-word, leaving a dangling vowel sign"
    assert out == "ప్లేస్‌మెంట్ ఫీజు 45000 రూపాయలు"


def test_a_heard_phrase_key_never_leaves_a_dangling_vowel_sign():
    """On the heard path the same unanchored replace corrupted a lead's
    QUESTION "...ఉటదా" via the statement-shaped key "...ఉటద", producing
    "ఉంటుందిా" — a dangling ా — in the LLM history, RAG query and stored
    transcript."""
    out = sarvam_prompts.normalize_heard_telugu("పలసమట సపరట కడ ఉటదా")

    assert "ఉంటుందిా" not in out, "phrase key consumed all but the interrogative -ా"
    assert out.endswith("ఉటదా"), "the ambiguous tail must pass through untouched"


def test_a_properly_bounded_phrase_still_fires():
    """The anchoring must not disable the map — a key followed by punctuation
    or end-of-string is a real match and must still be repaired."""
    out = sarvam_prompts.normalize_spoken_telugu("పలసమట సపరట కడ ఉటద.")

    assert "ప్లేస్‌మెంట్ సపోర్ట్ కూడా ఉంటుంది." in out


def test_the_longest_matching_phrase_wins():
    """"కరస ఎనన డస ఉటద" and its suffix "ఎనన డస ఉటద" are both keys; dict
    iteration order let the shorter fire first and the longer never match."""
    out = sarvam_prompts.normalize_spoken_telugu("కరస ఎనన డస ఉటద")

    assert out == "కోర్సు ఎన్ని రోజులు ఉంటుంది"


# ── ambiguous skeletons must not flip person or mood in the lead's words ───
#
# Dropping dependent vowel signs erases exactly the difference between మీకు
# ('to you') and మాకు ('to us'), and between the statement ఉంటుంది and the
# question ఉంటుందా — so a heard-map entry for the skeleton rewrites the
# lead's meaning. Same class as కల ('dream'→'call') / నల ('black'→'month').

def test_the_person_of_the_leads_words_is_never_flipped():
    out = sarvam_prompts.normalize_heard_telugu("మక వర కరస ఉద")

    assert "మీకు" not in out, "మాకు ('to us') was rewritten to మీకు ('to you')"


def test_a_leads_question_is_never_rewritten_into_a_statement():
    out = sarvam_prompts.normalize_heard_telugu("ఇటరనషప ఉటద")

    assert "ఉంటుంది" not in out, (
        "the lead may have asked ఉంటుందా — resolving the skeleton to the "
        "statement form puts an assertion in their mouth"
    )
    out2 = sarvam_prompts.normalize_heard_telugu("కరసల ఉననయ")
    assert "ఉన్నాయి" not in out2 and "ఉన్నాయా" not in out2


def test_a_leads_statement_is_never_rewritten_into_a_question():
    out = sarvam_prompts.normalize_heard_telugu("చపతర")

    assert "చెప్తారా" not in out and "చెప్తారు" not in out


def test_agent_text_still_gets_the_ambiguous_repairs():
    """The exclusions above are for LEAD speech only. The agent's own text is
    the model's words, not evidence — the spoken path keeps the repairs."""
    out = sarvam_prompts.normalize_spoken_telugu("మక ఉటద")

    assert "మీకు" in out and "ఉంటుంది" in out


# ── generic rupee amounts must be grammatical Telugu ───────────────────────
#
# The generic speller had singular forms for thousand (వెయ్యి) and hundred
# (వంద) but not lakh/crore, and always emitted the nominative plural
# లక్షలు/కోట్లు where the oblique లక్షల/కోట్ల (or the connective లక్షా) is
# required before a following word — so any fee outside the six-entry common
# map came out machine-broken in the one sentence quoting the price.

def test_one_lakh_plus_change_uses_the_connective_form():
    out = sarvam_prompts.normalize_spoken_telugu("ఫీజు ₹1,10,000")

    assert "ఒక లక్షా పది వేల రూపాయలు" in out
    assert "ఒకటి లక్షలు" not in out


def test_a_round_multi_lakh_amount_uses_the_oblique_plural():
    out = sarvam_prompts.normalize_spoken_telugu("ఫీజు ₹2,00,000")

    assert "రెండు లక్షల రూపాయలు" in out
    assert "లక్షలు" not in out


def test_crore_amounts_follow_the_same_grammar():
    assert "ఒక కోటి పది లక్షల రూపాయలు" in sarvam_prompts.normalize_spoken_telugu(
        "₹1,10,00,000")
    assert "రెండు కోట్ల రూపాయలు" in sarvam_prompts.normalize_spoken_telugu(
        "₹2,00,00,000")


# ── loanword orthography the conversation LLM gets wrong ───────────────────
#
# Heard on the 2026-08-27 test call: the conversation model, answering from
# RAG text that says "Google Ads", wrote "గూగుల్ అడ్స్" — fully voweled, so
# the skeleton map never fires — and Sarvam TTS read it as "aḍs". The
# campaign renders spell it "యాడ్స్" (yāḍs), so the same call can contain
# both spellings of the same product name.

def test_the_ads_loanword_is_normalized_to_its_spoken_form():
    out = sarvam_prompts.normalize_spoken_telugu("SEO, గూగుల్ అడ్స్, మెటా అడ్స్, సోషల్ మీడియా")

    assert "గూగుల్ యాడ్స్" in out and "మెటా యాడ్స్" in out
    assert "అడ్స్" not in out


def test_a_case_suffixed_loanword_is_still_repaired():
    """The conversation model welds case endings on with ZWNJ as a habit
    (ట్రైనింగ్‌లో, మార్కెటింగ్‌లో — its own live output), so "అడ్స్‌లో"
    passed through the whole-token map unrepaired and one call could speak
    the same product name two ways. ZWNJ is exactly the loanword-suffix
    seam, so a token that misses the map is retried on its pre-ZWNJ head."""
    out = sarvam_prompts.normalize_spoken_telugu("గూగుల్ అడ్స్‌లో నేర్చుకుంటారు")

    assert "యాడ్స్‌లో" in out


def test_seo_is_spoken_in_one_orthography():
    """Today's live calls spoke Latin "SEO" in one turn and "ఎస్ఈఓ" in the
    next — Sarvam TTS is not guaranteed to read the Latin acronym the way it
    reads the Telugu spelling, so the flagship module's name could sound
    different across turns of one call."""
    out = sarvam_prompts.normalize_spoken_telugu(
        "SEO, గూగుల్ యాడ్స్ నేర్పిస్తాం. SEO చాలా ముఖ్యం.")

    assert "SEO" not in out
    assert "ఎస్ఈఓ" in out


def test_seo_stays_english_in_rag_bound_heard_text():
    """Heard text becomes the RAG query, and the English-only corpus embeds
    English far better (Telugu-script queries measured 0.10-0.19 against it
    versus 0.43+ for English) — so the heard path must NOT convert SEO."""
    assert "SEO" in sarvam_prompts.normalize_heard_telugu("SEO గురించి చెప్పండి")


def test_details_is_spelled_one_way_everywhere():
    """The maps themselves disagreed: the token map wrote డిటైల్స్ while the
    phrase map wrote డీటెయిల్స్, so which spelling the lead heard depended on
    whether కోర్సు happened to precede the word."""
    alone = sarvam_prompts.normalize_spoken_telugu("డటయలస")
    phrased = sarvam_prompts.normalize_spoken_telugu("కరస డటయలస")

    assert alone == "డీటెయిల్స్"
    assert "డీటెయిల్స్" in phrased
    assert "డిటైల్స్" not in alone + phrased


# ── token repair must be idempotent ────────────────────────────────────────
#
# A token made only of quote characters was claimed by BOTH the leading and
# trailing strip slices and emitted twice — and sarvam_llm's cache-repair loop
# re-normalizes every cache hit and re-puts when the text changed, so the
# doubled quote doubled again on every later call to the campaign: 2^N growth
# in the cached script and the TTS payload.

def test_a_lone_quote_token_is_not_doubled():
    out = sarvam_prompts.normalize_spoken_telugu('అతను " అన్నారు')

    assert '""' not in out
    assert out == 'అతను " అన్నారు'


def test_normalize_spoken_telugu_is_idempotent():
    """The property the render cache actually depends on: repairing repaired
    text must change nothing, or the cache-repair loop amplifies forever."""
    for text in ('అతను " అన్నారు', "'", 'నమసత "గర" ఇద', "ఫీజు ₹45,000."):
        once = sarvam_prompts.normalize_spoken_telugu(text)
        twice = sarvam_prompts.normalize_spoken_telugu(once)
        assert twice == once, f"not idempotent for {text!r}: {once!r} -> {twice!r}"


# ── phrase repairs must not splice themselves inside a longer word ─────────
#
# _MANGLED_PHRASES was applied with a bare text.replace(), so a key that is a
# PREFIX of the real text spliced its replacement mid-word and left the
# remaining characters dangling — including orphaned dependent vowel signs,
# which Sarvam TTS then reads aloud to a live lead in the sentence quoting the
# price.

def test_a_phrase_key_does_not_splice_inside_a_longer_word():
    out = sarvam_prompts.normalize_spoken_telugu("పలసమట ఫీజు 45000 రూపాయలు")

    assert "ఫీజుీజు" not in out, f"spliced mid-word: {out!r}"
    assert "ప్లేస్‌మెంట్" in out and "ఫీజు" in out, out


def test_a_heard_phrase_key_does_not_leave_a_dangling_vowel_sign():
    out = sarvam_prompts.normalize_heard_telugu("పలసమట సపరట కడ ఉటదా")

    assert "ఉంటుందిా" not in out, f"dangling vowel sign: {out!r}"


# ── the lead's person and mood must survive repair ────────────────────────
#
# మక -> మీకు turns "WE have" into "YOU have"; ఉటద -> ఉంటుంది turns a question
# into an assertion. Both land in the LLM history, the RAG query and the stored
# transcript as things the lead never said.

def test_first_person_is_not_flipped_to_second_person():
    out = sarvam_prompts.normalize_heard_telugu("మక వర కరస ఉద")

    assert "మీకు" not in out, f"'we' became 'you': {out!r}"


def test_a_question_is_not_rewritten_into_an_assertion():
    out = sarvam_prompts.normalize_heard_telugu("ఇటరనషప ఉటద")

    assert "ఉంటుంది" not in out, f"question became a statement: {out!r}"


def test_unambiguous_domain_repairs_still_work():
    """The map must keep earning its place — these have no ordinary-word twin."""
    out = sarvam_prompts.normalize_heard_telugu("ఇటరనషప పలసమట డయరషన")

    assert "ఇంటర్న్‌షిప్" in out and "ప్లేస్‌మెంట్" in out and "డ్యూరేషన్" in out


# ── the default two-way opening needs no model ─────────────────────────────
#
# A two-way campaign normally has NO script, so the bridge falls back to
# DEFAULT_TWOWAY_SCRIPT — a fixed English constant of ours. Translating a
# constant is a constant, and on 2026-08-27 paying a model for it cost two
# live calls: Sarvam's reasoning outgrew its token budget, the render
# returned nothing, and both leads answered to silence and were hung up on.
# The Telugu below is not invented here — it is what the model itself
# produced for this script on the calls of 2026-08-27, which passed the
# disclosure check and were spoken to real leads.

def test_the_default_two_way_script_has_a_pre_translated_telugu_opening():
    from app.compliance.disclosure import has_ai_disclosure

    telugu = sarvam_prompts.DEFAULT_TWOWAY_TELUGU

    assert has_ai_disclosure(telugu), "the canned opening must disclose the AI"
    assert "డిజిటల్ బ్రోలీ" in telugu
    # It is spoken verbatim, so it must already survive the spoken repairs.
    assert sarvam_prompts.normalize_spoken_telugu(telugu) == telugu


def test_the_canned_opening_survives_the_greeting_the_bridge_splices_on():
    """compose_spoken prepends the lead's name; the disclosure has to stay
    inside the window has_ai_disclosure checks."""
    from app.compliance.disclosure import has_ai_disclosure

    spoken = sarvam_prompts.compose_spoken(
        sarvam_prompts.DEFAULT_TWOWAY_TELUGU, lead_name="రామ కృష్ణ")

    assert has_ai_disclosure(spoken)


# ── the documents are read BEFORE the call, not during it ───────────────────
#
# Owner's direction after the 2026-08-29 test round: "the documents I gave
# should be read and stored in advance" — every common question was paying a
# tool round-trip and a spoken "ఒక్క క్షణం, చూసి చెప్తాను", and the fee
# question STILL missed (top-k crowd-out needs no conjunction). A digest of
# the corpus's core facts now rides in the system prompt; the tool remains
# for the long tail.

def test_course_facts_ride_in_the_prompt_and_are_answered_directly():
    facts = "BDLP ఫీజు ఇరవై ఐదు వేల రూపాయలు. Duration: 3 months."
    out = sarvam_prompts.two_way_instructions("Asha", None, course_facts=facts)

    assert facts in out
    assert "KNOWN COURSE FACTS" in out
    lowered = out.lower()
    assert "no tool call" in lowered, (
        "facts in the prompt must be answered directly, without a lookup"
    )
    assert "search_course_material" in out, "the tool must remain for the long tail"


def test_without_facts_the_tool_stays_the_only_source():
    out = sarvam_prompts.two_way_instructions("Asha", None)

    assert "KNOWN COURSE FACTS" not in out
    assert "search_course_material" in out


def test_the_prompt_never_teaches_a_callback_promise():
    """Heard on live calls twice (2026-08-27 and 2026-08-29): 'మా టీమ్ నుంచి
    ఎవరో తిరిగి కాల్ చేస్తారు' — a promise nothing schedules. The model was
    not inventing it: _COURSE_FACTS_RULE itself said 'offer to have a person
    call back'. The prompt must forbid the promise, not teach it."""
    for facts in (None, "F"):
        out = sarvam_prompts.two_way_instructions(
            "Asha", None, course_facts=facts)
        lowered = out.lower()
        assert "offer to have a person call back" not in lowered
        assert "never promise" in lowered and "call" in lowered


def test_an_affirmation_is_acceptance_not_a_goodbye():
    """Live 2026-08-29: the agent asked 'మీకు కోర్సు వ్యవధి చెప్పాలా?', the
    lead said 'ఓకే సార్' — accepting — and the model called end_call and
    hung up on her mid-conversation."""
    out = sarvam_prompts.two_way_instructions("Asha", None)

    assert "ఓకే" in out and "సరే" in out
    assert "EXPLICITLY" in out or "explicitly" in out


# ── garbled speech is not a goodbye, and not an off-topic remark ────────────
#
# Live calls 2026-08-29, both endings wrong:
#   Mouli: agent asked WHICH fee structure he wanted; he answered "25000" —
#     naming the ₹25,000 course — and the agent replied "I can only help with
#     course questions" and ended the call.
#   Gayathri: the lead said "కో." (Telugu STT truncation) and the agent ended.
# Sarvam's Telugu STT garbles heavily on 8 kHz phone audio — the same calls
# produced "వాక్య తింగ ఏమేం కోషమ్స్" and "స్కూట్స్ ప్లాంట్స్" — so unclear
# input is the NORMAL case, not an edge case, and neither hanging up nor a
# scope refusal is an honest response to it.

def test_unclear_speech_is_asked_again_not_refused_or_ended():
    out = sarvam_prompts.two_way_instructions("Asha", None)
    lowered = out.lower()

    assert "UNCLEAR" in out
    assert "say it again" in lowered or "repeat" in lowered
    assert "never end the call" in lowered


def test_a_short_reply_to_your_own_question_is_an_answer():
    """A bare number, a course code, or a fragment right after the agent
    asked a question is the lead ANSWERING — never an off-topic remark."""
    out = sarvam_prompts.two_way_instructions("Asha", None)

    assert "25,000" in out or "25000" in out
    assert "off-topic" in out.lower()


def test_the_opening_is_pronounceable_telugu_with_no_latin_letters():
    """Audited across every call: "AI" was the only Latin token the agent
    ever spoke, and it is in EVERY opening. Sarvam's Telugu TTS reads Latin
    letters as an English word rather than the letter names, so the lead
    hears something that is not "ఏ-ఐ" in the one sentence India's rules
    require them to understand.

    Restructured rather than substituted: the compliance check matched the
    Latin \bai\b token, so swapping in ఏఐ alone made the opening fail
    has_ai_disclosure — the sentence now carries "ఆటోమేటెడ్ కాల్" adjacently
    instead, which is a marker in its own right."""
    import re

    from app.compliance.disclosure import has_ai_disclosure

    opening = sarvam_prompts.DEFAULT_TWOWAY_TELUGU

    assert not re.search(r"[A-Za-z]", opening), (
        f"Latin letters are spoken aloud in the opening: {opening!r}"
    )
    assert has_ai_disclosure(opening), "the AI disclosure was lost"
    assert sarvam_prompts.normalize_spoken_telugu(opening) == opening


# ── nothing but Telugu and English may reach the synthesiser ────────────────
#
# Live call 2026-09-01: the agent said "బ్యాచ్ వివరాలు నాకు ఈ պահին లేవు" —
# "պահին" is ARMENIAN. The model slipped a foreign-script word into an
# otherwise fine Telugu sentence and Sarvam's TTS read it out at a real lead.
# The prompt already forbids this; nothing enforced it.
#
# The same guard covers a documented older failure: CLAUDE.md records the
# model drifting into HINDI mid-call, "which a Telugu-speaking lead in
# Hyderabad experiences as being called by a stranger". Devanagari is caught
# by exactly the same rule.

def test_a_foreign_script_word_never_reaches_the_synthesiser():
    out = sarvam_prompts.normalize_spoken_telugu(
        "బ్యాచ్ వివరాలు నాకు ఈ պահին లేవు.")

    assert "պահին" not in out
    assert "బ్యాచ్ వివరాలు" in out and "లేవు" in out, (
        "the surrounding Telugu must survive; only the foreign word goes"
    )


def test_hindi_drift_is_caught_by_the_same_rule():
    out = sarvam_prompts.normalize_spoken_telugu(
        "కోర్సు फीस पचास हजार रुपये ఉంటుంది.")

    for devanagari in ("फीस", "पचास", "हजार", "रुपये"):
        assert devanagari not in out
    assert "కోర్సు" in out and "ఉంటుంది" in out


def test_telugu_and_english_both_survive():
    """Tinglish is a supported register and course names are English — the
    guard must only remove scripts that are neither."""
    text = "SEO ట్రైనింగ్ లో Google Ads, Meta Ads ఉన్నాయి. ఫీజు 25,000."
    out = sarvam_prompts.normalize_spoken_telugu(text)

    for keep in ("Google", "Ads", "Meta", "ట్రైనింగ్", "ఫీజు", "25,000"):
        assert keep in out, f"{keep!r} was wrongly removed"


def test_the_dropped_words_can_be_reported():
    """say() logs them: a model emitting Armenian is worth an operator's
    attention even though the sentence was rescued."""
    found = sarvam_prompts.foreign_script_tokens("నాకు ఈ պահին लेवу లేవు")

    assert "պահին" in found
    assert any("ल" in t for t in found)
    assert sarvam_prompts.foreign_script_tokens("కోర్సు ఫీజు SEO 25,000") == []


@pytest.mark.parametrize("text,keep", [
    ("మా దగ్గర కోర్సులు ఉన్నాయి।", "ఉన్నాయి"),      # danda — _SENTENCE_END's own terminator
    ("ఇంటర్న్‌షిప్ ఉంటుంది॥", "ఉంటుంది"),
    ("ఫీజు… ఇరవై ఐదు వేలు", "ఫీజు"),                # ellipsis
    ("కోర్సు «వివరాలు» ఇవి", "వివరాలు"),            # guillemets
    ("ఫీజు 50,000/- రూపాయలు", "50,000"),
])
def test_punctuation_never_makes_a_telugu_word_unspeakable(text, keep):
    """The foreign-script guard judged whole tokens, so a Telugu word ending
    in a danda — the very character _SENTENCE_END splits sentences on — was
    classified as foreign and DELETED. Words vanishing mid-sentence is
    exactly the "voice breaks in the middle" a lead reported. Script is a
    property of a word's LETTERS, never of the punctuation stuck to it."""
    assert sarvam_prompts.foreign_script_tokens(text) == []
    assert keep in sarvam_prompts.normalize_spoken_telugu(text)


def test_a_foreign_word_wearing_punctuation_is_still_caught():
    """...and the guard must not be disarmed by the same reasoning."""
    assert sarvam_prompts.foreign_script_tokens("నాకు պահին। లేవు") == ["պահին।"]


# ── Tinglish is a register of its own, not Telugu with a permission slip ─────
#
# languages.py's tinglish style keeps course, fees, batch, timings, online,
# demo, certificate and placement in English. Until 2026-09-01 three later
# rules in this module contradicted it on every call: OUTPUT FORMAT demanded
# "Telugu script", TELUGU SPELLING mandated కోర్సు / ఫీజు / ప్లేస్‌మెంట్, and
# the script framing said "never read English text aloud". The project's own
# design notes had already judged contradictory prompt instructions worse
# than a loose register (.superpowers/sdd/telugu-task-3-report.md), so the
# register and the contradictions are fixed as ONE change. These tests use
# the REAL catalogue text rather than a placeholder — the only way they could
# have caught it, and the reason the earlier placeholder tests never did.

def _two_way(style):
    return sarvam_prompts.two_way_instructions(None, None, language_style=style)


def _render(style):
    return sarvam_prompts.render_instructions("Our course starts Monday.",
                                              language_style=style)


def test_tinglish_instructions_keep_the_everyday_english_words_in_english():
    lowered = _two_way(languages.style("tinglish")).lower()
    for word in ("course", "fees", "batch", "timings", "certificate", "placement"):
        assert word in lowered
    assert "keep these everyday english words" in lowered
    assert "ordinary english letters" in lowered, (
        "the OUTPUT FORMAT rule must say how the kept English is written")


@pytest.mark.parametrize("build", [_two_way, _render])
def test_tinglish_instructions_do_not_contradict_themselves(build):
    prompt = build(languages.style("tinglish"))
    lowered = prompt.lower()
    assert "do not switch to english" not in lowered
    assert "never read english text aloud" not in lowered
    assert "in your own spoken telugu" not in lowered
    # The loanwords the register keeps in English must not be mandated in
    # Telugu spelling in the very same prompt.
    for telugu_spelling in ("కోర్సు", "ఫీజు", "ప్లేస్‌మెంట్"):
        assert telugu_spelling not in prompt, (
            f"{telugu_spelling!r} is mandated in a prompt that keeps that word in English")


@pytest.mark.parametrize("build", [_two_way, _render])
def test_pure_telugu_instructions_still_forbid_english(build):
    """The other half of the same change: te must not have loosened."""
    prompt = build(languages.style("te"))
    lowered = prompt.lower()
    assert "do not switch to english" in lowered
    assert "in telugu script" in lowered
    for telugu_spelling in ("కోర్సు", "ఫీజు", "ప్లేస్‌మెంట్"):
        assert telugu_spelling in prompt


@pytest.mark.parametrize("build", [_two_way, _render])
def test_pure_telugu_and_tinglish_produce_different_instructions(build):
    assert build(languages.style("te")) != build(languages.style("tinglish"))


def test_no_style_at_all_is_pure_telugu_never_tinglish():
    """A caller that bypasses call_routes' resolution must not be handed
    permission to mix in English by accident."""
    assert "do not switch to english" in _two_way(None).lower()


@pytest.mark.parametrize("token", ["te", "tinglish"])
def test_hindi_and_the_disclosure_hold_in_both_registers(token):
    lowered = _two_way(languages.style(token)).lower()
    assert "never use hindi" in lowered
    assert "automated ai call" in lowered


def test_the_tinglish_opening_discloses_and_survives_the_spoken_repairs():
    from app.compliance.disclosure import has_ai_disclosure

    opening = sarvam_prompts.DEFAULT_TWOWAY_TINGLISH

    assert has_ai_disclosure(opening), "the canned opening must disclose the AI"
    assert "డిజిటల్ బ్రోలీ" in opening
    assert sarvam_prompts.normalize_spoken_telugu(opening) == opening
    assert has_ai_disclosure(
        sarvam_prompts.compose_spoken(opening, lead_name="రామ కృష్ణ"))


def test_the_tinglish_opening_speaks_no_latin_ai_and_only_kept_english_words():
    """Latin "AI" is read by Sarvam's TTS as an English word, not the letter
    names — the reason the Telugu opening was restructured around ఏఐ, and it
    applies here unchanged. The English that IS present must be ordinary
    words the register keeps, never letters the synthesiser will misread."""
    import re

    opening = sarvam_prompts.DEFAULT_TWOWAY_TINGLISH

    assert not re.search(r"ai", opening, re.IGNORECASE)
    latin = {w.strip(".,").lower() for w in re.findall(r"[A-Za-z][A-Za-z.,]*", opening)}
    assert latin, "a Tinglish opening with no English in it is just the Telugu one"
    allowed = {"digital", "marketing", "course", "courses", "fees", "batch",
               "timings", "placement", "online", "demo", "certificate"}
    assert latin <= allowed, f"unexpected English in the opening: {latin - allowed}"


def test_the_scriptless_opening_is_chosen_by_register():
    pick = sarvam_prompts.default_twoway_opening
    assert pick(languages.style("tinglish")) == sarvam_prompts.DEFAULT_TWOWAY_TINGLISH
    assert pick(languages.style("te")) == sarvam_prompts.DEFAULT_TWOWAY_TELUGU
    assert pick(None) == sarvam_prompts.DEFAULT_TWOWAY_TELUGU


# ── the lead is a person with a name ─────────────────────────────────────────
#
# Team lead's test call, 2026-09-01 21:07: "అవును గారు", "నమస్తే గారు" — a bare
# honorific every time, because the prompt carried the name in English
# letters the synthesiser cannot say, so the model avoided it. The bridge now
# passes the Telugu-script name and the prompt asks for it by name.

def test_the_lead_is_addressed_by_name_and_garu():
    prompt = sarvam_prompts.two_way_instructions(
        "గాయత్రీ", None, language_style=languages.style("te"))

    assert 'call them "గాయత్రీ గారు"' in prompt
    assert 'never a bare "గారు"' in prompt.lower() or "Never a bare" in prompt
    assert "never greet again" in prompt.lower(), (
        "a mid-call హలో was answered with a fresh నమస్తే on the live call")


def test_a_nameless_lead_is_still_never_sir_or_madam():
    prompt = sarvam_prompts.two_way_instructions(
        None, None, language_style=languages.style("te"))

    assert "You do not know the lead's name" in prompt
    assert "సార్" in prompt and "మేడమ్" in prompt


@pytest.mark.parametrize("token", ["te", "tinglish"])
def test_background_speech_and_garbled_course_questions_are_covered(token):
    """Two things the team lead heard as 'deviating to noise': other people
    near the phone being answered as if they were the lead, and a garbled
    on-topic question refused as off-topic."""
    prompt = sarvam_prompts.two_way_instructions(
        "మౌలి", None, language_style=languages.style(token))

    assert "recorded announcement" in prompt
    assert "never call such a question off-topic" in prompt
