"""Unit tests for telephony/sarvam_prompts.py — pure strings, no socket.

The rules under test here are not stylistic. Each one is a live-call failure
the sibling ../ai-voice-agent already paid for, carried forward from
app/telephony/openai_prompts.py, plus one that is new to this backend: Sarvam's
LLM writes text that a separate TTS then reads out VERBATIM, so anything the
model says that isn't the message — a preamble, a translation note, quotation
marks — is spoken aloud to a real person.
"""

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
