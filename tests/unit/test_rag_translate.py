"""app/rag/translate.py's _SYSTEM prompt — language coverage regression pin.

Static content checks only: no network call, no OpenAI client construction.
Guards against the prompt silently regressing back to naming only a subset
of the 5 catalogue languages (see app/languages.py's LANGUAGES) — the exact
gap this file exists to close for Hindi/Hinglish.
"""

from app.rag import translate


def test_system_prompt_names_all_five_catalogue_languages():
    prompt = translate._SYSTEM.lower()
    for language in ("english", "hindi", "telugu", "hinglish", "tinglish"):
        assert language in prompt, f"{language!r} is not named in translate._SYSTEM"


def test_system_prompt_still_says_nothing_about_courses_or_fees():
    """Regression guard for the documented failure mode in this module's
    docstring: naming the domain caused the model to turn an unrelated
    question ("how is the weather today?") into an invented course question.
    """
    prompt = translate._SYSTEM.lower()
    for banned in ("course", "fee", "curriculum", "price"):
        assert banned not in prompt, f"{banned!r} must not appear in translate._SYSTEM"
