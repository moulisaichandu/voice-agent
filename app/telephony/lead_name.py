"""telephony/lead_name.py — the lead's own name, in a script the voice can read.

Reported after a live call on 2026-08-31: "Mouli గారు" was not pronounced
properly. The greeting is "నమస్తే {name} గారు" and the name arrives from the
console or the sheet in LATIN letters, so Sarvam's Telugu TTS reads it with
English phonetics rather than saying మౌళి. It is the same defect as the Latin
"AI" that was fixed in the opening the same day — in the one word a lead cares
most about hearing right.

Transliteration, NOT translation: a name has no meaning to render, only
sounds to carry across scripts. The model is asked for exactly that and its
answer is refused unless it comes back as Telugu script, because the failure
this guards against is the model explaining itself ("The Telugu for Mouli
is…") and the synthesiser reading that sentence out as somebody's name.

Cached per name, effectively forever: a name's spelling does not change, the
same leads are dialled repeatedly, and the whole point is that no call waits
for this twice. Every failure path returns the ORIGINAL name — a
mispronounced name is a blemish, a crashed or nameless greeting is a broken
call, and the AI disclosure lives in the sentence right after it.
"""

from __future__ import annotations

import asyncio
import logging

from app import redis_client
from app.config import RAG_TRANSLATE_MODEL
from app.rag.embeddings import _get_client

logger = logging.getLogger(__name__)

_CACHE_PREFIX = "lead_name:te:"
_CACHE_TTL_S = 90 * 24 * 3600

# Short and single-purpose. The model is given a NAME and must return a NAME:
# any framing that invites it to be helpful ("explain", "translate") is how a
# greeting turns into a sentence read aloud at a stranger.
_SYSTEM = (
    "You transliterate Indian personal names into Telugu script.\n"
    "Rules:\n"
    "- Reply with ONLY the name in Telugu script. No quotes, no explanation, "
    "no extra words.\n"
    "- Transliterate the SOUND of the name. Never translate its meaning and "
    "never substitute a different name.\n"
    "- Use the spelling a Telugu speaker would recognise for that name.\n"
    "- If the input is already Telugu script, repeat it back unchanged."
)

# This runs before the opening is spoken, so it is bounded hard. On a cache
# miss the caller pays it once per name, ever.
_TIMEOUT_S = 4.0
_MAX_OUTPUT_CHARS = 40

_TELUGU_RANGE = ("ఀ", "౿")


def _is_telugu(text: str) -> bool:
    """Whether *text* is Telugu script and nothing else worth speaking.

    Combining marks and spaces are fine; a Latin letter or a digit is not —
    those are the shapes a refused reply takes (an echo of the input, or a
    sentence about the input).
    """
    letters = [c for c in text if not c.isspace()]
    if not letters:
        return False
    return all(_TELUGU_RANGE[0] <= c <= _TELUGU_RANGE[1] for c in letters)


def _transliterate_sync(name: str) -> str:
    resp = _get_client().chat.completions.create(
        model=RAG_TRANSLATE_MODEL,
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content": name}],
        temperature=0,
        max_tokens=30,
        timeout=_TIMEOUT_S,
    )
    return (resp.choices[0].message.content or "").strip()


async def telugu_name(name: str | None) -> str:
    """*name* in Telugu script, or *name* unchanged if that is not possible.

    Never raises and never returns empty for a non-empty input: the caller
    splices this straight into the spoken greeting.
    """
    if not name or not name.strip():
        return ""
    name = name.strip()
    if _is_telugu(name):
        return name              # already speakable; no model call at all

    key = _CACHE_PREFIX + name.casefold()
    redis = None
    try:
        redis = redis_client.get_redis()
        cached = await redis.get(key)
        if cached:
            return cached.decode() if isinstance(cached, bytes) else str(cached)
    except Exception:  # noqa: BLE001 - the cache is a speed-up, not a source
        redis = None

    try:
        result = await asyncio.to_thread(_transliterate_sync, name)
    except Exception as exc:  # noqa: BLE001 - degrade to the Latin spelling
        logger.warning(
            f"[lead-name] could not transliterate {name!r} "
            f"({type(exc).__name__}: {exc}) — greeting with it as written."
        )
        return name

    result = (result or "").strip()
    if not result or len(result) > _MAX_OUTPUT_CHARS or not _is_telugu(result):
        logger.warning(
            f"[lead-name] refused a transliteration of {name!r} that was not "
            "Telugu script — greeting with the name as written."
        )
        return name

    if redis is not None:
        try:
            await redis.set(key, result, ex=_CACHE_TTL_S)
        except Exception:  # noqa: BLE001 - see above
            pass
    logger.info(f"[lead-name] greeting {name!r} as {result!r}")
    return result
