"""rag/translate.py — Query normalisation to English for retrieval only.

Exists because the course documents are English-only while leads ask in
Telugu and Tinglish. Measured against the real corpus (91 chunks,
text-embedding-3-small): a Telugu-script question scores 0.13-0.19 against
its own answer chunk, where the English equivalent scores 0.37-0.46 — the
Telugu query is statistically indistinguishable from an off-topic English one
(0.128). No threshold setting separates "Telugu and relevant" from "English
and irrelevant", so the query is moved into the corpus's language instead of
the threshold being weakened. See app/config.py's RAG_TRANSLATE_ON_MISS.

This translates the QUERY only, for embedding. Retrieved course text is
returned verbatim to the agent, which answers in whatever language the lead
is speaking — nothing here changes what the lead is ultimately told, so a bad
translation can only cause a retrieval miss (the safe direction), never a
mistranslated course fact.
"""

from __future__ import annotations

import asyncio
import logging

from app.config import RAG_TRANSLATE_MODEL
from app.rag.embeddings import _get_client

logger = logging.getLogger(__name__)

# Deliberately says nothing about courses, fees, or what the translation will
# be used for. An earlier version framed this as "translate so it can be
# matched against course documents", and the model duly turned EVERY input
# into a plausible course question — "ఈరోజు వాతావరణం ఎలా ఉంది?" (how is the
# weather today) came back as "What is the course duration?", which then
# retrieved course material for a question that was never about the course.
# Any domain hint here silently becomes a licence to invent, so there is none.
_SYSTEM = (
    "You are a literal translator. Translate the user's message into English.\n"
    "The input may be Telugu, English, or Telugu written in Latin script "
    "('Tinglish').\n"
    "Rules:\n"
    "- Translate faithfully and literally. Preserve the exact subject matter.\n"
    "- Do NOT infer intent, add context, answer the question, or make the "
    "message about any particular topic.\n"
    "- If the message is unrelated to education or courses, translate it "
    "unchanged anyway — an unrelated question must stay unrelated.\n"
    "- If it is already entirely English, repeat it back verbatim.\n"
    "- Reply with ONLY the translation: no quotes, notes, or explanation."
)

# Retrieval quality degrades far less from a slightly-off translation than the
# call degrades from a long pause, and this runs while the lead is waiting.
_TIMEOUT_S = 6.0
_MAX_OUTPUT_CHARS = 512


def _translate_sync(query: str) -> str:
    resp = _get_client().chat.completions.create(
        model=RAG_TRANSLATE_MODEL,
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content": query}],
        temperature=0,
        max_tokens=120,
        timeout=_TIMEOUT_S,
    )
    return (resp.choices[0].message.content or "").strip()


async def translate_to_english(query: str) -> str | None:
    """Best-effort English rendering of *query*, or None if unavailable.

    None (not an exception, and not the original string) on every failure
    path, so the caller can cleanly distinguish "translation gave us nothing
    new to try" from "here is a different query worth retrying". Runs in a
    thread: the OpenAI client is synchronous, and this is called from the
    request path of a server also driving live calls — blocking the event
    loop here would stall every other in-flight call's tool lookups too.
    """
    try:
        translated = await asyncio.to_thread(_translate_sync, query)
    except Exception as exc:
        logger.warning(
            f"[rag] query translation unavailable, keeping the original-query "
            f"result: {type(exc).__name__}: {exc}"
        )
        return None

    if not translated or len(translated) > _MAX_OUTPUT_CHARS:
        return None
    # Nothing gained by re-embedding an identical string — skip the retry.
    if translated.strip().lower() == query.strip().lower():
        return None
    return translated
