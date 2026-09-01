"""rag/search.py — Retrieval, with the relevance-threshold safety pattern.

Two functions, two different contracts:

  search_relevant() — filters by min_score, returns "" / a "no relevant
    material" note when nothing clears the threshold. This is the ONLY
    function app/rag/endpoint.py's live /rag/search tool may call.

  search_permissive() — returns the raw top-k regardless of score. Debug/
    admin use ONLY. Never wire this into the live call path — that reintroduces
    the exact "reads out irrelevant course material" bug the blueprint's own
    "Mistakes to Avoid" section names from the sibling project's history: the
    agent, told to answer only from retrieved text, will confidently recite
    unrelated course content in response to a greeting or off-topic remark.

test_rag_endpoint_uses_filtered_search (tests/unit/test_rag_search.py) pins
this — it fails if a future refactor wires search_permissive() into endpoint.py.
"""

from __future__ import annotations

import asyncio
import logging
import re

from app.config import RAG_MIN_SCORE, RAG_TOP_K, RAG_TRANSLATE_ON_MISS
from app.db import rag_store
from app.rag.embeddings import embed_text
from app.rag.translate import translate_to_english

logger = logging.getLogger(__name__)

NO_MATERIAL_NOTE = "No relevant course material was found for that question."

# Multi-intent queries: "SEO details and fee structure" is ONE embedding, and
# it lands in SEO space because the corpus is SEO-dense. Measured on the live
# index after the 2026-08-27 two-way call: the SEO syllabus filled ranks 1-10
# (0.43-0.52) and the first fee-bearing chunk sat at rank 11 with 0.402 —
# comfortably above RAG_MIN_SCORE, invisible at RAG_TOP_K. The agent then
# truthfully told the lead it had no fee information, one turn after quoting
# the fee. So explicit conjunctions split the query into sub-queries that are
# retrieved CONCURRENTLY (wall-clock ≈ one embed) under the SAME min_score —
# a fragment that clears nothing contributes nothing, so the split can never
# admit material a single query of the same meaning would have excluded.
# English and Telugu coordinators both, because the conversation LLM writes
# tool queries in either.
_INTENT_SPLIT = re.compile(r"\s+(?:and|&|మరియు|అండ్)\s+", re.IGNORECASE)
# Embeds per tool call stay bounded even for a rambling compound question.
_MAX_INTENTS = 3


def _split_intents(query: str) -> list[str]:
    parts = [p.strip() for p in _INTENT_SPLIT.split(query) if p.strip()]
    if len(parts) < 2:
        return [query]
    return parts[:_MAX_INTENTS]


def _format(hits: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[From {h['section'] or 'course material'}]\n{h['content']}" for h in hits
    )


async def _embed_and_match(query: str, k: int, min_score: float) -> list[dict]:
    # embed_text is a synchronous HTTP call. Off-loaded to a thread because
    # this runs on the same event loop that drives the call worker and the
    # transcript webhook — blocking it here stalls every other in-flight
    # call, not just this one lookup.
    embedding = await asyncio.to_thread(embed_text, query)
    return await rag_store.match_chunks(embedding, match_count=k, min_score=min_score)


async def _retrieve(query: str, k: int, min_score: float) -> list[dict]:
    """Top-k chunks for *query*, splitting explicit conjunctions first.

    Sub-query pools are merged by interleaving ranks, deduplicated on
    content, and capped at *k*. Interleaving, not a global sort: similarity
    scores against DIFFERENT sub-queries are not comparable, and sorting by
    them would rebuild exactly the crowding-out the split exists to prevent.
    """
    parts = _split_intents(query)
    if len(parts) == 1:
        return await _embed_and_match(query, k, min_score)
    pools = await asyncio.gather(
        *(_embed_and_match(part, k, min_score) for part in parts))
    merged: list[dict] = []
    seen: set[str] = set()
    for rank in range(max(map(len, pools), default=0)):
        for pool in pools:
            if rank < len(pool) and pool[rank]["content"] not in seen:
                seen.add(pool[rank]["content"])
                merged.append(pool[rank])
    return merged[:k]


async def search_permissive(query: str, k: int = RAG_TOP_K) -> list[dict]:
    """Top-k chunks regardless of relevance score. DEBUG/ADMIN USE ONLY —
    see module docstring. Returns [] for an empty/whitespace query."""
    query = query.strip()
    if not query:
        return []
    return await _embed_and_match(query, k, min_score=0.0)


async def search_relevant(
    query: str, k: int = RAG_TOP_K, min_score: float = RAG_MIN_SCORE
) -> str:
    """Formatted, speakable answer text for the live RAG tool. Below
    min_score, returns NO_MATERIAL_NOTE rather than "" — the voice/call tool
    contract requires always returning something speakable (an empty string
    would leave the agent tool-calling with nothing to say), unlike a
    text-chat context-injection function that could safely return "" to mean
    "don't inject anything".

    On a miss, the query is translated to English and retried ONCE (see
    app/rag/translate.py): the course docs are English-only, and Telugu-script
    questions embed too far from them to clear any threshold that still
    excludes irrelevant material. min_score is applied identically to both
    attempts — the retry changes the QUERY's language, never the relevance
    bar, so a translated query cannot surface material an English query of
    the same meaning wouldn't have.
    """
    query = query.strip()
    if not query:
        return NO_MATERIAL_NOTE

    hits = await _retrieve(query, k, min_score)
    if hits:
        return _format(hits)

    if not RAG_TRANSLATE_ON_MISS:
        return NO_MATERIAL_NOTE

    translated = await translate_to_english(query)
    if not translated:
        return NO_MATERIAL_NOTE

    # _retrieve, not _embed_and_match: the translation carries the same
    # conjunction the original had, and hits the same crowding-out without
    # the split.
    hits = await _retrieve(translated, k, min_score)
    if not hits:
        return NO_MATERIAL_NOTE
    logger.info(f"[rag] miss on the original query, hit after translating to "
                f"{translated!r}")
    return _format(hits)
