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

from app.config import RAG_MIN_SCORE, RAG_TOP_K
from app.db import rag_store
from app.rag.embeddings import embed_text

NO_MATERIAL_NOTE = "No relevant course material was found for that question."


async def search_permissive(query: str, k: int = RAG_TOP_K) -> list[dict]:
    """Top-k chunks regardless of relevance score. DEBUG/ADMIN USE ONLY —
    see module docstring. Returns [] for an empty/whitespace query."""
    query = query.strip()
    if not query:
        return []
    embedding = embed_text(query)
    return await rag_store.match_chunks(embedding, match_count=k, min_score=0.0)


async def search_relevant(
    query: str, k: int = RAG_TOP_K, min_score: float = RAG_MIN_SCORE
) -> str:
    """Formatted, speakable answer text for the live RAG tool. Below
    min_score, returns NO_MATERIAL_NOTE rather than "" — the voice/call tool
    contract requires always returning something speakable (an empty string
    would leave the agent tool-calling with nothing to say), unlike a
    text-chat context-injection function that could safely return "" to mean
    "don't inject anything"."""
    query = query.strip()
    if not query:
        return NO_MATERIAL_NOTE
    embedding = embed_text(query)
    hits = await rag_store.match_chunks(embedding, match_count=k, min_score=min_score)
    if not hits:
        return NO_MATERIAL_NOTE
    return "\n\n---\n\n".join(
        f"[From {h['section'] or 'course material'}]\n{h['content']}" for h in hits
    )
