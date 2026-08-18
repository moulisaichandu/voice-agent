"""db/rag_store.py — doc_chunks storage + vector similarity search.

asyncpg has no native pgvector codec, so embeddings are passed as pgvector's
text literal format ('[0.1,0.2,...]') and cast with ::vector in SQL — the
standard approach that avoids a custom type-codec registration for a handful
of query sites.
"""

from __future__ import annotations

from app.db.pool import get_pool


def _to_vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


async def insert_chunk(*, doc_name: str, section: str | None, content: str,
                        embedding: list[float]) -> None:
    pool = await get_pool()
    await pool.execute(
        "insert into doc_chunks (doc_name, section, content, embedding) "
        "values ($1, $2, $3, $4::vector)",
        doc_name, section, content, _to_vector_literal(embedding),
    )


async def clear_doc(doc_name: str) -> None:
    """Delete all chunks for a doc before re-ingesting it, so a re-run of
    ingest.py doesn't accumulate stale duplicate chunks."""
    pool = await get_pool()
    await pool.execute("delete from doc_chunks where doc_name = $1", doc_name)


async def replace_doc_chunks(
    doc_name: str, rows: list[tuple[str, list[float]]]
) -> None:
    """Atomically replace a doc's chunks: delete the old ones and insert the new
    ones in ONE transaction. *rows* is [(content, embedding), ...].

    Atomicity matters twice: a concurrent match_chunks() on the live /rag/search
    path never sees a partially-rebuilt or empty corpus mid-re-ingest, and any
    failure during the inserts rolls back the delete so the prior chunks survive
    intact (ingest.py embeds BEFORE calling this, so an embed failure never even
    reaches here)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("delete from doc_chunks where doc_name = $1", doc_name)
            for content, embedding in rows:
                await conn.execute(
                    "insert into doc_chunks (doc_name, section, content, embedding) "
                    "values ($1, $2, $3, $4::vector)",
                    doc_name, None, content, _to_vector_literal(embedding),
                )


async def match_chunks(
    embedding: list[float], *, match_count: int, min_score: float
) -> list[dict]:
    """Calls the match_chunks() SQL function (migrations/0001_init.sql), which
    already filters by min_score server-side — this is the SQL-side half of
    the same relevance-threshold safety net app/rag/search.py enforces in
    Python. Returns [] if nothing clears the threshold; callers must NOT treat
    an empty list as "search failed", only as "nothing relevant"."""
    pool = await get_pool()
    rows = await pool.fetch(
        "select * from match_chunks($1::vector, $2, $3)",
        _to_vector_literal(embedding), match_count, min_score,
    )
    return [dict(r) for r in rows]
