A Telugu/Tinglish RAG query is failing to retrieve the right course-doc chunk
(or is retrieving an irrelevant one). Given the failing query:

1. Reproduce it against `app/rag/search.py`'s `search_relevant()` with the real
   (or a representative mocked) embeddings client — check the actual similarity
   scores returned, not just pass/fail.
2. Diagnose: is it a chunking problem (the right content exists but is split
   badly), a threshold problem (`RAG_MIN_SCORE` too strict/loose for this
   query), or an embedding problem (Telugu/Tinglish phrasing embeds far from
   the matching English-labelled chunk)?
3. Propose the smallest fix — re-chunk `app/rag/ingest.py`'s boundaries, adjust
   `RAG_MIN_SCORE` (with a comment explaining why), or add a query-rewriting
   step — and explain the tradeoff.
4. Add a regression test in `tests/unit/test_rag_search.py` covering this exact
   query so it can't silently regress.

Never fix this by wiring `search_permissive()` into the live path — that
reintroduces the exact "reads out irrelevant course material" bug this project
is built to avoid.

Failing query: $ARGUMENTS
