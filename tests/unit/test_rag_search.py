"""RAG search + the relevance-threshold safety pattern.

Mocks app.rag.embeddings.embed_text and app.db.rag_store's functions directly
— mirroring ai-voice-agent/backend/tests/test_rag.py's approach of patching
the embedding call — so these tests run with zero network calls and zero
Postgres/Docker dependency.
"""

import pytest

from app.rag import endpoint, search


@pytest.fixture(autouse=True)
def _stub_embedding(monkeypatch):
    monkeypatch.setattr(search, "embed_text", lambda text: [1.0, 0.0])
    # Translation-on-miss is off unless a test opts in, so the pre-existing
    # miss-path tests below keep asserting one clean behaviour instead of
    # silently exercising the retry too.
    monkeypatch.setattr(search, "RAG_TRANSLATE_ON_MISS", False)


async def test_search_relevant_returns_hits_above_threshold(monkeypatch):
    async def fake_match_chunks(embedding, *, match_count, min_score):
        assert min_score == 0.30  # the real threshold was actually passed through
        return [{"content": "The course fee is 5000 rupees.", "section": "fees", "score": 0.9}]

    monkeypatch.setattr("app.db.rag_store.match_chunks", fake_match_chunks)
    result = await search.search_relevant("what is the fee")
    assert "5000 rupees" in result
    assert "[From fees]" in result


async def test_search_relevant_returns_no_material_note_below_threshold(monkeypatch):
    async def fake_match_chunks(embedding, *, match_count, min_score):
        return []  # nothing cleared the SQL-side threshold

    monkeypatch.setattr("app.db.rag_store.match_chunks", fake_match_chunks)
    result = await search.search_relevant("hello, how are you")
    assert result == search.NO_MATERIAL_NOTE


async def test_search_relevant_empty_query_short_circuits(monkeypatch):
    called = {"n": 0}

    async def fake_match_chunks(*a, **kw):
        called["n"] += 1
        return []

    monkeypatch.setattr("app.db.rag_store.match_chunks", fake_match_chunks)
    result = await search.search_relevant("   ")
    assert result == search.NO_MATERIAL_NOTE
    assert called["n"] == 0  # never even embeds/queries an empty question


async def test_search_permissive_ignores_threshold(monkeypatch):
    async def fake_match_chunks(embedding, *, match_count, min_score):
        assert min_score == 0.0  # permissive really does pass 0, not the real threshold
        return [{"content": "x", "section": "s", "score": 0.01}]

    monkeypatch.setattr("app.db.rag_store.match_chunks", fake_match_chunks)
    hits = await search.search_permissive("anything")
    assert len(hits) == 1


# ── Telugu/Tinglish retrieval: translate-on-miss ──────────────────────────────
# Regression cover for a measured failure against the real 91-chunk corpus:
# "కోర్సు ఫీజు ఎంత?" (what is the course fee) scored 0.187 — below
# RAG_MIN_SCORE, and in the same band as a deliberately off-topic English
# query (0.128) — while the English equivalent scored 0.461. See
# app/rag/translate.py for why this is fixed by translating rather than by
# lowering the threshold.

TELUGU_FEE_QUERY = "కోర్సు ఫీజు ఎంత?"


async def test_telugu_miss_is_retried_in_english_and_then_hits(monkeypatch):
    monkeypatch.setattr(search, "RAG_TRANSLATE_ON_MISS", True)
    seen_queries = []

    def fake_embed(text):
        seen_queries.append(text)
        return [1.0, 0.0]

    async def fake_match_chunks(embedding, *, match_count, min_score):
        # Miss on the Telugu query, hit once it has been translated.
        if seen_queries[-1] == TELUGU_FEE_QUERY:
            return []
        return [{"content": "Total Fee: 1,50,000 rupees.", "section": "fees", "score": 0.46}]

    async def fake_translate(q):
        assert q == TELUGU_FEE_QUERY
        return "What is the course fee?"

    monkeypatch.setattr(search, "embed_text", fake_embed)
    monkeypatch.setattr(search, "translate_to_english", fake_translate)
    monkeypatch.setattr("app.db.rag_store.match_chunks", fake_match_chunks)

    result = await search.search_relevant(TELUGU_FEE_QUERY)

    assert "1,50,000" in result
    assert seen_queries == [TELUGU_FEE_QUERY, "What is the course fee?"]


async def test_translation_retry_never_weakens_the_relevance_threshold(monkeypatch):
    """The whole point of translating instead of lowering RAG_MIN_SCORE: the
    retry moves the QUERY's language, never the bar. If the retry ever ran at
    a looser threshold it would admit exactly the irrelevant material the
    threshold exists to exclude."""
    monkeypatch.setattr(search, "RAG_TRANSLATE_ON_MISS", True)
    thresholds = []

    async def fake_match_chunks(embedding, *, match_count, min_score):
        thresholds.append(min_score)
        return [] if len(thresholds) == 1 else [
            {"content": "Total Fee: 1,50,000 rupees.", "section": "fees", "score": 0.46}
        ]

    async def fake_translate(q):
        return "What is the course fee?"

    monkeypatch.setattr(search, "translate_to_english", fake_translate)
    monkeypatch.setattr("app.db.rag_store.match_chunks", fake_match_chunks)

    await search.search_relevant(TELUGU_FEE_QUERY)

    assert len(thresholds) == 2
    assert thresholds[0] == thresholds[1] == 0.30


async def test_a_hit_on_the_original_query_never_calls_the_translator(monkeypatch):
    """English is the common case and must not pay the translation latency —
    this runs while a lead is waiting on the phone."""
    monkeypatch.setattr(search, "RAG_TRANSLATE_ON_MISS", True)
    called = {"n": 0}

    async def fake_translate(q):
        called["n"] += 1
        return "should never happen"

    async def fake_match_chunks(embedding, *, match_count, min_score):
        return [{"content": "Total Fee: 1,50,000 rupees.", "section": "fees", "score": 0.46}]

    monkeypatch.setattr(search, "translate_to_english", fake_translate)
    monkeypatch.setattr("app.db.rag_store.match_chunks", fake_match_chunks)

    await search.search_relevant("What is the course fee?")
    assert called["n"] == 0


async def test_unavailable_translation_degrades_to_the_no_material_note(monkeypatch):
    """A dead/rate-limited OpenAI key must not turn a miss into an error the
    live agent can't voice — translate_to_english returns None on every
    failure path, and that must land on the normal fallback."""
    monkeypatch.setattr(search, "RAG_TRANSLATE_ON_MISS", True)

    async def fake_translate(q):
        return None

    async def fake_match_chunks(embedding, *, match_count, min_score):
        return []

    monkeypatch.setattr(search, "translate_to_english", fake_translate)
    monkeypatch.setattr("app.db.rag_store.match_chunks", fake_match_chunks)

    assert await search.search_relevant(TELUGU_FEE_QUERY) == search.NO_MATERIAL_NOTE


async def test_translation_retry_still_returns_the_note_when_nothing_matches(monkeypatch):
    """An off-topic question asked in Telugu must still be refused — the
    retry is a second chance at RETRIEVAL, not a bypass of relevance."""
    monkeypatch.setattr(search, "RAG_TRANSLATE_ON_MISS", True)

    async def fake_translate(q):
        return "What is the weather today?"

    async def fake_match_chunks(embedding, *, match_count, min_score):
        return []

    monkeypatch.setattr(search, "translate_to_english", fake_translate)
    monkeypatch.setattr("app.db.rag_store.match_chunks", fake_match_chunks)

    assert await search.search_relevant("ఈరోజు వాతావరణం ఎలా ఉంది?") == search.NO_MATERIAL_NOTE


# ── Pinning test: the live tool endpoint must never reach search_permissive ────

async def test_rag_endpoint_uses_filtered_search(monkeypatch):
    """If a future refactor wires search_permissive() into the live /rag/search
    endpoint, this must fail — that reintroduces the exact bug the blueprint's
    own "Mistakes to Avoid" section names: the agent reading out irrelevant
    course material in response to a greeting or off-topic remark."""
    calls = {"relevant": 0, "permissive": 0}

    async def fake_relevant(query, *a, **kw):
        calls["relevant"] += 1
        return "some course answer"

    async def fake_permissive(query, *a, **kw):
        calls["permissive"] += 1
        return []

    monkeypatch.setattr(endpoint, "search_relevant", fake_relevant)
    monkeypatch.setattr(search, "search_permissive", fake_permissive)

    result = await endpoint.rag_search(endpoint.RagQuery(query="what is the fee"))

    assert calls["relevant"] == 1
    assert calls["permissive"] == 0
    assert result == {"result": "some course answer"}


async def test_rag_endpoint_never_5xxs_when_retrieval_blows_up(monkeypatch):
    """The agent calling this tool is MID-CALL with a real person. A missing
    OPENAI_API_KEY (RuntimeError from embeddings._get_client), an OpenAI
    outage, or a DB blip must degrade to the speakable no-material note —
    not raise, which FastAPI would turn into a 500 the agent can't voice."""
    async def boom(query, *a, **kw):
        raise RuntimeError("OPENAI_API_KEY is not set — cannot embed text for RAG.")

    monkeypatch.setattr(endpoint, "search_relevant", boom)

    result = await endpoint.rag_search(endpoint.RagQuery(query="what is the fee"))

    assert result["result"] == search.NO_MATERIAL_NOTE
    assert result["degraded"] is True


async def test_rag_endpoint_does_not_flag_a_healthy_no_match_as_degraded(monkeypatch):
    """An off-topic question legitimately returns the same note — but that's
    a working search, not a degraded one, and the operator needs to tell the
    two apart in logs/metrics."""
    async def fake_relevant(query, *a, **kw):
        return search.NO_MATERIAL_NOTE

    monkeypatch.setattr(endpoint, "search_relevant", fake_relevant)

    result = await endpoint.rag_search(endpoint.RagQuery(query="what's the weather"))

    assert result == {"result": search.NO_MATERIAL_NOTE}
    assert "degraded" not in result


def test_endpoint_module_never_imports_search_permissive():
    """Static guard, independent of the behavioural test above: the endpoint
    module must not have search_permissive bound as a name at all (e.g. via
    `from app.rag.search import search_permissive`) — checking the module's
    actual namespace, not a source-text substring, so this doesn't false-fail
    on a comment that merely WARNS not to do this (see this module's own
    docstring, which names search_permissive for exactly that reason)."""
    assert "search_permissive" not in vars(endpoint)
