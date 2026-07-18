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


def test_endpoint_module_never_imports_search_permissive():
    """Static guard, independent of the behavioural test above: the endpoint
    module must not have search_permissive bound as a name at all (e.g. via
    `from app.rag.search import search_permissive`) — checking the module's
    actual namespace, not a source-text substring, so this doesn't false-fail
    on a comment that merely WARNS not to do this (see this module's own
    docstring, which names search_permissive for exactly that reason)."""
    assert "search_permissive" not in vars(endpoint)
