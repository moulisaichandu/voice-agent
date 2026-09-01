"""RAG search + the relevance-threshold safety pattern.

Mocks app.rag.embeddings.embed_text and app.db.rag_store's functions directly
— mirroring ai-voice-agent/backend/tests/test_rag.py's approach of patching
the embedding call — so these tests run with zero network calls and zero
Postgres/Docker dependency.
"""

import pytest
from fastapi import HTTPException

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


HINDI_FEE_QUERY = "कोर्स की फीस कितनी है?"
HINGLISH_FEE_QUERY = "Course ki fees kitni hai?"


async def test_hindi_miss_is_retried_in_english_and_then_hits(monkeypatch):
    monkeypatch.setattr(search, "RAG_TRANSLATE_ON_MISS", True)
    seen_queries = []

    def fake_embed(text):
        seen_queries.append(text)
        return [1.0, 0.0]

    async def fake_match_chunks(embedding, *, match_count, min_score):
        if seen_queries[-1] == HINDI_FEE_QUERY:
            return []
        return [{"content": "Total Fee: 1,50,000 rupees.", "section": "fees", "score": 0.46}]

    async def fake_translate(q):
        assert q == HINDI_FEE_QUERY
        return "What is the course fee?"

    monkeypatch.setattr(search, "embed_text", fake_embed)
    monkeypatch.setattr(search, "translate_to_english", fake_translate)
    monkeypatch.setattr("app.db.rag_store.match_chunks", fake_match_chunks)

    result = await search.search_relevant(HINDI_FEE_QUERY)

    assert "1,50,000" in result
    assert seen_queries == [HINDI_FEE_QUERY, "What is the course fee?"]


async def test_hinglish_miss_is_retried_in_english_and_then_hits(monkeypatch):
    monkeypatch.setattr(search, "RAG_TRANSLATE_ON_MISS", True)
    seen_queries = []

    def fake_embed(text):
        seen_queries.append(text)
        return [1.0, 0.0]

    async def fake_match_chunks(embedding, *, match_count, min_score):
        if seen_queries[-1] == HINGLISH_FEE_QUERY:
            return []
        return [{"content": "Total Fee: 1,50,000 rupees.", "section": "fees", "score": 0.46}]

    async def fake_translate(q):
        assert q == HINGLISH_FEE_QUERY
        return "What is the course fee?"

    monkeypatch.setattr(search, "embed_text", fake_embed)
    monkeypatch.setattr(search, "translate_to_english", fake_translate)
    monkeypatch.setattr("app.db.rag_store.match_chunks", fake_match_chunks)

    result = await search.search_relevant(HINGLISH_FEE_QUERY)

    assert "1,50,000" in result
    assert seen_queries == [HINGLISH_FEE_QUERY, "What is the course fee?"]


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


# ── multi-intent queries: every named topic must be retrieved ────────────────
# Measured on the live index after the 2026-08-27 two-way call: for
# "SEO details and fee structure" the SEO syllabus fills ranks 1-10
# (0.43-0.52 — the corpus is SEO-dense) and the first fee-bearing chunk sits
# at rank 11 with 0.402: comfortably relevant, but invisible at RAG_TOP_K=4.
# The tool therefore returned SEO material only, and the agent — correctly
# refusing to invent fees — told the lead fee details were unavailable and
# promised a callback, ONE TURN after it had quoted the starting fee.
# The fix splits explicit conjunctions into sub-queries retrieved
# concurrently under the SAME threshold, then interleaves the pools.

COMBINED_QUERY = "SEO details and fee structure"

_SEO_POOL = [
    {"content": f"SEO syllabus item {i}", "section": "seo", "score": 0.52 - i / 100}
    for i in range(4)
]
_FEE_CHUNK = {"content": "Total Fee: 1,50,000 rupees.", "section": "fees", "score": 0.55}


def _embed_as_text(text):
    """Embedding stub that lets fake stores key off the exact query."""
    return text


async def test_a_two_intent_query_retrieves_both_topics(monkeypatch):
    monkeypatch.setattr(search, "embed_text", _embed_as_text)

    async def fake_match_chunks(embedding, *, match_count, min_score):
        assert min_score == 0.30  # sub-queries face the same bar, never looser
        if embedding == "fee structure":
            return [_FEE_CHUNK]
        if embedding in (COMBINED_QUERY, "SEO details"):
            return list(_SEO_POOL)  # SEO crowds out everything else
        return []

    monkeypatch.setattr("app.db.rag_store.match_chunks", fake_match_chunks)

    result = await search.search_relevant(COMBINED_QUERY)

    assert "1,50,000" in result, (
        "the fee intent was crowded out of top-k by the SEO-dense corpus"
    )
    assert "SEO syllabus item 0" in result, "the first intent must survive too"


async def test_split_sub_queries_are_deduplicated(monkeypatch):
    """A chunk matching two sub-queries must be spoken about once, not twice."""
    monkeypatch.setattr(search, "embed_text", _embed_as_text)

    async def fake_match_chunks(embedding, *, match_count, min_score):
        if embedding == COMBINED_QUERY:
            return list(_SEO_POOL)
        return [_FEE_CHUNK, _SEO_POOL[0]]  # both parts return the same chunks

    monkeypatch.setattr("app.db.rag_store.match_chunks", fake_match_chunks)

    result = await search.search_relevant(COMBINED_QUERY)

    assert result.count("Total Fee: 1,50,000 rupees.") == 1
    assert result.count("SEO syllabus item 0") == 1


async def test_an_off_topic_half_contributes_nothing(monkeypatch):
    """The split must not weaken relevance: a sub-query that clears nothing
    returns nothing — the threshold guards each part independently, so the
    split can never leak below-threshold material into the live tool."""
    monkeypatch.setattr(search, "embed_text", _embed_as_text)

    async def fake_match_chunks(embedding, *, match_count, min_score):
        if embedding == "fee structure":
            return [_FEE_CHUNK]
        return []  # "the weather today" clears nothing anywhere

    monkeypatch.setattr("app.db.rag_store.match_chunks", fake_match_chunks)

    result = await search.search_relevant("the weather today and fee structure")

    assert "1,50,000" in result
    assert "weather" not in result


async def test_the_translated_retry_is_also_split(monkeypatch):
    """A Telugu two-intent question misses in Telugu (scores ~0.1-0.19 against
    the English-only docs), gets translated — and the translation carries the
    same 'and', so it must get the same split or it hits the same rank-11
    wall the original fix exists for."""
    monkeypatch.setattr(search, "RAG_TRANSLATE_ON_MISS", True)
    monkeypatch.setattr(search, "embed_text", _embed_as_text)

    async def fake_translate(q):
        return COMBINED_QUERY

    async def fake_match_chunks(embedding, *, match_count, min_score):
        if embedding == "fee structure":
            return [_FEE_CHUNK]
        if embedding == "SEO details":
            return list(_SEO_POOL)
        return []  # every Telugu-script embedding misses

    monkeypatch.setattr(search, "translate_to_english", fake_translate)
    monkeypatch.setattr("app.db.rag_store.match_chunks", fake_match_chunks)

    result = await search.search_relevant(
        "ఎస్ఈఓ గురించి చెప్పండి మరియు ఫీజు స్ట్రక్చర్ చెప్పండి")

    assert "1,50,000" in result
    assert "SEO syllabus item 0" in result


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


# ── tool auth (RAG_TOOL_SECRET) ──────────────────────────────────────────────

def test_rag_search_is_open_when_no_tool_secret_is_configured(client, monkeypatch):
    """Local-dev default, matching CALL_WEBHOOK_SECRET. main.py refuses to boot
    in this state behind a PUBLIC_BASE_URL, so it can only ever be local."""
    monkeypatch.setattr(endpoint, "RAG_TOOL_SECRET", None)

    async def fake_search(q):
        return "course material"

    monkeypatch.setattr(endpoint, "search_relevant", fake_search)

    r = client.post("/rag/search", json={"query": "fees"})
    assert r.status_code == 200


def test_rag_search_rejects_a_missing_or_wrong_tool_token(client, monkeypatch):
    """Each request spends OpenAI credit and returns verbatim course material,
    so an unauthenticated caller is both a billing drain and a corpus leak."""
    monkeypatch.setattr(endpoint, "RAG_TOOL_SECRET", "s3cret")

    async def boom(q):
        raise AssertionError("must not reach the paid search path")

    monkeypatch.setattr(endpoint, "search_relevant", boom)

    assert client.post("/rag/search", json={"query": "fees"}).status_code == 401
    assert client.post("/rag/search", json={"query": "fees"},
                       headers={"X-RAG-Token": "wrong"}).status_code == 401


def test_rag_search_accepts_the_configured_tool_token(client, monkeypatch):
    monkeypatch.setattr(endpoint, "RAG_TOOL_SECRET", "s3cret")

    async def fake_search(q):
        return "course material"

    monkeypatch.setattr(endpoint, "search_relevant", fake_search)

    r = client.post("/rag/search", json={"query": "fees"},
                    headers={"X-RAG-Token": "s3cret"})
    assert r.status_code == 200
    assert r.json()["result"] == "course material"


async def test_a_non_ascii_tool_token_is_a_clean_401_not_a_crash(monkeypatch):
    """compare_digest raises TypeError on a non-ASCII str, which would surface
    as a 500 — an error the live agent has no script for, and an oracle
    confirming the real secret is ASCII.

    Exercised against the dependency directly rather than through the test
    client: HTTP headers are ASCII/latin-1 on the wire, so httpx refuses to
    send this before it ever reaches the server. The guard still encodes
    defensively because it is not the only possible caller."""
    monkeypatch.setattr(endpoint, "RAG_TOOL_SECRET", "s3cret")
    with pytest.raises(HTTPException) as exc:
        await endpoint.require_rag_tool_auth(x_rag_token="café")
    assert exc.value.status_code == 401
