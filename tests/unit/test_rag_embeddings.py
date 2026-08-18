"""app/rag/embeddings.py — the shared OpenAI embeddings client must be BOUNDED.

The embeddings call sits on the LIVE /rag/search tool path (a two-way call's
agent invokes search_course_material mid-conversation), so an unbounded request
could hang for the SDK default (~600s read x retries) and saturate the shared
ThreadPoolExecutor, cascading a stall across every in-flight call — the exact
degrade the never-5xx/always-speakable contract exists to avoid. translate.py
bounds the same live path at 6s; this pins that embeddings is bounded too.
"""

from collections import OrderedDict
from types import SimpleNamespace

import pytest

from app.rag import embeddings


def _read_timeout_seconds(t) -> float:
    """The client timeout may be a bare float (when we set one) or an
    httpx.Timeout (the SDK default). Normalise to the read-timeout seconds."""
    if isinstance(t, (int, float)):
        return float(t)
    return float(getattr(t, "read", None) or getattr(t, "timeout", None) or 10 ** 9)


def test_the_embeddings_client_is_built_with_a_bounded_timeout_and_few_retries(monkeypatch):
    monkeypatch.setattr(embeddings, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(embeddings, "_client", None)  # force a fresh build

    client = embeddings._get_client()

    assert _read_timeout_seconds(client.timeout) <= 30, (
        "an unbounded embed can hang the mid-call RAG tool for ~600s and "
        "saturate the thread pool"
    )
    assert client.max_retries <= 1, (
        f"too many retries multiplies the worst-case hang: {client.max_retries}"
    )


# ── the query cache ─────────────────────────────────────────────────────────
#
# Measured on a live call (2026-08-17): the FIRST embed of a container's life
# took 3.56s against 0.23-0.51s warm, and it landed mid-call. Two course
# lookups blew RAG_VOICE_DEADLINE_S and the lead was told "I don't have that
# information" about the courses this business sells. The queries the model
# writes are a small recurring set ('digital marketing course fees'), so they
# are worth remembering across leads.


class _FakeClient:
    """Counts API calls. `embeddings.create` mirrors the SDK's response shape."""

    def __init__(self):
        self.calls = 0
        self.embeddings = self

    def create(self, model, input):  # noqa: A002 - the SDK's own parameter name
        self.calls += 1
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(len(t)), 0.5]) for t in input])


@pytest.fixture
def fake_client(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(embeddings, "_get_client", lambda: client)
    monkeypatch.setattr(embeddings, "_QUERY_CACHE", OrderedDict())
    return client


def test_a_repeated_query_is_embedded_only_once(fake_client):
    first = embeddings.embed_text("digital marketing course fees")
    second = embeddings.embed_text("digital marketing course fees")

    assert fake_client.calls == 1, "the same query paid for a second embed"
    assert first == second


def test_switching_the_embedding_model_never_serves_a_stale_vector(
        fake_client, monkeypatch):
    """A cache keyed on the text alone would hand a text-embedding-3-small
    vector to a run configured for a different model."""
    embeddings.embed_text("course fees")
    monkeypatch.setattr(embeddings, "EMBED_MODEL", "some-other-model")

    embeddings.embed_text("course fees")

    assert fake_client.calls == 2


def test_the_query_cache_is_bounded(fake_client):
    """An unbounded dict on a long-running server is a slow memory leak: one
    entry is 1536 floats."""
    for i in range(embeddings._QUERY_CACHE_MAX + 25):
        embeddings.embed_text(f"query number {i}")

    assert len(embeddings._QUERY_CACHE) <= embeddings._QUERY_CACHE_MAX


def test_the_oldest_query_is_the_one_evicted(fake_client):
    embeddings.embed_text("first query")
    for i in range(embeddings._QUERY_CACHE_MAX):
        embeddings.embed_text(f"filler {i}")
    calls_before = fake_client.calls

    embeddings.embed_text("first query")

    assert fake_client.calls == calls_before + 1, \
        "the oldest entry should have been evicted, forcing a re-embed"


def test_ingest_batches_are_never_cached(fake_client):
    """embed_texts embeds thousands of unique document chunks — caching those
    would blow memory for no reuse at all. Only live queries repeat."""
    embeddings.embed_texts(["chunk one", "chunk two"])
    embeddings.embed_texts(["chunk one", "chunk two"])

    assert fake_client.calls == 2


# ── the startup warm-up ─────────────────────────────────────────────────────


async def test_warm_pays_the_handshake_up_front(monkeypatch):
    monkeypatch.setattr(embeddings, "OPENAI_API_KEY", "sk-test")
    seen = []
    monkeypatch.setattr(embeddings, "embed_text",
                        lambda text: seen.append(text) or [0.1])

    await embeddings.warm()

    assert seen, "startup did not open the connection, so a lead pays for it"


async def test_warm_never_raises_when_openai_is_unreachable(monkeypatch):
    """A cold OpenAI must not stop the app booting — this is an optimisation."""
    monkeypatch.setattr(embeddings, "OPENAI_API_KEY", "sk-test")

    def boom(text):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(embeddings, "embed_text", boom)

    await embeddings.warm()  # must not raise


async def test_warm_does_nothing_without_a_key(monkeypatch):
    monkeypatch.setattr(embeddings, "OPENAI_API_KEY", "")
    seen = []
    monkeypatch.setattr(embeddings, "embed_text",
                        lambda text: seen.append(text) or [0.1])

    await embeddings.warm()

    assert not seen
