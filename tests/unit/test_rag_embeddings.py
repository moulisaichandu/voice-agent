"""app/rag/embeddings.py — the shared OpenAI embeddings client must be BOUNDED.

The embeddings call sits on the LIVE /rag/search tool path (a two-way call's
agent invokes search_course_material mid-conversation), so an unbounded request
could hang for the SDK default (~600s read x retries) and saturate the shared
ThreadPoolExecutor, cascading a stall across every in-flight call — the exact
degrade the never-5xx/always-speakable contract exists to avoid. translate.py
bounds the same live path at 6s; this pins that embeddings is bounded too.
"""

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
