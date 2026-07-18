"""rag/embeddings.py — Shared OpenAI embeddings client for ingest.py + search.py.

A separate credential from ElevenLabs: text-embedding-3-small is an OpenAI
model, so this needs OPENAI_API_KEY specifically — see the plan's flagged
credential gap. The client is created lazily so importing this module never
fails even without a key; only a real embed call does.
"""

from __future__ import annotations

from openai import OpenAI

from app.config import EMBED_MODEL, OPENAI_API_KEY

_client: OpenAI | None = None

# OpenAI's embeddings endpoint accepts a batch of inputs per call — chunking
# keeps any single request well under its size limits.
_EMBED_BATCH = 96


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set — cannot embed text for RAG. This is "
                "an OpenAI credential, separate from ElevenLabs; you likely "
                "already have it from the ai-voice-agent project's backend/.env."
            )
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings. Tests mock this function directly (mirroring
    ai-voice-agent/backend/rag.py's `_embed` pattern) rather than mocking the
    OpenAI SDK, so ingest/search logic is testable with zero network calls."""
    client = _get_client()
    vecs: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + _EMBED_BATCH])
        vecs.extend(d.embedding for d in resp.data)
    return vecs


def embed_text(text: str) -> list[float]:
    """Single-string convenience wrapper (e.g. embedding a live search query)."""
    return embed_texts([text])[0]
