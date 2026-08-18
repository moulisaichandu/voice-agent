"""rag/embeddings.py — Shared OpenAI embeddings client for ingest.py + search.py.

A separate credential from ElevenLabs: text-embedding-3-small is an OpenAI
model, so this needs OPENAI_API_KEY specifically — see the plan's flagged
credential gap. The client is created lazily so importing this module never
fails even without a key; only a real embed call does.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict

from openai import OpenAI

from app.config import EMBED_MODEL, OPENAI_API_KEY

logger = logging.getLogger(__name__)

_client: OpenAI | None = None

# OpenAI's embeddings endpoint accepts a batch of inputs per call — chunking
# keeps any single request well under its size limits.
_EMBED_BATCH = 96

# The live /rag/search tool calls embed_text mid-conversation, so this MUST be
# bounded: the SDK default is a 600s read timeout with 2 retries, so a slow or
# hung OpenAI endpoint would occupy a ThreadPoolExecutor worker for ~20 minutes
# (a hang is not an exception, so endpoint.py's degrade path never fires), and
# ~32 concurrent hangs saturate the pool, stalling every other asyncio.to_thread
# in the process. translate.py bounds the same live path at 6s; embeddings gets
# a slightly larger ceiling since it may batch, plus a single retry.
_EMBED_TIMEOUT_S = 8.0
_EMBED_MAX_RETRIES = 1


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set — cannot embed text for RAG. This is "
                "an OpenAI credential, separate from ElevenLabs; you likely "
                "already have it from the ai-voice-agent project's backend/.env."
            )
        _client = OpenAI(api_key=OPENAI_API_KEY, timeout=_EMBED_TIMEOUT_S,
                         max_retries=_EMBED_MAX_RETRIES)
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


# Live search queries repeat; document chunks do not. The model is instructed
# to write short English keyword queries, so across a campaign the SAME handful
# of strings ('digital marketing course fees') recurs on lead after lead, and
# each one costs a network round trip inside RAG_VOICE_DEADLINE_S while a real
# person waits. Measured 2026-08-17: two course lookups on one call blew that
# 4s deadline and the lead was told there was no material about the courses
# this business sells.
#
# Keyed on the MODEL as well as the text: a bare-text key would hand a
# text-embedding-3-small vector to a run reconfigured for another model, which
# fails silently as slightly wrong relevance scores rather than as an error.
#
# Bounded, because this process is long-lived and one entry is 1536 floats
# (~50KB as a Python list). 128 entries is a few MB and far more distinct
# queries than a campaign produces. Only embed_text is cached — embed_texts is
# ingest, thousands of unique chunks with no reuse whatsoever.
_QUERY_CACHE: OrderedDict[tuple[str, str], list[float]] = OrderedDict()
_QUERY_CACHE_MAX = 128


def embed_text(text: str) -> list[float]:
    """Single-string convenience wrapper (e.g. embedding a live search query).

    Memoised — see _QUERY_CACHE. The memoisation is transparent: the same model
    and the same input always produce the same vector, so this cannot change
    which chunks match or how RAG_MIN_SCORE filters them. It also speeds up
    POST /rag/search, the ElevenLabs agent's live tool, for the same reason.
    """
    key = (EMBED_MODEL, text)
    cached = _QUERY_CACHE.get(key)
    if cached is not None:
        _QUERY_CACHE.move_to_end(key)          # least-recently-used ordering
        return list(cached)                    # a copy: callers must not mutate ours
    vec = embed_texts([text])[0]
    _QUERY_CACHE[key] = list(vec)
    while len(_QUERY_CACHE) > _QUERY_CACHE_MAX:
        _QUERY_CACHE.popitem(last=False)       # evict the oldest
    return vec


# Short and cheap, and never used as a real query — this exists only to make
# the TCP+TLS handshake happen somewhere that nobody is listening on a phone.
_WARM_QUERY = "warm"


async def warm() -> None:
    """Open the connection to OpenAI at startup rather than mid-call.

    Measured 2026-08-17: the first embed of a container's life took 3.56s
    against 0.23-0.51s once warm. That handshake landed inside a live call's
    4s course-lookup deadline, so the lookup was abandoned and the lead heard
    "I don't have that information" about this business's own courses.

    Best-effort by definition. A cold or unreachable OpenAI must not stop the
    app booting: without this the system behaves exactly as it does today, with
    the first real query paying the handshake instead.
    """
    if not OPENAI_API_KEY:
        return
    try:
        await asyncio.to_thread(embed_text, _WARM_QUERY)
    except Exception as exc:  # noqa: BLE001 - an optimisation must never fail boot
        logger.warning(
            f"[startup] could not warm the embeddings connection "
            f"({type(exc).__name__}: {exc}) — the first live course lookup will "
            "pay the handshake instead."
        )
