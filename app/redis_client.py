"""
redis_client.py — Shared async Redis connection.

Redis is the ephemeral speed layer (queue, cache, DND set, rate limits, live-call
registry, idempotency keys) — everything in it is rebuildable from Postgres, so a
restart is safe. Per the startup-checks design: an unreachable Redis must NOT
crash the whole app (health/RAG/Sheets-sync can still work), but the calling path
(campaign_tick / worker) must hard-refuse to enqueue anything and log loudly
rather than silently losing leads. See app/telephony/worker.py and app/scheduler.py.
"""

from __future__ import annotations

import asyncio
import logging

import redis.asyncio as redis

from app.config import REDIS_URL

logger = logging.getLogger(__name__)

_client: redis.Redis | None = None
_client_loop: asyncio.AbstractEventLoop | None = None


def get_redis() -> redis.Redis:
    """Lazily create the shared client. Connection is established on first use,
    not at import time, so importing this module never fails even if Redis is
    down — callers that need Redis to be reachable must call ping() themselves.

    Loop-aware for the same reason app/db/pool.py's get_pool() is: a client
    created under one asyncio event loop is unusable once that loop closes
    (raises "Event loop is closed"). The real app has exactly one long-lived
    loop for its whole process, so this never matters in production — but
    pytest-asyncio gives each test function its own loop by default, so a
    naive process-global client breaks on the second test that touches Redis."""
    global _client, _client_loop
    current_loop = asyncio.get_running_loop()
    if _client is not None and _client_loop is not current_loop:
        _client = None  # can't close it on a dead loop; just drop the reference

    if _client is None:
        _client = redis.from_url(REDIS_URL, decode_responses=True)
        _client_loop = current_loop
    return _client


async def is_available() -> bool:
    try:
        await get_redis().ping()
        return True
    except Exception as exc:
        logger.warning(f"[redis] unreachable: {type(exc).__name__}: {exc}")
        return False


async def close() -> None:
    global _client, _client_loop
    if _client is not None:
        # Same loop-aware guard as get_redis(): a client created under a now-
        # dead event loop (e.g. a previous pytest-asyncio test function's
        # loop) can't have its transport closed on THIS loop — that raises
        # "Event loop is closed" rather than actually closing anything. Just
        # drop the stale reference in that case, matching get_redis()'s own
        # handling of the identical situation.
        if _client_loop is asyncio.get_running_loop():
            await _client.aclose()
        _client = None
        _client_loop = None
