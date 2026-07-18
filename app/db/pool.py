"""db/pool.py — Shared asyncpg connection pool.

Raw SQL against DATABASE_URL, not the supabase-py client, for CRUD — see the
plan's design decision: Supabase Postgres IS Postgres, so a local pgvector
container gives dev/test parity, and pointing at real Supabase later is a
one-line .env change, zero code change.
"""

from __future__ import annotations

import asyncio

import asyncpg

from app.config import DATABASE_URL

_pool: asyncpg.Pool | None = None
_pool_loop: asyncio.AbstractEventLoop | None = None


async def get_pool() -> asyncpg.Pool:
    """A process-global pool is correct for the real app (one uvicorn worker,
    one long-lived event loop) but NOT for a test suite where pytest-asyncio
    gives each test function its own event loop by default — a pool created
    under test 1's loop is unusable (raises "Event loop is closed") once test
    2 starts on a new one. Detect the loop change and transparently recreate
    the pool rather than requiring every test file to manage pool lifecycle."""
    global _pool, _pool_loop
    current_loop = asyncio.get_running_loop()
    if _pool is not None and _pool_loop is not current_loop:
        try:
            await _pool.close()
        except Exception:
            pass  # the old loop may already be gone — nothing to clean up
        _pool = None

    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL is not set — cannot create a DB pool.")
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)
        _pool_loop = current_loop
    return _pool


async def close_pool() -> None:
    global _pool, _pool_loop
    if _pool is not None:
        await _pool.close()
        _pool = None
        _pool_loop = None
