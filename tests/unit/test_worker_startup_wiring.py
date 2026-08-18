"""Regression tests for app.main actually starting/stopping the call worker.

This was the single biggest gap the project audit found: run_worker()
existed, was fully tested in isolation, but nothing ever called it from the
running app — campaign_tick would enqueue leads onto Redis that no consumer
ever dequeued. These tests exercise app.main's lifespan directly (not through
a real Redis/DB) to pin down that a worker task actually gets created and
actually gets shut down.
"""

import asyncio

from app import main as app_main


async def _redis_up():
    return True


async def _redis_down():
    return False


async def test_worker_starts_and_stops_when_enabled_and_redis_is_up(monkeypatch):
    monkeypatch.setattr(app_main, "WORKER_ENABLED", True)
    monkeypatch.setattr(app_main, "SCHEDULER_ENABLED", False)
    monkeypatch.setattr(app_main.redis_client, "is_available", _redis_up)

    calls = {"entered": 0, "exited": 0}

    async def fake_run_worker(stop_event):
        calls["entered"] += 1
        await stop_event.wait()
        calls["exited"] += 1

    monkeypatch.setattr(app_main.worker, "run_worker", fake_run_worker)

    async with app_main.lifespan(app_main.app):
        await asyncio.sleep(0)  # let the background task reach its first await
        assert calls["entered"] == 1
        assert calls["exited"] == 0  # still running — stop_event not set yet

    assert calls["exited"] == 1  # lifespan exit must signal + await the task


async def test_worker_does_not_start_when_disabled(monkeypatch):
    monkeypatch.setattr(app_main, "WORKER_ENABLED", False)
    monkeypatch.setattr(app_main, "SCHEDULER_ENABLED", False)
    monkeypatch.setattr(app_main.redis_client, "is_available", _redis_up)

    calls = {"entered": 0}

    async def fake_run_worker(stop_event):
        calls["entered"] += 1

    monkeypatch.setattr(app_main.worker, "run_worker", fake_run_worker)

    async with app_main.lifespan(app_main.app):
        await asyncio.sleep(0)

    assert calls["entered"] == 0


async def test_worker_does_not_start_when_redis_is_down(monkeypatch):
    """Mirrors the scheduler's own rule: enqueueing/dialling infrastructure
    must not start against a Redis that isn't there — see app.main's own
    comment on SCHEDULER_ENABLED for the identical reasoning."""
    monkeypatch.setattr(app_main, "WORKER_ENABLED", True)
    monkeypatch.setattr(app_main, "SCHEDULER_ENABLED", False)
    monkeypatch.setattr(app_main.redis_client, "is_available", _redis_down)

    calls = {"entered": 0}

    async def fake_run_worker(stop_event):
        calls["entered"] += 1

    monkeypatch.setattr(app_main.worker, "run_worker", fake_run_worker)

    async with app_main.lifespan(app_main.app):
        await asyncio.sleep(0)

    assert calls["entered"] == 0


async def test_the_embeddings_connection_is_warmed_at_startup(monkeypatch):
    """The first embed of a container's life measured 3.56s against 0.23-0.51s
    warm, and on 2026-08-17 it landed mid-call: two course lookups blew
    RAG_VOICE_DEADLINE_S and the lead was told there was no material about the
    courses this business sells. Startup is the right place to pay that."""
    monkeypatch.setattr(app_main, "WORKER_ENABLED", False)
    monkeypatch.setattr(app_main, "SCHEDULER_ENABLED", False)
    monkeypatch.setattr(app_main.redis_client, "is_available", _redis_up)

    warmed = {"n": 0}

    async def fake_warm():
        warmed["n"] += 1

    monkeypatch.setattr(app_main.embeddings, "warm", fake_warm)

    async with app_main.lifespan(app_main.app):
        await asyncio.sleep(0)  # let the background task run
        assert warmed["n"] == 1


async def test_a_cold_openai_does_not_stop_the_app_booting(monkeypatch):
    """The warm-up is an optimisation. If it raises, the app must still serve —
    RAG simply pays the handshake on its first real query, as it does today."""
    monkeypatch.setattr(app_main, "WORKER_ENABLED", False)
    monkeypatch.setattr(app_main, "SCHEDULER_ENABLED", False)
    monkeypatch.setattr(app_main.redis_client, "is_available", _redis_up)

    async def boom():
        raise RuntimeError("openai unreachable")

    monkeypatch.setattr(app_main.embeddings, "warm", boom)

    async with app_main.lifespan(app_main.app):
        await asyncio.sleep(0)
