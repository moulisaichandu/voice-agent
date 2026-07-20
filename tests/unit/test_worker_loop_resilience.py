"""run_worker()'s loop must survive a failing iteration (worker.py docstring
point 8).

It runs as a background asyncio task started from app.main's lifespan, so an
escaping exception killed the coroutine while the process kept serving health
checks — the API looked fine, the scheduler kept enqueueing, and nothing ever
dialled again until a human noticed. A transient Redis/Postgres blip must
cost one retry, not the whole dialer.
"""

import asyncio

import pytest

from app.telephony import worker


@pytest.fixture(autouse=True)
def _no_real_backoff(monkeypatch):
    """The loop sleeps on failure; keep tests instant without patching
    asyncio.sleep globally (which would also break the stop_event waits)."""
    monkeypatch.setattr(worker, "_MAX_LOOP_BACKOFF_S", 0)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(worker.asyncio, "sleep", lambda d: real_sleep(0))


async def test_loop_survives_a_failing_reserve_and_keeps_going(monkeypatch):
    stop_event = asyncio.Event()
    attempts = {"n": 0}

    async def flaky_reserve(timeout_s=2):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ConnectionError("Redis went away mid-BLMOVE")
        if attempts["n"] >= 3:
            stop_event.set()
        return None

    monkeypatch.setattr(worker, "_reserve_next", flaky_reserve)

    await asyncio.wait_for(worker.run_worker(stop_event), timeout=5)

    # It kept looping past the failure rather than dying on iteration 1.
    assert attempts["n"] >= 3


async def test_loop_survives_a_failing_process_one(monkeypatch):
    stop_event = asyncio.Event()
    processed = []

    async def reserve(timeout_s=2):
        if len(processed) >= 2:
            stop_event.set()
            return None
        return f"lead-{len(processed)}"

    async def flaky_process(lead_id):
        processed.append(lead_id)
        raise RuntimeError("unhandled error that escaped process_one")

    monkeypatch.setattr(worker, "_reserve_next", reserve)
    monkeypatch.setattr(worker, "process_one", flaky_process)

    await asyncio.wait_for(worker.run_worker(stop_event), timeout=5)

    assert processed == ["lead-0", "lead-1"]  # second lead still attempted


async def test_loop_does_not_swallow_cancellation(monkeypatch):
    """Shutdown cancels the task. That must propagate, not be caught by the
    fault handler and retried — otherwise the task never stops."""
    async def hang(timeout_s=2):
        await asyncio.sleep(3600)

    monkeypatch.setattr(worker, "_reserve_next", hang)

    task = asyncio.create_task(worker.run_worker(asyncio.Event()))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
