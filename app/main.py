"""
main.py — FastAPI application entry point.

Run (dev):   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
Run (prod):  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

Single worker only: APScheduler (app/scheduler.py) runs in-process and must not
be started twice — see config.SCHEDULER_ENABLED and the blueprint's own
"one scheduler, one instance" warning. The Redis call worker (app/telephony/
worker.py) ALSO runs in-process, as a background asyncio task started below —
unlike the scheduler it's safe on multiple instances (see config.WORKER_ENABLED).
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from app.logging_setup import setup_logging

setup_logging()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import redis_client, scheduler
from app.config import (
    ALLOW_INSECURE_PUBLIC,
    ALLOWED_ORIGINS,
    APP_AUTH_TOKEN,
    CALL_WEBHOOK_SECRET,
    DATABASE_URL,
    ELEVENLABS_WEBHOOK_SECRET,
    LOG_LEVEL,
    PUBLIC_BASE_URL,
    RAG_TOOL_SECRET,
    SCHEDULER_ENABLED,
    WORKER_ENABLED,
)
from app.telephony import worker

setup_logging(LOG_LEVEL)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Fail-fast checks ────────────────────────────────────────────────────
    # Refuse to boot in dangerous or non-functional configurations, rather than
    # starting and failing confusingly (or insecurely) on the first request.

    # Almost nothing in this app works without a database — unlike an LLM key
    # that can degrade gracefully, there is no reduced-functionality mode here.
    # (This checks the var is SET; real connectivity is verified once app/db/
    # provides a pool to ping against — see Phase 1.)
    if not DATABASE_URL:
        raise RuntimeError(
            "Refusing to start: DATABASE_URL is not set. Point it at the local "
            "docker-compose pgvector service for dev, or your Supabase connection "
            "string for production."
        )

    # A public server with no webhook secret means anyone can POST a forged
    # transcript that gets written to Supabase / the lead's Google Sheet row.
    if PUBLIC_BASE_URL and not ELEVENLABS_WEBHOOK_SECRET and not ALLOW_INSECURE_PUBLIC:
        raise RuntimeError(
            "Refusing to start: PUBLIC_BASE_URL is set but ELEVENLABS_WEBHOOK_SECRET "
            "is not. The transcript webhook would accept forged payloads. Set "
            "ELEVENLABS_WEBHOOK_SECRET (from the ElevenLabs webhook dashboard), or "
            "set ALLOW_INSECURE_PUBLIC=1 to deliberately run open (not recommended)."
        )

    # The Plivo webhooks (/calls/answer, /calls/stream) can't carry
    # APP_AUTH_TOKEN, so CALL_WEBHOOK_SECRET is the only thing guarding them.
    # Unguarded on a public URL, anyone who finds it could open an audio
    # bridge to a paid agent, or feed audio into a call.
    if PUBLIC_BASE_URL and not CALL_WEBHOOK_SECRET and not ALLOW_INSECURE_PUBLIC:
        raise RuntimeError(
            "Refusing to start: PUBLIC_BASE_URL is set but CALL_WEBHOOK_SECRET is "
            "not. The Plivo call webhooks would be open to the internet. Set "
            "CALL_WEBHOOK_SECRET to a random string, or set ALLOW_INSECURE_PUBLIC=1 "
            "to deliberately run open (not recommended)."
        )

    # /rag/search is the two-way agent's live tool. Every request costs real
    # money — a paid OpenAI embedding, plus a paid gpt-4o-mini completion when
    # the first attempt misses — and the response is verbatim course material.
    # Open on a public URL it is both a billing drain and a way to extract the
    # whole corpus a few chunks at a time.
    if PUBLIC_BASE_URL and not RAG_TOOL_SECRET and not ALLOW_INSECURE_PUBLIC:
        raise RuntimeError(
            "Refusing to start: PUBLIC_BASE_URL is set but RAG_TOOL_SECRET is not. "
            "POST /rag/search would be open to the internet, where each request "
            "spends OpenAI credit and returns your course documents verbatim. Set "
            "RAG_TOOL_SECRET to a random string and add it as the X-RAG-Token "
            "header on the agent's search tool in the ElevenLabs dashboard, or set "
            "ALLOW_INSECURE_PUBLIC=1 to deliberately run open (not recommended)."
        )

    # The admin API fails OPEN when APP_AUTH_TOKEN is unset (see
    # admin/__init__.py's require_admin_auth) — fine on a laptop, unacceptable
    # once the app is reachable from the internet. Without this check the two
    # guards above protected the webhook surfaces while /admin, the surface
    # that can create campaigns and PLACE REAL CALLS, sat wide open to anyone
    # who learned the tunnel hostname. It is also the surface that returns
    # lead PII and full call transcripts.
    if PUBLIC_BASE_URL and not APP_AUTH_TOKEN and not ALLOW_INSECURE_PUBLIC:
        raise RuntimeError(
            "Refusing to start: PUBLIC_BASE_URL is set but APP_AUTH_TOKEN is not. "
            "Every /admin route would be open to the internet — including "
            "POST /admin/trigger-tick, which places real phone calls, and "
            "GET /admin/calls, which returns lead PII and transcripts. Set "
            "APP_AUTH_TOKEN to a random string, or set ALLOW_INSECURE_PUBLIC=1 "
            "to deliberately run open (not recommended)."
        )

    # A wildcard CORS origin lets any website read this API's responses from a
    # visitor's browser. NOT conditioned on APP_AUTH_TOKEN being set: with no
    # token the endpoints need no credentials at all, so a wildcard is MORE
    # dangerous there, not less — that combination used to boot happily.
    if "*" in ALLOWED_ORIGINS and not ALLOW_INSECURE_PUBLIC:
        raise RuntimeError(
            "Refusing to start: ALLOWED_ORIGINS contains '*'. Set it to your real "
            "origin(s), or set ALLOW_INSECURE_PUBLIC=1 to deliberately allow a "
            "wildcard origin."
        )

    # Redis unreachable at startup must NOT crash the app — health/RAG/Sheets-sync
    # can still function. The calling path (campaign_tick / worker) checks Redis
    # itself and hard-refuses to enqueue rather than silently losing leads.
    redis_ok = await redis_client.is_available()
    if not redis_ok:
        logger.warning(
            "[startup] Redis is unreachable — dialling/queueing is disabled until "
            "it recovers. Health, RAG, and Sheets-sync are unaffected."
        )

    # SCHEDULER_ENABLED must be true on exactly ONE running instance — see
    # config.py's note and the blueprint's own "one scheduler, one instance"
    # warning; APScheduler jobs would otherwise fire twice. Also skipped
    # outright when Redis is down: campaign_tick would just enqueue leads
    # nothing can dequeue.
    if SCHEDULER_ENABLED and redis_ok:
        scheduler.start()
    elif SCHEDULER_ENABLED:
        logger.warning("[startup] SCHEDULER_ENABLED but Redis is down — not starting it.")

    # Rebuild the dial-time DND set NOW rather than waiting for dnd_refresh's
    # 06:00 cron. CLAUDE.md calls Redis ephemeral and safe to flush, but the
    # DND set was written by that cron alone — so a flush, a recreated volume,
    # or a fresh managed Redis left the worker's SISMEMBER scrub matching
    # nothing at all, silently, for up to 20 hours. An empty set is
    # indistinguishable from "nobody has opted out". This makes the set
    # genuinely rebuildable, which is what "safe to flush" has to mean.
    if redis_ok:
        try:
            await scheduler.dnd_refresh()
        except Exception:
            logger.exception(
                "[startup] could not rebuild the DND set — the dial-time scrub "
                "may be incomplete until dnd_refresh next runs"
            )

    # The call worker: a background asyncio task in this same process (see
    # config.WORKER_ENABLED) — nothing else consumes calls:queue, so without
    # this, campaign_tick enqueues leads that never get dialled.
    worker_stop_event: asyncio.Event | None = None
    worker_task: asyncio.Task | None = None
    if WORKER_ENABLED and redis_ok:
        worker_stop_event = asyncio.Event()
        worker_task = asyncio.create_task(worker.run_worker(worker_stop_event))
    elif WORKER_ENABLED:
        logger.warning("[startup] WORKER_ENABLED but Redis is down — "
                       "the call worker is not starting.")

    yield

    if worker_task is not None:
        worker_stop_event.set()
        try:
            # Bounded: a worker wedged on an unresponsive Redis must not hang
            # shutdown forever. BLMOVE's own timeout is 2s, so this only trips
            # if something is genuinely stuck.
            await asyncio.wait_for(worker_task, timeout=10)
        except asyncio.TimeoutError:
            logger.warning("[shutdown] call worker did not stop in 10s — cancelling it.")
            worker_task.cancel()
        except Exception:
            # The worker died earlier (its own loop guard should prevent this).
            # Awaiting a task that failed re-raises here, which would abort the
            # rest of shutdown and leak the scheduler and the Redis pool.
            logger.exception("[shutdown] call worker had already failed.")
    scheduler.shutdown()
    await redis_client.close()


app = FastAPI(
    title="Voice Agent — Digital Brolly (ElevenLabs)",
    description="Outbound AI voice caller: cloned rep voice, EN/TE/Tinglish, "
                 "one-way + two-way calls with RAG over course docs.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
from app.admin import router as admin_router  # noqa: E402
from app.rag.endpoint import router as rag_router  # noqa: E402
from app.telephony.call_routes import router as call_router  # noqa: E402
from app.webhooks.elevenlabs import router as elevenlabs_webhook_router  # noqa: E402

app.include_router(rag_router)
app.include_router(elevenlabs_webhook_router)
app.include_router(admin_router)
app.include_router(call_router)


@app.get("/health", tags=["Health"])
def health():
    """Liveness check — used by the Dockerfile HEALTHCHECK and docker-compose."""
    return {"status": "ok"}


if __name__ == "__main__":  # pragma: no cover - manual dev entrypoint
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
