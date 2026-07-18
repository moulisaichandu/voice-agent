"""
main.py — FastAPI application entry point.

Run (dev):   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
Run (prod):  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

Single worker only: APScheduler (app/scheduler.py) runs in-process and must not
be started twice — see config.SCHEDULER_ENABLED and the blueprint's own
"one scheduler, one instance" warning.
"""

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
    DATABASE_URL,
    ELEVENLABS_WEBHOOK_SECRET,
    PUBLIC_BASE_URL,
    SCHEDULER_ENABLED,
)

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

    # A wildcard CORS origin combined with a real auth token lets any website
    # attach an Authorization header to a cross-origin request against this API.
    if APP_AUTH_TOKEN and "*" in ALLOWED_ORIGINS and not ALLOW_INSECURE_PUBLIC:
        raise RuntimeError(
            "Refusing to start: ALLOWED_ORIGINS contains '*' while APP_AUTH_TOKEN is "
            "set. Set ALLOWED_ORIGINS to your real origin(s), or set "
            "ALLOW_INSECURE_PUBLIC=1 to deliberately allow a wildcard origin."
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

    yield

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
from app.rag.endpoint import router as rag_router  # noqa: E402
from app.webhooks.elevenlabs import router as elevenlabs_webhook_router  # noqa: E402

app.include_router(rag_router)
app.include_router(elevenlabs_webhook_router)


@app.get("/health", tags=["Health"])
def health():
    """Liveness check — used by the Dockerfile HEALTHCHECK and docker-compose."""
    return {"status": "ok"}


if __name__ == "__main__":  # pragma: no cover - manual dev entrypoint
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
