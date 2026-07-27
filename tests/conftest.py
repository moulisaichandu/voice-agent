"""Shared test fixtures + a deterministic, OFFLINE test environment.

Env vars are set at import time — BEFORE any app module (app.config) is first
imported — because config reads os.environ once at import. Every external
credential is blanked so a test can never make a real network / telephony /
Sheets call by accident. Mirrors ai-voice-agent/backend/tests/conftest.py.

DATABASE_URL defaults to the docker-compose local pgvector service (same value
as .env.example) — good enough for app.main's fail-fast "is it set" check on
unit tests that only need the app to boot, AND it's the real connection
`@pytest.mark.integration` tests need. Run `docker compose --profile dev up -d
postgres redis` before running integration tests; unit tests need neither
container running.
"""

import os

os.environ.setdefault(
    "DATABASE_URL", "postgresql://voiceagent:voiceagent@localhost:5432/voiceagent"
)
os.environ["REDIS_URL"] = "redis://localhost:6379/15"  # a dedicated test DB index
os.environ["PUBLIC_BASE_URL"] = ""
os.environ["ALLOW_INSECURE_PUBLIC"] = "0"
os.environ["ALLOWED_ORIGINS"] = "http://localhost:8000"
os.environ["APP_AUTH_TOKEN"] = ""
os.environ["SCHEDULER_ENABLED"] = "false"
# Same reasoning as SCHEDULER_ENABLED: a handful of unit tests boot the real
# app via TestClient (test_health.py, test_startup_checks.py, ...). If a real
# local Redis happens to be reachable, an enabled worker would start a
# background BLMOVE loop with up to a ~2s shutdown grace period per boot —
# slow and pointless for tests that never enqueue anything. The worker's own
# behavior (run_worker, process_one, the Redis primitives) is exercised
# directly by tests/unit/test_worker_*.py and tests/integration/test_pipeline.py,
# neither of which goes through app.main's lifespan.
os.environ["WORKER_ENABLED"] = "false"
for _k in (
    "ELEVENLABS_API_KEY", "ELEVENLABS_WEBHOOK_SECRET",
    "ELEVENLABS_ONEWAY_AGENT_ID", "ELEVENLABS_TWOWAY_AGENT_ID",
    "ELEVENLABS_AGENT_PHONE_NUMBER_ID",
    "OPENAI_API_KEY", "SARVAM_API_KEY", "GOOGLE_SHEET_ID",
):
    os.environ[_k] = ""

import pytest  # noqa: E402


@pytest.fixture
def client():
    """A TestClient over the real app. Redis need not be running — the lifespan
    logs a warning and continues rather than crashing when Redis is unreachable."""
    from fastapi.testclient import TestClient

    from app.main import app
    with TestClient(app) as c:
        yield c
