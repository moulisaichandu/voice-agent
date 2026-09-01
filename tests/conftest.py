"""Shared test fixtures + a deterministic, OFFLINE test environment.

Env vars are set at import time — BEFORE any app module (app.config) is first
imported — because config reads os.environ once at import. Every external
credential is blanked so a test can never make a real network / telephony /
Sheets call by accident. Mirrors ai-voice-agent/backend/tests/conftest.py.

DATABASE_URL points at a SEPARATE database — `voiceagent_test`, not the
`voiceagent` the running backend uses. Integration tests create campaigns and
leads, mark them dnd/failed/calling, and run the reaper over whatever they find;
against the live database that is indistinguishable from an operator's real
data. It is not hypothetical: on 2026-08-27 a single `pytest -q -m integration`
left 19 synthetic leads stuck in 'calling' among real ones, and an earlier run
deactivated every live campaign with a bare `update campaigns set active =
false`, which silently stopped all dialling. Redis was always isolated this way
(index 15, below); Postgres simply never was.

Create it once, alongside `docker compose --profile dev up -d postgres redis`:

    docker compose exec postgres psql -U voiceagent -d voiceagent \\
        -c "create database voiceagent_test owner voiceagent"
    docker compose exec postgres psql -U voiceagent -d voiceagent_test \\
        -c "create extension if not exists vector"
    DATABASE_URL=postgresql://voiceagent:voiceagent@localhost:5432/voiceagent_test \\
        python scripts/apply_migrations.py

Unit tests need no container at all — the value only has to satisfy app.main's
fail-fast "is it set" check. Overriding DATABASE_URL in the environment still
works, and is the one way to aim the suite somewhere else; do not aim it at a
database whose contents you would miss.
"""

import os

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql://voiceagent:voiceagent@localhost:5432/voiceagent_test",
)
os.environ["REDIS_URL"] = "redis://localhost:6379/15"  # a dedicated test DB index
os.environ["PUBLIC_BASE_URL"] = ""
os.environ["ALLOW_INSECURE_PUBLIC"] = "0"
os.environ["ALLOWED_ORIGINS"] = "http://localhost:8000"
os.environ["APP_AUTH_TOKEN"] = ""
os.environ["SCHEDULER_ENABLED"] = "false"
# The two-way enable flags must never leak in from the developer's .env:
# with SARVAM_TWOWAY_ENABLED=true on the machine, the refusal-guard test
# sailed past the very guard it exists to pin and passed for an unrelated
# reason (SarvamNotConfigured, one layer deeper) — the third layer of
# CLAUDE.md's three-layer two-way gate could have been deleted and this
# suite would have stayed green. Tests that WANT two-way monkeypatch the
# flag on explicitly (the two_way fixture does).
os.environ["SARVAM_TWOWAY_ENABLED"] = "false"
os.environ["OPENAI_TWOWAY_ENABLED"] = "false"
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
