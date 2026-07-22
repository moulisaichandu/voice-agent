"""admin/system.py — Dial-readiness, aggregate stats, effective config, and
Sheets sync visibility.

The readiness endpoint exists because of a real, expensive lesson: every
blocker hit while bringing this project up (a dead cloudflared tunnel, the
ElevenLabs account going `past_due`, a stale PUBLIC_BASE_URL pointing at a
different project, Zentrunk never provisioned) was invisible until a call
actually failed, and each cost a diagnostic cycle. "Can this system dial
right now, and if not why" is answerable at a glance instead.

See app/admin/__init__.py for the router assembly and auth dependency shared
by every admin submodule.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable, Literal

from fastapi import APIRouter
from pydantic import BaseModel

from app import config as app_config
from app import languages, redis_client
from app import scheduler as scheduler_module
from app.compliance.calling_hours import within_calling_hours
from app.config import CALLING_HOURS_END, CALLING_HOURS_START, CALLING_HOURS_TZ
from app.db import calls as calls_db
from app.db import campaigns as campaigns_db
from app.db import leads as leads_db
from app.db.pool import get_pool
from app.telephony import elevenlabs_client
from app.telephony import preflight as preflight_module
from app.telephony import worker as worker_module

logger = logging.getLogger(__name__)
router = APIRouter()

Tone = Literal["good", "warning", "serious", "critical"]

# ── Readiness ──────────────────────────────────────────────────────────────
# preflight and billing are the two EXPENSIVE checks (an HTTP round-trip to
# our own /calls/answer, an ElevenLabs agent lookup, and a separate
# ElevenLabs API call for billing) — cached in Redis so hitting this endpoint
# repeatedly (the dashboard polls it) doesn't hammer either. Every other
# check is sub-millisecond and always computed fresh.
_READINESS_CACHE_TTL_S = 60


class ReadinessCheck(BaseModel):
    key: str
    status: Tone
    label: str
    detail: str
    cached: bool
    checked_at: str


class ReadinessResponse(BaseModel):
    can_dial: bool
    checked_at: str
    checks: list[ReadinessCheck]


async def _check_database() -> dict:
    try:
        pool = await get_pool()
        await pool.fetchval("select 1")
        return {"key": "database", "status": "good", "label": "Database", "detail": "Reachable"}
    except Exception as exc:
        return {"key": "database", "status": "critical", "label": "Database",
                "detail": f"Unreachable: {type(exc).__name__}: {exc}"}


async def _check_redis() -> dict:
    ok = await redis_client.is_available()
    if ok:
        return {"key": "redis", "status": "good", "label": "Redis", "detail": "Reachable"}
    return {"key": "redis", "status": "critical", "label": "Redis", "detail": "Unreachable"}


async def _check_worker() -> dict:
    try:
        beat = await redis_client.get_redis().get(worker_module.HEARTBEAT_KEY)
    except Exception as exc:
        return {"key": "worker", "status": "critical", "label": "Call worker",
                "detail": f"Could not check: {type(exc).__name__}: {exc}"}
    if not beat:
        return {"key": "worker", "status": "critical", "label": "Call worker",
                "detail": "No heartbeat — the worker is not running (crashed, or "
                          "WORKER_ENABLED=false). Queued leads will never be dialled "
                          "until it's restarted."}
    return {"key": "worker", "status": "good", "label": "Call worker",
            "detail": f"Alive, last heartbeat {beat}"}


def _check_calling_hours() -> dict:
    window = f"{CALLING_HOURS_START}:00-{CALLING_HOURS_END}:00 {CALLING_HOURS_TZ}"
    if within_calling_hours():
        return {"key": "calling_hours", "status": "good", "label": "Calling hours",
                "detail": f"Within the {window} window"}
    return {"key": "calling_hours", "status": "warning", "label": "Calling hours",
            "detail": f"Outside the {window} window — campaign_tick will not enqueue "
                      "anything until it reopens"}


_REQUIRED_FOR_DIALING = [
    "ELEVENLABS_API_KEY", "PLIVO_AUTH_ID", "PLIVO_AUTH_TOKEN",
    "PLIVO_FROM_NUMBER", "PUBLIC_BASE_URL",
]


def _check_config() -> dict:
    # Looked up live via getattr, not imported by value: app.config's
    # constants are only ever set once at process start in the real app (see
    # CLAUDE.md's "every env var is read once" convention) — but reading them
    # live here, rather than baking values into a module-level list at import
    # time, is what lets a test simulate "what if this were unset" without
    # actually restarting the process.
    missing = [name for name in _REQUIRED_FOR_DIALING if not getattr(app_config, name, None)]
    if missing:
        return {"key": "config", "status": "critical", "label": "Configuration",
                "detail": f"Not set: {', '.join(missing)}"}
    return {"key": "config", "status": "good", "label": "Configuration",
            "detail": "All variables required to dial are set"}


async def _check_queue_depth(*, worker_ok: bool) -> dict:
    try:
        depth = await redis_client.get_redis().llen(worker_module.QUEUE_KEY)
    except Exception as exc:
        return {"key": "queue_depth", "status": "critical", "label": "Queue depth",
                "detail": f"Could not check: {type(exc).__name__}: {exc}"}
    if depth > 0 and not worker_ok:
        return {"key": "queue_depth", "status": "warning", "label": "Queue depth",
                "detail": f"{depth} lead(s) waiting, but the worker isn't running to "
                          "dequeue them"}
    return {"key": "queue_depth", "status": "good", "label": "Queue depth",
            "detail": f"{depth} lead(s) waiting"}


async def _compute_preflight() -> dict:
    active = await campaigns_db.list_active_campaigns()
    if not active:
        return {"key": "preflight", "status": "warning", "label": "Preflight",
                "detail": "No active campaigns to check — create or activate one first"}
    campaign = active[0]
    # No specific lead at dashboard time, only a campaign — resolve with the
    # campaign's language alone, the same call the worker makes per-lead (see
    # worker.py's own preflight call) collapsed to its campaign-only case.
    # Without this, a campaign misconfigured for its language (e.g. pointed at
    # a v2.5 agent that can't speak Telugu) stayed green here while the dial
    # path correctly refused every call — leads then cycled
    # queued -> calling -> pending with the reason visible only in a
    # container log, exactly the diagnostic cycle this dashboard exists to
    # prevent.
    call_language = languages.iso_code(languages.resolve(None, campaign.language))
    reason = await preflight_module.preflight(campaign.agent_id, call_language)
    if reason:
        return {"key": "preflight", "status": "critical", "label": "Preflight", "detail": reason}
    return {"key": "preflight", "status": "good", "label": "Preflight",
            "detail": f"Passing (checked against {campaign.name!r})"}


_GOOD_SUBSCRIPTION_STATUSES = {"active", "trialing"}


async def _compute_billing() -> dict:
    try:
        sub = await asyncio.to_thread(elevenlabs_client.subscription_status)
    except Exception as exc:
        return {"key": "billing", "status": "critical", "label": "ElevenLabs billing",
                "detail": f"Could not check: {type(exc).__name__}: {exc}"}
    status = sub["status"]
    if status in _GOOD_SUBSCRIPTION_STATUSES:
        usage = ""
        if sub.get("character_limit"):
            usage = f" ({sub['character_count']}/{sub['character_limit']} characters used)"
        return {"key": "billing", "status": "good", "label": "ElevenLabs billing",
                "detail": f"{status}{usage}"}
    return {"key": "billing", "status": "critical", "label": "ElevenLabs billing",
            "detail": f"Account status is {status!r} — no ElevenLabs conversation can run "
                      "until this is resolved (the agent WebSocket refuses the handshake "
                      "with a payment-issue error in this state)."}


async def _cached_check(key: str, compute: Callable[[], Awaitable[dict]]) -> dict:
    r = redis_client.get_redis()
    try:
        raw = await r.get(f"readiness:{key}")
    except Exception:
        raw = None
    if raw:
        result = json.loads(raw)
        result["cached"] = True
        return result
    return await _refresh_check(key, compute)


async def _refresh_check(key: str, compute: Callable[[], Awaitable[dict]]) -> dict:
    r = redis_client.get_redis()
    try:
        result = await compute()
    except Exception as exc:
        result = {"key": key, "status": "critical", "label": key.replace("_", " ").title(),
                  "detail": f"Check failed: {type(exc).__name__}: {exc}"}
    result["cached"] = False
    result["checked_at"] = datetime.now(timezone.utc).isoformat()
    try:
        await r.set(f"readiness:{key}", json.dumps(result), ex=_READINESS_CACHE_TTL_S)
    except Exception:
        pass  # best-effort cache; _check_redis() reports an unreachable Redis separately
    return result


async def _build_readiness(*, force_refresh: bool) -> ReadinessResponse:
    now_iso = datetime.now(timezone.utc).isoformat()

    db_check = await _check_database()
    redis_check = await _check_redis()
    worker_check = await _check_worker()
    hours_check = _check_calling_hours()
    config_check = _check_config()
    queue_check = await _check_queue_depth(worker_ok=worker_check["status"] == "good")

    cheap = [db_check, redis_check, worker_check, hours_check, config_check, queue_check]
    for c in cheap:
        c["cached"] = False
        c["checked_at"] = now_iso

    if force_refresh:
        preflight_check = await _refresh_check("preflight", _compute_preflight)
        billing_check = await _refresh_check("billing", _compute_billing)
    else:
        preflight_check = await _cached_check("preflight", _compute_preflight)
        billing_check = await _cached_check("billing", _compute_billing)

    checks = [ReadinessCheck(**c) for c in [*cheap, preflight_check, billing_check]]
    can_dial = not any(c.status in ("warning", "serious", "critical") for c in checks)
    return ReadinessResponse(can_dial=can_dial, checked_at=now_iso, checks=checks)


@router.get("/readiness", response_model=ReadinessResponse)
async def get_readiness() -> ReadinessResponse:
    return await _build_readiness(force_refresh=False)


@router.post("/readiness/refresh", response_model=ReadinessResponse)
async def refresh_readiness() -> ReadinessResponse:
    """Bypasses the 60s cache on the two expensive checks — for "I just fixed
    it, tell me right now" rather than waiting out the TTL."""
    return await _build_readiness(force_refresh=True)


# ── Stats ──────────────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    leads_by_status: dict[str, int]
    calls_today: int
    calls_today_with_transcript: int


@router.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """Scoped to active campaigns only — see
    lead_status_counts_for_active_campaigns()'s docstring for why: this dev
    database's stale test-*/pipeline-* campaigns would otherwise dominate
    every number here."""
    since_midnight_utc = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    status_counts, call_counts = await asyncio.gather(
        leads_db.lead_status_counts_for_active_campaigns(),
        calls_db.call_counts_since(since_midnight_utc, active_campaigns_only=True),
    )
    return StatsResponse(
        leads_by_status=status_counts,
        calls_today=call_counts["total"],
        calls_today_with_transcript=call_counts["with_transcript"],
    )


# ── Config (names/booleans only — see module docstring) ─────────────────────

# Anything that could identify or locate the deployment, or that IS a
# secret, is reported as presence-only: {name, is_set}, never the value.
# Three API keys and an account password were pasted into chat over the
# course of this project already; this console must not become a fourth leak.
_PRESENCE_ONLY = [
    "DATABASE_URL", "REDIS_URL",
    "PLIVO_AUTH_ID", "PLIVO_AUTH_TOKEN", "PLIVO_FROM_NUMBER", "CALL_WEBHOOK_SECRET",
    "ELEVENLABS_API_KEY", "ELEVENLABS_WEBHOOK_SECRET",
    "ELEVENLABS_ONEWAY_AGENT_ID", "ELEVENLABS_TWOWAY_AGENT_ID",
    "OPENAI_API_KEY", "GOOGLE_SERVICE_ACCOUNT_FILE", "GOOGLE_SHEET_ID",
    "PUBLIC_BASE_URL", "APP_AUTH_TOKEN", "RAG_TOOL_SECRET",
]

# Tuning knobs with no secrecy or identifying value — genuinely useful to see
# at a glance ("what is it actually running with?") without shelling into
# the container. Looked up live via getattr (see _check_config's note above)
# rather than by value, so these stay correct if config is ever reloaded and
# so a test can simulate a different value without a process restart.
_VALUE_VAR_NAMES = [
    "CALLING_HOURS_START", "CALLING_HOURS_END", "CALLING_HOURS_TZ",
    "MAX_CONCURRENT_CALLS", "CAMPAIGN_TICK_SECONDS", "RAG_MIN_SCORE",
    "SCHEDULER_ENABLED", "WORKER_ENABLED", "ALLOW_INSECURE_PUBLIC",
]


class ConfigVar(BaseModel):
    name: str
    is_set: bool
    value: str | int | float | bool | None = None


class ConfigResponse(BaseModel):
    variables: list[ConfigVar]


@router.get("/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    presence = [
        ConfigVar(name=name, is_set=bool(getattr(app_config, name, None)))
        for name in _PRESENCE_ONLY
    ]
    valued = [
        ConfigVar(name=name, is_set=True, value=getattr(app_config, name))
        for name in _VALUE_VAR_NAMES
    ]
    return ConfigResponse(variables=[*presence, *valued])


# ── Sheets sync visibility ───────────────────────────────────────────────────

class SheetsStatusResponse(BaseModel):
    configured: bool
    last_sync_at: str | None = None
    last_synced_count: int | None = None
    last_error: str | None = None


@router.get("/sheets-status", response_model=SheetsStatusResponse)
async def sheets_status() -> SheetsStatusResponse:
    configured = bool(getattr(app_config, "GOOGLE_SHEET_ID", None))
    try:
        raw = await redis_client.get_redis().get(scheduler_module.SHEETS_LAST_SYNC_KEY)
    except Exception:
        raw = None
    if not raw:
        return SheetsStatusResponse(configured=configured)
    data = json.loads(raw)
    return SheetsStatusResponse(
        configured=configured,
        last_sync_at=data.get("at"),
        last_synced_count=data.get("synced"),
        last_error=data.get("error"),
    )
