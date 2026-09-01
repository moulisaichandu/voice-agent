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

import httpx
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
from app.sheets import client as sheets_client
from app.telephony import elevenlabs_client, public_url, sarvam_circuit_breaker
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


_LAST_PUBLIC_URL_KEY = "public_url:last_seen"
# Short: this runs on every dashboard poll, and a tunnel that needs longer than
# this to answer is not one Plivo will wait for either.
_PUBLIC_URL_TIMEOUT_S = 8.0


async def _check_public_url() -> dict:
    """Whether the public URL RESOLVES AND ANSWERS, and whether it has changed.

    Reachability first, because that is the property that matters: if Plivo
    cannot fetch the answer URL, every call rings and drops the moment it is
    answered. Critical, not a warning.

    This check originally compared hostnames only, so it noticed ROTATION and
    nothing else. On 2026-08-20 Cloudflare tore down the quick tunnel's record
    while cloudflared stayed "Up" and went on serving the dead hostname from
    /quicktunnel for hours — resolving nowhere, from the host or the container.
    This check reported "Stable at https://…" throughout. Only preflight's real
    round-trip caught it. A name that has not changed is not evidence of a
    tunnel that works.

    Rotation, when the new URL DOES answer, stays a warning: dialling genuinely
    still works because this app re-resolves at dial time. What breaks is
    everything configured OUTSIDE this repo against the old hostname — the
    ElevenLabs agent's search-tool URL and its transcript webhook — and nothing
    here can repair those. The real fix is a named Cloudflare tunnel; see the
    CLOUDFLARE_TUNNEL_TOKEN block in docker-compose.yml.
    """
    try:
        current = await public_url.refresh()
    except Exception as exc:  # noqa: BLE001 - one dead check must not take the dashboard down
        return {"key": "public_url", "status": "critical", "label": "Public URL",
                "detail": f"Could not resolve the tunnel: {type(exc).__name__}: {exc}"}
    if not current:
        return {"key": "public_url", "status": "critical", "label": "Public URL",
                "detail": "No public URL is resolvable — Plivo cannot reach this "
                          "app, so every call would ring and drop on answer."}

    # Does it actually answer? /health is unauthenticated and cheap by design.
    try:
        async with httpx.AsyncClient(timeout=_PUBLIC_URL_TIMEOUT_S) as client:
            response = await client.get(f"{current.rstrip('/')}/health")
        reachable = response.status_code == 200
        why = f"HTTP {response.status_code}"
    except Exception as exc:  # noqa: BLE001 - any failure to reach it is the finding
        reachable = False
        why = f"{type(exc).__name__}"

    if not reachable:
        return {
            "key": "public_url", "status": "critical", "label": "Public URL",
            "detail": (
                f"{current} is unreachable ({why}). Plivo cannot fetch the "
                "answer URL, so every call would ring and drop on answer. If "
                "this is a quick tunnel, cloudflared can stay 'Up' and keep "
                "serving a hostname Cloudflare has already torn down — restart "
                "the cloudflared service, then re-point the ElevenLabs agent's "
                "tool URL and webhook at the new hostname."
            ),
        }

    try:
        r = redis_client.get_redis()
        previous = await r.get(_LAST_PUBLIC_URL_KEY)
        await r.set(_LAST_PUBLIC_URL_KEY, current)
    except Exception as exc:  # noqa: BLE001
        return {"key": "public_url", "status": "warning", "label": "Public URL",
                "detail": f"Reachable at {current}, but could not check for "
                          f"rotation: {type(exc).__name__}: {exc}"}

    if previous and previous != current:
        return {
            "key": "public_url", "status": "warning", "label": "Public URL",
            "detail": (
                f"The tunnel rotated to {current} (was {previous}) and the new "
                "one answers, so dialling is unaffected — this app re-resolves "
                "it. But anything configured against the OLD hostname is now "
                "dead, and nothing here can update it: in the ElevenLabs "
                "dashboard, re-point the agent's search-tool URL and the "
                "transcript webhook. A named Cloudflare tunnel would stop this "
                "recurring."
            ),
        }
    return {"key": "public_url", "status": "good", "label": "Public URL",
            "detail": f"Reachable and stable at {current}"}


async def _check_sarvam_circuit_breaker() -> dict:
    """Whether a Sarvam credits failure has paused dialling on that backend.

    Cheap by design — one Redis read, unlike billing's real API round-trip —
    so unlike that check this needs no caching of its own.

    A read failure reports critical rather than "not tripped". This endpoint
    exists to answer "can this system dial right now, and if not why", and
    "I could not tell" is the honest answer to that; silently reading as
    healthy is how the original problem stayed invisible until a real call
    failed.
    """
    try:
        reason = await sarvam_circuit_breaker.tripped_reason()
    except Exception as exc:  # noqa: BLE001 - one dead check must not take the dashboard down
        return {"key": "sarvam_circuit_breaker", "status": "critical",
                "label": "Sarvam circuit breaker",
                "detail": f"Could not check: {type(exc).__name__}: {exc}"}
    if reason:
        return {"key": "sarvam_circuit_breaker", "status": "critical",
                "label": "Sarvam circuit breaker",
                "detail": f"Tripped: {reason}. Check credits at "
                          "dashboard.sarvam.ai/usage, then clear it: POST "
                          "/admin/system/sarvam-circuit-breaker/clear."}
    return {"key": "sarvam_circuit_breaker", "status": "good",
            "label": "Sarvam circuit breaker", "detail": "Not tripped"}


_REQUIRED_FOR_DIALING = [
    "PLIVO_AUTH_ID", "PLIVO_AUTH_TOKEN",
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


# Bound on how many DISTINCT (agent_id, language, mode) configurations are
# actually round-tripped to preflight() per computation. preflight() costs a
# real HTTP round-trip to our own /calls/answer plus, for a languaged
# campaign, a synchronous ElevenLabs API call — cheap for a handful of active
# campaigns, but a dev database that has had the integration suite run
# against it can carry hundreds of active test-*/pipeline-* campaigns (see
# app/admin/campaigns.py's list_campaigns docstring), and checking every one
# individually would make this 60s-cached check visibly slow to compute.
# Looked up live (module global, not a frozen import) so a test can lower it
# without a process restart — same convention as _REQUIRED_FOR_DIALING above.
_MAX_PREFLIGHT_CONFIGURATIONS = 20


async def _compute_preflight() -> dict:
    active = await campaigns_db.list_active_campaigns()
    if not active:
        return {"key": "preflight", "status": "warning", "label": "Preflight",
                "detail": "No active campaigns to check — create or activate one first"}

    # No specific lead at dashboard time, only a campaign — resolve with the
    # campaign's language alone, the same call the worker makes per-lead (see
    # worker.py's own preflight call) collapsed to its campaign-only case.
    # Without this, a campaign misconfigured for its language (e.g. pointed at
    # a v2.5 agent that can't speak Telugu) stayed green here while the dial
    # path correctly refused every call — leads then cycled
    # queued -> calling -> pending with the reason visible only in a
    # container log, exactly the diagnostic cycle this dashboard exists to
    # prevent.
    #
    # preflight()'s result depends ONLY on (agent_id, resolved ISO language,
    # mode) — many active campaigns commonly share all three (see the cap's
    # own comment above), so every campaign is grouped onto the ONE
    # preflight() call that answers for all of them, instead of repeating an
    # expensive round-trip once per campaign. This also fixes the original
    # defect: checking active[0] alone let one arbitrary campaign speak for
    # every other active campaign, whether or not they shared its
    # configuration.
    groups: dict[tuple[str, str | None, str | None], list] = {}
    for c in active:
        call_language = languages.iso_code(languages.resolve(None, c.language))
        key = (c.agent_id, call_language, c.mode)
        groups.setdefault(key, []).append(c)

    # Check the configurations covering the MOST campaigns first, so that if
    # the cap below is hit, the combinations affecting the fewest campaigns
    # are the ones skipped — not an arbitrary/insertion-order subset.
    ordered_keys = sorted(groups, key=lambda k: len(groups[k]), reverse=True)
    checked_keys = ordered_keys[:_MAX_PREFLIGHT_CONFIGURATIONS]
    skipped_keys = ordered_keys[_MAX_PREFLIGHT_CONFIGURATIONS:]
    skipped_campaign_count = sum(len(groups[k]) for k in skipped_keys)

    passing: list = []
    failing: list[tuple[object, str]] = []
    # Which DISTINCT (agent_id, language, mode) configurations produced each
    # reason — the only evidence this function has for telling "infrastructure
    # is down" apart from "this campaign's configuration is wrong". A reason
    # is checked once per configuration (the grouping above), so if a reason
    # only ever came from ONE configuration — whether that's because there's
    # only one failing campaign, or because many failing campaigns all share
    # one identical setup — there is nothing here that couldn't be explained
    # by that single configuration being wrong. Only a reason that recurs
    # across configurations that DIFFER rules that out.
    configs_by_reason: dict[str, set[tuple[str, str | None, str | None]]] = {}
    for key in checked_keys:
        agent_id, call_language, mode = key
        reason = await preflight_module.preflight(agent_id, call_language, mode=mode)
        campaigns_in_group = groups[key]
        if reason:
            failing.extend((c, reason) for c in campaigns_in_group)
            configs_by_reason.setdefault(reason, set()).add(key)
        else:
            passing.extend(campaigns_in_group)

    checked_total = len(passing) + len(failing)
    truncation_note = ""
    if skipped_campaign_count:
        truncation_note = (
            f" ({skipped_campaign_count} other active campaign(s) across "
            f"{len(skipped_keys)} more configuration(s) were NOT checked this "
            f"run — capped at {_MAX_PREFLIGHT_CONFIGURATIONS} distinct "
            "configurations per refresh. Deactivate unused campaigns, or wait "
            "for a later refresh, to confirm the rest.)"
        )

    if not failing:
        if skipped_campaign_count:
            return {"key": "preflight", "status": "warning", "label": "Preflight",
                    "detail": f"All {checked_total} checked active campaign(s) "
                              f"passing.{truncation_note}"}
        return {"key": "preflight", "status": "good", "label": "Preflight",
                "detail": f"All {checked_total} active campaign(s) passing preflight."}

    # Group failures by their exact reason text: several campaigns sharing a
    # configuration always share a reason (they were checked together above),
    # but distinct configurations can still fail identically — e.g. every
    # campaign refused by the same dead tunnel. Naming that reason once,
    # against every campaign it affects, is what tells "infrastructure is
    # down" apart from "campaign X in particular is misconfigured".
    by_reason: dict[str, list[str]] = {}
    for c, reason in failing:
        by_reason.setdefault(reason, []).append(c.name)
    blocked_desc = "; ".join(
        f"{', '.join(repr(n) for n in names)} — {reason}" for reason, names in by_reason.items()
    )

    if not passing:
        # NONE pass. Do NOT infer scope from the count alone — "all N
        # campaigns are blocked" is exactly the same observation whether N is
        # 1, or N campaigns that all happen to share one identical
        # configuration. Neither case is evidence of anything beyond "this
        # one configuration is wrong". The only real evidence of a shared/
        # infrastructure cause is a reason that recurs across configurations
        # that DIFFER — if the same reason blocks two-plus DISTINCT
        # (agent_id, language, mode) combinations, no single campaign's setup
        # can explain that, so it's safe to call it infrastructure. Otherwise
        # say what's actually known: these campaigns, this reason — no claim
        # about the cause.
        shared_reason = None
        if len(by_reason) == 1:
            only_reason = next(iter(by_reason))
            if len(configs_by_reason[only_reason]) >= 2:
                shared_reason = only_reason

        if shared_reason is not None:
            distinct_configs = len(configs_by_reason[shared_reason])
            detail = (f"All {checked_total} active campaign(s) are blocked by the SAME "
                      f"reason across {distinct_configs} different configurations — this "
                      f"looks like an infrastructure problem, not a per-campaign one: "
                      f"{shared_reason}{truncation_note}")
        else:
            detail = f"All {checked_total} active campaign(s) are blocked. {blocked_desc}" \
                      f"{truncation_note}"
        return {"key": "preflight", "status": "critical", "label": "Preflight", "detail": detail}

    # SOME pass, some fail — the operator's actual reported case. Naming the
    # blocked campaigns and reasons, AND how many can still dial, is what
    # makes "my dialer is broken" and "one campaign needs attention"
    # distinguishable at a glance instead of both reading as "critical".
    detail = (f"{len(passing)} of {checked_total} active campaign(s) can dial right now. "
              f"Blocked: {blocked_desc}{truncation_note}")
    return {"key": "preflight", "status": "warning", "label": "Preflight", "detail": detail}


async def _compute_billing(*, force_refresh: bool = False) -> dict:
    """Format the shared account status into a ReadinessCheck.

    The fetching and caching moved to elevenlabs_client.cached_subscription_status
    so preflight.py can share this exact cache entry — see that function. This
    is now only the presentation half; the dashboard output is unchanged.
    """
    if not app_config.ELEVENLABS_API_KEY:
        # Sarvam Telugu and OpenAI Realtime campaigns do not use ElevenLabs.
        # Only inspect billing when an active campaign actually routes there;
        # campaign-specific preflight still blocks that campaign honestly.
        try:
            active = await campaigns_db.list_active_campaigns()
        except Exception:
            active = []
        uses_elevenlabs = any(
            languages.backend_for(c.language) == languages.ELEVENLABS for c in active
        )
        if not uses_elevenlabs:
            return {
                "key": "billing", "status": "good", "label": "ElevenLabs billing",
                "detail": "Not configured; only required for ElevenLabs campaigns",
            }
    try:
        sub = await elevenlabs_client.cached_subscription_status(
            force_refresh=force_refresh)
    except Exception as exc:
        return {"key": "billing", "status": "critical", "label": "ElevenLabs billing",
                "detail": f"Could not check: {type(exc).__name__}: {exc}"}
    status = sub["status"]
    if status in elevenlabs_client.GOOD_SUBSCRIPTION_STATUSES:
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
        try:
            result = json.loads(raw)
        except (TypeError, ValueError):
            # A partial write or stale schema must not turn the dashboard
            # into a 500. Treat corrupted cache data as a cache miss.
            logger.warning("Ignoring malformed readiness cache for %s", key)
        else:
            if isinstance(result, dict):
                try:
                    ReadinessCheck(**{**result, "cached": True})
                except Exception:
                    logger.warning("Ignoring invalid readiness cache for %s", key)
                else:
                    result["cached"] = True
                    return result
            else:
                logger.warning("Ignoring non-object readiness cache for %s", key)
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


# can_dial-gating exceptions — see the comment inline at can_dial's
# computation in _build_readiness() for the full reasoning. Module-level (not
# inline in the set literal) so it reads as a considered, named policy rather
# than a throwaway expression, matching _REQUIRED_FOR_DIALING's convention.
_CAN_DIAL_IGNORED_WARNING_KEYS = {"calling_hours", "preflight", "public_url"}


async def _build_readiness(*, force_refresh: bool) -> ReadinessResponse:
    now_iso = datetime.now(timezone.utc).isoformat()

    db_check = await _check_database()
    redis_check = await _check_redis()
    worker_check = await _check_worker()
    hours_check = _check_calling_hours()
    config_check = _check_config()
    queue_check = await _check_queue_depth(worker_ok=worker_check["status"] == "good")

    breaker_check = await _check_sarvam_circuit_breaker()

    url_check = await _check_public_url()

    cheap = [db_check, redis_check, worker_check, hours_check, config_check,
             queue_check, breaker_check, url_check]
    for c in cheap:
        c["cached"] = False
        c["checked_at"] = now_iso

    if force_refresh:
        preflight_check = await _refresh_check("preflight", _compute_preflight)
        billing_check = await _refresh_check(
            "billing", lambda: _compute_billing(force_refresh=True))
    else:
        preflight_check = await _cached_check("preflight", _compute_preflight)
        billing_check = await _cached_check("billing", _compute_billing)

    checks = [ReadinessCheck(**c) for c in [*cheap, preflight_check, billing_check]]

    # can_dial means "is this system STRUCTURALLY CAPABLE of dialling" —
    # would a call go out the moment one is due — NOT "would a call be placed
    # in the next second". That's a narrower, more honest question than "did
    # every single check come back green", which is what this used to compute
    # (any warning at all flipped it false). Two real "warning" cases are NOT
    # capability problems and must not gate it:
    #
    #   - calling_hours is a warning for ~14 hours of every single day
    #     (outside 10:00-19:00 IST), which is the normal state of a healthy
    #     system overnight — not a fault: nothing is broken, dialling is just
    #     scheduled not to happen right now. can_dial stays true through that
    #     window on purpose, even though literally zero calls will be placed
    #     until it reopens — that fact is already stated once, by the
    #     calling_hours check itself ("will not dial until it reopens").
    #     can_dial repeating it by going false too would make a perfectly
    #     healthy overnight system look identical to a broken one.
    #   - preflight is a warning when SOME active campaigns pass and others
    #     don't (see _compute_preflight's "mixed" outcome) — one
    #     misconfigured campaign does not stop the system from placing calls
    #     for every OTHER active campaign, so it must not gate can_dial
    #     either. (preflight is still "critical" — and so still gates this —
    #     when NO active campaign can dial.)
    #
    # Everything else stays a hard gate: "critical" (dead database, dead
    # tunnel, missing key, an ElevenLabs account past due, every active
    # campaign refusing to dial) and "serious" (unused today, but reserved
    # for a future check that's worse than a warning and less than critical)
    # both mean a real call would actually fail right now. If a future check
    # ever needs a "warning" that SHOULD gate can_dial, that must be a
    # deliberate addition to this comment and set (_CAN_DIAL_IGNORED_WARNING_KEYS,
    # module-level above), not a side effect of adding another warning-tone
    # check elsewhere.
    can_dial = not any(
        c.status == "critical"
        or c.status == "serious"
        or (c.status == "warning" and c.key not in _CAN_DIAL_IGNORED_WARNING_KEYS)
        for c in checks
    )
    return ReadinessResponse(can_dial=can_dial, checked_at=now_iso, checks=checks)


@router.get("/readiness", response_model=ReadinessResponse)
async def get_readiness() -> ReadinessResponse:
    return await _build_readiness(force_refresh=False)


@router.post("/readiness/refresh", response_model=ReadinessResponse)
async def refresh_readiness() -> ReadinessResponse:
    """Bypasses the 60s cache on the two expensive checks — for "I just fixed
    it, tell me right now" rather than waiting out the TTL."""
    return await _build_readiness(force_refresh=True)


@router.post("/system/sarvam-circuit-breaker/clear")
async def clear_sarvam_circuit_breaker() -> dict:
    """Manual-only recovery, and the ONLY way out of a tripped breaker.

    Sarvam publishes no way to confirm credits have been topped up, so a human
    saying "this is fixed now" is the only reliable signal — an auto-expiring
    timer would either resume into the same failure or stay paused long after
    the account recovered. Mirrors PATCH /admin/campaigns/{id}
    {"active": false}, the manual stopgap this replaces.

    Clearing an untripped breaker is a no-op, so clicking it twice is safe.
    """
    await sarvam_circuit_breaker.clear()
    return {"cleared": True}


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
    # Of the rows imported, how many the dialer will actually call. Reported
    # separately because a sheet with no consent columns imports every row and
    # dials none — for weeks that was indistinguishable from a healthy sync.
    last_dialable_count: int | None = None
    last_skips: dict[str, int] | None = None


class SheetsSyncResponse(BaseModel):
    """What one hand-triggered sync did. Mirrors the record the scheduled job
    stamps into Redis, so the console shows the same numbers either way."""

    at: str
    synced: int | None = None
    received: int | None = None
    dialable: int | None = None
    skips: dict[str, int] | None = None
    error: str | None = None


@router.post("/sheets/sync", response_model=SheetsSyncResponse)
async def trigger_sheets_sync() -> SheetsSyncResponse:
    """Run the Google Sheet sync now instead of waiting for the next sweep.

    Delegates to the scheduled job itself rather than calling sheets_sync()
    directly, so the Redis status record is written in exactly one place and
    a manual run and a scheduled one can never report different things.
    That job already turns a Sheets outage into a recorded error rather than
    an exception, which is why this returns 200 with `error` set instead of
    surfacing a 5xx: the operator needs to read the reason.
    """
    record = await scheduler_module.sheets_sync()
    return SheetsSyncResponse(**record)


@router.get("/sheets-status", response_model=SheetsStatusResponse)
async def sheets_status() -> SheetsStatusResponse:
    # unconfigured_reason() is the same check the scheduler gates on, so a
    # sheet ID that is set but unusable (no service account, a URL that is
    # neither a spreadsheet nor a script) reports as unconfigured HERE rather
    # than reading as healthy until the next sweep happens to fail.
    reason = sheets_client.unconfigured_reason()
    configured = reason is None
    try:
        raw = await redis_client.get_redis().get(scheduler_module.SHEETS_LAST_SYNC_KEY)
    except Exception:
        raw = None
    if not raw:
        return SheetsStatusResponse(configured=configured, last_error=reason)
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("Ignoring malformed Sheets sync status cache")
        return SheetsStatusResponse(
            configured=configured,
            last_error="Cached Sheets status is malformed; waiting for the next sync",
        )
    if not isinstance(data, dict):
        logger.warning("Ignoring non-object Sheets sync status cache")
        return SheetsStatusResponse(
            configured=configured,
            last_error="Cached Sheets status is malformed; waiting for the next sync",
        )
    return SheetsStatusResponse(
        configured=configured,
        last_sync_at=data.get("at"),
        last_synced_count=data.get("synced"),
        last_error=data.get("error") or reason,
        last_dialable_count=data.get("dialable"),
        last_skips=data.get("skips") or None,
    )
