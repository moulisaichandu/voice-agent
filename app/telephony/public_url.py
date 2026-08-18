"""telephony/public_url.py — the URL Plivo must call us back on, kept current.

Cloudflare quick tunnels mint a BRAND NEW random hostname every time
cloudflared starts. scripts/resolve_public_url.sh already resolves one at BOOT
and exports it, which is why app.config.PUBLIC_BASE_URL is usually right — but
it runs exactly once. If cloudflared restarts while the backend keeps running,
nothing re-resolves: the backend goes on handing Plivo a hostname that now
belongs to nobody, and every call RINGS and then drops the instant it is
answered, with nothing in our logs because the request never arrives.

That is not hypothetical. 22 distinct hostnames were minted on this machine,
and a stale one silently defeated a live preflight check twice in one session.

WHY THIS SHAPE. `base()` is deliberately synchronous and does no I/O, so the
existing formatting helpers (plivo_stream.answer_xml,
plivo_client._webhook_urls) can stay ordinary sync functions instead of
growing async plumbing through the dial path. `refresh()` is the async half,
and app/telephony/preflight.py awaits it once per campaign_tick batch —
immediately before any dialling — so the value those helpers read is current
exactly when it matters.

A stable hostname (a named Cloudflare tunnel, or an ngrok reserved domain)
would remove the problem at source rather than absorbing it. This is the fix
that needs no account, no domain, and no credentials.
"""

from __future__ import annotations

import logging

import httpx

from app.config import PUBLIC_BASE_URL, TUNNEL_DISCOVERY_URL

logger = logging.getLogger(__name__)

# Short: this runs inside the compose network against cloudflared's own metrics
# port, so it is a local round-trip, and a dial blocked on it is worse than a
# dial made against a slightly stale URL.
_DISCOVERY_TIMEOUT_S = 3.0

# Last successfully resolved base URL. None means "never resolved", in which
# case base() falls back to whatever the entrypoint put in the environment.
_resolved: str | None = None


def _reset_for_tests() -> None:
    """Clear the cache. Module state would otherwise let one test decide the
    next one's answer."""
    global _resolved
    _resolved = None


def base() -> str:
    """The public base URL, as last resolved. Synchronous, no I/O.

    Always a string, never None: callers build URLs by concatenation, and None
    would quietly produce 'None/calls/answer' rather than failing.
    """
    if _resolved is not None:
        return _resolved
    return PUBLIC_BASE_URL or ""


async def refresh() -> str:
    """Ask cloudflared for the hostname it is serving right now.

    Never raises, and never blanks a known-good value. It runs on the dial
    path, where an exception would abort a dial outright and an empty result
    would read as "PUBLIC_BASE_URL is not set" — both far worse than one call
    placed against a slightly stale URL.

    With TUNNEL_DISCOVERY_URL unset (production, where PUBLIC_BASE_URL is a
    real domain and no cloudflared service exists) this is a no-op that keeps
    the configured value.
    """
    global _resolved

    if not TUNNEL_DISCOVERY_URL:
        return base()

    try:
        async with httpx.AsyncClient(timeout=_DISCOVERY_TIMEOUT_S) as client:
            response = await client.get(f"{TUNNEL_DISCOVERY_URL}/quicktunnel")
        hostname = (response.json() or {}).get("hostname")
    except Exception as exc:  # noqa: BLE001 - a discovery blip must not stop dialling
        logger.debug(f"[public-url] could not reach cloudflared at "
                     f"{TUNNEL_DISCOVERY_URL}: {type(exc).__name__}: {exc}")
        return base()

    if not hostname:
        logger.debug("[public-url] cloudflared returned no hostname")
        return base()

    discovered = f"https://{hostname}"
    if discovered != base():
        logger.info(
            f"[public-url] tunnel hostname changed to {discovered} — the "
            "previous one no longer routes here, so any call placed against "
            "it would have rung and dropped on answer."
        )
    _resolved = discovered
    return _resolved
