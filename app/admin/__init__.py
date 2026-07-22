"""app/admin/ — Minimal admin/test API: campaigns, leads, calls, and system
status (readiness, stats, config, Sheets sync).

Built to support a local test console (create a campaign, add leads, see
calls/transcripts, and tell at a glance whether the system can actually dial
right now) without hand-editing Supabase or waiting on a real Google Sheet —
NOT a full production admin surface. Every write goes through the exact same
validated paths the rest of the app uses (create_campaign's AI-disclosure/
script checks, upsert_lead's normal upsert-on-phone+campaign semantics,
campaign_tick's compliance gates, mark_dnd's existing DND propagation) —
there is no bypass of any hard rule anywhere in this package.

Protected by APP_AUTH_TOKEN when set (see require_admin_auth) — open by
default in local dev, matching .env.example's blank default. This makes
APP_AUTH_TOKEN load-bearing; previously it was read in app/config.py but only
consulted by app.main's CORS-wildcard fail-fast check, never actually
enforced on a request.

Split into one module per concern (campaigns.py, leads.py, calls.py,
system.py) rather than one growing file — each defines its own unprefixed
APIRouter, assembled here under the shared "/admin" prefix and auth
dependency via include_router(), which is FastAPI's own recommended pattern
for this rather than a hand-rolled registration scheme.
"""

from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, Header, HTTPException

from app.config import APP_AUTH_TOKEN


async def require_admin_auth(authorization: str | None = Header(default=None)) -> None:
    """Bearer-token gate on every admin route.

    Open when APP_AUTH_TOKEN is unset, matching the local-dev default — but
    app/main.py now refuses to boot in that state behind a PUBLIC_BASE_URL, so
    it can only happen on a non-public server.

    compare_digest rather than `!=`: str equality short-circuits on the first
    differing byte and so leaks the token prefix by timing. app/telephony/
    call_routes.py already did this correctly for the sibling secret; only
    this path was missed. compare_digest raises TypeError on non-ASCII, so the
    header is encoded first — a non-ASCII header must be a clean 401, not a
    500 that doubles as an oracle for the token being ASCII."""
    if not APP_AUTH_TOKEN:
        return
    expected = f"Bearer {APP_AUTH_TOKEN}".encode()
    supplied = (authorization or "").encode("utf-8", errors="replace")
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="unauthorized")


router = APIRouter(prefix="/admin", tags=["Admin"],
                   dependencies=[Depends(require_admin_auth)])

from app.admin.calls import router as _calls_router  # noqa: E402
from app.admin.campaigns import router as _campaigns_router  # noqa: E402
from app.admin.leads import router as _leads_router  # noqa: E402
from app.admin.system import router as _system_router  # noqa: E402

router.include_router(_campaigns_router)
router.include_router(_leads_router)
router.include_router(_calls_router)
router.include_router(_system_router)
