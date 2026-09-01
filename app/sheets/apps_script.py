"""sheets/apps_script.py — the Apps Script Web-App transport.

WHY THIS EXISTS. Discovered live 2026-08-27: this deployment's
GOOGLE_SHEET_ID held not a spreadsheet ID but the /exec URL of a deployed
"Voice Agent Transcript Logger" — the sibling ../ai-voice-agent project's
Code.gs, already bound to the operator's actual sheet. That web app IS the
operator's sheet infrastructure: it appends transcripts to a "Transcripts"
tab, reads lead rows from a "Leads" tab, and writes a lead's Status back,
with no service account and no sharing step. Demanding a bare sheet ID plus
a shared service account would mean re-plumbing something that already
works, so when the configured value is a script URL the sheets layer speaks
the script's own POST protocol instead. gspread remains the transport for a
real spreadsheet ID.

The protocol shapes are taken from the sibling's Code.gs (READ-ONLY
reference, per CLAUDE.md): every request is a JSON POST of
{"action": ..., "token": ...?, ...}; every response is {"status": "ok"|
"error", ...}. Apps Script 302-redirects POSTs to the execution host, so
redirects must be followed.

SECURITY NOTE, verified live: the deployed script currently has no
SHEETS_TOKEN script property, which per Code.gs leaves the webhook OPEN —
anyone with the URL can append rows. This module sends the token whenever
config.SHEETS_TOKEN is set, and omits the field entirely when it is not (an
empty-string token would FAIL a script whose property is set). Set the same
value on both sides once the integration is proven.

Failure contract, per helper: export_transcript and update_lead_status
degrade to False with an ERROR log (the write-back queue provides the
retry, and only export_transcript's False should requeue — export_session
appends blindly, so a retry after a successful export would duplicate the
whole transcript in the sheet). fetch_rows RAISES instead: its only caller
is the scheduler's sheets_sync job, which records the failure into
/admin/sheets-status's last_error — degrading to [] there made a hard-down
transport indistinguishable from a healthy empty sheet.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from app.config import GOOGLE_SHEET_ID, LEADS_WORKSHEET_NAME, SHEETS_TOKEN

logger = logging.getLogger(__name__)

_TIMEOUT_S = 20.0


def is_apps_script(value: str | None) -> bool:
    """Whether *value* selects this transport rather than gspread."""
    if not value:
        return False
    lowered = value.lower()
    return (lowered.startswith(("http://", "https://"))
            and "script.google.com" in lowered)


async def _http_post_json(url: str, payload: dict) -> dict:
    """One JSON POST, redirects followed, parsed body returned.

    Raises on transport errors and on a non-JSON body (a misconfigured or
    unauthorised deployment returns an HTML login page) — callers translate
    that into their degrade value.
    """
    async with httpx.AsyncClient(follow_redirects=True,
                                 timeout=_TIMEOUT_S) as client:
        resp = await client.post(url, json=payload)
        try:
            return resp.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Apps Script returned non-JSON (HTTP {resp.status_code}) — "
                "usually a deployment whose access is not 'Anyone'."
            ) from exc


# In-call retries for TRANSPORT-shaped failures only. Observed live
# 2026-08-27, twice: each container's first sweep after idle got HTTP 404
# (non-JSON) while probes seconds later succeeded — Apps Script cold-start
# flakiness the sibling's client also retries for. The sweep cadence cannot
# be the retry: 10-30 minutes later the script is cold again, so without
# this every sweep can fail forever while every manual probe passes. The
# first attempt warms the script; the retry lands. A retried POST that DID
# execute before its response was lost re-runs the action — for the append
# actions that is the documented duplicate-beats-a-loss trade. Application-
# level {"status":"error"} bodies are real responses and are never retried.
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_S = 2.0


async def _post(action: str, **fields) -> dict:
    payload: dict = {"action": action, **fields}
    if SHEETS_TOKEN:
        payload["token"] = SHEETS_TOKEN
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            return await _http_post_json(GOOGLE_SHEET_ID, payload)
        except Exception as exc:  # noqa: BLE001 - transport-shaped, retried
            if attempt == _RETRY_ATTEMPTS:
                raise
            logger.warning(
                f"[sheets] Apps Script {action} attempt {attempt} failed "
                f"({type(exc).__name__}: {exc}) — retrying in "
                f"{_RETRY_BACKOFF_S * attempt:.0f}s"
            )
            await asyncio.sleep(_RETRY_BACKOFF_S * attempt)
    raise RuntimeError("unreachable")


async def fetch_rows(worksheet_name: str) -> list[tuple[int | None, dict]]:
    """Lead rows from *worksheet_name*, as (sheet_row, row_dict) pairs.

    The script's _getLeads skips blank rows and reports each row's REAL
    1-based sheet row as ``_row`` — authoritative in a way enumerate() can
    never be, and exactly what writeback.py's stale-hint verification wants
    cached in leads.sheet_row.

    RAISES on transport failures and script-level errors, deliberately: the
    only caller is the scheduler's sheets_sync job, which catches, logs at
    ERROR and stamps the failure into /admin/sheets-status's last_error.
    Degrading to [] here made a hard-down transport look like a healthy
    empty sheet on the status endpoint (confirmed 2026-08-27).

    The MISSING-TAB case used to be the one blind spot no client code could
    fix: Code.gs answered a missing tab with status ok and leads:[], exactly
    like an empty one. Live on 2026-09-01 that was the whole reason no lead
    was ever dialled from the sheet — /admin/sheets-status read
    "synced 0, no error" for weeks while the operator's spreadsheet simply had
    no Leads tab. The script now returns a status error for it (matching what
    its own _updateLead always did), so it arrives here as a raise.
    """
    body = await _post("get_leads", sheet=worksheet_name)
    if body.get("status") != "ok":
        raise RuntimeError(f"Apps Script get_leads error: "
                           f"{body.get('message') or body}")
    rows: list[tuple[int | None, dict]] = []
    for row in body.get("leads") or []:
        if not isinstance(row, dict):
            continue
        row = dict(row)
        raw = row.pop("_row", None)
        try:
            sheet_row = int(raw) if raw is not None else None
        except (TypeError, ValueError):
            sheet_row = None
        rows.append((sheet_row, row))
    return rows


async def export_transcript(*, session: str, turns, timestamp: str) -> bool:
    """Append one call's transcript to the script's Transcripts tab.

    Roles map to the sibling's vocabulary — lead→user, agent→assistant —
    because Code.gs counts a new Turn # on role == 'user' and writes the
    role string verbatim into the sheet.
    """
    history = [
        {"role": "user" if t.role == "lead" else "assistant",
         "content": t.text}
        for t in (turns or [])
    ]
    try:
        body = await _post("export_session", session=session,
                           timestamp=timestamp, history=history)
    except Exception as exc:  # noqa: BLE001 - degrade, sweep retries
        logger.error(f"[sheets] Apps Script export_session failed: "
                     f"{type(exc).__name__}: {exc}")
        return False
    if body.get("status") == "ok":
        return True
    logger.error(f"[sheets] Apps Script export_session error: {body}")
    return False


async def update_lead_status(*, phone: str, status: str,
                             notes: str = "") -> bool:
    """Best-effort Status/Notes write to the lead's Leads-tab row.

    'Lead not found' is quiet success, exactly as the sibling treats it:
    console-added leads are not in the sheet at all, and a status cell for a
    row that does not exist is not a failure worth retrying forever.
    """
    try:
        body = await _post("update_lead", sheet=LEADS_WORKSHEET_NAME,
                           phone=phone, status=status, notes=notes)
    except Exception as exc:  # noqa: BLE001 - best-effort by contract
        logger.error(f"[sheets] Apps Script update_lead failed: "
                     f"{type(exc).__name__}: {exc}")
        return False
    if body.get("status") == "ok":
        return True
    # The exact per-lead miss ONLY. Code.gs has a second 'not found' message
    # — "Leads sheet not found." — which is a standing misconfiguration (a
    # renamed/missing tab) that kills every status write-back; a bare
    # substring match swallowed it before the ERROR log below, leaving zero
    # signal anywhere for a dead Status column.
    if str(body.get("message", "")).lower().startswith("lead not found"):
        return True
    logger.error(f"[sheets] Apps Script update_lead error: {body}")
    return False
