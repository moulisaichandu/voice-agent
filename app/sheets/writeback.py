"""sheets/writeback.py — Write a call's outcome back to the lead's Sheet row.

The blueprint's own write-back sketch (§7.2) trusts a row index cached at
sync time. If a human inserts/deletes a row in the Sheet between sync and
call completion, writing straight to that cached row silently corrupts a
DIFFERENT lead's data — and it looks like a successful write, which is worse
than a failed one. Fixed here by porting ai-voice-agent/backend/sheets.py's
proven pattern: sheet_row is used only as a fast-path HINT, verified against
the phone number actually at that row before trusting it; a mismatch falls
back to a full column scan (and updates the cached hint for next time).

Called from a Redis queue (app/scheduler.py consumes sheets:writeback:queue,
pushed by app/webhooks/elevenlabs.py) rather than directly from the webhook
handler — several two-way calls finishing near-simultaneously would otherwise
risk concurrent 429s from the Sheets API.
"""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID
from zoneinfo import ZoneInfo

import gspread.utils as gsutils

from app import redis_client
from app.compliance.dnd import normalize_phone_e164
from app.config import CALLING_HOURS_TZ, GOOGLE_SHEET_ID
from app.db import calls as calls_db
from app.db import leads as leads_db
from app.sheets import apps_script
from app.sheets.client import get_worksheet
from app.sheets.sync import PHONE_HINTS, find_column

logger = logging.getLogger(__name__)

# The Apps Script transport APPENDS transcripts (export_session has no
# overwrite), while transcript_reconcile is at-least-once by design — the
# same lead can sit in the queue twice for one call, and a crash between
# write and ack redrives. The gspread path shrugged that off by overwriting
# cells; this path needs a per-call marker so a call's transcript lands in
# the sheet exactly once. Set only AFTER a successful export: a marker set
# first would turn one transient failure into a transcript that never
# arrives. The crash window between export and marker is the remaining
# at-least-once residue — rare, and a duplicate beats a loss. Redis being
# flushable (CLAUDE.md) means a flush can also re-append old transcripts;
# same trade.
_EXPORTED_MARKER_FMT = "sheets:exported:{call_id}"
_EXPORTED_MARKER_TTL_S = 90 * 24 * 3600


def exported_marker_key(call_id) -> str:
    """The per-call exported-marker Redis key. Public because the ElevenLabs
    webhook must DELETE it when it overwrites the row with the provider's
    canonical transcript — otherwise a marker set by an earlier drain of the
    stream-teardown version suppresses the canonical re-export forever."""
    return _EXPORTED_MARKER_FMT.format(call_id=call_id)

_WRITEBACK_HEADERS = ["Status", "Turns", "CalledAt", "Transcript", "Summary"]


async def _ensure_writeback_columns(ws) -> dict[str, int]:
    """Returns {header: 1-indexed column}. Appends any missing write-back
    headers to the right of the sheet's existing columns — mirrors
    leadfiles.build_result_xlsx's "append, don't reshape" approach."""
    headers = await asyncio.to_thread(ws.row_values, 1)
    col_map = {h: i + 1 for i, h in enumerate(headers)}
    missing = [h for h in _WRITEBACK_HEADERS if h not in col_map]
    if missing:
        next_col = len(headers) + 1
        for offset, h in enumerate(missing):
            col_map[h] = next_col + offset
        cell_range = f"{gsutils.rowcol_to_a1(1, next_col)}:" \
                     f"{gsutils.rowcol_to_a1(1, next_col + len(missing) - 1)}"
        await asyncio.to_thread(ws.update, [missing], cell_range)
    return col_map


async def _resolve_row(ws, *, sheet_row: int | None, phone_e164: str, phone_col: int) -> int | None:
    """sheet_row is a fast-path hint, verified before use. Falls back to a
    full phone-column scan on a miss (or if no hint exists at all)."""
    if sheet_row:
        cell = await asyncio.to_thread(ws.cell, sheet_row, phone_col)
        if normalize_phone_e164(cell.value) == phone_e164:
            return sheet_row
        logger.warning(f"[sheets] sheet_row {sheet_row} hint stale for {phone_e164} "
                        "(row shifted?) — falling back to a full scan")

    phones = await asyncio.to_thread(ws.col_values, phone_col)
    for i, raw in enumerate(phones, start=1):
        if normalize_phone_e164(raw) == phone_e164:
            return i
    return None


def _format_transcript(turns) -> str:
    return "\n".join(f"{t.role}: {t.text}" for t in (turns or []))


async def _write_back_via_script(lead, call) -> bool:
    """The Apps Script transport's half of write_back_lead.

    The transcript goes to the script's Transcripts tab keyed by the call id
    (its Session ID column), and the Leads-tab Status rides along
    best-effort — 'Lead not found' is normal for console-added leads. ONLY
    the export's failure requests a retry: export_session appends blindly,
    so retrying after a successful export would write the whole transcript
    into the sheet twice.
    """
    call_id = getattr(call, "call_id", None)
    marker = exported_marker_key(call_id) if call_id else None
    already_exported = False
    if marker:
        try:
            already_exported = bool(
                await redis_client.get_redis().exists(marker))
        except Exception as exc:  # noqa: BLE001 - dedupe is best-effort
            logger.warning(f"[sheets] exported-marker check failed "
                           f"({type(exc).__name__}: {exc}) — exporting "
                           "without dedupe rather than dropping the write-back")

    if not already_exported:
        session = str(call_id or lead.lead_id)[:8]
        # In the operator's timezone, not UTC: the sheet has no tz column,
        # and Apps Script parses the bare string in ITS OWN timezone — a UTC
        # wall time showed every call 5½ hours early on the live sheet.
        ended_at = getattr(call, "ended_at", None)
        if ended_at is not None and ended_at.tzinfo is not None:
            ended_at = ended_at.astimezone(ZoneInfo(CALLING_HOURS_TZ))
        stamp = ended_at.strftime("%Y-%m-%d %H:%M:%S") if ended_at else ""
        exported = await apps_script.export_transcript(
            session=session, turns=call.transcript, timestamp=stamp)
        if not exported:
            return False
        if marker:
            try:
                await redis_client.get_redis().set(
                    marker, "1", ex=_EXPORTED_MARKER_TTL_S)
            except Exception as exc:  # noqa: BLE001 - see the check above
                logger.warning(f"[sheets] could not record exported marker "
                               f"({type(exc).__name__}: {exc})")

    # The post-call note when there is one — "Interested — asked the BDCP
    # fee, wants a callback" is what the team reads; the turn count is the
    # fallback for a call that produced no note.
    await apps_script.update_lead_status(
        phone=lead.phone_e164, status=call.status or "",
        notes=call.summary or f"{call.turns or 0} turns",
    )
    return True


async def write_back_lead(lead_id: UUID) -> bool:
    """Writes the lead's MOST RECENT call outcome to their Sheet row. Returns
    True on success, False if the row/phone-column couldn't be resolved (the
    caller — scheduler's transcript_reconcile — should retry later rather
    than treat this as permanent)."""
    lead = await leads_db.get_lead(lead_id)
    if lead is None:
        logger.error(f"[sheets] write_back_lead: lead {lead_id} not found")
        return True  # nothing to retry — the lead itself is gone

    call = await calls_db.get_latest_call_for_lead(lead_id)
    if call is None:
        logger.warning(f"[sheets] write_back_lead: no call found for lead {lead_id}")
        return True

    if apps_script.is_apps_script(GOOGLE_SHEET_ID):
        return await _write_back_via_script(lead, call)

    ws = await get_worksheet()
    col_map = await _ensure_writeback_columns(ws)
    headers = await asyncio.to_thread(ws.row_values, 1)
    # The SAME rule sheets_sync uses to pick the phone column — see
    # sync.find_column. This was an inline `"phone" in h or "mobile" in h`
    # scan, which could resolve to a different column than sync did and then
    # fail to match every lead forever: on ["Name", "Telephone Ext", "Mobile"]
    # sync picks "Mobile" (exact match wins) while the old rule picked
    # "Telephone Ext". It also found no column at all for a sheet whose header
    # is "WhatsApp Number", which sync handles fine.
    phone_header = find_column(headers, PHONE_HINTS)
    if phone_header is None:
        logger.error("[sheets] write_back_lead: no phone column found in the sheet")
        return False
    phone_col = headers.index(phone_header) + 1

    row = await _resolve_row(
        ws, sheet_row=lead.sheet_row, phone_e164=lead.phone_e164, phone_col=phone_col,
    )
    if row is None:
        logger.error(f"[sheets] write_back_lead: could not find a row for {lead.phone_e164}")
        return False

    if row != lead.sheet_row:
        await leads_db.upsert_lead(
            sheet_row=row, name=lead.name, phone_e164=lead.phone_e164,
            campaign_id=lead.campaign_id, language_pref=lead.language_pref,
            consent_basis=lead.consent_basis, consent_at=lead.consent_at,
        )

    updates = [
        {"range": gsutils.rowcol_to_a1(row, col_map["Status"]), "values": [[call.status or ""]]},
        {"range": gsutils.rowcol_to_a1(row, col_map["Turns"]), "values": [[call.turns or 0]]},
        {"range": gsutils.rowcol_to_a1(row, col_map["CalledAt"]),
         "values": [[call.ended_at.isoformat() if call.ended_at else ""]]},
        {"range": gsutils.rowcol_to_a1(row, col_map["Transcript"]),
         "values": [[_format_transcript(call.transcript)]]},
        {"range": gsutils.rowcol_to_a1(row, col_map["Summary"]), "values": [[call.summary or ""]]},
    ]
    await asyncio.to_thread(ws.batch_update, updates)
    return True
