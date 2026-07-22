"""sheets/sync.py — Pull leads from the Google Sheet into Supabase.

Header matching is flexible/case-insensitive (name/phone/campaign/consent hint
words), not a fixed exact header list — the blueprint's §7.1 only describes
the expected columns in prose, and hardcoding exact header strings would
break the moment a real sheet's headers differ even slightly. This mirrors
ai-voice-agent/backend/leadfiles.py's proven hint-based column detection
rather than assuming a rigid layout.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from app import languages
from app.compliance.dnd import normalize_phone_e164
from app.db import campaigns as campaigns_db
from app.db import leads as leads_db
from app.sheets.client import get_worksheet

logger = logging.getLogger(__name__)

# Public because three callers need the SAME answer to "which column is the
# phone number in this human's spreadsheet": this module, app/sheets/writeback.py
# (matching a transcript back to its row) and app/leads_import.py (an uploaded
# file). writeback.py used to carry its own inline rule, which could pick a
# different column than sync did — on headers like
# ["Name", "Telephone Ext", "Mobile"] sync matches "Mobile" and the old
# writeback rule matched "Telephone Ext", so every write-back silently failed
# forever. One implementation, one answer.
NAME_HINTS = ("name",)
PHONE_HINTS = ("phone", "mobile", "number", "contact", "whatsapp")
CAMPAIGN_HINTS = ("campaign",)
LANGUAGE_HINTS = ("language", "lang")
CONSENT_BASIS_HINTS = ("consent basis", "consent")
CONSENT_AT_HINTS = ("consent at", "consent timestamp", "consent date")


def find_column(headers: list[str], hints: tuple[str, ...]) -> str | None:
    """First header whose lowercased text contains any hint word, preferring
    an exact (case-insensitive) match over a substring match."""
    lowered = {h: h.lower().strip() for h in headers}
    for h, low in lowered.items():
        if low in hints:
            return h
    for h, low in lowered.items():
        if any(hint in low for hint in hints):
            return h
    return None


def parse_consent_at(raw: object) -> datetime | None:
    if not raw:
        return None
    text = str(raw).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    logger.warning(f"[sheets] unparseable consent timestamp: {text!r}")
    return None


async def sheets_sync() -> int:
    """Reads every row from the leads worksheet and upserts into Supabase.
    Returns the number of rows successfully upserted. Rows with no resolvable
    phone number or no matching campaign are skipped and logged — never
    dialled on guesswork."""
    ws = await get_worksheet()
    rows = await asyncio.to_thread(ws.get_all_records)
    if not rows:
        return 0

    headers = list(rows[0].keys())
    name_col = find_column(headers, NAME_HINTS)
    phone_col = find_column(headers, PHONE_HINTS)
    campaign_col = find_column(headers, CAMPAIGN_HINTS)
    language_col = find_column(headers, LANGUAGE_HINTS)
    consent_basis_col = find_column(headers, CONSENT_BASIS_HINTS)
    consent_at_col = find_column(headers, CONSENT_AT_HINTS)

    if not phone_col:
        logger.error("[sheets] no phone-like column found in the leads sheet — nothing synced")
        return 0

    campaign_cache: dict[str, object] = {}
    synced = 0

    for i, row in enumerate(rows, start=2):  # row 1 is the header
        phone = normalize_phone_e164(row.get(phone_col))
        if not phone:
            logger.warning(f"[sheets] row {i}: no valid phone number — skipped")
            continue

        campaign_name = str(row.get(campaign_col) or "").strip() if campaign_col else ""
        if not campaign_name:
            logger.warning(f"[sheets] row {i} ({phone}): no campaign specified — skipped")
            continue
        if campaign_name not in campaign_cache:
            campaign_cache[campaign_name] = await campaigns_db.get_campaign_by_name(campaign_name)
        campaign = campaign_cache[campaign_name]
        if campaign is None:
            logger.warning(f"[sheets] row {i} ({phone}): unknown campaign "
                            f"{campaign_name!r} — skipped")
            continue

        await leads_db.upsert_lead(
            sheet_row=i,
            name=str(row.get(name_col) or "").strip() if name_col else None,
            phone_e164=phone,
            campaign_id=campaign.campaign_id,
            language_pref=languages.normalize(row.get(language_col) if language_col else None),
            consent_basis=str(row.get(consent_basis_col) or "").strip() or None
            if consent_basis_col else None,
            consent_at=parse_consent_at(row.get(consent_at_col)) if consent_at_col else None,
        )
        synced += 1

    logger.info(f"[sheets] sync complete: {synced}/{len(rows)} row(s) upserted")
    return synced
