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
from dataclasses import dataclass, field
from datetime import datetime

from app import languages
from app.compliance.consent import has_valid_consent, normalize_consent_at
from app.compliance.dnd import normalize_phone_e164
from app.config import GOOGLE_SHEET_ID, LEADS_WORKSHEET_NAME
from app.db import campaigns as campaigns_db
from app.db import leads as leads_db
from app.sheets import apps_script
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
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return normalize_consent_at(parsed)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return normalize_consent_at(datetime.strptime(text, fmt))
        except ValueError:
            continue
    logger.warning(f"[sheets] unparseable consent timestamp: {text!r}")
    return None


@dataclass
class SyncResult:
    """What one sync actually did, in the operator's terms.

    A bare count of upserted rows answered the wrong question. The sheet path
    imports a row with no consent columns quite happily and the dialer then
    refuses it forever (has_valid_consent), so "40 rows synced" and "40 rows
    that will never be called" were the same number. `dialable` is the one an
    operator needs; `skips` says where the rest went.

    Mirrors what the file-upload endpoint has always reported
    (app/admin/leads.py's LeadImportResult) — the two importers should not
    describe their work differently.
    """

    received: int = 0
    imported: int = 0
    dialable: int = 0
    skips: dict[str, int] = field(default_factory=dict)

    @property
    def skipped(self) -> int:
        return self.received - self.imported

    def _skip(self, reason: str) -> None:
        self.skips[reason] = self.skips.get(reason, 0) + 1


async def sheets_sync() -> SyncResult:
    """Reads every row from the leads worksheet and upserts into Supabase.

    Rows with no resolvable phone number or no matching campaign are skipped
    and logged — never dialled on guesswork."""
    if apps_script.is_apps_script(GOOGLE_SHEET_ID):
        # The Apps Script transport reports each row's REAL sheet row as
        # _row (its reader skips blank rows), so its numbering is
        # authoritative where enumerate() below is only an assumption that
        # gspread's contiguous records make safe.
        numbered = await apps_script.fetch_rows(LEADS_WORKSHEET_NAME)
    else:
        ws = await get_worksheet()
        records = await asyncio.to_thread(ws.get_all_records)
        numbered = [(i, dict(r)) for i, r in enumerate(records, start=2)]
    if not numbered:
        return SyncResult()

    headers = list(numbered[0][1].keys())
    name_col = find_column(headers, NAME_HINTS)
    phone_col = find_column(headers, PHONE_HINTS)
    campaign_col = find_column(headers, CAMPAIGN_HINTS)
    language_col = find_column(headers, LANGUAGE_HINTS)
    consent_basis_col = find_column(headers, CONSENT_BASIS_HINTS)
    consent_at_col = find_column(headers, CONSENT_AT_HINTS)

    # A "Consent At" column alone matches CONSENT_BASIS_HINTS too (it contains
    # the bare word "consent"), which files the TIMESTAMP as the BASIS and
    # leaves every lead permanently dial-ineligible with nothing logged.
    # app/leads_import.py carries this same guard; the two importers must not
    # disagree about a compliance field.
    if consent_basis_col and consent_basis_col == consent_at_col:
        consent_basis_col = None

    if not phone_col:
        logger.error("[sheets] no phone-like column found in the leads sheet — nothing synced")
        return SyncResult(received=len(numbered), skips={"no phone column": len(numbered)})

    campaign_cache: dict[str, object] = {}
    result = SyncResult(received=len(numbered))

    for i, row in numbered:  # i: 1-based sheet row (row 1 is the header)
        phone = normalize_phone_e164(row.get(phone_col))
        if not phone:
            logger.warning(f"[sheets] row {i}: no valid phone number — skipped")
            result._skip("no phone")
            continue

        campaign_name = str(row.get(campaign_col) or "").strip() if campaign_col else ""
        if not campaign_name:
            logger.warning(f"[sheets] row {i} ({phone}): no campaign specified — skipped")
            result._skip("no campaign")
            continue
        # Cached on the name as typed, but resolved case-insensitively by
        # get_campaign_by_name, so "Demo" and "demo" cost two lookups and
        # reach the same campaign rather than one of them being skipped.
        if campaign_name not in campaign_cache:
            campaign_cache[campaign_name] = await campaigns_db.get_campaign_by_name(campaign_name)
        campaign = campaign_cache[campaign_name]
        if campaign is None:
            logger.warning(f"[sheets] row {i} ({phone}): unknown campaign "
                            f"{campaign_name!r} — skipped")
            result._skip("unknown campaign")
            continue

        # Lowercased to match VALID_CONSENT_BASES, which is what
        # has_valid_consent and due_leads compare against. A sheet saying
        # "Explicit" is the same consent as one saying "explicit"; only the
        # spelling differed, and the capitalised one silently never dialled.
        consent_basis = (
            (str(row.get(consent_basis_col) or "").strip().lower() or None)
            if consent_basis_col else None
        )
        consent_at = parse_consent_at(row.get(consent_at_col)) if consent_at_col else None

        await leads_db.upsert_lead(
            sheet_row=i,
            name=str(row.get(name_col) or "").strip() if name_col else None,
            phone_e164=phone,
            campaign_id=campaign.campaign_id,
            language_pref=languages.normalize(row.get(language_col) if language_col else None),
            consent_basis=consent_basis,
            consent_at=consent_at,
            # The sheet's own top-to-bottom order IS the operator's priority.
            # due_leads() sorts by `dial_order nulls last`, so leaving this
            # None put every sheet lead behind every uploaded one.
            dial_order=i,
        )
        result.imported += 1
        if has_valid_consent(consent_basis, consent_at):
            result.dialable += 1

    logger.info(
        f"[sheets] sync complete: {result.imported}/{result.received} row(s) upserted, "
        f"{result.dialable} dialable"
        + (f" — skipped {result.skips}" if result.skips else "")
    )
    return result
