"""admin/leads.py — Lead CRUD, do-not-call, and the cross-campaign lookup.

See app/admin/__init__.py for the router assembly and auth dependency shared
by every admin submodule.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from app import languages, leads_import
from app.compliance.consent import has_valid_consent
from app.compliance.dnd import normalize_phone_e164
from app.db import calls as calls_db
from app.db import campaigns as campaigns_db
from app.db import leads as leads_db
from app.db.models import Call, Lead
from app.telephony import worker as telephony_worker

router = APIRouter()
logger = logging.getLogger(__name__)


class LeadCreate(BaseModel):
    phone: str = Field(min_length=1, description="Any reasonable Indian phone "
                       "format — normalized server-side to +91E.164")
    name: str | None = None
    language_pref: str = "auto"
    consent_basis: Literal["explicit", "inferred"] | None = None
    consent_at: datetime | None = None


@router.get("/campaigns/{campaign_id}/leads", response_model=list[Lead])
async def list_leads(campaign_id: UUID) -> list[Lead]:
    if await campaigns_db.get_campaign(campaign_id) is None:
        raise HTTPException(status_code=404, detail="campaign not found")
    return await leads_db.list_leads_for_campaign(campaign_id)


@router.post("/campaigns/{campaign_id}/leads", response_model=Lead, status_code=201)
async def add_lead(campaign_id: UUID, body: LeadCreate) -> Lead:
    if await campaigns_db.get_campaign(campaign_id) is None:
        raise HTTPException(status_code=404, detail="campaign not found")
    phone = normalize_phone_e164(body.phone)
    if not phone:
        raise HTTPException(status_code=400,
                            detail=f"{body.phone!r} is not a valid Indian phone number")
    # leads.language_pref is CHECK-constrained to six canonical tokens, and this
    # endpoint accepts free text from the caller (e.g. "Telugu"), so normalize it
    # to one of those tokens before inserting — the same pattern as phone above.
    language_pref = languages.normalize(body.language_pref)
    return await leads_db.upsert_lead(
        sheet_row=None, name=body.name, phone_e164=phone, campaign_id=campaign_id,
        language_pref=language_pref,
        consent_basis=body.consent_basis, consent_at=body.consent_at,
    )


class LeadImportError(BaseModel):
    row_number: int
    reason: str


class LeadImportResult(BaseModel):
    received: int = Field(description="Data rows read from the file")
    imported: int = Field(description="Rows written (insert or update)")
    skipped: int = Field(description="Rows rejected — see errors")
    dialable: int = Field(
        description="Of those imported, how many pass the SAME consent check "
                    "the dialer applies. A file with no consent information "
                    "imports fine and dials nothing; this is how you see that."
    )
    errors: list[LeadImportError] = []


@router.post("/campaigns/{campaign_id}/leads/upload", response_model=LeadImportResult)
async def upload_leads(
    campaign_id: UUID,
    file: UploadFile = File(description="CSV or Excel file of leads"),
    consent_basis: Literal["explicit", "inferred"] | None = Form(
        default=None,
        description="Applied ONLY to rows whose own consent column is empty. "
                    "Omit it and those rows import with no consent — visible "
                    "in `dialable`, and never dialled.",
    ),
) -> LeadImportResult:
    """Bulk-import leads from an uploaded file.

    Consent resolution, in order: the row's own consent column wins; otherwise
    the operator-supplied *consent_basis* applies; otherwise the lead has none
    and due_leads() will not select it. Uploading a file is not itself consent —
    CLAUDE.md treats a lead with no consent record as exactly as dial-ineligible
    as a DND one, so this endpoint reports what it produced rather than
    quietly making everything dialable.
    """
    if await campaigns_db.get_campaign(campaign_id) is None:
        raise HTTPException(status_code=404, detail="campaign not found")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="the uploaded file is empty")

    # Off-thread: parsing is blocking CPU work and this process also serves
    # every live audio bridge — the same reason bridge.py offloads its SDK
    # call. A 5,000-row workbook parsed inline would stall calls in progress.
    try:
        parsed = await asyncio.to_thread(
            leads_import.parse_leads_file, file.filename or "", data
        )
    except leads_import.LeadsFileError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Stamped once, not per row, so every lead from one upload shares a single
    # consent timestamp — and it can never drift past `now` mid-import and be
    # rejected as future-dated by has_valid_consent().
    fallback_at = datetime.now(timezone.utc) if consent_basis else None

    imported = 0
    dialable = 0
    errors = [LeadImportError(row_number=e.row_number, reason=e.reason) for e in parsed.errors]

    for position, lead in enumerate(parsed.leads, start=1):
        basis = lead.consent_basis or consent_basis
        at = lead.consent_at if lead.consent_basis else fallback_at
        try:
            await leads_db.upsert_lead(
                sheet_row=None,  # not from the Sheet — writeback resolves by phone
                name=lead.name,
                phone_e164=lead.phone_e164,
                campaign_id=campaign_id,
                language_pref=lead.language_pref,
                consent_basis=basis,
                consent_at=at,
                # Position among the ACCEPTED rows, not the raw file line —
                # so a rejected row in the middle doesn't leave a gap, and the
                # order is stable if the operator fixes that row and re-uploads.
                dial_order=position,
            )
        except Exception as exc:
            logger.exception(f"[admin] upload: row {lead.row_number} failed")
            errors.append(LeadImportError(
                row_number=lead.row_number,
                reason=f"could not be saved: {type(exc).__name__}",
            ))
            continue
        imported += 1
        if has_valid_consent(basis, at):
            dialable += 1

    logger.info(
        f"[admin] uploaded {file.filename!r} to campaign {campaign_id}: "
        f"{imported} imported, {len(errors)} skipped, {dialable} dialable"
    )
    return LeadImportResult(
        received=parsed.received, imported=imported, skipped=len(errors),
        dialable=dialable, errors=errors,
    )


@router.get("/leads", response_model=list[Lead])
async def search_leads(phone: str = Query(min_length=1)) -> list[Lead]:
    """Every lead across every campaign matching this phone number — "someone
    rang back / complained, what do we know about them?" without SQL. Any
    reasonable Indian format works; normalized the same way a dial-path phone
    number is, so this can never disagree with what the worker would match."""
    normalized = normalize_phone_e164(phone)
    if not normalized:
        raise HTTPException(status_code=400, detail=f"{phone!r} is not a valid Indian phone number")
    return await leads_db.search_leads_by_phone(normalized)


@router.get("/leads/{lead_id}/calls", response_model=list[Call])
async def lead_calls(lead_id: UUID) -> list[Call]:
    if await leads_db.get_lead(lead_id) is None:
        raise HTTPException(status_code=404, detail="lead not found")
    return await calls_db.list_calls_for_lead(lead_id)


@router.post("/leads/{lead_id}/do-not-call", response_model=Lead)
async def do_not_call(lead_id: UUID) -> Lead:
    """Permanently opt this lead out — the compliance-correct way to "remove"
    a lead. Wraps db/leads.mark_dnd() rather than deleting: the lead keeps its
    consent record and call history, and calls.lead_id has no ON DELETE clause
    to fall back on anyway.

    The Redis push is NOT optional and must not be left to
    scheduler.dnd_refresh's 06:00 cron — see worker.add_to_dnd_set. mark_dnd
    also covers every campaign this number appears on, not just this row."""
    lead = await leads_db.get_lead(lead_id)
    if lead is None:
        raise HTTPException(status_code=404, detail="lead not found")
    phone = await leads_db.mark_dnd(lead_id)
    if phone:
        # Best-effort: the Postgres flag above is the source of truth, so a
        # Redis blip must not fail the operator's opt-out. It costs at most a
        # cross-campaign gap until the next dnd_refresh, and the per-row
        # `l.dnd = false` filter in due_leads still holds meanwhile.
        try:
            await telephony_worker.add_to_dnd_set(phone)
        except Exception:
            logger.exception(
                f"[admin] opted {lead_id} out in Postgres but could not reach "
                "Redis — the dial-time scrub lags until the next dnd_refresh"
            )
    updated = await leads_db.get_lead(lead_id)
    assert updated is not None  # just fetched successfully above
    return updated
