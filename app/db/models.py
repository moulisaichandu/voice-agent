"""db/models.py — Pydantic models mirroring the migrations/0001_init.sql schema."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, computed_field

from app import languages

CampaignMode = Literal["oneway", "twoway"]
# The six tokens in app/languages.py and in 0003's CHECK constraint. Kept as a
# Literal rather than a bare str so an unknown language fails at the API
# boundary, not at the ElevenLabs initiation frame mid-call.
CampaignLanguage = Literal["auto", "en", "te", "tinglish", "hi", "hinglish"]
LeadStatus = Literal["pending", "queued", "calling", "done", "failed", "dnd"]
# NOT a narrow Literal on purpose: migrations/0001_init.sql's calls.status is
# unconstrained `text`, and the real ElevenLabs webhook only confirms "done" as
# one observed value for `data.status` — the full vocabulary (and how it
# relates to `data.analysis.call_successful`) isn't documented anywhere I could
# verify. Storing whatever ElevenLabs actually reports is more honest than
# force-fitting into a guessed enum that could silently misclassify real calls.
CallStatus = str


class Campaign(BaseModel):
    campaign_id: UUID
    name: str
    mode: CampaignMode
    agent_id: str
    script: str | None = None
    language: CampaignLanguage = "auto"
    max_attempts: int = 2
    active: bool = True
    created_at: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def voice_backend(self) -> str:
        """Which voice backend will actually carry this campaign's calls.

        Derived, never stored: app/languages.py owns the mapping and it depends
        on TELUGU_BACKEND, so a stored column would go stale the moment the
        rollback switch was used.

        Computed here rather than in the dashboard because the dashboard cannot
        see .env. It used to mirror the table in TypeScript, which was fine
        while the mapping was a constant and became a lie as soon as it became
        an operator decision — a rolled-back deployment would have shown every
        Telugu campaign running on a backend it no longer used, to exactly the
        person trying to work out why a call sounded wrong.
        """
        return languages.backend_display(languages.backend_for(self.language))


class Lead(BaseModel):
    lead_id: UUID
    sheet_row: int | None = None
    name: str | None = None
    phone_e164: str
    language_pref: str = "auto"
    campaign_id: UUID | None = None
    consent_basis: str | None = None
    consent_at: datetime | None = None
    # Position in the file this lead was imported from — what due_leads()
    # orders by. NULL for Google-Sheet and single-lead-form leads, which have
    # no file position and sort after the ordered ones.
    dial_order: int | None = None
    dnd: bool = False
    status: LeadStatus = "pending"
    attempts: int = 0
    last_called_at: datetime | None = None
    created_at: datetime


class TranscriptTurn(BaseModel):
    role: Literal["agent", "lead"]
    text: str


class Call(BaseModel):
    call_id: UUID
    lead_id: UUID | None = None
    campaign_id: UUID | None = None
    el_conversation_id: str | None = None
    provider_call_id: str | None = None
    mode: str | None = None
    status: CallStatus | None = None
    turns: int | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    transcript: list[TranscriptTurn] | None = None
    summary: str | None = None
    created_at: datetime


class DocChunk(BaseModel):
    chunk_id: UUID
    doc_name: str | None = None
    section: str | None = None
    content: str
    created_at: datetime
