"""db/models.py — Pydantic models mirroring the migrations/0001_init.sql schema."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel

CampaignMode = Literal["oneway", "twoway"]
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
    max_attempts: int = 2
    active: bool = True
    created_at: datetime


class Lead(BaseModel):
    lead_id: UUID
    sheet_row: int | None = None
    name: str | None = None
    phone_e164: str
    language_pref: str = "auto"
    campaign_id: UUID | None = None
    consent_basis: str | None = None
    consent_at: datetime | None = None
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
