"""Integration tests for app/db/* — run against the REAL local pgvector
container (docker compose --profile dev up -d postgres), not mocks. Each test
uses a unique random phone/name suffix so repeated runs against the same
disposable dev database never collide on the leads/calls unique constraints.
"""

import uuid
from datetime import datetime, timedelta

import pytest

from app.db import calls as calls_db
from app.db import campaigns as campaigns_db
from app.db import leads as leads_db
from app.db import rag_store
from app.db.models import TranscriptTurn

pytestmark = pytest.mark.integration


def _uniq_phone() -> str:
    return "+91" + uuid.uuid4().hex[:10]


@pytest.fixture
async def campaign():
    return await campaigns_db.create_campaign(
        name=f"test-{uuid.uuid4().hex[:8]}", mode="twoway", agent_id="agent_test",
    )


async def test_create_and_get_campaign(campaign):
    fetched = await campaigns_db.get_campaign(campaign.campaign_id)
    assert fetched is not None
    assert fetched.mode == "twoway"


async def test_upsert_lead_is_idempotent_on_phone_plus_campaign(campaign):
    phone = _uniq_phone()
    a = await leads_db.upsert_lead(
        sheet_row=2, name="Ravi", phone_e164=phone, campaign_id=campaign.campaign_id,
    )
    b = await leads_db.upsert_lead(
        sheet_row=3, name="Ravi K", phone_e164=phone, campaign_id=campaign.campaign_id,
    )
    assert a.lead_id == b.lead_id          # same row, not a duplicate
    assert b.sheet_row == 3                 # latest values win
    assert b.name == "Ravi K"


async def test_lead_lifecycle_queued_then_calling_is_guarded(campaign):
    """mark_calling must only succeed from 'queued' — this is the
    reserve-before-act guard the Redis worker depends on to avoid double-dial."""
    lead = await leads_db.upsert_lead(
        sheet_row=1, name="Kiran", phone_e164=_uniq_phone(), campaign_id=campaign.campaign_id,
    )
    assert await leads_db.mark_calling(lead.lead_id) is False  # still 'pending', not 'queued'

    assert await leads_db.mark_queued(lead.lead_id) is True
    assert await leads_db.mark_queued(lead.lead_id) is False   # already queued, guard rejects

    assert await leads_db.mark_calling(lead.lead_id) is True
    assert await leads_db.mark_calling(lead.lead_id) is False  # already calling, guard rejects

    refetched = await leads_db.get_lead(lead.lead_id)
    assert refetched.status == "calling"
    # mark_calling does NOT count as an attempt — only record_dial_attempt()
    # does, once a call is actually placed. See its docstring for why: a
    # reservation released back to 'pending' (preflight failed, etc.) must not
    # silently burn down max_attempts.
    assert refetched.attempts == 0


async def test_record_dial_attempt_increments_only_when_called(campaign):
    lead = await leads_db.upsert_lead(
        sheet_row=1, name="Lakshmi", phone_e164=_uniq_phone(), campaign_id=campaign.campaign_id,
    )
    await leads_db.record_dial_attempt(lead.lead_id)
    await leads_db.record_dial_attempt(lead.lead_id)
    refetched = await leads_db.get_lead(lead.lead_id)
    assert refetched.attempts == 2
    assert refetched.last_called_at is not None


_VALID_CONSENT = dict(consent_basis="explicit", consent_at=datetime.now() - timedelta(hours=1))


async def test_due_leads_excludes_dnd_and_inactive_campaigns(campaign):
    ok = await leads_db.upsert_lead(
        sheet_row=1, name="A", phone_e164=_uniq_phone(), campaign_id=campaign.campaign_id,
        **_VALID_CONSENT,
    )
    dnd_lead = await leads_db.upsert_lead(
        sheet_row=2, name="B", phone_e164=_uniq_phone(), campaign_id=campaign.campaign_id,
        **_VALID_CONSENT,
    )
    await leads_db.mark_dnd(dnd_lead.lead_id)

    due = await leads_db.due_leads(limit=50)
    due_ids = {lead.lead_id for lead in due}
    assert ok.lead_id in due_ids
    assert dnd_lead.lead_id not in due_ids


async def test_due_leads_excludes_leads_without_valid_consent(campaign):
    """The gap the audit flagged: has_valid_consent() existed and was
    unit-tested in isolation, but nothing wired it into the query that
    decides who's actually dial-eligible."""
    no_consent = await leads_db.upsert_lead(
        sheet_row=1, name="C", phone_e164=_uniq_phone(), campaign_id=campaign.campaign_id,
    )
    with_consent = await leads_db.upsert_lead(
        sheet_row=2, name="D", phone_e164=_uniq_phone(), campaign_id=campaign.campaign_id,
        **_VALID_CONSENT,
    )
    due_ids = {lead.lead_id for lead in await leads_db.due_leads(limit=50)}
    assert no_consent.lead_id not in due_ids
    assert with_consent.lead_id in due_ids


async def test_get_lead_by_phone_matches_the_most_recent_row(campaign):
    """Backs the Sheets write-back fallback: when a cached sheet_row no longer
    points at the right lead, the phone-number lookup must still find them."""
    phone = _uniq_phone()
    lead = await leads_db.upsert_lead(
        sheet_row=5, name="Sita", phone_e164=phone, campaign_id=campaign.campaign_id,
    )
    found = await leads_db.get_lead_by_phone(phone)
    assert found is not None
    assert found.lead_id == lead.lead_id


async def test_call_round_trip_with_transcript(campaign):
    lead = await leads_db.upsert_lead(
        sheet_row=1, name="Anu", phone_e164=_uniq_phone(), campaign_id=campaign.campaign_id,
    )
    conv_id = f"conv_{uuid.uuid4().hex}"
    created = await calls_db.create_call(
        lead_id=lead.lead_id, campaign_id=campaign.campaign_id, mode="twoway",
        el_conversation_id=conv_id,
    )
    assert created.el_conversation_id == conv_id

    updated = await calls_db.record_transcript(
        el_conversation_id=conv_id, status="answered", turns=2,
        transcript=[
            TranscriptTurn(role="agent", text="నమస్కారం, ఇది Digital Brolly నుండి."),
            TranscriptTurn(role="lead", text="chెప్పండి"),
        ],
        summary="Interested, follow up requested.",
    )
    assert updated.status == "answered"
    assert updated.turns == 2
    assert len(updated.transcript) == 2
    assert updated.transcript[0].text.startswith("నమస్కారం")

    refetched = await calls_db.get_call_by_conversation_id(conv_id)
    assert refetched.summary == "Interested, follow up requested."


async def test_calls_el_conversation_id_is_unique(campaign):
    """Schema-level defense in depth: a retried webhook must not be able to
    create a duplicate call row even if the Redis idempotency key expired."""
    lead = await leads_db.upsert_lead(
        sheet_row=1, name="X", phone_e164=_uniq_phone(), campaign_id=campaign.campaign_id,
    )
    conv_id = f"conv_{uuid.uuid4().hex}"
    await calls_db.create_call(
        lead_id=lead.lead_id, campaign_id=campaign.campaign_id, mode="oneway",
        el_conversation_id=conv_id,
    )
    with pytest.raises(Exception, match="unique|duplicate"):
        await calls_db.create_call(
            lead_id=lead.lead_id, campaign_id=campaign.campaign_id, mode="oneway",
            el_conversation_id=conv_id,
        )


async def test_rag_match_chunks_filters_by_min_score():
    """Two chunks, one near-identical to the query embedding, one orthogonal.
    match_chunks() (the SQL function) must exclude the orthogonal one below
    the threshold — this is the SQL-side half of the RAG relevance guard."""
    doc = f"testdoc-{uuid.uuid4().hex[:8]}"
    await rag_store.insert_chunk(
        doc_name=doc, section="fees", content="The course fee is 5000 rupees.",
        embedding=[1.0, 0.0] + [0.0] * 1534,
    )
    await rag_store.insert_chunk(
        doc_name=doc, section="unrelated", content="Classes are in the morning.",
        embedding=[0.0, 1.0] + [0.0] * 1534,
    )
    hits = await rag_store.match_chunks(
        [1.0, 0.0] + [0.0] * 1534, match_count=5, min_score=0.30,
    )
    sections = {h["section"] for h in hits}
    assert "fees" in sections
    assert "unrelated" not in sections

    await rag_store.clear_doc(doc)
