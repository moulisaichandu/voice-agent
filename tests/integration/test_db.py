"""Integration tests for app/db/* — run against the REAL local pgvector
container (docker compose --profile dev up -d postgres), not mocks. Each test
uses a unique random phone/name suffix so repeated runs against the same
disposable dev database never collide on the leads/calls unique constraints.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.db import calls as calls_db
from app.db import campaigns as campaigns_db
from app.db import leads as leads_db
from app.db import rag_store
from app.db.models import TranscriptTurn

pytestmark = pytest.mark.integration


def _uniq_phone() -> str:
    return "+91" + uuid.uuid4().hex[:10]


@pytest.fixture(autouse=True)
async def _deactivate_campaigns_this_test_creates():
    """Leave no active campaign behind.

    create_campaign() defaults to active=true, and this module makes one (often
    several) per test against a database that is ALSO the running app's. Nothing
    here switched them off, so they accumulated: 258 of them, and the readiness
    dashboard reported "All 32 active campaign(s) are blocked" for agent ids
    that only ever existed in a test.

    That went unnoticed because test_pipeline.py's isolation fixture ran a bare
    `update campaigns set active = false` and was silently cleaning up after
    this module as a side effect. Fixing that one to restore what it deactivated
    (correctly — it was switching off the operator's live campaigns) removed the
    accidental cleanup and left this visible.

    Scoped by ID, not by name: matching 'test-%' would also catch a real
    campaign an operator happened to name that way.
    """
    from app.db.pool import get_pool

    pool = await get_pool()
    before = {
        r["campaign_id"] for r in
        await pool.fetch("select campaign_id from campaigns")
    }
    try:
        yield
    finally:
        after = await pool.fetch("select campaign_id from campaigns where active")
        created = [r["campaign_id"] for r in after if r["campaign_id"] not in before]
        if created:
            await pool.execute(
                "update campaigns set active = false "
                "where campaign_id = any($1::uuid[])",
                created,
            )


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


async def test_due_leads_scoped_to_a_campaign_excludes_other_campaigns(campaign):
    """Backs the admin API's per-campaign trigger: against real Postgres, a
    scoped call must return only that campaign's leads."""
    other = await campaigns_db.create_campaign(
        name=f"other-{uuid.uuid4().hex[:8]}", mode="twoway", agent_id="agent_other",
    )
    mine = await leads_db.upsert_lead(
        sheet_row=1, name="Mine", phone_e164=_uniq_phone(),
        campaign_id=campaign.campaign_id, **_VALID_CONSENT,
    )
    theirs = await leads_db.upsert_lead(
        sheet_row=1, name="Theirs", phone_e164=_uniq_phone(),
        campaign_id=other.campaign_id, **_VALID_CONSENT,
    )

    scoped = {lead.lead_id for lead in
              await leads_db.due_leads(limit=50, campaign_id=campaign.campaign_id)}
    assert mine.lead_id in scoped
    assert theirs.lead_id not in scoped

    # Unscoped still sees both — the scheduler's own run must stay global.
    everyone = {lead.lead_id for lead in await leads_db.due_leads(limit=200)}
    assert {mine.lead_id, theirs.lead_id} <= everyone


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


async def test_set_campaign_active_round_trips_and_gates_due_leads(campaign):
    """The admin console's activate/deactivate toggle, against real Postgres:
    flipping active=false must make due_leads() stop selecting this
    campaign's leads immediately, and flipping it back must restore them."""
    lead = await leads_db.upsert_lead(
        sheet_row=1, name="Toggle", phone_e164=_uniq_phone(),
        campaign_id=campaign.campaign_id, **_VALID_CONSENT,
    )

    updated = await campaigns_db.set_campaign_active(campaign.campaign_id, False)
    assert updated is not None
    assert updated.active is False
    due_ids = {lead.lead_id for lead in await leads_db.due_leads(limit=50)}
    assert lead.lead_id not in due_ids

    reactivated = await campaigns_db.set_campaign_active(campaign.campaign_id, True)
    assert reactivated.active is True
    due_ids = {lead.lead_id for lead in await leads_db.due_leads(limit=50)}
    assert lead.lead_id in due_ids


async def test_set_campaign_active_returns_none_for_an_unknown_campaign():
    result = await campaigns_db.set_campaign_active(uuid.uuid4(), False)
    assert result is None


async def test_lead_status_counts_groups_by_status(campaign):
    a = await leads_db.upsert_lead(
        sheet_row=1, name="A", phone_e164=_uniq_phone(), campaign_id=campaign.campaign_id,
    )
    b = await leads_db.upsert_lead(
        sheet_row=2, name="B", phone_e164=_uniq_phone(), campaign_id=campaign.campaign_id,
    )
    await leads_db.upsert_lead(
        sheet_row=3, name="C", phone_e164=_uniq_phone(), campaign_id=campaign.campaign_id,
    )
    await leads_db.mark_dnd(a.lead_id)
    await leads_db.mark_queued(b.lead_id)

    counts = await leads_db.lead_status_counts(campaign_id=campaign.campaign_id)

    assert counts.get("dnd") == 1
    assert counts.get("queued") == 1
    assert counts.get("pending") == 1


async def test_lead_status_counts_for_active_campaigns_excludes_inactive_ones(campaign):
    """Delta-based rather than absolute, since this runs against a shared,
    already-polluted dev DB: an inactive campaign's lead must move the count
    by exactly ZERO, and an active campaign's lead by exactly ONE."""
    inactive = await campaigns_db.create_campaign(
        name=f"inactive-{uuid.uuid4().hex[:8]}", mode="twoway", agent_id="agent_inactive",
    )
    await campaigns_db.set_campaign_active(inactive.campaign_id, False)

    before = (await leads_db.lead_status_counts_for_active_campaigns()).get("pending", 0)

    await leads_db.upsert_lead(
        sheet_row=1, name="Inactive", phone_e164=_uniq_phone(), campaign_id=inactive.campaign_id,
    )
    after_inactive = (await leads_db.lead_status_counts_for_active_campaigns()).get("pending", 0)
    assert after_inactive == before

    await leads_db.upsert_lead(
        sheet_row=1, name="Active", phone_e164=_uniq_phone(), campaign_id=campaign.campaign_id,
    )
    after_active = (await leads_db.lead_status_counts_for_active_campaigns()).get("pending", 0)
    assert after_active == before + 1


async def test_search_leads_by_phone_finds_the_same_number_across_campaigns(campaign):
    other = await campaigns_db.create_campaign(
        name=f"other-{uuid.uuid4().hex[:8]}", mode="twoway", agent_id="agent_other",
    )
    phone = _uniq_phone()
    a = await leads_db.upsert_lead(
        sheet_row=1, name="Ravi (campaign 1)", phone_e164=phone, campaign_id=campaign.campaign_id,
    )
    b = await leads_db.upsert_lead(
        sheet_row=1, name="Ravi (campaign 2)", phone_e164=phone, campaign_id=other.campaign_id,
    )

    found = {lead.lead_id for lead in await leads_db.search_leads_by_phone(phone)}
    assert found == {a.lead_id, b.lead_id}


async def test_search_leads_by_phone_finds_nothing_for_an_unknown_number():
    assert await leads_db.search_leads_by_phone(_uniq_phone()) == []


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


async def test_list_calls_for_lead_returns_only_that_leads_calls(campaign):
    lead_a = await leads_db.upsert_lead(
        sheet_row=1, name="A", phone_e164=_uniq_phone(), campaign_id=campaign.campaign_id,
    )
    lead_b = await leads_db.upsert_lead(
        sheet_row=2, name="B", phone_e164=_uniq_phone(), campaign_id=campaign.campaign_id,
    )
    call_a = await calls_db.create_call(
        lead_id=lead_a.lead_id, campaign_id=campaign.campaign_id, mode="twoway",
        el_conversation_id=f"conv_{uuid.uuid4().hex}",
    )
    await calls_db.create_call(
        lead_id=lead_b.lead_id, campaign_id=campaign.campaign_id, mode="twoway",
        el_conversation_id=f"conv_{uuid.uuid4().hex}",
    )

    calls = await calls_db.list_calls_for_lead(lead_a.lead_id)
    assert [c.call_id for c in calls] == [call_a.call_id]


async def test_list_recent_calls_orders_newest_first_and_respects_limit(campaign):
    lead = await leads_db.upsert_lead(
        sheet_row=1, name="A", phone_e164=_uniq_phone(), campaign_id=campaign.campaign_id,
    )
    ids = []
    for _ in range(3):
        c = await calls_db.create_call(
            lead_id=lead.lead_id, campaign_id=campaign.campaign_id, mode="twoway",
            el_conversation_id=f"conv_{uuid.uuid4().hex}",
        )
        ids.append(c.call_id)

    recent = await calls_db.list_recent_calls(limit=2)
    recent_ids = [c.call_id for c in recent]
    # Newest-first: the two most recently created of the three must be first,
    # in reverse creation order. This dev DB is shared across tests, so only
    # assert the relative order of OUR ids among whatever else is present.
    ours_in_order = [i for i in recent_ids if i in ids]
    assert ours_in_order == list(reversed(ids))[: len(ours_in_order)]
    assert len(recent_ids) <= 2


async def test_call_counts_since_includes_recent_calls_and_counts_transcripts(campaign):
    lead = await leads_db.upsert_lead(
        sheet_row=1, name="A", phone_e164=_uniq_phone(), campaign_id=campaign.campaign_id,
    )
    before = datetime.now() - timedelta(hours=1)
    await calls_db.create_call(
        lead_id=lead.lead_id, campaign_id=campaign.campaign_id, mode="twoway",
        el_conversation_id=f"conv_{uuid.uuid4().hex}",
    )
    connected = await calls_db.create_call(
        lead_id=lead.lead_id, campaign_id=campaign.campaign_id, mode="twoway",
        el_conversation_id=f"conv_{uuid.uuid4().hex}",
    )
    await calls_db.record_transcript(
        el_conversation_id=connected.el_conversation_id, status="done", turns=3,
        transcript=[], summary=None,
    )

    counts = await calls_db.call_counts_since(before)
    assert counts["total"] >= 2
    assert counts["with_transcript"] >= 1


async def test_call_counts_since_active_campaigns_only_excludes_inactive_campaigns(campaign):
    inactive = await campaigns_db.create_campaign(
        name=f"inactive-{uuid.uuid4().hex[:8]}", mode="twoway", agent_id="agent_inactive",
    )
    await campaigns_db.set_campaign_active(inactive.campaign_id, False)
    lead = await leads_db.upsert_lead(
        sheet_row=1, name="Inactive", phone_e164=_uniq_phone(), campaign_id=inactive.campaign_id,
    )
    before = datetime.now() - timedelta(hours=1)
    baseline = await calls_db.call_counts_since(before, active_campaigns_only=True)

    await calls_db.create_call(
        lead_id=lead.lead_id, campaign_id=inactive.campaign_id, mode="twoway",
        el_conversation_id=f"conv_{uuid.uuid4().hex}",
    )

    after = await calls_db.call_counts_since(before, active_campaigns_only=True)
    assert after["total"] == baseline["total"]  # the inactive campaign's call must not count


async def test_call_counts_since_excludes_calls_before_the_cutoff(campaign):
    """Deterministic even against a shared dev DB: a cutoff an hour in the
    FUTURE can't legitimately include anything, since started_at is set to
    now() at creation — no test's call can have a start time later than
    "now" was when it ran."""
    lead = await leads_db.upsert_lead(
        sheet_row=1, name="A", phone_e164=_uniq_phone(), campaign_id=campaign.campaign_id,
    )
    await calls_db.create_call(
        lead_id=lead.lead_id, campaign_id=campaign.campaign_id, mode="twoway",
        el_conversation_id=f"conv_{uuid.uuid4().hex}",
    )

    future_cutoff = datetime.now() + timedelta(hours=1)
    counts = await calls_db.call_counts_since(future_cutoff)
    assert counts == {"total": 0, "with_transcript": 0}


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


# ── dial order (migrations/0002) ─────────────────────────────────────────────

@pytest.mark.integration
async def test_due_leads_returns_file_imported_leads_in_file_order(campaign):
    """A campaign calls its uploaded list top-to-bottom. Ordering by
    created_at could not express this: a bulk import writes every row in the
    same instant, so the values tie and Postgres is free to return any order."""
    phones = ["+919700001001", "+919700001002", "+919700001003", "+919700001004"]
    # Inserted in REVERSE, so passing only on created_at is impossible.
    for position, phone in reversed(list(enumerate(phones, start=1))):
        await leads_db.upsert_lead(
            sheet_row=None, name=f"L{position}", phone_e164=phone,
            campaign_id=campaign.campaign_id, consent_basis="explicit",
            consent_at=datetime.now(timezone.utc) - timedelta(days=1),
            dial_order=position,
        )

    due = await leads_db.due_leads(limit=10, campaign_id=campaign.campaign_id)
    assert [lead.phone_e164 for lead in due] == phones


@pytest.mark.integration
async def test_re_uploading_a_reordered_file_changes_the_dial_order(campaign):
    """The failure that motivated the column: upsert_lead never rewrites
    created_at, so before this a reordered re-upload left the calling order
    exactly as it was, with nothing to indicate why."""
    a, b = "+919700002001", "+919700002002"
    consent = dict(consent_basis="explicit",
                   consent_at=datetime.now(timezone.utc) - timedelta(days=1))
    for position, phone in enumerate([a, b], start=1):
        await leads_db.upsert_lead(sheet_row=None, name=None, phone_e164=phone,
                                   campaign_id=campaign.campaign_id,
                                   dial_order=position, **consent)
    first = await leads_db.due_leads(limit=10, campaign_id=campaign.campaign_id)
    assert [lead.phone_e164 for lead in first] == [a, b]

    for position, phone in enumerate([b, a], start=1):  # swapped in the file
        await leads_db.upsert_lead(sheet_row=None, name=None, phone_e164=phone,
                                   campaign_id=campaign.campaign_id,
                                   dial_order=position, **consent)
    second = await leads_db.due_leads(limit=10, campaign_id=campaign.campaign_id)
    assert [lead.phone_e164 for lead in second] == [b, a]


@pytest.mark.integration
async def test_a_sheet_sync_does_not_wipe_an_uploaded_leads_dial_order(campaign):
    """sheets_sync calls upsert_lead with no dial_order. If that NULLed the
    column, a lead that also appears in the Sheet would silently drop to the
    back of the calling order on the next 10-minute sync."""
    phone = "+919700003001"
    consent = dict(consent_basis="explicit",
                   consent_at=datetime.now(timezone.utc) - timedelta(days=1))
    await leads_db.upsert_lead(sheet_row=None, name="From file", phone_e164=phone,
                               campaign_id=campaign.campaign_id, dial_order=1, **consent)
    # Same shape sheets_sync uses: a sheet_row, and no dial_order at all.
    updated = await leads_db.upsert_lead(sheet_row=7, name="From sheet", phone_e164=phone,
                                         campaign_id=campaign.campaign_id, **consent)

    assert updated.dial_order == 1, "the file position must survive a Sheet sync"
    assert updated.sheet_row == 7


@pytest.mark.integration
async def test_leads_without_a_dial_order_sort_after_those_with_one(campaign):
    """Sheet/single-form leads have no file position; they must not jump ahead
    of an uploaded list just because they happen to be older."""
    consent = dict(consent_basis="explicit",
                   consent_at=datetime.now(timezone.utc) - timedelta(days=1))
    await leads_db.upsert_lead(sheet_row=None, name="no position",
                               phone_e164="+919700004001",
                               campaign_id=campaign.campaign_id, **consent)
    await leads_db.upsert_lead(sheet_row=None, name="from file",
                               phone_e164="+919700004002",
                               campaign_id=campaign.campaign_id, dial_order=1, **consent)

    due = await leads_db.due_leads(limit=10, campaign_id=campaign.campaign_id)
    assert [lead.phone_e164 for lead in due] == ["+919700004002", "+919700004001"]



# ── a failed call must not retire the lead ──────────────────────────────────
#
# _finalise_call writes 'failed' for any call that connected but did not run
# its course, and due_leads() selects only 'pending', so max_attempts was never
# honoured for exactly the calls it exists for. Observed live: an ElevenLabs
# account in past_due failed every EN/HI call at attempt 1 of 2, retiring each
# lead permanently.

async def test_a_failed_lead_under_max_attempts_is_returned_to_pending(campaign):
    lead = await leads_db.upsert_lead(
        sheet_row=None, name="retry-me", phone_e164=_uniq_phone(),
        campaign_id=campaign.campaign_id, consent_basis="explicit",
        consent_at=datetime.now(timezone.utc),
    )
    await leads_db.record_dial_attempt(lead.lead_id)   # attempts = 1 of 2
    await leads_db.mark_result(lead.lead_id, "failed")

    moved = await leads_db.requeue_failed_leads(cooldown_s=0)

    assert lead.lead_id in moved
    again = await leads_db.get_lead(lead.lead_id)
    assert again.status == "pending"
    due = await leads_db.due_leads(limit=50, campaign_id=campaign.campaign_id)
    assert lead.lead_id in [x.lead_id for x in due], "still not dialable"


async def test_a_failed_lead_at_max_attempts_stays_failed(campaign):
    """Exhausted is terminal — this must not become an infinite redial loop."""
    lead = await leads_db.upsert_lead(
        sheet_row=None, name="exhausted", phone_e164=_uniq_phone(),
        campaign_id=campaign.campaign_id, consent_basis="explicit",
        consent_at=datetime.now(timezone.utc),
    )
    for _ in range(campaign.max_attempts):
        await leads_db.record_dial_attempt(lead.lead_id)
    await leads_db.mark_result(lead.lead_id, "failed")

    moved = await leads_db.requeue_failed_leads(cooldown_s=0)

    assert lead.lead_id not in moved
    assert (await leads_db.get_lead(lead.lead_id)).status == "failed"


async def test_the_cooldown_holds_a_just_failed_lead_back(campaign):
    """Without a cooldown a provider outage becomes a tight redial loop against
    the same numbers."""
    lead = await leads_db.upsert_lead(
        sheet_row=None, name="too-soon", phone_e164=_uniq_phone(),
        campaign_id=campaign.campaign_id, consent_basis="explicit",
        consent_at=datetime.now(timezone.utc),
    )
    await leads_db.record_dial_attempt(lead.lead_id)   # last_called_at = now
    await leads_db.mark_result(lead.lead_id, "failed")

    moved = await leads_db.requeue_failed_leads(cooldown_s=3600)

    assert lead.lead_id not in moved
    assert (await leads_db.get_lead(lead.lead_id)).status == "failed"


async def test_a_done_lead_is_never_resurrected(campaign):
    lead = await leads_db.upsert_lead(
        sheet_row=None, name="finished", phone_e164=_uniq_phone(),
        campaign_id=campaign.campaign_id, consent_basis="explicit",
        consent_at=datetime.now(timezone.utc),
    )
    await leads_db.record_dial_attempt(lead.lead_id)
    await leads_db.mark_result(lead.lead_id, "done")

    moved = await leads_db.requeue_failed_leads(cooldown_s=0)

    assert lead.lead_id not in moved
    assert (await leads_db.get_lead(lead.lead_id)).status == "done"


async def test_a_sheet_sync_does_not_erase_consent_captured_by_file_import(campaign):
    """sheets_sync passes consent_basis=None for a Sheet that has no consent
    column. The upsert overwrote the column unconditionally, so a lead imported
    with operator-affirmed consent silently became undialable — a compliance
    field cleared by a background job."""
    phone = _uniq_phone()
    granted = datetime.now(timezone.utc)
    await leads_db.upsert_lead(
        sheet_row=None, name="from-file", phone_e164=phone,
        campaign_id=campaign.campaign_id,
        consent_basis="explicit", consent_at=granted,
    )

    # the Sheet knows this phone but carries no consent columns
    after = await leads_db.upsert_lead(
        sheet_row=7, name="from-sheet", phone_e164=phone,
        campaign_id=campaign.campaign_id,
        consent_basis=None, consent_at=None,
    )

    assert after.consent_basis == "explicit", "sheet sync erased the consent"
    assert after.consent_at is not None
    due = await leads_db.due_leads(limit=50, campaign_id=campaign.campaign_id)
    assert after.lead_id in [x.lead_id for x in due], "lead became undialable"


async def test_a_calling_lead_that_was_never_dialled_can_be_released(campaign):
    """last_called_at is NULL only if record_dial_attempt never ran, and that
    runs before placement — so such a lead cannot have a live call."""
    lead = await leads_db.upsert_lead(
        sheet_row=None, name="crashed", phone_e164=_uniq_phone(),
        campaign_id=campaign.campaign_id, consent_basis="explicit",
        consent_at=datetime.now(timezone.utc),
    )
    await leads_db.mark_queued(lead.lead_id)
    await leads_db.mark_calling(lead.lead_id)          # crash lands right here

    assert await leads_db.release_undialed_calling_lead(lead.lead_id) is True
    assert (await leads_db.get_lead(lead.lead_id)).status == "pending"
    due = await leads_db.due_leads(limit=50, campaign_id=campaign.campaign_id)
    assert lead.lead_id in [x.lead_id for x in due], "still uncallable"


async def test_a_lead_with_a_real_call_in_flight_is_never_released(campaign):
    """The guard that stops this becoming a double-dial: a lead whose attempt
    was actually placed must be left to reap_stranded_calls, not reset here."""
    lead = await leads_db.upsert_lead(
        sheet_row=None, name="in-flight", phone_e164=_uniq_phone(),
        campaign_id=campaign.campaign_id, consent_basis="explicit",
        consent_at=datetime.now(timezone.utc),
    )
    await leads_db.mark_queued(lead.lead_id)
    await leads_db.mark_calling(lead.lead_id)
    await leads_db.record_dial_attempt(lead.lead_id)   # the call went out

    assert await leads_db.release_undialed_calling_lead(lead.lead_id) is False
    assert (await leads_db.get_lead(lead.lead_id)).status == "calling"
