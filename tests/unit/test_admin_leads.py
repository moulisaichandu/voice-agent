"""Unit tests for admin/leads.py — the DB layer is mocked at the
app.admin.leads module's imported names, so these run with no Docker.
"""

from datetime import datetime, timezone
from uuid import uuid4

from app.admin import leads as admin_leads


def _campaign(**overrides):
    defaults = dict(
        campaign_id=uuid4(), name="Demo", mode="twoway", agent_id="agent_1",
        script=None, max_attempts=2, active=True, created_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    from app.db.models import Campaign
    return Campaign(**defaults)


def _lead(**overrides):
    defaults = dict(
        lead_id=uuid4(), sheet_row=None, name="Ravi", phone_e164="+919876543210",
        language_pref="auto", campaign_id=uuid4(), consent_basis=None, consent_at=None,
        dnd=False, status="pending", attempts=0, last_called_at=None,
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    from app.db.models import Lead
    return Lead(**defaults)


# ── campaign-scoped leads ────────────────────────────────────────────────────

def test_list_leads_404s_for_an_unknown_campaign(client, monkeypatch):
    async def not_found(cid):
        return None

    monkeypatch.setattr(admin_leads.campaigns_db, "get_campaign", not_found)

    r = client.get(f"/admin/campaigns/{uuid4()}/leads")
    assert r.status_code == 404


def test_add_lead_normalizes_the_phone_before_storing(client, monkeypatch):
    campaign = _campaign()

    async def get_campaign(cid):
        return campaign

    captured = {}

    async def fake_upsert(**kwargs):
        captured.update(kwargs)
        return _lead(campaign_id=campaign.campaign_id, phone_e164=kwargs["phone_e164"])

    monkeypatch.setattr(admin_leads.campaigns_db, "get_campaign", get_campaign)
    monkeypatch.setattr(admin_leads.leads_db, "upsert_lead", fake_upsert)

    r = client.post(f"/admin/campaigns/{campaign.campaign_id}/leads", json={
        "phone": "09876543210", "name": "Sita",
    })
    assert r.status_code == 201
    assert captured["phone_e164"] == "+919876543210"


def test_add_lead_rejects_an_unparseable_phone_number(client, monkeypatch):
    campaign = _campaign()

    async def get_campaign(cid):
        return campaign

    monkeypatch.setattr(admin_leads.campaigns_db, "get_campaign", get_campaign)

    r = client.post(f"/admin/campaigns/{campaign.campaign_id}/leads", json={
        "phone": "123", "name": "Bad Number",
    })
    assert r.status_code == 400


# ── cross-campaign search ────────────────────────────────────────────────────

def test_search_leads_normalizes_the_phone_before_searching(client, monkeypatch):
    captured = {}

    async def fake_search(phone_e164):
        captured["phone_e164"] = phone_e164
        return [_lead(phone_e164=phone_e164)]

    monkeypatch.setattr(admin_leads.leads_db, "search_leads_by_phone", fake_search)

    r = client.get("/admin/leads", params={"phone": "09876543210"})
    assert r.status_code == 200
    assert len(r.json()) == 1
    assert captured["phone_e164"] == "+919876543210"


def test_search_leads_rejects_an_unparseable_phone(client, monkeypatch):
    async def must_not_be_called(phone_e164):
        raise AssertionError("must not query the DB for an invalid phone")

    monkeypatch.setattr(admin_leads.leads_db, "search_leads_by_phone", must_not_be_called)

    r = client.get("/admin/leads", params={"phone": "123"})
    assert r.status_code == 400


def test_search_leads_returns_every_match_across_campaigns(client, monkeypatch):
    matches = [_lead(campaign_id=uuid4()), _lead(campaign_id=uuid4())]

    async def fake_search(phone_e164):
        return matches

    monkeypatch.setattr(admin_leads.leads_db, "search_leads_by_phone", fake_search)

    r = client.get("/admin/leads", params={"phone": "+919876543210"})
    assert r.status_code == 200
    assert len(r.json()) == 2


# ── per-lead call history ────────────────────────────────────────────────────

def test_lead_calls_404s_for_an_unknown_lead(client, monkeypatch):
    async def not_found(lead_id):
        return None

    monkeypatch.setattr(admin_leads.leads_db, "get_lead", not_found)

    r = client.get(f"/admin/leads/{uuid4()}/calls")
    assert r.status_code == 404


def test_lead_calls_returns_that_leads_history(client, monkeypatch):
    lead = _lead()

    async def get_lead(lead_id):
        return lead

    async def fake_calls(lead_id):
        return []

    monkeypatch.setattr(admin_leads.leads_db, "get_lead", get_lead)
    monkeypatch.setattr(admin_leads.calls_db, "list_calls_for_lead", fake_calls)

    r = client.get(f"/admin/leads/{lead.lead_id}/calls")
    assert r.status_code == 200
    assert r.json() == []


# ── do-not-call (NOT delete) ─────────────────────────────────────────────────

def test_do_not_call_404s_for_an_unknown_lead(client, monkeypatch):
    async def not_found(lead_id):
        return None

    monkeypatch.setattr(admin_leads.leads_db, "get_lead", not_found)

    r = client.post(f"/admin/leads/{uuid4()}/do-not-call")
    assert r.status_code == 404


def test_do_not_call_wraps_mark_dnd_and_returns_the_updated_lead(client, monkeypatch):
    """The compliance-correct "remove a lead" action reuses mark_dnd() — the
    same function the dial path's own DND check and dnd_refresh() already
    depend on — rather than any kind of delete."""
    lead = _lead(dnd=False, status="pending")
    calls = {"mark_dnd": []}

    fetch_sequence = [lead, _lead(lead_id=lead.lead_id, dnd=True, status="dnd")]

    async def get_lead(lead_id):
        return fetch_sequence.pop(0)

    async def fake_mark_dnd(lead_id):
        calls["mark_dnd"].append(lead_id)

    monkeypatch.setattr(admin_leads.leads_db, "get_lead", get_lead)
    monkeypatch.setattr(admin_leads.leads_db, "mark_dnd", fake_mark_dnd)

    r = client.post(f"/admin/leads/{lead.lead_id}/do-not-call")

    assert r.status_code == 200
    assert calls["mark_dnd"] == [lead.lead_id]
    assert r.json()["dnd"] is True
    assert r.json()["status"] == "dnd"


# ── leads file upload ────────────────────────────────────────────────────────

def _upload(client, campaign_id, csv_text, **data):
    return client.post(
        f"/admin/campaigns/{campaign_id}/leads/upload",
        files={"file": ("leads.csv", csv_text.encode("utf-8"), "text/csv")},
        data=data,
    )


def _mock_upload_deps(monkeypatch, *, campaign=True):
    """Campaign lookup + upsert, capturing every upserted row."""
    upserted = []

    async def get_campaign(cid):
        return _campaign(campaign_id=cid) if campaign else None

    async def upsert_lead(**kwargs):
        upserted.append(kwargs)
        return _lead(**{k: v for k, v in kwargs.items() if k != "sheet_row"},
                     sheet_row=kwargs.get("sheet_row"))

    monkeypatch.setattr(admin_leads.campaigns_db, "get_campaign", get_campaign)
    monkeypatch.setattr(admin_leads.leads_db, "upsert_lead", upsert_lead)
    return upserted


def test_upload_404s_for_an_unknown_campaign(client, monkeypatch):
    _mock_upload_deps(monkeypatch, campaign=False)
    r = _upload(client, uuid4(), "Phone\n+919876543210\n")
    assert r.status_code == 404


def test_upload_imports_leads_and_reports_bad_rows(client, monkeypatch):
    upserted = _mock_upload_deps(monkeypatch)

    r = _upload(client, uuid4(),
                "Name,Phone\nRavi,+919876543210\nBroken,12345\nSita,9812345678\n")

    assert r.status_code == 200
    body = r.json()
    assert (body["received"], body["imported"], body["skipped"]) == (3, 2, 1)
    assert body["errors"][0]["row_number"] == 3
    assert [u["phone_e164"] for u in upserted] == ["+919876543210", "+919812345678"]


def test_upload_without_a_consent_affirmation_imports_nothing_dialable(client, monkeypatch):
    """Uploading a file is NOT consent. These leads exist, keep their history,
    and are simply never selected by due_leads() — and the response says so
    rather than leaving the operator to discover it when nothing dials."""
    upserted = _mock_upload_deps(monkeypatch)

    r = _upload(client, uuid4(), "Phone\n+919876543210\n")

    assert r.status_code == 200
    assert r.json()["imported"] == 1
    assert r.json()["dialable"] == 0
    assert upserted[0]["consent_basis"] is None
    assert upserted[0]["consent_at"] is None


def test_an_affirmed_consent_basis_is_applied_with_a_timestamp(client, monkeypatch):
    """has_valid_consent() needs BOTH a basis and a real timestamp — a basis
    alone would import and still never dial."""
    upserted = _mock_upload_deps(monkeypatch)

    r = _upload(client, uuid4(), "Phone\n+919876543210\n", consent_basis="explicit")

    assert r.status_code == 200
    assert r.json()["dialable"] == 1
    assert upserted[0]["consent_basis"] == "explicit"
    assert upserted[0]["consent_at"] is not None


def test_a_rows_own_consent_column_beats_the_affirmed_default(client, monkeypatch):
    """Per-row data from the file is more specific than a blanket affirmation,
    so it must win — including its own timestamp."""
    upserted = _mock_upload_deps(monkeypatch)

    r = _upload(
        client, uuid4(),
        "Phone,Consent Basis,Consent At\n+919876543210,inferred,2026-07-01 10:00:00\n",
        consent_basis="explicit",
    )

    assert r.status_code == 200
    assert upserted[0]["consent_basis"] == "inferred"
    assert upserted[0]["consent_at"].year == 2026


def test_upload_never_writes_a_sheet_row(client, monkeypatch):
    """sheet_row is a Google-Sheet row index. An uploaded file has no relation
    to the Sheet, and a bogus hint would send write-back to a wrong row."""
    upserted = _mock_upload_deps(monkeypatch)
    _upload(client, uuid4(), "Phone\n+919876543210\n")
    assert upserted[0]["sheet_row"] is None


def test_an_unparseable_file_is_a_422_not_a_500(client, monkeypatch):
    _mock_upload_deps(monkeypatch)
    r = _upload(client, uuid4(), "Name,City\nRavi,Hyderabad\n")
    assert r.status_code == 422
    assert "phone" in r.json()["detail"].lower()


def test_an_empty_upload_is_rejected(client, monkeypatch):
    _mock_upload_deps(monkeypatch)
    r = _upload(client, uuid4(), "")
    assert r.status_code == 400


def test_one_failing_row_does_not_abort_the_whole_import(client, monkeypatch):
    """A transient DB error on row 2 must not cost rows 3+ — the operator would
    have no way to tell which ones landed."""
    async def get_campaign(cid):
        return _campaign(campaign_id=cid)

    seen = []

    async def flaky_upsert(**kwargs):
        seen.append(kwargs["phone_e164"])
        if kwargs["phone_e164"] == "+919812345678":
            raise RuntimeError("connection reset")
        return _lead(**{k: v for k, v in kwargs.items() if k != "sheet_row"})

    monkeypatch.setattr(admin_leads.campaigns_db, "get_campaign", get_campaign)
    monkeypatch.setattr(admin_leads.leads_db, "upsert_lead", flaky_upsert)

    r = _upload(client, uuid4(),
                "Phone\n+919876543210\n+919812345678\n+919700000001\n")

    assert r.status_code == 200
    body = r.json()
    assert body["imported"] == 2
    assert body["skipped"] == 1
    assert len(seen) == 3, "must keep going past the failing row"


def test_upload_numbers_leads_by_their_position_so_calls_follow_file_order(
    client, monkeypatch
):
    """due_leads() orders by dial_order, so this is what makes a campaign call
    its list top-to-bottom the way the operator wrote it."""
    upserted = _mock_upload_deps(monkeypatch)

    _upload(client, uuid4(),
            "Name,Phone\nFirst,+919876543210\nSecond,9812345678\nThird,+919700000001\n")

    assert [(u["phone_e164"], u["dial_order"]) for u in upserted] == [
        ("+919876543210", 1), ("+919812345678", 2), ("+919700000001", 3),
    ]


def test_a_rejected_row_does_not_leave_a_gap_in_the_dial_order(client, monkeypatch):
    """Positions count ACCEPTED rows, not raw file lines — so fixing a bad row
    and re-uploading doesn't renumber everything after it."""
    upserted = _mock_upload_deps(monkeypatch)

    _upload(client, uuid4(),
            "Phone\n+919876543210\n12345\n9812345678\n")

    assert [u["dial_order"] for u in upserted] == [1, 2]
