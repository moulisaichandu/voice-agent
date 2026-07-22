"""Unit tests for sheets/sync.py — the worksheet is a plain stub object (its
methods run through asyncio.to_thread exactly like the real gspread.Worksheet
would, so this exercises the same code path with zero network calls)."""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.sheets import sync


class _StubWorksheet:
    def __init__(self, records):
        self._records = records

    def get_all_records(self):
        return self._records


@pytest.fixture
def campaign():
    return SimpleNamespace(campaign_id=uuid4(), name="Demo Class Outreach")


async def _patch(monkeypatch, records, campaign):
    ws = _StubWorksheet(records)

    async def fake_get_worksheet():
        return ws

    monkeypatch.setattr(sync, "get_worksheet", fake_get_worksheet)

    async def fake_get_campaign_by_name(name):
        return campaign if name == campaign.name else None

    monkeypatch.setattr(sync.campaigns_db, "get_campaign_by_name", fake_get_campaign_by_name)

    upserted = []

    async def fake_upsert_lead(**kwargs):
        upserted.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(sync.leads_db, "upsert_lead", fake_upsert_lead)
    return ws, upserted


async def test_sync_upserts_valid_rows(monkeypatch, campaign):
    records = [
        {"Name": "Ravi", "Phone": "9876543210", "Campaign": campaign.name},
        {"Name": "Sita", "Phone": "9812345678", "Campaign": campaign.name},
    ]
    _, upserted = await _patch(monkeypatch, records, campaign)

    n = await sync.sheets_sync()

    assert n == 2
    assert len(upserted) == 2
    assert upserted[0]["phone_e164"] == "+919876543210"
    assert upserted[0]["campaign_id"] == campaign.campaign_id


async def test_sync_skips_rows_with_no_valid_phone(monkeypatch, campaign):
    records = [
        {"Name": "Ravi", "Phone": "not-a-phone", "Campaign": campaign.name},
        {"Name": "Sita", "Phone": "9812345678", "Campaign": campaign.name},
    ]
    _, upserted = await _patch(monkeypatch, records, campaign)

    n = await sync.sheets_sync()
    assert n == 1
    assert upserted[0]["phone_e164"] == "+919812345678"


async def test_sync_skips_rows_with_no_campaign(monkeypatch, campaign):
    records = [{"Name": "Ravi", "Phone": "9876543210", "Campaign": ""}]
    _, upserted = await _patch(monkeypatch, records, campaign)

    n = await sync.sheets_sync()
    assert n == 0
    assert upserted == []


async def test_sync_skips_rows_with_unknown_campaign(monkeypatch, campaign):
    records = [{"Name": "Ravi", "Phone": "9876543210", "Campaign": "Nonexistent Campaign"}]
    _, upserted = await _patch(monkeypatch, records, campaign)

    n = await sync.sheets_sync()
    assert n == 0
    assert upserted == []


async def test_sync_is_case_insensitive_and_flexible_on_headers(monkeypatch, campaign):
    """Blueprint's §7.1 only describes expected columns in prose — headers
    must resolve by hint word, not an exact hardcoded string."""
    records = [{"Full Name": "Ravi", "Mobile Number": "9876543210",
               "Campaign": campaign.name}]
    _, upserted = await _patch(monkeypatch, records, campaign)

    n = await sync.sheets_sync()
    assert n == 1
    assert upserted[0]["name"] == "Ravi"
    assert upserted[0]["phone_e164"] == "+919876543210"


async def test_sync_empty_sheet_returns_zero(monkeypatch, campaign):
    _, upserted = await _patch(monkeypatch, [], campaign)
    n = await sync.sheets_sync()
    assert n == 0
    assert upserted == []


def test_find_column_prefers_exact_match_over_substring():
    # "contact" is a phone-hint word too, so "Contact Person" would ALSO
    # substring-match — an exact "phone" header must still win over it.
    assert sync.find_column(["phone", "Contact Person"], sync.PHONE_HINTS) == "phone"
