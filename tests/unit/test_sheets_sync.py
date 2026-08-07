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


# ── language normalization ───────────────────────────────────────────────────

async def test_sync_normalizes_language_values_to_canonical_tokens(monkeypatch, campaign):
    """The DB now CHECKs language_pref against six tokens, so a hand-typed
    'Telugu' must become 'te' before it is inserted — otherwise the import
    raises on a perfectly ordinary spreadsheet."""
    records = [
        {"Name": "Asha", "Phone": "9876543210", "Campaign": campaign.name, "Language": "Telugu"},
        {"Name": "Ravi", "Phone": "9876543211", "Campaign": campaign.name, "Language": "tinglish"},
        {"Name": "Sita", "Phone": "9876543212", "Campaign": campaign.name, "Language": "Hindi"},
    ]
    _, upserted = await _patch(monkeypatch, records, campaign)

    n = await sync.sheets_sync()

    assert n == 3
    assert [u["language_pref"] for u in upserted] == ["te", "tinglish", "hi"]


async def test_sync_normalizes_unrecognised_language_to_auto(monkeypatch, campaign):
    """One junk cell must not fail the row — the lead still imports and dials
    in the campaign's language."""
    records = [
        {"Name": "Asha", "Phone": "9876543210", "Campaign": campaign.name, "Language": "Klingon"},
    ]
    _, upserted = await _patch(monkeypatch, records, campaign)

    n = await sync.sheets_sync()

    assert n == 1
    assert upserted[0]["language_pref"] == "auto"


async def test_sync_normalizes_missing_language_to_auto(monkeypatch, campaign):
    """A file with no Language column should default to auto."""
    records = [
        {"Name": "Asha", "Phone": "9876543210", "Campaign": campaign.name},
    ]
    _, upserted = await _patch(monkeypatch, records, campaign)

    n = await sync.sheets_sync()

    assert n == 1
    assert upserted[0]["language_pref"] == "auto"


# ── is the Sheet reachable at all? ───────────────────────────────────────────

def test_a_missing_credential_file_is_reported_as_a_reason(monkeypatch, tmp_path):
    """Distinguishes "you have not set this up" from "the API call failed",
    which the scheduler needs in order to stop logging nine per-lead ERRORs
    every ten minutes for one missing file."""
    from app.sheets import client as sheets_client
    monkeypatch.setattr(sheets_client, "GOOGLE_SHEET_ID", "sheet-1")
    monkeypatch.setattr(sheets_client, "GOOGLE_SERVICE_ACCOUNT_FILE",
                        str(tmp_path / "nope.json"))
    reason = sheets_client.unconfigured_reason()
    assert reason and "nope.json" in reason


def test_a_missing_sheet_id_is_reported_as_a_reason(monkeypatch):
    from app.sheets import client as sheets_client
    monkeypatch.setattr(sheets_client, "GOOGLE_SHEET_ID", "")
    reason = sheets_client.unconfigured_reason()
    assert reason and "GOOGLE_SHEET_ID" in reason


def test_a_sheet_url_is_rejected_in_favour_of_the_spreadsheet_id(monkeypatch):
    from app.sheets import client as sheets_client
    monkeypatch.setattr(sheets_client, "GOOGLE_SHEET_ID", "https://docs.google.com/spreadsheets/d/sheet-1/edit")
    reason = sheets_client.unconfigured_reason()
    assert reason and "spreadsheet ID" in reason


def test_a_fully_configured_sheet_reports_no_reason(monkeypatch, tmp_path):
    from app.sheets import client as sheets_client
    creds = tmp_path / "sa.json"
    creds.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sheets_client, "GOOGLE_SHEET_ID", "sheet-1")
    monkeypatch.setattr(sheets_client, "GOOGLE_SERVICE_ACCOUNT_FILE", str(creds))
    assert sheets_client.unconfigured_reason() is None
