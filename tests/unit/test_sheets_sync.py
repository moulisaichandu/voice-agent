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

    # Models get_campaign_by_name's SQL, which matches on lower(btrim(name)).
    # A fake comparing exactly would let a case-sensitivity regression pass
    # here while still failing against Postgres.
    async def fake_get_campaign_by_name(name):
        return campaign if name.strip().lower() == campaign.name.lower() else None

    monkeypatch.setattr(sync.campaigns_db, "get_campaign_by_name", fake_get_campaign_by_name)

    upserted = []

    async def fake_upsert_lead(**kwargs):
        upserted.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(sync.leads_db, "upsert_lead", fake_upsert_lead)
    return ws, upserted


async def test_sync_reads_through_the_apps_script_transport(monkeypatch, campaign):
    """When GOOGLE_SHEET_ID is the deployed logger's /exec URL, rows come
    from its get_leads action — gspread is never touched, and the script's
    own _row numbering (which skips blank rows) is what lands in sheet_row,
    not an enumerate() guess."""
    monkeypatch.setattr(sync, "GOOGLE_SHEET_ID",
                        "https://script.google.com/macros/s/x/exec")

    async def fake_fetch_rows(worksheet_name):
        assert worksheet_name == sync.LEADS_WORKSHEET_NAME
        return [(7, {"Name": "Ravi", "Phone": "9876543210",
                     "Campaign": campaign.name})]

    monkeypatch.setattr(sync.apps_script, "fetch_rows", fake_fetch_rows)

    async def never_gspread():
        raise AssertionError("the gspread path must not run for a script URL")

    monkeypatch.setattr(sync, "get_worksheet", never_gspread)

    async def fake_get_campaign_by_name(name):
        return campaign if name.strip().lower() == campaign.name.lower() else None

    monkeypatch.setattr(sync.campaigns_db, "get_campaign_by_name",
                        fake_get_campaign_by_name)
    upserted = []

    async def fake_upsert_lead(**kwargs):
        upserted.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(sync.leads_db, "upsert_lead", fake_upsert_lead)

    result = await sync.sheets_sync()

    assert result.imported == 1
    assert upserted[0]["phone_e164"] == "+919876543210"
    assert upserted[0]["sheet_row"] == 7, "the script's real row number was discarded"


async def test_sync_upserts_valid_rows(monkeypatch, campaign):
    records = [
        {"Name": "Ravi", "Phone": "9876543210", "Campaign": campaign.name},
        {"Name": "Sita", "Phone": "9812345678", "Campaign": campaign.name},
    ]
    _, upserted = await _patch(monkeypatch, records, campaign)

    result = await sync.sheets_sync()

    assert result.imported == 2
    assert len(upserted) == 2
    assert upserted[0]["phone_e164"] == "+919876543210"
    assert upserted[0]["campaign_id"] == campaign.campaign_id


async def test_sync_skips_rows_with_no_valid_phone(monkeypatch, campaign):
    records = [
        {"Name": "Ravi", "Phone": "not-a-phone", "Campaign": campaign.name},
        {"Name": "Sita", "Phone": "9812345678", "Campaign": campaign.name},
    ]
    _, upserted = await _patch(monkeypatch, records, campaign)

    result = await sync.sheets_sync()
    assert result.imported == 1
    assert upserted[0]["phone_e164"] == "+919812345678"


async def test_sync_skips_rows_with_no_campaign(monkeypatch, campaign):
    records = [{"Name": "Ravi", "Phone": "9876543210", "Campaign": ""}]
    _, upserted = await _patch(monkeypatch, records, campaign)

    result = await sync.sheets_sync()
    assert result.imported == 0
    assert upserted == []


async def test_sync_skips_rows_with_unknown_campaign(monkeypatch, campaign):
    records = [{"Name": "Ravi", "Phone": "9876543210", "Campaign": "Nonexistent Campaign"}]
    _, upserted = await _patch(monkeypatch, records, campaign)

    result = await sync.sheets_sync()
    assert result.imported == 0
    assert upserted == []


async def test_sync_is_case_insensitive_and_flexible_on_headers(monkeypatch, campaign):
    """Blueprint's §7.1 only describes expected columns in prose — headers
    must resolve by hint word, not an exact hardcoded string."""
    records = [{"Full Name": "Ravi", "Mobile Number": "9876543210",
               "Campaign": campaign.name}]
    _, upserted = await _patch(monkeypatch, records, campaign)

    result = await sync.sheets_sync()
    assert result.imported == 1
    assert upserted[0]["name"] == "Ravi"
    assert upserted[0]["phone_e164"] == "+919876543210"


async def test_sync_empty_sheet_returns_zero(monkeypatch, campaign):
    _, upserted = await _patch(monkeypatch, [], campaign)
    result = await sync.sheets_sync()
    assert result.imported == 0
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

    result = await sync.sheets_sync()

    assert result.imported == 3
    assert [u["language_pref"] for u in upserted] == ["te", "tinglish", "hi"]


async def test_sync_normalizes_unrecognised_language_to_auto(monkeypatch, campaign):
    """One junk cell must not fail the row — the lead still imports and dials
    in the campaign's language."""
    records = [
        {"Name": "Asha", "Phone": "9876543210", "Campaign": campaign.name, "Language": "Klingon"},
    ]
    _, upserted = await _patch(monkeypatch, records, campaign)

    result = await sync.sheets_sync()

    assert result.imported == 1
    assert upserted[0]["language_pref"] == "auto"


async def test_sync_normalizes_missing_language_to_auto(monkeypatch, campaign):
    """A file with no Language column should default to auto."""
    records = [
        {"Name": "Asha", "Phone": "9876543210", "Campaign": campaign.name},
    ]
    _, upserted = await _patch(monkeypatch, records, campaign)

    result = await sync.sheets_sync()

    assert result.imported == 1
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


# ── what the sync reports, and the columns it must not confuse ───────────────

async def test_a_consent_at_column_alone_is_not_read_as_the_basis(
        monkeypatch, campaign):
    """A sheet carrying only "Consent At" must not have its TIMESTAMP filed as
    the consent BASIS.

    CONSENT_BASIS_HINTS contains the bare word "consent", so find_column's
    substring pass matches "Consent At" for both roles. The lead then holds a
    basis of "2026-08-01" — not in VALID_CONSENT_BASES — and is permanently
    undialable with nothing logged. app/leads_import.py already guards this
    (its own comment describes the same trap); the two importers must not
    disagree about a compliance field.
    """
    records = [{"Name": "Ravi", "Phone": "9876543210",
                "Campaign": campaign.name, "Consent At": "2026-08-01"}]
    _, upserted = await _patch(monkeypatch, records, campaign)

    await sync.sheets_sync()

    assert upserted[0]["consent_basis"] is None, (
        "the consent timestamp was filed as the consent basis")
    assert upserted[0]["consent_at"] is not None, "the timestamp itself was lost"


async def test_sheet_rows_keep_their_dialling_order(monkeypatch, campaign):
    """due_leads() orders by `dial_order nulls last`, so a sync that leaves it
    None puts every sheet lead behind every uploaded one — the sheet's own
    top-to-bottom order is the operator's stated priority."""
    records = [
        {"Name": "First", "Phone": "9876543210", "Campaign": campaign.name},
        {"Name": "Second", "Phone": "9812345678", "Campaign": campaign.name},
    ]
    _, upserted = await _patch(monkeypatch, records, campaign)

    await sync.sheets_sync()

    assert [u["dial_order"] for u in upserted] == [2, 3], (
        "sheet order was not carried into dial_order")


async def test_the_result_says_how_many_rows_can_actually_dial(
        monkeypatch, campaign):
    """The operator's real question is never "how many rows synced" but "how
    many will the dialer call". A sheet with no consent columns imports every
    row and dials none, which read as a healthy sync for weeks."""
    records = [
        {"Name": "Ravi", "Phone": "9876543210", "Campaign": campaign.name,
         "Consent Basis": "explicit", "Consent At": "2026-08-01"},
        {"Name": "Sita", "Phone": "9812345678", "Campaign": campaign.name},
        {"Name": "NoPhone", "Phone": "", "Campaign": campaign.name},
    ]
    await _patch(monkeypatch, records, campaign)

    result = await sync.sheets_sync()

    assert result.received == 3
    assert result.imported == 2
    assert result.skipped == 1
    assert result.dialable == 1, "only the consented row may count as dialable"
    assert result.skips.get("no phone") == 1


async def test_a_campaign_name_matches_despite_case_and_spacing(
        monkeypatch, campaign):
    """The console generates campaign names the operator then retypes into the
    Sheet. An exact, case-sensitive comparison turns a stray capital into a
    silently skipped row."""
    records = [{"Name": "Ravi", "Phone": "9876543210",
                "Campaign": f"  {campaign.name.upper()}  "}]
    _, upserted = await _patch(monkeypatch, records, campaign)

    result = await sync.sheets_sync()

    assert result.imported == 1, "a case/whitespace variant was treated as unknown"
    assert upserted[0]["campaign_id"] == campaign.campaign_id
