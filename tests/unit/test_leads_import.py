"""Unit tests for app/leads_import.py — pure parsing, no DB, no Docker.

The bar here is that a bad row is REPORTED rather than dropped: an operator who
uploads 200 numbers and dials 180 needs to know which 20 didn't make it and why,
or the 20 just silently never get called.
"""

import io

import pytest
from openpyxl import Workbook

from app import leads_import


def _csv(text: str) -> bytes:
    return text.encode("utf-8")


def _xlsx(rows: list[list[object]]) -> bytes:
    wb = Workbook()
    ws = wb.active
    for row in rows:
        ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ── happy path ───────────────────────────────────────────────────────────────

def test_parses_a_simple_csv():
    parsed = leads_import.parse_leads_file("leads.csv", _csv(
        "Name,Phone\n"
        "Ravi,+919876543210\n"
        "Sita,9812345678\n"
    ))
    assert [lead.phone_e164 for lead in parsed.leads] == ["+919876543210", "+919812345678"]
    assert [lead.name for lead in parsed.leads] == ["Ravi", "Sita"]
    assert parsed.errors == []
    assert parsed.received == 2


def test_normalizes_every_indian_phone_format_the_dial_path_accepts():
    """Same normalizer the worker and the DND scrub use, so the console can
    never disagree with what actually gets dialled."""
    parsed = leads_import.parse_leads_file("leads.csv", _csv(
        "Phone\n"
        "+919876543210\n"
        "09812345678\n"
        "919700000001\n"
        "98 76 54 32 19\n"
    ))
    assert [lead.phone_e164 for lead in parsed.leads] == [
        "+919876543210", "+919812345678", "+919700000001", "+919876543219",
    ]


def test_finds_the_phone_column_by_hint_word_not_exact_header():
    """Real sheets say "Mobile No", "WhatsApp Number", "Contact" — the same
    hint-based detection sheets_sync uses."""
    for header in ("Mobile No", "WhatsApp Number", "Contact", "PHONE"):
        parsed = leads_import.parse_leads_file("leads.csv", _csv(f"{header}\n+919876543210\n"))
        assert [lead.phone_e164 for lead in parsed.leads] == ["+919876543210"], header


def test_parses_xlsx():
    data = _xlsx([["Name", "Phone"], ["Ravi", "+919876543210"], ["Sita", 9812345678]])
    parsed = leads_import.parse_leads_file("leads.xlsx", data)
    assert [lead.phone_e164 for lead in parsed.leads] == ["+919876543210", "+919812345678"]


def test_strips_the_excel_utf8_bom():
    """Excel's "CSV UTF-8" export prefixes a BOM, which would otherwise become
    part of the first header and stop it matching any hint word."""
    parsed = leads_import.parse_leads_file("leads.csv", "﻿Phone\n+919876543210\n".encode())
    assert [lead.phone_e164 for lead in parsed.leads] == ["+919876543210"]


def test_handles_semicolon_delimited_csv():
    parsed = leads_import.parse_leads_file("leads.csv", _csv(
        "Name;Phone\nRavi;+919876543210\n"
    ))
    assert [lead.phone_e164 for lead in parsed.leads] == ["+919876543210"]


# ── per-row rejections ───────────────────────────────────────────────────────

def test_a_bad_number_is_reported_with_its_row_number_not_dropped():
    parsed = leads_import.parse_leads_file("leads.csv", _csv(
        "Name,Phone\n"
        "Ravi,+919876543210\n"
        "Broken,12345\n"
        "Empty,\n"
    ))
    assert [lead.phone_e164 for lead in parsed.leads] == ["+919876543210"]
    assert [(e.row_number, "12345" in e.reason) for e in parsed.errors] == [(3, True), (4, False)]
    assert parsed.received == 3


def test_duplicate_numbers_within_one_file_are_reported_once():
    """upsert_lead would collapse these silently; naming the earlier row makes
    a copy-paste mistake visible instead."""
    parsed = leads_import.parse_leads_file("leads.csv", _csv(
        "Phone\n+919876543210\n09876543210\n"
    ))
    assert len(parsed.leads) == 1
    assert len(parsed.errors) == 1
    assert "duplicate of row 2" in parsed.errors[0].reason


# ── consent ──────────────────────────────────────────────────────────────────

def test_consent_columns_are_read_when_present():
    parsed = leads_import.parse_leads_file("leads.csv", _csv(
        "Phone,Consent Basis,Consent At\n"
        "+919876543210,explicit,2026-07-01 10:00:00\n"
    ))
    lead = parsed.leads[0]
    assert lead.consent_basis == "explicit"
    assert lead.consent_at is not None


def test_no_consent_columns_means_no_consent_not_an_assumed_one():
    """Uploading a file is not consent. These import and simply don't dial."""
    parsed = leads_import.parse_leads_file("leads.csv", _csv("Phone\n+919876543210\n"))
    assert parsed.leads[0].consent_basis is None
    assert parsed.leads[0].consent_at is None


def test_a_lone_consent_at_column_is_not_mistaken_for_the_basis():
    """"Consent At" contains "consent", so it substring-matches the BASIS hints
    too. Feeding a timestamp in as the basis makes every lead permanently
    dial-ineligible, silently — the same trap sheets_sync has."""
    parsed = leads_import.parse_leads_file("leads.csv", _csv(
        "Phone,Consent At\n+919876543210,2026-07-01 10:00:00\n"
    ))
    lead = parsed.leads[0]
    assert lead.consent_basis is None
    assert lead.consent_at is not None


# ── whole-file failures ──────────────────────────────────────────────────────

def test_no_phone_column_is_a_file_level_error_naming_what_it_found():
    with pytest.raises(leads_import.LeadsFileError) as exc:
        leads_import.parse_leads_file("leads.csv", _csv("Name,City\nRavi,Hyderabad\n"))
    assert "phone" in str(exc.value).lower()
    assert "City" in str(exc.value)


def test_unsupported_extension_is_rejected():
    with pytest.raises(leads_import.LeadsFileError, match="Unsupported file type"):
        leads_import.parse_leads_file("leads.pdf", b"%PDF-1.4")


def test_empty_file_is_rejected():
    with pytest.raises(leads_import.LeadsFileError, match="empty"):
        leads_import.parse_leads_file("leads.csv", b"")


def test_header_only_file_is_rejected():
    with pytest.raises(leads_import.LeadsFileError, match="no data rows"):
        leads_import.parse_leads_file("leads.csv", _csv("Name,Phone\n"))


def test_a_file_over_the_row_cap_is_rejected_rather_than_parsed():
    rows = "\n".join(f"+9198765{i:05d}" for i in range(leads_import.MAX_ROWS + 1))
    with pytest.raises(leads_import.LeadsFileError, match="limit"):
        leads_import.parse_leads_file("leads.csv", _csv(f"Phone\n{rows}\n"))


# ── language normalization ───────────────────────────────────────────────────

def test_a_language_column_is_normalized_to_a_canonical_token():
    """The DB now CHECKs language_pref against six tokens, so a hand-typed
    'Telugu' must become 'te' before it is inserted — otherwise the import
    raises on a perfectly ordinary spreadsheet."""
    csv = (
        "Name,Phone,Language\n"
        "Asha,9876543210,Telugu\n"
        "Ravi,9876543211,tinglish\n"
        "Sita,9876543212,Hindi\n"
    )
    parsed = leads_import.parse_leads_file("leads.csv", csv.encode())
    assert [lead.language_pref for lead in parsed.leads] == ["te", "tinglish", "hi"]


def test_an_unrecognised_language_value_imports_as_auto():
    """One junk cell must not fail the row — the lead still imports and dials
    in the campaign's language."""
    csv = "Name,Phone,Language\nAsha,9876543210,Klingon\n"
    parsed = leads_import.parse_leads_file("leads.csv", csv.encode())
    assert parsed.leads[0].language_pref == "auto"


def test_a_file_with_no_language_column_imports_as_auto():
    csv = "Name,Phone\nAsha,9876543210\n"
    parsed = leads_import.parse_leads_file("leads.csv", csv.encode())
    assert parsed.leads[0].language_pref == "auto"
