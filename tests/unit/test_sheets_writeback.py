"""Unit tests for sheets/writeback.py — the phone-verification-before-trusting
-sheet_row pattern this module exists to fix. The worksheet is a stub that
tracks batch_update calls; all Worksheet methods run through asyncio.to_thread
exactly like the real gspread client, so this exercises the same code path.
"""

from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

import gspread.utils as gsutils

from app.db.models import TranscriptTurn
from app.sheets import writeback

# ── the Apps Script transport ───────────────────────────────────────────────

def _script_call():
    return SimpleNamespace(
        call_id=uuid4(), status="done", turns=2,
        ended_at=datetime(2026, 8, 27, 13, 0, 0),
        transcript=[TranscriptTurn(role="agent", text="నమస్తే."),
                    TranscriptTurn(role="lead", text="హలో.")],
        summary=None,
    )


def _patch_script_branch(monkeypatch, lead, call, *, export_ok=True):
    monkeypatch.setattr(writeback, "GOOGLE_SHEET_ID",
                        "https://script.google.com/macros/s/x/exec")

    async def fake_get_lead(lead_id):
        return lead

    async def fake_latest(lead_id):
        return call

    monkeypatch.setattr(writeback.leads_db, "get_lead", fake_get_lead)
    monkeypatch.setattr(writeback.calls_db, "get_latest_call_for_lead", fake_latest)

    async def never_gspread():
        raise AssertionError("gspread must not run for a script URL")

    monkeypatch.setattr(writeback, "get_worksheet", never_gspread)
    exported, updated = [], []

    async def fake_export(*, session, turns, timestamp):
        exported.append((session, list(turns or []), timestamp))
        return export_ok

    async def fake_update(*, phone, status, notes=""):
        updated.append((phone, status, notes))
        return True

    monkeypatch.setattr(writeback.apps_script, "export_transcript", fake_export)
    monkeypatch.setattr(writeback.apps_script, "update_lead_status", fake_update)
    return exported, updated


async def test_write_back_speaks_the_apps_script_protocol(monkeypatch):
    """Script URL configured: the transcript exports to the Transcripts tab
    keyed by the call id, the Leads-tab status update rides along
    best-effort, and gspread is never touched."""
    lead = _lead()
    call = _script_call()
    exported, updated = _patch_script_branch(monkeypatch, lead, call)

    ok = await writeback.write_back_lead(lead.lead_id)

    assert ok is True
    session, turns, timestamp = exported[0]
    assert session == str(call.call_id)[:8]
    assert turns == call.transcript
    assert timestamp == "2026-08-27 13:00:00"
    assert updated and updated[0][0] == lead.phone_e164
    assert updated[0][1] == "done"


async def test_a_failed_export_requests_a_retry_and_skips_the_status(monkeypatch):
    """Only the export's failure may request a retry: export_session appends
    blindly, so retrying after a SUCCESSFUL export would write the whole
    transcript into the sheet twice."""
    lead = _lead()
    exported, updated = _patch_script_branch(
        monkeypatch, lead, _script_call(), export_ok=False)

    ok = await writeback.write_back_lead(lead.lead_id)

    assert ok is False, "a failed export must be retried by the next sweep"
    assert updated == [], "no status write for a transcript that never landed"


class _FakeRedis:
    """Just the two calls the exported-marker needs."""

    def __init__(self):
        self.store: dict[str, str] = {}

    async def exists(self, key):
        return 1 if key in self.store else 0

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True


async def test_an_aware_timestamp_lands_in_the_operators_timezone(monkeypatch):
    """calls.ended_at is a UTC timestamptz; the sheet's timezone is the
    operator's (Asia/Kolkata). Sending the UTC wall time verbatim showed
    every call 5½ hours early in the Timestamp column (verified on the live
    sheet 2026-08-27)."""
    from datetime import timezone

    lead = _lead()
    call = _script_call()
    call.ended_at = datetime(2026, 8, 27, 7, 19, 9, tzinfo=timezone.utc)
    exported, _updated = _patch_script_branch(monkeypatch, lead, call)

    await writeback.write_back_lead(lead.lead_id)

    assert exported[0][2] == "2026-08-27 12:49:09", (
        f"UTC leaked into the sheet: {exported[0][2]!r}"
    )


async def test_the_same_call_is_never_exported_twice(monkeypatch):
    """transcript_reconcile is at-least-once BY DESIGN (reserve-then-ack, and
    the queue can hold the same lead twice for one call). The gspread path
    overwrote cells, so re-runs were harmless; export_session APPENDS, so
    without a marker every re-run pastes the whole transcript into the sheet
    again. The marker is per call_id and set only after a successful export."""
    lead = _lead()
    call = _script_call()
    exported, updated = _patch_script_branch(monkeypatch, lead, call)
    fake = _FakeRedis()
    monkeypatch.setattr(writeback.redis_client, "get_redis", lambda: fake)

    assert await writeback.write_back_lead(lead.lead_id) is True
    assert await writeback.write_back_lead(lead.lead_id) is True

    assert len(exported) == 1, "the second drain re-appended the transcript"


async def test_a_failed_export_does_not_mark_the_call_as_done(monkeypatch):
    """The marker lands AFTER success — a marker set before the export would
    turn one transient failure into a transcript that never reaches the
    sheet at all."""
    lead = _lead()
    call = _script_call()
    exported, _updated = _patch_script_branch(
        monkeypatch, lead, call, export_ok=False)
    fake = _FakeRedis()
    monkeypatch.setattr(writeback.redis_client, "get_redis", lambda: fake)

    assert await writeback.write_back_lead(lead.lead_id) is False
    assert fake.store == {}, "a failed export left a marker behind"


class _StubWorksheet:
    def __init__(self, headers, rows):
        """rows: list of row-value-lists, index 0 = row 2 (row 1 is headers)."""
        self._headers = headers
        self._rows = rows
        self.batch_update_calls = []
        self.update_calls = []

    def row_values(self, row):
        if row == 1:
            return self._headers
        return self._rows[row - 2]

    def col_values(self, col):
        return [self._headers[col - 1]] + [
            r[col - 1] if col - 1 < len(r) else "" for r in self._rows
        ]

    def cell(self, row, col):
        val = self._rows[row - 2][col - 1] if row > 1 else self._headers[col - 1]
        return SimpleNamespace(value=val)

    def update(self, values, range_name):
        self.update_calls.append((values, range_name))
        # Simulate the new columns actually landing in the header row.
        self._headers.extend(values[0])

    def batch_update(self, updates):
        self.batch_update_calls.append(updates)


def _lead(**overrides):
    defaults = dict(
        lead_id=uuid4(), sheet_row=2, name="Ravi", phone_e164="+919876543210",
        campaign_id=uuid4(), language_pref="auto", consent_basis=None, consent_at=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _call(**overrides):
    defaults = dict(
        status="done", turns=2, ended_at=None, summary="Interested.",
        transcript=[TranscriptTurn(role="agent", text="నమస్కారం!"),
                   TranscriptTurn(role="lead", text="hello")],
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


async def _patch_common(monkeypatch, ws, lead, call):
    async def fake_get_worksheet():
        return ws

    monkeypatch.setattr(writeback, "get_worksheet", fake_get_worksheet)

    async def fake_get_lead(lead_id):
        return lead

    async def fake_get_latest_call(lead_id):
        return call

    monkeypatch.setattr(writeback.leads_db, "get_lead", fake_get_lead)
    monkeypatch.setattr(writeback.calls_db, "get_latest_call_for_lead", fake_get_latest_call)

    upserted = []

    async def fake_upsert(**kwargs):
        upserted.append(kwargs)

    monkeypatch.setattr(writeback.leads_db, "upsert_lead", fake_upsert)
    return upserted


def _values_by_col_letter(updates: list[dict]) -> dict[str, object]:
    """{'C': value, 'D': value, ...} — strips the row number so assertions
    don't hardcode which row a test happens to use."""
    out = {}
    for u in updates:
        col_letter = "".join(ch for ch in u["range"] if ch.isalpha())
        out[col_letter] = u["values"][0][0]
    return out


async def test_write_back_uses_the_sheet_row_hint_when_it_still_matches(monkeypatch):
    ws = _StubWorksheet(
        headers=["Name", "Phone"],
        rows=[["Ravi", "+919876543210"]],  # row 2 — matches the hint
    )
    lead = _lead(sheet_row=2)
    call = _call()
    upserted = await _patch_common(monkeypatch, ws, lead, call)

    ok = await writeback.write_back_lead(lead.lead_id)

    assert ok is True
    assert upserted == []  # hint was correct — no re-sync of sheet_row needed
    assert len(ws.batch_update_calls) == 1
    ranges = [u["range"] for u in ws.batch_update_calls[0]]
    # New columns were appended at C..G (after Name, Phone); row 2 throughout.
    assert ranges == [
        gsutils.rowcol_to_a1(2, 3), gsutils.rowcol_to_a1(2, 4), gsutils.rowcol_to_a1(2, 5),
        gsutils.rowcol_to_a1(2, 6), gsutils.rowcol_to_a1(2, 7),
    ]


async def test_write_back_falls_back_to_full_scan_when_hint_is_stale(monkeypatch):
    """A row was inserted above the lead's original row — sheet_row=2 now
    points at a DIFFERENT lead. Must NOT write to the wrong row."""
    ws = _StubWorksheet(
        headers=["Name", "Phone"],
        rows=[
            ["Someone Else", "+919999999999"],  # now at row 2 (stale hint)
            ["Ravi", "+919876543210"],           # actually at row 3
        ],
    )
    lead = _lead(sheet_row=2)  # stale
    call = _call()
    upserted = await _patch_common(monkeypatch, ws, lead, call)

    ok = await writeback.write_back_lead(lead.lead_id)

    assert ok is True
    # The stale hint must be corrected, not silently left wrong.
    assert len(upserted) == 1
    assert upserted[0]["sheet_row"] == 3
    # And the actual Sheet write must target row 3, not the stale row 2.
    ranges = [u["range"] for u in ws.batch_update_calls[0]]
    assert ranges[0] == gsutils.rowcol_to_a1(3, 3)  # Status column, row 3
    assert all(gsutils.rowcol_to_a1(2, c) not in ranges for c in range(3, 8))


async def test_write_back_appends_missing_writeback_columns(monkeypatch):
    ws = _StubWorksheet(headers=["Name", "Phone"], rows=[["Ravi", "+919876543210"]])
    lead = _lead()
    call = _call()
    await _patch_common(monkeypatch, ws, lead, call)

    await writeback.write_back_lead(lead.lead_id)

    assert ws._headers[-5:] == ["Status", "Turns", "CalledAt", "Transcript", "Summary"]
    # Appended once, at C1:G1 — not overwriting the existing Name/Phone columns.
    # gspread's update() takes a 2D structure (list of rows); one row here.
    assert ws.update_calls == [
        ([["Status", "Turns", "CalledAt", "Transcript", "Summary"]],
         f"{gsutils.rowcol_to_a1(1, 3)}:{gsutils.rowcol_to_a1(1, 7)}"),
    ]


async def test_write_back_writes_correct_values_including_telugu_transcript(monkeypatch):
    ws = _StubWorksheet(headers=["Name", "Phone"], rows=[["Ravi", "+919876543210"]])
    lead = _lead()
    call = _call()
    await _patch_common(monkeypatch, ws, lead, call)

    await writeback.write_back_lead(lead.lead_id)

    by_col = _values_by_col_letter(ws.batch_update_calls[0])
    assert by_col["C"] == "done"    # Status
    assert by_col["D"] == 2         # Turns
    assert by_col["E"] == ""        # CalledAt — call.ended_at is None here
    assert by_col["F"] == "agent: నమస్కారం!\nlead: hello"  # Transcript
    assert by_col["G"] == "Interested."  # Summary


async def test_write_back_returns_false_when_phone_not_found_anywhere(monkeypatch):
    ws = _StubWorksheet(headers=["Name", "Phone"], rows=[["Someone", "+911111111111"]])
    lead = _lead(sheet_row=None, phone_e164="+919876543210")  # not in the sheet at all
    call = _call()
    await _patch_common(monkeypatch, ws, lead, call)

    ok = await writeback.write_back_lead(lead.lead_id)
    assert ok is False
    assert ws.batch_update_calls == []


async def test_write_back_no_call_yet_is_treated_as_nothing_to_do(monkeypatch):
    ws = _StubWorksheet(headers=["Name", "Phone"], rows=[["Ravi", "+919876543210"]])
    lead = _lead()
    await _patch_common(monkeypatch, ws, lead, call=None)

    ok = await writeback.write_back_lead(lead.lead_id)
    assert ok is True
    assert ws.batch_update_calls == []


async def test_write_back_missing_lead_does_not_crash(monkeypatch):
    async def fake_get_lead(lead_id):
        return None

    monkeypatch.setattr(writeback.leads_db, "get_lead", fake_get_lead)
    ok = await writeback.write_back_lead(uuid4())
    assert ok is True
