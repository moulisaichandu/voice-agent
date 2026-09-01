"""app/sheets/apps_script.py — the Apps Script transport.

Discovered live 2026-08-27: GOOGLE_SHEET_ID held not a spreadsheet ID but the
/exec URL of a deployed "Voice Agent Transcript Logger" — the sibling
project's Code.gs, already wired to the operator's sheet. Demanding a service
account + bare ID would mean re-plumbing infrastructure that already works,
so the sheets layer speaks the script's own POST protocol instead. These
tests mock the single HTTP seam (_http_post_json); the protocol shapes are
taken verbatim from ../ai-voice-agent/backend/Code.gs (read-only reference).
"""

from types import SimpleNamespace

import pytest

from app.sheets import apps_script

# ── transport detection ─────────────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    ("https://script.google.com/macros/s/AKfycb123/exec", True),
    ("http://script.google.com/macros/s/x/exec", True),
    ("1AbC-dEf_1234567890", False),                       # bare sheet ID
    ("https://docs.google.com/spreadsheets/d/1AbC/edit", False),
    ("", False),
    (None, False),
])
def test_only_script_urls_select_this_transport(value, expected):
    assert apps_script.is_apps_script(value) is expected


# ── the POST envelope ───────────────────────────────────────────────────────

def _capture_posts(monkeypatch, reply):
    posts = []

    async def fake_http(url, payload):
        posts.append((url, payload))
        return reply if not isinstance(reply, Exception) else (_ for _ in ()).throw(reply)

    monkeypatch.setattr(apps_script, "_http_post_json", fake_http)
    return posts


async def test_the_shared_token_is_sent_only_when_configured(monkeypatch):
    """Code.gs validates data.token against its SHEETS_TOKEN script property;
    when our side has no token there must be no token FIELD either — an empty
    string would fail a script whose property IS set."""
    monkeypatch.setattr(apps_script, "GOOGLE_SHEET_ID", "https://script.google.com/macros/s/x/exec")
    posts = _capture_posts(monkeypatch, {"status": "ok", "leads": []})

    monkeypatch.setattr(apps_script, "SHEETS_TOKEN", "")
    await apps_script.fetch_rows("Leads")
    assert "token" not in posts[-1][1]

    monkeypatch.setattr(apps_script, "SHEETS_TOKEN", "s3cret")
    await apps_script.fetch_rows("Leads")
    assert posts[-1][1]["token"] == "s3cret"
    assert posts[-1][0] == "https://script.google.com/macros/s/x/exec"


async def test_a_cold_start_404_is_retried_within_the_call(monkeypatch):
    """Observed live 2026-08-27, twice: each container's FIRST sweep after
    idle got HTTP 404 (non-JSON) from the script, while probes seconds later
    succeeded — Apps Script cold-start flakiness. The sweep cadence cannot be
    the retry, because 10-30 minutes later the script is COLD AGAIN, so
    without an in-call retry every sweep can fail forever while manual
    probes all pass. The sibling's client retries with backoff for exactly
    this reason; the first attempt warms the script, the retry lands."""
    monkeypatch.setattr(apps_script, "GOOGLE_SHEET_ID", "https://script.google.com/x/exec")
    monkeypatch.setattr(apps_script, "_RETRY_BACKOFF_S", 0.01)
    attempts = []

    async def flaky_http(url, payload):
        attempts.append(payload["action"])
        if len(attempts) == 1:
            raise RuntimeError("Apps Script returned non-JSON (HTTP 404)")
        return {"status": "ok", "leads": []}

    monkeypatch.setattr(apps_script, "_http_post_json", flaky_http)

    assert await apps_script.fetch_rows("Leads") == []
    assert len(attempts) == 2


async def test_an_application_level_error_is_not_retried(monkeypatch):
    """{"status":"error"} is a VALID response (auth failure, bad action) —
    retrying it re-runs a possibly-completed action for no benefit and slows
    every real failure by the full backoff."""
    monkeypatch.setattr(apps_script, "GOOGLE_SHEET_ID", "https://script.google.com/x/exec")
    monkeypatch.setattr(apps_script, "_RETRY_BACKOFF_S", 0.01)
    posts = _capture_posts(monkeypatch, {"status": "error", "message": "Unauthorized."})

    with pytest.raises(RuntimeError, match="Unauthorized"):
        await apps_script.fetch_rows("Leads")
    assert len(posts) == 1


# ── lead import ─────────────────────────────────────────────────────────────

async def test_fetch_rows_carries_the_scripts_own_row_numbers(monkeypatch):
    """Code.gs's _getLeads skips blank rows and reports each row's REAL sheet
    row as _row — enumerate(start=2) would mis-number every row after a gap,
    which is exactly the stale-sheet_row corruption writeback.py guards
    against. The script's numbering is authoritative."""
    monkeypatch.setattr(apps_script, "GOOGLE_SHEET_ID", "https://script.google.com/x/exec")
    _capture_posts(monkeypatch, {"status": "ok", "leads": [
        {"_row": 2, "Name": "Ravi", "Phone": "9876543210"},
        {"_row": 5, "Name": "Sita", "Phone": "9812345678"},   # rows 3-4 blank
        "not-a-dict",
    ]})

    rows = await apps_script.fetch_rows("Leads")

    assert rows == [
        (2, {"Name": "Ravi", "Phone": "9876543210"}),
        (5, {"Name": "Sita", "Phone": "9812345678"}),
    ]


async def test_fetch_rows_raises_on_script_error_so_the_operator_sees_it(monkeypatch):
    """Confirmed by the 2026-08-27 review: degrading to [] here made the
    script transport invisible to /admin/sheets-status — sheets_sync
    returned a healthy 0 and the scheduler stamped error:null while lead
    ingestion was hard-down (revoked deployment, token mismatch). fetch_rows'
    only caller is the scheduler job, which catches a raise, logs at ERROR
    and records it as last_error — so raising IS the degrade path."""
    monkeypatch.setattr(apps_script, "GOOGLE_SHEET_ID", "https://script.google.com/x/exec")
    _capture_posts(monkeypatch, {"status": "error", "message": "Unauthorized."})

    with pytest.raises(RuntimeError, match="Unauthorized"):
        await apps_script.fetch_rows("Leads")


async def test_fetch_rows_propagates_transport_failures(monkeypatch):
    monkeypatch.setattr(apps_script, "GOOGLE_SHEET_ID", "https://script.google.com/x/exec")
    _capture_posts(monkeypatch, RuntimeError("connect timeout"))

    with pytest.raises(RuntimeError, match="connect timeout"):
        await apps_script.fetch_rows("Leads")


# ── transcript export ───────────────────────────────────────────────────────

def _turns():
    return [
        SimpleNamespace(role="agent", text="నమస్తే, ఇది AI కాల్."),
        SimpleNamespace(role="lead", text="ఫీజు ఎంత?"),
        SimpleNamespace(role="agent", text="ఇరవై ఐదు వేల రూపాయలు."),
    ]


async def test_export_transcript_speaks_the_export_session_shape(monkeypatch):
    """Code.gs's _exportSession counts a new Turn # on role=='user' and writes
    the role verbatim — so our lead/agent roles map to user/assistant, the
    same vocabulary the sibling's own client sends."""
    monkeypatch.setattr(apps_script, "GOOGLE_SHEET_ID", "https://script.google.com/x/exec")
    posts = _capture_posts(monkeypatch, {"status": "ok", "rows_added": 3})

    ok = await apps_script.export_transcript(
        session="abc12345", turns=_turns(), timestamp="2026-08-27 13:00:00")

    assert ok is True
    payload = posts[-1][1]
    assert payload["action"] == "export_session"
    assert payload["session"] == "abc12345"
    assert payload["timestamp"] == "2026-08-27 13:00:00"
    assert payload["history"] == [
        {"role": "assistant", "content": "నమస్తే, ఇది AI కాల్."},
        {"role": "user", "content": "ఫీజు ఎంత?"},
        {"role": "assistant", "content": "ఇరవై ఐదు వేల రూపాయలు."},
    ]


async def test_export_transcript_reports_failure_for_retry(monkeypatch):
    monkeypatch.setattr(apps_script, "GOOGLE_SHEET_ID", "https://script.google.com/x/exec")
    _capture_posts(monkeypatch, RuntimeError("HTTP 502"))

    assert await apps_script.export_transcript(
        session="s", turns=_turns(), timestamp="t") is False


# ── status write-back ───────────────────────────────────────────────────────

async def test_update_lead_status_posts_phone_status_and_notes(monkeypatch):
    monkeypatch.setattr(apps_script, "GOOGLE_SHEET_ID", "https://script.google.com/x/exec")
    posts = _capture_posts(monkeypatch, {"status": "ok", "row": 4})

    ok = await apps_script.update_lead_status(
        phone="+919876543210", status="done", notes="9 turns")

    assert ok is True
    payload = posts[-1][1]
    assert payload["action"] == "update_lead"
    assert payload["phone"] == "+919876543210"
    assert payload["status"] == "done"
    assert payload["notes"] == "9 turns"


async def test_a_lead_missing_from_the_sheet_is_not_an_error(monkeypatch):
    """Console-added leads are not in the sheet's Leads tab at all. The
    sibling treats 'Lead not found' as quiet success, and so do we — the
    transcript export is the payload that matters; a status cell for a row
    that does not exist is not a failure to retry forever."""
    monkeypatch.setattr(apps_script, "GOOGLE_SHEET_ID", "https://script.google.com/x/exec")
    _capture_posts(monkeypatch, {"status": "error", "message": "Lead not found: +91987..."})

    assert await apps_script.update_lead_status(
        phone="+91987", status="done", notes="") is True


async def test_a_missing_leads_tab_is_a_logged_failure_not_quiet_success(monkeypatch):
    """Confirmed by the 2026-08-27 review: Code.gs has TWO 'not found'
    messages — the benign per-lead one, and 'Leads sheet not found.' (a
    renamed/missing tab: a STANDING misconfiguration that kills every status
    write-back). A bare 'not found' substring match returned True before the
    ERROR log, leaving zero signal anywhere for a dead Status column."""
    monkeypatch.setattr(apps_script, "GOOGLE_SHEET_ID", "https://script.google.com/x/exec")
    _capture_posts(monkeypatch, {"status": "error", "message": "Leads sheet not found."})

    assert await apps_script.update_lead_status(
        phone="+91987", status="done", notes="") is False


async def test_a_missing_leads_tab_is_an_error_not_an_empty_sheet(monkeypatch):
    """The live failure of 2026-09-01, and the reason no sheet lead was ever
    dialled: the operator's spreadsheet had no Leads tab, Code.gs answered
    status ok with leads:[], and /admin/sheets-status reported a healthy
    "synced 0" for weeks. The script now reports it the same way its own
    _updateLead always has, and it must reach the scheduler as a failure."""
    _capture_posts(monkeypatch, {"status": "error",
                                 "message": "Leads sheet not found."})

    with pytest.raises(RuntimeError, match="Leads sheet not found"):
        await apps_script.fetch_rows("Leads")


async def test_a_genuinely_empty_tab_is_still_just_empty(monkeypatch):
    """The other half of the same distinction — a Leads tab that exists and
    holds only its header row is not an error, and must not page anyone."""
    _capture_posts(monkeypatch, {"status": "ok", "leads": []})

    assert await apps_script.fetch_rows("Leads") == []
