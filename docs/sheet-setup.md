# Google Sheet setup — the `Leads` tab

This is the one piece of the lead pipeline that lives outside the codebase.
Until it exists, **no lead in the Sheet is ever dialled**, and the sync reports
success while importing nothing.

## Why this document exists

On 2026-09-01 `/admin/sheets-status` reported:

```json
{"configured": true, "last_synced_count": 0, "last_error": null}
```

every ten minutes. The real cause was in the container log:

```
[sheets] Apps Script update_lead error: {'status':'error','message':'Leads sheet not found.'}
```

The spreadsheet had no `Leads` tab. The Apps Script's `_getLeads` answered a
missing tab with `{status: "ok", leads: []}`, which is byte-identical to an
empty one — so a misconfiguration was indistinguishable from "no new leads".

Both halves are now fixed: the script reports the missing tab as an error
(below), and `apps_script.fetch_rows()` raises on it so it reaches
`/admin/sheets-status` as `last_error` instead of a healthy `0`.

## 1. Add the `Leads` tab

Create a worksheet named exactly **`Leads`** (override with
`LEADS_WORKSHEET_NAME` in `.env` if you want a different name). Row 1 is the
header:

| Name | Phone | Campaign | Language | Consent Basis | Consent At | Status | Notes |
|------|-------|----------|----------|---------------|------------|--------|-------|
| Mouli | 7993399336 | Demo Class Outreach | te | explicit | 2026-09-01 | | |

### Column rules

| Column | Required | Notes |
|---|---|---|
| **Phone** | **yes** | Any reasonable Indian format; normalized to `+91…` E.164. A row without a valid one is skipped. |
| **Campaign** | **yes** | Must name a campaign that already exists. Matched ignoring case and surrounding spaces. A row naming an unknown campaign is skipped. |
| **Consent Basis** | for dialling | `explicit` or `inferred`. Any other value, and the lead imports but is **never called**. |
| **Consent At** | for dialling | A date (`2026-09-01`) or timestamp. A future date is rejected. |
| Name | no | Used for the spoken greeting. Latin names are transliterated to Telugu so the synthesiser pronounces them correctly. |
| Language | no | `te`, `tinglish`, `en`, `hi`, `hinglish`, or blank for `auto`. |
| Status / Notes | no | Written *by* the agent after each call. Create them if you want write-back. |

Header names are matched loosely — `Mobile`, `Contact`, `Phone Number` all
work for the phone column. Only the first row is read as the header.

**Consent is what decides whether a lead is dialled.** Both `Consent Basis`
*and* `Consent At` must be present and valid. A sheet with neither column
imports every row and calls nobody — the console now says so explicitly
("No imported lead can be dialled").

## 2. Patch the Apps Script

The script is the sibling project's `Code.gs`, already deployed against your
Sheet. Open **Extensions → Apps Script** in the spreadsheet and change one
line in `_getLeads`:

```javascript
function _getLeads(data) {
  var name = data.sheet || "Leads";
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(name);

  // BEFORE — a missing tab is indistinguishable from an empty one, so the
  // backend reported "synced 0, no error" for weeks while nothing worked:
  //   if (!sheet) return _json({ status: "ok", leads: [] });

  // AFTER — say so, exactly as _updateLead already does:
  if (!sheet) return _json({ status: "error",
                             message: "Leads sheet not found: " + name });
  ...
```

Then **Deploy → Manage deployments → edit → Deploy** to publish the change.
The `/exec` URL stays the same, so nothing in `.env` needs touching.

## 3. Check it worked

Press **Sync now** on the console's Settings page, or:

```bash
curl -X POST -H "Authorization: Bearer $APP_AUTH_TOKEN" \
     http://localhost:8091/admin/sheets/sync
```

A healthy response names both numbers that matter:

```json
{"synced": 2, "received": 2, "dialable": 2, "skips": {}, "error": null}
```

`synced` is how many rows were imported. **`dialable` is how many the dialer
will actually call** — if it is 0 while `synced` is not, the consent columns
are the reason.

To prove the failure path is no longer silent, rename the tab and sync again:
you should now get `"error": "... Leads sheet not found ..."` rather than a
cheerful `0`.

## A note on the `/exec` URL

`GOOGLE_SHEET_ID` holds the Apps Script `/exec` URL rather than a spreadsheet
ID, by design — the deployed script is already wired to your Sheet, so no
service account or `sa.json` is needed. `SHEETS_TOKEN` is currently unset and
the script has no token property, which means **the endpoint is publicly
callable by anyone who has the URL**. Worth closing before the Sheet holds a
large lead list: set a `SHEETS_TOKEN` script property and the matching `.env`
value, and the transport starts sending it.
