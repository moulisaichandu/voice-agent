"""sheets/client.py — gspread authentication + worksheet access.

gspread is a SYNCHRONOUS HTTP client (no async variant) — every call here runs
in a thread via asyncio.to_thread so it never blocks the event loop that must
stay free to serve webhooks and the RAG tool endpoint.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import gspread
from google.oauth2.service_account import Credentials

from app.config import (
    GOOGLE_SERVICE_ACCOUNT_FILE,
    GOOGLE_SERVICE_ACCOUNT_JSON,
    GOOGLE_SHEET_ID,
    LEADS_WORKSHEET_NAME,
)
from app.sheets import apps_script

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

_client: gspread.Client | None = None


def _sheet_id_reason() -> str | None:
    if not GOOGLE_SHEET_ID:
        return ("GOOGLE_SHEET_ID is not set, so there is no Sheet to read or "
                "write. Set it in .env.")
    if GOOGLE_SHEET_ID.lower().startswith(("http://", "https://")):
        # app/config.py's _sheet_id already extracts the ID from a pasted
        # docs.google.com/spreadsheets/d/<id>/... link, so a URL surviving to
        # here has no recognisable /d/<id> segment in it at all.
        #
        # An Apps Script /exec URL never reaches this function: it selects
        # the apps_script transport in unconfigured_reason below, which needs
        # neither a sheet ID nor a service account.
        return (
            "GOOGLE_SHEET_ID contains a URL, but gspread.open_by_key needs the "
            "spreadsheet ID only. Copy the ID between /d/ and /edit in the "
            "Google Sheets URL and set that value in .env."
        )
    return None


def unconfigured_reason() -> str | None:
    """Why the Sheet cannot be reached at all, or None if it can.

    Separates "this deployment has no Sheet set up" from "a Sheets API call
    failed". The scheduler needs that distinction: a missing credential is one
    settings problem, not a transient failure of every queued write-back, and
    reporting it as the latter produced one ERROR per queued lead per sweep,
    indefinitely — noise that buries the real per-row failures this logging is
    for, while re-attempting doomed API calls every ten minutes.

    Deliberately a cheap local check (is the id set, does the key file exist),
    not a probe of the API. It runs on every sweep and must never itself be the
    slow or failing thing.
    """
    if apps_script.is_apps_script(GOOGLE_SHEET_ID):
        # The Apps Script transport (app/sheets/apps_script.py): the deployed
        # web app owns the spreadsheet binding and the write access, so
        # neither a sheet ID nor a service-account credential is needed here.
        return None
    sheet_id_reason = _sheet_id_reason()
    if sheet_id_reason:
        return sheet_id_reason
    if GOOGLE_SERVICE_ACCOUNT_JSON.strip():
        try:
            json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
        except ValueError as exc:
            return (
                f"GOOGLE_SERVICE_ACCOUNT_JSON is set but is not valid JSON "
                f"({exc}). Paste the whole service-account key file as a single "
                "line, quoted, in .env."
            )
        return None
    if not Path(GOOGLE_SERVICE_ACCOUNT_FILE).is_file():
        return (
            f"no Google service-account credential. The file "
            f"{GOOGLE_SERVICE_ACCOUNT_FILE!r} was not found — and note that a "
            "file cannot work inside the container, since sa.json is in "
            ".dockerignore and nothing mounts it. Set "
            "GOOGLE_SERVICE_ACCOUNT_JSON in .env to the key file's contents "
            "instead, and share the Sheet with that service account's email."
        )
    return None


def _get_client() -> gspread.Client:
    global _client
    if _client is None:
        sheet_id_reason = _sheet_id_reason()
        if sheet_id_reason:
            raise RuntimeError(sheet_id_reason)
        if GOOGLE_SERVICE_ACCOUNT_JSON.strip():
            creds = Credentials.from_service_account_info(
                json.loads(GOOGLE_SERVICE_ACCOUNT_JSON), scopes=_SCOPES,
            )
        else:
            creds = Credentials.from_service_account_file(
                GOOGLE_SERVICE_ACCOUNT_FILE, scopes=_SCOPES,
            )
        _client = gspread.authorize(creds)
    return _client


def _get_worksheet_sync() -> gspread.Worksheet:
    return _get_client().open_by_key(GOOGLE_SHEET_ID).worksheet(LEADS_WORKSHEET_NAME)


async def get_worksheet() -> gspread.Worksheet:
    return await asyncio.to_thread(_get_worksheet_sync)
