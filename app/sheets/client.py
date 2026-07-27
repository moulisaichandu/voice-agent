"""sheets/client.py — gspread authentication + worksheet access.

gspread is a SYNCHRONOUS HTTP client (no async variant) — every call here runs
in a thread via asyncio.to_thread so it never blocks the event loop that must
stay free to serve webhooks and the RAG tool endpoint.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import gspread
from google.oauth2.service_account import Credentials

from app.config import GOOGLE_SERVICE_ACCOUNT_FILE, GOOGLE_SHEET_ID, LEADS_WORKSHEET_NAME

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

_client: gspread.Client | None = None


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
    if not GOOGLE_SHEET_ID:
        return ("GOOGLE_SHEET_ID is not set, so there is no Sheet to read or "
                "write. Set it in .env.")
    if not Path(GOOGLE_SERVICE_ACCOUNT_FILE).is_file():
        return (
            f"the Google service-account key file "
            f"{GOOGLE_SERVICE_ACCOUNT_FILE!r} was not found. Download it from "
            "the Google Cloud console, put it at that path, and share the "
            "Sheet with the service account's email address."
        )
    return None


def _get_client() -> gspread.Client:
    global _client
    if _client is None:
        if not GOOGLE_SHEET_ID:
            raise RuntimeError(
                "GOOGLE_SHEET_ID is not set — cannot reach the leads Sheet."
            )
        creds = Credentials.from_service_account_file(
            GOOGLE_SERVICE_ACCOUNT_FILE, scopes=_SCOPES,
        )
        _client = gspread.authorize(creds)
    return _client


def _get_worksheet_sync() -> gspread.Worksheet:
    return _get_client().open_by_key(GOOGLE_SHEET_ID).worksheet(LEADS_WORKSHEET_NAME)


async def get_worksheet() -> gspread.Worksheet:
    return await asyncio.to_thread(_get_worksheet_sync)
