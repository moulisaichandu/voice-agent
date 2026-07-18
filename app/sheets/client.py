"""sheets/client.py — gspread authentication + worksheet access.

gspread is a SYNCHRONOUS HTTP client (no async variant) — every call here runs
in a thread via asyncio.to_thread so it never blocks the event loop that must
stay free to serve webhooks and the RAG tool endpoint.
"""

from __future__ import annotations

import asyncio

import gspread
from google.oauth2.service_account import Credentials

from app.config import GOOGLE_SERVICE_ACCOUNT_FILE, GOOGLE_SHEET_ID, LEADS_WORKSHEET_NAME

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

_client: gspread.Client | None = None


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
