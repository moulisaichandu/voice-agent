"""Unit tests for app/admin/__init__.py's require_admin_auth — shared by
every admin submodule, so it's tested once here rather than once per file.
"""

import pytest

from app import admin


def test_admin_routes_are_open_when_app_auth_token_is_unset(client, monkeypatch):
    """conftest.py sets APP_AUTH_TOKEN='' — matches .env.example's local-dev
    default. Confirms the dependency is a true no-op in that configuration."""
    async def fake_list():
        return []

    monkeypatch.setattr(admin.campaigns.campaigns_db, "list_active_campaigns", fake_list)
    r = client.get("/admin/campaigns")
    assert r.status_code == 200


async def test_admin_routes_require_the_bearer_token_when_configured(monkeypatch):
    monkeypatch.setattr(admin, "APP_AUTH_TOKEN", "secret-token")

    with pytest.raises(Exception):
        # require_admin_auth is a plain function — call it directly rather
        # than spinning up a TestClient, since APP_AUTH_TOKEN is read at
        # request time (a FastAPI Header dependency), not import time.
        await admin.require_admin_auth(authorization=None)


async def test_admin_routes_accept_the_correct_bearer_token_when_configured(monkeypatch):
    monkeypatch.setattr(admin, "APP_AUTH_TOKEN", "secret-token")
    await admin.require_admin_auth(authorization="Bearer secret-token")  # must not raise
