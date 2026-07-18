"""Regression tests for app.main's fail-fast startup checks.

These exist because a silently-misconfigured deploy is worse than one that
refuses to boot: an unset DATABASE_URL means almost nothing works, a missing
webhook secret on a public server means anyone can forge a transcript, and a
wildcard CORS origin alongside a real auth token defeats the token entirely.
Config is read as module-level constants at import time, so each test reloads
app.config (then app.main, which imports those constants by name) after
patching os.environ — patching os.environ alone would not be picked up.
"""

import importlib

import pytest
from fastapi.testclient import TestClient


def _boot_with(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import app.config as config
    import app.main as main
    importlib.reload(config)
    importlib.reload(main)
    return main


def test_refuses_to_boot_without_database_url(monkeypatch):
    # Must SET to empty, not delenv: config.py calls load_dotenv() on every
    # reload, and dotenv only fills in keys ABSENT from os.environ — deleting
    # the key lets the real .env file on disk silently refill it, defeating
    # the test. An empty string stays empty (dotenv never overrides a key
    # that's merely empty-but-present), and config's own DATABASE_URL read
    # treats "" as falsy exactly like a missing var.
    main = _boot_with(monkeypatch, DATABASE_URL="")
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        with TestClient(main.app):
            pass


def test_refuses_public_url_without_webhook_secret(monkeypatch):
    main = _boot_with(
        monkeypatch,
        DATABASE_URL="postgresql://t:t@localhost/t",
        PUBLIC_BASE_URL="https://example.ngrok.dev",
        ELEVENLABS_WEBHOOK_SECRET="",
        ALLOW_INSECURE_PUBLIC="0",
    )
    with pytest.raises(RuntimeError, match="ELEVENLABS_WEBHOOK_SECRET"):
        with TestClient(main.app):
            pass


def test_allow_insecure_public_bypasses_the_webhook_secret_check(monkeypatch):
    main = _boot_with(
        monkeypatch,
        DATABASE_URL="postgresql://t:t@localhost/t",
        PUBLIC_BASE_URL="https://example.ngrok.dev",
        ELEVENLABS_WEBHOOK_SECRET="",
        ALLOW_INSECURE_PUBLIC="1",
    )
    with TestClient(main.app) as c:
        assert c.get("/health").status_code == 200


def test_refuses_wildcard_cors_with_auth_token_set(monkeypatch):
    main = _boot_with(
        monkeypatch,
        DATABASE_URL="postgresql://t:t@localhost/t",
        PUBLIC_BASE_URL="",
        APP_AUTH_TOKEN="realtoken",
        ALLOWED_ORIGINS="*",
        ALLOW_INSECURE_PUBLIC="0",
    )
    with pytest.raises(RuntimeError, match="ALLOWED_ORIGINS"):
        with TestClient(main.app):
            pass


def test_boots_clean_with_a_safe_configuration(monkeypatch):
    main = _boot_with(
        monkeypatch,
        DATABASE_URL="postgresql://t:t@localhost/t",
        PUBLIC_BASE_URL="",
        APP_AUTH_TOKEN="",
        ALLOWED_ORIGINS="http://localhost:3000",
        ALLOW_INSECURE_PUBLIC="0",
    )
    with TestClient(main.app) as c:
        assert c.get("/health").status_code == 200
