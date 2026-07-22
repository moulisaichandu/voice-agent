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


@pytest.fixture(autouse=True)
def _restore_module_state():
    """Undo _boot_with's reloads after every test in this file.

    _boot_with reloads app.config and app.main IN PLACE, so the constants the
    last test booted with stay bound in those modules for the rest of the
    session — monkeypatch restores os.environ but cannot un-reload a module.
    Before this fixture, the file only stayed honest because it happened to
    end on a benign config; a wildcard-CORS test added at the end leaked
    ALLOWED_ORIGINS=['*'] into every later test file and broke six unrelated
    tests. Not autouse-dependent on monkeypatch on purpose: this must tear
    down AFTER monkeypatch has restored the environment, which autouse
    ordering gives us."""
    yield
    import app.config as config
    import app.main as main
    importlib.reload(config)
    importlib.reload(main)


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


def test_refuses_public_url_without_an_admin_token(monkeypatch):
    """REGRESSION. Both webhook surfaces were guarded but /admin was not, so a
    public tunnel exposed POST /admin/trigger-tick — which places REAL phone
    calls on the owner's Plivo account — and GET /admin/calls, which returns
    lead PII and full transcripts, to anyone who learned the hostname.
    require_admin_auth fails OPEN when APP_AUTH_TOKEN is unset, and nothing
    caught that combination."""
    main = _boot_with(
        monkeypatch,
        DATABASE_URL="postgresql://t:t@localhost/t",
        PUBLIC_BASE_URL="https://example.ngrok.dev",
        ELEVENLABS_WEBHOOK_SECRET="whsec",
        CALL_WEBHOOK_SECRET="callsec",
        RAG_TOOL_SECRET="ragsec",
        APP_AUTH_TOKEN="",
        ALLOWED_ORIGINS="http://localhost:3000",
        ALLOW_INSECURE_PUBLIC="0",
    )
    with pytest.raises(RuntimeError, match="APP_AUTH_TOKEN"):
        with TestClient(main.app):
            pass


def test_allow_insecure_public_bypasses_the_admin_token_check(monkeypatch):
    """The deliberate escape hatch stays available for local dev."""
    main = _boot_with(
        monkeypatch,
        DATABASE_URL="postgresql://t:t@localhost/t",
        PUBLIC_BASE_URL="https://example.ngrok.dev",
        ELEVENLABS_WEBHOOK_SECRET="whsec",
        CALL_WEBHOOK_SECRET="callsec",
        RAG_TOOL_SECRET="ragsec",
        APP_AUTH_TOKEN="",
        ALLOWED_ORIGINS="http://localhost:3000",
        ALLOW_INSECURE_PUBLIC="1",
    )
    with TestClient(main.app) as c:
        assert c.get("/health").status_code == 200


def test_refuses_wildcard_cors_even_with_no_auth_token(monkeypatch):
    """REGRESSION. The wildcard guard was conditioned on APP_AUTH_TOKEN being
    set, so it did NOT fire in the configuration where a wildcard is most
    dangerous: with no token the admin endpoints need no credentials at all,
    so any website the operator visits could read /admin/calls out of them."""
    main = _boot_with(
        monkeypatch,
        DATABASE_URL="postgresql://t:t@localhost/t",
        PUBLIC_BASE_URL="",
        APP_AUTH_TOKEN="",
        ALLOWED_ORIGINS="*",
        ALLOW_INSECURE_PUBLIC="0",
    )
    with pytest.raises(RuntimeError, match="ALLOWED_ORIGINS"):
        with TestClient(main.app):
            pass


def test_refuses_public_url_without_a_rag_tool_secret(monkeypatch):
    """/rag/search is the live agent's tool. Open on a public URL, every
    request spends OpenAI credit (an embedding, plus a completion on a miss)
    and returns course documents verbatim — a billing drain and a way to
    extract the whole corpus four chunks at a time."""
    main = _boot_with(
        monkeypatch,
        DATABASE_URL="postgresql://t:t@localhost/t",
        PUBLIC_BASE_URL="https://example.ngrok.dev",
        ELEVENLABS_WEBHOOK_SECRET="whsec",
        CALL_WEBHOOK_SECRET="callsec",
        RAG_TOOL_SECRET="",
        APP_AUTH_TOKEN="tok",
        ALLOWED_ORIGINS="http://localhost:3000",
        ALLOW_INSECURE_PUBLIC="0",
    )
    with pytest.raises(RuntimeError, match="RAG_TOOL_SECRET"):
        with TestClient(main.app):
            pass


def test_a_fully_configured_public_deployment_boots(monkeypatch):
    """All four guards satisfied at once — catches a new check that is
    unsatisfiable, or one that fires on a correct configuration."""
    main = _boot_with(
        monkeypatch,
        DATABASE_URL="postgresql://t:t@localhost/t",
        PUBLIC_BASE_URL="https://example.ngrok.dev",
        ELEVENLABS_WEBHOOK_SECRET="whsec",
        CALL_WEBHOOK_SECRET="callsec",
        RAG_TOOL_SECRET="ragsec",
        APP_AUTH_TOKEN="tok",
        ALLOWED_ORIGINS="https://console.example.dev",
        ALLOW_INSECURE_PUBLIC="0",
    )
    with TestClient(main.app) as c:
        assert c.get("/health").status_code == 200
