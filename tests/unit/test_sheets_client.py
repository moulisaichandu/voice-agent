"""app/sheets/client.py — the credential the container can actually reach.

GOOGLE_SERVICE_ACCOUNT_FILE defaults to the relative path 'sa.json', but that
file is in .dockerignore (correctly, it is a secret) and nothing mounts it, so
it is structurally absent from every container. Sheets write-back therefore
could not work under `docker compose up` no matter what else was fixed, and the
error message told the operator to put a file at a path the container cannot
see. These pin the environment route that does work.
"""

from app.sheets import client

# ── credentials must be reachable from inside a container ──────────────────
#
# GOOGLE_SERVICE_ACCOUNT_FILE defaults to the relative path 'sa.json', but
# sa.json is in .dockerignore (correctly — it is a secret) and nothing mounts
# it, so the file is structurally absent from every container. Sheets could
# therefore never work under `docker compose up`, and the error message's own
# remedy ("put it at that path") could not be followed. Accepting the JSON from
# the environment is the container-native route, and matches CLAUDE.md's rule
# that secrets come from .env.

def test_inline_credentials_from_the_environment_satisfy_the_check(monkeypatch):
    monkeypatch.setattr(client, "GOOGLE_SHEET_ID", "1AbC_sheet_id")
    monkeypatch.setattr(client, "GOOGLE_SERVICE_ACCOUNT_FILE", "definitely-absent.json")
    monkeypatch.setattr(client, "GOOGLE_SERVICE_ACCOUNT_JSON",
                        '{"type": "service_account", "client_email": "x@y.iam"}')

    assert client.unconfigured_reason() is None


def test_a_missing_file_with_no_inline_json_still_explains_both_routes(monkeypatch):
    monkeypatch.setattr(client, "GOOGLE_SHEET_ID", "1AbC_sheet_id")
    monkeypatch.setattr(client, "GOOGLE_SERVICE_ACCOUNT_FILE", "definitely-absent.json")
    monkeypatch.setattr(client, "GOOGLE_SERVICE_ACCOUNT_JSON", "")

    reason = client.unconfigured_reason()

    assert reason is not None
    assert "GOOGLE_SERVICE_ACCOUNT_JSON" in reason, (
        "the operator is told to put a file at a path that a container cannot "
        "see, with no mention of the route that works"
    )


def test_an_apps_script_url_is_a_valid_transport_not_an_error(monkeypatch):
    """Observed live 2026-08-27: GOOGLE_SHEET_ID held a script.google.com
    /macros/.../exec deployment URL — the sibling project's transcript-logger
    web app, already wired to the operator's sheet. That is a working
    transport of its own (app/sheets/apps_script.py), needing no service
    account and no bare sheet ID — so it must not be reported as a
    misconfiguration, or every sync/write-back sweep skips a Sheet that is
    perfectly reachable."""
    monkeypatch.setattr(client, "GOOGLE_SHEET_ID",
                        "https://script.google.com/macros/s/AKfycb_x123/exec")
    # No service-account credential on purpose: the script transport does not
    # use one, so its absence must not be a reason either.
    monkeypatch.setattr(client, "GOOGLE_SERVICE_ACCOUNT_JSON", "")
    monkeypatch.setattr(client, "GOOGLE_SERVICE_ACCOUNT_FILE", "definitely-absent.json")

    assert client.unconfigured_reason() is None


def test_malformed_inline_json_is_reported_as_a_settings_problem(monkeypatch):
    monkeypatch.setattr(client, "GOOGLE_SHEET_ID", "1AbC_sheet_id")
    monkeypatch.setattr(client, "GOOGLE_SERVICE_ACCOUNT_FILE", "definitely-absent.json")
    monkeypatch.setattr(client, "GOOGLE_SERVICE_ACCOUNT_JSON", "{not json")

    reason = client.unconfigured_reason()

    assert reason is not None and "not valid JSON" in reason
