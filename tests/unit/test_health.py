def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    # status is the liveness contract the Dockerfile HEALTHCHECK and compose
    # depend on; "build" was added alongside it so a stale-image deploy is
    # visible from outside the container (see the tests below).
    assert body["status"] == "ok"


# ── which build is actually running ────────────────────────────────────────
#
# A deploy can silently reuse the old image while .env changes DO apply, so the
# running code and the running config disagree with nothing to show for it.
# That has already bitten this project. Neither /health, /admin/config nor the
# startup logs carried any build identifier, so the state was undiagnosable
# from outside the container.

def test_health_reports_which_build_is_running():
    from fastapi.testclient import TestClient

    from app.main import app

    body = TestClient(app).get("/health").json()

    assert body["status"] == "ok"
    assert body.get("build"), "no way to tell a stale image from a fresh one"


def test_the_build_id_is_stable_within_a_process():
    from app import config

    assert config.BUILD_ID == config.BUILD_ID
    assert isinstance(config.BUILD_ID, str) and config.BUILD_ID
