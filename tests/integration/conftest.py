"""A refusal to run destructive integration tests against a real database.

Everything in this directory writes: it creates campaigns and leads, flips them
to dnd/failed/calling, and runs the reaper over whatever else it finds. That is
fine against a scratch database and unrecoverable against the operator's one.

The guard is a name check rather than anything clever, because the failure it
prevents is not subtle — it is DATABASE_URL quietly still pointing at the
database the running backend uses. That was the state of this repo until
2026-08-27, when one `pytest -q -m integration` left 19 synthetic leads stuck in
'calling' among the real ones, and an earlier run deactivated every live
campaign and stopped all dialling until someone noticed.

Opt out with VOICEAGENT_ALLOW_UNSAFE_TEST_DB=1 if you genuinely mean to point
the suite at a differently-named scratch database.
"""

import os

import pytest

_ALLOW = "VOICEAGENT_ALLOW_UNSAFE_TEST_DB"


def _database_name(url: str) -> str:
    """The path component of a postgres URL, minus any ?query suffix."""
    without_query = url.split("?", 1)[0]
    return without_query.rsplit("/", 1)[-1] if "/" in without_query else ""


@pytest.fixture(scope="session", autouse=True)
def refuse_to_write_to_a_non_test_database() -> None:
    if os.environ.get(_ALLOW) == "1":
        return
    url = os.environ.get("DATABASE_URL", "")
    name = _database_name(url)
    if name.endswith("_test"):
        return
    pytest.exit(
        f"Refusing to run integration tests against database {name!r}: these "
        "tests create, mutate and delete campaigns and leads, and the name does "
        "not end in '_test'. See tests/conftest.py for how to create "
        "voiceagent_test, or set "
        f"{_ALLOW}=1 if you are certain this database is disposable.",
        returncode=2,
    )
