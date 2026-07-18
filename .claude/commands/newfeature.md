Scaffold a new feature end-to-end, following CLAUDE.md's conventions:

1. Add/extend a pydantic model in `app/db/models.py` if new data is involved.
2. Add the endpoint or module logic in the appropriate `app/` subpackage.
3. Wire it into `app/main.py` (router mount) if it's an HTTP endpoint.
4. Add a test in `tests/unit/` (mock external calls — ElevenLabs, Sheets, Redis)
   or `tests/integration/` if it genuinely needs the local Postgres/Redis
   containers, marked `@pytest.mark.integration`.
5. Run `pytest -q -m "not integration"` and `ruff check .` before reporting done.

Feature to scaffold: $ARGUMENTS
