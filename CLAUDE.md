# Voice Agent — Digital Brolly (ElevenLabs)

## What this is
Outbound AI voice caller. One cloned rep voice, EN/TE/Tinglish.
Two modes: one-way (info + hangup), two-way (voice RAG over course docs).
Transcripts written back to Google Sheets (source of truth: Supabase).

Sibling project: `../ai-voice-agent/` runs a SEPARATE, working OpenAI-Realtime +
Plivo voice agent for the same business. Do not touch that repo from here.

## Stack (do not swap without asking)
FastAPI (3.12) · ElevenLabs Agents (hosted LLM, no separate LLM key) · Plivo
(STANDARD Voice API: Plivo dials and streams the call audio to /calls/stream,
and app/telephony/bridge.py bridges it to the ElevenLabs agent WebSocket. NOT
SIP trunking — that needs Zentrunk, which isn't provisioned on our account.
Both ends are ulaw_8000, so audio is a passthrough with no transcoding)
· Supabase (Postgres+pgvector, accessed via raw SQL/asyncpg — NOT supabase-py
for CRUD) · Redis · APScheduler · Docker. Async everywhere.

## Hard rules
- India telecom compliance is mandatory: 140-series caller ID, dial-time DND
  scrub, AI disclosure as the FIRST line of every script (enforced by a test,
  not just review), calling hours 10:00-19:00 IST only. Never generate code
  that dials outside these rules.
- RAG: `app/rag/search.py`'s `search_relevant()` is the ONLY function the live
  `/rag/search` tool endpoint may call — it filters by `RAG_MIN_SCORE` (0.30)
  and returns "no relevant material" below it. `search_permissive()` exists
  for debug/admin use only and must never be wired to the live tool. Never
  invent course facts, prices, or dates.
- Supabase is source of truth; Redis is ephemeral (rebuildable, safe to flush);
  Sheets is a human-friendly mirror, not authoritative. Never block a call on
  a Sheets write — queue it.
- Sheets write-back matches leads by PHONE NUMBER, with `sheet_row` only as a
  fast-path hint (verify before trusting it) — a cached row index alone can
  silently write a transcript to the wrong lead if a row was inserted/deleted.
- Verify every inbound webhook's HMAC signature (`ElevenLabs-Signature` header).
- The Redis call-worker uses reserve-then-act: transition the lead's DB status
  to `calling` (guarded by `WHERE status='queued'`) BEFORE placing the call,
  and use `BLMOVE`+ack (not bare `BRPOP`) so a crash mid-call doesn't silently
  lose the lead. See app/telephony/worker.py's own docstring for the exact
  pattern before touching it.
- `campaigns.mode` (oneway/twoway) is admin-set at campaign-creation time,
  never LLM-inferred per call. If a natural-language campaign creator is ever
  added, default to the safer option (two-way) and require an unambiguous
  explicit signal for one-way — see the git history / ai-voice-agent sibling
  for why (a past bug there silently produced calls leads couldn't respond to).
- Secrets come from `.env` only; never hardcode keys. `sa.json` (Google service
  account) is equally sensitive — never read it into a commit or log its contents.

## Conventions
- Type hints + pydantic models on every boundary.
- New endpoints/modules get a test in `tests/unit/` (mocked, no Docker) or
  `tests/integration/` (needs local Redis+Postgres, `@pytest.mark.integration`).
- Keep TTS output format `ulaw_8000` for the phone path (skips transcoding).
- Config: every env var is read ONCE in `app/config.py` via the `_int`/`_float`/
  `_bool`/`_list` helpers (malformed override → warn + default, never crash at
  import). No other module reads `os.environ` directly.

## Commands
- Local dev DB+Redis: `docker compose --profile dev up -d`
- Run: `docker compose up` (or `uvicorn app.main:app --reload` with the above running)
- Test (no Docker needed): `pytest -q -m "not integration"`
- Test (full, needs the profile-dev containers up): `pytest -q`
- Lint: `ruff check .`
- Apply DB schema: `python scripts/apply_migrations.py`
