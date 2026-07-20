# Voice Agent — Digital Brolly (ElevenLabs)

Outbound AI voice caller for Digital Brolly: one cloned rep voice speaking
Telugu / English / Tinglish, calling leads from a Google Sheet. Two call
modes — **one-way** (deliver a message, hang up) and **two-way** (a real
conversation, answering course questions from RAG over the course documents).
Transcripts are written back to the lead's Sheet row.

Built on the ElevenLabs Agents Platform (which owns the hard real-time part:
STT + LLM + cloned TTS + turn-taking) with a Plivo number for India telephony,
connected to ElevenLabs via a SIP trunk (the ElevenLabs SDK has no native
Plivo integration — only `exotel`, `twilio`, and generic `sip_trunk`; see
`app/telephony/elevenlabs_client.py`). This backend is the orchestration
layer: it syncs leads, enforces India calling rules, paces campaigns,
receives transcript webhooks, and writes results back. **No audio flows
through this server** — only control messages and text.

> Sibling project: `../ai-voice-agent/` is a **separate, working** OpenAI
> Realtime + Plivo voice agent for the same business (Plivo directly, not via
> ElevenLabs). This repo does not touch it. Several hard-won lessons from it
> are ported here deliberately — see [Design decisions](#design-decisions-worth-knowing).

---

## Architecture

```
Google Sheet ──sync──> Supabase (Postgres + pgvector) <──── RAG (course docs)
                            │                                    ▲
                     APScheduler                                 │ /rag/search
                     (calling hours,                              │ (server tool)
                      campaign pacing)                            │
                            │ enqueue                             │
                            ▼                                     │
                       Redis queue ──> Call worker ──────> ElevenLabs Agents
                                                                  │
                                                       Plivo (SIP │ (India, DLT,
                                                        trunk)    │  140-series)
                                                                  ▼
                                                             Lead's phone
                                                                  │
        Sheet row  <──write-back──  Supabase  <──transcript webhook┘
```

## Quick start (local dev, no cloud accounts needed)

```bash
# 1. Start Redis + a local pgvector Postgres (stands in for Supabase)
docker compose --profile dev up -d postgres redis

# 2. Set up the environment
cp .env.example .env          # defaults already point at the local containers
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt

# 3. Apply the schema
.venv/Scripts/python scripts/apply_migrations.py

# 4. (optional) Seed a demo campaign + leads
.venv/Scripts/python scripts/seed_dev_data.py

# 5. (optional) Ingest the course docs for RAG — otherwise /rag/search has
#    nothing to retrieve
.venv/Scripts/python scripts/ingest_docs.py

# 6. Run it — this also starts the call worker in-process (see below)
docker compose up -d backend      # http://localhost:8091/health
# ...or directly, without Docker:
.venv/Scripts/python -m uvicorn app.main:app --reload --port 8091
```

The backend listens on host port **8091** (container-internal 8000) to avoid
colliding with the sibling `ai-voice-agent` project on 8000.

The call worker (`app/telephony/worker.py`) starts automatically as a
background task alongside the API and scheduler — no separate process or
`docker compose` service is needed at this project's pilot scale. See
`config.WORKER_ENABLED` and the commented-out `worker` service in
`docker-compose.yml` if it's ever split out to scale independently.

## Test frontend (`frontend/`)

A minimal Next.js console for exercising the backend by hand — create a
campaign, add leads, trigger a call, browse transcripts, and test RAG search —
without hand-editing Supabase or wiring up a real Google Sheet first.

```bash
cd frontend
npm install
cp .env.local.example .env.local   # BACKEND_URL, defaults to localhost:8091
npm run dev                        # http://localhost:3200
```

Port **3200**, not Next.js's usual 3000: this machine already runs other
projects' containers on 3000/3001/3006. Change it in `frontend/package.json`
if that's not true for you.

The browser never calls FastAPI directly — it calls this Next server at
`/api/backend/*`, which proxies to `BACKEND_URL` server-side
(`frontend/next.config.js`). That keeps every request same-origin, so the
console doesn't break when the backend's `ALLOWED_ORIGINS` doesn't happen to
list the console's port — a cross-origin block surfaces in the browser as an
opaque "failed to fetch" that's indistinguishable from the backend being down.
`BACKEND_URL` is read at startup, so restart `npm run dev` after changing it.

It talks to a small admin API (`app/admin/endpoint.py`, mounted at `/admin`)
that didn't exist before — `GET/POST /admin/campaigns`,
`GET/POST /admin/campaigns/{id}/leads`, `GET /admin/campaigns/{id}/calls`, and
`POST /admin/trigger-tick`. Every write goes through the exact same validated
paths the rest of the app uses: `create_campaign()`'s AI-disclosure check,
`upsert_lead()`'s normal upsert semantics, and **`trigger-tick` calls the real
`scheduler.campaign_tick()`** — same compliance gates (calling hours, DND,
consent, `max_attempts`) as the automatic scheduler tick, no test-only bypass.
Protected by `APP_AUTH_TOKEN` when set (open by default in local dev, matching
`.env.example`).

**`trigger-tick` places a real phone call** if `ELEVENLABS_API_KEY` and a real
agent/phone number are configured and the worker is running — the frontend
warns about this inline, but know that before clicking it against real
credentials.

The campaign list defaults to **active campaigns only**: a dev database that's
had the integration suite run against it accumulates hundreds of throwaway
`test-*`/`pipeline-*` campaigns (each run deactivates every pre-existing one),
which otherwise bury the real entries. Tick "include inactive" to see them all.

## Tests

```bash
pytest -q -m "not integration"   # fast, no containers needed
pytest -q                        # full suite (needs postgres + redis up)
ruff check .
```

Integration tests run against the **real** local Postgres and Redis — the
crash-recovery and no-double-dial guarantees are verified end-to-end
(`tests/integration/test_pipeline.py`), not just mocked per-unit.

> Gotcha: the integration suite's isolation fixture runs
> `update campaigns set active = false` before each test, so **running
> `pytest` (without `-m "not integration"`) deactivates every campaign in
> your dev database** — including any you seeded by hand. The test console
> will look empty afterwards; re-run `scripts/seed_dev_data.py`, or tick
> "include inactive" to confirm they're still there.

## Configuration

Every env var is read once, in `app/config.py`, via safe parsing helpers (a
malformed override logs a warning and falls back to the default rather than
crashing at import). See `.env.example` for the full list. The ones that
matter most:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Local pgvector for dev; the real Supabase connection string in production. One-line swap, no code change. |
| `REDIS_URL` | Queue, DND set, concurrency semaphore, idempotency keys |
| `ELEVENLABS_API_KEY` | Agents Platform |
| `ELEVENLABS_WEBHOOK_SECRET` | HMAC verification for the transcript webhook |
| `ELEVENLABS_{ONEWAY,TWOWAY}_AGENT_ID` | The two configured agents |
| `ELEVENLABS_AGENT_PHONE_NUMBER_ID` | The Plivo number linked in ElevenLabs via a SIP trunk |
| `OPENAI_API_KEY` | **Embeddings only** (`text-embedding-3-small` for RAG) — not the agent's brain, which is ElevenLabs-hosted |
| `PUBLIC_BASE_URL` | Where ElevenLabs reaches `/rag/search` and `/webhooks/elevenlabs` |
| `SCHEDULER_ENABLED` | Must be true on **exactly one** running instance |
| `WORKER_ENABLED` | The call worker — safe on multiple instances (unlike the scheduler), on by default |
| `RAG_MIN_SCORE` | Relevance floor (0.30) — see below |

## Design decisions worth knowing

**RAG has two search functions, and only one is safe for the live agent.**
`search_relevant()` filters by `RAG_MIN_SCORE` and is the *only* function the
`/rag/search` tool endpoint may call. `search_permissive()` returns top-k
regardless of score and is debug-only. Wiring the permissive one into the live
path makes the agent — which is instructed to answer only from retrieved text —
confidently recite unrelated course material in response to a greeting. A
pinning test (`tests/unit/test_rag_search.py`) fails if anyone ever does this;
it's been verified to actually fail when deliberately sabotaged.

**The call worker is crash-safe by construction.** It uses `BLMOVE` into a
processing list with an explicit ack (not bare `BRPOP`, which loses the lead
entirely if the worker dies mid-flight), and reserves the lead in the DB
(`WHERE status='queued'`) *before* placing the call. A reaper requeues anything
stuck. See `app/telephony/worker.py`'s module docstring for the full list of
issues this design fixes.

**`attempts` counts calls actually placed, not reservations.** Releasing a
reservation because preflight failed or the concurrency cap was hit does not
burn an attempt — otherwise infrastructure hiccups would silently exhaust a
lead's `max_attempts` without a single real call.

**Sheets write-back never trusts a cached row index alone.** `sheet_row` is a
fast-path hint, verified against the phone number at that row before writing; a
mismatch falls back to a full column scan. Without this, a human inserting a row
mid-campaign silently writes a transcript onto a *different* lead's row — and it
looks like a successful write.

**TTS output format (`ulaw_8000`) is an ElevenLabs agent-dashboard setting,
not application code.** No audio flows through this server, so there's
nothing in this repo to configure — but it must be set on each agent
(dashboard or the agent-creation API) when it's created, or the phone path
pays for a transcoding step it doesn't need. Written down here so it isn't
lost.

**`campaigns.mode` is admin-set, never LLM-inferred.** A past bug in the sibling
project let an LLM pick one-way vs two-way per call from free text, silently
producing calls the lead couldn't respond to. Here it's bound to the campaign
row with a DB `CHECK` constraint.

**Preflight runs once per campaign batch, not per lead.** Unlike the sibling
project (where a dead tunnel meant every call rang and dropped), Plivo talks
directly to ElevenLabs here (via the SIP trunk) — so a dead `PUBLIC_BASE_URL`
doesn't stop calls connecting, it silently breaks RAG tool calls and
transcript webhooks instead. Preflight catches that before a whole campaign
runs half-blind.

**AI disclosure is validated at campaign-creation time, not just by review.**
`app/compliance/disclosure.py`'s `has_ai_disclosure()` requires a recognisable
AI-disclosure marker (English "AI"/"artificial intelligence", or Telugu
"కృత్రిమ మేధ") in a script's *first* sentence; `create_campaign()` refuses to
create a one-way campaign without one. A one-way campaign's `script` is passed
to the call as a `script` dynamic variable — the ElevenLabs agent's first
message must reference `{{script}}` for it to actually be spoken (an
agent-dashboard configuration step, not code in this repo).

## Compliance (India / TCCCPR)

Built into the call path, not bolted on:

- **Calling hours** — `campaign_tick` only enqueues between 10:00–19:00 IST
- **Dial-time DND scrub** — `SISMEMBER` check against the Redis DND set before every dial
- **Consent audit** — every lead records a consent basis + timestamp, and
  `due_leads()` excludes any lead without a valid one (`app/compliance/consent.py`)
- **AI disclosure** — required as the first line of every script, enforced at
  campaign-creation time (`app/compliance/disclosure.py`), not just by review

> `dnd_refresh` propagates `leads.dnd` flags from Supabase into Redis. It is
> **not** an integration with India's real NCPR/DND registry — that requires
> business registration outside this codebase's scope. DLT registration, a
> 140-series number, and legal sign-off remain manual prerequisites.
> This is technical guidance, not legal advice.

## Project status

Phases 0–5 are complete and verified: scaffold, database layer, compliance +
RAG, ElevenLabs client + webhooks, the queue/worker/scheduler (the worker now
actually runs — see above), and Sheets sync + write-back.

What remains is **credential-blocked, not code-blocked**:

- [ ] Plivo account + DLT registration + 140-series number *(real lead time)*
- [ ] Link the Plivo number to ElevenLabs via a SIP trunk; set the agent IDs in `.env`
- [ ] Set `ulaw_8000` as the TTS output format on each agent (dashboard/API — see above)
- [ ] Real Supabase connection string → `DATABASE_URL`, then run migrations
- [ ] Google service-account JSON (`sa.json`) + share the Sheet with it
- [ ] `OPENAI_API_KEY` for RAG embeddings, then `python scripts/ingest_docs.py`
- [ ] Voice clone: recorded sample + **written consent** from the salesperson
- [ ] `PUBLIC_BASE_URL` (ngrok/cloudflared for dev; a real HTTPS domain in prod)
- [ ] Decide whether an admin HTTP API for campaigns/leads is in scope, or the
      dev seed script + Sheets-sync path are sufficient for the pilot
- [ ] First live one-way test call, then the Telugu/Tinglish voice bake-off
