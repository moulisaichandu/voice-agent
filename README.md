# Voice Agent — Digital Brolly (ElevenLabs)

Outbound AI voice caller for Digital Brolly: one cloned rep voice speaking
Telugu / English / Tinglish, calling leads from a Google Sheet. Two call
modes — **one-way** (deliver a message, hang up) and **two-way** (a real
conversation, answering course questions from RAG over the course documents).
Transcripts are written back to the lead's Sheet row.

Built on the ElevenLabs Agents Platform (which owns the hard real-time part:
STT + LLM + cloned TTS + turn-taking) with Exotel for India telephony. This
backend is the orchestration layer: it syncs leads, enforces India calling
rules, paces campaigns, receives transcript webhooks, and writes results back.
**No audio flows through this server** — only control messages and text.

> Sibling project: `../ai-voice-agent/` is a **separate, working** OpenAI
> Realtime + Plivo voice agent for the same business. This repo does not
> touch it. Several hard-won lessons from it are ported here deliberately —
> see [Design decisions](#design-decisions-worth-knowing).

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
                                                            Exotel │ (India, DLT,
                                                                  │  140-series)
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

# 5. Run it
docker compose up -d backend      # http://localhost:8091/health
# ...or directly, without Docker:
.venv/Scripts/python -m uvicorn app.main:app --reload --port 8091
```

The backend listens on host port **8091** (container-internal 8000) to avoid
colliding with the sibling `ai-voice-agent` project on 8000.

## Tests

```bash
pytest -q -m "not integration"   # fast, no containers needed
pytest -q                        # full suite (needs postgres + redis up)
ruff check .
```

Integration tests run against the **real** local Postgres and Redis — the
crash-recovery and no-double-dial guarantees are verified end-to-end
(`tests/integration/test_pipeline.py`), not just mocked per-unit.

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
| `ELEVENLABS_AGENT_PHONE_NUMBER_ID` | The Exotel number linked in ElevenLabs |
| `OPENAI_API_KEY` | **Embeddings only** (`text-embedding-3-small` for RAG) — not the agent's brain, which is ElevenLabs-hosted |
| `PUBLIC_BASE_URL` | Where ElevenLabs reaches `/rag/search` and `/webhooks/elevenlabs` |
| `SCHEDULER_ENABLED` | Must be true on **exactly one** running instance |
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

**`campaigns.mode` is admin-set, never LLM-inferred.** A past bug in the sibling
project let an LLM pick one-way vs two-way per call from free text, silently
producing calls the lead couldn't respond to. Here it's bound to the campaign
row with a DB `CHECK` constraint.

**Preflight runs once per campaign batch, not per lead.** Unlike the sibling
project (where a dead tunnel meant every call rang and dropped), Exotel talks
directly to ElevenLabs here — so a dead `PUBLIC_BASE_URL` doesn't stop calls
connecting, it silently breaks RAG tool calls and transcript webhooks instead.
Preflight catches that before a whole campaign runs half-blind.

## Compliance (India / TCCCPR)

Built into the call path, not bolted on:

- **Calling hours** — `campaign_tick` only enqueues between 10:00–19:00 IST
- **Dial-time DND scrub** — `SISMEMBER` check against the Redis DND set before every dial
- **Consent audit** — every lead records a consent basis + timestamp
- **AI disclosure** — required as the first line of every script

> `dnd_refresh` propagates `leads.dnd` flags from Supabase into Redis. It is
> **not** an integration with India's real NCPR/DND registry — that requires
> business registration outside this codebase's scope. DLT registration, a
> 140-series number, and legal sign-off remain manual prerequisites.
> This is technical guidance, not legal advice.

## Project status

Phases 0–5 are complete and verified: scaffold, database layer, compliance +
RAG, ElevenLabs client + webhooks, the queue/worker/scheduler, and Sheets
sync + write-back.

What remains is **credential-blocked, not code-blocked**:

- [ ] Exotel account + DLT registration + 140-series number *(real lead time)*
- [ ] Link the Exotel number to an ElevenLabs agent; set the agent IDs in `.env`
- [ ] Real Supabase connection string → `DATABASE_URL`, then run migrations
- [ ] Google service-account JSON (`sa.json`) + share the Sheet with it
- [ ] `OPENAI_API_KEY` for RAG embeddings
- [ ] Voice clone: recorded sample + **written consent** from the salesperson
- [ ] `PUBLIC_BASE_URL` (ngrok/cloudflared for dev; a real HTTPS domain in prod)
- [ ] First live one-way test call, then the Telugu/Tinglish voice bake-off
