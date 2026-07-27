# Voice Agent — Digital Brolly

## What this is
Outbound AI voice caller. EN / HI / Hinglish / TE / Tinglish, selected per
campaign (`campaigns.language`), which also decides the voice backend.
Two modes: one-way (info + hangup), two-way (voice RAG over course docs).
Transcripts written back to Google Sheets (source of truth: Supabase).

Two-way is live on ElevenLabs. On Sarvam it is BUILT but gated off
(`SARVAM_TWOWAY_ENABLED`) until a human has judged it on a real call.

Sibling project: `../ai-voice-agent/` runs a SEPARATE, working OpenAI-Realtime +
Plivo voice agent for the same business. **Do not modify that repo from here.**
It is READ-ONLY reference material, and a good one: it is where this project's
OpenAI Realtime bridge pattern and Telugu prompts came from, including two
fixes it earned on real 8 kHz calls — pinning the STT language (auto-detection
mis-hears Telugu as Croatian/Urdu) and treating a script as MEANING to convey
rather than words to recite. It has no DB, no scheduler, and none of this
project's compliance machinery, so never route real campaigns through it.

## Stack (do not swap without asking)
FastAPI (3.12) · Plivo (STANDARD Voice API: Plivo dials and streams the call
audio to /calls/stream. NOT SIP trunking — that needs Zentrunk, which isn't
provisioned on our account) · Supabase (Postgres+pgvector, accessed via raw
SQL/asyncpg — NOT supabase-py for CRUD) · Redis · APScheduler · Docker.
Async everywhere.

### THREE voice backends — which one runs is derived from the language
- **ElevenLabs Agents** (hosted LLM, no separate LLM key) — English, Hindi,
  Hinglish, and `auto`. `app/telephony/bridge.py`.
- **Sarvam** (`SARVAM_API_KEY`) — **Telugu and Tinglish**, one-way live and
  two-way built-but-gated. `app/telephony/sarvam_bridge.py`. NOT a
  speech-to-speech API: STT (`saaras:v3`), LLM (`sarvam-105b`) and TTS
  (`bulbul:v3`) are three separate services and **the turn-taking between them
  is ours**. That is why two-way is possible here at all, and why the barge-in
  handling is this backend's most delicate code — see `SpokenLedger`.
- **OpenAI Realtime** (`gpt-realtime-2.1`, uses the existing `OPENAI_API_KEY`) —
  the PREVIOUS Telugu backend, kept working and kept tested as a rollback.
  `app/telephony/openai_bridge.py`. Reached only when
  `TELUGU_BACKEND=openai_realtime`.

Why not ElevenLabs for Telugu: the ElevenLabs Agents platform **does not offer
Telugu at all**. Not a plan tier, not a model setting — its API enumerates the
agent languages it accepts and `te` is not among them (Hindi and Tamil are).
Verified against the live API 2026-07-22; the list is recorded in
`app/languages.py`'s `ELEVENLABS_AGENT_LANGUAGES`. Telugu is this business's
primary market language, so a second backend was unavoidable. **Do not try to
"fix" Telugu by reconfiguring ElevenLabs — it cannot be done, and preflight
will tell you so.**

Why Sarvam rather than OpenAI Realtime: every Realtime voice is English-first —
there is no Telugu-native voice to choose, at any price, and no code change
fixes that. Sarvam's TTS voices are recorded by Indian voice artists and its
STT is trained on Indian telephony audio. `TELUGU_BACKEND` in `.env` switches
Telugu and Tinglish between the two backends together (never separately — see
`app/languages.py`'s `_TELUGU_TOKENS` for why splitting them breaks preflight),
so a bad live call is reverted with one line and a restart.

All three backends drive the same `PlivoCall` (`app/telephony/plivo_stream.py`) and
populate the identical `outcome` contract, so everything downstream — call
recording, Sheets write-back, slot release, retry accounting — is
backend-agnostic and must stay that way. Outbound is mu-law 8 kHz on every
backend (`ulaw_8000` / `audio/pcmu` / Sarvam's `mulaw`+`8000`), so agent audio
is always a passthrough with no transcoding. **One exception, inbound:**
Sarvam's STT does not accept mu-law at all (only WAV/PCM), so a two-way Sarvam
call decodes the lead's audio µ-law→PCM16 in `app/telephony/ulaw.py`. That is
the only transcoding anywhere in the project. It is pure-Python on purpose —
`audioop` was removed in Python 3.13, which this venv already runs.

## Hard rules
- India telecom compliance is mandatory: 140-series caller ID, dial-time DND
  scrub, AI disclosure as the FIRST line of every script (enforced by a test,
  not just review). Never generate code that dials outside these rules.
- CALLING HOURS ARE OPERATOR-CONFIGURED, and are currently set FULLY OPEN
  (`CALLING_HOURS_START=0`, `CALLING_HOURS_END=24` in `.env`), on the owner's
  explicit instruction, 2026-07-22. This is a deliberate decision, not drift —
  **do not "fix" it back to 10:00-19:00.** If you think it is wrong, raise it
  with the owner; do not change it unilaterally.
  - The gate itself still exists and is still tested (`within_calling_hours()`,
    enforced in `app/scheduler.py`'s campaign_tick and again in
    `app/telephony/worker.py`). Only the WINDOW is open. Never delete the gate
    or its tests — restoring the old behaviour must stay a one-line `.env`
    change.
  - What the owner is carrying: India's TCCCPR generally restricts promotional
    calls to roughly 09:00-21:00, and penalties attach to the telecom resource
    — the Plivo number and its DLT registration — not to a line of code. The
    realistic failure is a blacklisted or disconnected number, which stops
    every campaign at once. Surface that if the subject comes up; do not
    re-litigate it unprompted.
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
- `campaigns.language` is admin-set the same way, and the VOICE BACKEND is
  derived from it (`app/languages.py`'s `backend_for`). Never choose a backend
  per call, and never infer one from a model. The backend is fully determined
  before a call is placed, exactly as `mode` is.
- Two-way Telugu/Tinglish is refused three times while its flag is off: at
  campaign creation, in preflight (which also catches campaigns that predate
  the guard), and in `sarvam_bridge.bridge()` itself. Do not remove any of them
  — an unproven two-way call connects, never speaks, never listens, sits in
  silence for the full `CALL_MAX_DURATION_S` of billed airtime, and is then
  recorded as a SUCCESSFUL zero-turn call, because `max_duration` counts as a
  clean exit. Each backend has its OWN flag (`SARVAM_TWOWAY_ENABLED`,
  `OPENAI_TWOWAY_ENABLED`), resolved through `languages.twoway_enabled()`;
  enabling one must never enable the other, because each needs its own live
  call to prove it.
- **Barge-in on Sarvam is the one thing no other backend needs.** ElevenLabs
  and OpenAI own their own conversation history, and OpenAI is told what the
  lead actually heard via `conversation.item.truncate`. On Sarvam the history
  is OURS, so when a lead interrupts, `SpokenLedger` truncates the agent's last
  turn down to the sentences that finished playing, in both the LLM history and
  the stored transcript. Remove it and every later turn is built on words
  nobody heard, and the agent starts referring back to things it never said.
  A partly-played sentence is DROPPED, not kept — under-claiming makes the
  agent repeat itself, over-claiming makes it incoherent.
- On both Telugu backends a script may be written in English and rendered into
  Telugu by a model, so `has_ai_disclosure()`'s creation-time check on
  `campaigns.script` does NOT guarantee what the lead actually hears.
  `app/telephony/call_routes.py` therefore re-checks the FIRST SPOKEN agent
  turn after every call and logs `[compliance]` at ERROR when it fails. Don't
  remove it, and treat any `[compliance]` ERROR as a stop-dialling event.
  - On Sarvam there is also a PREVENTIVE half: `sarvam_llm.render()` holds the
    exact Telugu as a string before anything is spoken, so it checks the
    disclosure itself and prepends a standard one (logging `[compliance]` at
    WARNING) rather than only reporting the failure afterwards. That does not
    make the post-call check redundant — it is still the only thing covering
    what happens after rendering.
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
