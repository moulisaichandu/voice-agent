"""
config.py — Centralised configuration: env vars, safe parsing, single source of truth.
All other modules import from here; nothing else reads os.environ directly.
"""

import logging
import math
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── SAFE ENV PARSING ──────────────────────────────────────────────────────────
# os.getenv(name, default) only returns the default when the var is ABSENT; an
# empty or malformed override (e.g. `CALL_MAX_ATTEMPTS=` or `PORT=abc`) is a
# non-None string that would crash int()/float() at import and take the whole
# app down. These helpers fall back to the default with a warning instead.


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        logger.warning(f"{name}={raw!r} is not an integer — using default {default}.")
        return default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except ValueError:
        logger.warning(f"{name}={raw!r} is not a number — using default {default}.")
        return default
    # nan/inf parse cleanly but poison every comparison downstream. A
    # RAG_MIN_SCORE of nan makes `score >= min_score` False for every chunk,
    # so the agent silently answers "no relevant material" to everything.
    if not math.isfinite(value):
        logger.warning(f"{name}={raw!r} is not finite — using default {default}.")
        return default
    return value


_TRUE_VALUES = ("1", "true", "yes", "on")
_FALSE_VALUES = ("0", "false", "no", "off")


def _bool(name: str, default: bool = False) -> bool:
    """Parse a boolean env var. True for 1/true/yes/on, False for 0/false/no/off
    (case-insensitive). Anything else warns and falls back to *default*.

    That fallback matters: this used to be `raw.lower() in (...)`, which
    returned False for every unrecognised value and ignored *default*
    entirely — silently, with no warning, breaking CLAUDE.md's "malformed
    override -> warn + default" rule. WORKER_ENABLED=enabled (or =y, or
    =True_) therefore turned the call worker OFF while campaign_tick kept
    filling calls:queue with leads nothing would ever consume: a total
    outbound-dialling outage whose only symptom was silence."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    cleaned = raw.strip().lower()
    if cleaned in _TRUE_VALUES:
        return True
    if cleaned in _FALSE_VALUES:
        return False
    logger.warning(f"{name}={raw!r} is not a boolean — using default {default}.")
    return default


def _list(name: str, default: tuple[str, ...] = ()) -> list[str]:
    """Comma-separated env var -> list of stripped, non-empty strings."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return list(default)
    return [p.strip() for p in raw.split(",") if p.strip()]


# ── DATABASE ──────────────────────────────────────────────────────────────────
# Supabase Postgres IS Postgres — this is a plain connection string. Point it at
# the local docker-compose pgvector container for dev/test, or the real Supabase
# connection string in production. No code changes either way.
DATABASE_URL: str | None = os.getenv("DATABASE_URL")

# ── REDIS ─────────────────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ── PLIVO (telephony) ─────────────────────────────────────────────────────────
# Plivo's STANDARD Voice API — the same one the sibling ai-voice-agent project
# uses — not SIP trunking. Plivo dials the lead, then streams the call audio to
# this server over a WebSocket, and app/telephony/bridge.py bridges that to the
# ElevenLabs agent. That needs only these credentials.
#
# The alternative (ElevenLabs placing the call itself) requires the number to be
# registered inside ElevenLabs over a SIP trunk, which on Plivo means Zentrunk —
# a product that has to be provisioned per-account and was NOT enabled on ours.
# Hence this path: it works with credentials that already exist.
PLIVO_AUTH_ID: str | None = os.getenv("PLIVO_AUTH_ID")
PLIVO_AUTH_TOKEN: str | None = os.getenv("PLIVO_AUTH_TOKEN")
PLIVO_FROM_NUMBER: str | None = os.getenv("PLIVO_FROM_NUMBER")

# Plivo's webhooks (/calls/answer, /calls/stream) can't carry APP_AUTH_TOKEN, so
# they authenticate with this shared secret in the query string instead. If
# PUBLIC_BASE_URL is set and this is empty, those routes are open to anyone who
# finds the URL — app/main.py's startup check refuses to boot in that state.
CALL_WEBHOOK_SECRET: str | None = os.getenv("CALL_WEBHOOK_SECRET")

# Guards POST /rag/search, the two-way agent's live tool. Same shape as
# CALL_WEBHOOK_SECRET and for the same reason: ElevenLabs calls that endpoint
# from its own servers and cannot send APP_AUTH_TOKEN, so a dedicated shared
# secret is the only thing standing in front of it. Sent as the
# X-RAG-Token HEADER, not a query param — the audit found query strings land
# verbatim in uvicorn's access log, and this one would be logged on every tool
# call of every conversation.
#
# Unguarded on a public URL, anyone who finds the URL can burn OpenAI credit
# (each request is a paid embedding, plus a paid completion on a miss) and
# extract the course corpus verbatim, four chunks at a time. app/main.py's
# startup check refuses to boot in that state.
RAG_TOOL_SECRET: str | None = os.getenv("RAG_TOOL_SECRET")

CALL_RING_TIMEOUT_S = _int("CALL_RING_TIMEOUT_S", 30)
CALL_MAX_DURATION_S = _int("CALL_MAX_DURATION_S", 300)

# How a ONE-WAY call knows it is over. A one-way bridge never forwards the
# lead's audio, so ElevenLabs gets no turn-end signal and never closes its
# socket — nothing else sets the bridge's stop event either. Without these the
# bridge sat on CALL_MAX_DURATION_S (300s) for every one-way call: the lead
# heard a 20-second message and then four and a half minutes of silence, and
# we paid Plivo airtime plus an ElevenLabs conversation for all of it.
#
# TAIL: silence after the agent's queued audio has finished playing before the
# call is ended. Long enough not to clip the last syllable (play_end is an
# estimate from the base64 byte count, not a report from Plivo), short enough
# not to be a noticeable pause.
ONEWAY_SILENCE_TAIL_S = _int("ONEWAY_SILENCE_TAIL_S", 2)
# And the backstop: an agent that never emits any audio at all (misconfigured
# prompt, a dynamic variable the agent is waiting on) should cost seconds, not
# the full CALL_MAX_DURATION_S.
ONEWAY_MAX_SILENT_S = _int("ONEWAY_MAX_SILENT_S", 30)

# On an outbound call the lead's first sound is almost always them answering
# ("Hello?") — an acknowledgement, not an interruption. Without a grace window
# that cancels the agent's opening line mid-sentence, and it restarts the
# greeting from the top. Ported from the sibling project, which hit exactly
# this. Barge-in is fully active after this window; 0 disables it.
PLIVO_GREETING_GRACE_MS = _int("PLIVO_GREETING_GRACE_MS", 2500)

# ── ELEVENLABS ────────────────────────────────────────────────────────────────
ELEVENLABS_API_KEY: str | None = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_WEBHOOK_SECRET: str | None = os.getenv("ELEVENLABS_WEBHOOK_SECRET")
# Set once an agent exists in the ElevenLabs dashboard / is created via API.
ELEVENLABS_ONEWAY_AGENT_ID: str | None = os.getenv("ELEVENLABS_ONEWAY_AGENT_ID")
ELEVENLABS_TWOWAY_AGENT_ID: str | None = os.getenv("ELEVENLABS_TWOWAY_AGENT_ID")
# Only used if ElevenLabs ever places calls itself (its own SIP-trunk number).
# The live path does NOT use it: Plivo places the call and app/telephony/
# bridge.py bridges the audio, so no number is registered inside ElevenLabs.
# Kept so switching back doesn't need a config change.
ELEVENLABS_AGENT_PHONE_NUMBER_ID: str | None = os.getenv("ELEVENLABS_AGENT_PHONE_NUMBER_ID")

# ── OpenAI Realtime (the Telugu voice backend) ──────────────────────────────
# OPENAI_API_KEY is declared below — the RAG embedder uses the same key.
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1")
# Every Realtime voice is English-first; there is no Telugu-native voice. The
# sibling project chose theirs by generating the same Telugu sentence in each
# voice at real phone quality (8 kHz mu-law) and listening. Blank uses the
# API default.
OPENAI_REALTIME_VOICE = os.getenv("OPENAI_REALTIME_VOICE", "")
OPENAI_REALTIME_STT_MODEL = os.getenv("OPENAI_REALTIME_STT_MODEL", "gpt-4o-transcribe")
# THE most load-bearing value here. On 8 kHz telephony audio, automatic
# language detection was observed guessing Croatian and Urdu for Telugu
# speech — the lead is then transcribed as nonsense and the model answers
# nonsense. Pinning the language is what stops that. Blank restores
# auto-detection, which is almost never what you want on a phone call.
OPENAI_REALTIME_STT_LANGUAGE = os.getenv("OPENAI_REALTIME_STT_LANGUAGE", "te")
# Energy-gated VAD, standard for telephony. The threshold is a "voice radius":
# higher means only louder/nearer speech ends a turn, rejecting background.
OPENAI_REALTIME_VAD_THRESHOLD = _float("OPENAI_REALTIME_VAD_THRESHOLD", 0.5)
# How long a lead may pause before the model takes its turn. Too low and it
# interrupts someone mid-sentence; the sibling floors this at 600ms.
OPENAI_REALTIME_SILENCE_MS = _int("OPENAI_REALTIME_SILENCE_MS", 700)
# Suppress background noise before it reaches the model. Blank disables.
OPENAI_REALTIME_NOISE_REDUCTION = os.getenv("OPENAI_REALTIME_NOISE_REDUCTION", "near_field")
# Two-way (conversational) Telugu/Tinglish is fully built — turn detection,
# barge-in truncation, RAG — but the plan that shipped it gates real use on a
# human confirming it on a LIVE call first (Milestone B's own verification
# task), the same discipline the AI-disclosure and mode rules already apply
# to anything a lead could be harmed by getting wrong. False refuses two-way
# for that backend at campaign creation and at dial time; flip to true only
# once a live two-way Telugu call has been placed and judged. One-way is
# never affected by this flag.
OPENAI_TWOWAY_ENABLED = _bool("OPENAI_TWOWAY_ENABLED", False)

# ── EMBEDDINGS (OpenAI text-embedding-3-small) ────────────────────────────────
# A separate credential from ElevenLabs — text-embedding-3-small is an OpenAI
# model. RAG ingestion/search cannot run for real without this; code + tests run
# fine against a mocked embeddings client regardless.
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = _int("EMBED_DIM", 1536)

# ── RAG RETRIEVAL ─────────────────────────────────────────────────────────────
# Below this cosine-similarity score, search_relevant() returns "" / a "no
# relevant material" note instead of the top-k regardless of score — the exact
# bug class the blueprint's own "Mistakes to Avoid" section names from the
# sibling project's history. See app/rag/search.py.
RAG_MIN_SCORE = _float("RAG_MIN_SCORE", 0.30)
RAG_TOP_K = _int("RAG_TOP_K", 4)

# Telugu-script queries embed FAR from this English-only corpus with
# text-embedding-3-small: measured top scores of 0.13-0.19 for questions whose
# English equivalents score 0.37-0.46 — i.e. the same band as a deliberately
# off-topic English query (0.128). Lowering RAG_MIN_SCORE to admit them would
# admit noise too, and their top hits are the WRONG chunks anyway, so it would
# trade "no answer" for "confidently wrong answer" — the exact failure this
# project exists to avoid. Instead, on a miss ONLY, the query is translated to
# English and retried once. English queries (the common case) never pay for it.
RAG_TRANSLATE_ON_MISS = _bool("RAG_TRANSLATE_ON_MISS", True)
RAG_TRANSLATE_MODEL = os.getenv("RAG_TRANSLATE_MODEL", "gpt-4o-mini")

# ── GOOGLE SHEETS ─────────────────────────────────────────────────────────────
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "sa.json")
GOOGLE_SHEET_ID: str | None = os.getenv("GOOGLE_SHEET_ID")
LEADS_WORKSHEET_NAME = os.getenv("LEADS_WORKSHEET_NAME", "Leads")

# ── PUBLIC URL / WEBHOOKS ─────────────────────────────────────────────────────
PUBLIC_BASE_URL: str | None = os.getenv("PUBLIC_BASE_URL")

# ── COMPLIANCE (India TCCCPR) ─────────────────────────────────────────────────
# Calling-hours window, IST. APScheduler's campaign_tick only enqueues leads
# inside this window; anything outside is deferred to the next window.
CALLING_HOURS_START = _int("CALLING_HOURS_START", 10)   # 10:00 IST
CALLING_HOURS_END = _int("CALLING_HOURS_END", 19)        # 19:00 IST
CALLING_HOURS_TZ = os.getenv("CALLING_HOURS_TZ", "Asia/Kolkata")

# ── CAMPAIGN / DIALLING ────────────────────────────────────────────────────────
MAX_CONCURRENT_CALLS = _int("MAX_CONCURRENT_CALLS", 10)
PER_NUMBER_RETRY_COOLDOWN_S = _int("PER_NUMBER_RETRY_COOLDOWN_S", 3600)
CAMPAIGN_TICK_SECONDS = _int("CAMPAIGN_TICK_SECONDS", 60)
RETRY_SWEEP_MINUTES = _int("RETRY_SWEEP_MINUTES", 15)
TRANSCRIPT_RECONCILE_MINUTES = _int("TRANSCRIPT_RECONCILE_MINUTES", 30)
SHEETS_SYNC_MINUTES = _int("SHEETS_SYNC_MINUTES", 10)
DIALING_LOCK_TTL_S = _int("DIALING_LOCK_TTL_S", 60)
PROCESSING_REAPER_TIMEOUT_S = _int("PROCESSING_REAPER_TIMEOUT_S", 300)

# ── SCHEDULER ─────────────────────────────────────────────────────────────────
# APScheduler runs in-process. If this backend is ever scaled to multiple
# replicas, it MUST run in exactly one of them — see the blueprint's own "one
# scheduler, one instance" warning. This flag exists from day one so that rule
# is enforceable later without a rewrite: only the designated instance sets it.
SCHEDULER_ENABLED = _bool("SCHEDULER_ENABLED", True)

# ── CALL WORKER ───────────────────────────────────────────────────────────────
# Unlike the scheduler, the worker is safe to run on MULTIPLE instances at
# once — BLMOVE's pop is atomic, so concurrent consumers of calls:queue can't
# double-process the same lead_id. Runs in-process (a background asyncio task
# started from app.main's lifespan) by default at this project's pilot scale;
# set to false only if it's been split into the separate docker-compose
# `worker` service, to avoid two redundant (harmless, just wasteful) consumers.
WORKER_ENABLED = _bool("WORKER_ENABLED", True)

# ── AUTH / CORS ───────────────────────────────────────────────────────────────
APP_AUTH_TOKEN: str | None = os.getenv("APP_AUTH_TOKEN")
ALLOWED_ORIGINS = _list("ALLOWED_ORIGINS", ("http://localhost:3000",))
ALLOW_INSECURE_PUBLIC = _bool("ALLOW_INSECURE_PUBLIC", False)

# ── LOGGING ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
