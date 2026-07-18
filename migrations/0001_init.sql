-- 0001_init.sql — Initial schema: leads, campaigns, calls, doc_chunks.
--
-- Two additions over the blueprint's own SQL (see plan's "Blueprint issues
-- fixed" section):
--   1. CHECK (mode IN ('oneway','twoway')) on campaigns — the blueprint's SQL
--      left `mode text not null` unconstrained.
--   2. UNIQUE(el_conversation_id) on calls — schema-level defense in depth
--      alongside the Redis idempotency key, so a retried transcript webhook
--      cannot create a duplicate call row even if the Redis key expired or
--      was never set.
--
-- Applied via scripts/apply_migrations.py — a 4-table schema doesn't justify
-- Alembic.

create extension if not exists vector;
create extension if not exists pgcrypto;  -- gen_random_uuid()

-- ── CAMPAIGNS ───────────────────────────────────────────────────────────────
-- mode is admin-set at campaign-creation time, never LLM-inferred per call —
-- see CLAUDE.md's hard rules for why (a past bug in the sibling project let an
-- LLM silently pick a one-way mode the lead couldn't respond to).
create table campaigns (
  campaign_id  uuid primary key default gen_random_uuid(),
  name         text not null,
  mode         text not null check (mode in ('oneway', 'twoway')),
  agent_id     text not null,               -- ElevenLabs agent id
  script       text,                        -- one-way message template
  max_attempts int default 2,
  active       boolean default true,
  created_at   timestamptz default now()
);

-- ── LEADS ───────────────────────────────────────────────────────────────────
-- Mirrors the Google Sheet, plus dialling state. sheet_row is a fast-path hint
-- ONLY — app/sheets/writeback.py verifies the phone number at that row before
-- trusting it, since a row inserted/deleted in the Sheet between sync and call
-- completion would otherwise silently write a transcript to the wrong lead.
create table leads (
  lead_id        uuid primary key default gen_random_uuid(),
  sheet_row      int,
  name           text,
  phone_e164     text not null,
  language_pref  text default 'auto',       -- 'en' | 'te' | 'auto'
  campaign_id    uuid references campaigns(campaign_id),
  consent_basis  text,                      -- 'explicit' | 'inferred'
  consent_at     timestamptz,
  dnd            boolean default false,
  status         text default 'pending',    -- pending|queued|calling|done|failed|dnd
  attempts       int  default 0,
  last_called_at timestamptz,
  created_at     timestamptz default now(),
  unique (phone_e164, campaign_id)
);

create index idx_leads_status on leads (status);
create index idx_leads_campaign on leads (campaign_id);

-- ── CALLS ───────────────────────────────────────────────────────────────────
-- One row per dial attempt. el_conversation_id is how the transcript webhook
-- (keyed by conversation_id) maps back to a lead — see app/webhooks/elevenlabs.py.
create table calls (
  call_id            uuid primary key default gen_random_uuid(),
  lead_id            uuid references leads(lead_id),
  campaign_id        uuid references campaigns(campaign_id),
  el_conversation_id text unique,           -- NULL until ElevenLabs assigns one
  provider_call_id   text,                  -- Exotel call sid
  mode               text,
  status             text,                  -- answered|no_answer|voicemail|failed
  turns              int,
  started_at         timestamptz,
  ended_at           timestamptz,
  transcript         jsonb,                  -- full turn list
  summary            text,
  created_at         timestamptz default now()
);

create index idx_calls_lead on calls (lead_id);
create index idx_calls_conversation on calls (el_conversation_id);

-- ── DOC_CHUNKS (RAG) ────────────────────────────────────────────────────────
create table doc_chunks (
  chunk_id   uuid primary key default gen_random_uuid(),
  doc_name   text,
  section    text,
  content    text,                           -- short, speakable chunk
  embedding  vector(1536),                   -- text-embedding-3-small
  created_at timestamptz default now()
);

create index idx_doc_chunks_embedding on doc_chunks
  using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- ── match_chunks() — server-side relevance filter ──────────────────────────
-- Mirrors the Python-side filter in app/rag/search.py's search_relevant() as
-- defense in depth: below min_score, the row is excluded here too, not just
-- in application code. See CLAUDE.md's hard rules on RAG.
create or replace function match_chunks(
  query_embedding vector(1536),
  match_count int default 4,
  min_score float default 0.30
) returns table (content text, section text, score float)
language sql stable as $$
  select content, section,
         1 - (embedding <=> query_embedding) as score
  from doc_chunks
  where 1 - (embedding <=> query_embedding) >= min_score
  order by embedding <=> query_embedding
  limit match_count;
$$;
