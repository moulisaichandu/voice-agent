-- ── LEADS.DIAL_ORDER ────────────────────────────────────────────────────────
-- Dial an uploaded file's leads in the order they appear IN THE FILE.
--
-- due_leads() used to order by leads.created_at, which only approximates file
-- order and breaks in two ways that matter:
--
--   1. A bulk import writes every row within the same moment, so created_at
--      values differ by microseconds at best and can tie outright — after
--      which the order Postgres returns is arbitrary.
--   2. upsert_lead() deliberately does NOT touch created_at on conflict, so
--      re-uploading a REORDERED file left the original dial order in place.
--      The operator changes the file, re-uploads, and nothing about the
--      calling order changes — with no indication why.
--
-- Nullable because it only means something for leads that came from a file:
-- Google-Sheet and single-lead-form leads have no file position, keep NULL,
-- and sort after the ordered ones (see due_leads' `nulls last`).
alter table leads add column if not exists dial_order int;

-- Serves due_leads' `order by dial_order nulls last, created_at` for the
-- common per-campaign selection. status leads because that query filters on
-- status = 'pending' first.
create index if not exists idx_leads_dial_order
  on leads (campaign_id, status, dial_order, created_at);
