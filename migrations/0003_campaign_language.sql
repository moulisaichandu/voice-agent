-- 0003_campaign_language.sql — The language a campaign dials in.
--
-- Two columns, one vocabulary. campaigns.language is new; leads.language_pref
-- has existed since 0001 as unconstrained text populated from a hand-typed
-- spreadsheet column, and until now nothing read it. Both are constrained to
-- the same six tokens here so a value that reaches the dial path can never be
-- one app/languages.py doesn't know.
--
-- 'auto' means "send no override to ElevenLabs" — whatever the agent is
-- already configured with. It is the default for every existing campaign,
-- which is what keeps the working one-way and two-way flows byte-for-byte
-- unchanged by this feature.

alter table campaigns
  add column if not exists language text not null default 'auto';

-- Normalize the arbitrary free text already sitting in leads.language_pref
-- BEFORE constraining it. These rows came from hand-typed "Language" columns,
-- so anything is possible; anything unrecognised becomes 'auto', which means
-- "use the agent's configured language" and is the safe unknown answer.
--
-- This CASE mirrors app/languages.py's _ALIASES dict and MUST be kept in step
-- with it. Any free-text variant that app/languages.py's normalize() accepts
-- must appear in the appropriate WHEN branch here, otherwise a pre-existing lead
-- typed in native script loses their preference during migration.
update leads set language_pref = case
    when lower(btrim(coalesce(language_pref, ''))) in ('en', 'eng', 'english')
      then 'en'
    when lower(btrim(coalesce(language_pref, ''))) in ('te', 'tel', 'telugu', 'telegu', 'తెలుగు')
      then 'te'
    when lower(btrim(coalesce(language_pref, ''))) in ('tinglish', 'tenglish', 'telugish', 'te-en')
      then 'tinglish'
    when lower(btrim(coalesce(language_pref, ''))) in ('hi', 'hin', 'hindi', 'हिंदी', 'हिन्दी')
      then 'hi'
    when lower(btrim(coalesce(language_pref, ''))) in ('hinglish', 'hi-en')
      then 'hinglish'
    else 'auto'
  end;

alter table campaigns
  add constraint campaigns_language_check
  check (language in ('auto', 'en', 'te', 'tinglish', 'hi', 'hinglish'));

alter table leads
  alter column language_pref set default 'auto';

alter table leads
  add constraint leads_language_pref_check
  check (language_pref in ('auto', 'en', 'te', 'tinglish', 'hi', 'hinglish'));
