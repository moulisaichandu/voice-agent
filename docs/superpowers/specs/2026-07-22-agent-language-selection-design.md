# Per-campaign agent language (Telugu / Tinglish / Hindi / Hinglish)

Date: 2026-07-22
Status: approved, ready for an implementation plan

## Problem

The agent speaks whatever language its ElevenLabs agent is configured with.
An operator has no way to say "this campaign dials in Telugu". A `Language`
column is already parsed out of uploaded lead files into `leads.language_pref`
(`app/leads_import.py`, `app/sheets/sync.py`) and then **never read by
anything** — it reaches the database and stops there.

A previous Telugu two-way test failed three ways at once: the agent replied in
English, mis-heard Telugu speech, and produced broken audio when it did attempt
Telugu.

## Root cause of the failed Telugu test

Not a code defect. The ElevenLabs Agents platform defaults to the **Flash v2.5**
TTS model, whose 32 languages are:

```
en ja zh de hi fr ko pt it es id nl tr fil pl sv bg ro ar cs el fi hr ms sk da ta uk ru hu no vi
```

Hindi and Tamil are present. **Telugu is not.** Telugu exists only in Eleven v3,
and for live calls specifically in **Eleven v3 Conversational** (v3 optimised for
real-time dialogue, 70+ languages including Telugu).

The agent was therefore being asked to speak a language its model had no
coverage for, which explains all three symptoms as one fault: English ASR turned
Telugu speech into nonsense, the LLM answered the nonsense in English, and any
Telugu text it did emit was voiced by a model that cannot pronounce it.

No amount of application code fixes this. It is an agent-level model setting.

## Prerequisite (ElevenLabs dashboard, one-time, no code)

For **both** agents (`ELEVENLABS_ONEWAY_AGENT_ID`, `ELEVENLABS_TWOWAY_AGENT_ID`):

1. TTS model → **Eleven v3 Conversational** (covers all four languages plus English).
2. **Additional Languages** → add Telugu and Hindi.
3. **Security** tab → enable the `language` override. Overrides are disabled by
   default, and ElevenLabs *raises an error* if a conversation sends an override
   for a field that is not enabled — so without this step every call on a
   non-`auto` campaign would fail.

Until this is done the feature is inert by design, and preflight says so in
words rather than letting calls fail.

## Design

### 1. `app/languages.py` — new, pure, no I/O

One module that owns the whole catalogue. Same shape as
`app/compliance/disclosure.py`: pure logic, fully unit-testable, no imports from
the DB or the network.

| Choice   | Stored token | ElevenLabs `language` | Style instruction |
|----------|--------------|------------------------|-------------------|
| Auto     | `auto`       | *(no override sent)*   | — |
| English  | `en`         | `en`                   | plain English |
| Telugu   | `te`         | `te`                   | pure Telugu |
| Tinglish | `tinglish`   | `te`                   | Telugu, keeping *course, fees, batch, online, demo* in English |
| Hindi    | `hi`         | `hi`                   | pure Hindi |
| Hinglish | `hinglish`   | `hi`                   | Hindi, keeping the same words in English |

Code-mixed registers map to their **base** ISO language, never to `en`. The
mixing is a matter of style and belongs in the prompt; the ISO code drives
speech recognition and synthesis, and pinning Tinglish to `en` would mis-hear a
lead who answers mostly in Telugu.

Public surface:

- `LANGUAGES` — the catalogue: token → (display name, ISO code, style text).
- `normalize(value: str | None) -> str` — free-text spreadsheet values
  (`Telugu`, `TE`, `telegu`, `Tinglish`, blank) → a canonical token, falling
  back to `auto`. Unknown values fall back to `auto` rather than raising: a
  typo in one spreadsheet cell must not fail a 5,000-row import.
- `resolve(lead_pref: str | None, campaign_language: str) -> str` — the lead's
  own value wins when it is not `auto`; otherwise the campaign's. Returns a
  token, possibly `auto`.
- `iso_code(token) -> str | None` — `None` for `auto`, meaning "send no override".
- `style(token) -> str | None`.

### 2. Schema — `migrations/0003_campaign_language.sql`

- `alter table campaigns add column language text not null default 'auto'`
  with a `CHECK (language in ('auto','en','te','tinglish','hi','hinglish'))`,
  mirroring how `mode` is constrained in `0001_init.sql`.
- Normalize existing `leads.language_pref` values to the canonical tokens, then
  add the same `CHECK`. The column exists today as unconstrained text with a
  stale `'en' | 'te' | 'auto'` comment; the values in it came from arbitrary
  spreadsheet cells.

`app/db/models.py`: add `language: CampaignLanguage = "auto"` to `Campaign`,
with `CampaignLanguage` a `Literal` over the six tokens, alongside `CampaignMode`.

Language is chosen **once, before the call**, and is admin-set — the same
discipline `campaigns.mode` already follows, and it matches ElevenLabs' own
constraint that a conversation's language is fixed for its duration.

### 3. Dial path

**`app/db/campaigns.py`** — `create_campaign()` accepts and persists `language`.

**`app/admin/campaigns.py`** — `CampaignCreate` gains
`language: CampaignLanguage = "auto"`.

**`app/telephony/call_routes.py`** (around line 193, where `dynamic_variables`
is assembled) — resolve once per call:

```python
token = languages.resolve(lead.language_pref, campaign.language)
```

Add `language` (display name) and `language_style` (the instruction sentence) to
`dynamic_variables` so the agent prompt can reference `{{language}}` and
`{{language_style}}`, and pass `languages.iso_code(token)` into `bridge()`.

**`app/telephony/bridge.py`** (the initiation frame, around line 173) — gains a
`language: str | None = None` parameter (an ISO code). When it is not `None`,
the frame carries:

```json
{
  "type": "conversation_initiation_client_data",
  "dynamic_variables": { "...": "..." },
  "conversation_config_override": { "agent": { "language": "te" } }
}
```

When it is `None` the frame is **byte-for-byte what it sends today**. This is
the central safety property of the design: `auto` is the default for every
existing campaign, so nothing that currently works can regress. The wire format
stays inside `bridge.py`; `call_routes.py` deals only in ISO codes.

### 4. Preflight — `app/telephony/preflight.py`

`preflight()` gains a `language: str | None` parameter, passed from
`worker.process_one` (call site at `app/telephony/worker.py:255`, which already
passes `campaign.agent_id`). When a language is requested, it verifies against
the agent's config:

- the `language` override is enabled in the agent's security settings;
- the language is present in the agent's language presets;
- the agent's TTS model can actually speak it — a Telugu campaign pointed at an
  agent still on a v2.5-class model is refused with a message naming that as the
  cause. This check exists because it is precisely the failure that produced the
  broken Telugu test, and it is invisible from the outside.

Consistent with every other preflight check: a string return means "do not dial"
and the reason surfaces in the dashboard readiness strip once, not as N failed
calls. Reading the agent config needs a new read-only helper in
`app/telephony/elevenlabs_client.py` (`agent_language_support()`), wrapped in
`asyncio.to_thread` like `agent_exists` — the SDK is synchronous and this runs
on the loop that carries every live audio bridge.

### 5. Compliance — `app/compliance/disclosure.py`

Add Hindi markers to `_DISCLOSURE_MARKERS`: `एआई`, `कृत्रिम बुद्धिमत्ता`,
`स्वचालित`. A Hindi one-way script that discloses correctly in Devanagari is
currently rejected, because the marker list only covers English and Telugu. The
AI-disclosure hard rule must hold in every language the product offers — the
enforcement point cannot be narrower than the language menu.

### 6. Frontend

- **`frontend/app/campaigns/page.tsx`** — a Language `<select>` at the top of
  the *Create a campaign* card, directly above the leads-file picker (not buried
  in *Advanced*: it is a per-campaign decision the operator makes every time,
  unlike mode). Hint text notes that a `Language` column in the uploaded file
  overrides it per row.
- **`frontend/lib/api.ts`** — `language` on the `Campaign` type and on
  `createCampaign`'s body; a `CampaignLanguage` union mirroring the backend.
- Language badge in the campaign list table and on the campaign detail page,
  next to the existing mode badge.

## Testing

New `tests/unit/test_languages.py`:

- `normalize()` over real spreadsheet spellings, including case, whitespace,
  the `telegu` misspelling, blanks, and unknown values → `auto`.
- `resolve()` precedence: lead wins when set, campaign otherwise, `auto` when
  neither.
- Every catalogue token maps to an ISO code ElevenLabs accepts, and every
  non-`auto` token has a style string.

Additions to existing suites:

- `test_bridge.py` — the initiation frame carries `conversation_config_override`
  when a language is passed, and is unchanged when it is not.
- `test_call_routes.py` — resolution precedence end-to-end; `dynamic_variables`
  carries `language` and `language_style`.
- `test_disclosure.py` — a Devanagari Hindi script with a correct disclosure
  passes; one without still fails.
- `test_admin_campaigns.py` — `language` round-trips; an invalid token is a 422.
- `test_leads_import.py` — a `Language` column normalizes to canonical tokens.
- A preflight test for each refusal (override disabled, language absent from
  presets, wrong TTS model).

All unit tests, no Docker: `pytest -q -m "not integration"`.

## Risks

1. **v3 Conversational latency.** v3 is slower than Flash v2.5's ~75 ms. On a
   phone call a longer pause before the agent speaks is audible and can read as
   a dropped line. Only a live test call answers this. If it proves too slow,
   the fallback is a per-language agent split — Flash v2.5 agents for
   English/Hindi/Hinglish, a v3 Conversational agent for Telugu/Tinglish —
   which this design can accommodate by resolving `agent_id` by
   `(mode, language)` instead of `mode` alone. Nothing else changes.
2. **The cloned rep voice in Telugu.** A voice cloned from English speech often
   degrades on Indian-script languages, and an 8 kHz phone line is unforgiving.
   This is a voice-asset problem no configuration fixes; it may need a voice
   cloned on Telugu speech. Test before dialling a real list.
3. **Dashboard prerequisite not done.** Mitigated by design — preflight blocks
   the campaign and names the missing step.

## Out of scope

- Mid-call language switching. ElevenLabs fixes a conversation's language for
  its duration, and `campaigns.mode`'s precedent is that per-call inference of
  a call's shape is a hard rule violation.
- Automatic language detection from a lead's name, phone series, or region.
  Guessing a person's language from their name is unreliable and would be
  invisible when wrong.
- Translating one-way scripts. A one-way campaign's script is written by the
  operator in the campaign's language; `has_ai_disclosure()` validates it in
  whichever language it is written.
- Tamil. Supported by both models and trivially addable to the catalogue later,
  but not requested.

## References

- [Overrides](https://elevenlabs.io/docs/eleven-agents/customization/personalization/overrides) — the `conversation_config_override` shape; overrides disabled by default and error when not enabled.
- [Agent language](https://elevenlabs.io/docs/eleven-agents/customization/voice/customization/language) — additional languages, language fixed for a call's duration.
- [Models](https://elevenlabs.io/docs/overview/models) — Flash/Turbo v2.5's 32-language list, v3's Telugu support.
