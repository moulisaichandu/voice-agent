# Per-campaign Agent Language Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an operator choose the language a campaign dials in (English, Telugu, Tinglish, Hindi, Hinglish) on the create-campaign form, and make the ElevenLabs agent actually speak it.

**Architecture:** A new pure module `app/languages.py` owns the whole catalogue — token → ISO code → prompt style — plus normalization of free-text spreadsheet values and the lead-over-campaign precedence rule. `campaigns.language` is persisted with a `CHECK` constraint. At dial time `app/telephony/call_routes.py` resolves one token and hands the ISO code to `bridge()`, which adds `conversation_config_override.agent.language` to the initiation frame **only when a language was chosen**. Preflight refuses to dial when the agent can't honour the request.

**Tech Stack:** FastAPI 3.12 · asyncpg raw SQL · pydantic · ElevenLabs Agents (WebSocket initiation frame) · Next.js + Tailwind · pytest

## Global Constraints

- **`auto` must be byte-for-byte identical to today.** Every existing campaign defaults to `auto`, and `auto` sends no `conversation_config_override` key at all. One-way and two-way both work in production right now; nothing in this plan may change what they send.
- **Canonical tokens, exactly these six, everywhere:** `auto`, `en`, `te`, `tinglish`, `hi`, `hinglish`. Same list in the SQL `CHECK`, the `Literal`, and the TypeScript union.
- **Code-mixed registers map to their BASE ISO language**, never to `en`. `tinglish` → `te`, `hinglish` → `hi`.
- Type hints + pydantic models on every boundary (CLAUDE.md).
- Every env var is read once in `app/config.py`. **This feature adds no new env vars.**
- No module reads `os.environ` directly.
- Unit tests must run with no Docker: `pytest -q -m "not integration"`.
- Lint clean: `ruff check .`
- TTS output format stays `ulaw_8000` on the phone path.

## Prerequisite — ElevenLabs dashboard (no code, do this first)

For **both** `ELEVENLABS_ONEWAY_AGENT_ID` and `ELEVENLABS_TWOWAY_AGENT_ID`:

1. TTS model → **Eleven v3 Conversational**. (Flash v2.5's 32 languages do not include Telugu — this is the root cause of the failed Telugu test.)
2. **Additional Languages** → add **Telugu** and **Hindi**.
3. **Security** tab → enable the **`language`** override. Overrides are off by default and ElevenLabs *raises* if one arrives for a field that isn't enabled.

Task 6's preflight check exists to tell you, in words, if any of these is missing.

---

### Task 1: The language catalogue

**Files:**
- Create: `app/languages.py`
- Test: `tests/unit/test_languages.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `TOKENS: tuple[str, ...]`, `AUTO: str`, `normalize(value: str | None) -> str`, `resolve(lead_pref: str | None, campaign_language: str | None) -> str`, `iso_code(token: str) -> str | None`, `style(token: str) -> str | None`, `display(token: str) -> str | None`, `V3_ONLY_ISO: frozenset[str]`. Every later task depends on this module.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_languages.py`:

```python
"""Unit tests for app/languages.py.

The catalogue is pure, but it decides what language a real person is spoken to
in, and every failure mode here is silent: a mis-normalized spreadsheet cell
doesn't raise, it just calls someone in the wrong language.
"""

import pytest

from app import languages


def test_auto_sends_no_override():
    """The whole safety property of this feature. Every pre-existing campaign
    is 'auto', and 'auto' must produce no conversation_config_override at all
    — ElevenLabs raises if an override arrives for a field that isn't enabled
    in the agent's Security tab, so a wrong answer here breaks every call."""
    assert languages.iso_code("auto") is None
    assert languages.style("auto") is None


@pytest.mark.parametrize("raw", ["Telugu", "telugu", "  TELUGU  ", "te", "TE", "telegu", "tel"])
def test_telugu_spellings_normalize(raw):
    """Real spreadsheets contain all of these, including the common
    'telegu' misspelling."""
    assert languages.normalize(raw) == "te"


@pytest.mark.parametrize("raw", ["Hindi", "hi", "HIN", " hindi "])
def test_hindi_spellings_normalize(raw):
    assert languages.normalize(raw) == "hi"


@pytest.mark.parametrize("raw", ["Tinglish", "tenglish", "TINGLISH"])
def test_tinglish_spellings_normalize(raw):
    assert languages.normalize(raw) == "tinglish"


@pytest.mark.parametrize("raw", [None, "", "   ", "Klingon", "n/a", "-"])
def test_unknown_and_blank_values_fall_back_to_auto(raw):
    """A typo in ONE cell of a 5,000-row file must not fail the import or
    silently pick a language for that lead. 'auto' means 'whatever the agent
    is already configured with', which is the safe answer."""
    assert languages.normalize(raw) == "auto"


def test_code_mixed_registers_map_to_their_base_language():
    """Tinglish is Telugu with English words in it, so speech recognition must
    run as Telugu. Mapping it to 'en' would pin ASR to English and mis-hear a
    lead who answers mostly in Telugu — silent, and it lands on the lead."""
    assert languages.iso_code("tinglish") == "te"
    assert languages.iso_code("hinglish") == "hi"


def test_a_lead_language_beats_the_campaign_default():
    assert languages.resolve("hindi", "te") == "hi"


def test_the_campaign_default_applies_when_the_lead_has_none():
    assert languages.resolve("auto", "te") == "te"
    assert languages.resolve(None, "tinglish") == "tinglish"
    assert languages.resolve("", "hi") == "hi"


def test_auto_everywhere_stays_auto():
    assert languages.resolve(None, None) == "auto"
    assert languages.resolve("auto", "auto") == "auto"


def test_an_unknown_lead_value_does_not_override_the_campaign():
    """Junk in a spreadsheet cell must not silently downgrade a Telugu
    campaign to the agent's default language."""
    assert languages.resolve("Klingon", "te") == "te"


def test_every_token_is_complete():
    """Guards against a language being half-added — in the catalogue but with
    no ISO code, so it silently behaves like 'auto'."""
    for token in languages.TOKENS:
        assert languages.display(token), f"{token} has no display name"
        if token == languages.AUTO:
            continue
        assert languages.iso_code(token), f"{token} has no ISO code"
        assert languages.style(token), f"{token} has no style instruction"


def test_telugu_is_flagged_as_needing_a_v3_model():
    """Flash/Turbo v2.5's 32 languages do not include Telugu. This set is what
    preflight uses to refuse a Telugu campaign pointed at a v2.5 agent — the
    exact misconfiguration that produced garbled Telugu audio in testing."""
    assert "te" in languages.V3_ONLY_ISO
    assert "hi" not in languages.V3_ONLY_ISO
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_languages.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.languages'`

- [ ] **Step 3: Write the implementation**

Create `app/languages.py`:

```python
"""languages.py — The catalogue of languages this product dials in.

Pure logic, no I/O — the same shape as app/compliance/disclosure.py, and for
the same reason: what language a real person is spoken to in must be decidable
and testable without a database or a network.

Two things are deliberately kept apart here:

  * the ISO code, which drives ElevenLabs' speech RECOGNITION and SYNTHESIS
    and is the only part of this the platform understands;
  * the style, which is prose for the agent's prompt.

That split is what lets "Tinglish" exist at all. Tinglish and Hinglish are
code-mixed registers, not ISO languages, so they map to their BASE language
(te / hi) and express the mixing through the style text. Mapping them to 'en'
instead would pin speech recognition to English and mis-hear a lead who
answers mostly in Telugu — a silent failure that lands on the lead, not on us.

'auto' means "send no override at all", i.e. whatever the ElevenLabs agent is
already configured with. It is the default for every campaign, so a campaign
created before this feature existed dials exactly as it did before. That is
not a nicety: ElevenLabs RAISES if a conversation sends an override for a
field that is not enabled in the agent's Security tab, so 'auto' sending
nothing is what keeps the working one-way and two-way flows working.
"""

from __future__ import annotations

from dataclasses import dataclass

AUTO = "auto"

# Everyday English words an Indian lead uses untranslated in ordinary speech.
# Naming them explicitly beats "mix in some English": an unanchored instruction
# produced either near-pure Telugu or near-pure English depending on the turn.
_KEEP_IN_ENGLISH = "course, fees, batch, timings, online, demo, certificate, placement"


@dataclass(frozen=True)
class Language:
    """One offered language. *iso* is None only for AUTO, which is the signal
    to send no override; see the module docstring."""
    token: str
    display: str
    iso: str | None
    style: str | None


LANGUAGES: dict[str, Language] = {
    AUTO: Language(
        token=AUTO,
        display="Agent default",
        iso=None,
        style=None,
    ),
    "en": Language(
        token="en",
        display="English",
        iso="en",
        style="Speak in clear, simple English throughout.",
    ),
    "te": Language(
        token="te",
        display="Telugu",
        iso="te",
        style="Speak only in Telugu (తెలుగు) throughout. Do not switch to English.",
    ),
    "tinglish": Language(
        token="tinglish",
        display="Tinglish (Telugu + English)",
        iso="te",
        style=(
            "Speak in Telugu, in the natural mixed way people in Hyderabad "
            f"speak. Keep these everyday English words in English rather than "
            f"translating them: {_KEEP_IN_ENGLISH}."
        ),
    ),
    "hi": Language(
        token="hi",
        display="Hindi",
        iso="hi",
        style="Speak only in Hindi (हिन्दी) throughout. Do not switch to English.",
    ),
    "hinglish": Language(
        token="hinglish",
        display="Hinglish (Hindi + English)",
        iso="hi",
        style=(
            "Speak in Hindi, in the natural mixed way people speak in Indian "
            f"cities. Keep these everyday English words in English rather than "
            f"translating them: {_KEEP_IN_ENGLISH}."
        ),
    ),
}

TOKENS: tuple[str, ...] = tuple(LANGUAGES)

# ISO codes absent from ElevenLabs Flash/Turbo v2.5's 32-language list
# (en ja zh de hi fr ko pt it es id nl tr fil pl sv bg ro ar cs el fi hr ms sk
#  da ta uk ru hu no vi) and available only on Eleven v3 / v3 Conversational.
#
# This is not trivia. An agent left on a v2.5-class model cannot pronounce
# Telugu at all: asked to, it produces garbled audio and drifts back to
# English. That was the cause of this project's failed Telugu two-way test,
# and it is invisible from the outside — which is why preflight checks it.
V3_ONLY_ISO = frozenset({"te"})

# Free-text spellings seen in real uploaded files, mapped to canonical tokens.
# Lower-cased and stripped before lookup.
_ALIASES: dict[str, str] = {
    "auto": AUTO, "default": AUTO, "any": AUTO,
    "en": "en", "eng": "en", "english": "en",
    "te": "te", "tel": "te", "telugu": "te", "telegu": "te", "తెలుగు": "te",
    "tinglish": "tinglish", "tenglish": "tinglish", "te-en": "tinglish",
    "telugish": "tinglish",
    "hi": "hi", "hin": "hi", "hindi": "hi", "हिंदी": "hi", "हिन्दी": "hi",
    "hinglish": "hinglish", "hi-en": "hinglish",
}


def normalize(value: str | None) -> str:
    """A free-text language value → a canonical token, defaulting to AUTO.

    Never raises. A spreadsheet's "Language" column is typed by hand, and one
    unrecognised cell in a 5,000-row import must not fail the import or pick a
    language on the lead's behalf — falling back to AUTO means "use whatever
    the agent is configured with", which is the only safe unknown answer.
    """
    if value is None:
        return AUTO
    key = str(value).strip().lower()
    if not key:
        return AUTO
    if key in LANGUAGES:
        return key
    return _ALIASES.get(key, AUTO)


def resolve(lead_pref: str | None, campaign_language: str | None) -> str:
    """The language THIS call runs in.

    The lead's own value wins when it is set to something real; otherwise the
    campaign's. This mirrors how consent already resolves in
    app/admin/leads.py's upload path — the row's own value beats the
    operator-supplied default — so a mixed-language file works without
    splitting it into separate campaigns.
    """
    lead = normalize(lead_pref)
    if lead != AUTO:
        return lead
    return normalize(campaign_language)


def iso_code(token: str) -> str | None:
    """The ElevenLabs `language` value, or None for AUTO meaning "send no
    override". Callers MUST treat None as "omit the key entirely"."""
    return LANGUAGES[normalize(token)].iso


def style(token: str) -> str | None:
    """The prompt instruction for this language, or None for AUTO."""
    return LANGUAGES[normalize(token)].style


def display(token: str) -> str | None:
    """The human-readable name — shown in the dashboard and passed to the
    agent as the {{language}} dynamic variable."""
    return LANGUAGES[normalize(token)].display
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_languages.py -q`
Expected: PASS, all tests.

- [ ] **Step 5: Lint**

Run: `ruff check app/languages.py tests/unit/test_languages.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add app/languages.py tests/unit/test_languages.py
git commit -m "Add the language catalogue"
```

---

### Task 2: Persist a campaign's language

**Files:**
- Create: `migrations/0003_campaign_language.sql`
- Modify: `app/db/models.py` (add `CampaignLanguage`, add `language` to `Campaign`)
- Modify: `app/db/campaigns.py:24-52` (`create_campaign`)
- Modify: `app/admin/campaigns.py:24-39` (`CampaignCreate`), `:86-98` (`create_campaign`)
- Test: `tests/unit/test_admin_campaigns.py`

**Interfaces:**
- Consumes: `app.languages.TOKENS` (Task 1).
- Produces: `Campaign.language: CampaignLanguage` (default `"auto"`); `campaigns_db.create_campaign(..., language: str = "auto")`; `POST /admin/campaigns` accepts `language`.

- [ ] **Step 1: Write the migration**

Create `migrations/0003_campaign_language.sql`:

```sql
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
update leads set language_pref = case
    when lower(btrim(coalesce(language_pref, ''))) in ('en', 'eng', 'english')
      then 'en'
    when lower(btrim(coalesce(language_pref, ''))) in ('te', 'tel', 'telugu', 'telegu')
      then 'te'
    when lower(btrim(coalesce(language_pref, ''))) in ('tinglish', 'tenglish', 'telugish', 'te-en')
      then 'tinglish'
    when lower(btrim(coalesce(language_pref, ''))) in ('hi', 'hin', 'hindi')
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
```

- [ ] **Step 2: Apply the migration**

Run:
```bash
docker compose --profile dev up -d
python scripts/apply_migrations.py
```
Expected: `apply 0003_campaign_language.sql ...` followed by no error. Re-running prints `skip 0003_campaign_language.sql (already applied)`.

- [ ] **Step 3: Add the model type**

In `app/db/models.py`, add next to `CampaignMode` (line 11):

```python
CampaignMode = Literal["oneway", "twoway"]
# The six tokens in app/languages.py and in 0003's CHECK constraint. Kept as a
# Literal rather than a bare str so an unknown language fails at the API
# boundary, not at the ElevenLabs initiation frame mid-call.
CampaignLanguage = Literal["auto", "en", "te", "tinglish", "hi", "hinglish"]
```

and add to the `Campaign` model, after `script`:

```python
class Campaign(BaseModel):
    campaign_id: UUID
    name: str
    mode: CampaignMode
    agent_id: str
    script: str | None = None
    language: CampaignLanguage = "auto"
    max_attempts: int = 2
    active: bool = True
    created_at: datetime
```

- [ ] **Step 4: Persist it in the DB layer**

In `app/db/campaigns.py`, change `create_campaign`'s signature and INSERT:

```python
async def create_campaign(
    *, name: str, mode: CampaignMode, agent_id: str, script: str | None = None,
    max_attempts: int = 2, language: str = "auto",
) -> Campaign:
```

and replace the INSERT with:

```python
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        insert into campaigns (name, mode, agent_id, script, max_attempts, language)
        values ($1, $2, $3, $4, $5, $6)
        returning *
        """,
        name, mode, agent_id, script, max_attempts, language,
    )
    return _row_to_campaign(row)
```

- [ ] **Step 5: Accept it at the admin boundary**

In `app/admin/campaigns.py`, import the type and add the field to `CampaignCreate` after `script`:

```python
from app.db.models import Campaign, CampaignLanguage
```

```python
    script: str | None = None
    language: CampaignLanguage = Field(
        default="auto",
        description="The language this campaign's calls run in. 'auto' sends "
                    "no override and uses whatever the ElevenLabs agent is "
                    "configured with — the behaviour of every campaign created "
                    "before this field existed. A lead's own language column "
                    "still overrides this per row.",
    )
    max_attempts: int = Field(default=2, ge=1, le=10)
```

and pass it through in the route:

```python
        return await campaigns_db.create_campaign(
            name=body.name, mode=body.mode, agent_id=agent_id,
            script=body.script, max_attempts=body.max_attempts,
            language=body.language,
        )
```

- [ ] **Step 6: Write the tests**

Add to `tests/unit/test_admin_campaigns.py`. First update the shared `_campaign` helper so every existing test still builds a valid `Campaign`:

```python
def _campaign(**overrides):
    defaults = dict(
        campaign_id=uuid4(), name="Demo", mode="twoway", agent_id="agent_1",
        script=None, language="auto", max_attempts=2, active=True,
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    from app.db.models import Campaign
    return Campaign(**defaults)
```

then append:

```python
def test_create_campaign_defaults_to_auto_language(client, monkeypatch):
    """The safety default. Every campaign created before this field existed
    reads back as 'auto', which sends no override — so nothing that works
    today changes."""
    created = {}

    async def fake_create(**kwargs):
        created.update(kwargs)
        return _campaign()

    monkeypatch.setattr(admin_campaigns.campaigns_db, "create_campaign", fake_create)
    monkeypatch.setattr(admin_campaigns.app_config, "ELEVENLABS_TWOWAY_AGENT_ID", "agent_1")

    r = client.post("/admin/campaigns", json={"name": "Demo"})
    assert r.status_code == 201
    assert created["language"] == "auto"


def test_create_campaign_persists_the_chosen_language(client, monkeypatch):
    created = {}

    async def fake_create(**kwargs):
        created.update(kwargs)
        return _campaign(language="tinglish")

    monkeypatch.setattr(admin_campaigns.campaigns_db, "create_campaign", fake_create)
    monkeypatch.setattr(admin_campaigns.app_config, "ELEVENLABS_TWOWAY_AGENT_ID", "agent_1")

    r = client.post("/admin/campaigns", json={"name": "Demo", "language": "tinglish"})
    assert r.status_code == 201
    assert created["language"] == "tinglish"
    assert r.json()["language"] == "tinglish"


def test_an_unknown_language_is_rejected_at_the_boundary(client, monkeypatch):
    """422 here, not a mid-call ElevenLabs error. An unknown language must not
    reach the initiation frame — the failure would be a dropped call."""
    async def fake_create(**kwargs):
        raise AssertionError("must not reach the DB with an invalid language")

    monkeypatch.setattr(admin_campaigns.campaigns_db, "create_campaign", fake_create)
    monkeypatch.setattr(admin_campaigns.app_config, "ELEVENLABS_TWOWAY_AGENT_ID", "agent_1")

    r = client.post("/admin/campaigns", json={"name": "Demo", "language": "klingon"})
    assert r.status_code == 422
```

- [ ] **Step 7: Run the tests**

Run: `pytest tests/unit/test_admin_campaigns.py -q`
Expected: PASS, all tests including the pre-existing ones.

- [ ] **Step 8: Run the whole unit suite for regressions**

Run: `pytest -q -m "not integration"`
Expected: PASS. Any failure here is a `Campaign(...)` constructed somewhere without the new field — fix by relying on the `"auto"` default rather than adding the field at every call site.

- [ ] **Step 9: Commit**

```bash
git add migrations/0003_campaign_language.sql app/db/models.py app/db/campaigns.py app/admin/campaigns.py tests/unit/test_admin_campaigns.py
git commit -m "Persist the language a campaign dials in"
```

---

### Task 3: Normalize lead-level language on import

**Files:**
- Modify: `app/leads_import.py:187-189`
- Modify: `app/sheets/sync.py:114`
- Test: `tests/unit/test_leads_import.py`, `tests/unit/test_sheets_sync.py`

**Interfaces:**
- Consumes: `app.languages.normalize` (Task 1).
- Produces: `ParsedLead.language_pref` is always one of the six canonical tokens.

Without this, the `CHECK` added in Task 2 rejects any future import whose file has a `Language` column reading "Telugu" — an inserted row would raise instead of importing.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_leads_import.py`:

```python
def test_a_language_column_is_normalized_to_a_canonical_token():
    """The DB now CHECKs language_pref against six tokens, so a hand-typed
    'Telugu' must become 'te' before it is inserted — otherwise the import
    raises on a perfectly ordinary spreadsheet."""
    csv = (
        "Name,Phone,Language\n"
        "Asha,9876543210,Telugu\n"
        "Ravi,9876543211,tinglish\n"
        "Sita,9876543212,Hindi\n"
    )
    parsed = leads_import.parse_leads_file("leads.csv", csv.encode())
    assert [lead.language_pref for lead in parsed.leads] == ["te", "tinglish", "hi"]


def test_an_unrecognised_language_value_imports_as_auto():
    """One junk cell must not fail the row — the lead still imports and dials
    in the campaign's language."""
    csv = "Name,Phone,Language\nAsha,9876543210,Klingon\n"
    parsed = leads_import.parse_leads_file("leads.csv", csv.encode())
    assert parsed.leads[0].language_pref == "auto"


def test_a_file_with_no_language_column_imports_as_auto():
    csv = "Name,Phone\nAsha,9876543210\n"
    parsed = leads_import.parse_leads_file("leads.csv", csv.encode())
    assert parsed.leads[0].language_pref == "auto"
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/unit/test_leads_import.py -q -k language`
Expected: FAIL — the first test gets `['Telugu', 'tinglish', 'Hindi']`.

- [ ] **Step 3: Normalize in the importer**

In `app/leads_import.py`, add to the imports:

```python
from app import languages
```

and replace the `language_pref=` argument in the `ParsedLead(...)` construction (around line 187):

```python
            # Normalized here, at the boundary, because leads.language_pref is
            # CHECK-constrained to the six canonical tokens (migrations/0003)
            # and this column is typed by hand into a spreadsheet.
            language_pref=languages.normalize(row.get(language_col) if language_col else None),
```

- [ ] **Step 4: Normalize in the Sheets sync**

In `app/sheets/sync.py`, add `from app import languages` to the imports and replace line 114:

```python
            language_pref=languages.normalize(row.get(language_col) if language_col else None),
```

- [ ] **Step 5: Run the tests**

Run: `pytest tests/unit/test_leads_import.py tests/unit/test_sheets_sync.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/leads_import.py app/sheets/sync.py tests/unit/test_leads_import.py tests/unit/test_sheets_sync.py
git commit -m "Normalize lead language values at the import boundary"
```

---

### Task 4: Send the language override in the initiation frame

**Files:**
- Modify: `app/telephony/bridge.py:114-116` (signature), `:172-177` (initiation frame)
- Test: `tests/unit/test_bridge.py`

**Interfaces:**
- Consumes: nothing (takes a plain ISO string).
- Produces: `bridge.bridge(..., language: str | None = None, ...)` — `language` is an **ISO code** (`"te"`), not a token. `None` means send no override.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_bridge.py`, after the existing one-way tests:

```python
# ── the language override ────────────────────────────────────────────────────

def _initiation(el_ws) -> dict:
    """The first frame the bridge sends — the initiation payload."""
    return json.loads(el_ws.sent[0])


async def test_no_language_sends_no_override(bridged):
    """THE regression guard for this whole feature. ElevenLabs raises if an
    override arrives for a field that isn't enabled in the agent's Security
    tab, so a campaign left on 'auto' must send exactly what it always sent —
    otherwise enabling this feature breaks every existing campaign."""
    el_ws = _FakeElevenLabsWS([_metadata(), _agent_says("Hi."), _audio()])
    bridged(el_ws)

    await asyncio.wait_for(
        bridge.bridge(_FakePlivoWS(), agent_id="agent_1", lead_id="lead-1",
                      one_way=True),
        timeout=5,
    )

    assert "conversation_config_override" not in _initiation(el_ws)


async def test_a_language_is_sent_as_a_conversation_config_override(bridged):
    el_ws = _FakeElevenLabsWS([_metadata(), _agent_says("Hi."), _audio()])
    bridged(el_ws)

    await asyncio.wait_for(
        bridge.bridge(_FakePlivoWS(), agent_id="agent_1", lead_id="lead-1",
                      language="te", one_way=True),
        timeout=5,
    )

    assert _initiation(el_ws)["conversation_config_override"] == {
        "agent": {"language": "te"}
    }


async def test_the_override_does_not_disturb_the_dynamic_variables(bridged):
    """Both travel in the same frame; adding one must not drop the other —
    lead_id in particular is how the transcript maps back to a lead."""
    el_ws = _FakeElevenLabsWS([_metadata(), _agent_says("Hi."), _audio()])
    bridged(el_ws)

    await asyncio.wait_for(
        bridge.bridge(_FakePlivoWS(), agent_id="agent_1", lead_id="lead-1",
                      dynamic_variables={"lead_id": "lead-1"},
                      language="hi", one_way=True),
        timeout=5,
    )

    frame = _initiation(el_ws)
    assert frame["dynamic_variables"] == {"lead_id": "lead-1"}
    assert frame["conversation_config_override"]["agent"]["language"] == "hi"
    assert frame["type"] == "conversation_initiation_client_data"
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/unit/test_bridge.py -q -k "override or language"`
Expected: FAIL — `TypeError: bridge() got an unexpected keyword argument 'language'`.

- [ ] **Step 3: Add the parameter**

In `app/telephony/bridge.py`, change the signature (line 114):

```python
async def bridge(plivo_ws: WebSocket, *, agent_id: str, lead_id: str,
                 dynamic_variables: dict | None = None,
                 language: str | None = None,
                 one_way: bool = False, outcome: dict | None = None) -> dict:
```

and add to the docstring, after the `one_way:` paragraph:

```
    language: an ISO code ('te', 'hi', 'en') for ElevenLabs to run the
    conversation in, or None to send no override at all. None is not the same
    as 'the default language': ElevenLabs RAISES if an override arrives for a
    field that is not enabled in the agent's Security tab, so a campaign that
    never asked for a language must send a frame with no override key in it.
    See app/languages.py.
```

- [ ] **Step 4: Build the frame conditionally**

Replace the `await el_ws.send(...)` at line 173:

```python
            init: dict = {
                "type": "conversation_initiation_client_data",
                "dynamic_variables": dynamic_variables or {},
            }
            if language:
                # Only when a language was actually chosen — see the docstring.
                init["conversation_config_override"] = {"agent": {"language": language}}
            await el_ws.send(json.dumps(init))
```

- [ ] **Step 5: Run the tests**

Run: `pytest tests/unit/test_bridge.py -q`
Expected: PASS, all tests including the pre-existing bridge-loop ones.

- [ ] **Step 6: Commit**

```bash
git add app/telephony/bridge.py tests/unit/test_bridge.py
git commit -m "Let the bridge request a conversation language"
```

---

### Task 5: Resolve the language on the dial path

**Files:**
- Modify: `app/telephony/call_routes.py:193-215`
- Test: `tests/unit/test_call_routes.py`

**Interfaces:**
- Consumes: `app.languages.resolve/iso_code/style/display` (Task 1), `bridge.bridge(language=...)` (Task 4), `Campaign.language` (Task 2).
- Produces: `dynamic_variables` now carries `language` and `language_style` for non-`auto` calls.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_call_routes.py`:

```python
# ── language resolution on the dial path ─────────────────────────────────────

def _lead_and_campaign(lead_language="auto", campaign_language="auto"):
    from datetime import datetime, timezone
    from uuid import uuid4
    from app.db.models import Campaign, Lead

    campaign_id = uuid4()
    lead = Lead(
        lead_id=uuid4(), phone_e164="+919876543210", campaign_id=campaign_id,
        language_pref=lead_language, created_at=datetime.now(timezone.utc),
    )
    campaign = Campaign(
        campaign_id=campaign_id, name="Demo", mode="twoway", agent_id="agent_1",
        language=campaign_language, created_at=datetime.now(timezone.utc),
    )
    return lead, campaign


def test_build_call_language_uses_the_campaign_default():
    lang, dyn = call_routes._call_language(*_lead_and_campaign(campaign_language="te"))
    assert lang == "te"
    assert dyn["language"] == "Telugu"
    assert "Telugu" in dyn["language_style"]


def test_build_call_language_lets_the_lead_override_the_campaign():
    """A mixed-language file must work without splitting it into separate
    campaigns — the row's own value wins, exactly as consent already does."""
    lang, dyn = call_routes._call_language(
        *_lead_and_campaign(lead_language="hi", campaign_language="te")
    )
    assert lang == "hi"
    assert dyn["language"] == "Hindi"


def test_build_call_language_maps_tinglish_to_telugu():
    """Speech recognition and synthesis must run as Telugu; the code-mixing
    is carried by the style text, not by the ISO code."""
    lang, dyn = call_routes._call_language(*_lead_and_campaign(campaign_language="tinglish"))
    assert lang == "te"
    assert "English" in dyn["language_style"]


def test_build_call_language_sends_nothing_for_auto():
    """The safety default: an untouched campaign adds no language variables
    and requests no override."""
    lang, dyn = call_routes._call_language(*_lead_and_campaign())
    assert lang is None
    assert dyn == {}
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/unit/test_call_routes.py -q -k language`
Expected: FAIL — `AttributeError: module 'app.telephony.call_routes' has no attribute '_call_language'`.

- [ ] **Step 3: Add the helper**

In `app/telephony/call_routes.py`, add `from app import languages` to the imports, and add this function above `stream()`:

```python
def _call_language(lead, campaign) -> tuple[str | None, dict]:
    """This call's ElevenLabs language code, and the prompt variables that go
    with it.

    Returns (None, {}) for 'auto' — which is not "the default language" but
    "send nothing at all". Kept as a pure function of two rows so the
    precedence rule is testable without a WebSocket; see app/languages.py.
    """
    token = languages.resolve(lead.language_pref, campaign.language)
    if token == languages.AUTO:
        return None, {}
    return languages.iso_code(token), {
        "language": languages.display(token),
        "language_style": languages.style(token),
    }
```

- [ ] **Step 4: Use it in `stream()`**

Replace lines 193-197 of `app/telephony/call_routes.py`:

```python
    call_language, language_vars = _call_language(lead, campaign)

    dynamic_variables = {"lead_id": str(lead.lead_id)}
    if lead.name:
        dynamic_variables["lead_name"] = lead.name
    if campaign.mode == "oneway" and campaign.script:
        dynamic_variables["script"] = campaign.script
    dynamic_variables.update(language_vars)
```

and add the argument to the `bridge_module.bridge(...)` call:

```python
        await bridge_module.bridge(
            ws,
            agent_id=campaign.agent_id,
            lead_id=str(lead.lead_id),
            dynamic_variables=dynamic_variables,
            language=call_language,
            one_way=(campaign.mode == "oneway"),
            outcome=outcome,
        )
```

- [ ] **Step 5: Run the tests**

Run: `pytest tests/unit/test_call_routes.py -q`
Expected: PASS, all tests.

- [ ] **Step 6: Commit**

```bash
git add app/telephony/call_routes.py tests/unit/test_call_routes.py
git commit -m "Resolve each call's language before bridging"
```

---

### Task 6: Refuse to dial when the agent can't speak the language

**Files:**
- Modify: `app/telephony/elevenlabs_client.py` (add `agent_language_support`)
- Modify: `app/telephony/preflight.py:37` (signature), and add the check before the final `return None`
- Modify: `app/telephony/worker.py:255` (call site)
- Test: `tests/unit/test_startup_checks.py`

**Interfaces:**
- Consumes: `app.languages.V3_ONLY_ISO` (Task 1).
- Produces: `elevenlabs_client.agent_language_support(agent_id) -> dict` with keys `override_allowed: bool`, `languages: set[str]`, `tts_model: str | None`; `preflight(agent_id, language=None)`.

**This task starts with a discovery step, deliberately.** `elevenlabs_client.py`'s own docstring records that this project has already been bitten by assuming SDK shapes instead of checking them (`agent_exists` caught the wrong exception class for months). Do not skip Step 1.

- [ ] **Step 1: Dump a real agent's config and confirm the attribute paths**

Run, with the backend's `.env` loaded and a real agent id:

```bash
python -c "
from app.config import ELEVENLABS_TWOWAY_AGENT_ID
from app.telephony.elevenlabs_client import get_client
a = get_client().conversational_ai.agents.get(ELEVENLABS_TWOWAY_AGENT_ID)
print(a.model_dump_json(indent=2))
"
```

Read the output and note the real paths for:
- the agent's configured language,
- the additional-language presets,
- the TTS model id,
- the overrides allowlist (whether `language` is enabled).

**If any path below differs from what you see, change the code to match what the API returned, not the other way round.** The paths used in Step 3 are the expected ones; the dump is the authority.

- [ ] **Step 2: Write the failing tests**

Add to `tests/unit/test_startup_checks.py`:

```python
# ── preflight: can this agent actually speak the campaign's language? ────────

import pytest

from app.telephony import preflight as preflight_module


@pytest.fixture
def reachable(monkeypatch):
    """Make every non-language preflight check pass, so these tests isolate
    the language gate."""
    monkeypatch.setattr(preflight_module, "ELEVENLABS_API_KEY", "sk-test")
    monkeypatch.setattr(preflight_module, "PLIVO_AUTH_ID", "id")
    monkeypatch.setattr(preflight_module, "PLIVO_AUTH_TOKEN", "tok")
    monkeypatch.setattr(preflight_module, "PLIVO_FROM_NUMBER", "+918035383564")
    monkeypatch.setattr(preflight_module, "PUBLIC_BASE_URL", "https://x.invalid")
    monkeypatch.setattr(preflight_module, "CALL_WEBHOOK_SECRET", "s3cret")

    class _Resp:
        status_code = 200
        text = '<Response><Stream>x</Stream></Response>'

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url):
            return _Resp()

    monkeypatch.setattr(preflight_module.httpx, "AsyncClient", lambda **kw: _Client())
    monkeypatch.setattr(preflight_module, "agent_exists", lambda agent_id: True)


def _support(monkeypatch, **overrides):
    support = {"override_allowed": True, "languages": {"en", "te", "hi"},
               "tts_model": "eleven_v3_conversational"}
    support.update(overrides)
    monkeypatch.setattr(preflight_module, "agent_language_support",
                        lambda agent_id: support)


async def test_preflight_passes_with_no_language_requested(reachable, monkeypatch):
    """'auto' campaigns must not gain a new way to fail — nothing is
    overridden, so nothing needs checking."""
    def boom(agent_id):
        raise AssertionError("must not query language support for an auto campaign")

    monkeypatch.setattr(preflight_module, "agent_language_support", boom)
    assert await preflight_module.preflight("agent_1") is None


async def test_preflight_passes_when_the_agent_supports_the_language(reachable, monkeypatch):
    _support(monkeypatch)
    assert await preflight_module.preflight("agent_1", "te") is None


async def test_preflight_blocks_when_the_override_is_disabled(reachable, monkeypatch):
    """Overrides are off by default and ElevenLabs raises when one arrives
    unannounced — every call in the campaign would fail. Say which box to
    tick, once, instead of failing N calls."""
    _support(monkeypatch, override_allowed=False)
    reason = await preflight_module.preflight("agent_1", "te")
    assert reason is not None
    assert "Security" in reason


async def test_preflight_blocks_when_the_language_is_not_on_the_agent(reachable, monkeypatch):
    _support(monkeypatch, languages={"en"})
    reason = await preflight_module.preflight("agent_1", "te")
    assert reason is not None
    assert "Additional Languages" in reason


async def test_preflight_blocks_telugu_on_a_v2_model(reachable, monkeypatch):
    """REGRESSION for the real failure this feature was built around: Flash
    v2.5's 32 languages don't include Telugu, so the agent produced garbled
    audio and fell back to English. Invisible from the outside."""
    _support(monkeypatch, tts_model="eleven_flash_v2_5")
    reason = await preflight_module.preflight("agent_1", "te")
    assert reason is not None
    assert "v3" in reason


async def test_hindi_is_fine_on_a_v2_model(reachable, monkeypatch):
    """Hindi IS in Flash v2.5's language list — the model gate must be
    specific to the languages that actually need v3, not a blanket rule."""
    _support(monkeypatch, tts_model="eleven_flash_v2_5")
    assert await preflight_module.preflight("agent_1", "hi") is None


async def test_a_language_lookup_failure_does_not_block_the_campaign(reachable, monkeypatch):
    """An unreachable or changed API must not become a dial-stopping outage:
    the language check is an EXTRA guard, and the call can still succeed if
    the agent happens to be configured correctly. Every other preflight check
    still applies."""
    def boom(agent_id):
        raise RuntimeError("ElevenLabs API changed")

    monkeypatch.setattr(preflight_module, "agent_language_support", boom)
    assert await preflight_module.preflight("agent_1", "te") is None
```

- [ ] **Step 3: Run to verify they fail**

Run: `pytest tests/unit/test_startup_checks.py -q -k "preflight and language or v2_model or Security"`
Expected: FAIL — `AttributeError: module 'app.telephony.preflight' has no attribute 'agent_language_support'`.

- [ ] **Step 4: Add the SDK helper**

Append to `app/telephony/elevenlabs_client.py`:

```python
def agent_language_support(agent_id: str) -> dict:
    """What languages this agent can actually be asked to speak.

    Returns {"override_allowed": bool, "languages": set[str],
             "tts_model": str | None}.

    Read with getattr chains rather than direct attribute access on purpose:
    this reaches four levels into an SDK response whose shape is not part of
    any contract we control, and a missing intermediate must degrade to "I
    don't know" rather than raise inside preflight. Verified against the live
    API by dumping a real agent — see this module's docstring on why the
    SDK's type stubs are not trusted here.
    """
    agent = get_client().conversational_ai.agents.get(agent_id)

    conv = getattr(agent, "conversation_config", None)
    agent_cfg = getattr(conv, "agent", None)
    tts_cfg = getattr(conv, "tts", None)

    languages_: set[str] = set()
    default = getattr(agent_cfg, "language", None)
    if default:
        languages_.add(str(default))
    presets = getattr(conv, "language_presets", None) or {}
    try:
        languages_.update(str(k) for k in presets)
    except TypeError:
        pass

    overrides = getattr(getattr(agent, "platform_settings", None), "overrides", None)
    ov_conv = getattr(overrides, "conversation_config_override", None)
    ov_agent = getattr(ov_conv, "agent", None)

    return {
        "override_allowed": bool(getattr(ov_agent, "language", False)),
        "languages": languages_,
        "tts_model": getattr(tts_cfg, "model_id", None),
    }
```

- [ ] **Step 5: Add the preflight gate**

In `app/telephony/preflight.py`, extend the imports:

```python
from app import languages as languages_module
from app.telephony.elevenlabs_client import agent_exists, agent_language_support
```

change the signature (line 37):

```python
async def preflight(agent_id: str, language: str | None = None) -> str | None:
```

and add this block just before the final `return None`:

```python
    # Language, last: only reached when the chain is otherwise sound, and only
    # when a language was actually requested. An 'auto' campaign overrides
    # nothing and so cannot fail here — it must not gain a new way to not dial.
    if language:
        try:
            support = await asyncio.to_thread(agent_language_support, agent_id)
        except Exception as exc:
            # Deliberately NOT a refusal. This check is an extra guard over a
            # correctly-configured agent; an SDK or API change must not become
            # a dial-stopping outage when the call would have worked fine.
            # Every check above still applies.
            logger.warning(
                f"[preflight] could not read agent {agent_id}'s language support "
                f"({type(exc).__name__}: {exc}) — dialling anyway"
            )
            return None

        if not support["override_allowed"]:
            return (
                f"This campaign dials in '{language}', but agent {agent_id} does "
                "not allow the language override. ElevenLabs rejects an override "
                "for a field that isn't enabled, so every call would fail. Open "
                "the agent's Security tab and enable the 'language' override."
            )
        if support["languages"] and language not in support["languages"]:
            return (
                f"Agent {agent_id} is not configured for '{language}' "
                f"(it has: {', '.join(sorted(support['languages'])) or 'none'}). "
                "Add it under Additional Languages in the agent's settings."
            )
        model = (support["tts_model"] or "").lower()
        if language in languages_module.V3_ONLY_ISO and model and "v3" not in model:
            return (
                f"Agent {agent_id} runs the '{support['tts_model']}' TTS model, "
                f"which cannot speak '{language}' — Flash/Turbo v2.5's 32 "
                "languages do not include Telugu. The call would produce garbled "
                "audio and fall back to English. Switch the agent's model to "
                "Eleven v3 Conversational."
            )

    return None
```

Add a module logger at the top of `preflight.py` if one is not already there:

```python
import logging

logger = logging.getLogger(__name__)
```

- [ ] **Step 6: Pass the language from the worker**

In `app/telephony/worker.py`, add `from app import languages` to the imports and change line 255:

```python
        reachability_error = await preflight_module.preflight(
            campaign.agent_id,
            languages.iso_code(languages.resolve(lead.language_pref, campaign.language)),
        )
```

- [ ] **Step 7: Run the tests**

Run: `pytest tests/unit/test_startup_checks.py tests/unit/test_worker_process_one.py -q`
Expected: PASS.

- [ ] **Step 8: Run the whole unit suite**

Run: `pytest -q -m "not integration"`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add app/telephony/elevenlabs_client.py app/telephony/preflight.py app/telephony/worker.py tests/unit/test_startup_checks.py
git commit -m "Refuse to dial when the agent cannot speak the campaign's language"
```

---

### Task 7: Accept a Hindi AI disclosure

**Files:**
- Modify: `app/compliance/disclosure.py:61-70`
- Test: `tests/unit/test_disclosure.py`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing new — `has_ai_disclosure()` keeps its signature.

A Hindi one-way script that discloses correctly in Devanagari is rejected today, because `_DISCLOSURE_MARKERS` covers only English and Telugu. The AI-disclosure hard rule must hold in every language the product offers, and the enforcement point cannot be narrower than the language menu.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_disclosure.py`:

```python
def test_a_hindi_disclosure_in_devanagari_is_accepted():
    """Offering Hindi means accepting a Hindi disclosure. Without a Devanagari
    marker, a correctly-disclosing Hindi script is refused at campaign
    creation and the operator has no way to comply."""
    assert has_ai_disclosure(
        "नमस्ते! यह एक स्वचालित एआई वॉइस असिस्टेंट है, डिजिटल ब्रॉली की ओर से।"
    )


def test_a_hindi_script_without_a_disclosure_still_fails():
    """The marker list must not become a rubber stamp: ordinary Hindi with no
    disclosure has to keep failing."""
    assert not has_ai_disclosure(
        "नमस्ते! हम डिजिटल ब्रॉली से बात कर रहे हैं। हमारा नया कोर्स शुरू हो रहा है।"
    )


def test_hindi_artificial_intelligence_spelled_out_is_accepted():
    assert has_ai_disclosure("यह कॉल कृत्रिम बुद्धिमत्ता द्वारा की जा रही है।")
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/unit/test_disclosure.py -q -k hindi`
Expected: FAIL on the two positive tests; the negative one already passes.

- [ ] **Step 3: Add the markers**

In `app/compliance/disclosure.py`, extend `_DISCLOSURE_MARKERS`:

```python
_DISCLOSURE_MARKERS = (
    "artificial intelligence",
    "automated voice",
    "automated call",
    "voice assistant",
    "voice bot",
    "virtual assistant",
    "కృత్రిమ మేధ",  # Telugu: "artificial intelligence"
    "ఆటోమేటెడ్",  # Telugu: "automated"
    "एआई",  # Hindi: "AI" spelled out in Devanagari
    "कृत्रिम बुद्धिमत्ता",  # Hindi: "artificial intelligence"
    "स्वचालित",  # Hindi: "automated"
)
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/unit/test_disclosure.py -q`
Expected: PASS, all tests including the pre-existing brand-name and greeting cases.

- [ ] **Step 5: Commit**

```bash
git add app/compliance/disclosure.py tests/unit/test_disclosure.py
git commit -m "Accept a Hindi AI disclosure"
```

---

### Task 8: Choose the language in the dashboard

**Files:**
- Modify: `frontend/lib/api.ts:14-25` (types), `:174-188` (`createCampaign`)
- Modify: `frontend/app/campaigns/page.tsx`
- Modify: `frontend/app/campaigns/[id]/page.tsx:164`

**Interfaces:**
- Consumes: `POST /admin/campaigns` accepting `language` (Task 2).
- Produces: nothing other tasks depend on.

- [ ] **Step 1: Add the types**

In `frontend/lib/api.ts`, below `CampaignMode`:

```typescript
export type CampaignMode = "oneway" | "twoway";

/** Mirrors app/languages.py's TOKENS and migrations/0003's CHECK constraint.
 * "auto" sends no override to ElevenLabs and uses whatever the agent is
 * configured with — the behaviour of every campaign created before this
 * existed. */
export type CampaignLanguage = "auto" | "en" | "te" | "tinglish" | "hi" | "hinglish";

export const CAMPAIGN_LANGUAGES: { value: CampaignLanguage; label: string }[] = [
  { value: "auto", label: "Agent default" },
  { value: "en", label: "English" },
  { value: "te", label: "Telugu" },
  { value: "tinglish", label: "Tinglish (Telugu + English)" },
  { value: "hi", label: "Hindi" },
  { value: "hinglish", label: "Hinglish (Hindi + English)" },
];

export function languageLabel(value: string): string {
  return CAMPAIGN_LANGUAGES.find((l) => l.value === value)?.label ?? value;
}
```

add `language` to the `Campaign` type:

```typescript
export type Campaign = {
  campaign_id: string;
  name: string;
  mode: CampaignMode;
  agent_id: string;
  script: string | null;
  language: CampaignLanguage;
  max_attempts: number;
  active: boolean;
  created_at: string;
};
```

and to `createCampaign`'s body:

```typescript
  createCampaign: (body: {
    name: string;
    /** Defaults to twoway server-side — the safer of the two. Still admin-set
     * at creation time, never inferred per call. */
    mode?: CampaignMode;
    /** Omit to use the agent configured for the mode in the backend's .env.
     * Kept optional rather than removed so an explicit id still works. */
    agent_id?: string;
    script?: string;
    /** Omit for "auto" — no override, the agent's own language. */
    language?: CampaignLanguage;
    max_attempts?: number;
  }) =>
```

- [ ] **Step 2: Add the selector to the create form**

In `frontend/app/campaigns/page.tsx`, extend the imports:

```typescript
import {
  api,
  CAMPAIGN_LANGUAGES,
  errorMessage,
  languageLabel,
  type Campaign,
  type CampaignLanguage,
  type CampaignMode,
  type ConsentBasis,
  type LeadImportResult,
} from "@/lib/api";
```

add state next to `mode`:

```typescript
  const [language, setLanguage] = useState<CampaignLanguage>("auto");
```

add the field **above** `<LeadsUpload />` inside the form — it is a decision the operator makes for every campaign, so it belongs in the main flow rather than under *Advanced*:

```tsx
          <Field
            label="Language"
            hint="Every lead in this file is called in this language. A Language column in the file overrides it for that row. “Agent default” changes nothing about how your agent already speaks."
          >
            <select
              className={fieldControlClass}
              value={language}
              onChange={(e) => setLanguage(e.target.value as CampaignLanguage)}
            >
              {CAMPAIGN_LANGUAGES.map((l) => (
                <option key={l.value} value={l.value}>
                  {l.label}
                </option>
              ))}
            </select>
          </Field>

          <LeadsUpload
```

and pass it when creating:

```typescript
      const campaign = await api.createCampaign({
        name: uniqueCampaignName(leadsFile.name, campaigns),
        mode,
        language,
        script: script.trim() || undefined,
      });
```

- [ ] **Step 3: Show it in the campaigns table**

In the same file, add a header cell after `Mode`:

```tsx
              <Th>Mode</Th>
              <Th>Language</Th>
```

a body cell after the mode badge:

```tsx
                  <Td>
                    <Badge tone={c.language === "auto" ? "neutral" : "accent"}>
                      {languageLabel(c.language)}
                    </Badge>
                  </Td>
```

and bump both `colSpan={5}` occurrences in the `TableMessageRow`s to `colSpan={6}`.

- [ ] **Step 4: Show it on the campaign detail page**

In `frontend/app/campaigns/[id]/page.tsx`, add `languageLabel` to the `@/lib/api` import and add a badge after the mode badge on line 164:

```tsx
          <Badge tone={campaign.mode === "twoway" ? "accent" : "neutral"}>{campaign.mode}</Badge>
          <Badge tone={campaign.language === "auto" ? "neutral" : "accent"}>
            {languageLabel(campaign.language)}
          </Badge>
```

- [ ] **Step 5: Type-check and build**

Run:
```bash
cd frontend && npm run lint && npx tsc --noEmit
```
Expected: no errors. (`Campaign.language` being required will surface any object literal that needs the field.)

- [ ] **Step 6: Commit**

```bash
git add frontend/lib/api.ts frontend/app/campaigns/page.tsx "frontend/app/campaigns/[id]/page.tsx"
git commit -m "Choose a campaign's language in the dashboard"
```

---

### Task 9: Verify against a real call

**Files:** none — this is the live verification the whole plan exists for.

**Interfaces:**
- Consumes: everything above, plus the dashboard prerequisite at the top of this plan.

- [ ] **Step 1: Confirm the full suite is green**

Run: `pytest -q -m "not integration"` then `ruff check .`
Expected: PASS, `All checks passed!`

- [ ] **Step 2: Confirm the dashboard prerequisite is done**

Both agents: Eleven v3 Conversational model, Telugu + Hindi under Additional Languages, `language` override enabled in Security. If any is missing, Task 6's preflight will refuse the campaign and name it — that is the check working, not a bug.

- [ ] **Step 3: Prove the safety property first — an `auto` campaign still works**

Create a campaign with Language left on **Agent default**, upload a one-lead file with your own number, and dial. It must behave exactly as it does today. **If this regresses, stop** — the `auto` path is what protects every existing campaign, and nothing else matters until it is right.

- [ ] **Step 4: Hindi two-way, to your own number**

Create a two-way campaign with Language **Hindi**. Hindi is in Flash v2.5 *and* v3, so it is the cleanest test of the override path itself. Confirm on the call:
- the agent opens in Hindi,
- it understands a Hindi reply (check the transcript in the dashboard, not just your ears),
- it stays in Hindi rather than drifting to English.

- [ ] **Step 5: Telugu two-way, to your own number**

The one that failed before. Confirm the same three things, and additionally judge:
- **Audio quality** — is the cloned voice intelligible in Telugu, or does it slur and mispronounce? This is a voice-asset problem no configuration fixes; if it is bad, the voice needs re-cloning on Telugu speech.
- **Latency** — how long between you finishing a sentence and the agent replying? v3 Conversational is slower than Flash v2.5. If the pause reads as a dropped call, take the fallback in the spec's Risk 1: resolve `agent_id` by `(mode, language)` and keep Flash v2.5 agents for English/Hindi.

- [ ] **Step 6: Tinglish two-way**

Confirm the code-mixing actually happens — the agent should say "course", "fees", "batch" in English inside Telugu sentences. If it speaks pure Telugu instead, the agent's prompt is not using the `{{language_style}}` dynamic variable; add it to the prompt in the ElevenLabs dashboard.

- [ ] **Step 7: One-way in Telugu**

Create a one-way campaign with a Telugu script whose first sentence discloses AI (it must contain `AI` or `కృత్రిమ మేధ` — enforced at creation). Confirm the message plays and the call hangs up on its own.

- [ ] **Step 8: A per-row override**

Upload a file with a `Language` column mixing `Telugu` and `Hindi` into one campaign whose default is Telugu, and confirm from the transcripts that the Hindi rows were called in Hindi.

- [ ] **Step 9: Record what you found**

Append a short "Live verification" section to `docs/superpowers/specs/2026-07-22-agent-language-selection-design.md` with the measured latency, the verdict on the cloned voice in Telugu, and anything that needed a prompt change. Commit it.

---

## Notes for the implementer

- **The agent prompt is not in this repo.** `{{language}}` and `{{language_style}}` are delivered as dynamic variables, but they only do something if the ElevenLabs agent's system prompt references them. Add a line like `{{language_style}}` to both agents' prompts in the dashboard. Task 9 Step 6 is where you find out if you forgot.
- **Don't add a language column to the single-lead form** (`app/admin/leads.py`'s `LeadCreate` already has `language_pref`, and it now normalizes through the same `CHECK`). It works; it just isn't surfaced in the UI, and no one asked for it.
- **`place_outbound_call` in `elevenlabs_client.py` is the old SIP path** and is not on the live dial path (Plivo places calls now — see CLAUDE.md). Don't wire language into it.
