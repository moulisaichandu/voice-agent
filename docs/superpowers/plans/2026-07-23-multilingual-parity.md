# Multilingual Pipeline Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the two real gaps found while investigating "two-way worked in
Telugu but not Hindi" — the RAG query-translation shim never explicitly
covered Hindi/Hinglish, and there was no fast way to confirm live ElevenLabs
agent language configuration — and hand the owner an executable path to
verify and enable Telugu/Tinglish two-way.

**Architecture:** No new subsystems. Task 1 extends the existing
`app/rag/translate.py` query-translation shim (already used by
`search_relevant()`'s retry-on-miss path) to name all 5 catalogue languages
instead of 3. Task 2 adds one read-only CLI script that calls the
already-existing `agent_language_support()` against the live ElevenLabs
two-way agent. Task 3 is a manual, owner-run verification pass — no code —
against the already-implemented Telugu two-way path in `openai_bridge.py`.

**Tech Stack:** Python 3.12, FastAPI, pytest (`pytest-asyncio`, `monkeypatch`
for mocking), `httpx`/`elevenlabs` SDK (already a dependency), `ruff`.

## Global Constraints

- Zero network calls in `tests/unit/` — mock `embed_text`,
  `app.db.rag_store.match_chunks`, and `translate_to_english` exactly as the
  existing tests in `tests/unit/test_rag_search.py` do. (Design doc, spec's
  Testing section.)
- The RAG corpus stays English-only — no new source documents, no language
  column, no ingestion changes. (Design doc, Non-goals.)
- `app/rag/translate.py`'s system prompt must say nothing about
  courses/fees/domain — a prior version that hinted at the domain caused the
  model to invent course-shaped answers for unrelated input (see the
  module's existing docstring, lines 29-35). Any prompt edit preserves this.
- Never wire `search_permissive()` into any live-reachable path — untouched
  by this plan, but every new/edited test in `test_rag_search.py` must keep
  living in the same file that pins this guarantee.
- No changes to `preflight.py`, `app/admin/campaigns.py`, or `worker.py` —
  investigation found their guards already fail closed correctly; this plan
  only adds a diagnostic script alongside them. (Design doc, Workstream B.)
- `OPENAI_TWOWAY_ENABLED` is not flipped by this plan under any task — it is
  a production config change gated on the owner completing Task 10 in
  `docs/superpowers/plans/2026-07-22-telugu-openai-realtime-backend.md` on a
  real phone call. (Design doc, Workstream A.)
- Secrets come from `.env` only (`ELEVENLABS_API_KEY`,
  `ELEVENLABS_TWOWAY_AGENT_ID`) — the new script reads them via
  `app.config`, never hardcodes or logs their values. (CLAUDE.md.)

---

### Task 1: Hindi/Hinglish coverage in the RAG query-translation shim

**Files:**
- Create: `tests/unit/test_rag_translate.py`
- Modify: `app/rag/translate.py:36-48` (the `_SYSTEM` prompt)
- Modify: `tests/unit/test_rag_search.py` (append two tests after the
  existing Telugu section, before the "Pinning test" section at line 187)

**Interfaces:**
- Consumes: `app.rag.translate._SYSTEM` (module-level `str` constant),
  `app.rag.search.search_relevant(query: str) -> str` (existing, unchanged
  signature), `app.rag.search.translate_to_english` (existing, patched via
  `monkeypatch.setattr(search, "translate_to_english", fake)` — same pattern
  already used for Telugu at `test_rag_search.py:97`).
- Produces: nothing new consumed by later tasks — Task 2 is independent of
  this one.

- [ ] **Step 1: Write the failing prompt-coverage test**

Create `tests/unit/test_rag_translate.py`:

```python
"""app/rag/translate.py's _SYSTEM prompt — language coverage regression pin.

Static content checks only: no network call, no OpenAI client construction.
Guards against the prompt silently regressing back to naming only a subset
of the 5 catalogue languages (see app/languages.py's LANGUAGES) — the exact
gap this file exists to close for Hindi/Hinglish.
"""

from app.rag import translate


def test_system_prompt_names_all_five_catalogue_languages():
    prompt = translate._SYSTEM.lower()
    for language in ("english", "hindi", "telugu", "hinglish", "tinglish"):
        assert language in prompt, f"{language!r} is not named in translate._SYSTEM"


def test_system_prompt_still_says_nothing_about_courses_or_fees():
    """Regression guard for the documented failure mode in this module's
    docstring: naming the domain caused the model to turn an unrelated
    question ("how is the weather today?") into an invented course question.
    """
    prompt = translate._SYSTEM.lower()
    for banned in ("course", "fee", "curriculum", "price"):
        assert banned not in prompt, f"{banned!r} must not appear in translate._SYSTEM"
```

- [ ] **Step 2: Run it to verify the first test fails**

Run: `pytest tests/unit/test_rag_translate.py -v`
Expected: `test_system_prompt_names_all_five_catalogue_languages` FAILS
(`AssertionError: 'hindi' is not named in translate._SYSTEM`) —
`test_system_prompt_still_says_nothing_about_courses_or_fees` PASSES (the
current prompt already avoids those words).

- [ ] **Step 3: Update the prompt to name all 5 languages**

In `app/rag/translate.py`, replace lines 36-48 (the `_SYSTEM` constant):

```python
_SYSTEM = (
    "You are a literal translator. Translate the user's message into English.\n"
    "The input may be English, Hindi, Telugu, Hindi written in Latin script "
    "('Hinglish'), or Telugu written in Latin script ('Tinglish').\n"
    "Rules:\n"
    "- Translate faithfully and literally. Preserve the exact subject matter.\n"
    "- Do NOT infer intent, add context, answer the question, or make the "
    "message about any particular topic.\n"
    "- If the message is unrelated to education or courses, translate it "
    "unchanged anyway — an unrelated question must stay unrelated.\n"
    "- If it is already entirely English, repeat it back verbatim.\n"
    "- Reply with ONLY the translation: no quotes, notes, or explanation."
)
```

(Only the second line changes — from naming 3 languages to naming all 5. The
"unrelated to education" rule and every other line are unchanged, since they
already work correctly for every language and don't mention the domain.)

- [ ] **Step 4: Run it to verify both tests pass**

Run: `pytest tests/unit/test_rag_translate.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Add Hindi/Hinglish integration tests mirroring the Telugu ones**

In `tests/unit/test_rag_search.py`, immediately after
`test_telugu_miss_is_retried_in_english_and_then_hits` (after line 103, before
the blank line preceding `test_translation_retry_never_weakens_the_relevance_threshold`),
insert:

```python
HINDI_FEE_QUERY = "कोर्स की फीस कितनी है?"
HINGLISH_FEE_QUERY = "Course ki fees kitni hai?"


async def test_hindi_miss_is_retried_in_english_and_then_hits(monkeypatch):
    monkeypatch.setattr(search, "RAG_TRANSLATE_ON_MISS", True)
    seen_queries = []

    def fake_embed(text):
        seen_queries.append(text)
        return [1.0, 0.0]

    async def fake_match_chunks(embedding, *, match_count, min_score):
        if seen_queries[-1] == HINDI_FEE_QUERY:
            return []
        return [{"content": "Total Fee: 1,50,000 rupees.", "section": "fees", "score": 0.46}]

    async def fake_translate(q):
        assert q == HINDI_FEE_QUERY
        return "What is the course fee?"

    monkeypatch.setattr(search, "embed_text", fake_embed)
    monkeypatch.setattr(search, "translate_to_english", fake_translate)
    monkeypatch.setattr("app.db.rag_store.match_chunks", fake_match_chunks)

    result = await search.search_relevant(HINDI_FEE_QUERY)

    assert "1,50,000" in result
    assert seen_queries == [HINDI_FEE_QUERY, "What is the course fee?"]


async def test_hinglish_miss_is_retried_in_english_and_then_hits(monkeypatch):
    monkeypatch.setattr(search, "RAG_TRANSLATE_ON_MISS", True)
    seen_queries = []

    def fake_embed(text):
        seen_queries.append(text)
        return [1.0, 0.0]

    async def fake_match_chunks(embedding, *, match_count, min_score):
        if seen_queries[-1] == HINGLISH_FEE_QUERY:
            return []
        return [{"content": "Total Fee: 1,50,000 rupees.", "section": "fees", "score": 0.46}]

    async def fake_translate(q):
        assert q == HINGLISH_FEE_QUERY
        return "What is the course fee?"

    monkeypatch.setattr(search, "embed_text", fake_embed)
    monkeypatch.setattr(search, "translate_to_english", fake_translate)
    monkeypatch.setattr("app.db.rag_store.match_chunks", fake_match_chunks)

    result = await search.search_relevant(HINGLISH_FEE_QUERY)

    assert "1,50,000" in result
    assert seen_queries == [HINGLISH_FEE_QUERY, "What is the course fee?"]
```

These pin that `search_relevant()`'s retry-on-miss path is language-agnostic
(it dispatches to `translate_to_english` regardless of script), guarding
against a future regression where Hindi/Hinglish is special-cased out.

- [ ] **Step 6: Run the full RAG test file to verify everything passes**

Run: `pytest tests/unit/test_rag_search.py tests/unit/test_rag_translate.py -v`
Expected: all tests PASS (the two new integration tests pass immediately —
they pin already-correct language-agnostic behavior in `search.py`, not new
logic; only Step 2's prompt-coverage test was the genuinely failing one).

- [ ] **Step 7: Lint and commit**

Run: `ruff check app/rag/translate.py tests/unit/test_rag_translate.py tests/unit/test_rag_search.py`
Expected: clean, no output.

```bash
git add app/rag/translate.py tests/unit/test_rag_translate.py tests/unit/test_rag_search.py
git commit -m "Cover Hindi and Hinglish in the RAG query-translation prompt"
```

---

### Task 2: Live ElevenLabs agent language-support diagnostic script

**Files:**
- Create: `scripts/check_agent_language_support.py`

**Interfaces:**
- Consumes: `app.telephony.elevenlabs_client.agent_language_support(agent_id: str) -> dict`
  (existing, unchanged — returns `{"override_allowed": bool, "languages": set[str], "tts_model": str | None}`,
  per `elevenlabs_client.py:139-184`), `app.config.ELEVENLABS_TWOWAY_AGENT_ID: str | None`,
  `app.languages.LANGUAGES` (existing catalogue, for the human-readable
  language list printed on failure).
- Produces: nothing consumed by another task — this is a standalone
  operator tool, same category as `scripts/check_plivo_zentrunk.py`.

- [ ] **Step 1: Write the script**

Create `scripts/check_agent_language_support.py`:

```python
#!/usr/bin/env python
"""
check_agent_language_support.py — What can the live two-way ElevenLabs agent
actually speak?

Investigating "Hindi two-way doesn't work": app/telephony/preflight.py and
app/admin/campaigns.py both already refuse to dial a language the agent
isn't configured for, but the underlying dashboard state (which languages
are under "Additional Languages", and whether the `language` field is
enabled under Security -> Overrides) lives entirely on ElevenLabs' side, not
in this repo. This prints exactly what the API reports for
ELEVENLABS_TWOWAY_AGENT_ID, so a missing language shows up directly instead
of being inferred from a failed dial.

Read-only: calls conversational_ai.agents.get(), never modifies the agent.

Usage:
    python scripts/check_agent_language_support.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ELEVENLABS_TWOWAY_AGENT_ID  # noqa: E402
from app.telephony.elevenlabs_client import agent_language_support  # noqa: E402

if not ELEVENLABS_TWOWAY_AGENT_ID:
    sys.exit("ELEVENLABS_TWOWAY_AGENT_ID is not set in .env — nothing to check.")

print(f"Checking agent {ELEVENLABS_TWOWAY_AGENT_ID} ...")
try:
    support = agent_language_support(ELEVENLABS_TWOWAY_AGENT_ID)
except Exception as exc:
    sys.exit(f"Could not reach ElevenLabs ({type(exc).__name__}: {exc}). "
              "Check ELEVENLABS_API_KEY and network access.")

print(f"  override_allowed : {support['override_allowed']}")
print(f"  languages         : {sorted(support['languages']) or '(none reported)'}")
print(f"  tts_model         : {support['tts_model']}")
print()

problems = []
if not support["override_allowed"]:
    problems.append(
        "The 'language' override is NOT enabled. Open this agent's Security "
        "tab in the ElevenLabs dashboard and enable the 'language' override."
    )
if "hi" not in support["languages"]:
    problems.append(
        "'hi' (Hindi) is not in this agent's configured languages. Open the "
        "agent's Additional Languages setting and add Hindi."
    )

if problems:
    print("PROBLEMS FOUND:")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)
else:
    print("Hindi is fully configured on this agent — override allowed, "
          "'hi' present. If Hindi calls are still failing, the cause is "
          "something other than this agent's language configuration.")
```

- [ ] **Step 2: Verify it imports and parses cleanly**

Run: `python -c "import ast; ast.parse(open('scripts/check_agent_language_support.py').read())"`
Expected: no output, exit code 0 (confirms valid syntax without needing live
credentials or executing the ElevenLabs call).

- [ ] **Step 3: Lint and commit**

Run: `ruff check scripts/check_agent_language_support.py`
Expected: clean, no output.

```bash
git add scripts/check_agent_language_support.py
git commit -m "Add a read-only diagnostic for the live two-way agent's language config"
```

- [ ] **Step 4: Owner runs it against the real agent (manual, not part of this commit)**

```
python scripts/check_agent_language_support.py
```

If it reports `hi` missing or `override_allowed: False`, fix it on the
ElevenLabs dashboard exactly as the script's own output instructs, then
re-run to confirm. If it reports everything correctly configured, Hindi's
root cause is not the agent's language settings — escalate with the actual
call's `conversation_id` and the transcript from that call for further
diagnosis (out of scope for this plan).

---

### Task 3: Telugu/Tinglish two-way go-live (manual, owner-run — no code)

**Files:** none created or modified by this task. References
`docs/superpowers/plans/2026-07-22-telugu-openai-realtime-backend.md`
(existing file, Task 10, lines 1448-1456).

This task has no automated test cycle — it is the live-call verification
already scoped as Task 10 in the existing Telugu plan doc, which this plan
does not duplicate. Steps for the owner:

- [ ] **Step 1: Confirm the full suite is green and lint is clean**

```bash
pytest -q -m "not integration"
ruff check .
```

Expected: both clean, including Task 1 and Task 2's additions above.

- [ ] **Step 2: Run Task 10's checklist on a real call**

Open `docs/superpowers/plans/2026-07-22-telugu-openai-realtime-backend.md`
and work through Task 10 (lines 1448-1456) exactly as written: create a
two-way Telugu campaign using your own number, place the call, and verify
each item (disclosure-first, real Telugu Q&A sourced from the RAG documents,
barge-in truncation, off-topic refusal, no `[compliance]` ERROR log line).

- [ ] **Step 3: Record the result in that same plan doc and commit**

Check off Task 10's items in
`docs/superpowers/plans/2026-07-22-telugu-openai-realtime-backend.md` as
they're verified, and commit that file's update — do not create a separate
runbook file (see this plan's spec, Workstream A, on why).

```bash
git add docs/superpowers/plans/2026-07-22-telugu-openai-realtime-backend.md
git commit -m "Record Task 10 live verification results"
```

- [ ] **Step 4: Flip the flag, only after Step 2 fully passes**

In `.env`, set:

```
OPENAI_TWOWAY_ENABLED=true
```

This is a production behavior change — per this plan's Global Constraints,
it is not done automatically by any earlier task, and should only happen
once every item in Task 10 has genuinely passed on a real call.

---

## Self-Review Notes

- **Spec coverage:** Workstream A -> Task 3 (runbook pointer, no duplicate
  file). Workstream B -> Task 2 (diagnostic script). Workstream C -> Task 1
  (prompt + tests). All three design-doc workstreams have a task.
- **Placeholder scan:** no TBD/TODO; every code step shows complete code,
  not a description of code.
- **Type/name consistency:** `agent_language_support()`'s return shape
  (`override_allowed`, `languages`, `tts_model`) is used identically in
  Task 2's script and matches `elevenlabs_client.py:139-184` and its
  existing caller in `preflight.py:172-218`. `search.RAG_TRANSLATE_ON_MISS`
  and `search.translate_to_english` patch targets in Task 1's new tests
  match the exact names already used by the pre-existing Telugu tests in
  the same file.
