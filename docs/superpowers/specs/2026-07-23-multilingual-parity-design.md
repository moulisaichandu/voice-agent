# Multilingual pipeline parity — design

## Context

The owner reported that two-way conversation worked when talking in Telugu
but not in Hindi. That report is surprising against CLAUDE.md's documented
state ("two-way currently works on the ElevenLabs backend only" — English,
Hindi, Hinglish — while Telugu/Tinglish are one-way-only on OpenAI Realtime),
so this investigation set out to explain the gap before treating it as either
a bug or a go-live decision.

Three findings came out of exploring the RAG pipeline, the language/backend
routing, and the two-way guards:

1. **Telugu/Tinglish two-way is already built**, not missing. `openai_bridge.py`
   implements RAG tool-calling, barge-in truncation, and an end-call tool for
   two-way calls (Milestone B of
   `docs/superpowers/plans/2026-07-22-telugu-openai-realtime-backend.md`), but
   it's gated behind `OPENAI_TWOWAY_ENABLED` (default `False`,
   `app/config.py:200`) pending a human-verified live call — Task 10 in that
   plan doc, never completed. If that flag was flipped locally for testing,
   that fully explains "Telugu worked."
2. **Hindi two-way failing is very likely a live ElevenLabs dashboard
   configuration gap, not a code defect.** There is one shared
   `ELEVENLABS_TWOWAY_AGENT_ID` for English/Hindi/Hinglish
   (`app/config.py:161-162`), distinguished only by a per-call
   `conversation_config_override.agent.language` (`bridge.py:113-115`). That
   override only works if the agent has Hindi added under "Additional
   Languages" *and* the `language` field is enabled in its Security tab.
   Reading `preflight.py:172-218` and `worker.py:261-266` shows the existing
   guard already fails closed correctly when ElevenLabs reports the language
   isn't supported or the override isn't allowed — it only fails open on a
   genuine SDK/API exception, which is intentional (an unrelated outage
   shouldn't block a call that would otherwise succeed). There is no
   code gap to patch here; the fix is confirming and, if needed, changing
   the live agent's dashboard config.
3. **The RAG query-translation shim never explicitly covered Hindi/Hinglish.**
   `app/rag/translate.py`'s system prompt only names "Telugu, English, or
   Telugu written in Latin script ('Tinglish')" — Hindi and Hinglish queries
   go through the same code path but were never named in the prompt or
   covered by a test. The corpus itself stays English-only by design (see
   `translate.py`'s module docstring); only the query side needs to be
   language-complete across all 5 catalogue tokens in `app/languages.py`'s
   `LANGUAGES`.

## Non-goals

- No new voice backend, no change to which language routes to which backend
  (`app/languages.py::backend_for` stays as-is).
- No RAG corpus changes — no native-language course documents are added; the
  corpus stays English-only and only the query-translation shim is hardened.
- No rewrite of the preflight/creation-time guards — investigation found them
  already correct; only a diagnostic tool is added.
- Flipping `OPENAI_TWOWAY_ENABLED` is a production behavior change gated on a
  real phone call the owner places — not done as part of this work, only
  prepared for.

## Workstream A — Telugu/Tinglish two-way verification runbook

Task 10 already exists, unchecked, in
`docs/superpowers/plans/2026-07-22-telugu-openai-realtime-backend.md` (lines
1448-1456) — no new runbook file is created; that would just duplicate it.
The implementation plan walks the owner through executing that existing
checklist in place and checking items off there:

- Full test suite green, `ruff check .` clean, before dialling.
- Create a two-way Telugu campaign using the owner's own number.
- On the call, confirm: AI disclosure is the first thing said; a spoken
  Telugu question is understood (verified via the stored transcript, not by
  ear); answers come from the course documents via the RAG tool, and the
  agent says it doesn't know rather than inventing when asked something
  outside them.
- Barge-in: interrupt mid-sentence, then ask about what it was saying — if it
  behaves as though it finished, `conversation.item.truncate` isn't working.
- Ask something off-topic (e.g. the weather) — it must decline rather than
  answer from general knowledge.
- Confirm no `[compliance]` ERROR line fired in the logs.
- Record the outcome directly in that plan doc's Task 10 section and commit.

Only after every item passes does `OPENAI_TWOWAY_ENABLED=True` get set in
`.env` — done by the owner (or by the assistant with the owner's explicit
go-ahead after Task 10 passes), never automatically.

## Workstream B — Hindi two-way diagnostic

Add a small read-only diagnostic script,
`scripts/check_agent_language_support.py`, that calls the existing
`app.telephony.elevenlabs_client.agent_language_support()` against
`ELEVENLABS_TWOWAY_AGENT_ID` and prints:

- `override_allowed` (must be `True` for any non-default language to work)
- `languages` (must include `hi`)
- `tts_model`

This is a thin CLI wrapper around code that already exists and is already
exercised by `preflight.py` — it exists purely so the owner can see the live
agent's actual state directly instead of inferring it from a failed dial.

The script's usage docstring includes the two concrete dashboard fixes if the
output shows the gap:

1. Add `hi` under the agent's **Additional Languages**.
2. Enable the **`language`** field under **Security → Overrides** on that
   same agent.

No changes to `preflight.py`, `campaigns.py`'s `_check_language_support`, or
`worker.py` — they were read closely during investigation and already do the
right thing.

## Workstream C — RAG translation coverage for all languages

`app/rag/translate.py`:

- Update `_SYSTEM` to explicitly enumerate all 5 catalogue tokens the query
  can arrive in — English, Hindi, Telugu, Hinglish (Hindi in Latin script),
  and Tinglish (Telugu in Latin script) — instead of naming only
  Telugu/English/Tinglish. Keep the existing constraint that the prompt says
  nothing about courses/fees (the module docstring explains why: naming the
  domain caused the model to invent course-shaped answers for unrelated
  input).
- Add test cases alongside the existing Telugu case (wherever it currently
  lives — `tests/unit/test_rag_search.py` or a dedicated
  `tests/unit/test_rag_translate.py`) for a Hindi-script query and a
  Hinglish-script query, asserting the translate-and-retry path in
  `search_relevant()` behaves the same way it does for Telugu today.

## Testing / verification

- `pytest -q -m "not integration"` — covers the new Hindi/Hinglish
  translation tests and confirms nothing else regresses.
- `ruff check .`
- Workstream A's runbook: manual, executed by the owner on a real call.
- Workstream B's diagnostic script: manual, run and read by the owner (or run
  by the assistant with explicit permission, since it's a live API call
  against production ElevenLabs infrastructure).

## Risks

- Task 10's live call is the only thing standing between "code complete" and
  "Telugu two-way live" — if it surfaces a real defect (e.g. barge-in not
  truncating), that becomes new follow-up work, not something this design
  anticipates further.
- The Hindi diagnostic may show the dashboard is already configured
  correctly, in which case the root cause is something not yet identified —
  the runbook for workstream B should say so plainly rather than assume the
  dashboard theory is confirmed until the script actually runs.
