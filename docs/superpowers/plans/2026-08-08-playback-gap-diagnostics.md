# Playback Gap Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log one `[sarvam]` line per spoken turn on the Sarvam two-way path
reporting how much dead air, if any, occurred *in the middle* of the
agent's own reply — total gap time, largest single gap, and how many
sentences the turn covered — so a batch of real calls can confirm or rule
out sequential per-sentence synthesis as the cause of a reported "voice
latency"/choppiness complaint.

**Architecture:** All new logic lives inside
`app/telephony/sarvam_bridge.py`'s `_Conversation.say()`, the method that
already loops sentence-by-sentence and calls `self.call.play()` per audio
frame. `PlivoCall` (`app/telephony/plivo_stream.py`) already tracks
`play_end` — the wall-clock time its queued audio will finish playing —
for every backend; `say()` compares that against the current time
immediately before each frame (after the response's own first frame) to
detect and accumulate any gap, then logs one summary line when the turn
is done.

**Tech Stack:** Pure Python, no new dependency. pytest/pytest-asyncio for
tests, using real (not mocked) event-loop timing — the same style already
used elsewhere in `tests/unit/test_sarvam_bridge.py` for barge-in and
reply-gate tests.

## Global Constraints

- No new dependency, no new env var / config toggle.
- No change to `app/telephony/plivo_stream.py` — this reads the existing
  public `PlivoCall.play_end` field only; `PlivoCall.play()`'s send
  behavior is untouched. `PlivoCall` is shared with the production
  ElevenLabs and OpenAI Realtime backends, so this stays entirely inside
  `sarvam_bridge.py`.
- No pacing/buffering/synthesis look-ahead added to close a gap — this
  plan only measures and logs.
- Log line format is fixed by the spec:
  `[sarvam] lead=<id> playback gaps: total=<f:.1f>ms max=<f:.1f>ms sentences=<n>`
- The response's very first frame is never counted as a gap — only frames
  after the response has already started playing.
- The line is logged once per `say()` call (success, interruption, or
  cancellation), but only if at least one frame was actually played.
- Every new/modified module needs tests in `tests/unit/` (mocked, no
  Docker) per CLAUDE.md conventions.

---

## Task 1: Log playback gaps in `say()`

**Files:**
- Modify: `app/telephony/sarvam_bridge.py:331-387` (`_Conversation.say()`),
  plus a new method added near `_log_turn_latency` (currently line 290)
- Test: `tests/unit/test_sarvam_bridge.py`

**Interfaces:**
- Consumes: `self.call.play_end` (existing public field on `PlivoCall`,
  `app/telephony/plivo_stream.py:135`) and `self._loop.time()` (existing
  on `_Conversation`).
- Produces: `_Conversation._log_playback_gaps(self, gap_total_ms: float, gap_max_ms: float, sentences: int) -> None`
  — logs the one `[sarvam] ... playback gaps: ...` line. No other task
  depends on this; it is called only from `say()`.

- [ ] **Step 1: Write the failing tests**

Add `import re` to the top of `tests/unit/test_sarvam_bridge.py`'s import
block (alongside the existing `import asyncio`, `import base64`,
`import json`).

Add these three tests after
`test_barge_in_forgets_what_the_lead_never_heard` (in the `# ── barge-in
──` section, or immediately after it — either is fine, this is a new
concern so a new comment header is appropriate):

```python
# ── playback gaps: dead air in the middle of the agent's own reply ──────────
#
# say() synthesizes one sentence at a time, sequentially — each is a fresh
# WebSocket round trip to Sarvam. If a short sentence's synthesis takes
# longer than its predecessor took to actually play out on the phone line,
# Plivo's queue drains before the next chunk arrives: dead air mid-reply.
# PlivoCall already tracks play_end (when queued audio finishes playing);
# these tests prove say() actually reads it. See
# docs/superpowers/specs/2026-08-08-playback-gap-diagnostics-design.md.

async def test_a_playback_gap_mid_response_is_logged(caplog):
    """A frame arriving well after the previous one finished playing is a
    real gap the lead heard, and it must show up in the logged total."""
    caplog.set_level("INFO")
    call = sarvam_bridge.plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="gap-test", one_way=False, protect_opening=False)
    turns = []
    convo = sarvam_bridge._Conversation(
        call, lead_id="gap-test", system_prompt="s", turns=turns)

    class _GappyTTS:
        async def speak(self, text):
            yield _FRAME, 20
            await asyncio.sleep(0.1)  # far longer than _FRAME's ~20ms play-out
            yield _FRAME, 20

    # No sentence-ending punctuation, so _split_sentences keeps this as ONE
    # sentence — one call to speak(), isolating exactly the gap under test.
    await convo.say(_GappyTTS(), "hello there")

    lines = [r.message for r in caplog.records if "playback gaps" in r.message]
    assert lines, "no playback-gap line was logged"
    assert "lead=gap-test" in lines[0]
    assert "sentences=1" in lines[0]
    match = re.search(r"total=(\d+\.\d+)ms", lines[0])
    assert match, f"couldn't parse total from: {lines[0]}"
    assert float(match.group(1)) > 50, f"expected a gap of at least 50ms: {lines[0]}"


async def test_fast_delivery_logs_zero_gap(caplog):
    """Frames arriving well within the previous frame's play-out window —
    the common case tonight's real calls already showed — must not be
    reported as a gap."""
    caplog.set_level("INFO")
    call = sarvam_bridge.plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="fast-test", one_way=False, protect_opening=False)
    turns = []
    convo = sarvam_bridge._Conversation(
        call, lead_id="fast-test", system_prompt="s", turns=turns)

    class _FastTTS:
        async def speak(self, text):
            yield _FRAME, 20
            yield _FRAME, 20
            yield _FRAME, 20

    await convo.say(_FastTTS(), "hello there")

    lines = [r.message for r in caplog.records if "playback gaps" in r.message]
    assert lines, "no playback-gap line was logged"
    assert "total=0.0ms max=0.0ms" in lines[0], (
        f"fast delivery must not read as a gap: {lines[0]}"
    )
    assert "sentences=1" in lines[0]


async def test_no_gap_line_when_nothing_was_spoken(caplog):
    """A TTS response that produces no audio frames at all (e.g. Sarvam
    returned an error before any audio chunk) has nothing to report — same
    guard _log_turn_latency already uses for an unanswered turn."""
    caplog.set_level("INFO")
    call = sarvam_bridge.plivo_stream.PlivoCall(
        _FakePlivoWS(), lead_id="silent-test", one_way=False, protect_opening=False)
    turns = []
    convo = sarvam_bridge._Conversation(
        call, lead_id="silent-test", system_prompt="s", turns=turns)

    class _SilentTTS:
        async def speak(self, text):
            return
            yield  # pragma: no cover - makes this an async generator

    await convo.say(_SilentTTS(), "hello there")

    lines = [r.message for r in caplog.records if "playback gaps" in r.message]
    assert not lines, f"a gap line was logged for a turn that never spoke: {lines}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_sarvam_bridge.py -k "playback_gap or fast_delivery or no_gap_line" -v`
Expected: FAIL — no "playback gaps" line is ever logged yet, so the first
two tests fail on `assert lines, "no playback-gap line was logged"` and
the third passes vacuously (skip re-checking it now; it will still be
exercised in Step 4).

- [ ] **Step 3: Implement**

In `app/telephony/sarvam_bridge.py`, replace `say()` (currently lines
331-387):

```python
    async def say(self, tts: sarvam_tts.SarvamTTS, text: str) -> None:
        """Speak *text*, recording what actually reached the lead.

        Each sentence is synthesised, played, and logged to the ledger with its
        audio duration. If the generation changes mid-way the lead has
        interrupted, so the rest is abandoned unplayed.
        """
        sentences = _split_sentences(text)
        if not sentences:
            return

        generation = self.generation
        self.ledger.reset()
        # Restart the playback clock: played_ms() is relative to the current
        # response, and a barge-in truncates against it.
        self.call.mark_response_boundary()

        self._agent_turn_index = len(self.turns)
        self.turns.append(TranscriptTurn(role="agent", text=text))
        self.history.append({"role": "assistant", "content": text})

        self._turn_say_started = self._loop.time()
        spoke_a_frame = False
        # Dead air detected IN THE MIDDLE of this response — see
        # _log_playback_gaps for what these mean and why they're measured
        # this way.
        gap_total_ms = 0.0
        gap_max_ms = 0.0
        sentences_started = 0

        async def reset_tts_after_cancel() -> None:
            # The production SarvamTTS exposes this hook. Keep the bridge
            # tolerant of the tiny test/tool TTS adapters that only implement
            # speak().
            reset = getattr(tts, "reset_after_cancel", None)
            if reset is not None:
                await reset()

        try:
            for sentence in sentences:
                if self.generation != generation or self.call.stop.is_set():
                    await reset_tts_after_cancel()
                    return
                sentences_started += 1
                spoken_ms = 0
                async for payload, duration_ms in tts.speak(sentence):
                    if self.generation != generation or self.call.stop.is_set():
                        await reset_tts_after_cancel()
                        return
                    if spoke_a_frame:
                        # Only meaningful once THIS response has already
                        # started playing — the delay before the very first
                        # frame is normal startup latency, already captured
                        # by _log_turn_latency's tts= figure, not a gap.
                        now = self._loop.time()
                        if now > self.call.play_end:
                            gap_ms = (now - self.call.play_end) * 1000
                            gap_total_ms += gap_ms
                            gap_max_ms = max(gap_max_ms, gap_ms)
                    await self.call.play(payload)
                    if not spoke_a_frame:
                        # The lead's wait ends HERE, at the first frame — not
                        # when the whole answer has been sent.
                        spoke_a_frame = True
                        self._log_turn_latency(spoke=True)
                    spoken_ms += duration_ms
                self.ledger.add(sentence, spoken_ms)
                self.note_activity()
        except asyncio.CancelledError:
            # The cancellation can land while call.play() is awaiting Plivo,
            # after speak() already yielded a frame. Reset explicitly because
            # the async generator may not receive that cancellation itself.
            await reset_tts_after_cancel()
            raise
        finally:
            # Runs whether the response finished, was interrupted, or was
            # cancelled — as long as something was actually played, there is
            # something to report.
            if spoke_a_frame:
                self._log_playback_gaps(gap_total_ms, gap_max_ms, sentences_started)
```

Then add a new method immediately after `_reset_turn_metrics` (currently
ending around line 327, right before `say()`):

```python
    def _log_playback_gaps(self, gap_total_ms: float, gap_max_ms: float,
                           sentences: int) -> None:
        """One INFO line per spoken turn: how much dead air, if any, the lead
        heard IN THE MIDDLE of this response.

        self.call.play_end is the wall-clock time Plivo's queued audio will
        finish playing, already maintained by PlivoCall.play() for every
        backend. If a later frame arrives after play_end has passed, the
        queue had already drained — dead air between whatever played last
        and this new chunk. say() accumulates that across the whole
        response and logs it here regardless of how the response ended, as
        long as at least one frame was actually played.

        A turn with zero gaps still logs (total=0.0ms) — the point is a
        dataset across many real calls, the same reasoning
        app/telephony/sarvam_stt.py's _log_audio_health uses for the
        inbound side. See
        docs/superpowers/specs/2026-08-08-playback-gap-diagnostics-design.md.
        """
        logger.info(
            f"[sarvam] lead={self.lead_id} playback gaps: "
            f"total={gap_total_ms:.1f}ms max={gap_max_ms:.1f}ms "
            f"sentences={sentences}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_sarvam_bridge.py -k "playback_gap or fast_delivery or no_gap_line" -v`
Expected: 3 passed

- [ ] **Step 5: Run the full non-integration suite**

Run: `pytest -q -m "not integration"`
Expected: all tests PASS (no regressions — `say()`'s public behavior,
return value, and side effects on `self.turns`/`self.history`/`self.ledger`
are unchanged; only new local bookkeeping and one new log call were added)

- [ ] **Step 6: Lint**

Run: `ruff check app/telephony/sarvam_bridge.py tests/unit/test_sarvam_bridge.py`
Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add app/telephony/sarvam_bridge.py tests/unit/test_sarvam_bridge.py
git commit -m "$(cat <<'EOF'
Log playback gaps in the middle of the agent's own reply

say() synthesizes one sentence at a time, sequentially - each a fresh
round trip to Sarvam. If a short sentence's synthesis outlasts its
predecessor's play-out time, Plivo's queue drains before the next
chunk arrives: dead air mid-reply, not silence between turns.
PlivoCall already tracks play_end; this reads it and logs one
[sarvam] ... playback gaps: ... line per spoken turn, mirroring the
inbound audio-health diagnostic already shipped tonight. Diagnostic
only - no change to how or when audio is actually sent.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Deploy and collect real data

**Files:** none (operational task)

- [ ] **Step 1: Rebuild the backend image**

Run: `docker compose build backend`

- [ ] **Step 2: Recreate the running container**

Run: `docker compose up -d --force-recreate backend`

- [ ] **Step 3: Verify the new code is live**

Run:
```bash
sleep 3
docker exec voice-agent-elevenlabs-backend-1 sh -c "grep -c '_log_playback_gaps' /app_root/app/telephony/sarvam_bridge.py"
curl -s -o /dev/null -w "health=%{http_code}\n" http://localhost:8091/health
```
Expected: a non-zero count and `health=200`.

- [ ] **Step 4: Place a real two-way Telugu/Tinglish call**

Manual — done by the owner, or by the assistant with the owner's explicit
go-ahead per CLAUDE.md's compliance rules on dialing. A longer reply (one
that produces several sentences, e.g. a course-fee or course-list
question that gets a multi-sentence answer) is the useful case to
listen to and correlate against.

- [ ] **Step 5: Pull and read the logs**

Run:
```bash
docker logs voice-agent-elevenlabs-backend-1 --since 1h 2>&1 | grep -E "\[sarvam\].*playback gaps"
```

Line up each `playback gaps` line's `total`/`max`/`sentences` against
what was actually heard on that turn (the owner's ear is the ground
truth here — this project has no call recording). Read the result
against the hypothesis:

- Nonzero `total`/`max` correlating with turns the owner heard as choppy
  → the inter-sentence-synthesis hypothesis is confirmed; the fix is a
  new, separate design (e.g. pipelining the next sentence's synthesis
  while the current one is still playing), not something to attempt from
  this plan.
- `total=0.0ms` on every turn, even ones the owner heard as choppy → the
  hypothesis is ruled out; the latency complaint is coming from
  somewhere else not yet instrumented (turn-taking dead air, which is
  already measured by the existing `[sarvam] ... turn latency: ...`
  line, or something outside this codebase entirely — the phone network,
  Plivo's own playback, or the device on the lead's end).
- Inconclusive from one call → place more calls before drawing a
  conclusion; this task's job is done once the logging exists and is
  verified live, not once the underlying question is answered from one
  batch.

One confound to read around: `self._loop.time()` and `self.call.play_end`
are both measured on this process's own event loop, so a gap this logs
could be inflated by THIS PROCESS being busy elsewhere (another
concurrent call's LLM round trip, another call's TTS) rather than by
Sarvam's synthesis actually being slow. If a gap shows up specifically on
calls that ran concurrently with others, re-test with only one call in
flight before concluding the inter-sentence-synthesis hypothesis is
confirmed.

This task has no automated pass/fail — its deliverable is the data and
the owner's read of it against what they actually heard.
