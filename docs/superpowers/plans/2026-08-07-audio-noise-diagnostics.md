# Audio Noise Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log two lightweight, per-utterance audio-quality signals (noise-floor
RMS and frame-delivery timing) on the Sarvam two-way STT path, from data
already in memory, so a batch of real calls can show whether garbled
transcripts come from genuine background noise or a connection/quality
problem — before any noise-suppression code gets written.

**Architecture:** All new logic lives inside `app/telephony/sarvam_stt.py`'s
`SarvamSTT`, which already decodes every inbound mu-law frame to PCM16 and
already parses Sarvam's START_SPEECH/END_SPEECH signals. Two pure functions
compute a sum-of-squares/count and an RMS from that; `SarvamSTT` accumulates
them into two windows (silence vs. speech) reset at VAD boundaries, plus a
frame-arrival timing gap, and logs one `[sarvam-stt]` INFO line per utterance
at END_SPEECH. `app/telephony/sarvam_bridge.py` threads its `lead_id` through
to `SarvamSTT`'s constructor so the log line correlates with existing
`[sarvam]` transcript/turn-latency lines.

**Tech Stack:** Pure Python (no numpy/scipy — matches `ulaw.py`'s existing
"pure Python on purpose" constraint), pytest/pytest-asyncio for tests.

## Global Constraints

- No new dependency — everything is plain arithmetic over PCM16 samples
  already decoded by `ulaw.decode()`.
- No new env var / config toggle — the logging always runs (matches
  `sarvam_bridge.py`'s existing turn-latency logging, which has no toggle).
- No behavior change to the call itself — `SarvamSTT`'s public interface
  gains one optional constructor keyword (`lead_id`); `send_audio()` and
  `events()` keep their existing signatures and return values.
- `app/telephony/sarvam_stt.py`'s module docstring states it "knows nothing
  about calls, turns or prompts" — this plan must not violate that. The new
  state is self-contained (VAD signals + decoded PCM this module already
  owns); no reference to `_Conversation` or `sarvam_bridge` internals.
- Log line format is fixed by the spec:
  `[sarvam-stt] lead=<id> audio health: silence_rms=<f> speech_rms=<f> frame_gap_max_ms=<f> frames=<n>`
- Every new/modified module needs tests in `tests/unit/` (mocked, no
  Docker) per CLAUDE.md conventions.

---

## Task 1: RMS accumulation helper functions

**Files:**
- Modify: `app/telephony/sarvam_stt.py` (add module-level functions, near
  the top of the file after `_stt_url()`, around line 100)
- Test: `tests/unit/test_sarvam_stt.py`

**Interfaces:**
- Produces: `_sumsq_and_count(pcm: bytes) -> tuple[int, int]` — sum of
  squared little-endian signed-16-bit samples, and how many samples were
  read. A trailing odd byte (not a full sample) is dropped, matching
  `ulaw.encode()`'s existing convention for the same situation. Empty input
  returns `(0, 0)`.
- Produces: `_rms(sumsq: int, count: int) -> float` — `sqrt(sumsq / count)`,
  or `0.0` if `count == 0` (nothing measured, not "silence measured as
  zero"). Combining raw sums this way — rather than averaging per-frame RMS
  values — is required for correctness: the RMS of several frames
  concatenated is NOT the average of each frame's own RMS.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_sarvam_stt.py`, in a new section after the existing
imports/helpers (after line 27, before the `_transcript` helper):

```python
# ── audio-health accumulation ────────────────────────────────────────────────

def test_sumsq_and_count_of_known_samples():
    """Two samples, 3 and 4 — sum of squares is 25, count is 2. Chosen so the
    RMS ends up being sqrt(12.5), an easy value to check by hand."""
    pcm = (3).to_bytes(2, "little", signed=True) + (4).to_bytes(2, "little", signed=True)
    assert sarvam_stt._sumsq_and_count(pcm) == (25, 2)


def test_sumsq_and_count_of_empty_pcm_is_zero():
    assert sarvam_stt._sumsq_and_count(b"") == (0, 0)


def test_sumsq_and_count_drops_a_trailing_odd_byte():
    """A stray half-sample must not be read as a sample — same reasoning as
    ulaw.encode()'s trailing-byte handling."""
    pcm = (5).to_bytes(2, "little", signed=True) + b"\x01"
    assert sarvam_stt._sumsq_and_count(pcm) == (25, 1)


def test_rms_of_the_3_4_example():
    assert sarvam_stt._rms(25, 2) == pytest.approx(12.5 ** 0.5)


def test_rms_with_no_samples_is_zero():
    """Zero samples measured, not a silent signal — the caller (send_audio's
    accumulator) never has zero samples for a frame it actually received, but
    the function must not divide by zero if it's ever called with none."""
    assert sarvam_stt._rms(0, 0) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_sarvam_stt.py -k "sumsq or rms" -v`
Expected: FAIL with `AttributeError: module 'app.telephony.sarvam_stt' has no attribute '_sumsq_and_count'`

- [ ] **Step 3: Implement the functions**

In `app/telephony/sarvam_stt.py`, add after `_stt_url()` (before
`class SarvamSTT:`, currently line 102):

```python
def _sumsq_and_count(pcm: bytes) -> tuple[int, int]:
    """Sum of squared samples and sample count, for RMS accumulation.

    Raw sums, not a single RMS — multiple frames combine correctly this way;
    averaging per-frame RMS values is NOT the same as the RMS of the
    combined signal. A trailing odd byte cannot be half a sample, so it is
    dropped, matching ulaw.encode()'s existing convention.
    """
    usable = len(pcm) - (len(pcm) % 2)
    samples = [
        int.from_bytes(pcm[i:i + 2], "little", signed=True)
        for i in range(0, usable, 2)
    ]
    return sum(s * s for s in samples), len(samples)


def _rms(sumsq: int, count: int) -> float:
    """Root-mean-square from an accumulated sum-of-squares and sample count.

    0 samples returns 0.0 — "nothing was measured", not "the signal was
    silent". Callers only pass a genuine zero-sample window when nothing
    happened during it (e.g. an utterance with no leading silence).
    """
    return (sumsq / count) ** 0.5 if count else 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_sarvam_stt.py -k "sumsq or rms" -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add app/telephony/sarvam_stt.py tests/unit/test_sarvam_stt.py
git commit -m "$(cat <<'EOF'
Add RMS accumulation helpers for audio-health diagnostics

Pure functions, no class changes yet: sum-of-squares/count and RMS
from that, the building blocks for per-utterance noise-floor logging
on the Sarvam STT path.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Wire per-utterance audio-health logging into `SarvamSTT`

**Files:**
- Modify: `app/telephony/sarvam_stt.py:34-55` (imports), `:109` (`__init__`),
  `:130` (`send_audio`), `:162` (`events`)
- Test: `tests/unit/test_sarvam_stt.py`

**Interfaces:**
- Consumes: `_sumsq_and_count(pcm: bytes) -> tuple[int, int]`,
  `_rms(sumsq: int, count: int) -> float` (Task 1)
- Produces: `SarvamSTT.__init__(self, *, lead_id: str = "") -> None` — new
  optional keyword, defaults to `""` so every existing call site (including
  all of `test_sarvam_stt.py`'s current tests, which construct
  `SarvamSTT()` with no arguments) keeps working unchanged.
- Produces: one `logging.INFO` line per utterance, matching exactly:
  `[sarvam-stt] lead=<lead_id> audio health: silence_rms=<f:.1f> speech_rms=<f:.1f> frame_gap_max_ms=<f:.1f> frames=<n>`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_sarvam_stt.py`, after the existing
`test_speech_start_and_end_are_surfaced` test:

```python
async def test_audio_health_is_logged_once_per_utterance(connected, caplog):
    """One INFO line at END_SPEECH, correlating this utterance's noise floor
    (before it started) against how loud the speech itself was — the whole
    point being to tell genuine background noise apart from a connection
    problem using real calls, not recorded audio (this project has none).
    See docs/superpowers/specs/2026-08-07-audio-noise-diagnostics-design.md.
    """
    caplog.set_level("INFO")
    connected(_FakeSTTWS([_vad("START_SPEECH"), _vad("END_SPEECH")]))

    quiet = (0).to_bytes(2, "little", signed=True) * 80
    loud = (12000).to_bytes(2, "little", signed=True) * 80
    quiet_mulaw_b64 = base64.b64encode(ulaw.encode(quiet)).decode()
    loud_mulaw_b64 = base64.b64encode(ulaw.encode(loud)).decode()

    async with sarvam_stt.SarvamSTT(lead_id="lead-9") as stt:
        await stt.send_audio(quiet_mulaw_b64)  # before START_SPEECH: silence window
        events_iter = stt.events()
        started = await events_iter.__anext__()
        assert started.kind == "speech_started"
        await stt.send_audio(loud_mulaw_b64)   # after START_SPEECH: speech window
        ended = await events_iter.__anext__()
        assert ended.kind == "speech_ended"

    expected_silence = sarvam_stt._rms(*sarvam_stt._sumsq_and_count(ulaw.decode(
        base64.b64decode(quiet_mulaw_b64))))
    expected_speech = sarvam_stt._rms(*sarvam_stt._sumsq_and_count(ulaw.decode(
        base64.b64decode(loud_mulaw_b64))))

    lines = [r.message for r in caplog.records if "audio health" in r.message]
    assert lines, "no audio-health line was logged for the utterance"
    line = lines[0]
    assert "lead=lead-9" in line
    assert f"silence_rms={expected_silence:.1f}" in line
    assert f"speech_rms={expected_speech:.1f}" in line
    assert "frames=2" in line


async def test_lead_id_defaults_to_empty_string(connected, caplog):
    """The constructor's lead_id is optional so every other test in this
    file (and every existing call site) keeps constructing SarvamSTT() with
    no arguments."""
    caplog.set_level("INFO")
    connected(_FakeSTTWS([_vad("START_SPEECH"), _vad("END_SPEECH")]))

    async with sarvam_stt.SarvamSTT() as stt:
        async for event in stt.events():
            if event.kind == "speech_ended":
                break

    lines = [r.message for r in caplog.records if "audio health" in r.message]
    assert lines, "no audio-health line was logged"
    assert "lead= audio health" in lines[0], (
        f"expected an empty lead_id to render as 'lead= audio health', got: {lines[0]!r}"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_sarvam_stt.py -k "audio_health or lead_id_defaults" -v`
Expected: FAIL — `SarvamSTT() takes no arguments` / no "audio health" log
line exists yet.

- [ ] **Step 3: Implement the wiring**

In `app/telephony/sarvam_stt.py`:

1. Add `import time` to the import block (after `import logging` at line 39):

```python
import time
```

2. Replace `__init__` (currently line 109-110):

```python
    def __init__(self) -> None:
        self._ws: Any = None
```

with:

```python
    def __init__(self, *, lead_id: str = "") -> None:
        self._ws: Any = None
        # Audio-health accumulators — see _log_audio_health. All of this is
        # measured from data this module already decodes for Sarvam; no new
        # coupling to sarvam_bridge or call/turn state.
        self._lead_id = lead_id
        self._speaking = False
        self._silence_sumsq = 0
        self._silence_count = 0
        self._speech_sumsq = 0
        self._speech_count = 0
        self._last_frame_at: float | None = None
        self._frame_gap_max_ms = 0.0
        self._frame_count = 0
```

3. In `send_audio` (currently lines 130-160), after the successful decode
(right after the `except (binascii.Error, ValueError):` block, before the
`await self._ws.send(...)` call), add a call to track the frame:

```python
        self._track_frame(pcm)

        await self._ws.send(json.dumps({
```

(the rest of the method body is unchanged — only the one new line is
inserted before the existing `await self._ws.send(...)`)

4. After `send_audio`, before `events` (i.e. as a new method on `SarvamSTT`),
add:

```python
    def _track_frame(self, pcm: bytes) -> None:
        """Accumulate one successfully decoded inbound frame into whichever
        RMS window is currently active, and update frame-delivery timing.
        """
        sumsq, count = _sumsq_and_count(pcm)
        if self._speaking:
            self._speech_sumsq += sumsq
            self._speech_count += count
        else:
            self._silence_sumsq += sumsq
            self._silence_count += count

        now = time.monotonic()
        if self._last_frame_at is not None:
            gap_ms = (now - self._last_frame_at) * 1000
            self._frame_gap_max_ms = max(self._frame_gap_max_ms, gap_ms)
        self._last_frame_at = now
        self._frame_count += 1

    def _log_audio_health(self) -> None:
        """One INFO line per utterance: what the audio actually looked like,
        from frames already decoded for Sarvam.

        silence_rms covers the window since the PREVIOUS END_SPEECH (the
        quiet stretch before this utterance); speech_rms covers the window
        since the START_SPEECH that opened it. Correlate against
        app/telephony/sarvam_bridge.py's [sarvam] transcript/turn-latency
        lines by lead_id and timestamp to tell genuine background noise
        (elevated silence_rms) apart from a connection/quality problem
        (irregular frame_gap_max_ms) — see
        docs/superpowers/specs/2026-08-07-audio-noise-diagnostics-design.md.
        """
        silence_rms = _rms(self._silence_sumsq, self._silence_count)
        speech_rms = _rms(self._speech_sumsq, self._speech_count)
        logger.info(
            f"[sarvam-stt] lead={self._lead_id} audio health: "
            f"silence_rms={silence_rms:.1f} speech_rms={speech_rms:.1f} "
            f"frame_gap_max_ms={self._frame_gap_max_ms:.1f} "
            f"frames={self._frame_count}"
        )
        self._silence_sumsq = 0
        self._silence_count = 0
        self._frame_gap_max_ms = 0.0
        self._frame_count = 0
```

5. In `events` (currently lines 162-194), update the VAD branch:

```python
            elif etype == "events":
                signal = data.get("signal_type")
                if signal == "START_SPEECH":
                    yield STTEvent("speech_started")
                elif signal == "END_SPEECH":
                    yield STTEvent("speech_ended")
```

becomes:

```python
            elif etype == "events":
                signal = data.get("signal_type")
                if signal == "START_SPEECH":
                    self._speaking = True
                    self._speech_sumsq = 0
                    self._speech_count = 0
                    yield STTEvent("speech_started")
                elif signal == "END_SPEECH":
                    self._speaking = False
                    self._log_audio_health()
                    yield STTEvent("speech_ended")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_sarvam_stt.py -v`
Expected: all tests in the file PASS (existing tests unaffected — `lead_id`
defaults to `""`, `send_audio`'s return value and side effects on `self._ws`
are unchanged)

- [ ] **Step 5: Run the full non-integration suite to check for regressions**

Run: `pytest -q -m "not integration"`
Expected: all tests PASS (no other module references `SarvamSTT.__init__`'s
old zero-argument-only signature in a way that would break — it still
accepts zero arguments)

- [ ] **Step 6: Commit**

```bash
git add app/telephony/sarvam_stt.py tests/unit/test_sarvam_stt.py
git commit -m "$(cat <<'EOF'
Log per-utterance audio health on the Sarvam STT path

One [sarvam-stt] INFO line per utterance: RMS during the silence
before it vs. during the speech itself, plus frame-delivery timing.
Diagnostic only, no behavior change - see
docs/superpowers/specs/2026-08-07-audio-noise-diagnostics-design.md
for why this needs to exist before any noise-suppression code does.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Thread `lead_id` through from the bridge

**Files:**
- Modify: `app/telephony/sarvam_bridge.py:824-825`
- Test: `tests/unit/test_sarvam_bridge.py`

**Interfaces:**
- Consumes: `sarvam_stt.SarvamSTT(lead_id: str = "")` (Task 2)

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_sarvam_bridge.py`, near the other `caplog`-based
turn-latency tests (after `test_a_turn_logs_where_the_lead_s_wait_actually_went`):

```python
async def test_the_lead_id_reaches_the_audio_health_log(two_way, caplog):
    """SarvamSTT logs its own [sarvam-stt] audio-health line (see
    app/telephony/sarvam_stt.py) — this only confirms the bridge actually
    threads the call's lead_id into it, so that line can be correlated
    against this call's other [sarvam] lines afterward."""
    caplog.set_level("INFO")
    two_way([_speech_ended()], replies=[])

    await _run(_FakePlivoWS(), one_way=False, lead_id="lead-audio-health")

    lines = [r.message for r in caplog.records if "audio health" in r.message]
    assert lines, "no audio-health line was logged"
    assert "lead=lead-audio-health" in lines[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_sarvam_bridge.py -k audio_health -v`
Expected: FAIL — the logged line contains `lead=` (empty), not
`lead=lead-audio-health`, because `SarvamSTT()` is still constructed with no
`lead_id`.

- [ ] **Step 3: Wire it up**

In `app/telephony/sarvam_bridge.py`, the `_run_two_way` function currently
has (lines 824-825):

```python
    async with sarvam_tts.SarvamTTS(language=SARVAM_STT_LANGUAGE) as tts, \
            sarvam_stt.SarvamSTT() as stt:
```

Change to:

```python
    async with sarvam_tts.SarvamTTS(language=SARVAM_STT_LANGUAGE) as tts, \
            sarvam_stt.SarvamSTT(lead_id=lead_id) as stt:
```

(`lead_id` is already an existing parameter of `_run_two_way` — no new
parameter needed here, just passing the one that's already in scope.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_sarvam_bridge.py -k audio_health -v`
Expected: PASS

- [ ] **Step 5: Run the full non-integration suite**

Run: `pytest -q -m "not integration"`
Expected: all tests PASS

- [ ] **Step 6: Lint**

Run: `ruff check app/telephony/sarvam_stt.py app/telephony/sarvam_bridge.py tests/unit/test_sarvam_stt.py tests/unit/test_sarvam_bridge.py`
Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add app/telephony/sarvam_bridge.py tests/unit/test_sarvam_bridge.py
git commit -m "$(cat <<'EOF'
Thread lead_id into SarvamSTT so audio-health logs correlate

Without this every [sarvam-stt] audio health line logged lead= empty,
making it useless for lining up against a specific call's other
[sarvam] transcript/turn-latency lines.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Deploy and collect real data

**Files:** none (operational task — rebuild/redeploy the existing Docker
setup, same commands already used earlier this session)

- [ ] **Step 1: Rebuild the backend image**

Run: `docker compose build backend`

- [ ] **Step 2: Recreate the running container**

Run: `docker compose up -d --force-recreate backend`

- [ ] **Step 3: Verify the new code is live**

Run:
```bash
sleep 3
docker exec voice-agent-elevenlabs-backend-1 sh -c "grep -c '_log_audio_health' /app_root/app/telephony/sarvam_stt.py"
curl -s -o /dev/null -w "health=%{http_code}\n" http://localhost:8091/health
```
Expected: a non-zero count and `health=200`.

- [ ] **Step 4: Place several real (or test) two-way Telugu/Tinglish calls**

This is manual — done by the owner, or by the assistant with the owner's
explicit go-ahead per CLAUDE.md's compliance rules on dialing. Include at
least one call from a location believed to be noisy and one from a quiet
location, if that distinction is available, to give the two RMS windows
something to actually differ on.

- [ ] **Step 5: Pull and read the logs**

Run:
```bash
docker logs voice-agent-elevenlabs-backend-1 --since 1h 2>&1 | grep -E "\[sarvam-stt\]|\[sarvam\]"
```

For each call, line up its `[sarvam-stt] lead=<id> audio health: ...` lines
against that same `lead_id`'s `[sarvam] lead=<id> ...` transcript lines by
timestamp. Read the result against the two hypotheses from the design doc:

- Elevated `silence_rms` next to a garbled transcript → genuine ambient
  noise. Proceed to scoping the actual suppression technique as a new,
  separate design (do not start that work from this plan).
- Normal-looking `silence_rms` (and `speech_rms`) next to a garbled
  transcript, but an irregular/high `frame_gap_max_ms` → connection/quality
  problem, not noise. This needs a different, likely harder investigation
  (the network path between Plivo and this service) — flag it rather than
  building a noise filter that would not help.
- Inconclusive from the first batch → place more calls before drawing a
  conclusion; this plan's job is done once the logging exists and is
  verified live, not once the underlying question is answered from one
  batch.

**Two confounds to read around**, found in the whole-branch review after this
plan's three tasks landed:

- **The FIRST utterance's `silence_rms` of every call is not clean.**
  `sarvam_bridge.py`'s `converse()` calls `deliver_opening()` — which blocks
  in real time until the AI-disclosure opening finishes playing — before it
  starts consuming `stt.events()`. So for the whole 15-30s of the opening, no
  VAD signal is processed and every inbound frame during that window lands
  in the silence accumulator. The first `[sarvam-stt]` line of every call
  will show an inflated `silence_rms` and a large `frames` count reflecting
  conditions DURING the agent's own opening speech (echo/leakage risk
  included), not genuine pre-speech quiet. Don't read an elevated
  first-utterance `silence_rms` as ambient noise — read the call's later
  utterances instead, or treat the first line as uninformative for the
  noise-vs-connection question.
- **`frame_gap_max_ms` can reflect event-loop scheduling delay, not only
  network jitter** — it's measured between successive `_track_frame` calls in
  a process shared with TTS playback, the reply task, and LLM HTTP
  round-trips. A healthy loop schedules in well under a millisecond; treat
  gaps in the tens-of-ms range as the meaningful signal for a connection
  problem, not sub-millisecond variation.

This task has no automated pass/fail — its deliverable is the data and the
owner's read of it, which determines whether there is a Task 5 at all and
what it should be.
