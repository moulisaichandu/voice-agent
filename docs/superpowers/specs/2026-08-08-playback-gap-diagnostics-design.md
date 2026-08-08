# Playback gap diagnostics — design

## Context

The owner reported two things heard on real two-way Sarvam calls tonight:
a "voice latency issue" and pronunciation that sounds off. Investigating
both without audio recording (this project has none) meant working from
what the code can actually prove, not guessing at a fix.

**Pronunciation** turned out to have a concrete, already-built answer:
`app/config.py`'s `SARVAM_TTS_SPEAKER` defaults to `"priya"`, Sarvam's
documented default for `te-IN` — but the comment right next to it says
this was never actually judged on a real 8 kHz call, only assumed.
`scripts/compare_telugu_voices.py` exists for exactly this and had never
been run. It has now been run — 8 voice samples are rendered and waiting
on the owner's ear, not on any code change. This design does not cover
that half.

**Latency** is murkier. `app/telephony/plivo_stream.py`'s `PlivoCall.play()`
sends audio to Plivo as fast as it arrives from Sarvam with no artificial
pacing — which is the correct pattern for this class of telephony
WebSocket API (Plivo paces actual playback on the phone line, not the
sender). That part is not the bug. But `app/telephony/sarvam_bridge.py`'s
`say()` synthesizes a response one sentence at a time, sequentially: each
sentence is a fresh `SarvamTTS.speak()` call, which is a new
text-then-flush round trip over Sarvam's WebSocket. If synthesizing a
short sentence takes longer than the previous sentence took to actually
play out on the phone line, Plivo's queue drains before the next chunk
arrives — dead air in the middle of the agent's own reply. That is a real,
plausible mechanism for "the agent's voice stutters," and `PlivoCall`
already tracks precisely the state needed to detect it (`play_end`,
`play_start`) — it is just never logged.

This design adds that logging, mirroring the per-utterance audio-health
signal already shipped tonight for the inbound (STT) side, this time for
the outbound (TTS/playback) side.

## Non-goals

- No fix to the pronunciation issue — that's an operator decision (which
  voice sample sounds right), not a code change, and is already unblocked.
- No change to how audio is sent to Plivo. `PlivoCall.play()`'s
  fire-as-it-arrives pattern is not touched — this only measures it.
- No pacing, buffering, or synthesis look-ahead added to close a gap if
  one is found. That's a follow-up decision made from this diagnostic's
  data, the same relationship tonight's audio-noise-diagnostics work has
  to an eventual suppression fix.
- No change to `app/telephony/plivo_stream.py`. Per the owner's explicit
  choice, this stays entirely inside `app/telephony/sarvam_bridge.py` —
  `PlivoCall` is shared by the ElevenLabs (production) and OpenAI Realtime
  backends too, and the inter-sentence-gap mechanism is specific to
  Sarvam's per-sentence synthesis loop; ElevenLabs streams from a single
  continuous platform connection with no equivalent seam. Reading
  `self.call.play_end`, a public field `PlivoCall` already exposes, is the
  only touch point.
- No new dependency, no new env var.

## Design

**Where:** `app/telephony/sarvam_bridge.py`'s `_Conversation.say()`, the
same method that already loops over sentences and calls `self.call.play()`
per frame.

**Signal:** immediately before each `call.play()` call *after the first
frame of the response has already played* (skipping the first frame is
deliberate — the delay before it is normal startup latency, already
captured by the existing turn-latency `tts=` metric; this signal is about
gaps *within* an already-started response, not time-to-first-word):

```
if spoke_a_frame:  # skip the response's very first frame — see above
    now = self._loop.time()
    if now > self.call.play_end:
        gap_ms = (now - self.call.play_end) * 1000
```

`spoke_a_frame` is the flag `say()` already maintains, currently flipped to
`True` right after the response's first frame is played. This check must
run *before* that flip and before `await self.call.play(payload)` — so on
the response's first-ever frame it is still `False` and the check is
skipped; on every frame after, it is `True` and the check runs.

`self.call.play_end` is the wall-clock time Plivo's queued audio will
finish playing, already maintained by `PlivoCall.play()` for every backend.
If `now` is past it, the queue had already drained — dead air the lead
heard between whatever played last and this new chunk. Accumulate two
things across the whole `say()` call: `gap_total_ms` (sum of every gap) and
`gap_max_ms` (the single largest).

**Output:** one `[sarvam]` INFO line per spoken turn (matching the existing
turn-latency line's prefix and per-turn scope, since both live in the same
class and describe the same turn), emitted once `say()` is done — success,
interrupted, or cancelled, via a `finally` block, but only if at least one
frame was actually played (nothing to report on a turn that never spoke):

```
[sarvam] lead=<id> playback gaps: total=<f:.1f>ms max=<f:.1f>ms sentences=<n>
```

`sentences` counts how many sentences this turn started synthesizing
(not necessarily finished, if interrupted) — context for reading
`total`/`max` against how much material the turn was.

A turn with zero gaps still logs (`total=0.0ms max=0.0ms`) — the point is
building a dataset across many real calls, the same reasoning tonight's
inbound diagnostic already used. A human reads a batch of these against
what the owner actually heard to confirm or rule out the inter-sentence
hypothesis, not an automated threshold.

## Testing / verification

- A test driving `say()` with a fake TTS whose `speak()` yields two frames
  with a real `asyncio.sleep()` between them long enough to guarantee
  `self.call.play_end` has passed before the second frame's `play()` call,
  asserting the logged line shows a nonzero `total`/`max`. Same
  `_FakePlivoWS`/real-`PlivoCall` harness `tests/unit/test_sarvam_bridge.py`
  already uses for barge-in tests, which exercise real timing rather than
  mocking it.
- A test driving `say()` with frames that arrive well within their
  predecessor's play-out window, asserting `total=0.0ms max=0.0ms` —
  proves the signal doesn't false-positive on ordinary fast delivery,
  which is the common case tonight's real calls already showed (TTS
  frequently outruns real time).
- A test confirming the line is NOT logged when `say()` returns before
  playing any frame (e.g., interrupted before the first frame, or empty
  text) — mirrors `_log_turn_latency`'s existing "nobody was waiting, no
  log" guard.
- `pytest -q -m "not integration"` and `ruff check .` for the full suite.
- Manual: place another real two-way Sarvam call and read the new
  `[sarvam] ... playback gaps: ...` lines back against what was actually
  heard, the same way tonight's audio-health lines were read against the
  transcripts already collected.

## Risks

- This measures Sarvam's own synthesis-round-trip time indirectly (a slow
  gap is time Sarvam spent generating the next sentence's audio, not
  something in this codebase's control) — if the data confirms the
  hypothesis, the fix is a follow-up design (e.g., pipelining the next
  sentence's synthesis while the current one is still playing), not
  something this diagnostic itself should attempt.
- `self._loop.time()` and `self.call.play_end` are both measured in the
  same process on the same event loop, so this is not sensitive to clock
  skew the way a cross-machine measurement would be — but it also means a
  busy event loop (this process doing other work — a concurrent call's LLM
  round trip, another call's TTS) could itself delay when the gap-check
  code even runs, inflating the measured gap beyond what Sarvam's own
  synthesis actually took. Worth remembering when reading results from a
  call that ran concurrently with others, same caveat as tonight's
  `frame_gap_max_ms` finding for the inbound side.
