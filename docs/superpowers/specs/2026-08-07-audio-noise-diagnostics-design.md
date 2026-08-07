# Audio noise diagnostics — design

## Context

The owner asked for "noise cancellation" on the Sarvam two-way voice path
after observing garbled STT transcripts on real calls. Two independent test
calls placed earlier the same session back this up: `4a7f503d` produced
`"కో సీజన్ కా"` and `"ఎన్ఎం కోర్సెస్ ఎన్ఎం డిసొర్న్ మార్కెటింగ్లో"`, neither a
real Telugu sentence; `8ef724bb` produced `"స్పేస్ గురించి అప్పుడు డిస్టర్బ్
మాట్లాడడానికి స్పేస్ గురించి."`, and `871724cc` produced the truncated
`"డిస్టప్."` — the same "disturb"/"disturbance" fragment surfacing, garbled,
across two independent calls. That recurrence is a real signal, not noise in
the statistical sense.

Before building a fix, this needs a cause. Two different problems produce
identical-looking garbled transcripts: genuine ambient background noise on
the lead's end, or a connection/audio-quality problem (jitter, dropped or
corrupted mu-law frames) on the call leg. They need different fixes — noise
suppression helps the first and does nothing for the second. Research into
Sarvam's own API turned up no noise-suppression parameter on the
speech-to-text endpoint (streaming or REST), and Sarvam's own documentation
for `saaras:v3` — the exact model this project already uses
(`SARVAM_STT_MODEL`) — states the model is trained on real noisy Indian
telephony audio specifically so client-side pre-processing is unnecessary.
That raises the risk that a classic noise-gate/spectral-subtraction filter
either does nothing or actively hurts accuracy by introducing artifacts the
model wasn't trained on.

This project also has no call recording today (verified — no
`recording_url`/audio-file handling anywhere in `app/`), so there is no
after-the-fact audio to listen to. This design is the cheapest way to get
real evidence: log two lightweight signals per utterance, from data the
pipeline already has in memory, and use a batch of real campaign calls to
tell the two hypotheses apart before committing to a suppression technique.

## Non-goals

- No noise suppression / cancellation is implemented in this step. This is
  diagnostics only — the suppression approach (and whether it's even needed)
  is a follow-up decision made from the data this produces.
- No new dependency. Both signals are computed from the PCM16 samples
  `app/telephony/sarvam_stt.py` already decodes via `ulaw.decode()` — plain
  arithmetic over 80 samples/frame, no numpy.
- No new env var / toggle. The logging is cheap enough (a handful of
  arithmetic ops per 20ms frame, one INFO line per utterance) to always run,
  matching how `sarvam_bridge.py`'s existing turn-latency logging works.
- No behavior change to the call itself — this is additive logging on an
  already-open code path, nothing in the STT/TTS/turn-taking flow changes.
- No alerting, dashboard, or automated threshold. A human reads the logs
  after a batch of real calls.

## Design

**Where:** `app/telephony/sarvam_stt.py`, inside `SarvamSTT`. This module
already decodes every inbound frame in `send_audio()` and already parses
Sarvam's START_SPEECH/END_SPEECH VAD signals in `events()`, so both signals
below come from data it already owns — no new coupling to `sarvam_bridge.py`
or conversation state, keeping the module's documented boundary ("It knows
nothing about calls, turns or prompts") intact.

**Signal 1 — noise floor vs. speech-band energy.** A simple RMS
(root-mean-square) computed over each frame's decoded PCM16 samples:
`sqrt(sum(sample**2 for sample in frame) / len(frame))`. Accumulate two
running averages, reset at each VAD transition:
- `silence_rms` — frames arriving while NOT between a START_SPEECH and the
  matching END_SPEECH (i.e., presumed quiet).
- `speech_rms` — frames arriving WHILE between START_SPEECH and END_SPEECH.

`silence_rms` resets and starts accumulating at each END_SPEECH;
`speech_rms` resets and starts accumulating at each START_SPEECH — so the
value logged for a given utterance is always "since the boundary that
started this window," never a carry-over from a previous one.

An elevated `silence_rms` (the "quiet" parts of the call are not actually
quiet) points at genuine ambient background noise. A normal-looking
`silence_rms` next to a garbled transcript points elsewhere.

**Signal 2 — frame delivery health.** Track the wall-clock gap between
consecutive `send_audio()` calls. Plivo streams frames roughly every 20ms;
irregular gaps (jitter, drops) point at a connection/quality problem rather
than noise. Keep a running max and count for the current utterance window.

**Output.** One `[sarvam-stt]` INFO line per utterance, emitted from
`events()` at the point an `STTEvent("speech_ended")` is about to be
yielded (i.e., right when `on_lead_stopped` would be REACHED downstream, but
this module doesn't need to know that — it just logs on its own END_SPEECH
signal):

```
[sarvam-stt] lead=<id> audio health: silence_rms=<f> speech_rms=<f> frame_gap_max_ms=<f> frames=<n>
```

`lead_id` and wall-clock timestamp already correlate with the existing
`[sarvam]` transcript/turn-latency lines (`sarvam_bridge.py`'s
`_log_turn_latency`), so a human reviewing a batch of calls can line up "this
utterance came out garbled" against "here's what the audio actually looked
like" without needing recorded audio.

`SarvamSTT` currently has no `lead_id` — it is constructed with none in
`sarvam_bridge.py`'s `_run_two_way`. Threading one through (an optional
constructor argument, defaulting to an empty string the way other modules
default missing IDs) is the only interface change; everything else is
internal state added to `SarvamSTT.__init__`.

## Testing / verification

- Pure-function unit tests for the RMS computation given known sample
  sequences (silence-like low-amplitude samples vs. loud samples), matching
  `tests/unit/test_ulaw.py`'s existing style of exact-value assertions
  against hand-computed expectations — no Docker/network needed.
- A test driving `SarvamSTT.events()` through a fake `START_SPEECH` →
  frames → `END_SPEECH` sequence (same `_FakeSTTWS`-style harness already
  used in `tests/unit/test_sarvam_bridge.py`) asserting the log line fires
  once per utterance with the right `silence_rms`/`speech_rms` split.
- `pytest -q -m "not integration"` and `ruff check .` for the full suite.
- Manual: place several real campaign calls (or test calls, as done earlier
  this session) and read the `[sarvam-stt]` lines back against the
  transcript to actually answer the noise-vs-connection question. This is
  the point of the whole feature — code review alone cannot verify it.

## Risks

- A handful of calls may not be enough data to distinguish the two
  hypotheses conclusively; if the first batch is inconclusive, the answer is
  more calls, not a bigger diagnostic.
- RMS-based noise-floor detection is a coarse heuristic — a call with a
  brief loud interruption (a dog barking once) will look different from
  steady background noise (traffic hum) in ways this single number doesn't
  distinguish. If the first round of data is ambiguous, the next step is
  refining the signal (e.g., a noise-floor histogram instead of a mean),
  not jumping straight to a suppression implementation.
- If the data shows the frame-gap signal is elevated (connection quality),
  that is a materially different and likely harder follow-up problem
  (network path between Plivo and this service) than anything a noise
  suppression filter would address — worth flagging early rather than
  discovering after building the wrong fix.
