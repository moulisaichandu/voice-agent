"""telephony/sarvam_stt.py — Sarvam's streaming speech-to-text, as a client.

Hearing the lead. Owns one thing: turning Plivo's inbound mu-law frames into
transcripts and speech-boundary signals. It knows nothing about calls, turns or
prompts, beyond an opaque lead_id used only to correlate its own diagnostic
log lines — app/telephony/sarvam_bridge.py decides what to do with what it
hears.

Wire protocol (docs.sarvam.ai, verified 2026-07-27):

    connect  wss://api.sarvam.ai/speech-to-text/ws
               ?model=saaras:v3&language-code=te-IN&sample_rate=8000
               &input_audio_codec=pcm_s16le&vad_signals=true
             header Api-Subscription-Key: <SARVAM_API_KEY>
    ->       {"audio": {"data": <b64 pcm16>, "encoding": "audio/pcm_s16le",
                        "sample_rate": "8000"}}
    <-       {"type": "data",   "data": {"transcript": "...", ...}}
    <-       {"type": "events", "data": {"signal_type": "START_SPEECH"|"END_SPEECH"}}
    <-       {"type": "error",  "data": {"message": "...", "code": "..."}}

TWO THINGS HERE ARE LOAD-BEARING.

1. THE LANGUAGE IS PINNED. On 8 kHz telephony audio, automatic language
   detection was observed guessing Croatian and Urdu for Telugu speech. The
   lead is then transcribed as nonsense, and the agent answers the nonsense
   fluently and confidently. That failure was paid for on real calls by the
   sibling ../ai-voice-agent and is why SARVAM_STT_LANGUAGE exists.

2. THE AUDIO IS DECODED, NOT FORWARDED. Sarvam's STT accepts only
   wav / pcm_s16le / pcm_l16 / pcm_raw — mu-law is not in the list. This is the
   one leg in the whole project that transcodes, and app/telephony/ulaw.py
   exists for it. Forwarding Plivo's bytes untouched would not raise anywhere:
   Sarvam would simply transcribe noise as nothing, on every call.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlencode

import websockets

from app.config import (
    SARVAM_API_KEY,
    SARVAM_STT_LANGUAGE,
    SARVAM_STT_MODEL,
    SARVAM_VAD_HIGH_SENSITIVITY,
)
from app.telephony import sarvam_circuit_breaker, ulaw

logger = logging.getLogger(__name__)

_STT_URL = "wss://api.sarvam.ai/speech-to-text/ws"

EventKind = Literal["transcript", "speech_started", "speech_ended"]


@dataclass(frozen=True)
class STTEvent:
    """Something the lead did. `text` is meaningful only for 'transcript'."""

    kind: EventKind
    text: str = ""


class SarvamNotConfigured(RuntimeError):
    """No SARVAM_API_KEY. Raised before connecting."""


async def _connect(url: str, headers: dict[str, str]) -> Any:
    """Open the socket, tolerating both `websockets` header kwargs — the same
    shim the other two Sarvam modules carry."""
    try:
        return await websockets.connect(url, additional_headers=headers, max_size=None)
    except TypeError:
        return await websockets.connect(url, extra_headers=headers, max_size=None)


def _stt_url() -> str:
    params = {
        "model": SARVAM_STT_MODEL,
        # See this module's docstring, point 1. Never leave this to detection.
        "language-code": SARVAM_STT_LANGUAGE,
        "sample_rate": "8000",
        "input_audio_codec": "pcm_s16le",
        # START_SPEECH is the ONLY barge-in signal this backend gets. Without
        # it the lead physically cannot interrupt the agent.
        "vad_signals": "true",
        "high_vad_sensitivity": "true" if SARVAM_VAD_HIGH_SENSITIVITY else "false",
    }
    # safe=":" keeps model names like 'saaras:v3' literal. Percent-encoding the
    # colon is legal and any conformant server decodes it, but a colon is also
    # legal unencoded in a query string, and matching the form Sarvam's own
    # documentation shows is the cheaper bet against an API we cannot inspect.
    return f"{_STT_URL}?{urlencode(params, safe=':')}"


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


class SarvamSTT:
    """One STT socket for the life of a call.

    Async context manager: opened on entry, closed on exit, so no error path
    can leak one.
    """

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

    async def __aenter__(self) -> SarvamSTT:
        if not SARVAM_API_KEY:
            raise SarvamNotConfigured(
                "SARVAM_API_KEY is not set — the lead cannot be heard."
            )
        self._ws = await _connect(_stt_url(),
                                  {"Api-Subscription-Key": SARVAM_API_KEY})
        return self

    async def __aexit__(self, *exc_info) -> bool:
        if self._frame_count:
            # The call ended without a clean final END_SPEECH — a hangup
            # mid-utterance, or no VAD boundary ever fired. Flush whatever
            # was accumulated rather than discarding it silently: these
            # pathological calls are exactly the ones most worth measuring.
            # A call that never tracked a single frame has nothing to say.
            self._log_audio_health()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001 - closing must not mask the real error
                logger.debug("[sarvam-stt] socket close failed", exc_info=True)
            self._ws = None
        return False

    async def send_audio(self, mulaw_b64: str) -> None:
        """Forward one Plivo frame, decoded to PCM16.

        A frame that will not decode is DROPPED, not raised on: one corrupt
        20 ms frame should cost the lead a flicker of audio, not the call.
        """
        try:
            pcm = ulaw.decode(base64.b64decode(mulaw_b64))
        except (binascii.Error, ValueError):
            logger.warning("[sarvam-stt] undecodable inbound frame — dropped")
            return

        self._track_frame(pcm)

        await self._ws.send(json.dumps({
            "audio": {
                "data": base64.b64encode(pcm).decode(),
                # 'audio/wav', NOT 'audio/pcm_s16le'. These are two different
                # fields with two different enums: the CONNECTION's
                # input_audio_codec query param (above) takes pcm_s16le and
                # declares what the bytes are; this per-message field takes
                # only audio/wav. Sending pcm_s16le here is rejected outright —
                # "audio.encoding: Input should be 'audio/wav'" — and a
                # rejected stream means the agent hears nothing at all for the
                # whole call.
                #
                # The bytes stay raw PCM16; no WAV container is added or
                # wanted. Verified live by transcribing real Telugu through
                # exactly this frame.
                "encoding": "audio/wav",
                "sample_rate": "8000",
            },
        }))

    def _track_frame(self, pcm: bytes) -> None:
        """Accumulate one successfully decoded inbound frame into whichever
        RMS window is currently active, and update frame-delivery timing.

        A frame that failed to decode never reaches here (see send_audio's
        early return), so `_last_frame_at` isn't advanced for it either — the
        gap is simply absorbed into whichever successfully-tracked frame
        arrives next. That's desirable, not a bug: a dropped frame IS a
        delivery problem, and widening the next gap is how it shows up here.
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
        # _speech_sumsq/_speech_count are ALSO reset here, not only at
        # START_SPEECH: if END_SPEECH ever fires twice with no intervening
        # START_SPEECH (a real reachable shape — see
        # test_sarvam_bridge.py's test_the_lead_id_reaches_the_audio_health_log,
        # which drives exactly a bare END_SPEECH with no prior START_SPEECH),
        # the second line must honestly report "nothing measured"
        # (speech_rms=0.0, per _rms's own convention) rather than carry over
        # whatever speech was last measured.
        self._speech_sumsq = 0
        self._speech_count = 0
        self._frame_gap_max_ms = 0.0
        self._frame_count = 0

    async def events(self) -> AsyncIterator[STTEvent]:
        """Yield what the lead does until the socket ends or errors.

        An error frame ends the stream rather than being swallowed: a rejected
        connection otherwise looks exactly like a lead who never speaks, and
        costs a full CALL_MAX_DURATION_S of billed silence to discover.
        """
        async for raw in self._ws:
            try:
                event = json.loads(raw)
            except (ValueError, TypeError):
                continue
            etype = event.get("type")
            data = event.get("data") or {}

            if etype == "data":
                text = (data.get("transcript") or "").strip()
                if text:
                    # Sarvam emits empty transcripts for non-speech. Treating
                    # one as a turn would have the agent answer a cough.
                    yield STTEvent("transcript", text)
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
            elif etype == "error":
                # 'message', not 'error'. Sarvam populates the former, so the
                # old key always missed and this fell through to dumping the
                # raw event dict. That was not merely untidy: the credits
                # check below reads this same value, so with the wrong key it
                # could never have fired whatever Sarvam sent.
                message = data.get("message") or event
                logger.error(
                    f"[sarvam-stt] refused to transcribe: "
                    f"{message} (code={data.get('code')})"
                )
                # isinstance guard: message falls back to `event`, a dict, when
                # the frame carries no 'message' — .lower() on that would raise
                # inside the error handler and turn a reported failure into an
                # unreported one.
                if isinstance(message, str) and "insufficient credit" in message.lower():
                    await sarvam_circuit_breaker.trip(message)
                return
