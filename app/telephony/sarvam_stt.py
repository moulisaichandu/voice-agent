"""telephony/sarvam_stt.py — Sarvam's streaming speech-to-text, as a client.

Hearing the lead. Owns one thing: turning Plivo's inbound mu-law frames into
transcripts and speech-boundary signals. It knows nothing about calls, turns or
prompts — app/telephony/sarvam_bridge.py decides what to do with what it hears.

Wire protocol (docs.sarvam.ai, verified 2026-07-27):

    connect  wss://api.sarvam.ai/speech-to-text/ws
               ?model=saaras:v3&language-code=te-IN&sample_rate=8000
               &input_audio_codec=pcm_s16le&vad_signals=true
             header Api-Subscription-Key: <SARVAM_API_KEY>
    ->       {"audio": {"data": <b64 pcm16>, "encoding": "audio/pcm_s16le",
                        "sample_rate": "8000"}}
    <-       {"type": "data",   "data": {"transcript": "...", ...}}
    <-       {"type": "events", "data": {"signal_type": "START_SPEECH"|"END_SPEECH"}}
    <-       {"type": "error",  "data": {"error": "...", "code": "..."}}

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
from app.telephony import ulaw

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

    def __init__(self) -> None:
        self._ws: Any = None

    async def __aenter__(self) -> SarvamSTT:
        if not SARVAM_API_KEY:
            raise SarvamNotConfigured(
                "SARVAM_API_KEY is not set — the lead cannot be heard."
            )
        self._ws = await _connect(_stt_url(),
                                  {"Api-Subscription-Key": SARVAM_API_KEY})
        return self

    async def __aexit__(self, *exc_info) -> bool:
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
                    yield STTEvent("speech_started")
                elif signal == "END_SPEECH":
                    yield STTEvent("speech_ended")
            elif etype == "error":
                logger.error(
                    f"[sarvam-stt] refused to transcribe: "
                    f"{data.get('error') or event} (code={data.get('code')})"
                )
                return
