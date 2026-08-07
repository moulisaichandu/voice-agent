"""telephony/sarvam_tts.py — Sarvam's streaming text-to-speech, as a client.

The Telugu voice. Owns exactly one thing: turning text into base64 mu-law
frames a PlivoCall can play. It knows nothing about calls, leads, prompts or
compliance, which is what keeps app/telephony/sarvam_bridge.py readable.

WHY THIS IS A SEPARATE SERVICE AT ALL. ElevenLabs Agents and OpenAI Realtime
are speech-to-speech: one socket, and the platform does turn-taking. Sarvam has
no such API — STT, LLM and TTS are three independent services. That is the
central difference of this backend, and the reason two-way Telugu becomes
possible here: the turn-taking is ours to write.

Wire protocol (docs.sarvam.ai, verified 2026-07-27):

    connect  wss://api.sarvam.ai/text-to-speech/ws?model=<model>
             header Api-Subscription-Key: <SARVAM_API_KEY>
    ->       {"type": "config", "data": {...}}
    ->       {"type": "text",   "data": {"text": "..."}}
    ->       {"type": "flush"}
    <-       {"type": "audio",  "data": {"audio": <base64>, "request_id": ...}}
    <-       {"type": "event",  "data": {"event_type": "final"}}
    <-       {"type": "error",  "data": {"message": ..., "code": ...}}

THE FORMAT IS THE WHOLE JOB. Sarvam defaults to 24 kHz MP3. Plivo is told
every payload it receives is mu-law 8 kHz (see plivo_stream.PlivoCall.play), so
any other format is misframed on arrival and the lead hears static — with
nothing logged anywhere, because both sides believe they did their job. The
config frame below pins `mulaw` / `8000`, which also keeps the outbound leg a
pure base64 passthrough with no transcoding, exactly as the other two backends
have.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import websockets

from app.config import (
    SARVAM_API_KEY,
    SARVAM_TTS_MODEL,
    SARVAM_TTS_PACE,
    SARVAM_TTS_SPEAKER,
)

logger = logging.getLogger(__name__)

# send_completion_event=true is NOT optional, despite reading like a nicety.
# Verified against the live API on 2026-07-27: without it Sarvam streams the
# audio and then sends nothing at all — no terminating frame — and the socket
# sits open until the server kills it as idle with a 408, 25s+ later. With it,
# {"event_type": "final"} arrives about 0.6s after the last audio chunk.
#
# speak() ends its stream on that event, so omitting this makes every utterance
# block for the server's idle timeout: survivable on a one-way call, where the
# audio has already reached the lead, and fatal on a two-way one, where every
# sentence of every turn would stall.
_TTS_URL = ("wss://api.sarvam.ai/text-to-speech/ws"
            "?model={model}&send_completion_event=true")

# Mu-law at 8 kHz is one byte per sample, so a byte is 1/8 of a millisecond.
# Milestone B's barge-in ledger uses this to work out how much of a sentence
# the lead actually heard before interrupting.
_BYTES_PER_MS = 8.0


class SarvamNotConfigured(RuntimeError):
    """No SARVAM_API_KEY. Raised before connecting, so the bridge can turn it
    into a failed outcome instead of discovering it mid-call."""


async def _connect(url: str, headers: dict[str, str]) -> Any:
    """Open the socket, tolerating both `websockets` header kwargs.

    websockets >= 14 renamed `extra_headers` to `additional_headers`. The same
    shim app/telephony/openai_bridge.py carries, for the same reason: the
    pinned version is 15.x but this must not break on an older environment.
    """
    try:
        return await websockets.connect(url, additional_headers=headers, max_size=None)
    except TypeError:
        return await websockets.connect(url, extra_headers=headers, max_size=None)


def _config_frame(language: str, speaker: str) -> dict:
    """The frame that decides whether the lead hears Telugu or static."""
    return {
        "type": "config",
        "data": {
            "model": SARVAM_TTS_MODEL,
            "target_language_code": language,
            "speaker": speaker,
            # Both pinned for the phone path — see this module's docstring.
            "output_audio_codec": "mulaw",
            "speech_sample_rate": "8000",
            "pace": SARVAM_TTS_PACE,
        },
    }


class SarvamTTS:
    """One TTS socket, reusable across the turns of a single call.

    Async context manager: the socket is opened and configured on entry and
    closed on exit, so a caller cannot leave one open on an error path.
    """

    def __init__(self, *, language: str, speaker: str | None = None):
        """*speaker* defaults to SARVAM_TTS_SPEAKER, which is how every real
        call gets its voice — so the voice can be changed in .env without a
        deploy. The override exists for tools that need to hear several voices
        in one process (scripts/compare_telugu_voices.py), which would
        otherwise have to mutate module state and stop exercising the same
        path a live call takes."""
        self._language = language
        self._speaker = speaker or SARVAM_TTS_SPEAKER
        self._ws: Any = None

    async def __aenter__(self) -> SarvamTTS:
        await self._open()
        return self

    async def _open(self) -> None:
        """Open and configure the socket used by the next synthesis."""
        if not SARVAM_API_KEY:
            raise SarvamNotConfigured(
                "SARVAM_API_KEY is not set — Telugu calls cannot be voiced."
            )
        url = _TTS_URL.format(model=SARVAM_TTS_MODEL)
        self._ws = await _connect(url, {"Api-Subscription-Key": SARVAM_API_KEY})
        await self._ws.send(json.dumps(
            _config_frame(self._language, self._speaker)))

    async def _reset_connection(self) -> None:
        """Discard a socket whose utterance was cancelled mid-stream.

        Sarvam keeps one WebSocket open across utterances. If a barge-in
        cancels ``speak()`` while the server is still streaming audio, the
        unread audio/final frames remain on that socket. Reusing it lets the
        next turn consume stale frames (often the old ``final`` immediately),
        producing no audio for the answer the lead is waiting for. A fresh
        configured socket is cheap compared with another silent turn.
        """
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001 - cleanup must not mask cancellation
                logger.debug("[sarvam-tts] cancelled socket close failed",
                             exc_info=True)

    async def reset_after_cancel(self) -> None:
        """Public cancellation hook for a consumer stopped during a chunk.

        Cancelling the caller while it is inside ``call.play`` can happen
        after ``speak()`` yielded a frame, before the async generator receives
        the cancellation itself. The bridge calls this hook so that case also
        cannot leave the reusable socket mid-utterance.
        """
        await self._reset_connection()

    async def __aexit__(self, *exc_info) -> bool:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001 - closing must never mask the real error
                logger.debug("[sarvam-tts] socket close failed", exc_info=True)
            self._ws = None
        return False

    async def speak(self, text: str) -> AsyncIterator[tuple[str, int]]:
        """Synthesise *text*, yielding `(base64_mulaw, duration_ms)` per chunk.

        Ends on Sarvam's `final` event. Honouring it is not optional: Sarvam
        keeps the socket open for the next utterance rather than closing it, so
        a caller that waits for the socket to end instead would hang until
        CALL_MAX_DURATION_S and bill the silence.

        An error frame also ends the stream. A rejected config — an expired key,
        a speaker that does not exist for this language — otherwise produces
        exactly the same symptom as a hang, and a call that ends with a logged
        reason beats one that ends with a bill.
        """
        cleaned = (text or "").strip()
        if not cleaned:
            # No text means no turn. Sending an empty frame and then awaiting
            # audio Sarvam has no reason to produce is a hang, not a no-op.
            return

        if self._ws is None:
            # This is the normal path only after a cancelled prior utterance;
            # __aenter__ opens the first socket for a call.
            await self._open()
        try:
            await self._ws.send(json.dumps({"type": "text", "data": {"text": cleaned}}))
            await self._ws.send(json.dumps({"type": "flush"}))

            async for raw in self._ws:
                try:
                    event = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                etype = event.get("type")
                data = event.get("data") or {}

                if etype == "audio":
                    payload = data.get("audio")
                    if payload:
                        yield payload, _duration_ms(payload)
                elif etype == "event" and data.get("event_type") == "final":
                    return
                elif etype == "error":
                    logger.error(
                        f"[sarvam-tts] refused to synthesise: "
                        f"{data.get('message') or event} (code={data.get('code')})"
                    )
                    return
        except asyncio.CancelledError:
            await self._reset_connection()
            raise

    async def collect(self, text: str) -> list[tuple[str, int]]:
        """speak() drained into a list. For one-way calls, which have a single
        utterance and no barge-in to react to, and for tests."""
        return [chunk async for chunk in self.speak(text)]


def _duration_ms(payload_b64: str) -> int:
    """How many milliseconds of audio *payload_b64* is.

    Decodes only to measure the length; the payload is handed to Plivo
    untouched. Undecodable base64 counts as zero rather than raising — a
    corrupt chunk should cost the lead a gap in the audio, not the whole call.
    """
    try:
        return int(len(base64.b64decode(payload_b64)) / _BYTES_PER_MS)
    except (binascii.Error, ValueError):
        logger.warning("[sarvam-tts] undecodable audio chunk — counting it as 0 ms")
        return 0
