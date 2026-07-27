#!/usr/bin/env python
"""
smoke_sarvam_oneway.py — Run a one-way Telugu call's whole pipeline, minus the phone.

This is everything app/telephony/sarvam_bridge.py does on a real one-way call,
in the same order, through the same modules:

    campaign script (English)
      -> sarvam_llm.render()          the Telugu, cached, disclosure-repaired
      -> sarvam_prompts.compose_spoken()   greeting + body, as the lead hears it
      -> sarvam_tts.speak()           mu-law 8 kHz frames
      -> a .wav                       instead of PlivoCall.play()

Only the Plivo leg is missing, and that leg is already proven by the two
backends shipping today — it is the same PlivoCall, the same mu-law framing.

So this settles, for the price of one API call and no phone call:

  * whether SARVAM_API_KEY works at all;
  * whether Sarvam's TTS really accepts output_audio_codec=mulaw at 8000 on the
    WebSocket path. Its own reference enumerates mulaw and alaw, while one line
    of the same page claims MP3 only. If it is MP3, the lead hears static and
    NOTHING logs an error — so this is the single most important unknown in the
    backend, and the file this writes is the answer;
  * whether sarvam-105b actually renders usable spoken Telugu from English;
  * whether the AI disclosure survives that rendering (and, if it does not,
    that the repair in sarvam_llm fires and says so);
  * time-to-first-audio, cold cache and warm — the number Milestone A's gate
    asks for, and the one that decides whether two-way is worth building on
    this foundation.

Usage:
    python scripts/smoke_sarvam_oneway.py
    python scripts/smoke_sarvam_oneway.py --script "This is an AI call. Fees dropped."
    python scripts/smoke_sarvam_oneway.py --name Asha --speaker kavitha
    python scripts/smoke_sarvam_oneway.py --style tinglish
"""

import argparse
import asyncio
import base64
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import languages  # noqa: E402
from app.compliance.disclosure import has_ai_disclosure  # noqa: E402
from app.config import (  # noqa: E402
    SARVAM_API_KEY,
    SARVAM_LLM_MODEL,
    SARVAM_STT_LANGUAGE,
    SARVAM_TTS_MODEL,
    SARVAM_TTS_SPEAKER,
)
from app.telephony import sarvam_llm, sarvam_prompts, sarvam_tts  # noqa: E402
from scripts.compare_telugu_voices import mulaw_wav  # noqa: E402

# Written in English on purpose: the operator workflow this backend has to
# support is "write English, the lead hears Telugu". A script already in Telugu
# would skip the interesting half of the test.
_DEFAULT_SCRIPT = (
    "This is an automated AI call from Digital Brolly. "
    "Our new digital marketing course starts on Monday. "
    "Classes are online in the evening, and there is a free demo class first."
)


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a one-way Telugu call's pipeline without placing a call."
    )
    parser.add_argument("--script", default=_DEFAULT_SCRIPT,
                        help="the campaign script, as an operator would write it")
    parser.add_argument("--name", default=None,
                        help="lead name to greet (default: no greeting)")
    parser.add_argument("--speaker", default=None,
                        help=f"Sarvam voice (default: {SARVAM_TTS_SPEAKER})")
    parser.add_argument("--style", choices=("te", "tinglish"), default="te",
                        help="language register (default: te)")
    parser.add_argument("--out", type=Path, default=Path("voice-samples"))
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass

    if not SARVAM_API_KEY:
        print("SARVAM_API_KEY is not set. Add it to .env and re-run.", file=sys.stderr)
        return 1

    style = languages.style(args.style)
    speaker = args.speaker or SARVAM_TTS_SPEAKER

    _rule("configuration")
    print(f"  llm       {SARVAM_LLM_MODEL}")
    print(f"  tts       {SARVAM_TTS_MODEL} / {speaker} / {SARVAM_STT_LANGUAGE}")
    print(f"  register  {args.style}")
    print(f"  script    {args.script}")

    # ── step 1: English -> Telugu, exactly as the bridge does it ──────────────
    _rule("1. render (sarvam_llm.render)")
    started = time.perf_counter()
    try:
        body = await sarvam_llm.render(args.script, language_style=style)
    except (sarvam_llm.SarvamNotConfigured, sarvam_llm.SarvamRenderFailed) as exc:
        print(f"  FAILED: {exc}", file=sys.stderr)
        print("\n  A real call would refuse to dial here rather than read the "
              "English script at a Telugu speaker.", file=sys.stderr)
        return 1
    cold_s = time.perf_counter() - started

    warm_started = time.perf_counter()
    await sarvam_llm.render(args.script, language_style=style)
    warm_s = time.perf_counter() - warm_started

    print(f"  cold {cold_s:5.2f}s   warm {warm_s:5.2f}s (Redis cache)")
    if warm_s > cold_s * 0.5:
        print("  NOTE: the warm render was not much faster — Redis is probably "
              "unreachable, so every call would pay the cold cost in silence.")
    print(f"\n  {body}")

    # ── step 2: what the lead actually hears ─────────────────────────────────
    _rule("2. spoken text (sarvam_prompts.compose_spoken)")
    spoken = sarvam_prompts.compose_spoken(body, lead_name=args.name)
    print(f"  {spoken}")

    ok = has_ai_disclosure(spoken)
    print(f"\n  AI disclosure in the opening: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  This is a stop-dialling result. The repair in sarvam_llm "
              "should have caught it — check the [compliance] warning above.")

    # ── step 3: speech, at real phone quality ────────────────────────────────
    _rule("3. synthesise (sarvam_tts, mulaw 8 kHz)")
    chunks: list[tuple[str, int]] = []
    first_audio_s = None
    started = time.perf_counter()
    try:
        async with sarvam_tts.SarvamTTS(
            language=SARVAM_STT_LANGUAGE, speaker=speaker,
        ) as tts:
            async for payload, ms in tts.speak(spoken):
                if first_audio_s is None:
                    first_audio_s = time.perf_counter() - started
                chunks.append((payload, ms))
    except Exception as exc:  # noqa: BLE001 - this is the diagnostic
        print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if not chunks:
        print("  FAILED: no audio. Either the voice does not exist for te-IN, "
              "or mulaw/8000 was rejected — see the [sarvam-tts] error above.",
              file=sys.stderr)
        return 1

    audio = b"".join(base64.b64decode(c) for c, _ in chunks)
    duration_s = sum(ms for _, ms in chunks) / 1000
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"oneway-{speaker}-{args.style}.wav"
    path.write_bytes(mulaw_wav(audio))

    print(f"  time to first audio  {first_audio_s:.2f}s")
    print(f"  audio duration       {duration_s:.1f}s ({len(chunks)} chunks, "
          f"{len(audio)} bytes)")
    print(f"  written              {path}")

    # ── the verdict ──────────────────────────────────────────────────────────
    _rule("what this proves")
    opening_s = (cold_s if warm_s > cold_s * 0.5 else warm_s) + (first_audio_s or 0)
    print("  The key works, and Sarvam did not reject mulaw/8000 on the WebSocket")
    print("  path. That is not yet proof the bytes ARE mu-law — a server that")
    print("  ignored the codec would also send audio. PLAY THE FILE: it is wrapped")
    print("  as mu-law without re-encoding, so if it sounds like speech rather than")
    print("  static, the format question is genuinely settled.")
    print(f"\n  A lead would hear the first word about {opening_s:.1f}s after "
          "answering.\n  Under ~1.5s is fine; much more and they hear silence "
          "and may hang up.")
    print("\n  NOT proven here: the Plivo leg, and whether the Telugu is any GOOD."
          "\n  The second is a human judgement and the whole reason for this "
          "backend —\n  play the file before dialling anyone.")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
