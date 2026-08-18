#!/usr/bin/env python
"""
compare_telugu_voices.py — Hear the same Telugu line in several Sarvam voices,
at real phone quality, before dialling anyone.

WHY THIS EXISTS. Choosing the Telugu voice is the single biggest product
decision in this backend and the one thing no amount of code review settles.
Sarvam ships ~39 voices; their names say nothing about how any of them sounds
reading Telugu, and a voice that is pleasant at 24 kHz in a browser demo can be
mushy or shrill once it has been squeezed through an 8 kHz mu-law phone codec.

So this synthesises through app/telephony/sarvam_tts.py — the SAME client the
live call uses, pinned to the same mulaw/8000 output — and writes one .wav per
voice. What you hear is byte-for-byte what a lead hears, not an approximation.

The default line is a realistic call opening: the AI disclosure first (as India
telecom rules require and app/compliance/disclosure.py enforces), then the
company, then the reason for the call. Judge the voice on the words it will
actually say, and listen for the disclosure specifically — it is the sentence
every lead hears and the one that must be unambiguous.

A voice that produces no audio almost always means Sarvam does not offer it for
te-IN; the run reports that per voice and carries on rather than stopping.

Usage:
    python scripts/compare_telugu_voices.py
    python scripts/compare_telugu_voices.py --speakers priya,kavitha,aditya
    python scripts/compare_telugu_voices.py --text "మీ కోర్సు రేపు మొదలవుతుంది."
    python scripts/compare_telugu_voices.py --out C:/tmp/voices

Then play the files, pick a winner, and put it in .env:
    SARVAM_TTS_SPEAKER=<name>
"""

import argparse
import asyncio
import base64
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SARVAM_API_KEY, SARVAM_STT_LANGUAGE, SARVAM_TTS_MODEL  # noqa: E402
from app.telephony import sarvam_tts  # noqa: E402

# A realistic opening, not a test phrase: disclosure, caller, reason. Contains
# "కృత్రిమ మేధ", one of the markers app/compliance/disclosure.py recognises, so
# what you are judging is the line that actually has to work.
_DEFAULT_TEXT = (
    "ఇది కృత్రిమ మేధ ద్వారా చేసే ఆటోమేటెడ్ కాల్. "
    "నేను డిజిటల్ బ్రోలీ నుండి మాట్లాడుతున్నాను. "
    "మా కొత్త డిజిటల్ మార్కెటింగ్ కోర్సు సోమవారం మొదలవుతుంది."
)

# A starting shortlist, not a recommendation — I have no way to judge an
# Indian-language voice from a name. 'priya' is Sarvam's documented default for
# te-IN and is listed first so the current config value is in the comparison.
# Pass --speakers to try any others from bulbul:v3's ~39.
_DEFAULT_SPEAKERS = (
    "priya", "kavitha", "shruti", "roopa",      # female
    "aditya", "vijay", "gokul", "shubh",        # male
)

_SAMPLE_RATE = 8000
_WAVE_FORMAT_MULAW = 7


def mulaw_wav(payload: bytes) -> bytes:
    """Wrap raw mu-law samples in a WAV container, WITHOUT re-encoding them.

    Written by hand because the stdlib `wave` module only writes PCM, and
    converting to PCM would defeat the point: the whole exercise is judging how
    a voice survives the mu-law telephony codec, so the bytes in the file must
    be the bytes that went down the wire.

    Format 7 (mu-law) needs an 18-byte fmt chunk (a cbSize field PCM omits) and
    a `fact` chunk giving the sample count — some players reject a non-PCM WAV
    without it.
    """
    n = len(payload)
    fmt = struct.pack(
        "<HHIIHHH",
        _WAVE_FORMAT_MULAW, 1, _SAMPLE_RATE,
        _SAMPLE_RATE,  # byte rate: mu-law is 1 byte per sample
        1,             # block align
        8,             # bits per sample
        0,             # cbSize
    )
    chunks = (b"fmt " + struct.pack("<I", len(fmt)) + fmt
              + b"fact" + struct.pack("<II", 4, n)
              + b"data" + struct.pack("<I", n) + payload
              + (b"\x00" if n % 2 else b""))
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


async def _render_one(speaker: str, text: str,
                      out_dir: Path) -> tuple[str, bool, str]:
    """Synthesise *text* as *speaker*. Returns (speaker, wrote_a_file, message).

    Never raises: one unavailable voice must not end the comparison. A voice
    Sarvam does not offer for te-IN comes back as an error frame, which
    sarvam_tts logs and turns into zero chunks — reported here as "no audio".
    """
    try:
        async with sarvam_tts.SarvamTTS(
            language=SARVAM_STT_LANGUAGE, speaker=speaker,
        ) as tts:
            chunks = await tts.collect(text)
    except Exception as exc:  # noqa: BLE001 - report and keep going
        return speaker, False, f"FAILED  {type(exc).__name__}: {exc}"

    if not chunks:
        return speaker, False, ("no audio — Sarvam probably has no te-IN voice "
                                "by this name")

    payload = b"".join(base64.b64decode(c) for c, _ in chunks)
    duration_ms = sum(ms for _, ms in chunks)
    path = out_dir / f"{speaker}.wav"
    path.write_bytes(mulaw_wav(payload))
    return speaker, True, f"{duration_ms / 1000:5.1f}s  ->  {path}"


def _make_stdout_speak_telugu() -> None:
    """Stop a Windows console killing the run on the first Telugu character.

    A default Windows terminal is cp1252, which cannot encode Telugu at all, so
    printing the line being synthesised raises UnicodeEncodeError and takes the
    whole comparison down before a single voice is rendered. Switching stdout
    to UTF-8 fixes it where the terminal can cope and, with errors='replace',
    degrades to '?' rather than crashing where it cannot.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass  # already UTF-8, or redirected somewhere that cannot be reconfigured


async def main() -> int:
    _make_stdout_speak_telugu()
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0].strip())
    parser.add_argument("--text", default=_DEFAULT_TEXT,
                        help="Telugu line to synthesise (default: a realistic "
                             "call opening including the AI disclosure)")
    parser.add_argument("--speakers", default=",".join(_DEFAULT_SPEAKERS),
                        help="comma-separated Sarvam speaker names")
    parser.add_argument("--out", type=Path, default=Path("voice-samples"),
                        help="directory to write the .wav files into")
    args = parser.parse_args()

    if not SARVAM_API_KEY:
        print("SARVAM_API_KEY is not set. Add it to .env and re-run.",
              file=sys.stderr)
        return 1

    speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]
    if not speakers:
        print("No speakers to compare.", file=sys.stderr)
        return 1

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"model    {SARVAM_TTS_MODEL}")
    print(f"language {SARVAM_STT_LANGUAGE}")
    print("format   mu-law 8 kHz — exactly what a lead hears on the phone")
    print(f"text     {args.text}")
    print(f"output   {out_dir}\n")

    # Sequential on purpose: eight concurrent sockets against a key you have
    # just created is a good way to meet a rate limit and misread it as the
    # voices being unavailable.
    results = [await _render_one(s, args.text, out_dir) for s in speakers]

    width = max(len(s) for s in speakers)
    for speaker, _wrote, message in results:
        print(f"  {speaker:<{width}}  {message}")

    ok = [speaker for speaker, wrote, _ in results if wrote]
    print(f"\n{len(ok)}/{len(speakers)} voices rendered.")
    if ok:
        print("Play them, pick one, then set it in .env:")
        print("    SARVAM_TTS_SPEAKER=<name>")
        print("\nListen for: is the DISCLOSURE sentence unambiguous at 8 kHz? "
              "Does the Telugu sound like a person from Hyderabad on a phone, "
              "or like a reader? Are the English words (course, batch) natural?")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
