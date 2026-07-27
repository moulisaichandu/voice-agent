"""telephony/ulaw.py — G.711 mu-law <-> 16-bit PCM, in pure Python.

Exists because of one asymmetry in the Sarvam backend. Its TTS emits mu-law at
8 kHz, so the OUTBOUND leg stays the pure passthrough every backend in this
project has. Its STT does not accept mu-law at all — `input_audio_codec` takes
only wav / pcm_s16le / pcm_l16 / pcm_raw — so the INBOUND leg has to be
decoded. That makes this the one place the project transcodes call audio, and
CLAUDE.md says so explicitly rather than letting the claim quietly rot.

NO `audioop`. The stdlib had `audioop.ulaw2lin` for exactly this, but it was
deprecated in 3.12 and REMOVED in 3.13 — and this venv is already running
3.13.7. An audio path that stops existing on a Python upgrade is not a
dependency worth having for forty lines of arithmetic.

Cost is negligible: two 256-entry tables built once at import, then a dict-free
lookup per byte. A call streams 50 frames a second of 160 bytes — 8000
lookups/second/call against a table that fits in L1.
"""

from __future__ import annotations

# G.711 constants, named as the standard names them.
_BIAS = 0x84       # 132; added before the exponent search, removed after
_CLIP = 32635      # the largest magnitude mu-law can represent (32124 + BIAS)
_SIGN_BIT = 0x80
_QUANT_MASK = 0x0F  # mantissa
_SEG_MASK = 0x70    # exponent
_SEG_SHIFT = 4


def _decode_sample(byte_: int) -> int:
    """One mu-law byte -> one signed 16-bit sample.

    Straight from G.711: the stored byte is bitwise-inverted, the mantissa is
    shifted back up by the exponent, and the bias that made the logarithm work
    is subtracted again.
    """
    inverted = ~byte_ & 0xFF
    magnitude = ((inverted & _QUANT_MASK) << 3) + _BIAS
    magnitude <<= (inverted & _SEG_MASK) >> _SEG_SHIFT
    return _BIAS - magnitude if inverted & _SIGN_BIT else magnitude - _BIAS


def _encode_sample(sample: int) -> int:
    """One signed 16-bit sample -> one mu-law byte.

    Saturates rather than wrapping. A sample past mu-law's range that wrapped
    would turn the loudest part of a word into the loudest possible sound of
    the OPPOSITE sign — an audible click on every peak.
    """
    sign = _SIGN_BIT if sample < 0 else 0
    magnitude = min(abs(sample), _CLIP) + _BIAS
    exponent = max(magnitude.bit_length() - 8, 0)
    mantissa = (magnitude >> (exponent + 3)) & _QUANT_MASK
    return ~(sign | (exponent << _SEG_SHIFT) | mantissa) & 0xFF


# Built once at import. _DECODE holds the two little-endian bytes directly, so
# decoding a frame is a map over a list of ready-made byte pairs with no
# per-sample struct work.
_DECODE: tuple[bytes, ...] = tuple(
    _decode_sample(b).to_bytes(2, "little", signed=True) for b in range(256)
)
# 65536 entries keyed by the unsigned 16-bit view of the sample, so encoding is
# also a lookup rather than arithmetic.
_ENCODE: bytes = bytes(
    _encode_sample(value - 65536 if value >= 32768 else value)
    for value in range(65536)
)


def decode_sample(byte_: int) -> int:
    """One mu-law byte as a signed 16-bit int. For tests and diagnostics."""
    return int.from_bytes(_DECODE[byte_ & 0xFF], "little", signed=True)


def encode_sample(sample: int) -> int:
    """One signed 16-bit sample as a mu-law byte. For tests and diagnostics."""
    return _ENCODE[max(-32768, min(32767, sample)) & 0xFFFF]


def decode(payload: bytes) -> bytes:
    """Mu-law bytes -> little-endian signed 16-bit PCM.

    This is the inbound call leg: what Plivo streams, in the format Sarvam's
    STT will accept. One byte in, two out — the sample COUNT is unchanged, so
    the duration is unchanged and the call stays in sync.
    """
    return b"".join(map(_DECODE.__getitem__, payload))


def encode(pcm: bytes) -> bytes:
    """Little-endian signed 16-bit PCM -> mu-law bytes.

    Not needed by the live call path today — Sarvam's TTS already emits mu-law.
    It is here because that is one documented line away from being untrue: the
    same reference that lists `mulaw` as an output codec also claims MP3 only
    in one place. If mu-law output turns out unavailable, the fallback is
    linear16 at 8 kHz plus this function, and the table is already built.

    A trailing odd byte cannot be half a sample, so it is dropped rather than
    silently misaligning every sample after it.
    """
    usable = len(pcm) - (len(pcm) % 2)
    return bytes(
        _ENCODE[pcm[i] | (pcm[i + 1] << 8)]
        for i in range(0, usable, 2)
    )
