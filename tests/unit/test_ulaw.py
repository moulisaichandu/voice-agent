"""Unit tests for telephony/ulaw.py — the G.711 mu-law codec.

Pure arithmetic, no I/O. Worth its own tests because a wrong table does not
raise: it produces audio that is quiet, distorted, or inverted, and the only
symptom is a lead being misheard on a live call.

The canonical values below come from the G.711 standard itself rather than
from this implementation, so they catch a table that is self-consistently
wrong — the failure a round-trip test alone cannot see.
"""

import pytest

from app.telephony import ulaw

# ── the values G.711 itself pins ─────────────────────────────────────────────

@pytest.mark.parametrize("byte_, expected", [
    (0xFF, 0),        # positive zero
    (0x7F, 0),        # negative zero — mu-law has both, both decode to silence
    (0x80, 32124),    # largest positive
    (0x00, -32124),   # largest negative
])
def test_the_standard_s_own_endpoints(byte_, expected):
    assert ulaw.decode_sample(byte_) == expected


def test_silence_decodes_to_silence():
    """0xFF is the mu-law idle byte. If this were not ~0 the line would carry a
    DC offset — an audible hum under every call."""
    assert ulaw.decode(b"\xff" * 160) == b"\x00\x00" * 160


# ── shape of the output ──────────────────────────────────────────────────────

def test_decode_produces_two_little_endian_bytes_per_sample():
    """Sarvam is told the stream is pcm_s16le. One byte in, two bytes out, low
    byte first — get the order wrong and every sample is scrambled."""
    out = ulaw.decode(bytes([0x80]))
    assert len(out) == 2
    assert out == (32124).to_bytes(2, "little", signed=True)


def test_a_full_plivo_frame_keeps_its_length():
    """Plivo streams 160-byte frames (20 ms at 8 kHz). Anything that changes
    the sample count changes the duration and desynchronises the call."""
    assert len(ulaw.decode(bytes(range(256)) * 10)) == 2560 * 2


def test_empty_input_is_empty_output():
    assert ulaw.decode(b"") == b""


# ── round trip ───────────────────────────────────────────────────────────────

_NEGATIVE_ZERO = 0x7F


def test_mu_law_has_two_zeroes_and_the_encoder_picks_the_positive_one():
    """A property of G.711, not of this implementation, and the reason the
    round-trip test below has exactly one exemption. Both 0x7F and 0xFF mean
    silence; an encoder handed a zero sample has to choose, and choosing
    consistently is what matters."""
    assert ulaw.decode_sample(_NEGATIVE_ZERO) == 0
    assert ulaw.decode_sample(0xFF) == 0
    assert ulaw.encode_sample(0) == 0xFF


def test_every_byte_survives_a_round_trip():
    """encode(decode(b)) == b for all 256 codes but negative zero. Mu-law is
    lossy from PCM, but each of its own codes maps to a distinct level that
    must encode back to itself — otherwise the two tables disagree and audio
    drifts. The one exemption is the second zero, above."""
    for byte_ in range(256):
        if byte_ == _NEGATIVE_ZERO:
            continue
        decoded = ulaw.decode_sample(byte_)
        assert ulaw.encode_sample(decoded) == byte_, f"0x{byte_:02X} did not survive"


# Mu-law's smallest step is 8, so anything quieter than that quantises to
# silence and has no sign left to preserve. Asserting sign below this would be
# asserting something the format cannot do.
_QUANTISATION_FLOOR = 8


@pytest.mark.parametrize("sample", [0, 1, -1, 100, -100, 1000, -1000,
                                    32767, -32768, 8000, -8000])
def test_encoding_a_pcm_sample_stays_close_to_it(sample):
    """Mu-law is logarithmic, so error grows with amplitude — but a decoded
    value must never land in a different magnitude band, and above the
    quantisation floor it must never flip sign. Both are what an exponent or
    sign-bit bug looks like."""
    round_tripped = ulaw.decode_sample(ulaw.encode_sample(sample))
    clipped = max(-32124, min(32124, sample))
    assert abs(round_tripped - clipped) <= max(_QUANTISATION_FLOOR,
                                               abs(clipped) * 0.08)
    if abs(clipped) >= _QUANTISATION_FLOOR:
        assert (round_tripped >= 0) == (clipped >= 0)


def test_quiet_samples_stay_quiet():
    """Near-silence must round to the nearest representable level, which near
    zero means 0 or +-8 — never to something loud. A quiet sample coming back
    large is what an exponent bug looks like, and it shows up on a live call
    as crackle on an otherwise silent line."""
    for sample in range(-_QUANTISATION_FLOOR, _QUANTISATION_FLOOR + 1):
        decoded = ulaw.decode_sample(ulaw.encode_sample(sample))
        assert abs(decoded) <= _QUANTISATION_FLOOR, f"{sample} -> {decoded}"


def test_loud_samples_clip_instead_of_wrapping():
    """A sample beyond mu-law's range must saturate. Wrapping would turn the
    loudest part of a word into the loudest possible sound of the opposite
    sign — a click on every peak."""
    assert ulaw.decode_sample(ulaw.encode_sample(32767)) == 32124
    assert ulaw.decode_sample(ulaw.encode_sample(-32768)) == -32124


def test_encode_and_decode_are_inverse_over_a_whole_buffer():
    pcm = b"".join(
        ulaw.decode_sample(b_).to_bytes(2, "little", signed=True)
        for b_ in range(256)
    )
    assert ulaw.decode(ulaw.encode(pcm)) == pcm
