"""Tests for app/config.py's env-parsing helpers.

CLAUDE.md's rule for these is specific: a malformed override must "warn +
default, never crash at import". Two of the four helpers did not honour it,
and because config is read once at import the consequences were silent and
process-wide — see the regression tests below.
"""

import math

import pytest

from app import config

# ── _bool ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " True "])
def test_bool_accepts_the_documented_true_values(monkeypatch, raw):
    monkeypatch.setenv("SOME_FLAG", raw)
    assert config._bool("SOME_FLAG", default=False) is True


@pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off", " off "])
def test_bool_accepts_the_documented_false_values(monkeypatch, raw):
    monkeypatch.setenv("SOME_FLAG", raw)
    assert config._bool("SOME_FLAG", default=True) is False


@pytest.mark.parametrize("raw", ["enabled", "y", "n", "True_", "maybe", "2"])
def test_bool_falls_back_to_the_default_on_anything_unrecognised(monkeypatch, raw, caplog):
    """REGRESSION. This was `raw.lower() in ("1","true","yes","on")`, which
    returned False for every unrecognised value and ignored the declared
    default — with no warning. WORKER_ENABLED=enabled therefore turned the
    call worker OFF while campaign_tick kept filling calls:queue with leads
    nothing would ever consume: a complete outbound-dialling outage whose only
    symptom was silence. main.py's "worker not starting" warning only fires
    when the flag is truthy, so not even that appeared."""
    monkeypatch.setenv("SOME_FLAG", raw)
    with caplog.at_level("WARNING"):
        assert config._bool("SOME_FLAG", default=True) is True
        assert config._bool("SOME_FLAG", default=False) is False
    assert "SOME_FLAG" in caplog.text


def test_bool_unset_and_blank_use_the_default(monkeypatch):
    monkeypatch.delenv("SOME_FLAG", raising=False)
    assert config._bool("SOME_FLAG", default=True) is True
    monkeypatch.setenv("SOME_FLAG", "   ")
    assert config._bool("SOME_FLAG", default=True) is True


# ── _float ───────────────────────────────────────────────────────────────────

def test_float_parses_a_normal_value(monkeypatch):
    monkeypatch.setenv("SOME_NUM", "0.45")
    assert config._float("SOME_NUM", 0.30) == 0.45


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "NaN", "Infinity"])
def test_float_rejects_non_finite_values(monkeypatch, raw, caplog):
    """REGRESSION. nan/inf parse cleanly through float() but poison every
    comparison downstream. RAG_MIN_SCORE=nan makes `score >= min_score` False
    for every chunk, so the live agent answers "no relevant material" to
    every question ever asked — silently, with nothing logged."""
    monkeypatch.setenv("SOME_NUM", raw)
    with caplog.at_level("WARNING"):
        assert config._float("SOME_NUM", 0.30) == 0.30
    assert "SOME_NUM" in caplog.text


def test_float_warns_and_defaults_on_garbage(monkeypatch, caplog):
    monkeypatch.setenv("SOME_NUM", "abc")
    with caplog.at_level("WARNING"):
        assert config._float("SOME_NUM", 0.30) == 0.30
    assert "SOME_NUM" in caplog.text


def test_configured_rag_min_score_is_a_usable_threshold():
    """A guard on the live value rather than the parser: the whole RAG safety
    story rests on this being a finite score strictly between 0 and 1. At 0 the
    filtered search behaves exactly like search_permissive, which CLAUDE.md
    forbids wiring to the live tool."""
    assert math.isfinite(config.RAG_MIN_SCORE)
    assert 0 < config.RAG_MIN_SCORE < 1


# ── _int ─────────────────────────────────────────────────────────────────────

def test_int_warns_and_defaults_on_garbage(monkeypatch, caplog):
    monkeypatch.setenv("SOME_INT", "12abc")
    with caplog.at_level("WARNING"):
        assert config._int("SOME_INT", 10) == 10
    assert "SOME_INT" in caplog.text


# ── _list ────────────────────────────────────────────────────────────────────

def test_list_strips_and_drops_empty_entries(monkeypatch):
    monkeypatch.setenv("SOME_LIST", " a , ,b,  ")
    assert config._list("SOME_LIST") == ["a", "b"]


# ── _choice ──────────────────────────────────────────────────────────────────

def test_choice_accepts_an_allowed_value(monkeypatch):
    monkeypatch.setenv("SOME_CHOICE", "sarvam")
    assert config._choice("SOME_CHOICE", "openai_realtime",
                          ("sarvam", "openai_realtime")) == "sarvam"


def test_choice_is_case_and_whitespace_insensitive(monkeypatch):
    """Operators edit .env by hand. ' Sarvam ' meaning something different from
    'sarvam' would route every Telugu call to the wrong backend over a stray
    space, and the only symptom would be the wrong voice on a live call."""
    monkeypatch.setenv("SOME_CHOICE", "  SARVAM ")
    assert config._choice("SOME_CHOICE", "openai_realtime",
                          ("sarvam", "openai_realtime")) == "sarvam"


def test_choice_warns_and_defaults_on_an_unknown_value(monkeypatch, caplog):
    """CLAUDE.md's rule for every config helper: a malformed override warns and
    falls back, it never crashes at import. A typo'd backend name must not be
    able to stop the app booting, and must not pass silently either."""
    monkeypatch.setenv("SOME_CHOICE", "elevnlabs")
    with caplog.at_level("WARNING"):
        assert config._choice("SOME_CHOICE", "sarvam",
                              ("sarvam", "openai_realtime")) == "sarvam"
    assert "SOME_CHOICE" in caplog.text


def test_choice_unset_and_blank_use_the_default(monkeypatch):
    monkeypatch.delenv("SOME_CHOICE", raising=False)
    assert config._choice("SOME_CHOICE", "sarvam", ("sarvam", "x")) == "sarvam"
    monkeypatch.setenv("SOME_CHOICE", "   ")
    assert config._choice("SOME_CHOICE", "sarvam", ("sarvam", "x")) == "sarvam"


# ── Sarvam / Telugu backend config ───────────────────────────────────────────

def test_configured_telugu_backend_is_one_this_product_can_dial(monkeypatch):
    """A guard on the live value rather than the parser. TELUGU_BACKEND is the
    rollback switch: it decides which bridge carries every Telugu and Tinglish
    call. A value outside this set would make languages.backend_for() return a
    backend call_routes has no bridge for, and every Telugu call would fail at
    dial time."""
    assert config.TELUGU_BACKEND in ("sarvam", "openai_realtime")


# ── OpenAI Realtime config tests ──────────────────────────────────────────────

def test_a_malformed_realtime_threshold_warns_and_keeps_the_default(monkeypatch, caplog):
    """CLAUDE.md's config rule: a malformed override warns and falls back, it
    never crashes at import. A voice agent that won't boot because someone
    typo'd a VAD threshold is worse than one running the default."""
    monkeypatch.setenv("OPENAI_REALTIME_VAD_THRESHOLD", "not-a-number")
    with caplog.at_level("WARNING"):
        assert config._float("OPENAI_REALTIME_VAD_THRESHOLD", 0.5) == 0.5
    assert "OPENAI_REALTIME_VAD_THRESHOLD" in caplog.text


# ── a dial lock must outlive the call it guards ────────────────────────────
#
# The `dialing:<phone>` lock stops one number being dialled twice at once, but
# it was a flat 60s while a call may run CALL_MAX_DURATION_S (300s) — so it
# stopped protecting four minutes before the first call could end. A person
# enrolled in two campaigns could be rung a second time while still talking to
# the agent: both legs billed, and the second attempt burned against
# max_attempts for a conversation that already happened.

def test_the_dialing_lock_outlives_the_longest_possible_call():
    from app import config

    assert config.DIALING_LOCK_TTL_S >= (
        config.CALL_RING_TIMEOUT_S + config.CALL_MAX_DURATION_S
    ), (
        f"lock expires after {config.DIALING_LOCK_TTL_S}s but a call can ring "
        f"for {config.CALL_RING_TIMEOUT_S}s then run for "
        f"{config.CALL_MAX_DURATION_S}s — the same person can be dialled twice"
    )


# ── a calling window that can never open must not pass silently ────────────
#
# `END=0` is the natural way to write midnight, but the code means [start, end)
# on a 24-hour clock and wants 24. `9 <= hour < 0` is false at every hour of
# every day: all dialling stops permanently while the readiness dashboard still
# reports it can dial.

def test_an_inverted_calling_window_falls_back_to_the_defaults():
    from app import config

    assert config._validated_hours(9, 0) == (
        config._DEFAULT_HOURS_START, config._DEFAULT_HOURS_END)


def test_an_out_of_range_calling_window_falls_back_to_the_defaults():
    from app import config

    assert config._validated_hours(-3, 99) == (
        config._DEFAULT_HOURS_START, config._DEFAULT_HOURS_END)


def test_a_fully_open_window_is_still_allowed():
    """0-24 is the owner's deliberate current setting; it must keep working."""
    from app import config

    assert config._validated_hours(0, 24) == (0, 24)


def test_an_ordinary_window_is_left_alone():
    from app import config

    assert config._validated_hours(10, 19) == (10, 19)


# ── _sheet_id ────────────────────────────────────────────────────────────────
#
# Observed live 2026-08-27: GOOGLE_SHEET_ID held the full Google Sheets URL,
# so every sheets_sync and transcript_reconcile sweep skipped with the
# open_by_key hint while 63 transcript write-backs sat queued in Redis. A
# pasted URL is the single most likely malformed value for this variable —
# the browser address bar is where the operator sees the sheet — so the
# config boundary extracts the ID rather than punting to a .env edit,
# following the same "malformed override → warn + degrade" rule as the
# other helpers.

def test_sheet_id_passes_a_bare_id_through(monkeypatch):
    monkeypatch.setenv("SOME_SHEET", "1AbC-dEf_1234567890")
    assert config._sheet_id("SOME_SHEET") == "1AbC-dEf_1234567890"


def test_sheet_id_extracts_the_id_from_a_pasted_url(monkeypatch, caplog):
    monkeypatch.setenv(
        "SOME_SHEET",
        "https://docs.google.com/spreadsheets/d/1AbC-dEf_1234567890/edit?gid=0#gid=0",
    )
    with caplog.at_level("WARNING"):
        assert config._sheet_id("SOME_SHEET") == "1AbC-dEf_1234567890"
    assert "SOME_SHEET" in caplog.text
    # /admin/config hides this variable's value as sensitive; the extracted
    # ID must not leak into the log line either.
    assert "1AbC-dEf_1234567890" not in caplog.text


def test_sheet_id_leaves_an_unrecognisable_url_alone(monkeypatch):
    """A URL with no /d/<id> segment passes through unchanged so
    sheets.client's unconfigured_reason can still explain it to the operator —
    guessing an ID out of an arbitrary URL would hide the mistake instead."""
    monkeypatch.setenv("SOME_SHEET", "https://example.com/not-a-sheet")
    assert config._sheet_id("SOME_SHEET") == "https://example.com/not-a-sheet"


def test_sheet_id_unset_or_blank_is_none(monkeypatch):
    monkeypatch.delenv("SOME_SHEET", raising=False)
    assert config._sheet_id("SOME_SHEET") is None
    monkeypatch.setenv("SOME_SHEET", "   ")
    assert config._sheet_id("SOME_SHEET") is None


# ── how fast the agent speaks ────────────────────────────────────────────────

def test_the_speaking_pace_is_faster_than_sarvams_default():
    """The lead said so, on the call of 2026-08-31, in Telugu, to the agent
    itself: "చాలా మెల్లగా మాట్లాడుతున్నాడు" — he is speaking very slowly.
    The old comment here reasoned that telephony audio is hard to follow so
    it should "err slow rather than fast"; a real listener on a real 8 kHz
    call has now disagreed. Measured on that same sentence: pace 1.0 took
    ~9.5s of audio, 1.2 took ~8.0s."""
    from app import config

    # The DEFAULT, not the resolved value: .env legitimately overrides this
    # per deployment (and on the machine this was written on it pins 1.0),
    # so asserting the live value would make this test a report on somebody's
    # .env rather than on the shipped behaviour.
    assert config._DEFAULT_TTS_PACE > 1.0, (
        "still at Sarvam's neutral pace, which a live lead called too slow"
    )
    assert config._DEFAULT_TTS_PACE <= 1.4, (
        "fast enough to lose a listener on a phone line"
    )


def test_the_pace_can_still_be_tuned_from_env(monkeypatch):
    """It is a taste judgement on a real voice, so it stays an operator knob
    rather than a constant — including back to 1.0."""
    monkeypatch.setenv("SOME_PACE", "1.0")
    assert config._float("SOME_PACE", 1.2) == 1.0
