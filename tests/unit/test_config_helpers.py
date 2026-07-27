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
