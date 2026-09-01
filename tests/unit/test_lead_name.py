"""app/telephony/lead_name.py — the lead's name, in a script the voice can read.

Reported by the owner after a live call on 2026-08-31: "Mouli గారు" was not
pronounced properly. The greeting is built as "నమస్తే {name} గారు", and the
name arrives from the console or the sheet in LATIN letters — so Sarvam's
Telugu TTS reads a Latin run with English phonetics instead of saying మౌళి.
It is the same defect as the Latin "AI" in the opening, in the one word the
lead cares most about hearing right: their own.

An audit of every agent turn had already flagged 'Mouli' and 'Gayathri' as
Latin tokens spoken aloud; they were dismissed as unavoidable. They are not.
"""

import pytest

from app.telephony import lead_name


class _FakeRedis:
    def __init__(self, store=None):
        self.store = dict(store or {})
        self.sets = []

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.sets.append((key, value, ex))
        self.store[key] = value


@pytest.fixture
def redis(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(lead_name.redis_client, "get_redis", lambda: fake)
    return fake


def _model(monkeypatch, reply, calls=None):
    def fake(name: str) -> str:
        if calls is not None:
            calls.append(name)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(lead_name, "_transliterate_sync", fake)


async def test_a_latin_name_becomes_telugu_script(redis, monkeypatch):
    _model(monkeypatch, "మౌళి")

    assert await lead_name.telugu_name("Mouli") == "మౌళి"


async def test_a_name_already_in_telugu_is_left_alone(redis, monkeypatch):
    calls = []
    _model(monkeypatch, "SHOULD NOT BE CALLED", calls)

    assert await lead_name.telugu_name("మౌళి") == "మౌళి"
    assert calls == [], "spent a model call on a name that was already Telugu"


async def test_the_result_is_cached_so_a_name_is_transliterated_once(redis, monkeypatch):
    calls = []
    _model(monkeypatch, "మౌళి", calls)

    first = await lead_name.telugu_name("Mouli")
    second = await lead_name.telugu_name("Mouli")

    assert first == second == "మౌళి"
    assert len(calls) == 1, "re-transliterated a name it had already resolved"
    assert redis.sets, "never cached the result"


async def test_a_failure_keeps_the_original_name(redis, monkeypatch):
    """A mispronounced name is a blemish; a missing or crashed greeting is a
    broken call. This never raises and never returns empty."""
    _model(monkeypatch, RuntimeError("model down"))

    assert await lead_name.telugu_name("Mouli") == "Mouli"


async def test_a_reply_that_is_not_telugu_is_rejected(redis, monkeypatch):
    """The model occasionally answers with an explanation or echoes the Latin
    input. Anything that is not Telugu script is not an improvement, so the
    original stands rather than a sentence being read out as a name."""
    for bad in ("Mouli", "The Telugu for Mouli is మౌళి.", "", "   "):
        _model(monkeypatch, bad)
        redis.store.clear()
        assert await lead_name.telugu_name("Mouli") == "Mouli", bad


async def test_a_blank_or_missing_name_is_handled(redis, monkeypatch):
    _model(monkeypatch, "మౌళి")

    assert await lead_name.telugu_name("") == ""
    assert await lead_name.telugu_name(None) == ""


async def test_a_dead_cache_still_transliterates(monkeypatch):
    """Redis is ephemeral by design (CLAUDE.md). Losing it must cost speed,
    never the correct pronunciation."""
    class _Dead:
        async def get(self, key):
            raise RuntimeError("redis down")

        async def set(self, key, value, ex=None):
            raise RuntimeError("redis down")

    monkeypatch.setattr(lead_name.redis_client, "get_redis", lambda: _Dead())
    _model(monkeypatch, "మౌళి")

    assert await lead_name.telugu_name("Mouli") == "మౌళి"
