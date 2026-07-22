# Telugu via an OpenAI Realtime Backend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Telugu and Tinglish campaigns dial, by adding a second voice backend (OpenAI Realtime) behind the existing `campaigns.language` selection — one-way first, then two-way with RAG.

**Architecture:** `app/languages.py` gains a capability table mapping each language to the backend that can carry it. `app/telephony/call_routes.py` dispatches to `bridge.py` (ElevenLabs) or the new `openai_bridge.py` (OpenAI Realtime) on that basis. Both bridges drive the same `PlivoCall` from `app/telephony/plivo_stream.py` and honour the identical `outcome` contract, so everything downstream — recording, Sheets write-back, slot release, retry accounting — is unchanged and backend-agnostic.

**Tech Stack:** FastAPI 3.12 · OpenAI Realtime (`gpt-realtime-2.1`, μ-law 8 kHz) · Plivo Voice API · asyncpg · pgvector RAG · pytest

## Why this exists

ElevenLabs Agents does not support Telugu. Not a plan tier, not a model setting — its API enumerates the languages it accepts and `te` is not among them (`app/languages.py`'s `ELEVENLABS_AGENT_LANGUAGES`, verified against the live API on 2026-07-22). Telugu is this business's primary market language, so the product needs a second backend.

`../ai-voice-agent/` — a separate, working system for the same business — already runs Telugu phone calls on OpenAI Realtime over Plivo, with the same μ-law-8 kHz-passthrough shape this project uses. It is the source material for the bridge and the prompts. It is **read-only**: do not modify anything in that repo.

What it does NOT have, and what this project supplies by wrapping the ported bridge in its existing worker: DND scrub, 10:00–19:00 IST calling-hours gate, AI-disclosure enforcement, `max_attempts`, reserve-then-act queueing, a database. That asymmetry is the whole reason we port *into* here rather than calling out.

## Global Constraints

- **Nothing about the ElevenLabs path may change.** English and Hindi campaigns work in production today. `bridge.py`'s behaviour, its `outcome` contract, and `auto`'s no-override frame are all fixed points.
- **`outcome` contract is identical across backends:** `{status, turns, transcript, conversation_id}`, populated in a `finally` so a mid-call exception still yields what was collected. `app/telephony/call_routes.py::_finalise_call` must not learn which backend ran.
- **μ-law 8 kHz passthrough both directions**, no transcoding — `audio/pcmu` on the OpenAI side, matching Plivo natively.
- **AI disclosure is a hard rule.** The owner chose to allow English scripts rendered into Telugu by the model, so the validated script is *not* what the lead hears. Task 6's post-call check on the actually-spoken first turn is what keeps the rule real. It is not optional.
- **RAG:** `search_relevant()` is the ONLY function the live tool may call. `search_permissive()` must never be wired to it. Never invent course facts.
- Every env var read ONCE in `app/config.py` via its `_int`/`_float`/`_bool`/`_list` helpers.
- Type hints + pydantic models on every boundary. No module reads `os.environ` directly.
- Unit tests run with no Docker: `./.venv/Scripts/python.exe -m pytest -m "not integration"`. Lint: `ruff check .`
- Suite is currently **433 passing**. It may only go up.

## Verified protocol facts

Confirmed by reading the sibling's working code, not from documentation:

- Connect: `wss://api.openai.com/v1/realtime?model=<model>`, header `Authorization: Bearer <OPENAI_API_KEY>`.
- `websockets` ≥ 14 renamed the header kwarg `extra_headers` → `additional_headers`. The sibling tries the new name and falls back (`openai_live.py:288-294`). Do the same.
- Inbound audio to the model: `{"type": "input_audio_buffer.append", "audio": <base64 μ-law>}`
- Outbound audio from the model: `response.output_audio.delta` (older alias `response.audio.delta`), field `delta`.
- Agent transcript: `response.output_audio_transcript.delta` / `response.audio_transcript.delta`
- Lead transcript: `conversation.item.input_audio_transcription.completed`
- Barge-in signal: `input_audio_buffer.speech_started`
- Response lifecycle: `response.created`, `response.done` (function calls arrive in `response.done`'s `output` as items with `"type": "function_call"`)
- Errors: `{"type": "error", ...}`
- Session is configured by sending `{"type": "session.update", "session": {...}}` after connect.
- A one-way call sets `audio.input.turn_detection = None` so the model never auto-responds, and attaches no tools.

---

# MILESTONE A — One-way Telugu (Tasks 1–7)

Ends with real Telugu calls dialable and testable. Do not start Milestone B until a live one-way Telugu call has been made and judged.

---

### Task 1: Which backend carries which language

**Files:**
- Modify: `app/languages.py`
- Test: `tests/unit/test_languages.py`

**Interfaces:**
- Consumes: existing `TOKENS`, `AUTO`, `iso_code`, `ELEVENLABS_AGENT_LANGUAGES`.
- Produces: `VoiceBackend = Literal["elevenlabs", "openai_realtime"]`, `backend_for(token: str) -> str`, `ELEVENLABS = "elevenlabs"`, `OPENAI_REALTIME = "openai_realtime"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_languages.py`:

```python
# ── which backend carries which language ─────────────────────────────────────

def test_telugu_and_tinglish_route_to_openai():
    """ElevenLabs cannot speak Telugu at all — see ELEVENLABS_AGENT_LANGUAGES.
    Routing them anywhere else is what makes these campaigns dialable."""
    assert languages.backend_for("te") == languages.OPENAI_REALTIME
    assert languages.backend_for("tinglish") == languages.OPENAI_REALTIME


def test_english_hindi_and_auto_stay_on_elevenlabs():
    """The working path. Every campaign in production today is one of these,
    and none of them may move backend as a side effect of this feature."""
    for token in ("auto", "en", "hi", "hinglish"):
        assert languages.backend_for(token) == languages.ELEVENLABS


def test_every_catalogue_token_has_a_backend():
    """A language in the dropdown with no backend would fail at dial time with
    a KeyError rather than a message anyone can act on."""
    for token in languages.TOKENS:
        assert languages.backend_for(token) in (
            languages.ELEVENLABS, languages.OPENAI_REALTIME
        )


def test_an_unknown_token_falls_back_to_the_working_backend():
    """normalize() sends junk to 'auto', so this can only happen if a caller
    hand-builds a token. Degrade to the backend that works, not to a crash."""
    assert languages.backend_for("klingon") == languages.ELEVENLABS


def test_no_language_routes_to_elevenlabs_for_something_it_cannot_speak():
    """The consistency guard between the two tables: if a language is routed
    to ElevenLabs, ElevenLabs must actually offer it. Adding Tamil later and
    forgetting to route it would otherwise dial into a wall."""
    for token in languages.TOKENS:
        iso = languages.iso_code(token)
        if iso and languages.backend_for(token) == languages.ELEVENLABS:
            assert languages.elevenlabs_can_speak(iso), (
                f"{token} routes to ElevenLabs but ElevenLabs cannot speak {iso}"
            )
```

- [ ] **Step 2: Run to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_languages.py -q -k backend`
Expected: FAIL — `AttributeError: module 'app.languages' has no attribute 'backend_for'`

- [ ] **Step 3: Implement**

In `app/languages.py`, add after `ELEVENLABS_AGENT_LANGUAGES` / `elevenlabs_can_speak`:

```python
# ── which backend carries which language ─────────────────────────────────────
# Two voice backends exist because no single one covers this business's
# languages. ElevenLabs Agents sounds better and is already in production, but
# does not offer Telugu at any price (see ELEVENLABS_AGENT_LANGUAGES). OpenAI
# Realtime does, and the sibling ../ai-voice-agent proves it on real 8 kHz
# phone calls.
#
# The backend is DERIVED from the language, never chosen per call and never
# inferred by a model — campaigns.language is admin-set at creation, so the
# backend is fully determined before a call is placed. That is the same
# discipline campaigns.mode follows, for the same reason.
ELEVENLABS = "elevenlabs"
OPENAI_REALTIME = "openai_realtime"

VoiceBackend = Literal["elevenlabs", "openai_realtime"]

_BACKEND_BY_TOKEN: dict[str, str] = {
    AUTO: ELEVENLABS,
    "en": ELEVENLABS,
    "hi": ELEVENLABS,
    "hinglish": ELEVENLABS,
    "te": OPENAI_REALTIME,
    "tinglish": OPENAI_REALTIME,
}


def backend_for(token: str) -> str:
    """The voice backend that can carry *token*.

    Falls back to ELEVENLABS for anything unrecognised: normalize() already
    routes junk to AUTO, so reaching this means a caller hand-built a token,
    and degrading to the backend that is known to work beats raising mid-dial.
    """
    return _BACKEND_BY_TOKEN.get(normalize(token), ELEVENLABS)
```

Add `Literal` to the `typing` import at the top of the file.

- [ ] **Step 4: Run the tests**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_languages.py -q`
Expected: PASS.

- [ ] **Step 5: Full suite + lint**

Run: `./.venv/Scripts/python.exe -m pytest -m "not integration"` then `./.venv/Scripts/python.exe -m ruff check .`
Expected: 433 + 5 passing, lint clean.

- [ ] **Step 6: Commit**

```bash
git add app/languages.py tests/unit/test_languages.py
git commit -m "Map each language to the voice backend that can carry it"
```

---

### Task 2: Configuration for the OpenAI Realtime backend

**Files:**
- Modify: `app/config.py`, `.env.example`
- Test: `tests/unit/test_config_helpers.py`

**Interfaces:**
- Produces: `OPENAI_REALTIME_MODEL`, `OPENAI_REALTIME_VOICE`, `OPENAI_REALTIME_STT_MODEL`, `OPENAI_REALTIME_STT_LANGUAGE`, `OPENAI_REALTIME_VAD_THRESHOLD`, `OPENAI_REALTIME_SILENCE_MS`, `OPENAI_REALTIME_NOISE_REDUCTION`.

`OPENAI_API_KEY` already exists (RAG embeddings use it) — do not add it again.

- [ ] **Step 1: Add the config**

In `app/config.py`, near the other ElevenLabs/telephony settings:

```python
# ── OpenAI Realtime (the Telugu voice backend) ──────────────────────────────
# OPENAI_API_KEY is declared above — the RAG embedder uses the same key.
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1")
# Every Realtime voice is English-first; there is no Telugu-native voice. The
# sibling project chose theirs by generating the same Telugu sentence in each
# voice at real phone quality (8 kHz mu-law) and listening. Blank uses the
# API default.
OPENAI_REALTIME_VOICE = os.getenv("OPENAI_REALTIME_VOICE", "")
OPENAI_REALTIME_STT_MODEL = os.getenv("OPENAI_REALTIME_STT_MODEL", "gpt-4o-transcribe")
# THE most load-bearing value here. On 8 kHz telephony audio, automatic
# language detection was observed guessing Croatian and Urdu for Telugu
# speech — the lead is then transcribed as nonsense and the model answers
# nonsense. Pinning the language is what stops that. Blank restores
# auto-detection, which is almost never what you want on a phone call.
OPENAI_REALTIME_STT_LANGUAGE = os.getenv("OPENAI_REALTIME_STT_LANGUAGE", "te")
# Energy-gated VAD, standard for telephony. The threshold is a "voice radius":
# higher means only louder/nearer speech ends a turn, rejecting background.
OPENAI_REALTIME_VAD_THRESHOLD = _float("OPENAI_REALTIME_VAD_THRESHOLD", 0.5)
# How long a lead may pause before the model takes its turn. Too low and it
# interrupts someone mid-sentence; the sibling floors this at 600ms.
OPENAI_REALTIME_SILENCE_MS = _int("OPENAI_REALTIME_SILENCE_MS", 700)
# Suppress background noise before it reaches the model. Blank disables.
OPENAI_REALTIME_NOISE_REDUCTION = os.getenv("OPENAI_REALTIME_NOISE_REDUCTION", "near_field")
```

- [ ] **Step 2: Document them**

Append to `.env.example`:

```
# ── OpenAI Realtime — the Telugu/Tinglish voice backend ─────────────────────
# Uses OPENAI_API_KEY above (the same key the RAG embedder uses).
# OPENAI_REALTIME_MODEL=gpt-realtime-2.1
# OPENAI_REALTIME_VOICE=
# OPENAI_REALTIME_STT_MODEL=gpt-4o-transcribe
# Pin the STT language. On 8kHz phone audio, auto-detection mis-hears Telugu
# as Croatian/Urdu. Blank restores auto-detection.
# OPENAI_REALTIME_STT_LANGUAGE=te
# OPENAI_REALTIME_VAD_THRESHOLD=0.5
# OPENAI_REALTIME_SILENCE_MS=700
# OPENAI_REALTIME_NOISE_REDUCTION=near_field
```

- [ ] **Step 3: Test the malformed-override behaviour**

Append to `tests/unit/test_config_helpers.py`:

```python
def test_a_malformed_realtime_threshold_warns_and_keeps_the_default(monkeypatch, caplog):
    """CLAUDE.md's config rule: a malformed override warns and falls back, it
    never crashes at import. A voice agent that won't boot because someone
    typo'd a VAD threshold is worse than one running the default."""
    monkeypatch.setenv("OPENAI_REALTIME_VAD_THRESHOLD", "not-a-number")
    import importlib

    from app import config as config_module
    importlib.reload(config_module)
    assert config_module.OPENAI_REALTIME_VAD_THRESHOLD == 0.5
    importlib.reload(config_module)
```

- [ ] **Step 4: Run and lint**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_config_helpers.py -q` then `ruff check .`
Expected: PASS, clean.

- [ ] **Step 5: Commit**

```bash
git add app/config.py .env.example tests/unit/test_config_helpers.py
git commit -m "Configure the OpenAI Realtime voice backend"
```

---

### Task 3: The OpenAI Realtime bridge — one-way

**Files:**
- Create: `app/telephony/openai_bridge.py`
- Test: `tests/unit/test_openai_bridge.py`

**Interfaces:**
- Consumes: `PlivoCall` from `app/telephony/plivo_stream.py` — `play(b64)`, `interrupt()`, `read_events(on_media)`, `note_exit(reason)`, `ran_its_course`, `oneway_watchdog()`, `run(*tasks)`, `.stop`.
- Produces: `bridge(plivo_ws, *, agent_id, lead_id, dynamic_variables=None, language=None, one_way=False, outcome=None) -> dict` — **the same signature and the same `outcome` contract as `app/telephony/bridge.py::bridge`**, so `call_routes` can dispatch to either without knowing the difference.

`agent_id` is accepted and ignored: OpenAI Realtime has no agent objects. Keeping the signature identical is what makes the dispatch in Task 4 a one-line choice rather than a branch with two call shapes.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_openai_bridge.py`:

```python
"""Unit tests for telephony/openai_bridge.py — the Telugu voice backend.

Driven against a fake OpenAI Realtime socket and the same fake Plivo socket
`PlivoCall` is tested with, so no network and no Docker.

The behaviour that matters most here is the same as the ElevenLabs bridge's:
a one-way call must end once its message has been delivered, and a mid-call
failure must still hand back the transcript of a conversation that really
happened. Both are inherited from PlivoCall, so these tests prove the wiring
is right, not that the watchdog works (tests/unit/test_plivo_stream.py owns
that).
"""

import asyncio
import base64
import json

import pytest

from app.telephony import openai_bridge

_AUDIO_B64 = base64.b64encode(b"\xff" * 640).decode()


def _audio_delta(b64: str = _AUDIO_B64) -> str:
    return json.dumps({"type": "response.output_audio.delta", "delta": b64})


def _agent_transcript(text: str) -> str:
    return json.dumps({
        "type": "response.output_audio_transcript.delta", "delta": text,
    })


def _response_done() -> str:
    return json.dumps({"type": "response.done", "response": {"output": []}})


class _FakeOpenAIWS:
    """Yields *messages*, then goes quiet — matching a one-way call, where the
    model has no turn to end and never closes the socket itself."""

    def __init__(self, messages: list[str]):
        self._messages = list(messages)
        self.sent: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def send(self, payload):
        self.sent.append(payload)

    async def close(self):
        pass

    def __aiter__(self):
        async def _gen():
            for m in self._messages:
                yield m
            await asyncio.Event().wait()
        return _gen()


class _FakePlivoWS:
    def __init__(self):
        self.sent: list[dict] = []

    async def receive_text(self):
        await asyncio.Event().wait()

    async def send_text(self, raw):
        self.sent.append(json.loads(raw))


@pytest.fixture
def bridged(monkeypatch):
    monkeypatch.setattr(openai_bridge, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(openai_bridge.plivo_stream, "ONEWAY_SILENCE_TAIL_S", 0)
    monkeypatch.setattr(openai_bridge.plivo_stream, "ONEWAY_MAX_SILENT_S", 0)

    def _install(oa_ws):
        async def fake_connect(*a, **kw):
            return oa_ws
        monkeypatch.setattr(openai_bridge, "_connect", fake_connect)

    return _install


def _session(oa_ws) -> dict:
    """The session.update frame — the first thing sent after connecting."""
    return json.loads(oa_ws.sent[0])


async def test_a_one_way_call_delivers_its_message_and_ends(bridged):
    """The expensive one. Nothing but the watchdog ends a one-way call: the
    model gets no turn-end signal because we never forward the lead's audio."""
    oa = _FakeOpenAIWS([_agent_transcript("Namaste."), _audio_delta(), _response_done()])
    bridged(oa)
    plivo = _FakePlivoWS()

    outcome = await asyncio.wait_for(
        openai_bridge.bridge(plivo, agent_id="ignored", lead_id="lead-1",
                             language="te", one_way=True),
        timeout=5,
    )

    assert outcome["status"] == "done"
    assert [t.text for t in outcome["transcript"]] == ["Namaste."]
    assert any(m["event"] == "playAudio" for m in plivo.sent)


async def test_a_one_way_session_disables_turn_detection(bridged):
    """A one-way call must never auto-respond to the lead. The bridge also
    never forwards their audio, but belt and braces: with turn detection on,
    the model would try to take turns against silence."""
    oa = _FakeOpenAIWS([_audio_delta()])
    bridged(oa)

    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=True),
        timeout=5,
    )

    session = _session(oa)["session"]
    assert session["audio"]["input"]["turn_detection"] is None
    assert "tools" not in session, "a one-way call has nothing to call tools for"


async def test_the_session_pins_mulaw_both_directions(bridged):
    """Plivo is told the stream is mu-law 8kHz. Anything else is misframed and
    the lead hears noise — the same rule the ElevenLabs path follows."""
    oa = _FakeOpenAIWS([_audio_delta()])
    bridged(oa)

    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=True),
        timeout=5,
    )

    audio = _session(oa)["session"]["audio"]
    assert audio["input"]["format"]["type"] == "audio/pcmu"
    assert audio["output"]["format"]["type"] == "audio/pcmu"


async def test_the_script_is_given_as_content_not_words_to_recite(bridged):
    """A script written in English must be CONVEYED in Telugu, not read out in
    English. The sibling hit exactly this: leads got an English call from a
    Telugu-first agent because the prompt said 'say this'."""
    oa = _FakeOpenAIWS([_audio_delta()])
    bridged(oa)

    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=True,
                             dynamic_variables={"script": "Our new course starts Monday."}),
        timeout=5,
    )

    instructions = _session(oa)["session"]["instructions"]
    assert "Our new course starts Monday." in instructions
    assert "never read English text aloud" in instructions.lower() or \
           "convey its meaning" in instructions.lower()


async def test_the_lead_name_reaches_the_prompt(bridged):
    oa = _FakeOpenAIWS([_audio_delta()])
    bridged(oa)

    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=True,
                             dynamic_variables={"lead_name": "Asha"}),
        timeout=5,
    )

    assert "Asha" in _session(oa)["session"]["instructions"]


async def test_no_api_key_fails_cleanly_without_dialling(monkeypatch):
    """Same contract as the ElevenLabs bridge: return a failed outcome rather
    than raising into the caller's finally."""
    monkeypatch.setattr(openai_bridge, "OPENAI_API_KEY", None)
    outcome = await openai_bridge.bridge(
        _FakePlivoWS(), agent_id="x", lead_id="l", language="te", one_way=True
    )
    assert outcome["status"] == "failed"
    assert outcome["turns"] == 0
```

- [ ] **Step 2: Run to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_openai_bridge.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.telephony.openai_bridge'`

- [ ] **Step 3: Write the prompts module**

Create `app/telephony/openai_prompts.py`:

```python
"""telephony/openai_prompts.py — Telugu call personas for the Realtime backend.

Adapted from the sibling ../ai-voice-agent's backend/prompts.py, which earned
these rules on real 8 kHz phone calls to real Telugu-speaking leads. Two of
them exist because of specific observed failures and must not be softened:

  1. The script is CONTENT, not words to recite. The sibling's reminder prompt
     originally said "say this", and the model dutifully read English script
     text aloud to Telugu speakers. Both labels below say the script is meaning
     to convey.

  2. Telugu and English only, never Hindi. Without it the model drifted into
     Hindi mid-call, which a Telugu-speaking lead in Hyderabad experiences as
     being called by a stranger who does not know them.

Pure strings, no I/O — testable without a socket.
"""

from __future__ import annotations

_LANGUAGE_RULE = (
    "LANGUAGE: Speak natural, everyday spoken Telugu — the register a person "
    "from Hyderabad actually uses on the phone, not literary Telugu. You may "
    "mix in the English words Telugu speakers themselves use (course, fees, "
    "batch, online, demo, certificate, placement). Never use Hindi or any "
    "other language under any circumstances."
)

_DISCLOSURE_RULE = (
    "FIRST SENTENCE: Your very first sentence must state plainly, in Telugu, "
    "that this is an automated AI call from Digital Brolly. This is a legal "
    "requirement in India and is not optional. Say it before anything else — "
    "before greeting them, before your name, before the reason for the call."
)


def one_way_instructions(lead_name: str | None, script: str | None) -> str:
    """The persona for a call that delivers a message and hangs up."""
    who = f"You are calling {lead_name}. " if lead_name else ""
    parts = [
        f"You are a voice assistant calling on behalf of Digital Brolly, an "
        f"education company in Hyderabad. {who}"
        "Deliver one short message, then say goodbye and stop. Do not ask "
        "questions and do not wait for a reply — this call does not listen.",
        _DISCLOSURE_RULE,
        _LANGUAGE_RULE,
    ]
    if script and script.strip():
        parts.append(
            "MESSAGE TO DELIVER — this is the MEANING to convey in natural "
            "spoken Telugu, not words to recite. It may be written in English; "
            "if so, convey its meaning in Telugu and never read English text "
            f"aloud: {script.strip()}"
        )
    return "\n\n".join(parts)
```

- [ ] **Step 4: Write the bridge**

Create `app/telephony/openai_bridge.py`:

```python
"""telephony/openai_bridge.py — Bridge a live Plivo call to OpenAI Realtime.

The Telugu backend. Exists because the ElevenLabs Agents platform does not
offer Telugu at all (see app/languages.py's ELEVENLABS_AGENT_LANGUAGES) and
Telugu is this business's primary market language.

Deliberately mirrors app/telephony/bridge.py's signature and `outcome`
contract exactly, so app/telephony/call_routes.py picks a backend rather than
branching into two different call shapes, and everything downstream — call
recording, Sheets write-back, slot release, retry accounting — never learns
which one ran.

All the Plivo-side behaviour (playing audio, barge-in, the one-way watchdog
that stops a message-only call billing 300 seconds of silence) lives in
app/telephony/plivo_stream.py and is shared with the ElevenLabs bridge. This
module owns only OpenAI's wire protocol.

Wire protocol, captured from the sibling ../ai-voice-agent's working
implementation rather than assumed from documentation:

    connect  wss://api.openai.com/v1/realtime?model=<model>
             header Authorization: Bearer <OPENAI_API_KEY>
    ->       {"type": "session.update", "session": {...}}
    ->       {"type": "input_audio_buffer.append", "audio": <b64 mu-law>}
    <-       response.output_audio.delta           {delta: <b64 mu-law>}
    <-       response.output_audio_transcript.delta {delta: <text>}
    <-       conversation.item.input_audio_transcription.completed {transcript}
    <-       input_audio_buffer.speech_started      (barge-in)
    <-       response.done                          {response: {output: [...]}}
    <-       error

Both legs are mu-law 8 kHz (`audio/pcmu`), so audio is a passthrough with no
transcoding — the same property the ElevenLabs path has.
"""

from __future__ import annotations

import json
import logging

import websockets
from fastapi import WebSocket

from app.config import (
    OPENAI_API_KEY,
    OPENAI_REALTIME_MODEL,
    OPENAI_REALTIME_NOISE_REDUCTION,
    OPENAI_REALTIME_STT_LANGUAGE,
    OPENAI_REALTIME_STT_MODEL,
    OPENAI_REALTIME_VOICE,
)
from app.db.models import TranscriptTurn
from app.telephony import openai_prompts, plivo_stream
from app.telephony.plivo_stream import PlivoCall

logger = logging.getLogger(__name__)

_REALTIME_URL = "wss://api.openai.com/v1/realtime?model={model}"


async def _connect(url: str, headers: dict):
    """websockets >= 14 renamed `extra_headers` to `additional_headers`.
    Try the new name, fall back to the old — the sibling hit this too."""
    try:
        return await websockets.connect(url, additional_headers=headers, max_size=None)
    except TypeError:
        return await websockets.connect(url, extra_headers=headers, max_size=None)


def _session_update(*, lead_name: str | None, script: str | None,
                    one_way: bool) -> dict:
    """The session frame. One-way disables turn detection and attaches no
    tools, so the model can never take a turn against a lead it isn't
    listening to."""
    output_cfg: dict = {"format": {"type": "audio/pcmu"}}
    if OPENAI_REALTIME_VOICE:
        output_cfg["voice"] = OPENAI_REALTIME_VOICE

    input_cfg: dict = {"format": {"type": "audio/pcmu"}}
    if OPENAI_REALTIME_NOISE_REDUCTION:
        input_cfg["noise_reduction"] = {"type": OPENAI_REALTIME_NOISE_REDUCTION}
    input_cfg["turn_detection"] = None  # two-way overrides this in Milestone B

    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": openai_prompts.one_way_instructions(lead_name, script),
            "output_modalities": ["audio"],
            "audio": {"input": input_cfg, "output": output_cfg},
        },
    }


async def bridge(plivo_ws: WebSocket, *, agent_id: str, lead_id: str,
                 dynamic_variables: dict | None = None,
                 language: str | None = None,
                 one_way: bool = False, outcome: dict | None = None) -> dict:
    """Bridge one answered call to OpenAI Realtime until either side ends.

    Same contract as app/telephony/bridge.py's bridge(). *agent_id* is
    accepted and ignored — OpenAI Realtime has no agent objects — so that
    call_routes can dispatch on backend without reshaping the call.

    *outcome* is populated in place and returned, so a mid-call exception
    still hands the caller every turn collected before it. The ElevenLabs
    bridge's docstring explains why that matters: a three-minute conversation
    that ended badly used to be recorded as zero turns and a 'failed' lead
    that then burned a retry.
    """
    if outcome is None:
        outcome = {}
    outcome.update({"status": "failed", "turns": 0, "transcript": [],
                    "conversation_id": None})
    if not OPENAI_API_KEY:
        logger.error("[openai] OPENAI_API_KEY is not set — cannot bridge the call.")
        return outcome

    variables = dynamic_variables or {}
    call = PlivoCall(plivo_ws, lead_id=lead_id, one_way=one_way)
    turns: list[TranscriptTurn] = []
    agent_text: list[str] = []

    url = _REALTIME_URL.format(model=OPENAI_REALTIME_MODEL)
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    try:
        oa = await _connect(url, headers)
        async with oa:
            await oa.send(json.dumps(_session_update(
                lead_name=variables.get("lead_name"),
                script=variables.get("script"),
                one_way=one_way,
            )))
            # One-way: ask for the message immediately. Nothing else will —
            # with turn detection off the model waits for an explicit cue.
            if one_way:
                await oa.send(json.dumps({"type": "response.create"}))

            async def to_openai(payload_b64: str) -> None:
                await oa.send(json.dumps({
                    "type": "input_audio_buffer.append", "audio": payload_b64,
                }))

            async def from_openai() -> None:
                try:
                    async for raw in oa:
                        if call.stop.is_set():
                            break
                        try:
                            event = json.loads(raw)
                        except (ValueError, TypeError):
                            continue
                        etype = event.get("type", "")

                        if etype in ("response.output_audio.delta",
                                     "response.audio.delta"):
                            delta = event.get("delta")
                            if delta:
                                await call.play(delta)
                        elif etype in ("response.output_audio_transcript.delta",
                                       "response.audio_transcript.delta"):
                            piece = event.get("delta")
                            if piece:
                                agent_text.append(piece)
                        elif etype == "response.done":
                            # Transcript deltas arrive piecemeal; a completed
                            # response is the only place they form a turn.
                            said = "".join(agent_text).strip()
                            agent_text.clear()
                            if said:
                                turns.append(TranscriptTurn(role="agent", text=said))
                        elif etype == "error":
                            logger.error(f"[openai] lead={lead_id} error event: "
                                         f"{event.get('error') or event}")
                except websockets.exceptions.ConnectionClosed as exc:
                    call.note_exit(f"openai_closed:{exc.code}")
                    logger.warning(f"[openai] socket closed: {exc}")
                except Exception as exc:
                    call.note_exit(f"error:{type(exc).__name__}")
                    logger.error(f"[openai] reader failed: "
                                 f"{type(exc).__name__}: {exc}")
                finally:
                    call.stop.set()

            await call.run(call.read_events(to_openai), from_openai())
    except Exception as exc:
        call.note_exit(f"error:{type(exc).__name__}")
        logger.exception(f"[openai] bridge failed for lead {lead_id}: "
                         f"{type(exc).__name__}: {exc}")
    finally:
        # In a finally for the same reason the ElevenLabs bridge is: a call
        # that really happened must not be recorded as nothing.
        trailing = "".join(agent_text).strip()
        if trailing:
            turns.append(TranscriptTurn(role="agent", text=trailing))
        outcome["turns"] = len(turns)
        outcome["transcript"] = turns
        outcome["status"] = "done" if call.ran_its_course else "failed"
        logger.info(f"[openai] lead={lead_id} ended turns={len(turns)}")
    return outcome
```

Note `conversation_id` stays `None` — OpenAI Realtime has no equivalent, and `calls.el_conversation_id` is nullable. `_finalise_call` already records unconditionally with no conversation id (that path exists because an ElevenLabs billing failure produced the same shape).

- [ ] **Step 5: Run the tests**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_openai_bridge.py -q`
Expected: PASS.

- [ ] **Step 6: Full suite + lint, then commit**

```bash
./.venv/Scripts/python.exe -m pytest -m "not integration"
./.venv/Scripts/python.exe -m ruff check .
git add app/telephony/openai_bridge.py app/telephony/openai_prompts.py tests/unit/test_openai_bridge.py
git commit -m "Bridge a Plivo call to OpenAI Realtime for one-way Telugu"
```

---

### Task 4: Dispatch the dial path by backend

**Files:**
- Modify: `app/telephony/call_routes.py`
- Test: `tests/unit/test_call_routes.py`

**Interfaces:**
- Consumes: `languages.backend_for` (Task 1), `openai_bridge.bridge` (Task 3).
- Produces: `_backend_bridge(token) -> Callable` — the bridge function for a resolved language token.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_call_routes.py`:

```python
# ── backend dispatch ─────────────────────────────────────────────────────────

def test_telugu_dispatches_to_the_openai_bridge():
    from app.telephony import openai_bridge
    assert call_routes._backend_bridge("te") is openai_bridge.bridge
    assert call_routes._backend_bridge("tinglish") is openai_bridge.bridge


def test_everything_else_stays_on_the_elevenlabs_bridge():
    """The production path. English and Hindi campaigns are dialling today and
    must not move backend as a side effect of adding Telugu."""
    from app.telephony import bridge as el_bridge
    for token in ("auto", "en", "hi", "hinglish"):
        assert call_routes._backend_bridge(token) is el_bridge.bridge
```

- [ ] **Step 2: Run to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_call_routes.py -q -k backend`
Expected: FAIL — no attribute `_backend_bridge`.

- [ ] **Step 3: Implement**

In `app/telephony/call_routes.py`, add the import and the helper:

```python
from app.telephony import openai_bridge as openai_bridge_module
```

```python
def _backend_bridge(token: str):
    """The bridge function for this call's language.

    Both bridges take the same arguments and populate the same `outcome`, so
    the choice is a function reference rather than a branch — everything after
    this point is backend-agnostic, including _finalise_call.
    """
    if languages.backend_for(token) == languages.OPENAI_REALTIME:
        return openai_bridge_module.bridge
    return bridge_module.bridge
```

In `stream()`, resolve the token once and use it for both the language variables and the dispatch. Replace the `_call_language(lead, campaign)` call and the `bridge_module.bridge(...)` call:

```python
    token = languages.resolve(lead.language_pref, campaign.language)
    call_language, language_vars = _call_language(lead, campaign)
    run_bridge = _backend_bridge(token)
```

and

```python
        await run_bridge(
            ws,
            agent_id=campaign.agent_id,
            lead_id=str(lead.lead_id),
            dynamic_variables=dynamic_variables,
            language=call_language,
            one_way=(campaign.mode == "oneway"),
            outcome=outcome,
        )
```

- [ ] **Step 4: Run tests, full suite, lint, commit**

```bash
./.venv/Scripts/python.exe -m pytest -m "not integration"
./.venv/Scripts/python.exe -m ruff check .
git add app/telephony/call_routes.py tests/unit/test_call_routes.py
git commit -m "Dispatch each call to the backend its language needs"
```

---

### Task 5: Preflight for the OpenAI backend

**Files:**
- Modify: `app/telephony/preflight.py`
- Test: `tests/unit/test_preflight.py`

The current language block asks ElevenLabs what the agent supports. For an OpenAI-backed call every one of those questions is meaningless — there is no agent, no language preset, no override switch — and running them would refuse every Telugu campaign for the wrong reason.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_preflight.py`:

```python
# ── the OpenAI backend has entirely different prerequisites ──────────────────

async def test_a_telugu_campaign_does_not_ask_elevenlabs_anything(monkeypatch):
    """Telugu never touches ElevenLabs. Asking it about an agent that isn't
    involved would refuse every Telugu campaign for a reason that has nothing
    to do with why it can or can't dial."""
    _configured(monkeypatch)
    monkeypatch.setattr(pf, "OPENAI_API_KEY", "sk-test")

    def boom(agent_id):
        raise AssertionError("must not consult ElevenLabs for an OpenAI-backed call")

    monkeypatch.setattr(pf, "agent_language_support", boom)
    assert await pf.preflight("agent_1", "te") is None


async def test_a_telugu_campaign_needs_an_openai_key(monkeypatch):
    _configured(monkeypatch)
    monkeypatch.setattr(pf, "OPENAI_API_KEY", None)
    reason = await pf.preflight("agent_1", "te")
    assert reason is not None
    assert "OPENAI_API_KEY" in reason


async def test_hindi_still_runs_the_elevenlabs_checks(monkeypatch):
    """The other side of the branch: an ElevenLabs-backed language must keep
    every check it had."""
    _configured(monkeypatch)
    _support(monkeypatch, override_allowed=False)
    reason = await pf.preflight("agent_1", "hi")
    assert reason is not None
    assert "Security" in reason
```

- [ ] **Step 2: Run to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_preflight.py -q -k "openai or telugu"`
Expected: FAIL — Telugu currently falls into the ElevenLabs branch.

- [ ] **Step 3: Implement**

Add `OPENAI_API_KEY` to `preflight.py`'s config imports, and replace the opening of the `if language:` block:

```python
    if language:
        # Which questions are worth asking depends entirely on which backend
        # will carry this call. An OpenAI-backed language has no ElevenLabs
        # agent, no language presets and no override switch — running those
        # checks against it would refuse the campaign for reasons unrelated
        # to whether it can dial.
        if languages_module.backend_for_iso(language) == languages_module.OPENAI_REALTIME:
            if not OPENAI_API_KEY:
                return (
                    f"This campaign dials in '{language}', which runs on the "
                    "OpenAI Realtime backend, but OPENAI_API_KEY is not set. "
                    "Set it in .env — it is the same key the RAG embedder uses."
                )
            return None
        # ... existing ElevenLabs checks unchanged from here
```

This needs a helper that maps an **ISO code** (not a token) to a backend, because preflight receives the ISO code. Add to `app/languages.py`:

```python
def backend_for_iso(iso: str | None) -> str:
    """The backend for an ISO code, for callers that only have the code.

    Several tokens can share an ISO code (te and tinglish are both 'te'), but
    they never disagree about the backend — a language is carried by exactly
    one backend — so resolving through the first token that matches is safe.
    """
    for token, backend in _BACKEND_BY_TOKEN.items():
        if iso_code(token) == iso:
            return backend
    return ELEVENLABS
```

with a test in `tests/unit/test_languages.py`:

```python
def test_backend_for_iso_agrees_with_backend_for_token():
    """The two lookups must never disagree — preflight uses the ISO one and
    the dial path uses the token one, on the same call."""
    for token in languages.TOKENS:
        iso = languages.iso_code(token)
        if iso:
            assert languages.backend_for_iso(iso) == languages.backend_for(token)
```

- [ ] **Step 4: Run, lint, commit**

```bash
./.venv/Scripts/python.exe -m pytest -m "not integration"
./.venv/Scripts/python.exe -m ruff check .
git add app/telephony/preflight.py app/languages.py tests/unit/test_preflight.py tests/unit/test_languages.py
git commit -m "Preflight the backend a campaign will actually use"
```

---

### Task 6: Verify the disclosure that was actually spoken

**Files:**
- Modify: `app/telephony/call_routes.py`
- Test: `tests/unit/test_call_routes.py`

**Why this is not optional.** India telecom rules require the AI disclosure as the first line of the call, and `has_ai_disclosure()` enforces it against `campaigns.script` at creation time. On the OpenAI path the operator may write the script in English and the model renders it into Telugu — so **the text that was validated is not the text the lead hears**. The model could reword or drop the disclosure and nothing would notice.

We already capture every agent turn. Checking the first one closes the gap.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_call_routes.py`:

```python
# ── the disclosure that was actually spoken ──────────────────────────────────

def _turn(role, text):
    from app.db.models import TranscriptTurn
    return TranscriptTurn(role=role, text=text)


def test_a_spoken_telugu_disclosure_passes(caplog):
    outcome = {"transcript": [_turn("agent", "నమస్తే! ఇది కృత్రిమ మేధ ద్వారా చేసే కాల్.")]}
    assert call_routes._spoken_disclosure_ok(outcome) is True


def test_a_missing_spoken_disclosure_is_flagged():
    """The model dropped the legally-required disclosure. The call already
    happened — this is a detective control, and its job is to make sure the
    operator finds out."""
    outcome = {"transcript": [_turn("agent", "నమస్తే! మా కొత్త కోర్సు గురించి చెప్తాను.")]}
    assert call_routes._spoken_disclosure_ok(outcome) is False


def test_only_the_first_agent_turn_counts():
    """Disclosing in turn three is not disclosing. The rule is first line."""
    outcome = {"transcript": [
        _turn("agent", "నమస్తే! మా కొత్త కోర్సు గురించి చెప్తాను."),
        _turn("agent", "ఇది కృత్రిమ మేధ ద్వారా చేసే కాల్."),
    ]}
    assert call_routes._spoken_disclosure_ok(outcome) is False


def test_a_call_with_no_agent_turns_is_not_flagged():
    """No speech means no call worth judging — the lead heard nothing, so
    there is no disclosure failure to report, just a failed call."""
    assert call_routes._spoken_disclosure_ok({"transcript": []}) is True
```

- [ ] **Step 2: Run to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/unit/test_call_routes.py -q -k disclosure`
Expected: FAIL — no attribute `_spoken_disclosure_ok`.

- [ ] **Step 3: Implement**

In `app/telephony/call_routes.py`:

```python
from app.compliance.disclosure import has_ai_disclosure
```

```python
def _spoken_disclosure_ok(outcome: dict) -> bool:
    """Whether the agent's FIRST spoken turn disclosed that it is an AI.

    campaigns.script is validated at creation, but on the OpenAI backend the
    operator may write English and the model renders it into Telugu — so the
    validated text is not the spoken text. This checks what was actually said.

    A call where the agent never spoke returns True: the lead heard nothing,
    so there is no disclosure failure to report, only a failed call.
    """
    for turn in outcome.get("transcript") or []:
        if turn.role == "agent":
            return has_ai_disclosure(turn.text)
    return True
```

and call it from `_finalise_call`, after the call is recorded:

```python
    if not _spoken_disclosure_ok(outcome):
        # Loud on purpose. This is a compliance failure on a call that has
        # already happened to a real person — it cannot be prevented here,
        # only surfaced, and it must never be silent.
        logger.error(
            f"[compliance] lead={lead.lead_id} the agent's FIRST SPOKEN LINE did "
            "not disclose AI. India telecom rules require it. Review this "
            "campaign's script and the agent prompt before dialling more leads."
        )
```

- [ ] **Step 4: Run, lint, commit**

```bash
./.venv/Scripts/python.exe -m pytest -m "not integration"
./.venv/Scripts/python.exe -m ruff check .
git add app/telephony/call_routes.py tests/unit/test_call_routes.py
git commit -m "Check the AI disclosure that was actually spoken, not just the one validated"
```

---

### Task 7: Surface the backend in the dashboard

**Files:**
- Modify: `frontend/lib/api.ts`, `frontend/app/campaigns/page.tsx`, `frontend/app/campaigns/[id]/page.tsx`

- [ ] **Step 1: Add the mapping to `api.ts`**

```typescript
/** Mirrors app/languages.py's _BACKEND_BY_TOKEN. Telugu and Tinglish run on
 * OpenAI Realtime because ElevenLabs Agents does not offer Telugu at all. */
export function backendFor(language: string): "ElevenLabs" | "OpenAI Realtime" {
  return language === "te" || language === "tinglish" ? "OpenAI Realtime" : "ElevenLabs";
}
```

- [ ] **Step 2: Show it**

In the campaigns table, add a `<Th>Voice</Th>` after Language and a matching cell:

```tsx
                  <Td className="text-xs text-neutral-500 dark:text-neutral-400">
                    {backendFor(c.language)}
                  </Td>
```

Bump the two `TableMessageRow` `colSpan={6}` values to `colSpan={7}`.

On the campaign detail page, add next to the language badge:

```tsx
          <Badge tone="neutral">{backendFor(campaign.language)}</Badge>
```

- [ ] **Step 3: Update the create-form hint**

The Language field's hint should tell the truth about what changed:

```tsx
            hint="Every lead in this file is called in this language. A Language column in the file overrides it for that row. Telugu and Tinglish run on a different voice engine (OpenAI Realtime) because ElevenLabs does not support Telugu."
```

- [ ] **Step 4: Type-check and commit**

```bash
cd frontend && npx tsc --noEmit && cd ..
git add frontend/lib/api.ts frontend/app/campaigns/page.tsx "frontend/app/campaigns/[id]/page.tsx"
git commit -m "Show which voice backend a campaign runs on"
```

---

### MILESTONE A GATE — live one-way Telugu call

**Do not start Milestone B until this passes.**

- [ ] Bring the tunnel up (`scripts/resolve_public_url.sh`) and confirm `PUBLIC_BASE_URL` is reachable.
- [ ] Create a one-way Telugu campaign with one lead — your own number. The script must pass `has_ai_disclosure()`; write it in Telugu with `కృత్రిమ మేధ` or `ఆటోమేటెడ్ కాల్` in the first sentence.
- [ ] Confirm an English campaign still dials correctly first. If the ElevenLabs path regressed, stop.
- [ ] Take the Telugu call. Judge: **is the Telugu intelligible?** Is the disclosure the first thing said? Does it hang up on its own rather than running to `CALL_MAX_DURATION_S`?
- [ ] Check the logs for `[compliance]` — if it fired, the model dropped the disclosure and the prompt needs strengthening before any list is dialled.
- [ ] Measure time-to-first-audio. If it is unacceptable, that is a finding worth acting on before building two-way on the same foundation.
- [ ] Record what you found in this plan file under a "Milestone A results" heading, and commit.

---

# MILESTONE B — Two-way Telugu with RAG (Tasks 8–10)

Only start once Milestone A's gate has passed.

**A deliberate note on detail level.** Milestone A's tasks carry complete code because
nothing about them is contingent. Milestone B's carry complete tests and complete
interface decisions, but two implementation details are specified rather than written
out — `PlivoCall.played_ms()` and `openai_prompts.two_way_instructions()`. That is
because Milestone A's live call will inform both: the amount of audio actually delivered
before barge-in depends on real network timing, and the two-way persona depends on how
the one-way persona actually sounds to a Telugu speaker. Writing that code now would be
guessing dressed up as a plan. Fill it in from the Milestone A results, not from here.

---

### Task 8: The two-way conversation path

**Files:**
- Modify: `app/telephony/openai_bridge.py`, `app/telephony/openai_prompts.py`
- Test: `tests/unit/test_openai_bridge.py`

**What changes from one-way:** turn detection on, the lead's audio forwarded (`PlivoCall.read_events` already does this for `one_way=False`), lead transcripts captured, and barge-in handled.

**The bug we must NOT inherit.** The sibling never sends `conversation.item.truncate` on its WebSocket paths, and its own `ARCHITECTURE.md:268-276` flags this as unresolved: when a lead barges in, the model still believes it said everything it generated, while the lead only heard what had played. The conversation then proceeds from a false premise. We must send it.

- [ ] **Step 1: Write the failing tests**

```python
async def test_a_two_way_session_enables_turn_detection(bridged):
    oa = _FakeOpenAIWS([_audio_delta()])
    bridged(oa)
    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False),
        timeout=5,
    )
    td = _session(oa)["session"]["audio"]["input"]["turn_detection"]
    assert td["type"] == "server_vad"


async def test_a_two_way_session_pins_the_transcription_language(bridged):
    """On 8kHz phone audio, auto-detection was observed hearing Telugu as
    Croatian or Urdu. The lead is then transcribed as nonsense and the model
    answers nonsense. Pinning the language is the fix."""
    oa = _FakeOpenAIWS([_audio_delta()])
    bridged(oa)
    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False),
        timeout=5,
    )
    tx = _session(oa)["session"]["audio"]["input"]["transcription"]
    assert tx["language"] == "te"


async def test_the_leads_words_are_recorded_as_lead_turns(bridged):
    oa = _FakeOpenAIWS([
        json.dumps({"type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "ఫీజు ఎంత?"}),
        _agent_transcript("Fees are..."), _response_done(),
    ])
    bridged(oa)
    outcome = await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False),
        timeout=5,
    )
    roles = [(t.role, t.text) for t in outcome["transcript"]]
    assert ("lead", "ఫీజు ఎంత?") in roles


async def test_barge_in_truncates_the_models_belief_about_what_it_said(bridged):
    """REGRESSION for a bug the source material has and we must not inherit.
    Without conversation.item.truncate the model believes it said everything
    it generated, while the lead only heard what had played — every later
    turn then builds on something the lead never heard."""
    oa = _FakeOpenAIWS([
        _audio_delta(),
        json.dumps({"type": "input_audio_buffer.speech_started"}),
    ])
    bridged(oa)
    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False),
        timeout=5,
    )
    sent = [json.loads(s).get("type") for s in oa.sent]
    assert "conversation.item.truncate" in sent
    assert "response.cancel" in sent
```

- [ ] **Step 2: Run to verify they fail, then implement**

In `_session_update`, replace the unconditional `turn_detection = None`:

```python
    if one_way:
        # Never auto-respond on a call that isn't listening.
        input_cfg["turn_detection"] = None
    else:
        transcription: dict = {"model": OPENAI_REALTIME_STT_MODEL}
        if OPENAI_REALTIME_STT_LANGUAGE:
            # See config: 8kHz auto-detection mis-hears Telugu badly.
            transcription["language"] = OPENAI_REALTIME_STT_LANGUAGE
        input_cfg["transcription"] = transcription
        input_cfg["turn_detection"] = {
            "type": "server_vad",
            "threshold": OPENAI_REALTIME_VAD_THRESHOLD,
            "prefix_padding_ms": 300,
            "silence_duration_ms": max(OPENAI_REALTIME_SILENCE_MS, 600),
        }
```

and use `openai_prompts.two_way_instructions(...)` when `one_way` is False.

In `from_openai`, add the lead-transcript and barge-in handlers:

```python
                        elif etype == "conversation.item.input_audio_transcription.completed":
                            said = (event.get("transcript") or "").strip()
                            if said:
                                turns.append(TranscriptTurn(role="lead", text=said))
                        elif etype == "input_audio_buffer.speech_started":
                            # Barge-in. Honour the greeting grace window, then
                            # drop queued audio AND tell the model how much of
                            # its last message the lead actually heard.
                            if await call.interrupt():
                                await oa.send(json.dumps({"type": "response.cancel"}))
                                await oa.send(json.dumps({
                                    "type": "conversation.item.truncate",
                                    "item_id": last_item_id,
                                    "content_index": 0,
                                    "audio_end_ms": call.played_ms(),
                                }))
```

This needs two things:
- `last_item_id`: track the current response's item id from `response.created` / audio delta events (`event.get("item_id")`).
- `PlivoCall.played_ms()`: how many milliseconds of the current response actually reached the lead. Add it to `plivo_stream.py` — it is Plivo-side knowledge, derived from the same `play_end` accounting the watchdog uses, and belongs there rather than in a backend.

Add `two_way_instructions(lead_name, script)` to `openai_prompts.py` — same `_DISCLOSURE_RULE` and `_LANGUAGE_RULE`, but framing the script as the GOAL of a conversation rather than a message to deliver, and instructing the model to answer course questions only from the search tool.

- [ ] **Step 3: Run, lint, commit**

```bash
git add app/telephony/openai_bridge.py app/telephony/openai_prompts.py app/telephony/plivo_stream.py tests/unit/test_openai_bridge.py tests/unit/test_plivo_stream.py
git commit -m "Hold a two-way Telugu conversation, and fix barge-in truncation"
```

---

### Task 9: RAG over course documents

**Files:**
- Modify: `app/telephony/openai_bridge.py`
- Test: `tests/unit/test_openai_bridge.py`

**The hard rule:** `search_relevant()` is the ONLY function the live tool may call. `search_permissive()` must never be reachable from here. `search_relevant(query) -> str` already returns speakable text and returns `NO_MATERIAL_NOTE` below `RAG_MIN_SCORE`, so the tool always has something to say.

Called in-process rather than over HTTP — no network hop, no `RAG_TOOL_SECRET` to manage — but through the exact same function the `/rag/search` endpoint uses.

- [ ] **Step 1: Write the failing tests**

```python
_SEARCH_TOOL_NAME = "search_course_material"


async def test_a_two_way_session_attaches_the_search_tool(bridged):
    oa = _FakeOpenAIWS([_audio_delta()])
    bridged(oa)
    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False),
        timeout=5,
    )
    names = [t["name"] for t in _session(oa)["session"].get("tools", [])]
    assert _SEARCH_TOOL_NAME in names


async def test_a_tool_call_is_answered_from_search_relevant(bridged, monkeypatch):
    """The hard rule: the live tool calls search_relevant and nothing else.
    search_permissive ignores the relevance floor and would let the agent
    recite irrelevant course text to a lead as though it were an answer."""
    seen = {}

    async def fake_relevant(query, *a, **kw):
        seen["query"] = query
        return "The fee is 25,000 rupees."

    monkeypatch.setattr(openai_bridge, "search_relevant", fake_relevant)

    def forbidden(*a, **kw):
        raise AssertionError("search_permissive must never be reachable live")

    monkeypatch.setattr(openai_bridge, "search_permissive", forbidden, raising=False)

    oa = _FakeOpenAIWS([json.dumps({
        "type": "response.done",
        "response": {"output": [{
            "type": "function_call", "name": _SEARCH_TOOL_NAME,
            "call_id": "call_1", "arguments": json.dumps({"query": "fees?"}),
        }]},
    })])
    bridged(oa)
    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=False),
        timeout=5,
    )

    assert seen["query"] == "fees?"
    outputs = [json.loads(s) for s in oa.sent]
    item = next(o for o in outputs
                if o.get("type") == "conversation.item.create")
    assert "25,000" in json.dumps(item)
    assert any(o.get("type") == "response.create" for o in outputs), (
        "the model must be prompted to speak the tool result"
    )


async def test_a_one_way_call_gets_no_tools(bridged):
    """Nothing to search when nobody can ask."""
    oa = _FakeOpenAIWS([_audio_delta()])
    bridged(oa)
    await asyncio.wait_for(
        openai_bridge.bridge(_FakePlivoWS(), agent_id="x", lead_id="l",
                             language="te", one_way=True),
        timeout=5,
    )
    assert "tools" not in _session(oa)["session"]
```

- [ ] **Step 2: Implement**

Add to `openai_bridge.py`:

```python
from app.rag.search import search_relevant

_SEARCH_TOOL = {
    "type": "function",
    "name": "search_course_material",
    "description": (
        "Search Digital Brolly's course documents for material relevant to the "
        "lead's question. Call this for every question about courses, fees, "
        "timings, batches or placement — never answer those from memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The lead's question, as a concise search query in English.",
            },
        },
        "required": ["query"],
    },
}
```

In `_session_update`, when not one-way: `session["tools"] = [_SEARCH_TOOL]` and `session["tool_choice"] = "auto"`.

In `from_openai`'s `response.done` handler, run any function calls:

```python
                            for item in (event.get("response") or {}).get("output") or []:
                                if item.get("type") != "function_call":
                                    continue
                                if item.get("name") != _SEARCH_TOOL["name"]:
                                    continue
                                try:
                                    args = json.loads(item.get("arguments") or "{}")
                                except (ValueError, TypeError):
                                    args = {}
                                # search_relevant, never search_permissive —
                                # CLAUDE.md's hard rule. It applies the
                                # relevance floor and always returns something
                                # speakable, so the agent is never left
                                # tool-calling with nothing to say.
                                answer = await search_relevant(str(args.get("query") or ""))
                                await oa.send(json.dumps({
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "function_call_output",
                                        "call_id": item.get("call_id"),
                                        "output": answer,
                                    },
                                }))
                                await oa.send(json.dumps({"type": "response.create"}))
```

- [ ] **Step 3: Run, lint, commit**

```bash
git add app/telephony/openai_bridge.py tests/unit/test_openai_bridge.py
git commit -m "Answer Telugu course questions from the RAG documents"
```

---

### Task 10: Two-way live verification

- [ ] Full suite green, `ruff check .` clean.
- [ ] Create a two-way Telugu campaign with your own number.
- [ ] Confirm on the call: the disclosure is the first thing said; it understands a Telugu question (check the stored transcript, not just your ears); it answers **from the course documents** and says it doesn't know rather than inventing when asked something outside them.
- [ ] **Barge-in:** interrupt it mid-sentence, then ask about what it was saying. If it behaves as though it finished, `conversation.item.truncate` is not working.
- [ ] Ask something off-topic (the weather). It must decline rather than answer from general knowledge.
- [ ] Check `[compliance]` did not fire.
- [ ] Record results in this file and commit.

---

## Risks

1. **Telugu voice quality.** Every OpenAI Realtime voice is English-first; there is no Telugu-native voice. The owner has heard the sibling's output and accepted it, but it remains the single biggest product risk and it is a voice-asset problem no code change fixes. `OPENAI_REALTIME_VOICE` exists so it can be changed without a deploy.
2. **Two providers, two failure modes.** English/Hindi outages and Telugu outages will now look different and need diagnosing differently. The backend column in the dashboard is the mitigation.
3. **Cost and latency are unmeasured** on the OpenAI path for this project's call profile. Measure at Milestone A before dialling a list.
4. **The disclosure detective control reports, it cannot prevent.** A dropped disclosure is found after the call happened. If Milestone A's gate shows `[compliance]` firing at all, treat English-script rendering as unsafe and require Telugu-written scripts before dialling any real list.
