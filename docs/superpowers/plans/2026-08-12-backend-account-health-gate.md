# Backend Account-Health Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the dialler from ever placing a call on a backend account
already known to be broken — ElevenLabs `past_due` billing, or a Sarvam
account out of credits — instead of ringing a real lead's phone into
guaranteed failure.

**Architecture:** Two independent gates, one per backend, both wired into
`app/telephony/preflight.py` (the actual dial-time check, run once per
campaign_tick batch) and surfaced on the `/admin/readiness` dashboard.
ElevenLabs gets a **proactive** check: `elevenlabs_client.py` gains a
Redis-cached `cached_subscription_status()` that both preflight and the
dashboard call, so an account status check costs at most one real API call
per 60s window regardless of how many places ask. Sarvam has no
programmatic balance API, so it gets a **reactive** circuit breaker: a new
`app/telephony/sarvam_circuit_breaker.py` module trips when a live call's
STT or TTS socket reports "insufficient credits", and stays tripped —
manual-clear-only, via a new admin endpoint — until an operator confirms
credits are topped up.

**Tech Stack:** FastAPI, Redis (`redis.asyncio`), pytest + pytest-asyncio
(`asyncio_mode = auto`), no new dependencies.

## Global Constraints

- Redis keys follow the existing `namespace:name` convention
  (`worker:heartbeat`, `sheets:last_sync`, `readiness:{key}`) — the new keys
  are `sarvam:circuit_breaker` and `elevenlabs:subscription_status`.
- The billing cache TTL is 60s, matching `_READINESS_CACHE_TTL_S` in
  `app/admin/system.py` — not a new constant to tune, an existing one to
  reuse in spirit.
- The circuit breaker key has **no TTL** — manual-clear-only is a deliberate
  design decision (see the design doc's Non-goals), not an oversight.
- `preflight()` must never raise — every new check follows the existing
  fail-closed-on-error precedent set by `agent_exists()` in the same
  function: an exception means "refuse to dial", not "must be fine".
- Redis-backed modules get tests run against the REAL local Redis (already
  up via `docker compose --profile dev up -d`, pointed at DB index 15 by
  `tests/conftest.py`), not a fake — see `tests/unit/test_worker_redis_primitives.py`
  for the convention this plan's new module tests follow.
- `pytest -q -m "not integration"` and `ruff check .` must stay green after
  every task.
- Full design context: `docs/superpowers/specs/2026-08-12-backend-account-health-gate-design.md`.

---

### Task 1: `sarvam_circuit_breaker.py` — the module itself

**Files:**
- Create: `app/telephony/sarvam_circuit_breaker.py`
- Test: `tests/unit/test_sarvam_circuit_breaker.py`

**Interfaces:**
- Produces: `async def trip(reason: str) -> None`, `async def tripped_reason() -> str | None`,
  `async def clear() -> None` — the only three functions later tasks call.
  Module-level constant `_BREAKER_KEY = "sarvam:circuit_breaker"` (private;
  later tasks/tests interact only through the three functions above, never
  the key name directly).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sarvam_circuit_breaker.py`:

```python
"""Unit tests for telephony/sarvam_circuit_breaker.py — run against the REAL
local Redis, same convention as test_worker_redis_primitives.py: the point
is verifying the actual trip/clear semantics, not that a mock echoes back
what it was told."""

from app import redis_client
from app.telephony import sarvam_circuit_breaker as breaker


async def _clean():
    r = redis_client.get_redis()
    await r.delete(breaker._BREAKER_KEY)


async def test_a_fresh_breaker_is_not_tripped():
    await _clean()
    assert await breaker.tripped_reason() is None


async def test_trip_then_tripped_reason_round_trip():
    await _clean()
    await breaker.trip("Insufficient credits")
    assert await breaker.tripped_reason() == "Insufficient credits"


async def test_a_second_trip_does_not_overwrite_the_first_reason():
    """First-trip-wins: several calls failing the same way in the minutes
    before anyone notices must not lose the ORIGINAL reason to a later,
    possibly less-informative one."""
    await _clean()
    await breaker.trip("Insufficient credits")
    await breaker.trip("a different later reason")
    assert await breaker.tripped_reason() == "Insufficient credits"


async def test_clear_removes_the_trip():
    await _clean()
    await breaker.trip("Insufficient credits")
    await breaker.clear()
    assert await breaker.tripped_reason() is None


async def test_clear_on_an_already_clear_breaker_is_a_noop():
    await _clean()
    await breaker.clear()  # must not raise
    assert await breaker.tripped_reason() is None
```

Note: cleanup runs as the first line of each test (`await _clean()`) rather
than an `autouse` fixture, because `app.telephony.sarvam_circuit_breaker`
does not exist yet — this whole file fails at import on Step 2, which is
what step 2 confirms. (There is no reason to switch this to an autouse
fixture afterward — either style is fine and this file only has one key to
clean up.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_sarvam_circuit_breaker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.telephony.sarvam_circuit_breaker'`
(needs `docker compose --profile dev up -d redis` running first, or a local
Redis on `localhost:6379` — the same requirement `test_worker_redis_primitives.py`
already has).

- [ ] **Step 3: Write the implementation**

Create `app/telephony/sarvam_circuit_breaker.py`:

```python
"""telephony/sarvam_circuit_breaker.py — manual-clear-only breaker for a
Sarvam account that has run out of credits.

Sarvam has no programmatic balance/credits API (only a web dashboard at
dashboard.sarvam.ai/usage), so unlike ElevenLabs' account-status check
(elevenlabs_client.cached_subscription_status) this cannot be proactive: a
call has to actually fail with the "insufficient credits" signal before
this trips. See
docs/superpowers/specs/2026-08-12-backend-account-health-gate-design.md for
the full design and why auto-recovery was deliberately rejected — Sarvam
gives no way to proactively confirm credits have been topped up, so only a
human clearing this explicitly is a reliable signal.

No TTL on the Redis key: this stays tripped until POST
/admin/system/sarvam-circuit-breaker/clear is called.
"""

from __future__ import annotations

from app import redis_client

_BREAKER_KEY = "sarvam:circuit_breaker"


async def trip(reason: str) -> None:
    """Trips the breaker with *reason*, unless already tripped.

    NX (set-if-not-exists): first-trip-wins, so the ORIGINAL failure reason
    survives repeated tripping — several calls failing the same way in the
    minutes before anyone notices — rather than being overwritten by a
    later one.
    """
    r = redis_client.get_redis()
    await r.set(_BREAKER_KEY, reason, nx=True)


async def tripped_reason() -> str | None:
    """The reason the breaker is tripped, or None if it isn't."""
    r = redis_client.get_redis()
    return await r.get(_BREAKER_KEY)


async def clear() -> None:
    """The only way out — an admin confirming the account is healthy
    again. Mirrors PATCH /admin/campaigns/{id} {"active": false}, the
    manual stopgap this replaces."""
    r = redis_client.get_redis()
    await r.delete(_BREAKER_KEY)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_sarvam_circuit_breaker.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Lint and commit**

Run: `ruff check app/telephony/sarvam_circuit_breaker.py tests/unit/test_sarvam_circuit_breaker.py`

```bash
git add app/telephony/sarvam_circuit_breaker.py tests/unit/test_sarvam_circuit_breaker.py
git commit -m "feat: add the Sarvam circuit breaker module"
```

---

### Task 2: Wire `trip()` into Sarvam's STT/TTS error handlers

**Files:**
- Modify: `app/telephony/sarvam_stt.py:19` (docstring), `:56` (import), `:305-310` (error handler)
- Modify: `app/telephony/sarvam_tts.py:50-52` (import), `:218-223` (error handler)
- Test: `tests/unit/test_sarvam_stt.py:458-468` (existing test), plus two new tests
- Test: `tests/unit/test_sarvam_tts.py:210-222` (existing test), plus two new tests

**Interfaces:**
- Consumes: `sarvam_circuit_breaker.trip(reason: str) -> None` from Task 1.
- Produces: nothing new consumed by later tasks — this task's only effect is
  that a live "insufficient credits" error now trips the breaker Task 1
  built. Also fixes a real bug: both files' error handlers previously read
  Sarvam's error message from the wrong dict key.

This task fixes a bug found while gathering context for this plan.
`sarvam_stt.py`'s error handler reads `data.get('error')`, but Sarvam
actually sends the field as `'message'` — confirmed live tonight:
`{'type': 'error', 'data': {'message': 'Insufficient credits'}}`. The
handler was silently falling back to printing the whole raw `event` dict
instead of a clean message. `sarvam_tts.py`'s equivalent handler already
reads `data.get('message')` correctly — this is what `sarvam_stt.py` should
have matched from the start. The module docstring's documented wire
protocol (line 19) has the same wrong key and needs the same fix, or it
keeps steering the next reader wrong.

Getting the message extraction right matters here specifically because
accurate extraction is what lets the "insufficient credit" substring match
below actually fire — the old code could never have detected it, no matter
what Sarvam sent, because it was reading a key Sarvam doesn't populate.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_sarvam_stt.py`, replace the existing
`test_an_error_frame_ends_the_stream_rather_than_hanging` (it currently
uses the wrong `"error"` key in its fixture — the exact key this task
fixes — so it needs to use the real key going forward) and add two new
tests directly after it:

```python
async def test_an_error_frame_ends_the_stream_rather_than_hanging(connected):
    """Same reasoning as the TTS client: a rejected connection otherwise looks
    exactly like a lead who never speaks, and costs the full call duration."""
    connected(_FakeSTTWS([json.dumps(
        {"type": "error", "data": {"message": "bad language", "code": "400"}}
    )]))

    async with sarvam_stt.SarvamSTT() as stt:
        events = [e async for e in stt.events()]

    assert events == []


async def test_an_insufficient_credits_error_trips_the_circuit_breaker(connected, monkeypatch):
    """The exact signal observed live tonight: Sarvam's STT reports the
    account is out of credits and the stream dies. Every later call on this
    backend would fail the same way until an admin clears the breaker."""
    tripped = {}

    async def fake_trip(reason):
        tripped["reason"] = reason

    monkeypatch.setattr(sarvam_stt.sarvam_circuit_breaker, "trip", fake_trip)
    connected(_FakeSTTWS([json.dumps(
        {"type": "error", "data": {"message": "Insufficient credits", "code": "402"}}
    )]))

    async with sarvam_stt.SarvamSTT() as stt:
        events = [e async for e in stt.events()]

    assert events == []
    assert tripped["reason"] == "Insufficient credits"


async def test_an_unrelated_error_does_not_trip_the_circuit_breaker(connected, monkeypatch):
    async def must_not_be_called(reason):
        raise AssertionError("an unrelated error must not trip the breaker")

    monkeypatch.setattr(sarvam_stt.sarvam_circuit_breaker, "trip", must_not_be_called)
    connected(_FakeSTTWS([json.dumps(
        {"type": "error", "data": {"message": "bad language", "code": "400"}}
    )]))

    async with sarvam_stt.SarvamSTT() as stt:
        events = [e async for e in stt.events()]

    assert events == []
```

In `tests/unit/test_sarvam_tts.py`, the existing
`test_an_error_frame_stops_the_stream_rather_than_hanging` already uses the
correct `"message"` key — leave it unchanged. Add two new tests directly
after it:

```python
async def test_an_insufficient_credits_error_trips_the_circuit_breaker(connected, monkeypatch):
    """The exact signal observed live tonight, this time from the TTS leg —
    a call can run out of credits mid-synthesis just as easily as at STT."""
    tripped = {}

    async def fake_trip(reason):
        tripped["reason"] = reason

    monkeypatch.setattr(sarvam_tts.sarvam_circuit_breaker, "trip", fake_trip)
    connected(_FakeSarvamWS([json.dumps(
        {"type": "error", "data": {"message": "Insufficient credits", "code": 402}}
    )]))

    async with sarvam_tts.SarvamTTS(language="te-IN") as tts:
        chunks = await asyncio.wait_for(tts.collect("ఒకటి"), timeout=5)

    assert chunks == []
    assert tripped["reason"] == "Insufficient credits"


async def test_an_unrelated_error_does_not_trip_the_circuit_breaker(connected, monkeypatch):
    async def must_not_be_called(reason):
        raise AssertionError("an unrelated error must not trip the breaker")

    monkeypatch.setattr(sarvam_tts.sarvam_circuit_breaker, "trip", must_not_be_called)
    connected(_FakeSarvamWS([json.dumps(
        {"type": "error", "data": {"message": "invalid speaker", "code": 400}}
    )]))

    async with sarvam_tts.SarvamTTS(language="te-IN") as tts:
        chunks = await asyncio.wait_for(tts.collect("ఒకటి"), timeout=5)

    assert chunks == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_sarvam_stt.py tests/unit/test_sarvam_tts.py -v`
Expected: the two new tests per file FAIL with `AttributeError: module
'app.telephony.sarvam_stt' has no attribute 'sarvam_circuit_breaker'` (and
the same for `sarvam_tts`) — the import doesn't exist yet. The updated
existing STT test should currently PASS even before Step 3 (changing the
fixture's key to `"message"` doesn't change today's behavior, since today's
buggy code falls back to printing the whole event either way and still
returns `[]`) — that's expected; it starts actually exercising the correct
key once Step 3 lands.

- [ ] **Step 3: Write the implementation**

In `app/telephony/sarvam_stt.py`, fix the docstring's documented wire
protocol (line 19):

```python
    <-       {"type": "error",  "data": {"message": "...", "code": "..."}}
```

Add the import after the existing `from app.telephony import ulaw` (line 56):

```python
from app.telephony import sarvam_circuit_breaker, ulaw
```

Replace the error handler (lines 305-310):

```python
            elif etype == "error":
                message = data.get("message") or event
                logger.error(
                    f"[sarvam-stt] refused to transcribe: "
                    f"{message} (code={data.get('code')})"
                )
                if isinstance(message, str) and "insufficient credit" in message.lower():
                    await sarvam_circuit_breaker.trip(message)
                return
```

In `app/telephony/sarvam_tts.py`, add the import after the existing
`from app.config import (...)` block (line 50), before `logger = logging.getLogger(__name__)`:

```python
from app.telephony import sarvam_circuit_breaker
```

Replace the error handler (lines 218-223):

```python
                elif etype == "error":
                    message = data.get("message") or event
                    logger.error(
                        f"[sarvam-tts] refused to synthesise: "
                        f"{message} (code={data.get('code')})"
                    )
                    if isinstance(message, str) and "insufficient credit" in message.lower():
                        await sarvam_circuit_breaker.trip(message)
                    return
```

The `isinstance(message, str)` guard matters: `message` falls back to
`event` (a dict) when Sarvam's frame has no `'message'` key, and calling
`.lower()` on a dict would raise instead of just skipping the trip check.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_sarvam_stt.py tests/unit/test_sarvam_tts.py -v`
Expected: PASS (all tests in both files)

- [ ] **Step 5: Lint and commit**

Run: `ruff check app/telephony/sarvam_stt.py app/telephony/sarvam_tts.py tests/unit/test_sarvam_stt.py tests/unit/test_sarvam_tts.py`

```bash
git add app/telephony/sarvam_stt.py app/telephony/sarvam_tts.py tests/unit/test_sarvam_stt.py tests/unit/test_sarvam_tts.py
git commit -m "fix: read Sarvam's error message from the right key, trip the circuit breaker on insufficient credits"
```

---

### Task 3: `elevenlabs_client.cached_subscription_status()` + `admin/system.py` delegation

**Files:**
- Modify: `app/telephony/elevenlabs_client.py:20-28` (imports), append new
  code after `subscription_status()` (currently ends at line 136)
- Modify: `app/admin/system.py:288-307` (`_GOOD_SUBSCRIPTION_STATUSES`,
  `_compute_billing`), `:375-380` (`_build_readiness`'s billing wiring)
- Test: `tests/unit/test_elevenlabs_client.py` (new tests appended)
- Test: `tests/unit/test_admin_system.py` (one new test appended; existing
  tests are NOT modified — see the note below on why they keep passing
  unchanged)

**Interfaces:**
- Produces: `async def cached_subscription_status(*, force_refresh: bool = False) -> dict`
  and module-level `GOOD_SUBSCRIPTION_STATUSES: set[str]` on
  `app.telephony.elevenlabs_client` — Task 4 imports both by name into
  `preflight.py`.

A note on why the ~15 existing tests in `test_admin_system.py` that
monkeypatch `admin_system.elevenlabs_client.subscription_status` directly
(via `_mock_all_healthy` and a few standalone tests) need **no changes**:
`admin/system.py` does `from app.telephony import elevenlabs_client`, and
`cached_subscription_status()` (defined inside `elevenlabs_client.py`)
looks up the bare name `subscription_status` in its own module's globals at
call time — which is the exact same module object `admin_system.elevenlabs_client`
refers to (Python caches modules by name; both names point at one object).
Monkeypatching `admin_system.elevenlabs_client.subscription_status` therefore
transparently changes what `cached_subscription_status()` calls too. The same
reasoning covers `redis_client`: both `elevenlabs_client.py` and `admin_system.py`
do `from app import redis_client`, so `_mock_all_healthy`'s
`monkeypatch.setattr(admin_system.redis_client, "get_redis", ...)` is
already visible from inside `cached_subscription_status()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_elevenlabs_client.py`:

```python
# ── cached_subscription_status: the shared 60s cache ─────────────────────────
# Shared by app/admin/system.py's readiness dashboard and
# app/telephony/preflight.py's dial-time gate — this is the function both
# call, so a dashboard poll and a preflight check within the same 60s window
# cost exactly one real ElevenLabs API call between them, not two.


class _FakeRedis:
    def __init__(self, store: dict | None = None):
        self.store: dict[str, str] = store or {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value
        return True


async def test_cached_subscription_status_caches_across_calls(monkeypatch):
    calls = {"n": 0}

    def counted():
        calls["n"] += 1
        return {"status": "active", "character_count": 1, "character_limit": 2}

    monkeypatch.setattr(ec, "subscription_status", counted)
    monkeypatch.setattr(ec.redis_client, "get_redis", lambda: _FakeRedis())

    first = await ec.cached_subscription_status()
    second = await ec.cached_subscription_status()

    assert calls["n"] == 1
    assert first == second == {"status": "active", "character_count": 1, "character_limit": 2}


async def test_cached_subscription_status_force_refresh_bypasses_the_cache(monkeypatch):
    calls = {"n": 0}

    def counted():
        calls["n"] += 1
        return {"status": "active", "character_count": calls["n"], "character_limit": 2}

    monkeypatch.setattr(ec, "subscription_status", counted)
    monkeypatch.setattr(ec.redis_client, "get_redis", lambda: _FakeRedis())

    await ec.cached_subscription_status()
    await ec.cached_subscription_status(force_refresh=True)

    assert calls["n"] == 2


async def test_cached_subscription_status_survives_a_malformed_cache_entry(monkeypatch):
    def fresh():
        return {"status": "active", "character_count": 1, "character_limit": 2}

    monkeypatch.setattr(ec, "subscription_status", fresh)
    monkeypatch.setattr(ec.redis_client, "get_redis",
                        lambda: _FakeRedis({ec._SUBSCRIPTION_CACHE_KEY: "not-json"}))

    result = await ec.cached_subscription_status()

    assert result == {"status": "active", "character_count": 1, "character_limit": 2}
```

Append to `tests/unit/test_admin_system.py`, in the `# ── readiness: caching ──`
section near the other caching tests:

```python
async def test_billing_check_reuses_a_cache_warmed_by_elevenlabs_client_directly(client, monkeypatch):
    """The whole point of moving billing's caching into elevenlabs_client.py:
    preflight.py and the readiness dashboard must not each pay for their own
    ElevenLabs API round-trip within the same cache window."""
    redis = _FakeRedis()
    _mock_all_healthy(monkeypatch, redis=redis)
    calls = {"n": 0}

    def counted():
        calls["n"] += 1
        return {"status": "active", "character_count": 1, "character_limit": 2}

    monkeypatch.setattr(admin_system.elevenlabs_client, "subscription_status", counted)

    # Simulates preflight.py having already warmed the cache moments earlier.
    await admin_system.elevenlabs_client.cached_subscription_status()
    assert calls["n"] == 1

    r = client.get("/admin/readiness")

    assert calls["n"] == 1, "the readiness dashboard must reuse the warm cache"
    billing_check = next(c for c in r.json()["checks"] if c["key"] == "billing")
    assert billing_check["status"] == "good"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_elevenlabs_client.py tests/unit/test_admin_system.py -v`
Expected: the three new `test_elevenlabs_client.py` tests FAIL with
`AttributeError: module 'app.telephony.elevenlabs_client' has no attribute
'cached_subscription_status'` (and `'_SUBSCRIPTION_CACHE_KEY'` for the third
one). The new `test_admin_system.py` test fails the same way. Every
pre-existing test in both files should still PASS at this point (nothing
about them has changed yet).

- [ ] **Step 3: Write the implementation**

In `app/telephony/elevenlabs_client.py`, update the imports (lines 20-28):

```python
from __future__ import annotations

import asyncio
import json

from elevenlabs.client import ElevenLabs
from elevenlabs.core.api_error import ApiError
from elevenlabs.types.conversation_initiation_client_data_request_input import (
    ConversationInitiationClientDataRequestInput,
)

from app import redis_client
from app.config import ELEVENLABS_API_KEY
```

Add one sentence to `subscription_status()`'s existing docstring (after its
last paragraph, still inside the `"""..."""`):

```python
    Most callers should use cached_subscription_status() below instead —
    this is the uncached primitive it wraps.
```

Append after `subscription_status()` (after its current closing line, before
`agent_language_support`):

```python
GOOD_SUBSCRIPTION_STATUSES = {"active", "trialing"}

_SUBSCRIPTION_CACHE_KEY = "elevenlabs:subscription_status"
_SUBSCRIPTION_CACHE_TTL_S = 60


async def cached_subscription_status(*, force_refresh: bool = False) -> dict:
    """subscription_status(), cached in Redis for _SUBSCRIPTION_CACHE_TTL_S
    seconds.

    Shared by app/admin/system.py's readiness dashboard and
    app/telephony/preflight.py's dial-time gate, so a dashboard poll and a
    preflight check within the same window agree on the account's billing
    state without each paying for their own ElevenLabs API round-trip.

    force_refresh=True bypasses the cache — used by POST /admin/readiness/
    refresh, where "I just fixed it, tell me right now" must not wait out
    the TTL.
    """
    r = redis_client.get_redis()
    if not force_refresh:
        try:
            raw = await r.get(_SUBSCRIPTION_CACHE_KEY)
        except Exception:
            raw = None
        if raw:
            try:
                return json.loads(raw)
            except (TypeError, ValueError):
                pass  # corrupted cache entry — fall through to a fresh call
    result = await asyncio.to_thread(subscription_status)
    try:
        await r.set(_SUBSCRIPTION_CACHE_KEY, json.dumps(result), ex=_SUBSCRIPTION_CACHE_TTL_S)
    except Exception:
        pass  # best-effort cache; a write failure must not fail the call
    return result
```

In `app/admin/system.py`, remove `_GOOD_SUBSCRIPTION_STATUSES = {"active",
"trialing"}` (line 288) and replace `_compute_billing()` (lines 291-307):

```python
async def _compute_billing(*, force_refresh: bool = False) -> dict:
    try:
        sub = await elevenlabs_client.cached_subscription_status(force_refresh=force_refresh)
    except Exception as exc:
        return {"key": "billing", "status": "critical", "label": "ElevenLabs billing",
                "detail": f"Could not check: {type(exc).__name__}: {exc}"}
    status = sub["status"]
    if status in elevenlabs_client.GOOD_SUBSCRIPTION_STATUSES:
        usage = ""
        if sub.get("character_limit"):
            usage = f" ({sub['character_count']}/{sub['character_limit']} characters used)"
        return {"key": "billing", "status": "good", "label": "ElevenLabs billing",
                "detail": f"{status}{usage}"}
    return {"key": "billing", "status": "critical", "label": "ElevenLabs billing",
            "detail": f"Account status is {status!r} — no ElevenLabs conversation can run "
                      "until this is resolved (the agent WebSocket refuses the handshake "
                      "with a payment-issue error in this state)."}
```

Update `_build_readiness()`'s billing wiring (lines 375-380):

```python
    if force_refresh:
        preflight_check = await _refresh_check("preflight", _compute_preflight)
        billing_check = await _refresh_check("billing", lambda: _compute_billing(force_refresh=True))
    else:
        preflight_check = await _cached_check("preflight", _compute_preflight)
        billing_check = await _cached_check("billing", _compute_billing)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_elevenlabs_client.py tests/unit/test_admin_system.py -v`
Expected: PASS (every test in both files, old and new)

- [ ] **Step 5: Run the full unit suite**

Run: `pytest -q -m "not integration"`
Expected: PASS — this is the point where a mistake in the shared-module
reasoning above would show up as a cascade of `test_admin_system.py`
failures, not just the new tests.

- [ ] **Step 6: Lint and commit**

Run: `ruff check app/telephony/elevenlabs_client.py app/admin/system.py tests/unit/test_elevenlabs_client.py tests/unit/test_admin_system.py`

```bash
git add app/telephony/elevenlabs_client.py app/admin/system.py tests/unit/test_elevenlabs_client.py tests/unit/test_admin_system.py
git commit -m "refactor: move ElevenLabs billing caching into elevenlabs_client, share it with preflight"
```

---

### Task 4: `preflight.py` gains both dial-time checks

**Files:**
- Modify: `app/telephony/preflight.py:39-40` (imports), `:185` (insert
  Sarvam breaker check), `:215-217` (insert ElevenLabs billing check)
- Test: `tests/unit/test_preflight.py:42-64` (`_configured()` helper,
  extended), plus new tests appended near the relevant existing sections

**Interfaces:**
- Consumes: `sarvam_circuit_breaker.tripped_reason()` (Task 1),
  `cached_subscription_status()` and `GOOD_SUBSCRIPTION_STATUSES` (Task 3).
- Produces: nothing new consumed by later tasks.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_preflight.py`, extend `_configured()` (lines 42-64) so
every existing test keeps its current "everything passes" default once the
two new checks exist — add these lines at the end of the function, after
`monkeypatch.setattr(pf.httpx, "AsyncClient", lambda **kw: _FakeAsyncClient())`:

```python
    async def good_billing(*, force_refresh: bool = False):
        return {"status": "active", "character_count": 1, "character_limit": 2}
    monkeypatch.setattr(pf, "cached_subscription_status", good_billing)

    async def not_tripped():
        return None
    monkeypatch.setattr(pf.sarvam_circuit_breaker, "tripped_reason", not_tripped)
```

Add these tests directly after `test_hindi_still_runs_the_elevenlabs_checks`
(around line 372):

```python
# ── ElevenLabs account billing must be healthy before any ElevenLabs call ────
#
# A `past_due` account refuses the agent WebSocket handshake with a
# payment-issue error — verified live 2026-07. Checked here, not just on the
# readiness dashboard, so a dead account stops dialling instead of ringing
# every lead into instant silence.

async def test_unhealthy_elevenlabs_billing_refuses_to_dial(monkeypatch):
    _configured(monkeypatch)

    async def past_due(*, force_refresh=False):
        return {"status": "past_due", "character_count": 1, "character_limit": 2}

    monkeypatch.setattr(pf, "cached_subscription_status", past_due)

    reason = await pf.preflight("agent_1")
    assert reason is not None
    assert "past_due" in reason


async def test_healthy_elevenlabs_billing_does_not_block(monkeypatch):
    _configured(monkeypatch)
    assert await pf.preflight("agent_1") is None  # good billing is _configured()'s default


async def test_an_unreachable_billing_check_refuses_rather_than_dials_blind(monkeypatch):
    """Same discipline as agent_exists() above: a check that cannot answer
    must not be read as "must be fine"."""
    _configured(monkeypatch)

    async def boom(*, force_refresh=False):
        raise RuntimeError("ElevenLabs API down")

    monkeypatch.setattr(pf, "cached_subscription_status", boom)

    reason = await pf.preflight("agent_1")
    assert reason is not None
    assert "Could not check" in reason


async def test_billing_is_not_checked_for_a_sarvam_backed_call(monkeypatch):
    _configured(monkeypatch)
    monkeypatch.setattr(pf, "SARVAM_API_KEY", "sk-test")

    async def must_not_be_called(*, force_refresh=False):
        raise AssertionError("must not check ElevenLabs billing for a Sarvam-backed call")

    monkeypatch.setattr(pf, "cached_subscription_status", must_not_be_called)
    assert await pf.preflight("agent_1", "te") is None


# ── the Sarvam circuit breaker must be clear before any Sarvam call ─────────
#
# Sarvam has no programmatic credits/balance API, so this is reactive: a
# call already failed with "insufficient credits" and tripped it, and every
# later Sarvam-backed dial must refuse until an admin clears it.

async def test_a_tripped_sarvam_breaker_refuses_to_dial(monkeypatch):
    _configured(monkeypatch)
    monkeypatch.setattr(pf, "SARVAM_API_KEY", "sk-test")

    async def tripped():
        return "Insufficient credits"

    monkeypatch.setattr(pf.sarvam_circuit_breaker, "tripped_reason", tripped)

    reason = await pf.preflight("agent_1", "te")
    assert reason is not None
    assert "Insufficient credits" in reason


async def test_a_clear_sarvam_breaker_does_not_block(monkeypatch):
    _configured(monkeypatch)
    monkeypatch.setattr(pf, "SARVAM_API_KEY", "sk-test")
    assert await pf.preflight("agent_1", "te") is None  # not tripped is _configured()'s default


async def test_the_sarvam_breaker_is_not_checked_for_an_elevenlabs_backed_call(monkeypatch):
    _configured(monkeypatch)

    async def must_not_be_called():
        raise AssertionError("must not check the Sarvam breaker for an ElevenLabs call")

    monkeypatch.setattr(pf.sarvam_circuit_breaker, "tripped_reason", must_not_be_called)
    assert await pf.preflight("agent_1") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_preflight.py -v`
Expected: the `_configured()` change itself fails immediately —
`AttributeError: module 'app.telephony.preflight' has no attribute
'cached_subscription_status'` — which every single test in the file
inherits (since every test calls `_configured()`). This is expected: Step 3
makes the whole file pass at once, not incrementally.

- [ ] **Step 3: Write the implementation**

In `app/telephony/preflight.py`, update the imports (lines 39-40):

```python
from app.telephony import conversation_llm, sarvam_circuit_breaker
from app.telephony.elevenlabs_client import (
    GOOD_SUBSCRIPTION_STATUSES,
    agent_exists,
    agent_language_support,
    cached_subscription_status,
)
```

Insert the Sarvam breaker check right after the existing `SARVAM_API_KEY`
refusal (after line 185's closing `)`, before the `# A two-way call...`
comment on line 186):

```python
        breaker_reason = await sarvam_circuit_breaker.tripped_reason()
        if breaker_reason:
            return (
                f"Sarvam reported {breaker_reason!r} on a recent call, and dialling "
                "on this backend has been paused rather than risk repeating it on "
                "every other lead. Check credits at dashboard.sarvam.ai/usage, top "
                "up if needed, then clear it: POST "
                "/admin/system/sarvam-circuit-breaker/clear."
            )
```

Insert the ElevenLabs billing check after the OPENAI_REALTIME branch's
`return None` (line 215) and before the `# Nothing overridden` comment
(line 217) — this ordering matters: it must run before the plain-`'auto'`
early return just below it, since billing applies to every ElevenLabs call
regardless of language/voice override, including the common case of an
`'auto'` campaign with no forced voice:

```python
    # ElevenLabs billing applies to every call this backend carries,
    # regardless of language/voice override — checked here, before the
    # 'auto'-with-no-override early return below, so it can't be skipped by
    # the very campaigns most likely to use it (a plain 'auto' campaign with
    # no forced voice).
    try:
        sub = await cached_subscription_status()
    except Exception as exc:
        return (f"Could not check ElevenLabs' billing status ({type(exc).__name__}: "
                f"{exc}). Not dialling while the account's state is unknown.")
    if sub["status"] not in GOOD_SUBSCRIPTION_STATUSES:
        return (
            f"ElevenLabs account status is {sub['status']!r} — no conversation can "
            "run until this is resolved at elevenlabs.io (almost always a payment "
            "issue). The agent WebSocket refuses the handshake in this state, so "
            "every call would ring, connect, and go silent. Resolve billing, then "
            "try again."
        )

```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_preflight.py -v`
Expected: PASS (every test in the file, old and new)

- [ ] **Step 5: Run the full unit suite**

Run: `pytest -q -m "not integration"`
Expected: PASS

- [ ] **Step 6: Lint and commit**

Run: `ruff check app/telephony/preflight.py tests/unit/test_preflight.py`

```bash
git add app/telephony/preflight.py tests/unit/test_preflight.py
git commit -m "feat: refuse to dial on a broken ElevenLabs billing account or a tripped Sarvam breaker"
```

---

### Task 5: Admin clear endpoint + readiness dashboard visibility

**Files:**
- Modify: `app/admin/system.py:35-37` (imports), add
  `_check_sarvam_circuit_breaker()` near the other cheap checks (after
  `_check_calling_hours`, around line 108), wire it into `_build_readiness()`
  (lines 370-373), add the new router endpoint (near the bottom of the
  readiness section, after `refresh_readiness()` around line 434)
- Test: `tests/unit/test_admin_system.py` (new tests appended)

**Interfaces:**
- Consumes: `sarvam_circuit_breaker.trip/tripped_reason/clear` (Task 1).
- Produces: `POST /admin/system/sarvam-circuit-breaker/clear` — the
  operator-facing recovery action referenced in Task 4's refusal message
  and Task 2's design.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_admin_system.py`, near the other readiness tests:

```python
# ── readiness: the Sarvam circuit breaker ────────────────────────────────────

async def test_readiness_can_dial_false_when_sarvam_circuit_breaker_is_tripped(client, monkeypatch):
    """The reactive half of the design: a call that already failed with
    Sarvam's "insufficient credits" signal must stop every later
    Sarvam-backed dial until an admin clears it."""
    redis = _FakeRedis()
    _mock_all_healthy(monkeypatch, redis=redis)
    await admin_system.sarvam_circuit_breaker.trip("Insufficient credits")

    r = client.get("/admin/readiness")

    body = r.json()
    assert body["can_dial"] is False
    breaker_check = next(c for c in body["checks"] if c["key"] == "sarvam_circuit_breaker")
    assert breaker_check["status"] == "critical"
    assert "Insufficient credits" in breaker_check["detail"]


async def test_readiness_sarvam_circuit_breaker_good_when_not_tripped(client, monkeypatch):
    _mock_all_healthy(monkeypatch)

    r = client.get("/admin/readiness")

    breaker_check = next(c for c in r.json()["checks"] if c["key"] == "sarvam_circuit_breaker")
    assert breaker_check["status"] == "good"


async def test_clear_sarvam_circuit_breaker_endpoint_clears_it(client, monkeypatch):
    redis = _FakeRedis()
    monkeypatch.setattr(admin_system.redis_client, "get_redis", lambda: redis)
    await admin_system.sarvam_circuit_breaker.trip("Insufficient credits")

    r = client.post("/admin/system/sarvam-circuit-breaker/clear")

    assert r.status_code == 200
    assert await admin_system.sarvam_circuit_breaker.tripped_reason() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_admin_system.py -v`
Expected: the three new tests FAIL — `AttributeError: module
'app.admin.system' has no attribute 'sarvam_circuit_breaker'` for the first
and third, and a 404 (missing `sarvam_circuit_breaker` key in `checks`) for
the second, once the import exists. Run once now to confirm the
`AttributeError` shape, then again after Step 3's import line to confirm
the check-shape failures before the full implementation lands.

- [ ] **Step 3: Write the implementation**

In `app/admin/system.py`, add the import (alongside the existing
`app.telephony` imports, lines 35-37):

```python
from app.telephony import elevenlabs_client
from app.telephony import preflight as preflight_module
from app.telephony import sarvam_circuit_breaker
from app.telephony import worker as worker_module
```

Add a new check function after `_check_calling_hours()` (after its closing
line, before `_REQUIRED_FOR_DIALING`):

```python
async def _check_sarvam_circuit_breaker() -> dict:
    try:
        reason = await sarvam_circuit_breaker.tripped_reason()
    except Exception as exc:
        return {"key": "sarvam_circuit_breaker", "status": "critical",
                "label": "Sarvam circuit breaker",
                "detail": f"Could not check: {type(exc).__name__}: {exc}"}
    if reason:
        return {"key": "sarvam_circuit_breaker", "status": "critical",
                "label": "Sarvam circuit breaker",
                "detail": f"Tripped: {reason}. Check credits at "
                          "dashboard.sarvam.ai/usage, then clear it: POST "
                          "/admin/system/sarvam-circuit-breaker/clear."}
    return {"key": "sarvam_circuit_breaker", "status": "good",
            "label": "Sarvam circuit breaker", "detail": "Not tripped"}
```

In `_build_readiness()`, add it to the cheap-checks block (lines 363-373):

```python
    db_check = await _check_database()
    redis_check = await _check_redis()
    worker_check = await _check_worker()
    hours_check = _check_calling_hours()
    config_check = _check_config()
    queue_check = await _check_queue_depth(worker_ok=worker_check["status"] == "good")
    breaker_check = await _check_sarvam_circuit_breaker()

    cheap = [db_check, redis_check, worker_check, hours_check, config_check,
             queue_check, breaker_check]
    for c in cheap:
        c["cached"] = False
        c["checked_at"] = now_iso
```

Add the clear endpoint after `refresh_readiness()` (after its closing line,
before the `# ── Stats ──` section header):

```python
@router.post("/system/sarvam-circuit-breaker/clear")
async def clear_sarvam_circuit_breaker() -> dict:
    """Manual-only recovery — Sarvam gives no way to confirm credits have
    been topped up, so a human saying "this is fixed now" is the only
    reliable signal. Mirrors PATCH /admin/campaigns/{id} {"active": false},
    the stopgap this replaces."""
    await sarvam_circuit_breaker.clear()
    return {"cleared": True}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_admin_system.py -v`
Expected: PASS (every test in the file, old and new — including
`test_readiness_can_dial_true_when_everything_passes`, which asserts every
check's status is `"good"` and must now also see the new
`sarvam_circuit_breaker` check land as `"good"`)

- [ ] **Step 5: Run the full unit suite and lint**

Run: `pytest -q -m "not integration"`
Run: `ruff check app/admin/system.py tests/unit/test_admin_system.py`
Expected: both clean

- [ ] **Step 6: Commit**

```bash
git add app/admin/system.py tests/unit/test_admin_system.py
git commit -m "feat: surface the Sarvam circuit breaker on the readiness dashboard, add a manual clear endpoint"
```

---

### Task 6: Deploy and manually verify (operational, no code changes)

This task has no test steps of its own — it exercises Tasks 1-5 against
the real running system, per the design doc's Testing/verification section.

- [ ] **Step 1: Rebuild and redeploy**

```bash
docker compose build backend
docker compose up -d --force-recreate backend
```

(Per this project's own operational note: `--force-recreate` alone reuses
the existing image and would silently skip all of this task's code
changes — the `build` step is required first.)

- [ ] **Step 2: Confirm the readiness dashboard shows the new check**

```bash
curl -s http://localhost:8000/admin/readiness | python -m json.tool
```

Expected: a `sarvam_circuit_breaker` entry in `checks`, status `"good"`
(assuming no breaker is currently tripped from earlier tonight's manual
work — Sarvam's credits are still exhausted, but nothing tripped this new
Redis key before it existed).

- [ ] **Step 3: Exercise the clear endpoint against the real Redis**

Trip it manually to confirm the whole loop end-to-end:

```bash
docker compose exec redis redis-cli SET sarvam:circuit_breaker "manual verification"
curl -s http://localhost:8000/admin/readiness | python -m json.tool   # sarvam_circuit_breaker: critical, can_dial: false
curl -s -X POST http://localhost:8000/admin/system/sarvam-circuit-breaker/clear
curl -s http://localhost:8000/admin/readiness | python -m json.tool   # sarvam_circuit_breaker: good again
```

- [ ] **Step 4: Deliberately confirm re-activating a campaign against a still-tripped breaker is refused**

This is the live proof the design exists for — the exact scenario tonight's
manual campaign pause was covering by hand, now covered by the code
instead. With Sarvam credits still genuinely exhausted (do not clear the
breaker for this step):

```bash
docker compose exec redis redis-cli SET sarvam:circuit_breaker "Insufficient credits"
# Re-activate one of the Sarvam campaigns paused earlier tonight:
curl -s -X PATCH http://localhost:8000/admin/campaigns/<campaign_id> \
  -H "Content-Type: application/json" -d '{"active": true}'
```

Expected: the campaign is active again in the database, but the next
`campaign_tick` must NOT place a real call — confirm via
`docker compose logs -f backend` around the next tick interval that
`preflight()` refuses with the "Sarvam reported..." message, and the
lead's status does not move past `queued`/`pending` into `calling`. Then
re-run Step 3's clear sequence (or leave it tripped, operator's call) and
deactivate the campaign again if credits genuinely have not been topped up
yet — this step is a verification, not a real dial-out.

- [ ] **Step 5: Record the result**

No commit for this task (nothing to commit — it's verification only).
Report back to the owner: readiness dashboard confirmed working, clear
endpoint confirmed working, and the refusal confirmed live against a real
`campaign_tick` — plus whatever the campaign's actual reactivation state
should be left at afterward (paused, pending real Sarvam credits top-up).

---

## Self-Review

**Spec coverage** — every section of
`docs/superpowers/specs/2026-08-12-backend-account-health-gate-design.md`
maps to a task:
- "ElevenLabs: proactive check, shared cache" → Task 3 (the function) + Task 4 (preflight wiring).
- "Sarvam: reactive circuit breaker, manual clear" → Task 1 (module) + Task 2 (trip wiring) + Task 4 (preflight wiring) + Task 5 (clear endpoint).
- "Visibility" (readiness dashboard) → Task 5.
- Testing/verification section → each task's own test steps, plus Task 6 for the manual items specifically called out (clear endpoint against real Redis, deliberate re-activation against a tripped breaker).
- Non-goals (no billing fix, no auto-recovery, no OpenAI Realtime change, no general-purpose Sarvam error net) — nothing in this plan does any of these; the substring match in Task 2 is scoped to "insufficient credit" exactly as specified.

**Placeholder scan** — no TBD/TODO markers; every step has literal code, not
a description of code. The one place that could look like a placeholder —
Task 6's manual verification steps — is deliberately operational (per the
design's own "Manual" testing note) and gives exact commands and expected
outcomes rather than vague instructions.

**Type/name consistency** — traced across all five code tasks:
`sarvam_circuit_breaker.trip(reason: str) -> None` /
`tripped_reason() -> str | None` / `clear() -> None` are the exact names
Tasks 2, 4, and 5 call. `elevenlabs_client.cached_subscription_status(*,
force_refresh: bool = False) -> dict` and `GOOD_SUBSCRIPTION_STATUSES` are
the exact names Task 4 imports and Task 3's own `_compute_billing` uses.
The Redis keys (`sarvam:circuit_breaker`, `elevenlabs:subscription_status`)
are private to their owning modules and never referenced by name from
outside — Tasks 4 and 5's tests interact only through the public functions,
matching Task 1's own test file.

## Execution Handoff

Plan complete and saved to
`docs/superpowers/plans/2026-08-12-backend-account-health-gate.md`. Two
execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task,
review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using
executing-plans, batch execution with checkpoints

**Which approach?**
