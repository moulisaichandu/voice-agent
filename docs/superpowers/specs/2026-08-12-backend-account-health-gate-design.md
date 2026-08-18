# Backend account-health gate — design

## Context

Two independent things went wrong tonight, discovered only by watching real
calls fail:

1. **ElevenLabs' account is `past_due`.** This has an existing, correct
   diagnosis in `app/admin/system.py`'s readiness dashboard (`billing`
   check, calls `elevenlabs_client.subscription_status()`) — but that check
   is dashboard-only. `app/telephony/preflight.py`, the actual dial-time
   gate, never consults it. No active campaign used ElevenLabs tonight, so
   this wasn't actively burning calls, but the gap is real: the moment an
   English/Hindi/Hinglish campaign goes active again, it would be dialed
   into guaranteed failure exactly the way CLAUDE.md already treats every
   other "config combination known not to work."

2. **Sarvam's account ran out of credits mid-session**, discovered live: a
   call's STT stream reported `{'message': 'Insufficient credits'}` and
   died; the next test call closed its Sarvam connection immediately
   (`ConnectionClosedOK`) before a single turn completed — confirming this
   wasn't a one-off, it was an ongoing account state. Every one of the 4
   active campaigns tonight used Sarvam, so every scheduled dial from that
   point on would have rung a real lead's phone and failed within seconds,
   with no code anywhere refusing to try. The campaigns were paused by hand
   (`PATCH /admin/campaigns/{id}` `{"active": false}`) as an immediate
   stopgap; this design is the permanent fix.

Both are the same class of problem — CLAUDE.md's own words, already applied
to other configs: *"a config combination known not to work must refuse to
dial, not degrade into a silent call."* An unhealthy backend account is
exactly that, just discovered later than the others.

**The two backends need different mechanisms, though.** ElevenLabs exposes
`GET /v1/user/subscription` — a real account-status API, already wrapped by
`elevenlabs_client.subscription_status()`. Sarvam has no equivalent
(confirmed by checking their API reference docs — only a web dashboard at
dashboard.sarvam.ai/usage, no programmatic balance/credits endpoint). So
ElevenLabs gets a **proactive** check (ask the account's status before
dialing); Sarvam gets a **reactive** one (a call actually fails with the
specific "insufficient credits" signal, and that trips a breaker that
blocks further dials until an admin clears it).

## Non-goals

- No fix to either underlying billing problem — those are resolved by the
  owner directly (elevenlabs.io payment, Sarvam credits top-up at
  dashboard.sarvam.ai). This is purely about not dialing leads into a
  backend known to be broken.
- No automatic recovery for the Sarvam breaker. Confirmed with the owner:
  manual clear only. Sarvam gives no way to proactively confirm credits
  have been topped up, so an auto-expiring timer risks either resuming
  into the same failure (timer too short) or staying paused long after the
  account is healthy again (timer too long) — a human confirming "I topped
  up" is the only reliable signal.
- No change to OpenAI Realtime's preflight path — unaffected, different
  account entirely.
- No general-purpose "any Sarvam error trips the breaker" mechanism. Scoped
  specifically to the "insufficient credits" signal actually observed
  tonight. A broader net would also trip on ordinary transient network
  blips, which is a correctness regression, not a fix — see Risks.

## Design

### ElevenLabs: proactive check, shared cache

`app/admin/system.py`'s `_compute_billing()` currently owns fetching
`subscription_status()` and caching it in Redis (`readiness:billing`, 60s
TTL) — see that module's own comment on why billing is one of the two
EXPENSIVE checks worth caching. Move that caching into
`app/telephony/elevenlabs_client.py` (which already owns
`subscription_status()`) as a new function, `cached_subscription_status()`.
`_GOOD_SUBSCRIPTION_STATUSES` moves there too, as a public constant.
`_compute_billing()` becomes a thin wrapper: call the shared function,
format its result into the existing `ReadinessCheck` shape — dashboard
output is unchanged.

`app/telephony/preflight.py` gains a check in its ElevenLabs branch — after
the OpenAI Realtime branch returns (`call_backend ==
languages_module.OPENAI_REALTIME`), before the existing early-return for a
plain `'auto'` campaign with no voice override (that early return exists to
avoid an unnecessary API call for the common case, but billing applies to
literally every ElevenLabs call regardless of override, so this check must
run before it, not after). Fetch the cached status; if it's not in
`GOOD_SUBSCRIPTION_STATUSES`, refuse to dial with a message naming the
account status and pointing at elevenlabs.io. If the check itself raises
(cache read failure, API error with nothing cached yet), also refuse —
matching the existing `agent_exists()` check's precedent immediately above
it in the same file ("not dialling while the platform's state is
unknown") — a false-refusal from a transient check failure costs far less
than dialing into a real failure.

### Sarvam: reactive circuit breaker, manual clear

New module, `app/telephony/sarvam_circuit_breaker.py` — small, three
functions, all thin Redis wrappers (no TTL; this is manual-clear-only by
design):

- `async def trip(reason: str) -> None` — sets a Redis key
  (`sarvam:circuit_breaker`) to `reason`, if not already set (first trip
  wins, so the ORIGINAL failure reason survives repeated tripping rather
  than being overwritten by a later one — useful when several calls fail
  the same way in the minutes before anyone notices).
- `async def tripped_reason() -> str | None` — the reason if tripped, else
  `None`.
- `async def clear() -> None` — deletes the key. The only way out.

**Where it trips:** `app/telephony/sarvam_stt.py`'s `events()` and
`app/telephony/sarvam_tts.py`'s `speak()` both already have an `elif etype
== "error":` branch that logs Sarvam's error frame and ends the stream.
Both gain one line: if the error message contains "insufficient credit"
(case-insensitive substring — matches the exact wording seen tonight,
`'Insufficient credits'`, without being so exact a minor wording change
from Sarvam breaks detection), call `sarvam_circuit_breaker.trip(message)`
before returning. Scoped to this specific message on purpose — see
Non-goals.

**Where it's checked:** `app/telephony/preflight.py`'s existing SARVAM
branch, right after the `SARVAM_API_KEY` presence check. If tripped,
refuse to dial with a message naming the recorded reason, pointing at
dashboard.sarvam.ai/usage to check/top up credits, and naming the clear
endpoint.

**Clearing it:** a new admin endpoint, `POST
/admin/system/sarvam-circuit-breaker/clear` (living in
`app/admin/system.py`, alongside the other system-health surfaces),
calling `sarvam_circuit_breaker.clear()`. Mirrors the manual
`PATCH /admin/campaigns/{id}` `{"active": false}` action already used
tonight as the stopgap — same "an admin explicitly says this is fixed now"
shape.

**Visibility:** the readiness dashboard (`GET /admin/readiness` /
`_compute_readiness` in `app/admin/system.py`) gains a new cheap check —
`sarvam_circuit_breaker` — reading `tripped_reason()` directly (no Redis
caching needed here; it's already a single fast Redis read, unlike the
billing check's real API round-trip). `status: "critical"` when tripped,
with the recorded reason in the detail message, `"good"` otherwise. This
is exactly the "can this system dial right now, and if not why" visibility
`app/admin/system.py`'s own module docstring says the endpoint exists for
— tonight's whole problem was that this state was invisible until a real
call failed.

## Testing / verification

- Unit tests for `sarvam_circuit_breaker.py`: trip-then-tripped_reason
  round-trip; a second `trip()` call does not overwrite an already-set
  reason; `clear()` removes it; `tripped_reason()` on a never-tripped
  breaker returns `None`. Mocked Redis, no Docker — same style as other
  `redis_client`-backed tests in this project.
- A test on `sarvam_stt.py`'s error handling: an error frame containing
  "Insufficient credits" trips the breaker; an error frame with an
  unrelated message (e.g. a bad language code) does not.
- Same pairing of tests for `sarvam_tts.py`'s error handling.
- `preflight.py` tests: a tripped breaker refuses a Sarvam-backed campaign
  with a message naming the reason; a clear breaker does not block it. An
  unhealthy (or unreachable) ElevenLabs billing status refuses an
  ElevenLabs-backed campaign; a healthy one does not.
- A test confirming `elevenlabs_client.cached_subscription_status()` and
  `app/admin/system.py`'s billing check share the same cache entry (one
  populates it, the other reads the same value without a second API call)
  — this is the property the whole point of the move is to guarantee.
- `pytest -q -m "not integration"` and `ruff check .` for the full suite.
- Manual: exercise the new admin clear endpoint against the real Redis
  instance and confirm `/admin/readiness` reflects it immediately.
  Re-activating a real campaign against the still-tripped breaker (before
  Sarvam credits are actually topped up) is the live proof this design
  exists for — do this deliberately once, since it's the exact scenario
  tonight's stopgap was covering by hand.

## Risks

- The credits-message substring match is a soft coupling to Sarvam's exact
  wording. If Sarvam changes it, the breaker silently stops tripping on
  future exhaustion (fails open, not closed) — worth a periodic manual
  check against the live error text the same way other Sarvam wire-format
  assumptions in this codebase are already flagged as "verified against
  the live API" on a specific date, not assumed permanent.
- Scoping the trip condition to this one message means OTHER kinds of
  sustained Sarvam outages (a genuine platform outage, not a credits
  issue) still dial into failure repeatedly, same as tonight before this
  design. Deliberately out of scope per Non-goals — broadening the net
  risks tripping on transient blips, which stops a campaign that would
  have succeeded on the next attempt. If sustained non-credits outages
  turn out to be a recurring problem, that's a new, separate design
  question, not a widening of this one.
- The breaker is per-process Redis state, not per-campaign — tripping it
  blocks ALL Sarvam-backed campaigns, not just the one whose call failed.
  This is intentional: Sarvam credits are account-wide, not per-campaign,
  so a credits failure on any call is evidence the account is out for
  every call.
