"""process_one() decision-logic tests — every collaborator (DB layer,
ElevenLabs client, preflight, Redis) is mocked, so these test ONLY the
sequencing/guard logic, not any real I/O. Each test targets one of the
specific safety issues fixed vs. the blueprint's original worker sketch (see
worker.py's module docstring).
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.telephony import worker


class _FakeRedis:
    def __init__(self, *, dnd=False, dialing_lock_free=True):
        self._dnd = dnd
        self._dialing_lock_free = dialing_lock_free
        self.set_calls = []
        self.deleted = []

    async def sismember(self, key, value):
        return self._dnd

    async def set(self, key, value, nx=False, ex=None):
        self.set_calls.append((key, value, nx, ex))
        return self._dialing_lock_free

    async def delete(self, *keys):
        self.deleted.extend(keys)


def _returns(result):
    """place_outbound_call is async now (Plivo over httpx) — a plain lambda
    would hand process_one a coroutine-less object and the await would fail."""
    async def _placer(**kwargs):
        return result
    return _placer


def _lead(**overrides):
    defaults = dict(
        lead_id=uuid4(), phone_e164="+919876543210", campaign_id=uuid4(),
        status="queued", attempts=0, dnd=False, language_pref="auto",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _campaign(**overrides):
    defaults = dict(campaign_id=uuid4(), mode="twoway", agent_id="agent_1", active=True,
                    language="auto")
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture
def mocks(monkeypatch):
    lead = _lead()
    campaign = _campaign(campaign_id=lead.campaign_id)
    calls = {"mark_calling": [], "mark_dnd": [], "mark_result": [],
             "record_dial_attempt": [], "create_call": [], "enqueue": []}

    async def get_lead(lead_id):
        return lead

    async def get_campaign(cid):
        return campaign

    monkeypatch.setattr(worker.leads_db, "get_lead", get_lead)
    monkeypatch.setattr(worker.campaigns_db, "get_campaign", get_campaign)

    async def mark_calling(lead_id):
        calls["mark_calling"].append(lead_id)
        return True

    async def mark_dnd(lead_id):
        calls["mark_dnd"].append(lead_id)

    async def mark_result(lead_id, status):
        calls["mark_result"].append((lead_id, status))

    async def record_dial_attempt(lead_id):
        calls["record_dial_attempt"].append(lead_id)

    async def create_call(**kwargs):
        calls["create_call"].append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(worker.leads_db, "mark_calling", mark_calling)
    monkeypatch.setattr(worker.leads_db, "mark_dnd", mark_dnd)
    monkeypatch.setattr(worker.leads_db, "mark_result", mark_result)
    monkeypatch.setattr(worker.leads_db, "record_dial_attempt", record_dial_attempt)
    monkeypatch.setattr(worker.calls_db, "create_call", create_call)

    async def fake_enqueue(lead_id):
        calls["enqueue"].append(lead_id)

    monkeypatch.setattr(worker, "enqueue_lead", fake_enqueue)

    async def fake_ack(lead_id):
        calls.setdefault("ack", []).append(lead_id)

    monkeypatch.setattr(worker, "_ack", fake_ack)

    async def no_preflight_error(agent_id, language=None, mode=None):
        return None

    monkeypatch.setattr(worker.preflight_module, "preflight", no_preflight_error)

    async def acquire_slot():
        return True

    async def release_slot():
        calls.setdefault("release_slot", []).append(1)

    monkeypatch.setattr(worker, "try_acquire_call_slot", acquire_slot)
    monkeypatch.setattr(worker, "release_call_slot", release_slot)

    # Pinned open, or every dial test here would depend on the wall clock of
    # whoever runs pytest and fail outside 10:00-19:00 IST. The closed case is
    # asserted explicitly below.
    monkeypatch.setattr(worker, "within_calling_hours", lambda: True)

    return SimpleNamespace(lead=lead, campaign=campaign, calls=calls)


async def test_outside_calling_hours_the_lead_is_never_dialled(mocks, monkeypatch):
    """REGRESSION. campaign_tick checked calling hours at ENQUEUE time and
    process_one never re-checked, but queued leads outlive that check:
    calls:queue is AOF-persisted on a named volume and the backend runs
    restart: unless-stopped, so a batch queued at 18:58 was dialled in full
    whenever the process next came back — 02:30, say. retry_sweeper's reaper
    runs 24/7 and could feed the same path. CLAUDE.md makes the window a hard
    rule, so it has to hold at the moment of dialling, not only at queueing."""
    monkeypatch.setattr(worker.redis_client, "get_redis", lambda: _FakeRedis())
    monkeypatch.setattr(worker, "within_calling_hours", lambda: False)

    await worker.process_one(str(mocks.lead.lead_id))

    assert mocks.calls["create_call"] == [], "must not place a call out of hours"
    assert mocks.calls["mark_calling"] == [], "must not even reserve the lead"
    assert mocks.calls["record_dial_attempt"] == [], "must not burn an attempt"
    # Back to 'pending' specifically: nothing else transitions 'queued' ->
    # 'pending', so leaving it queued would strand the lead forever because
    # due_leads() only ever selects 'pending'.
    assert mocks.calls["mark_result"] == [(mocks.lead.lead_id, "pending")]
    assert mocks.calls["ack"] == [str(mocks.lead.lead_id)]


async def test_dnd_lead_is_marked_and_never_dialled(mocks, monkeypatch):
    monkeypatch.setattr(worker.redis_client, "get_redis", lambda: _FakeRedis(dnd=True))
    await worker.process_one(str(mocks.lead.lead_id))

    assert mocks.calls["mark_dnd"] == [mocks.lead.lead_id]
    assert mocks.calls["mark_calling"] == []          # never even reserved
    assert mocks.calls["create_call"] == []
    assert mocks.calls["ack"] == [str(mocks.lead.lead_id)]  # still always acked


async def test_inactive_campaign_skips_without_dialling(mocks, monkeypatch):
    inactive = _campaign(campaign_id=mocks.lead.campaign_id, active=False)

    async def get_inactive(cid):
        return inactive

    monkeypatch.setattr(worker.campaigns_db, "get_campaign", get_inactive)
    monkeypatch.setattr(worker.redis_client, "get_redis", lambda: _FakeRedis())

    await worker.process_one(str(mocks.lead.lead_id))

    assert mocks.calls["mark_calling"] == []
    assert mocks.calls["create_call"] == []


async def test_mark_calling_guard_rejection_skips_cleanly(mocks, monkeypatch):
    """Simulates a race: another worker already moved this lead past 'queued'."""
    async def reject(lead_id):
        return False

    monkeypatch.setattr(worker.leads_db, "mark_calling", reject)
    monkeypatch.setattr(worker.redis_client, "get_redis", lambda: _FakeRedis())

    await worker.process_one(str(mocks.lead.lead_id))

    assert mocks.calls["create_call"] == []
    assert mocks.calls["record_dial_attempt"] == []   # never got that far
    assert mocks.calls["ack"] == [str(mocks.lead.lead_id)]


async def test_dialing_lock_contention_releases_reservation_without_counting_attempt(
    mocks, monkeypatch,
):
    """Two workers race for the same phone number — the loser must put the
    lead back to 'pending' WITHOUT incrementing attempts (see db/leads.py's
    record_dial_attempt docstring for why that split matters)."""
    monkeypatch.setattr(worker.redis_client, "get_redis",
                        lambda: _FakeRedis(dialing_lock_free=False))

    await worker.process_one(str(mocks.lead.lead_id))

    assert mocks.calls["mark_result"] == [(mocks.lead.lead_id, "pending")]
    assert mocks.calls["record_dial_attempt"] == []
    assert mocks.calls["create_call"] == []


async def test_preflight_failure_releases_reservation_without_counting_attempt(
    mocks, monkeypatch,
):
    monkeypatch.setattr(worker.redis_client, "get_redis", lambda: _FakeRedis())

    async def failing_preflight(agent_id, language=None, mode=None):
        return "PUBLIC_BASE_URL is unreachable"

    monkeypatch.setattr(worker.preflight_module, "preflight", failing_preflight)

    await worker.process_one(str(mocks.lead.lead_id))

    assert mocks.calls["mark_result"] == [(mocks.lead.lead_id, "pending")]
    assert mocks.calls["record_dial_attempt"] == []


async def test_preflight_receives_the_resolved_iso_code_not_the_raw_token(
    mocks, monkeypatch,
):
    """REGRESSION guard. app/languages.py's for_call() is the ONE place both
    this worker and call_routes._call_language() resolve a call's language.
    If the worker were ever "simplified" to pass campaign.language straight
    through instead of resolving it, preflight would receive the catalogue
    token "tinglish" — not an ISO code, not in any agent's language set — and
    refuse EVERY dial for every code-mixed campaign with no failing test to
    catch it. Pin the exact regression: a 'tinglish' campaign must reach
    preflight as 'te'.

    Also pins that the campaign's mode reaches preflight — the two-way
    dial-time guard needs it, and this is the worker's own call site."""
    monkeypatch.setattr(worker.redis_client, "get_redis", lambda: _FakeRedis())
    mocks.campaign.language = "tinglish"
    mocks.campaign.mode = "twoway"

    received = {}

    async def capturing_preflight(agent_id, language=None, mode=None):
        received["language"] = language
        received["mode"] = mode
        return None

    monkeypatch.setattr(worker.preflight_module, "preflight", capturing_preflight)

    await worker.process_one(str(mocks.lead.lead_id))

    assert received["language"] == "te"
    assert received["mode"] == "twoway"


async def test_at_capacity_requeues_instead_of_busy_looping(mocks, monkeypatch):
    monkeypatch.setattr(worker.redis_client, "get_redis", lambda: _FakeRedis())

    async def no_slot():
        return False

    monkeypatch.setattr(worker, "try_acquire_call_slot", no_slot)
    # Deliberately NOT monkeypatching asyncio.sleep here: patching it on the
    # shared module object would affect the whole async runtime for the
    # test's duration, not just this call site — a real (bounded, ~1s) sleep
    # is the safer choice.

    await worker.process_one(str(mocks.lead.lead_id))

    assert mocks.calls["mark_result"] == [(mocks.lead.lead_id, "pending")]
    assert mocks.calls["enqueue"] == [str(mocks.lead.lead_id)]
    assert mocks.calls["record_dial_attempt"] == []


async def test_successful_placement_records_attempt_and_creates_call(mocks, monkeypatch):
    monkeypatch.setattr(worker.redis_client, "get_redis", lambda: _FakeRedis())

    result = SimpleNamespace(success=True, message="ok", call_uuid="call_1")
    monkeypatch.setattr(worker.plivo_client, "place_outbound_call", _returns(result))

    await worker.process_one(str(mocks.lead.lead_id))

    assert mocks.calls["record_dial_attempt"] == [mocks.lead.lead_id]
    assert len(mocks.calls["create_call"]) == 1
    assert mocks.calls["create_call"][0]["provider_call_id"] == "call_1"
    # Slot ownership transfers to the call routes on success (released when the
    # stream ends or Plivo posts the hangup) — NOT released here.
    assert "release_slot" not in mocks.calls


async def test_worker_dials_the_lead_and_identifies_it_to_the_answer_url(mocks, monkeypatch):
    """The worker's whole job at placement is: dial this number, and tell
    Plivo which lead it is. lead_id travels in the answer URL, and is what
    lets /calls/stream find the lead and campaign again when the call is
    answered — get it wrong and the call connects to nothing."""
    monkeypatch.setattr(worker.redis_client, "get_redis", lambda: _FakeRedis())
    captured = {}

    async def fake_place(**kw):
        captured.update(kw)
        return SimpleNamespace(success=True, message="ok", call_uuid="call_1")

    monkeypatch.setattr(worker.plivo_client, "place_outbound_call", fake_place)

    await worker.process_one(str(mocks.lead.lead_id))

    assert captured["to_number"] == mocks.lead.phone_e164
    assert captured["lead_id"] == str(mocks.lead.lead_id)


async def test_call_row_starts_without_a_conversation_id(mocks, monkeypatch):
    """With Plivo placing the call there is no ElevenLabs conversation yet —
    one is created when the audio bridge connects, and call_routes fills it
    in then. Writing a placeholder here would collide with the UNIQUE
    constraint on el_conversation_id across concurrent calls."""
    monkeypatch.setattr(worker.redis_client, "get_redis", lambda: _FakeRedis())
    result = SimpleNamespace(success=True, message="ok", call_uuid="plivo_uuid_1")
    monkeypatch.setattr(worker.plivo_client, "place_outbound_call", _returns(result))

    await worker.process_one(str(mocks.lead.lead_id))

    created = mocks.calls["create_call"][0]
    assert created["el_conversation_id"] is None
    assert created["provider_call_id"] == "plivo_uuid_1"


async def test_placement_exception_releases_slot_and_marks_failed(mocks, monkeypatch):
    monkeypatch.setattr(worker.redis_client, "get_redis", lambda: _FakeRedis())

    async def boom(**kw):
        raise RuntimeError("Plivo API down")

    monkeypatch.setattr(worker.plivo_client, "place_outbound_call", boom)

    await worker.process_one(str(mocks.lead.lead_id))

    assert mocks.calls["record_dial_attempt"] == [mocks.lead.lead_id]  # attempt DID happen
    assert mocks.calls["mark_result"] == [(mocks.lead.lead_id, "failed")]
    assert mocks.calls["create_call"] == []
    assert mocks.calls["release_slot"] == [1]


async def test_unsuccessful_result_releases_slot_and_marks_failed(mocks, monkeypatch):
    monkeypatch.setattr(worker.redis_client, "get_redis", lambda: _FakeRedis())

    result = SimpleNamespace(success=False, message="no agent capacity", call_uuid=None)
    monkeypatch.setattr(worker.plivo_client, "place_outbound_call", _returns(result))

    await worker.process_one(str(mocks.lead.lead_id))

    assert mocks.calls["mark_result"] == [(mocks.lead.lead_id, "failed")]
    assert mocks.calls["create_call"] == []
    assert mocks.calls["release_slot"] == [1]


# ── the dialing-lock must not linger when no call was placed ─────────────────
# Regression tests for a real bug found by the end-to-end pipeline test: every
# path that bails AFTER acquiring `dialing:<phone>` but WITHOUT placing a call
# must delete the lock. Leaving it set blocked the lead from retrying for the
# full DIALING_LOCK_TTL_S even though nothing was ever dialled.

async def test_preflight_failure_releases_the_dialing_lock(mocks, monkeypatch):
    r = _FakeRedis()
    monkeypatch.setattr(worker.redis_client, "get_redis", lambda: r)

    async def failing_preflight(agent_id, language=None, mode=None):
        return "PUBLIC_BASE_URL is unreachable"

    monkeypatch.setattr(worker.preflight_module, "preflight", failing_preflight)

    await worker.process_one(str(mocks.lead.lead_id))

    assert f"dialing:{mocks.lead.phone_e164}" in r.deleted


async def test_at_capacity_releases_the_dialing_lock(mocks, monkeypatch):
    r = _FakeRedis()
    monkeypatch.setattr(worker.redis_client, "get_redis", lambda: r)

    async def no_slot():
        return False

    monkeypatch.setattr(worker, "try_acquire_call_slot", no_slot)

    await worker.process_one(str(mocks.lead.lead_id))

    assert f"dialing:{mocks.lead.phone_e164}" in r.deleted


async def test_placement_failure_releases_the_dialing_lock(mocks, monkeypatch):
    r = _FakeRedis()
    monkeypatch.setattr(worker.redis_client, "get_redis", lambda: r)

    async def boom(**kw):
        raise RuntimeError("Plivo API down")

    monkeypatch.setattr(worker.plivo_client, "place_outbound_call", boom)

    await worker.process_one(str(mocks.lead.lead_id))

    assert f"dialing:{mocks.lead.phone_e164}" in r.deleted


async def test_successful_placement_KEEPS_the_dialing_lock(mocks, monkeypatch):
    """The inverse: a genuinely in-flight call SHOULD keep the lock set (to
    expire on its own), since that's exactly the double-dial it prevents."""
    r = _FakeRedis()
    monkeypatch.setattr(worker.redis_client, "get_redis", lambda: r)

    result = SimpleNamespace(success=True, message="ok", call_uuid="sid_1")
    monkeypatch.setattr(worker.plivo_client, "place_outbound_call", _returns(result))

    await worker.process_one(str(mocks.lead.lead_id))

    assert f"dialing:{mocks.lead.phone_e164}" not in r.deleted


async def test_lead_not_found_is_dropped_without_crashing(monkeypatch):
    async def get_none(lead_id):
        return None

    monkeypatch.setattr(worker.leads_db, "get_lead", get_none)
    acked = []

    async def fake_ack(lead_id):
        acked.append(lead_id)

    monkeypatch.setattr(worker, "_ack", fake_ack)

    await worker.process_one(str(uuid4()))  # no exception
    assert len(acked) == 1


async def test_process_one_always_acks_even_on_early_return(mocks, monkeypatch):
    """The finally block must run regardless of which branch returned."""
    monkeypatch.setattr(worker.redis_client, "get_redis", lambda: _FakeRedis(dnd=True))
    await worker.process_one(str(mocks.lead.lead_id))
    assert mocks.calls["ack"] == [str(mocks.lead.lead_id)]


# ── unhandled failures must not strand the lead (docstring point 7) ──────────
# Before this, process_one had try/finally with no except: an error anywhere
# outside the placement call propagated, the finally acked it out of
# calls:processing anyway, and the lead was left status='calling' — invisible
# to both the reaper and due_leads(), i.e. never callable again.

async def test_unhandled_error_before_dialling_releases_the_lead(mocks, monkeypatch):
    r = _FakeRedis()
    monkeypatch.setattr(worker.redis_client, "get_redis", lambda: r)

    async def boom(agent_id, language=None):
        raise RuntimeError("Postgres went away")

    monkeypatch.setattr(worker.preflight_module, "preflight", boom)

    await worker.process_one(str(mocks.lead.lead_id))  # must not raise

    # Back to 'pending' so the next campaign_tick re-selects it...
    assert mocks.calls["mark_result"] == [(mocks.lead.lead_id, "pending")]
    # ...with no attempt burned, and the dialing lock released.
    assert mocks.calls["record_dial_attempt"] == []
    assert f"dialing:{mocks.lead.phone_e164}" in r.deleted
    assert mocks.calls["ack"] == [str(mocks.lead.lead_id)]


async def test_unhandled_error_after_dialling_never_releases_the_lead(mocks, monkeypatch):
    """create_call() failing AFTER a real call is ringing must NOT put the
    lead back to 'pending' — a later tick would dial someone already on the
    phone. The webhook owns the lead from placement onward."""
    monkeypatch.setattr(worker.redis_client, "get_redis", lambda: _FakeRedis())

    result = SimpleNamespace(success=True, message="ok", call_uuid="sid_1")
    monkeypatch.setattr(worker.plivo_client, "place_outbound_call", _returns(result))

    async def boom(**kwargs):
        raise RuntimeError("unique violation on el_conversation_id")

    monkeypatch.setattr(worker.calls_db, "create_call", boom)

    await worker.process_one(str(mocks.lead.lead_id))  # must not raise

    assert mocks.calls["mark_result"] == []                  # never released
    assert mocks.calls.get("release_slot", []) == []         # slot stays with the webhook
    assert mocks.calls.get("ack") == [str(mocks.lead.lead_id)]


async def test_lead_is_left_for_the_reaper_when_recovery_itself_fails(mocks, monkeypatch):
    """If the backing store is down hard, cleanup can't run either. Acking
    then would delete the only record that this lead was mid-flight — so the
    entry must stay in calls:processing for the reaper instead."""
    monkeypatch.setattr(worker.redis_client, "get_redis", lambda: _FakeRedis())

    async def boom(agent_id, language=None):
        raise RuntimeError("Postgres went away")

    async def also_boom(lead_id, status):
        raise RuntimeError("still down")

    monkeypatch.setattr(worker.preflight_module, "preflight", boom)
    monkeypatch.setattr(worker.leads_db, "mark_result", also_boom)

    await worker.process_one(str(mocks.lead.lead_id))  # must not raise

    assert mocks.calls.get("ack", []) == []  # deliberately NOT acked
