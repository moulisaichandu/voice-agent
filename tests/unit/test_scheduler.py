"""Unit tests for scheduler.py's job functions — DB/Redis/worker calls are
mocked so these test only the decision logic (calling-hours gate, batch
sizing, DND propagation), not real scheduling or I/O."""

from types import SimpleNamespace
from uuid import uuid4

from app import scheduler


async def test_campaign_tick_skips_outside_calling_hours(monkeypatch):
    monkeypatch.setattr(scheduler, "within_calling_hours", lambda: False)
    called = {"due_leads": 0}

    async def boom(limit):
        called["due_leads"] += 1
        return []

    monkeypatch.setattr(scheduler.leads_db, "due_leads", boom)
    await scheduler.campaign_tick()
    assert called["due_leads"] == 0


async def test_campaign_tick_queues_due_leads_inside_calling_hours(monkeypatch):
    monkeypatch.setattr(scheduler, "within_calling_hours", lambda: True)
    lead1, lead2 = SimpleNamespace(lead_id=uuid4()), SimpleNamespace(lead_id=uuid4())

    async def fake_due_leads(limit, campaign_id=None):
        return [lead1, lead2]

    queued = []

    async def fake_mark_queued(lead_id):
        queued.append(lead_id)
        return True

    enqueued = []

    async def fake_enqueue(lead_id):
        enqueued.append(lead_id)

    monkeypatch.setattr(scheduler.leads_db, "due_leads", fake_due_leads)
    monkeypatch.setattr(scheduler.leads_db, "mark_queued", fake_mark_queued)
    monkeypatch.setattr(scheduler.worker, "enqueue_lead", fake_enqueue)

    await scheduler.campaign_tick()

    assert queued == [lead1.lead_id, lead2.lead_id]
    assert enqueued == [str(lead1.lead_id), str(lead2.lead_id)]


async def test_campaign_tick_does_not_enqueue_a_lead_whose_queue_guard_rejects(monkeypatch):
    """A duplicate/racing tick's mark_queued() correctly returns False for an
    already-queued lead — that lead must NOT also be pushed onto Redis."""
    monkeypatch.setattr(scheduler, "within_calling_hours", lambda: True)
    lead = SimpleNamespace(lead_id=uuid4())

    async def fake_due_leads(limit, campaign_id=None):
        return [lead]

    async def reject(lead_id):
        return False

    enqueued = []

    async def fake_enqueue(lead_id):
        enqueued.append(lead_id)

    monkeypatch.setattr(scheduler.leads_db, "due_leads", fake_due_leads)
    monkeypatch.setattr(scheduler.leads_db, "mark_queued", reject)
    monkeypatch.setattr(scheduler.worker, "enqueue_lead", fake_enqueue)

    await scheduler.campaign_tick()
    assert enqueued == []


async def test_campaign_tick_rolls_back_when_redis_enqueue_fails(monkeypatch):
    monkeypatch.setattr(scheduler, "within_calling_hours", lambda: True)
    lead = SimpleNamespace(lead_id=uuid4())
    rolled_back = []

    async def fake_due_leads(limit, campaign_id=None):
        return [lead]

    async def mark_queued(lead_id):
        return True

    async def fail_enqueue(lead_id):
        raise ConnectionError("redis unavailable")

    async def mark_result(lead_id, status):
        rolled_back.append((lead_id, status))

    monkeypatch.setattr(scheduler.leads_db, "due_leads", fake_due_leads)
    monkeypatch.setattr(scheduler.leads_db, "mark_queued", mark_queued)
    monkeypatch.setattr(scheduler.worker, "enqueue_lead", fail_enqueue)
    monkeypatch.setattr(scheduler.leads_db, "mark_result", mark_result)

    assert await scheduler.campaign_tick() == 0
    assert rolled_back == [(lead.lead_id, "pending")]


async def test_campaign_tick_passes_the_scope_through_to_due_leads(monkeypatch):
    """Scoping must reach the QUERY, not be applied afterwards — filtering a
    global result set would still have queued other campaigns' leads on the
    way past."""
    monkeypatch.setattr(scheduler, "within_calling_hours", lambda: True)
    target = uuid4()
    seen = {}

    async def fake_due_leads(limit, campaign_id=None):
        seen["campaign_id"] = campaign_id
        return []

    monkeypatch.setattr(scheduler.leads_db, "due_leads", fake_due_leads)

    await scheduler.campaign_tick(campaign_id=target)
    assert seen["campaign_id"] == target

    await scheduler.campaign_tick()
    assert seen["campaign_id"] is None  # scheduler's own run stays global


async def test_scoped_tick_still_respects_calling_hours(monkeypatch):
    """Scoping narrows WHICH leads, it must never relax a compliance gate."""
    monkeypatch.setattr(scheduler, "within_calling_hours", lambda: False)

    async def boom(limit, campaign_id=None):
        raise AssertionError("must not select leads outside calling hours")

    monkeypatch.setattr(scheduler.leads_db, "due_leads", boom)

    assert await scheduler.campaign_tick(campaign_id=uuid4()) == 0


async def test_retry_sweeper_delegates_to_the_reaper(monkeypatch):
    calls = []

    async def fake_reaper(timeout_s):
        calls.append(timeout_s)
        return 0

    monkeypatch.setattr(scheduler.worker, "reaper_sweep", fake_reaper)
    await scheduler.retry_sweeper()
    assert calls == [scheduler.PROCESSING_REAPER_TIMEOUT_S]


async def test_dnd_refresh_syncs_flagged_phones_into_redis(monkeypatch):
    async def fake_dnd_phones():
        return ["+919876543210", "+919812345678"]

    sadd_calls = []

    class _FakeRedis:
        async def smembers(self, key):
            return set()

        async def sadd(self, key, *values):
            sadd_calls.append((key, values))

    monkeypatch.setattr(scheduler.leads_db, "dnd_phones", fake_dnd_phones)
    monkeypatch.setattr(scheduler.redis_client, "get_redis", lambda: _FakeRedis())

    await scheduler.dnd_refresh()

    assert sadd_calls == [(scheduler.worker.DND_SET_KEY,
                           ("+919876543210", "+919812345678"))]


async def test_dnd_refresh_is_a_noop_with_no_flagged_leads(monkeypatch):
    async def fake_dnd_phones():
        return []

    sadd_calls = []

    class _FakeRedis:
        async def smembers(self, key):
            return set()

        async def sadd(self, key, *values):
            sadd_calls.append((key, values))

        async def srem(self, key, *values):
            raise AssertionError("there are no stale members to remove")

    monkeypatch.setattr(scheduler.leads_db, "dnd_phones", fake_dnd_phones)
    monkeypatch.setattr(scheduler.redis_client, "get_redis", lambda: _FakeRedis())

    await scheduler.dnd_refresh()
    assert sadd_calls == []


async def test_dnd_refresh_removes_numbers_cleared_in_the_database(monkeypatch):
    async def fake_dnd_phones():
        return ["+919876543210"]

    removed, added = [], []

    class _FakeRedis:
        async def smembers(self, key):
            return {"+919876543210", "+919812345678"}

        async def srem(self, key, *values):
            removed.append((key, values))

        async def sadd(self, key, *values):
            added.append((key, values))

    monkeypatch.setattr(scheduler.leads_db, "dnd_phones", fake_dnd_phones)
    monkeypatch.setattr(scheduler.redis_client, "get_redis", lambda: _FakeRedis())

    await scheduler.dnd_refresh()

    assert removed == [(scheduler.worker.DND_SET_KEY, ("+919812345678",))]
    assert added == [(scheduler.worker.DND_SET_KEY, ("+919876543210",))]


_WB_QUEUE = "sheets:writeback:queue"
_WB_PROCESSING = "sheets:writeback:processing"


class _FakeRedisQueue:
    """Models the two lists transcript_reconcile uses (the write-back queue and
    its processing list) with real lmove/lrem/lpush semantics — index 0 is the
    head (LEFT), index -1 the tail (RIGHT)."""

    def __init__(self, items):
        # `items` seeds the queue.
        self._lists = {_WB_QUEUE: list(items), _WB_PROCESSING: []}
        self.lpush_calls = []  # values pushed back onto the QUEUE (re-queues)

    async def llen(self, key):
        return len(self._lists.setdefault(key, []))

    async def lpush(self, key, value):
        self._lists.setdefault(key, []).insert(0, value)
        if key == _WB_QUEUE:
            self.lpush_calls.append(value)

    async def lmove(self, src, dst, src_pos, dst_pos):
        s = self._lists.setdefault(src, [])
        if not s:
            return None
        elem = s.pop(0) if src_pos.upper() == "LEFT" else s.pop()
        d = self._lists.setdefault(dst, [])
        d.insert(0, elem) if dst_pos.upper() == "LEFT" else d.append(elem)
        return elem

    async def lrem(self, key, count, value):
        lst = self._lists.setdefault(key, [])
        removed = 0
        i = 0
        while i < len(lst) and (count == 0 or removed < count):
            if lst[i] == value:
                lst.pop(i)
                removed += 1
            else:
                i += 1
        return removed


async def test_sheets_sync_job_delegates_and_survives_failure(monkeypatch):
    """A Sheets outage must not crash the scheduler loop — it also runs
    campaign_tick/retry_sweeper/dnd_refresh, which must keep ticking."""
    # These exercise the DRAIN, which only runs when the Sheet is set up.
    # The test env blanks GOOGLE_SHEET_ID, so say so explicitly.
    monkeypatch.setattr(scheduler.sheets_client, "unconfigured_reason",
                        lambda: None)
    calls = {"n": 0}

    async def boom():
        calls["n"] += 1
        raise RuntimeError("Sheets API down")

    monkeypatch.setattr(scheduler.sheets_sync_module, "sheets_sync", boom)
    await scheduler.sheets_sync()  # must not raise
    assert calls["n"] == 1


async def test_transcript_reconcile_writes_back_and_requeues_failures(monkeypatch):
    # These exercise the DRAIN, which only runs when the Sheet is set up.
    # The test env blanks GOOGLE_SHEET_ID, so say so explicitly.
    monkeypatch.setattr(scheduler.sheets_client, "unconfigured_reason",
                        lambda: None)
    # Real lead_ids in production are always valid UUID strings (str(lead_id)
    # from a real Lead row) — using realistic values here so this test
    # actually exercises write_back_lead rather than failing at UUID parsing.
    ok_id, fail_id = str(uuid4()), str(uuid4())
    r = _FakeRedisQueue([ok_id, fail_id])
    monkeypatch.setattr(scheduler.redis_client, "get_redis", lambda: r)

    results = {ok_id: True, fail_id: False}

    async def fake_write_back(lead_id):
        return results[str(lead_id)]

    monkeypatch.setattr(scheduler.sheets_writeback, "write_back_lead", fake_write_back)

    await scheduler.transcript_reconcile()

    assert r.lpush_calls == [fail_id]  # only the failure is requeued


async def test_transcript_reconcile_reclaims_items_stranded_by_a_crash(monkeypatch):
    """REGRESSION: the drain used to rpop a whole batch at once (at-most-once),
    so a crash mid-write-back permanently dropped those Sheet write-backs. Items
    are now reserved onto a processing list and removed only after success, so a
    previous crashed run's in-flight item is reclaimed and retried next run."""
    # These exercise the DRAIN, which only runs when the Sheet is set up.
    # The test env blanks GOOGLE_SHEET_ID, so say so explicitly.
    monkeypatch.setattr(scheduler.sheets_client, "unconfigured_reason",
                        lambda: None)
    stranded = str(uuid4())
    r = _FakeRedisQueue([])
    # A previous run reserved this id into 'processing', then crashed before ack.
    await r.lpush(_WB_PROCESSING, stranded)
    monkeypatch.setattr(scheduler.redis_client, "get_redis", lambda: r)

    written = []

    async def fake_write_back(lead_id):
        written.append(str(lead_id))
        return True

    monkeypatch.setattr(scheduler.sheets_writeback, "write_back_lead", fake_write_back)

    await scheduler.transcript_reconcile()

    assert written == [stranded], "a stranded write-back must be reclaimed and retried"
    assert r._lists[_WB_PROCESSING] == [], "and acked out of the processing list"


async def test_transcript_reconcile_degrades_gracefully_on_malformed_lead_id(monkeypatch):
    """Anything landing in the queue that isn't a valid UUID (shouldn't
    happen in production — real entries always come from str(lead.lead_id)
    — but if it ever did) must be logged and requeued, not crash the job."""
    # These exercise the DRAIN, which only runs when the Sheet is set up.
    # The test env blanks GOOGLE_SHEET_ID, so say so explicitly.
    monkeypatch.setattr(scheduler.sheets_client, "unconfigured_reason",
                        lambda: None)
    r = _FakeRedisQueue(["not-a-real-uuid"])
    monkeypatch.setattr(scheduler.redis_client, "get_redis", lambda: r)

    await scheduler.transcript_reconcile()  # must not raise
    assert r.lpush_calls == ["not-a-real-uuid"]


async def test_transcript_reconcile_noop_on_empty_queue(monkeypatch):
    r = _FakeRedisQueue([])
    monkeypatch.setattr(scheduler.redis_client, "get_redis", lambda: r)

    called = []

    async def fake_write_back(lead_id):
        called.append(lead_id)
        return True

    monkeypatch.setattr(scheduler.sheets_writeback, "write_back_lead", fake_write_back)

    await scheduler.transcript_reconcile()
    assert called == []


async def test_start_is_idempotent():
    """Calling start() twice must not create a second scheduler instance —
    that's exactly the 'jobs fire twice' failure the blueprint warns about.
    Must be async: AsyncIOScheduler.start() attaches to the currently
    running event loop, which only exists inside an async test."""
    try:
        s1 = scheduler.start()
        s2 = scheduler.start()
        assert s1 is s2
    finally:
        scheduler.shutdown()


# ── an unconfigured Sheet is a settings problem, not a per-lead failure ──────
#
# From the live logs, repeating every ten minutes forever:
#   [scheduler] sheets_sync failed: FileNotFoundError: 'sa.json'
#   [scheduler] write-back raised for lead 9fb5d6d8-...: FileNotFoundError
#   [scheduler] write-back raised for lead ded0c97b-...: FileNotFoundError
#   ...one ERROR per queued lead, per sweep, indefinitely.
#
# The queue mechanics were fine — nothing was lost, everything requeued. But a
# missing credential is not a transient failure of nine separate writes, and
# reporting it as one buries the case this logging exists for: a REAL Sheets
# error on one row. It also re-attempts nine doomed API calls every sweep.

async def test_an_unconfigured_sheet_does_not_drain_the_queue(monkeypatch):
    """The transcripts must stay queued, ready for when the credential lands."""
    ids = [str(uuid4()) for _ in range(3)]
    r = _FakeRedisQueue(list(ids))
    monkeypatch.setattr(scheduler.redis_client, "get_redis", lambda: r)
    monkeypatch.setattr(scheduler.sheets_client, "unconfigured_reason",
                        lambda: "GOOGLE_SERVICE_ACCOUNT_FILE 'sa.json' not found")

    async def must_not_run(lead_id):
        raise AssertionError("must not attempt a write-back with no credential")

    monkeypatch.setattr(scheduler.sheets_writeback, "write_back_lead", must_not_run)

    await scheduler.transcript_reconcile()

    assert r._lists[_WB_QUEUE] == ids, "queued transcripts must be preserved"
    assert r._lists[_WB_PROCESSING] == []


async def test_an_unconfigured_sheet_is_reported_once_not_once_per_lead(
    monkeypatch, caplog
):
    """Nine ERROR lines per sweep drown the one case this logging is for."""
    r = _FakeRedisQueue([str(uuid4()) for _ in range(9)])
    monkeypatch.setattr(scheduler.redis_client, "get_redis", lambda: r)
    monkeypatch.setattr(scheduler.sheets_client, "unconfigured_reason",
                        lambda: "GOOGLE_SERVICE_ACCOUNT_FILE 'sa.json' not found")

    with caplog.at_level("WARNING"):
        await scheduler.transcript_reconcile()

    mentions = [rec for rec in caplog.records if "sa.json" in rec.getMessage()]
    assert len(mentions) == 1, f"expected one line, got {len(mentions)}"
    assert "9" in caplog.text, "the operator should be told how many are waiting"


async def test_sheets_sync_skips_cleanly_when_unconfigured(monkeypatch, caplog):
    monkeypatch.setattr(scheduler.sheets_client, "unconfigured_reason",
                        lambda: "GOOGLE_SHEET_ID is not set")

    async def must_not_run():
        raise AssertionError("must not call the Sheets API with no credential")

    monkeypatch.setattr(scheduler.sheets_sync_module, "sheets_sync", must_not_run)

    class _R:
        async def set(self, *a, **kw):
            return True

    monkeypatch.setattr(scheduler.redis_client, "get_redis", lambda: _R())

    with caplog.at_level("WARNING"):
        await scheduler.sheets_sync()

    assert "GOOGLE_SHEET_ID" in caplog.text
    assert not any(rec.levelname == "ERROR" for rec in caplog.records), (
        "a missing setting is not an error condition to alarm on every sweep"
    )


async def test_a_configured_sheet_still_drains_normally(monkeypatch):
    """The guard must not become a way to silently stop writing back."""
    ok_id = str(uuid4())
    r = _FakeRedisQueue([ok_id])
    monkeypatch.setattr(scheduler.redis_client, "get_redis", lambda: r)
    monkeypatch.setattr(scheduler.sheets_client, "unconfigured_reason", lambda: None)

    written = []

    async def fake_write_back(lead_id):
        written.append(str(lead_id))
        return True

    monkeypatch.setattr(scheduler.sheets_writeback, "write_back_lead", fake_write_back)

    await scheduler.transcript_reconcile()

    assert written == [ok_id]
