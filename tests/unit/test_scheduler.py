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
        async def sadd(self, key, *values):
            sadd_calls.append((key, values))

    monkeypatch.setattr(scheduler.leads_db, "dnd_phones", fake_dnd_phones)
    monkeypatch.setattr(scheduler.redis_client, "get_redis", lambda: _FakeRedis())

    await scheduler.dnd_refresh()
    assert sadd_calls == []


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
    calls = {"n": 0}

    async def boom():
        calls["n"] += 1
        raise RuntimeError("Sheets API down")

    monkeypatch.setattr(scheduler.sheets_sync_module, "sheets_sync", boom)
    await scheduler.sheets_sync()  # must not raise
    assert calls["n"] == 1


async def test_transcript_reconcile_writes_back_and_requeues_failures(monkeypatch):
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
