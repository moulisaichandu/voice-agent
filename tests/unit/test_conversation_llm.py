"""Unit tests for telephony/conversation_llm.py — the two-way conversation brain.

Split out of sarvam_llm because the two jobs have opposite constraints and, as
of the first live test, cannot use the same model.

Rendering a script happens once per campaign, is cached, and should use the
best Indic model available. Answering a lead happens every turn with a real
person waiting in silence, so it has to be fast — and Sarvam's chat models are
reasoning models that measured 21.8s on a realistic turn, which cannot be
switched off. Hence a provider-agnostic client speaking the OpenAI-compatible
protocol both vendors implement.
"""

import asyncio

import pytest

from app.telephony import conversation_llm

_HISTORY = [
    {"role": "system", "content": "You are a voice assistant."},
    {"role": "user", "content": "How much are the fees?"},
]


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {
            "choices": [{"message": {"content": "Twenty five thousand."}}]
        }

    def json(self):
        return self._payload


class _FakeAsyncClient:
    calls: list[dict] = []

    def __init__(self, *, status_code=200, payload=None, raise_error=None):
        self._status_code = status_code
        self._payload = payload
        self._raise_error = raise_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None):
        type(self).calls.append({"url": url, "json": json, "headers": headers})
        if self._raise_error:
            raise self._raise_error
        return _FakeResponse(self._status_code, self._payload)


@pytest.fixture
def llm(monkeypatch):
    _FakeAsyncClient.calls = []
    monkeypatch.setattr(conversation_llm, "OPENAI_API_KEY", "sk-openai")
    monkeypatch.setattr(conversation_llm, "SARVAM_API_KEY", "sk-sarvam")
    monkeypatch.setattr(conversation_llm, "CONVERSATION_LLM_PROVIDER", "openai")
    monkeypatch.setattr(conversation_llm, "CONVERSATION_LLM_MODEL", "")
    # turn() now reuses one client via _get_client() instead of opening one per
    # call — see that function's docstring. Without this reset, a client built
    # (and configured) by a PREVIOUS test would still be sitting in the module
    # global and every test after the first would silently talk to it instead
    # of the fake this test is about to install.
    monkeypatch.setattr(conversation_llm, "_client", None)
    monkeypatch.setattr(conversation_llm, "_client_loop", None)

    def _install(**client_kwargs):
        monkeypatch.setattr(
            conversation_llm.httpx, "AsyncClient",
            lambda **kw: _FakeAsyncClient(**client_kwargs),
        )

    return _install


def _reply(content=None, tool_calls=None):
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}]}


# ── which provider, and where ────────────────────────────────────────────────

async def test_it_talks_to_openai_by_default(llm):
    """The default exists because Sarvam's chat models measured 21.8s on a
    conversational turn and cannot be told to stop reasoning."""
    llm()
    await conversation_llm.turn(_HISTORY)

    sent = _FakeAsyncClient.calls[0]
    assert sent["url"] == "https://api.openai.com/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer sk-openai"
    assert sent["json"]["model"] == "gpt-4o-mini"


async def test_it_can_be_pointed_back_at_sarvam(llm, monkeypatch):
    """Single-vendor is still reachable without a code change, for whenever
    their latency story changes."""
    llm()
    monkeypatch.setattr(conversation_llm, "CONVERSATION_LLM_PROVIDER", "sarvam")
    await conversation_llm.turn(_HISTORY)

    sent = _FakeAsyncClient.calls[0]
    assert sent["url"] == "https://api.sarvam.ai/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer sk-sarvam"
    assert sent["json"]["model"] == "sarvam-105b"


async def test_an_explicit_model_overrides_the_provider_default(llm, monkeypatch):
    llm()
    monkeypatch.setattr(conversation_llm, "CONVERSATION_LLM_MODEL", "gpt-4.1-mini")
    await conversation_llm.turn(_HISTORY)
    assert _FakeAsyncClient.calls[0]["json"]["model"] == "gpt-4.1-mini"


async def test_a_missing_key_for_the_chosen_provider_raises_before_calling(llm, monkeypatch):
    """Distinguishable from a network failure: the operator has to set a
    different variable depending on which provider they chose."""
    llm()
    monkeypatch.setattr(conversation_llm, "OPENAI_API_KEY", None)
    with pytest.raises(conversation_llm.ConversationNotConfigured) as excinfo:
        await conversation_llm.turn(_HISTORY)
    assert "OPENAI_API_KEY" in str(excinfo.value)
    assert _FakeAsyncClient.calls == []


# ── the turn itself ──────────────────────────────────────────────────────────

async def test_a_turn_returns_the_assistant_text(llm):
    llm(payload=_reply(content="Twenty five thousand."))
    reply = await conversation_llm.turn(_HISTORY)
    assert reply.text == "Twenty five thousand."
    assert reply.tool_calls == []


async def test_a_turn_sends_the_tools_and_the_history(llm):
    llm(payload=_reply(content="Sure."))
    await conversation_llm.turn(_HISTORY)

    sent = _FakeAsyncClient.calls[0]["json"]
    assert sent["messages"] == _HISTORY
    names = [t["function"]["name"] for t in sent["tools"]]
    assert "search_course_material" in names
    assert "end_call" in names
    assert sent["tool_choice"] == "auto"


async def test_a_tool_call_comes_back_parsed(llm):
    llm(payload=_reply(tool_calls=[{
        "id": "call_1", "type": "function",
        "function": {"name": "search_course_material",
                     "arguments": '{"query": "fees"}'},
    }]))

    reply = await conversation_llm.turn(_HISTORY)

    assert reply.text == ""
    assert reply.tool_calls[0].name == "search_course_material"
    assert reply.tool_calls[0].arguments == {"query": "fees"}
    assert reply.tool_calls[0].call_id == "call_1"


async def test_unparseable_tool_arguments_become_empty_rather_than_raising(llm):
    """A model emitting malformed JSON should cost one useless tool call, not
    the conversation. search_relevant('') returns the no-material note, which
    the agent can say out loud."""
    llm(payload=_reply(tool_calls=[{
        "id": "call_1", "type": "function",
        "function": {"name": "search_course_material", "arguments": "{oh no"},
    }]))

    reply = await conversation_llm.turn(_HISTORY)
    assert reply.tool_calls[0].arguments == {}


async def test_a_reply_with_neither_words_nor_a_tool_call_raises(llm):
    """Silence on a lead who just asked a question is indistinguishable from a
    dropped call, so it must not be returned as an empty string something might
    quietly speak."""
    llm(payload=_reply(content=None))
    with pytest.raises(conversation_llm.NoAnswer):
        await conversation_llm.turn(_HISTORY)


async def test_a_tool_call_without_words_is_perfectly_normal(llm):
    """Look something up, then answer. Not the failure above."""
    llm(payload=_reply(tool_calls=[{
        "id": "c1", "type": "function",
        "function": {"name": "search_course_material",
                     "arguments": '{"query": "fees"}'},
    }]))
    reply = await conversation_llm.turn(_HISTORY)
    assert reply.text == ""
    assert reply.tool_calls


async def test_an_http_failure_raises(llm):
    llm(status_code=500, payload={})
    with pytest.raises(conversation_llm.TurnFailed):
        await conversation_llm.turn(_HISTORY)


async def test_the_turn_timeout_is_shorter_than_a_lead_s_patience(llm):
    """A render can take 20s because it is cached and off the call path. A turn
    cannot: past a few seconds the lead has decided the line is dead, and the
    answer is no longer worth having."""
    assert conversation_llm.CONVERSATION_LLM_TIMEOUT_S <= 15


async def test_the_search_tool_demands_an_english_query(llm):
    """Your course documents are English, so the embeddings are English.
    Measured against the real corpus: a Telugu query scores 0.10-0.17 against
    its own answer chunk, below the 0.30 relevance floor, while the English
    equivalent scores 0.35-0.65.

    search_relevant() does rescue a miss by translating, but that costs a round
    trip mid-call AND loses context: "లక్షణాలు గురించి చెప్పు" (tell me about
    the features) was translated in isolation as "Tell about the symptoms" and
    retrieved nothing. The MODEL is the right place to do this — unlike the
    standalone translator, it knows the call is about digital marketing
    courses. app/rag/translate.py explains why that translator must NOT be
    given the same hint."""
    llm()
    await conversation_llm.turn(_HISTORY)

    tools = _FakeAsyncClient.calls[0]["json"]["tools"]
    search = next(t["function"] for t in tools
                  if t["function"]["name"] == "search_course_material")
    described = (search["description"] + " "
                 + search["parameters"]["properties"]["query"]["description"])
    assert "English" in described
    lowered = described.lower()
    assert "always" in lowered or "must" in lowered, (
        "the model treated 'as a concise search query in English' as advice "
        "and sent Telugu anyway"
    )


# ── one connection, not one per turn ─────────────────────────────────────────
#
# turn() used to build and tear down an httpx.AsyncClient inside itself, so
# every completion paid a fresh DNS + TCP + TLS handshake to api.openai.com —
# and a course question pays it TWICE, once for the tool call and once for the
# answer, with a real person hearing silence for all of it.

async def test_two_turns_share_one_client_instead_of_handshaking_twice(llm, monkeypatch):
    """A course question costs two completions back to back. Each used to open
    its own connection; now both should reuse the same one."""
    llm()
    built = []
    monkeypatch.setattr(
        conversation_llm.httpx, "AsyncClient",
        lambda **kw: built.append(_FakeAsyncClient()) or built[-1],
    )

    await conversation_llm.turn(_HISTORY)
    await conversation_llm.turn(_HISTORY)

    assert len(built) == 1, (
        f"expected one shared client across two turns, got {len(built)} — "
        "turn() is opening a fresh connection per call again"
    )


def test_the_pool_survives_the_gap_between_two_turns():
    """httpx's default keepalive_expiry is 5s. Between two completions on one
    call the agent speaks its answer and the lead takes their own turn —
    commonly 10-20s — so the default would discard the connection between
    every single turn and this whole change would silently do nothing."""
    assert conversation_llm._LIMITS.keepalive_expiry >= 10.0


def test_the_pool_is_sized_for_every_concurrent_call_not_just_one():
    """The pool is process-wide. At MAX_CONCURRENT_CALLS there can be that
    many completions in flight at once, each idling between turns rather than
    closing — so a pool smaller than the call cap evicts live conversations'
    connections and hands them back the TLS handshake this exists to remove."""
    from app.config import MAX_CONCURRENT_CALLS

    assert conversation_llm._LIMITS.max_keepalive_connections >= 2 * MAX_CONCURRENT_CALLS


# ── one retry, and only when it is nearly free ───────────────────────────────
#
# sarvam_bridge._reply swallows TurnFailed, so at campaign scale a single 429
# is not an error the operator sees — it is a lead who hears NOTHING on a call
# that is still graded a clean exit.

async def test_a_rate_limit_is_retried_once(llm, monkeypatch):
    attempts = []

    class _FlakyClient:
        async def post(self, url, json=None, headers=None):
            attempts.append(1)
            code = 429 if len(attempts) == 1 else 200
            return _FakeResponse(code, _reply(content="Twenty five thousand."))

    monkeypatch.setattr(conversation_llm, "_get_client", lambda: _FlakyClient())
    monkeypatch.setattr(conversation_llm, "_RETRY_BACKOFF_S", 0.0)

    reply = await conversation_llm.turn(_HISTORY)

    assert len(attempts) == 2, "a 429 must be retried once"
    assert reply.text == "Twenty five thousand."


async def test_a_bad_request_is_not_retried(llm, monkeypatch):
    """401/400 are configuration errors. Retrying cannot fix them and only
    spends more of the lead's patience."""
    attempts = []

    class _RefusingClient:
        async def post(self, url, json=None, headers=None):
            attempts.append(1)
            return _FakeResponse(401, {})

    monkeypatch.setattr(conversation_llm, "_get_client", lambda: _RefusingClient())

    with pytest.raises(conversation_llm.TurnFailed):
        await conversation_llm.turn(_HISTORY)
    assert len(attempts) == 1


async def test_a_slow_failure_is_not_retried(llm, monkeypatch):
    """A 500 that burned most of the turn budget must not be retried — the
    lead has already been waiting, and a second attempt spends what is left."""
    attempts = []

    class _SlowFailClient:
        async def post(self, url, json=None, headers=None):
            attempts.append(1)
            await asyncio.sleep(0.05)
            return _FakeResponse(500, {})

    monkeypatch.setattr(conversation_llm, "_get_client", lambda: _SlowFailClient())
    monkeypatch.setattr(conversation_llm, "CONVERSATION_LLM_TIMEOUT_S", 0.05)

    with pytest.raises(conversation_llm.TurnFailed):
        await conversation_llm.turn(_HISTORY)
    assert len(attempts) == 1, "a slow failure left no budget for a retry"


async def test_the_client_is_bound_to_the_loop_that_made_it(llm):
    """Same guard as app/redis_client.get_redis() and app/db/pool.get_pool(),
    for the same reason: a client's connections belong to one event loop, and
    pytest-asyncio gives every test function its own."""
    llm()
    await conversation_llm.turn(_HISTORY)
    assert conversation_llm._client_loop is asyncio.get_running_loop()


async def test_closing_twice_is_safe():
    await conversation_llm.aclose()
    await conversation_llm.aclose()
