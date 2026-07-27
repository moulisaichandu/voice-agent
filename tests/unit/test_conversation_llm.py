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
