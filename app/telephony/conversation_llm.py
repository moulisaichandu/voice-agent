"""telephony/conversation_llm.py — the brain of a two-way call.

Split out of sarvam_llm.py because rendering a script and answering a lead have
opposite constraints, and as of the first live test they cannot use the same
model.

  * RENDERING happens once per campaign, is cached, and never runs while anyone
    is listening. It should use the best Indic model available, however slow.
    That is app/telephony/sarvam_llm.py's render().

  * ANSWERING happens every turn, with a real person holding a phone to their
    ear hearing nothing. It has to be fast, and quality matters less than the
    RAG documents it is quoting from.

Sarvam's chat models cannot do the second job. Measured against the live API on
2026-07-27, both sarvam-105b and sarvam-30b are REASONING models: 400-2000
completion tokens of `reasoning_content` before roughly a hundred characters of
answer. A realistic conversational turn took 21.8 seconds. It cannot be turned
off — `reasoning_effort` accepts only low/medium/high (and 'low' still reasoned
for 17s), `thinking.type=disabled` and `chat_template_kwargs.enable_thinking`
are accepted and silently ignored, and capping `max_tokens` truncates INSIDE
the reasoning so no answer comes back at all.

Both vendors implement the same OpenAI-compatible chat-completions protocol, so
supporting either is a URL and a key rather than a second integration. Sarvam
stays one env var away for whenever their latency story changes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import httpx

from app.config import (
    CONVERSATION_LLM_MODEL,
    CONVERSATION_LLM_PROVIDER,
    CONVERSATION_LLM_TIMEOUT_S,
    OPENAI_API_KEY,
    SARVAM_API_KEY,
)

logger = logging.getLogger(__name__)

OPENAI = "openai"
SARVAM = "sarvam"

# url, default model, and the name of the env var an operator has to set. The
# key itself is read at call time from this module's globals so tests (and a
# reload) see the current value rather than one captured in this table.
_PROVIDERS: dict[str, tuple[str, str, str]] = {
    OPENAI: ("https://api.openai.com/v1/chat/completions",
             "gpt-4o-mini", "OPENAI_API_KEY"),
    SARVAM: ("https://api.sarvam.ai/v1/chat/completions",
             "sarvam-105b", "SARVAM_API_KEY"),
}

SEARCH_TOOL_NAME = "search_course_material"
END_CALL_TOOL_NAME = "end_call"

# Chat-completions tool shape (nested under "function"), NOT the flat shape the
# OpenAI Realtime API uses — app/telephony/openai_bridge.py's tools look
# different for that reason and the two must not be copied between backends.
TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": SEARCH_TOOL_NAME,
            "description": (
                "Search Digital Brolly's course documents for material relevant "
                "to the lead's question. Call this for every question about "
                "courses, fees, timings, batches or placement — never answer "
                "those from memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": ("The lead's question, as a concise "
                                        "search query in English."),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": END_CALL_TOOL_NAME,
            "description": (
                "End the call. Use this together with your farewell the moment "
                "the conversation has finished — the lead has said goodbye, "
                "asked not to be called, or has nothing further to ask."
            ),
            # A required argument, not {}. Measured against the live API: with
            # an empty schema this tool fired 0 times out of 3 on an explicit
            # goodbye; with a required `reason` it fired 1 in 3. Still nowhere
            # near reliable — sarvam_bridge's silence watchdog is what actually
            # ends calls — but it costs nothing and gives the logs a why.
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "enum": ["said_goodbye", "not_interested",
                                 "no_more_questions"],
                        "description": "Why the call is ending.",
                    },
                },
                "required": ["reason"],
            },
        },
    },
]


@dataclass(frozen=True)
class ToolCall:
    """One tool the model asked for, with its arguments already parsed."""

    call_id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class LLMReply:
    """One assistant turn: something to say, tools to run, or both."""

    text: str
    tool_calls: list[ToolCall]


class ConversationNotConfigured(RuntimeError):
    """The chosen provider's API key is not set.

    Names the variable, because which one to set depends on
    CONVERSATION_LLM_PROVIDER and an operator should not have to read the code
    to find out which.
    """


class TurnFailed(RuntimeError):
    """The provider could not be reached, or refused the request."""


class NoAnswer(RuntimeError):
    """A turn with nothing to say and no tool to call.

    Raised rather than returned as an empty string: on a live call, silence
    from an agent a lead just asked a question is indistinguishable from a
    dropped line, so nothing should be able to speak it by accident.
    """


def _provider() -> tuple[str, str, str, str]:
    """(url, model, key, key_name) for the configured provider."""
    url, default_model, key_name = _PROVIDERS.get(
        CONVERSATION_LLM_PROVIDER, _PROVIDERS[OPENAI])
    key = OPENAI_API_KEY if key_name == "OPENAI_API_KEY" else SARVAM_API_KEY
    return url, CONVERSATION_LLM_MODEL or default_model, key or "", key_name


def _parse_tool_calls(message: dict) -> list[ToolCall]:
    """The model's tool_calls, with their JSON-string arguments decoded.

    Malformed arguments become an empty dict rather than raising. A model that
    emits broken JSON should cost one useless tool call, not the conversation —
    and search_relevant("") returns the no-material note, which the agent can
    say out loud instead of going silent on a lead who is waiting.
    """
    parsed: list[ToolCall] = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except (ValueError, TypeError):
            logger.warning(
                f"[conversation] tool {function.get('name')!r} was called with "
                "arguments that are not JSON — treating them as empty."
            )
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        parsed.append(ToolCall(
            call_id=str(raw.get("id") or ""),
            name=str(function.get("name") or ""),
            arguments=arguments,
        ))
    return parsed


async def turn(messages: list[dict]) -> LLMReply:
    """One conversational turn: what to say next, or which tools to run first.

    Deliberately not cached, unlike sarvam_llm.render(). That cache is keyed on
    a script's content and works because one script renders identically for
    every lead; a conversation turn depends on everything said so far, so
    serving a cached one would replay another lead's answer to this lead.

    Non-streaming. Streaming would let the first sentence reach the synthesiser
    before the model finished, which is the obvious next latency win — but
    streaming plus tool calls is a much larger piece of protocol handling, and
    the turn-taking has to be proven correct first.
    """
    url, model, key, key_name = _provider()
    if not key:
        raise ConversationNotConfigured(
            f"{key_name} is not set, but CONVERSATION_LLM_PROVIDER is "
            f"'{CONVERSATION_LLM_PROVIDER}' — a two-way call cannot answer "
            "anything without it."
        )

    payload = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        # Higher than render()'s: this is conversation, where identical
        # phrasing every time sounds robotic, not a translation with one right
        # answer.
        "temperature": 0.6,
    }

    try:
        async with httpx.AsyncClient(timeout=CONVERSATION_LLM_TIMEOUT_S) as client:
            response = await client.post(
                url, json=payload, headers={"Authorization": f"Bearer {key}"})
    except httpx.HTTPError as exc:
        raise TurnFailed(f"{CONVERSATION_LLM_PROVIDER} chat completions "
                         f"unreachable: {type(exc).__name__}: {exc}") from exc

    if response.status_code != 200:
        raise TurnFailed(f"{CONVERSATION_LLM_PROVIDER} chat completions "
                         f"returned HTTP {response.status_code}.")

    try:
        message = response.json()["choices"][0]["message"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise TurnFailed("chat completions returned an unreadable body: "
                         f"{type(exc).__name__}") from exc

    reply = LLMReply(
        text=(message.get("content") or "").strip(),
        tool_calls=_parse_tool_calls(message),
    )
    if not reply.text and not reply.tool_calls:
        # Calling a tool without speaking first is normal — the model looks
        # something up, then answers. Neither speaking NOR calling a tool is
        # not: see NoAnswer.
        raise NoAnswer(
            "the model returned a turn with no content and no tool call "
            f"(reasoning_content was "
            f"{len(message.get('reasoning_content') or '')} characters)"
        )
    return reply
