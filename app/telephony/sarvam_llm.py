"""telephony/sarvam_llm.py — Sarvam's LLM, as this backend uses it.

Today it does one job: turn the operator's campaign script into the Telugu the
lead actually hears. Milestone B adds the two-way turn loop here; the client
and the auth live in one place so that arrives as a function, not a second
vendor integration.

WHY A COMPLETION AT ALL ON A ONE-WAY CALL. The operator writes campaign scripts
in English. ElevenLabs and OpenAI Realtime both speak, so their model rendered
the Telugu itself as part of speaking. Sarvam's TTS only reads text, so
something has to write that text first.

TWO THINGS THIS MODULE OWNS BEYOND THE HTTP CALL:

  * The cache. A script belongs to a CAMPAIGN, not to a lead, so the identical
    completion would otherwise be paid for and waited on once per call — and
    the wait lands as silence on an answered phone, before the lead has heard
    a word. Rendered text is cached in Redis by content hash. Redis is
    ephemeral by CLAUDE.md's rule, so a cache outage costs latency, never a
    call.

  * The disclosure repair. This is the one compliance improvement this backend
    makes possible. On the OpenAI path the model speaks directly, so a dropped
    AI disclosure can only be DETECTED after the call
    (call_routes._spoken_disclosure_ok). Here the exact Telugu exists as a
    string before anything is spoken, so a dropped disclosure is fixed instead.
    The post-call check stays exactly as it is — it is the backstop for
    everything this cannot see.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass

import httpx

from app.compliance.disclosure import has_ai_disclosure
from app.config import SARVAM_API_KEY, SARVAM_LLM_MODEL
from app.redis_client import get_redis
from app.telephony import sarvam_prompts

logger = logging.getLogger(__name__)

_CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"
_TIMEOUT_S = 20.0

# A rendered script changes only when the script does, and the key is a content
# hash, so a stale entry is unreachable rather than wrong. The TTL exists to
# stop abandoned campaigns' renders living in Redis forever.
_CACHE_PREFIX = "sarvam:render:"
_CACHE_TTL_S = 7 * 24 * 3600

# Prepended when the model returns Telugu that does not disclose the call is
# automated. Deliberately a fixed string rather than another completion: the
# repair path must not be able to fail the same way the thing it is repairing
# did. "కృత్రిమ మేధ" (artificial intelligence) is one of the markers
# app/compliance/disclosure.py already recognises.
_DISCLOSURE_PREFIX = "ఇది కృత్రిమ మేధ ద్వారా చేసే ఆటోమేటెడ్ కాల్."


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
                "End the call. Use this once the conversation has genuinely "
                "finished — the lead has said goodbye, asked not to be called, "
                "or has nothing further to ask."
            ),
            "parameters": {"type": "object", "properties": {}},
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


class SarvamNotConfigured(RuntimeError):
    """No SARVAM_API_KEY. Raised before calling out, so the bridge turns it
    into a failed outcome rather than discovering it mid-call."""


class SarvamRenderFailed(RuntimeError):
    """The script could not be rendered into Telugu.

    Deliberately fatal to the call. The tempting fallback — speak the English
    script as written — would ring a Telugu-speaking lead and read English at
    them, which is precisely the failure the prompt rules exist to prevent.
    A call that does not happen beats a call that happens badly.
    """


def _cache_key(script: str, language_style: str | None) -> str:
    """Content hash of everything that changes the rendered output.

    language_style is in the key because 'te' and 'tinglish' produce genuinely
    different speech from the same script; the model is, because changing it
    changes the voice's phrasing and the old render should not be served as if
    it came from the new one.
    """
    material = "\x00".join([script, language_style or "", SARVAM_LLM_MODEL])
    return _CACHE_PREFIX + hashlib.sha256(material.encode("utf-8")).hexdigest()


async def _cache_get(key: str) -> str | None:
    try:
        return await get_redis().get(key)
    except Exception as exc:  # noqa: BLE001 - a cache miss and a cache outage are the same to the caller
        logger.warning(f"[sarvam-llm] cache read failed, rendering live: "
                       f"{type(exc).__name__}: {exc}")
        return None


async def _cache_put(key: str, value: str) -> None:
    try:
        await get_redis().set(key, value, ex=_CACHE_TTL_S)
    except Exception as exc:  # noqa: BLE001 - never let the cache fail a call
        logger.warning(f"[sarvam-llm] cache write failed: "
                       f"{type(exc).__name__}: {exc}")


async def _post_chat(payload: dict) -> dict:
    """One chat-completions request. Returns the assistant `message` object.

    Sarvam's API is OpenAI-compatible, so this is the ordinary shape: a
    `choices[0].message` with `content` and, when the model calls a tool,
    `tool_calls`.
    """
    headers = {"Authorization": f"Bearer {SARVAM_API_KEY}"}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            response = await client.post(_CHAT_URL, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        raise SarvamRenderFailed(
            f"Sarvam chat completions unreachable: {type(exc).__name__}: {exc}"
        ) from exc

    if response.status_code != 200:
        raise SarvamRenderFailed(
            f"Sarvam chat completions returned HTTP {response.status_code}."
        )

    try:
        return response.json()["choices"][0]["message"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise SarvamRenderFailed(
            f"Sarvam chat completions returned an unreadable body: "
            f"{type(exc).__name__}"
        ) from exc


async def _complete(system_prompt: str, user_prompt: str) -> str:
    """One non-streaming chat completion, for rendering a script."""
    message = await _post_chat({
        "model": SARVAM_LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # Low but not zero: this is a rendering task with one right meaning and
        # some freedom in phrasing, not a creative one.
        "temperature": 0.3,
    })
    text = (message.get("content") or "").strip()
    if not text:
        raise SarvamRenderFailed("Sarvam chat completions returned no text.")
    return text


def _ensure_disclosure(text: str) -> str:
    """*text*, guaranteed to open with an AI disclosure.

    Logged at WARNING with the [compliance] prefix the operator already greps
    for, but not fatal: the call is about to happen and a compliant opening is
    strictly better than a refused campaign. The ERROR-level [compliance] line
    in call_routes still fires if something downstream of here goes wrong.
    """
    if has_ai_disclosure(text):
        return text
    logger.warning(
        "[compliance] the rendered Telugu did not disclose the call is "
        "automated — prepending the standard disclosure. India telecom rules "
        "require it as the first sentence. Check this campaign's script: the "
        "model is dropping the disclosure when it renders."
    )
    return f"{_DISCLOSURE_PREFIX} {text}"


async def render(script: str, *, language_style: str | None = None) -> str:
    """*script*, as the Telugu a lead will hear. Cached by content.

    Returns "" for a blank script rather than paying for a completion that
    renders nothing — a one-way campaign cannot be created without a script,
    so a blank one means a caller bypassed that guard.
    """
    cleaned = (script or "").strip()
    if not cleaned:
        return ""
    if not SARVAM_API_KEY:
        raise SarvamNotConfigured(
            "SARVAM_API_KEY is not set — a Telugu script cannot be rendered."
        )

    key = _cache_key(cleaned, language_style)
    cached = await _cache_get(key)
    if cached:
        return cached

    rendered = _ensure_disclosure(
        await _complete(
            sarvam_prompts.render_instructions(cleaned, language_style=language_style),
            cleaned,
        )
    )
    # Cache the REPAIRED text, never the raw model output: a cached
    # non-compliant opening would be served to every later call on the
    # campaign, and would look compliant to anyone who trusted the cache.
    await _cache_put(key, rendered)
    return rendered


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
                f"[sarvam-llm] tool {function.get('name')!r} was called with "
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

    Deliberately NOT cached. render()'s cache is keyed on a script's content
    and works because one script renders identically for every lead; a
    conversation turn depends on everything said so far, so serving a cached
    one would replay another lead's answer to this lead.

    Non-streaming, for now. Streaming would let the first sentence reach the
    synthesiser before the model has finished thinking, which is the obvious
    latency win — but streaming and tool calls together is a much larger piece
    of protocol handling, and the first thing to establish is whether the
    turn-taking is right at all. Measure before optimising: the live-call gate
    exists to produce that number.
    """
    if not SARVAM_API_KEY:
        raise SarvamNotConfigured(
            "SARVAM_API_KEY is not set — the agent cannot hold a conversation."
        )

    message = await _post_chat({
        "model": SARVAM_LLM_MODEL,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        # Higher than render()'s: this is conversation, where identical
        # phrasing every time sounds robotic, not a translation with one right
        # answer.
        "temperature": 0.6,
    })
    return LLMReply(
        text=(message.get("content") or "").strip(),
        tool_calls=_parse_tool_calls(message),
    )
