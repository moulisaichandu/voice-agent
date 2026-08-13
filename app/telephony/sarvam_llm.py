"""telephony/sarvam_llm.py — Sarvam's LLM, as this backend uses it.

One job: turn the operator's campaign script into the Telugu the lead actually
hears. The two-way conversation loop deliberately does NOT live here — see
app/telephony/conversation_llm.py, and the timeout comment below, for why the
two jobs ended up on different models.

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
import logging

import httpx

from app.compliance.disclosure import has_ai_disclosure
from app.config import SARVAM_API_KEY, SARVAM_LLM_MODEL
from app.redis_client import get_redis
from app.telephony import sarvam_circuit_breaker, sarvam_prompts

logger = logging.getLogger(__name__)

_CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"

# Generous, and measured rather than guessed. Both Sarvam chat models are
# REASONING models: every response spends 400-2000 completion tokens on
# `reasoning_content` before emitting ~100 characters of actual `content`.
# Against the live API on 2026-07-27, rendering an ordinary three-sentence
# script took 14-22s on sarvam-105b and 6.7s on sarvam-30b.
#
# The reasoning cannot be turned off. `reasoning_effort` accepts only
# low/medium/high (not "none"), and "low" still reasoned for 17s;
# `thinking.type=disabled` and `chat_template_kwargs.enable_thinking=False`
# were both accepted and ignored. Capping max_tokens does not help either — it
# truncates INSIDE the reasoning, so `content` comes back null.
#
# This was 20s, chosen before anyone had run it against the real API, and the
# very first live render failed with ReadTimeout.
_TIMEOUT_S = 60.0

# How much of a failed response body is kept. Enough for Sarvam's own JSON
# error objects, short enough that an upstream HTML error page cannot become
# the breaker reason string shown on the readiness dashboard.
_MAX_ERROR_BODY = 300

# Sarvam's own default completion budget is 2048 tokens, and its reasoning
# routinely spends all of it before writing a word of the answer. Measured over
# four identical requests with no max_tokens: two came back
# finish_reason=length at exactly 2048 with content=0 characters — a render
# that "returned no text", intermittently, on a script that had worked minutes
# earlier. A failed render means the campaign does not dial at all.
#
# The same prompt with max_tokens=4000 succeeded 3/3, spending 1493-2234
# tokens. The ceiling has to sit well clear of the reasoning, not near it.
_MAX_TOKENS = 4096

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
        # The body, not just the status. A bare "HTTP 402" is a fact; "No
        # credits available" is the diagnosis, and discarding it is how this
        # leg stayed silent about an account outage that STT and TTS both
        # reported. Truncated because an upstream HTML error page must not
        # become the breaker reason an admin reads on the dashboard.
        detail = (getattr(response, "text", "") or "")[:_MAX_ERROR_BODY].strip()
        if sarvam_circuit_breaker.looks_like_credits_exhausted(detail):
            # The same account, the same failure, the same breaker as the STT
            # and TTS legs — rendering just reaches it over HTTP instead of a
            # WebSocket. Without this, a campaign warmed while the account was
            # empty would leave preflight happily dialling.
            await sarvam_circuit_breaker.trip(detail)
        raise SarvamRenderFailed(
            f"Sarvam chat completions returned HTTP {response.status_code}: "
            f"{detail or '(no body)'}"
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
        # See _MAX_TOKENS: below this the model runs out of budget mid-thought
        # and returns no answer at all.
        "max_tokens": _MAX_TOKENS,
    })
    text = (message.get("content") or "").strip()
    if not text:
        raise SarvamRenderFailed("Sarvam chat completions returned no text.")
    return text


def ensure_disclosure(text: str) -> str:
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

    rendered = ensure_disclosure(
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
