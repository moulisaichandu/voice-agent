"""telephony/call_routes.py — The Plivo-facing webhooks.

    POST /calls/answer   Plivo fetches this when the lead picks up; we return
                         the <Stream> XML that opens the audio WebSocket.
    WS   /calls/stream   The call audio itself, bridged to the ElevenLabs agent.
    POST /calls/hangup   Why the call ended — including calls that never
                         connected, which otherwise vanish with no reason.

Plivo cannot send APP_AUTH_TOKEN, so these authenticate with
CALL_WEBHOOK_SECRET in the query string, compared with compare_digest.

Concurrency-slot accounting lives here because this is where a call actually
ENDS. The worker acquires a slot before dialling; exactly one of the paths
below must release it, and a call can plausibly hit both (the stream ends,
then Plivo posts the hangup), so release is guarded by a Redis SET NX key —
double-releasing would let the semaphore over-admit and quietly break the
MAX_CONCURRENT_CALLS cap.
"""

from __future__ import annotations

import logging
import secrets
from uuid import UUID

from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import Response

from app import languages, redis_client
from app.compliance.disclosure import has_ai_disclosure
from app.config import CALL_WEBHOOK_SECRET
from app.db import calls as calls_db
from app.db import campaigns as campaigns_db
from app.db import leads as leads_db
from app.db.models import Campaign, Lead
from app.telephony import bridge as bridge_module
from app.telephony import openai_bridge, plivo_client, plivo_stream, sarvam_bridge
from app.telephony import worker as telephony_worker

router = APIRouter(tags=["Telephony"])
logger = logging.getLogger(__name__)

# Long enough to outlast any single call (CALL_MAX_DURATION_S plus slack), so
# the guard is still present when a late hangup arrives after the stream ended.
# Safe to keep generous now that the guard is keyed per ATTEMPT rather than per
# lead — see _release_slot_once.
_SLOT_GUARD_TTL_S = 3600

# The in-flight attempt's SLOT-RELEASE TOKEN. Written by worker.process_one
# and by nothing else — it is the same code path that acquired the slot, so it
# is authoritative, and it is written for every attempt whether or not the
# call is ever answered. The value is Plivo's `request_uuid`, which is unique
# per attempt. Read by /calls/stream and /calls/hangup, which Plivo reaches
# with only ?lead= in the URL.
#
# /calls/answer deliberately does NOT write here any more. It used to
# overwrite this key with Plivo's CallUUID — a DIFFERENT identifier from the
# request_uuid the worker had written. Whenever that overwrite didn't happen
# (the form failed to parse, Redis blipped, or the call was never answered),
# the stream's release resolved the token to `request_uuid` while the hangup
# post used `CallUUID`, so the two wrote different slot:released:* guard keys
# and calls:live:count was decremented TWICE for one call. That over-admits
# past MAX_CONCURRENT_CALLS — the exact failure _release_slot_once exists to
# prevent. One writer, one token.
CALL_UUID_KEY_FMT = "call:uuid:{lead_id}"

# Plivo's real CallUUID for the in-flight attempt, written by /calls/answer.
# Kept separate from the slot token above because it has a different job: it
# is the only id Plivo's DELETE /Call/{uuid}/ endpoint accepts, so it is what
# lets us force-end a leg (see the hangup in stream()'s finally).
CALL_PLIVO_UUID_KEY_FMT = "call:plivo_uuid:{lead_id}"


def _authorized(scope) -> bool:
    """Constant-time comparison of the shared secret.

    An unset secret means open, matching the sibling project's dev behaviour —
    but app/main.py refuses to boot with a public URL and no secret, so this
    only ever happens on a local, non-public server.
    """
    if not CALL_WEBHOOK_SECRET:
        return True
    token = scope.query_params.get("token") or ""
    # Compared as BYTES: compare_digest raises TypeError on a non-ASCII str,
    # and the token here is attacker-controlled. That turned
    # ?token=caf%C3%A9 into a 500 instead of a 403 — which /calls/hangup's
    # docstring explicitly forbids ("always 200, never raises" — an error
    # makes Plivo retry-storm), and which also confirmed the real secret is
    # ASCII by returning a different status than a plain wrong guess.
    return secrets.compare_digest(
        token.encode("utf-8", errors="replace"), CALL_WEBHOOK_SECRET.encode()
    )


async def _release_slot_once(lead_id: str, call_uuid: str | None = None) -> None:
    """Release this call ATTEMPT's concurrency slot, exactly once.

    The guard has to satisfy two opposing requirements at once:

      * dedupe — /calls/stream's `finally` and /calls/hangup BOTH fire for the
        same call, and the slot must be released once, not twice (a double
        release would let the semaphore over-admit);
      * but release once per ATTEMPT — a lead is dialled up to
        campaigns.max_attempts times (default 2), and every attempt acquires
        its own slot in worker.process_one.

    Keying this on lead_id alone satisfied the first and broke the second: the
    key lived for _SLOT_GUARD_TTL_S (1h) while a retry follows roughly one
    CAMPAIGN_TICK_SECONDS (60s) later, so attempt 2's release always found
    attempt 1's key and silently skipped. calls:live:count has no TTL and
    nothing resets it, so each retried lead burned a slot permanently; at
    MAX_CONCURRENT_CALLS the worker stopped dialling altogether while still
    heartbeating, and only a manual DEL recovered it. Found in production with
    4 of 10 slots already gone.

    So the key is a per-attempt provider call id. Resolution order: what
    worker.process_one recorded when it placed this attempt, then the caller's
    own value, then lead_id as a last resort — degrading to the old behaviour
    only when no attempt id can be established at all, which is still better
    than skipping the release.

    The recorded value comes FIRST, and that ordering is load-bearing. It used
    to be the other way round, on the reasoning that /calls/hangup gets a
    CallUUID straight from Plivo's form and so has the freshest value. But
    fresh is not the same as AGREED: the worker records Plivo's `request_uuid`
    and the hangup form carries its `CallUUID`, two different identifiers for
    one call. Preferring the argument meant the stream and the hangup keyed
    their guards differently and both released the same slot. Whichever id
    wins matters far less than both paths choosing the SAME one, and only the
    recorded value is written on every path (including calls that ring out and
    never reach /calls/answer at all).

    Never raises. It runs in /calls/hangup (whose contract is "always 200,
    never raises — an error makes Plivo retry-storm") and in /calls/stream's
    cleanup finally. A Redis failure here must degrade to a logged skip, not an
    exception that escapes the handler and turns a normal teardown into a 500.
    """
    try:
        r = redis_client.get_redis()
        recorded = await r.get(CALL_UUID_KEY_FMT.format(lead_id=lead_id))
        # NOT `recorded or call_uuid or lead_id`. That middle arm reopened
        # exactly the divergence this docstring argues against: with no
        # recorded value the stream (which passes no call_uuid) fell through to
        # lead_id while the hangup fell through to Plivo's CallUUID, so one
        # call wrote two guard keys and released two slots. Both paths now
        # agree in the fallback, and worker.process_one always records an
        # attempt id so the fallback is rarely reached at all.
        token = recorded or lead_id
        first = await r.set(f"slot:released:{token}", "1", nx=True, ex=_SLOT_GUARD_TTL_S)
        if first:
            await telephony_worker.release_call_slot()
    except Exception as exc:
        logger.error(f"[calls] could not release the concurrency slot for lead "
                     f"{lead_id}: {type(exc).__name__}: {exc}")


@router.post("/calls/answer")
async def answer(request: Request) -> Response:
    """Return the <Stream> XML that connects the call's audio to us."""
    lead_id = request.query_params.get("lead", "")
    if not _authorized(request):
        logger.warning(f"[calls] /calls/answer rejected (bad token) lead={lead_id}")
        return Response(status_code=403, content="forbidden")

    # Plivo posts the live CallUUID here; recording it is what makes a
    # force-hangup possible later. Its OWN key — see CALL_PLIVO_UUID_KEY_FMT
    # for why this must not touch the slot-release token.
    try:
        form = await request.form()
        call_uuid = form.get("CallUUID")
    except Exception:
        call_uuid = None
    if call_uuid and lead_id:
        try:
            r = redis_client.get_redis()
            await r.set(CALL_PLIVO_UUID_KEY_FMT.format(lead_id=lead_id), call_uuid,
                        ex=_SLOT_GUARD_TTL_S)
        except Exception as exc:
            # Best-effort: a Redis blip must not fail the answer webhook (500).
            # The call still proceeds via the <Stream> XML below; only the later
            # ability to force-hang-up this leg is lost.
            logger.warning(f"[calls] could not record the Plivo CallUUID for lead "
                           f"{lead_id}: {type(exc).__name__}: {exc}")

    logger.info(f"[calls] answered lead={lead_id} CallUUID={call_uuid}")
    return Response(content=plivo_stream.answer_xml(lead_id),
                    media_type="application/xml")


def _call_language(lead: Lead, campaign: Campaign) -> tuple[str | None, dict]:
    """This call's ElevenLabs language code, and the prompt variables that go
    with it.

    A thin adapter over app/languages.py's for_call() — kept as a function of
    two rows (rather than called inline at the one use site) so the
    precedence rule stays testable without a WebSocket. The actual resolution
    logic lives in for_call() so this and app/telephony/worker.py's dial-time
    preflight call can never drift apart; see for_call()'s own docstring for
    what drifting apart would break.
    """
    return languages.for_call(lead.language_pref, campaign.language)


# Every bridge takes identical arguments and populates the identical `outcome`
# dict, by deliberate design (see openai_bridge's and sarvam_bridge's module
# docstrings), so choosing between them is a table lookup rather than a branch.
# Adding a backend is a row here plus a row in app/languages.py — nothing else
# in the dial path is backend-aware, and _finalise_call, the Sheets write-back
# queue and slot release never learn which one ran.
_BRIDGE_BY_BACKEND = {
    languages.ELEVENLABS: bridge_module.bridge,
    languages.OPENAI_REALTIME: openai_bridge.bridge,
    languages.SARVAM: sarvam_bridge.bridge,
}


def _backend_bridge(token: str):
    """The bridge FUNCTION for *token*'s voice backend — a reference, not a
    branch.

    See app/languages.py's backend_for() for which languages route where and
    why; this must never diverge from that table, so it delegates to it rather
    than repeating the language list here.

    Falls back to the ElevenLabs bridge for a backend with no row above, which
    is the right answer for a hand-built token and the wrong one for a real
    backend somebody forgot to wire — ElevenLabs cannot speak Telugu, so that
    fallback would dial a lead into a wall. tests/unit/test_call_routes.py
    asserts every backend a language can name has a row here, so the omission
    is caught in CI rather than on a live call.
    """
    return _BRIDGE_BY_BACKEND.get(languages.backend_for(token),
                                  bridge_module.bridge)


async def _abort_stream(ws: WebSocket, lead_id: str) -> None:
    """Drop a call we cannot bridge, running the SAME cleanup the normal path
    does: close our socket, hang up the Plivo leg, and release the slot.

    All three are required on every early exit. keepCallAlive means closing the
    socket alone does not end the Plivo leg, and the worker already reserved a
    concurrency slot — skip either and the leg bills on, silent, while the slot
    leaks. Each step is individually best-effort (_end_plivo_leg and
    _release_slot_once swallow their own errors) so one failure can't skip the
    next."""
    try:
        await ws.close()
    except Exception as exc:
        logger.warning(f"[calls] /calls/stream: closing the socket for lead "
                       f"{lead_id} raised {type(exc).__name__}: {exc}")
    await _end_plivo_leg(lead_id)
    await _release_slot_once(lead_id)


@router.websocket("/calls/stream")
async def stream(ws: WebSocket) -> None:
    """The call's audio, bridged to the ElevenLabs agent."""
    lead_id = ws.query_params.get("lead", "")
    if not _authorized(ws):
        logger.warning(f"[calls] /calls/stream rejected (bad token) lead={lead_id}")
        await ws.close(code=1008)
        return

    await ws.accept()

    try:
        try:
            lead = await leads_db.get_lead(UUID(lead_id))
        except (ValueError, AttributeError):
            # A malformed lead id in the URL — not a DB failure.
            lead = None
        if lead is None or lead.campaign_id is None:
            logger.error(f"[calls] /calls/stream: no lead for {lead_id!r} — dropping the call")
            await _abort_stream(ws, lead_id)
            return

        campaign = await campaigns_db.get_campaign(lead.campaign_id)
        if campaign is None:
            logger.error(f"[calls] /calls/stream: no campaign for lead {lead_id} — dropping")
            await _abort_stream(ws, lead_id)
            return
    except Exception as exc:
        # A transient DB error (pool exhaustion, a connection blip) while
        # resolving the lead or campaign must NOT skip cleanup. Only UUID
        # parsing was caught before and get_campaign was unguarded, so a real
        # asyncpg error threw out of the handler before the finally below ever
        # ran — stranding the lead 'calling' forever (due_leads selects only
        # 'pending', so it is never retried), leaking the concurrency slot, and
        # leaving a billed, silent Plivo leg up (keepCallAlive survives our
        # socket closing).
        logger.exception(f"[calls] /calls/stream: could not resolve the call for "
                         f"{lead_id!r} ({type(exc).__name__}: {exc}) — dropping")
        await _abort_stream(ws, lead_id)
        return

    call_language, language_vars = _call_language(lead, campaign)
    # The resolved TOKEN (not the ISO code above) is what picks the backend —
    # 'te' and 'tinglish' both resolve to iso 'te', but only the token tells
    # _backend_bridge which voice backend can carry it. Recomputed here
    # rather than threaded out of _call_language()/for_call() because both are
    # pure, I/O-free functions shared with worker.py's preflight call, and
    # changing their return shape would ripple into a site that has no use
    # for a backend at all.
    bridge_fn = _backend_bridge(languages.resolve(lead.language_pref, campaign.language))

    dynamic_variables = {"lead_id": str(lead.lead_id)}
    if lead.name:
        dynamic_variables["lead_name"] = lead.name
    if campaign.mode == "oneway" and campaign.script:
        dynamic_variables["script"] = campaign.script
    dynamic_variables.update(language_vars)

    # Passed INTO the bridge rather than taken from its return value, so a
    # bridge that raises part-way through still leaves us the turns and the
    # conversation_id it had already collected. Rebinding this name from the
    # return value (as this did) meant an exception discarded the real
    # conversation and recorded an empty, 'failed' call — see bridge()'s
    # *outcome* parameter.
    outcome: dict = {"status": "failed", "turns": 0, "transcript": [],
                     "conversation_id": None}
    try:
        await bridge_fn(
            ws,
            agent_id=campaign.agent_id,
            lead_id=str(lead.lead_id),
            dynamic_variables=dynamic_variables,
            language=call_language,
            one_way=(campaign.mode == "oneway"),
            outcome=outcome,
        )
    except Exception as exc:
        # A failed bridge must still fall through to recording + slot release
        # below, or the lead is left 'calling' and a slot leaks for TTL.
        logger.exception(f"[calls] bridge failed for lead {lead_id}: "
                         f"{type(exc).__name__}: {exc}")
    finally:
        await _finalise_call(lead, outcome)
        await _end_plivo_leg(lead_id)
        await _release_slot_once(lead_id)


async def _end_plivo_leg(lead_id: str) -> None:
    """Hang up the Plivo call now that the bridge is finished.

    The answer XML sets keepCallAlive="true" so the leg survives the audio
    stream — necessary while the bridge runs, but it also means closing our
    WebSocket is not by itself a hangup. Nothing used to close the leg at all:
    plivo_client.hangup() was written, correct, and never called. A one-way
    call that had finished its message, or any call that hit
    CALL_MAX_DURATION_S, was left for the carrier to time out, billing the
    whole way.

    Best-effort by design — plivo_client.hangup swallows its own transport
    errors, and this sits between recording the outcome and releasing the
    slot, neither of which may be skipped because a hangup failed.
    """
    try:
        r = redis_client.get_redis()
        call_uuid = await r.get(CALL_PLIVO_UUID_KEY_FMT.format(lead_id=lead_id))
    except Exception as exc:
        logger.warning(f"[calls] could not look up the CallUUID to hang up "
                       f"lead {lead_id}: {type(exc).__name__}: {exc}")
        return
    if not call_uuid:
        # No CallUUID means /calls/answer never recorded one, which means the
        # leg almost certainly isn't up. Nothing to do.
        return
    await plivo_client.hangup(call_uuid)


def _spoken_disclosure_ok(outcome: dict) -> bool:
    """Whether the agent's FIRST spoken turn disclosed that it is an AI.

    campaigns.script is validated at creation, but on the OpenAI backend the
    operator may write English and the model renders it into Telugu — so the
    validated text is not the spoken text. This checks what was actually said.

    A call where the agent never spoke returns True: the lead heard nothing,
    so there is no disclosure failure to report, only a failed call.
    """
    for turn in outcome.get("transcript") or []:
        if turn.role == "agent":
            return has_ai_disclosure(turn.text)
    return True


async def _finalise_call(lead, outcome: dict) -> None:
    """Record the transcript and queue the Sheets write-back.

    This replaces what the post-call transcript webhook used to do. That
    webhook still exists and is still verified, but with the bridge in the
    media path the transcript is already here the moment the call ends — no
    round-trip, and no dependency on ElevenLabs reaching us.
    """
    conversation_id = outcome.get("conversation_id")
    try:
        # Unconditional, even with no conversation_id. Guarding this on one
        # being present meant that when ElevenLabs refused the handshake — the
        # signature of an account in `past_due`, per
        # elevenlabs_client.subscription_status — no
        # conversation_initiation_metadata ever arrived, so the calls row kept
        # status=NULL and ended_at=NULL forever and the Sheets mirror was
        # written blank. The failure with the most urgent cause looked like
        # nothing having happened.
        await calls_db.set_conversation_and_record(
            lead_id=lead.lead_id,
            el_conversation_id=conversation_id,
            status=outcome.get("status") or "unknown",
            turns=outcome.get("turns") or 0,
            transcript=outcome.get("transcript") or [],
            summary=None,
        )
        # Guarded on status='calling' so this and /calls/hangup — which both
        # fire for the same call, in either order — can't overwrite each
        # other. See mark_result_if_calling in db/leads.py.
        await leads_db.mark_result_if_calling(
            lead.lead_id, "done" if outcome.get("status") == "done" else "failed"
        )
    except Exception as exc:
        logger.exception(f"[calls] could not record outcome for lead "
                         f"{lead.lead_id}: {type(exc).__name__}: {exc}")
        return

    try:
        if not _spoken_disclosure_ok(outcome):
            # Loud on purpose. This is a compliance failure on a call that has
            # already happened to a real person — it cannot be prevented here,
            # only surfaced, and it must never be silent.
            logger.error(
                f"[compliance] lead={lead.lead_id} the agent's FIRST SPOKEN LINE did "
                "not disclose AI. India telecom rules require it. Review this "
                "campaign's script and the agent prompt before dialling more leads."
            )
    except Exception as exc:
        # This is a detective control, not a preventive one — a bug in the
        # check itself must never stop the call from being recorded or the
        # Sheets write-back from being queued below.
        logger.exception(f"[calls] could not check the spoken disclosure for "
                         f"lead {lead.lead_id}: {type(exc).__name__}: {exc}")

    try:
        r = redis_client.get_redis()
        await r.lpush("sheets:writeback:queue", str(lead.lead_id))
    except Exception as exc:
        # Never block or fail a call on the Sheets mirror — CLAUDE.md's rule.
        logger.error(f"[calls] could not queue Sheets write-back for "
                     f"{lead.lead_id}: {type(exc).__name__}: {exc}")


@router.post("/calls/hangup")
async def hangup(request: Request) -> Response:
    """Record why a call ended, and release the slot if no stream ever ran.

    Plivo posts here for EVERY call, including ones that never connected (the
    lead rejected it, the carrier dropped it, our answer URL was unreachable).
    Those never reach /calls/stream, so without this their concurrency slot
    would be held until its guard expired. Always 200, never raises — an error
    here makes Plivo retry-storm.
    """
    lead_id = request.query_params.get("lead", "")
    if not _authorized(request):
        logger.warning(f"[calls] /calls/hangup rejected (bad token) lead={lead_id}")
        return Response(status_code=403, content="forbidden")

    try:
        form = await request.form()
        fields = {k: form.get(k) for k in
                  ("CallUUID", "CallStatus", "HangupCause", "Duration")}
    except Exception:
        fields = {}

    duration = (fields.get("Duration") or "0").strip() or "0"
    if duration in ("0", "0.0"):
        # Duration 0 means the call never carried audio — it died on answer.
        # That is the failure worth shouting about; it is what a dead tunnel
        # or an unreachable answer URL looks like from Plivo's side.
        logger.warning(f"[calls] CALL ENDED WITHOUT CONNECTING lead={lead_id} "
                       f"status={fields.get('CallStatus')} "
                       f"cause={fields.get('HangupCause')}")
        # Back to 'pending', NOT a 'no_answer' status: that string isn't in
        # db/models.py's LeadStatus, and since the DB column is unconstrained
        # text it would write fine and then fail validation on every later
        # read of that row. 'pending' is also the correct retry semantics —
        # due_leads() re-selects it, still bounded by max_attempts because
        # record_dial_attempt() already counted this attempt.
        #
        # Guarded on status='calling': Plivo reports Duration 0 for a call
        # that connected but lasted under a second, and this used to race
        # _finalise_call's done/failed write with no ordering between them.
        # Last-writer-wins could re-dial someone whose call had actually
        # completed. Now whichever lands first wins and the other no-ops,
        # which is correct in both orders — hangup-first leaves the lead
        # retryable, finalise-first leaves it done.
        try:
            if lead_id:
                await leads_db.mark_result_if_calling(UUID(lead_id), "pending")
        except Exception:
            logger.exception(f"[calls] could not release lead {lead_id} for retry")
    else:
        logger.info(f"[calls] call ended lead={lead_id} duration={duration}s "
                    f"cause={fields.get('HangupCause')}")

    if lead_id:
        # Passed as the FALLBACK only. Plivo hands us this attempt's CallUUID
        # directly, which is tempting to prefer — but the worker recorded a
        # request_uuid, and the stream's release reads that, so preferring the
        # form value here would make the two paths key their guards on
        # different ids and release the same slot twice. It is still worth
        # passing: for a call that never reached /calls/answer it beats
        # degrading to lead_id scope. See _release_slot_once.
        await _release_slot_once(lead_id, call_uuid=fields.get("CallUUID"))
    return Response(status_code=200, content="ok")
