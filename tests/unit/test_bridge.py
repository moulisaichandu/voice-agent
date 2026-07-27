"""Unit tests for telephony/bridge.py — the ElevenLabs side of a call.

The bridge loop is driven here against a pair of fake WebSockets. That is
worth the fakes because the loop's two failure modes are both expensive and
both invisible in a passing call: a one-way call that never hangs up bills
CALL_MAX_DURATION_S of silence to every lead, and an exception mid-call used
to throw away the transcript of a conversation that really happened.

The Plivo side of the call — the answer XML, the audio channel and the one-way
watchdog — now lives in app/telephony/plivo_stream.py and is tested in
tests/unit/test_plivo_stream.py. The end-to-end behaviour of both together is
still asserted below, through bridge() itself.
"""

import asyncio
import base64
import json

import pytest

from app.telephony import bridge, plivo_stream

# 80 ms of mu-law: long enough to be realistic, short enough that waiting for
# it to "play out" costs the test nothing.
_AUDIO_B64 = base64.b64encode(b"\xff" * 640).decode()


def _metadata(fmt: str = "ulaw_8000") -> str:
    return json.dumps({
        "type": "conversation_initiation_metadata",
        "conversation_initiation_metadata_event": {
            "conversation_id": "conv_test_1", "agent_output_audio_format": fmt,
        },
    })


def _audio() -> str:
    return json.dumps({"type": "audio", "audio_event": {"audio_base_64": _AUDIO_B64}})


def _agent_says(text: str) -> str:
    return json.dumps({
        "type": "agent_response", "agent_response_event": {"agent_response": text},
    })


class _FakeElevenLabsWS:
    """Yields *messages*, then goes quiet forever — which is exactly what the
    real service does on a one-way call: with no user audio forwarded it has
    no turn to end, so it never closes the socket."""

    def __init__(self, messages: list[str], *, raise_on_exit: Exception | None = None):
        self._messages = list(messages)
        self._raise_on_exit = raise_on_exit
        self.sent: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        if self._raise_on_exit is not None:
            raise self._raise_on_exit
        return False

    async def send(self, payload):
        self.sent.append(payload)

    def __aiter__(self):
        async def _gen():
            for m in self._messages:
                yield m
            await asyncio.Event().wait()  # never closes on its own
        return _gen()


class _FakePlivoWS:
    """A lead who has answered and is saying nothing — the normal case on a
    one-way call."""

    def __init__(self):
        self.sent: list[dict] = []

    async def receive_text(self):
        await asyncio.Event().wait()

    async def send_text(self, raw):
        self.sent.append(json.loads(raw))


@pytest.fixture
def bridged(monkeypatch):
    """Wire bridge() to fakes and make the one-way timings test-fast."""
    monkeypatch.setattr(bridge, "ELEVENLABS_API_KEY", "sk-test")
    # No forced voice by default. config.py calls load_dotenv(), so without this
    # a real ELEVENLABS_VOICE_ID in .env would decide whether the "auto sends no
    # override" guard below passes — making the suite depend on whose machine it
    # runs on. Tests that want a forced voice set it explicitly.
    monkeypatch.setattr(bridge, "ELEVENLABS_VOICE_ID", None)
    monkeypatch.setattr(plivo_stream, "ONEWAY_SILENCE_TAIL_S", 0)
    monkeypatch.setattr(plivo_stream, "ONEWAY_MAX_SILENT_S", 0)

    async def fake_signed_url(agent_id):
        return "wss://elevenlabs.invalid/signed"

    monkeypatch.setattr(bridge, "_signed_url", fake_signed_url)

    def _install(el_ws):
        monkeypatch.setattr(bridge.websockets, "connect", lambda *a, **kw: el_ws)

    return _install


async def test_a_one_way_call_ends_once_the_message_has_played(bridged):
    """REGRESSION, and the most expensive bug this suite covers. A one-way
    bridge never forwards the lead's audio, so ElevenLabs gets no turn-end
    signal and never closes; from_plivo only returns if the lead hangs up.
    Nothing set `stop`, so the bridge sat on CALL_MAX_DURATION_S — 300
    seconds of billed Plivo airtime and a 300-second ElevenLabs conversation
    for a 20-second message, with the lead listening to silence throughout."""
    el_ws = _FakeElevenLabsWS([_metadata(), _agent_says("This is an AI call."), _audio()])
    bridged(el_ws)
    plivo_ws = _FakePlivoWS()

    outcome = await asyncio.wait_for(
        bridge.bridge(plivo_ws, agent_id="agent_1", lead_id="lead-1", one_way=True),
        timeout=5,
    )

    assert outcome["status"] == "done", (
        "a one-way call that delivered its message ran its course — recording it "
        "'failed' would mark the lead failed and burn a retry on a perfect call"
    )
    assert outcome["conversation_id"] == "conv_test_1"
    assert [t.text for t in outcome["transcript"]] == ["This is an AI call."]
    assert any(m["event"] == "playAudio" for m in plivo_ws.sent)


async def test_a_one_way_call_with_a_silent_agent_gives_up(bridged):
    """An agent that never speaks (a broken prompt, a dynamic variable it is
    waiting on) must cost ONEWAY_MAX_SILENT_S, not CALL_MAX_DURATION_S."""
    el_ws = _FakeElevenLabsWS([_metadata()])
    bridged(el_ws)

    outcome = await asyncio.wait_for(
        bridge.bridge(_FakePlivoWS(), agent_id="agent_1", lead_id="lead-1", one_way=True),
        timeout=5,
    )

    assert outcome["status"] == "failed", "nothing was said, so nothing was delivered"
    assert outcome["turns"] == 0


async def test_a_two_way_call_is_not_ended_by_the_watchdog(bridged, monkeypatch):
    """The watchdog must run for one-way calls ONLY. On a two-way call the
    silence after the agent's greeting is the lead thinking about their reply;
    hanging up on it would cut off every conversation at hello."""
    monkeypatch.setattr(plivo_stream, "CALL_MAX_DURATION_S", 1)
    el_ws = _FakeElevenLabsWS([_metadata(), _audio()])
    bridged(el_ws)

    loop = asyncio.get_running_loop()
    started = loop.time()
    await asyncio.wait_for(
        bridge.bridge(_FakePlivoWS(), agent_id="agent_1", lead_id="lead-1", one_way=False),
        timeout=5,
    )
    elapsed = loop.time() - started

    # Asserted on ELAPSED TIME, not on the outcome: a wrongly-running watchdog
    # would exit with "oneway_complete", which is also a 'done' status, so
    # only the timing distinguishes the two. The fakes fall silent
    # immediately, so anything near the 1s cap means nothing cut the call
    # short — with ONEWAY_SILENCE_TAIL_S monkeypatched to 0, a live watchdog
    # would have ended it within a poll interval.
    assert elapsed >= 0.9, f"the call was ended early, after {elapsed:.2f}s"


async def test_a_two_way_call_honours_barge_in_with_clearaudio(bridged, monkeypatch):
    """REGRESSION: the ElevenLabs backend is the product's only two-way backend,
    and the lead must be able to interrupt the agent. The opening-disclosure
    guard added for the OpenAI path (_opening_pending, previously default-on) is
    cleared only by mark_opening_delivered(), which this bridge never calls — so
    interrupt() returned False on EVERY turn and clearAudio was never sent,
    killing barge-in for the whole call. An `interruption` event after some audio
    must drop the buffered agent speech (clearAudio)."""
    monkeypatch.setattr(plivo_stream, "CALL_MAX_DURATION_S", 1)
    el_ws = _FakeElevenLabsWS([
        _metadata(), _audio(),
        json.dumps({"type": "interruption", "interruption_event": {}}),
    ])
    bridged(el_ws)
    plivo_ws = _FakePlivoWS()

    await asyncio.wait_for(
        bridge.bridge(plivo_ws, agent_id="agent_1", lead_id="lead-1", one_way=False),
        timeout=5,
    )

    assert any(m["event"] == "clearAudio" for m in plivo_ws.sent), \
        "barge-in must drop the buffered agent audio (clearAudio) on the ElevenLabs path"


async def test_a_mid_call_failure_still_yields_what_was_collected(bridged):
    """REGRESSION. bridge()'s results used to be reachable only through its
    return value, and call_routes kept its own default dict — so an exception
    escaping the `async with` discarded a real conversation and recorded zero
    turns, no conversation_id, and a 'failed' lead that then burned a retry."""
    el_ws = _FakeElevenLabsWS(
        [_metadata(), _agent_says("Hello."), _audio()],
        raise_on_exit=RuntimeError("socket died on close"),
    )
    bridged(el_ws)

    outcome: dict = {}
    with pytest.raises(RuntimeError):
        await asyncio.wait_for(
            bridge.bridge(_FakePlivoWS(), agent_id="agent_1", lead_id="lead-1",
                          one_way=True, outcome=outcome),
            timeout=5,
        )

    assert outcome["conversation_id"] == "conv_test_1"
    assert outcome["turns"] == 1
    assert [t.text for t in outcome["transcript"]] == ["Hello."]


# ── the language override ────────────────────────────────────────────────────

def _initiation(el_ws) -> dict:
    """The first frame the bridge sends — the initiation payload."""
    return json.loads(el_ws.sent[0])


async def test_no_language_sends_no_override(bridged):
    """THE regression guard for this whole feature. ElevenLabs raises if an
    override arrives for a field that isn't enabled in the agent's Security
    tab, so a campaign left on 'auto' must send exactly what it always sent —
    otherwise enabling this feature breaks every existing campaign."""
    el_ws = _FakeElevenLabsWS([_metadata(), _agent_says("Hi."), _audio()])
    bridged(el_ws)

    await asyncio.wait_for(
        bridge.bridge(_FakePlivoWS(), agent_id="agent_1", lead_id="lead-1",
                      one_way=True),
        timeout=5,
    )

    assert "conversation_config_override" not in _initiation(el_ws)


async def test_a_language_is_sent_as_a_conversation_config_override(bridged):
    el_ws = _FakeElevenLabsWS([_metadata(), _agent_says("Hi."), _audio()])
    bridged(el_ws)

    await asyncio.wait_for(
        bridge.bridge(_FakePlivoWS(), agent_id="agent_1", lead_id="lead-1",
                      language="te", one_way=True),
        timeout=5,
    )

    assert _initiation(el_ws)["conversation_config_override"] == {
        "agent": {"language": "te"}
    }


async def test_the_override_does_not_disturb_the_dynamic_variables(bridged):
    """Both travel in the same frame; adding one must not drop the other —
    lead_id in particular is how the transcript maps back to a lead."""
    el_ws = _FakeElevenLabsWS([_metadata(), _agent_says("Hi."), _audio()])
    bridged(el_ws)

    await asyncio.wait_for(
        bridge.bridge(_FakePlivoWS(), agent_id="agent_1", lead_id="lead-1",
                      dynamic_variables={"lead_id": "lead-1"},
                      language="hi", one_way=True),
        timeout=5,
    )

    frame = _initiation(el_ws)
    assert frame["dynamic_variables"] == {"lead_id": "lead-1"}
    assert frame["conversation_config_override"]["agent"]["language"] == "hi"
    assert frame["type"] == "conversation_initiation_client_data"


# ── the forced voice override (ELEVENLABS_VOICE_ID) ──────────────────────────
# The voice is sent on EVERY ElevenLabs-backed call, independently of language,
# because it exists to override an agent whose dashboard voice (or per-language
# preset) is not what the operator wants. app/telephony/preflight.py is what
# guarantees the agent actually allows the field.

_VOICE = "ohvvU75FpBEB8fdaLOMh"


async def test_the_configured_voice_is_forced_on_an_auto_campaign(bridged, monkeypatch):
    """The whole point of forcing from code: an 'auto' campaign previously sent
    NO override key at all, so a dashboard voice change was the only lever and a
    per-language preset could silently win. It must now carry the voice."""
    monkeypatch.setattr(bridge, "ELEVENLABS_VOICE_ID", _VOICE)
    el_ws = _FakeElevenLabsWS([_metadata(), _agent_says("Hi."), _audio()])
    bridged(el_ws)

    await asyncio.wait_for(
        bridge.bridge(_FakePlivoWS(), agent_id="agent_1", lead_id="lead-1",
                      one_way=True),
        timeout=5,
    )

    assert _initiation(el_ws)["conversation_config_override"] == {
        "tts": {"voice_id": _VOICE}
    }


async def test_the_voice_and_the_language_travel_in_one_override(bridged, monkeypatch):
    """Both are parts of the SAME override dict — adding the voice must not
    displace the language (or vice versa)."""
    monkeypatch.setattr(bridge, "ELEVENLABS_VOICE_ID", _VOICE)
    el_ws = _FakeElevenLabsWS([_metadata(), _agent_says("Hi."), _audio()])
    bridged(el_ws)

    await asyncio.wait_for(
        bridge.bridge(_FakePlivoWS(), agent_id="agent_1", lead_id="lead-1",
                      language="hi", one_way=True),
        timeout=5,
    )

    assert _initiation(el_ws)["conversation_config_override"] == {
        "agent": {"language": "hi"},
        "tts": {"voice_id": _VOICE},
    }


async def test_a_blank_voice_id_sends_no_tts_override(bridged, monkeypatch):
    """Clearing the var in .env is the rollback, so it must restore the old
    frame exactly — not send an empty voice_id, which would fail every call."""
    monkeypatch.setattr(bridge, "ELEVENLABS_VOICE_ID", "")
    el_ws = _FakeElevenLabsWS([_metadata(), _agent_says("Hi."), _audio()])
    bridged(el_ws)

    await asyncio.wait_for(
        bridge.bridge(_FakePlivoWS(), agent_id="agent_1", lead_id="lead-1",
                      one_way=True),
        timeout=5,
    )

    assert "conversation_config_override" not in _initiation(el_ws)
