"""Unit tests for telephony/public_url.py — surviving a rotating tunnel.

Cloudflare quick tunnels mint a BRAND NEW random hostname every time
cloudflared starts. scripts/resolve_public_url.sh already handles that at
BOOT, but it only runs once: if cloudflared restarts while the backend keeps
running, the backend goes on handing Plivo a hostname that now belongs to
nobody, and every call rings and drops on answer with nothing in our logs.

22 distinct hostnames were minted on this machine before anyone noticed the
pattern. This module is what makes that harmless.
"""

import pytest

from app.telephony import public_url


@pytest.fixture(autouse=True)
def _reset():
    """The cache is module state, so a value left by one test would otherwise
    decide the next one's answer."""
    public_url._reset_for_tests()
    yield
    public_url._reset_for_tests()


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeAsyncClient:
    def __init__(self, *, payload=None, raise_error=None, **kw):
        self._payload = payload
        self._raise_error = raise_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        if self._raise_error:
            raise self._raise_error
        return _FakeResponse(self._payload)


def _discovery(monkeypatch, payload=None, raise_error=None):
    monkeypatch.setattr(public_url, "TUNNEL_DISCOVERY_URL", "http://cloudflared:20241")
    monkeypatch.setattr(
        public_url.httpx, "AsyncClient",
        lambda **kw: _FakeAsyncClient(payload=payload, raise_error=raise_error))


# ── before anything is resolved ──────────────────────────────────────────────

def test_base_falls_back_to_the_configured_url(monkeypatch):
    """At import the entrypoint has already resolved one into the environment,
    so the configured value is the right starting point — not blank."""
    monkeypatch.setattr(public_url, "PUBLIC_BASE_URL", "https://configured.example")
    assert public_url.base() == "https://configured.example"


def test_base_is_a_string_even_with_nothing_configured(monkeypatch):
    """Callers build URLs by concatenation; None would produce 'None/calls/answer'."""
    monkeypatch.setattr(public_url, "PUBLIC_BASE_URL", None)
    assert public_url.base() == ""


# ── refreshing ───────────────────────────────────────────────────────────────

async def test_refresh_picks_up_the_live_tunnel(monkeypatch):
    monkeypatch.setattr(public_url, "PUBLIC_BASE_URL", "https://stale.example")
    _discovery(monkeypatch, payload={"hostname": "fresh-name.trycloudflare.com"})

    resolved = await public_url.refresh()

    assert resolved == "https://fresh-name.trycloudflare.com"
    assert public_url.base() == "https://fresh-name.trycloudflare.com"


async def test_a_rotation_is_picked_up_without_a_restart(monkeypatch):
    """THE failure this module exists for. cloudflared restarts mid-run and
    mints a new hostname; nothing restarts the backend, so without this the
    old one is handed to Plivo until someone notices."""
    monkeypatch.setattr(public_url, "PUBLIC_BASE_URL", "https://first.example")
    _discovery(monkeypatch, payload={"hostname": "first.example"})
    await public_url.refresh()
    assert public_url.base() == "https://first.example"

    _discovery(monkeypatch, payload={"hostname": "second.trycloudflare.com"})
    await public_url.refresh()

    assert public_url.base() == "https://second.trycloudflare.com"


async def test_an_unreachable_cloudflared_keeps_the_last_known_url(monkeypatch):
    """A discovery blip must not blank the URL — that would turn a momentary
    hiccup into 'PUBLIC_BASE_URL is not set' and stop dialling entirely."""
    monkeypatch.setattr(public_url, "PUBLIC_BASE_URL", "https://configured.example")
    _discovery(monkeypatch, payload={"hostname": "good.trycloudflare.com"})
    await public_url.refresh()

    _discovery(monkeypatch, raise_error=ConnectionError("cloudflared is down"))
    resolved = await public_url.refresh()

    assert resolved == "https://good.trycloudflare.com"
    assert public_url.base() == "https://good.trycloudflare.com"


async def test_discovery_disabled_keeps_the_configured_url(monkeypatch):
    """Production: PUBLIC_BASE_URL is a real domain and there is no cloudflared
    service, so TUNNEL_DISCOVERY_URL is blank and nothing must override it."""
    monkeypatch.setattr(public_url, "PUBLIC_BASE_URL", "https://voice.example.com")
    monkeypatch.setattr(public_url, "TUNNEL_DISCOVERY_URL", "")

    def must_not_be_called(**kw):
        raise AssertionError("must not ask cloudflared when discovery is disabled")

    monkeypatch.setattr(public_url.httpx, "AsyncClient", must_not_be_called)

    assert await public_url.refresh() == "https://voice.example.com"


async def test_a_malformed_discovery_reply_is_ignored(monkeypatch):
    monkeypatch.setattr(public_url, "PUBLIC_BASE_URL", "https://configured.example")
    _discovery(monkeypatch, payload={"not_a_hostname": True})

    assert await public_url.refresh() == "https://configured.example"


async def test_refresh_never_raises(monkeypatch):
    """It runs on the dial path. An exception here would abort a dial instead
    of degrading to the last-known URL."""
    monkeypatch.setattr(public_url, "PUBLIC_BASE_URL", "https://configured.example")
    _discovery(monkeypatch, raise_error=RuntimeError("anything at all"))

    assert await public_url.refresh() == "https://configured.example"


# ── the consumers actually read the refreshed value ─────────────────────────
#
# Resolving correctly is worthless if the two places that build URLs for Plivo
# still read a value bound at import.

async def test_the_stream_url_plivo_connects_back_on_follows_a_rotation(monkeypatch):
    """answer_xml builds the wss:// URL Plivo opens the audio stream to. Bound
    to a rotated-away hostname, the call connects and then goes silent."""
    from app.telephony import plivo_stream

    monkeypatch.setattr(public_url, "PUBLIC_BASE_URL", "https://stale.example")
    _discovery(monkeypatch, payload={"hostname": "fresh.trycloudflare.com"})
    await public_url.refresh()

    xml = plivo_stream.answer_xml("lead-1")

    assert "wss://fresh.trycloudflare.com/calls/stream" in xml
    assert "stale.example" not in xml


async def test_the_answer_url_plivo_is_given_follows_a_rotation(monkeypatch):
    """_webhook_urls is what Plivo is TOLD to fetch when the lead picks up. A
    stale one means the phone rings and the call drops on answer, with nothing
    in our logs because the request never arrives."""
    from app.telephony import plivo_client

    monkeypatch.setattr(public_url, "PUBLIC_BASE_URL", "https://stale.example")
    _discovery(monkeypatch, payload={"hostname": "fresh.trycloudflare.com"})
    await public_url.refresh()

    answer_url, hangup_url = plivo_client._webhook_urls("lead-1")

    assert answer_url.startswith("https://fresh.trycloudflare.com/calls/answer")
    assert hangup_url.startswith("https://fresh.trycloudflare.com/calls/hangup")


async def test_preflight_refreshes_before_it_probes(monkeypatch):
    """Preflight runs once per campaign_tick batch, immediately before any
    dialling — the one moment where being current actually matters. Without
    this the refresh would only ever happen by luck."""
    from app.telephony import preflight as pf

    refreshed = {"n": 0}

    async def counting_refresh():
        refreshed["n"] += 1
        return "https://fresh.trycloudflare.com"

    monkeypatch.setattr(pf.public_url, "refresh", counting_refresh)
    monkeypatch.setattr(pf.public_url, "base", lambda: "https://fresh.trycloudflare.com")
    monkeypatch.setattr(pf, "ELEVENLABS_API_KEY", "key")
    monkeypatch.setattr(pf, "PLIVO_AUTH_ID", "id")
    monkeypatch.setattr(pf, "PLIVO_AUTH_TOKEN", "token")
    monkeypatch.setattr(pf, "PLIVO_FROM_NUMBER", "+918035383564")

    class _Resp:
        status_code = 200
        text = "<Response><Stream>ws://x</Stream></Response>"

    class _Client:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url):
            assert url.startswith("https://fresh.trycloudflare.com"), url
            return _Resp()

    monkeypatch.setattr(pf.httpx, "AsyncClient", lambda **kw: _Client())
    monkeypatch.setattr(pf, "agent_exists", lambda agent_id: True)

    async def good_billing(*, force_refresh=False):
        return {"status": "active", "character_count": 1, "character_limit": 2}

    monkeypatch.setattr(pf, "cached_subscription_status", good_billing)

    async def not_tripped():
        return None

    monkeypatch.setattr(pf.sarvam_circuit_breaker, "tripped_reason", not_tripped)

    await pf.preflight("agent_1")

    assert refreshed["n"] == 1, "preflight must re-resolve the tunnel before probing"
