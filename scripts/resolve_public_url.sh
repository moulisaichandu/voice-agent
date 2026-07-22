#!/bin/sh
# resolve_public_url.sh — the backend container's entrypoint.
#
# Works out PUBLIC_BASE_URL and then execs the real command.
#
# WHY THIS EXISTS
# ---------------
# Plivo has to fetch our answer URL over the public internet, so a dev machine
# needs a tunnel. Cloudflare "quick tunnels" mint a BRAND NEW random hostname
# every time they start — so a hostname written into .env is stale the moment
# cloudflared restarts, and preflight then fails with
# "PUBLIC_BASE_URL (...) is unreachable ... Is the tunnel running?".
#
# That failed daily here: the terminal running cloudflared gets closed, or the
# machine reboots, and the URL in .env no longer belongs to anything. Editing
# .env by hand fixes it until the next restart, which is not a fix.
#
# So the tunnel now runs as a compose service, and this script ASKS it for the
# current hostname instead of anyone hardcoding one. cloudflared serves that on
# its metrics port:  GET /quicktunnel -> {"hostname":"foo.trycloudflare.com"}
#
# Precedence, highest first:
#   1. An explicit PUBLIC_BASE_URL that is actually REACHABLE — a real domain,
#      or a tunnel someone is deliberately running. Never overridden.
#   2. The compose cloudflared service's current quick-tunnel hostname.
#   3. Whatever PUBLIC_BASE_URL said, even if unreachable — so the existing
#      startup checks and the readiness panel still report the real problem
#      rather than silently pretending the variable was never set.
#
# Set TUNNEL_DISCOVERY_URL="" to switch discovery off entirely (production,
# where PUBLIC_BASE_URL is a real domain and there is no cloudflared service).

set -e

DISCOVERY="${TUNNEL_DISCOVERY_URL-http://cloudflared:20241}"
WAIT_SECONDS="${TUNNEL_DISCOVERY_WAIT_S:-20}"

log() { echo "[entrypoint] $*" >&2; }

reachable() {
    # --max-time keeps a black-holed hostname from stalling startup.
    curl -fsS --max-time 5 "$1/health" >/dev/null 2>&1
}

discover() {
    # cloudflared needs a moment to register the tunnel with Cloudflare's edge,
    # and compose may start us first, so poll rather than check once.
    i=0
    while [ "$i" -lt "$WAIT_SECONDS" ]; do
        host=$(curl -fsS --max-time 2 "$DISCOVERY/quicktunnel" 2>/dev/null \
               | sed -n 's/.*"hostname" *: *"\([^"]*\)".*/\1/p')
        if [ -n "$host" ]; then
            echo "https://$host"
            return 0
        fi
        i=$((i + 1))
        sleep 1
    done
    return 1
}

if [ -n "$PUBLIC_BASE_URL" ] && reachable "$PUBLIC_BASE_URL"; then
    log "PUBLIC_BASE_URL is set and reachable: $PUBLIC_BASE_URL"
elif [ -n "$DISCOVERY" ]; then
    if [ -n "$PUBLIC_BASE_URL" ]; then
        log "PUBLIC_BASE_URL ($PUBLIC_BASE_URL) is unreachable — asking cloudflared for the current one"
    else
        log "PUBLIC_BASE_URL is unset — asking cloudflared for the current one"
    fi
    if discovered=$(discover); then
        log "discovered tunnel: $discovered"
        PUBLIC_BASE_URL="$discovered"
        export PUBLIC_BASE_URL
    else
        log "could not reach $DISCOVERY — is the cloudflared service running?"
        log "keeping PUBLIC_BASE_URL=${PUBLIC_BASE_URL:-<unset>}; /admin/readiness will report the failure"
    fi
fi

exec "$@"
