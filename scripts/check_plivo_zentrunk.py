#!/usr/bin/env python
"""
check_plivo_zentrunk.py — Is Zentrunk (SIP trunking) enabled on the Plivo account yet?

ElevenLabs places the calls for this project, so it needs its own phone line.
Connecting a Plivo number to ElevenLabs requires Zentrunk, which Plivo has to
enable per-account — it is not a setting in the console. Run this after asking
Plivo support to enable it, to see whether it has actually happened.

Distinguishes "not enabled" (Zentrunk endpoints 404 while ordinary ones 200)
from "bad credentials" (everything 401), because those look identical if you
only test one endpoint.

Credentials come from the environment so they stay out of shell history:

    set PLIVO_AUTH_ID=...
    set PLIVO_AUTH_TOKEN=...
    python scripts/check_plivo_zentrunk.py
"""

import os
import sys

import httpx

AUTH_ID = os.getenv("PLIVO_AUTH_ID")
AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN")

if not AUTH_ID or not AUTH_TOKEN:
    sys.exit("Set PLIVO_AUTH_ID and PLIVO_AUTH_TOKEN in the environment first.")

BASE = f"https://api.plivo.com/v1/Account/{AUTH_ID}"
AUTH = (AUTH_ID, AUTH_TOKEN)

# Ordinary endpoints prove the credentials work; the Zentrunk ones prove
# whether the product is provisioned. Checking both is what separates
# "not enabled" from "wrong token".
CONTROL = [("/Number/", "phone numbers"), ("/Application/", "applications")]
ZENTRUNK = [("/OutboundTrunk/", "outbound trunks"), ("/CredentialList/", "credential lists")]


def probe(path: str) -> int:
    try:
        return httpx.get(BASE + path, auth=AUTH, timeout=20).status_code
    except Exception as exc:  # network, DNS, TLS
        print(f"  request failed: {type(exc).__name__}: {exc}")
        return -1


print("=== credentials / ordinary endpoints ===")
control_ok = True
for path, label in CONTROL:
    code = probe(path)
    print(f"  {label:20} HTTP {code}")
    control_ok &= code == 200

if not control_ok:
    print()
    sys.exit("Ordinary endpoints are failing too — check PLIVO_AUTH_ID / "
             "PLIVO_AUTH_TOKEN before drawing any conclusion about Zentrunk.")

print()
print("=== Zentrunk (SIP trunking) ===")
enabled = True
for path, label in ZENTRUNK:
    code = probe(path)
    state = "available" if code == 200 else "NOT enabled" if code == 404 else f"unexpected ({code})"
    print(f"  {label:20} HTTP {code}   {state}")
    enabled &= code == 200

print()
if enabled:
    print("Zentrunk IS enabled. Next:")
    print("  1. Plivo console -> Zentrunk -> Credential Lists -> create a SIP username/password")
    print("  2. Plivo console -> Zentrunk -> Outbound Trunks -> create a trunk using that list")
    print("  3. Copy the trunk's Termination SIP URI")
    print("  4. python scripts/register_plivo_number.py --number +918035383564 \\")
    print("         --address <trunk>.zt.plivo.com --username <sip-user>")
else:
    print("Zentrunk is NOT enabled yet — the credentials are fine, the product isn't")
    print("provisioned. Only Plivo support can turn it on; re-run this after they reply.")
