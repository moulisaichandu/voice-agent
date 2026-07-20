#!/usr/bin/env python
"""
register_plivo_number.py — Register a Plivo number with ElevenLabs over SIP.

This project does NOT talk to Plivo directly (unlike the sibling
ai-voice-agent). ElevenLabs places the call, so the number must be
registered inside ElevenLabs as a SIP trunk endpoint; ElevenLabs then sends
the SIP INVITE to Plivo, and Plivo dials the lead.

Prompts for the SIP password rather than taking it as an argument, so it
never lands in shell history. Prints the resulting phone_number_id, which is
what goes in ELEVENLABS_AGENT_PHONE_NUMBER_ID.

Usage:
    python scripts/register_plivo_number.py \
        --number +918035383564 \
        --address yourtrunk.zt.plivo.com \
        --username your_sip_user
"""

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ELEVENLABS_API_KEY  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--number", required=True, help="E.164, e.g. +918035383564")
    p.add_argument("--address", required=True,
                   help="Plivo termination SIP URI, e.g. yourtrunk.zt.plivo.com")
    p.add_argument("--username", required=True, help="SIP username (Plivo credential list)")
    p.add_argument("--label", default="Plivo India (SIP trunk)")
    p.add_argument("--transport", default="auto", choices=["auto", "udp", "tcp", "tls"])
    p.add_argument("--media-encryption", default="allowed",
                   choices=["disabled", "allowed", "required"])
    args = p.parse_args()

    if not ELEVENLABS_API_KEY:
        sys.exit("ELEVENLABS_API_KEY is not set in .env")

    password = getpass.getpass("Plivo SIP password (not echoed): ")
    if not password:
        sys.exit("A SIP password is required.")

    from elevenlabs.client import ElevenLabs
    from elevenlabs.conversational_ai.phone_numbers.types.phone_numbers_create_request_body import (  # noqa: E501
        PhoneNumbersCreateRequestBody_SipTrunk,
    )

    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    body = PhoneNumbersCreateRequestBody_SipTrunk(
        phone_number=args.number,
        label=args.label,
        # Outbound only: this project dials out. Inbound would need its own
        # config and an agent assigned to answer, which isn't in scope.
        supports_outbound=True,
        supports_inbound=False,
        outbound_trunk_config={
            "address": args.address,
            "transport": args.transport,
            "media_encryption": args.media_encryption,
            "credentials": {"username": args.username, "password": password},
        },
    )

    try:
        resp = client.conversational_ai.phone_numbers.create(request=body)
    except Exception as exc:
        sys.exit(f"Registration failed: {type(exc).__name__}: {exc}")

    phone_number_id = getattr(resp, "phone_number_id", None) or getattr(resp, "id", None)
    print()
    print("Registered.")
    print()
    print("Put this in .env, then `docker compose up -d --force-recreate backend`:")
    print(f"ELEVENLABS_AGENT_PHONE_NUMBER_ID={phone_number_id}")


if __name__ == "__main__":
    main()
