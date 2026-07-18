#!/usr/bin/env python
"""
seed_dev_data.py — Create a demo campaign + a few leads in the local dev DB.

For local development only: gives campaign_tick/worker something to pick up
without needing a real Google Sheet wired in yet. Points at whatever
DATABASE_URL is set to, so DO NOT run it against production Supabase.

Usage:
    python scripts/seed_dev_data.py                    # two-way demo campaign
    python scripts/seed_dev_data.py --mode oneway      # one-way reminder campaign
    python scripts/seed_dev_data.py --phones +919876543210,+919812345678
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.compliance.dnd import normalize_phone_e164  # noqa: E402
from app.config import DATABASE_URL  # noqa: E402
from app.db import campaigns as campaigns_db  # noqa: E402
from app.db import leads as leads_db  # noqa: E402
from app.db.pool import close_pool  # noqa: E402

_DEFAULT_PHONES = ["+919876543210", "+919812345678", "+919700000001"]
_DEFAULT_NAMES = ["Ravi", "Sita", "Kiran"]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["oneway", "twoway"], default="twoway")
    parser.add_argument("--name", default="Dev Demo Campaign")
    parser.add_argument("--agent-id", default="agent_dev_placeholder",
                        help="ElevenLabs agent id; a placeholder is fine until one exists")
    parser.add_argument("--phones", default=",".join(_DEFAULT_PHONES),
                        help="comma-separated phone numbers")
    args = parser.parse_args()

    if not DATABASE_URL:
        print("DATABASE_URL is not set.", file=sys.stderr)
        sys.exit(1)
    if "supabase" in DATABASE_URL.lower():
        print("Refusing to seed: DATABASE_URL looks like real Supabase, not a local dev DB.",
              file=sys.stderr)
        sys.exit(1)

    try:
        campaign = await campaigns_db.create_campaign(
            name=args.name, mode=args.mode, agent_id=args.agent_id,
            script="ఈ డెమో క్లాస్ గురించి తెలియజేయడానికి కాల్ చేస్తున్నాము.",
        )
        print(f"campaign  {campaign.campaign_id}  {campaign.name}  ({campaign.mode})")

        raw_phones = [p.strip() for p in args.phones.split(",") if p.strip()]
        for i, raw in enumerate(raw_phones):
            phone = normalize_phone_e164(raw)
            if not phone:
                print(f"  skip  {raw!r} — not a valid Indian phone number")
                continue
            lead = await leads_db.upsert_lead(
                sheet_row=i + 2,
                name=_DEFAULT_NAMES[i] if i < len(_DEFAULT_NAMES) else f"Lead {i + 1}",
                phone_e164=phone,
                campaign_id=campaign.campaign_id,
            )
            print(f"  lead    {lead.lead_id}  {lead.name}  {lead.phone_e164}")

        print("\nSeeded. campaign_tick will pick these up inside calling hours "
              "(10:00-19:00 IST by default).")
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
