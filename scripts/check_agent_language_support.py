#!/usr/bin/env python
"""
check_agent_language_support.py — What can the live two-way ElevenLabs agent
actually speak?

Investigating "Hindi two-way doesn't work": app/telephony/preflight.py and
app/admin/campaigns.py both already refuse to dial a language the agent
isn't configured for, but the underlying dashboard state (which languages
are under "Additional Languages", and whether the `language` field is
enabled under Security -> Overrides) lives entirely on ElevenLabs' side, not
in this repo. This prints exactly what the API reports for
ELEVENLABS_TWOWAY_AGENT_ID, so a missing language shows up directly instead
of being inferred from a failed dial.

Read-only: calls conversational_ai.agents.get(), never modifies the agent.

Usage:
    python scripts/check_agent_language_support.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ELEVENLABS_TWOWAY_AGENT_ID  # noqa: E402
from app.telephony.elevenlabs_client import agent_language_support  # noqa: E402

if not ELEVENLABS_TWOWAY_AGENT_ID:
    sys.exit("ELEVENLABS_TWOWAY_AGENT_ID is not set in .env — nothing to check.")

print(f"Checking agent {ELEVENLABS_TWOWAY_AGENT_ID} ...")
try:
    support = agent_language_support(ELEVENLABS_TWOWAY_AGENT_ID)
except Exception as exc:
    sys.exit(f"Could not reach ElevenLabs ({type(exc).__name__}: {exc}). "
              "Check ELEVENLABS_API_KEY and network access.")

print(f"  override_allowed : {support['override_allowed']}")
print(f"  languages         : {sorted(support['languages']) or '(none reported)'}")
print(f"  tts_model         : {support['tts_model']}")
print()

problems = []
if not support["override_allowed"]:
    problems.append(
        "The 'language' override is NOT enabled. Open this agent's Security "
        "tab in the ElevenLabs dashboard and enable the 'language' override."
    )
if "hi" not in support["languages"]:
    problems.append(
        "'hi' (Hindi) is not in this agent's configured languages. Open the "
        "agent's Additional Languages setting and add Hindi."
    )

if problems:
    print("PROBLEMS FOUND:")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)
else:
    print("Hindi is fully configured on this agent — override allowed, "
          "'hi' present. If Hindi calls are still failing, the cause is "
          "something other than this agent's language configuration.")
