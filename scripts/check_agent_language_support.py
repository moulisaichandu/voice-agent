#!/usr/bin/env python
"""
check_agent_language_support.py — What can the live ElevenLabs agents actually
speak, and in which VOICE?

Two questions this answers, both of which live entirely on ElevenLabs' side
rather than in this repo:

  1. "Hindi doesn't work" — is `hi` under Additional Languages, and is the
     `language` field enabled under Security -> Overrides? preflight.py and
     admin/campaigns.py already refuse to dial when they aren't, but this shows
     the state directly instead of inferring it from a failed dial.

  2. "I changed the voice in the dashboard but calls still use the old one" —
     there are three usual causes, and this distinguishes them:
       * WRONG AGENT. A campaign freezes its agent_id at creation
         (admin/campaigns.py::_resolve_agent_id), so editing the two-way agent
         does nothing for a one-way campaign, and changing .env never
         retro-updates an existing campaign row. Both agents are printed below;
         compare them against the campaign's agent_id in the admin UI.
       * A PER-LANGUAGE PRESET. language_presets[iso] can PIN a voice for one
         language, which beats the agent's base voice for calls in it.
       * The voice override not being enabled, when ELEVENLABS_VOICE_ID is set.

Read-only: calls conversational_ai.agents.get(), never modifies the agent.

Usage:
    python scripts/check_agent_language_support.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (  # noqa: E402
    ELEVENLABS_ONEWAY_AGENT_ID,
    ELEVENLABS_TWOWAY_AGENT_ID,
    ELEVENLABS_VOICE_ID,
)
from app.telephony.elevenlabs_client import agent_language_support  # noqa: E402

AGENTS = [("one-way", ELEVENLABS_ONEWAY_AGENT_ID),
          ("two-way", ELEVENLABS_TWOWAY_AGENT_ID)]

if not any(agent_id for _label, agent_id in AGENTS):
    sys.exit("Neither ELEVENLABS_ONEWAY_AGENT_ID nor ELEVENLABS_TWOWAY_AGENT_ID "
             "is set in .env — nothing to check.")

print(f"Forcing voice (ELEVENLABS_VOICE_ID): {ELEVENLABS_VOICE_ID or '(not set)'}")
print("A campaign uses whichever agent id it was CREATED with — compare these "
      "against the campaign's agent_id in the admin UI.")
print()

problems: list[str] = []

for label, agent_id in AGENTS:
    print(f"=== {label} agent: {agent_id or '(not set — skipped)'} ===")
    if not agent_id:
        print()
        continue

    try:
        support = agent_language_support(agent_id)
    except Exception as exc:
        print(f"  could not read it ({type(exc).__name__}: {exc})")
        problems.append(f"The {label} agent {agent_id} could not be read. Check "
                        "ELEVENLABS_API_KEY, the agent id, and network access.")
        print()
        continue

    print(f"  languages              : {sorted(support['languages']) or '(none reported)'}")
    print(f"  language override      : {support['override_allowed']}")
    print(f"  tts_model              : {support['tts_model']}")
    print(f"  configured voice       : {support['voice_id'] or '(none reported)'}")
    print(f"  voice override allowed : {support['voice_override_allowed']}")
    print(f"  per-language presets   : {support['preset_voice_ids'] or '(none)'}")

    if ELEVENLABS_VOICE_ID:
        if support["voice_override_allowed"]:
            print(f"  -> calls WILL be forced to voice {ELEVENLABS_VOICE_ID}")
        else:
            problems.append(
                f"The {label} agent ({agent_id}) does NOT allow the voice "
                "override, but ELEVENLABS_VOICE_ID is set — preflight will "
                "refuse to dial it. Open that agent's Security tab -> "
                "Overrides -> TTS -> enable Voice ID."
            )
    elif support["voice_id"]:
        print(f"  -> calls use this agent's own voice ({support['voice_id']}); "
              "set ELEVENLABS_VOICE_ID in .env to force a different one")

    # A preset pins a voice for ONE language, beating the agent's base voice —
    # so changing the base voice in the dashboard cannot affect that language.
    for iso, pinned in (support["preset_voice_ids"] or {}).items():
        if pinned != (ELEVENLABS_VOICE_ID or support["voice_id"]):
            print(f"  !! '{iso}' is PINNED to voice {pinned} by a language preset. "
                  "Changing this agent's base voice in the dashboard will NOT "
                  f"affect {iso} calls.")

    print()

# Language checks stay scoped to the two-way agent, which is the one that holds
# conversations — keeping this script's exit semantics as they were.
if ELEVENLABS_TWOWAY_AGENT_ID:
    try:
        two_way = agent_language_support(ELEVENLABS_TWOWAY_AGENT_ID)
    except Exception:
        two_way = None
    if two_way is not None:
        if not two_way["override_allowed"]:
            problems.append(
                "The two-way agent's 'language' override is NOT enabled. Open "
                "its Security tab and enable the 'language' override."
            )
        if "hi" not in two_way["languages"]:
            problems.append(
                "'hi' (Hindi) is not in the two-way agent's configured "
                "languages. Add it under Additional Languages."
            )

if problems:
    print("PROBLEMS FOUND:")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)

print("No configuration problems found on the agents above.")
