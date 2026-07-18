Review the current diff (`git diff`) against the hard compliance rules in
CLAUDE.md — specifically:

- Does any changed code path place or schedule a call outside the
  `CALLING_HOURS_START`–`CALLING_HOURS_END` window (`app/compliance/calling_hours.py`)?
- Does any changed dialling path skip the dial-time DND scrub
  (`app/compliance/dnd.py`)?
- Does every one-way and two-way script/prompt still open with the AI-disclosure
  line? Is there a test asserting this, not just a manual check?
- Does the RAG tool endpoint (`app/rag/endpoint.py`) call anything other than
  `search_relevant()` (the filtered function)? Flag immediately if so —
  `search_permissive()` must never be reachable from the live call path.
- Does any Sheets write-back trust `sheet_row` without verifying the phone
  number at that row first?
- Are secrets read only via `app/config.py`, never hardcoded or read from
  `os.environ` elsewhere?

Report each violation with the file:line, why it's a violation, and a concrete fix.
If nothing is found, say so plainly — do not invent findings.
