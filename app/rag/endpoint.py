"""rag/endpoint.py — POST /rag/search, the ElevenLabs two-way agent's server tool.

Calls ONLY search_relevant() (see search.py's module docstring for why this
matters) — never search_permissive().

Never returns a 5xx: this endpoint is called by an agent that is MID-CALL with
a real person, and search_relevant()'s contract is that the tool always yields
something speakable. A missing/revoked OPENAI_API_KEY, an OpenAI outage, or a
Postgres blip would otherwise surface as a tool error the agent has no script
for, in the middle of a live conversation. Those failures degrade to the same
"no relevant material" note an off-topic question gets — logged loudly here so
the operator sees it, while the lead just hears the agent say it doesn't have
that information. That still satisfies CLAUDE.md's hard rule (never invent
course facts): the fallback asserts nothing.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.rag.search import NO_MATERIAL_NOTE, search_relevant

router = APIRouter(tags=["RAG"])
logger = logging.getLogger(__name__)


class RagQuery(BaseModel):
    # Bounded so a malformed/abusive tool call can't send a huge or empty
    # query (each triggers a paid embedding call).
    query: str = Field(min_length=1, max_length=512)


@router.post("/rag/search")
async def rag_search(body: RagQuery) -> dict:
    """Look up course material for a query — called by the ElevenLabs agent
    when it invokes the search_course_material tool. Returns text the agent
    voices back to the lead. See the module docstring: this deliberately
    never propagates an exception as a 5xx."""
    try:
        result = await search_relevant(body.query)
    except Exception as exc:
        logger.error(
            f"[rag] search failed, returning the no-material fallback so the "
            f"live agent still has something speakable: {type(exc).__name__}: {exc}"
        )
        return {"result": NO_MATERIAL_NOTE, "degraded": True}
    return {"result": result}
