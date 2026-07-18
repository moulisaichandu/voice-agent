"""rag/endpoint.py — POST /rag/search, the ElevenLabs two-way agent's server tool.

Calls ONLY search_relevant() (see search.py's module docstring for why this
matters) — never search_permissive().
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.rag.search import search_relevant

router = APIRouter(tags=["RAG"])


class RagQuery(BaseModel):
    # Bounded so a malformed/abusive tool call can't send a huge or empty
    # query (each triggers a paid embedding call).
    query: str = Field(min_length=1, max_length=512)


@router.post("/rag/search")
async def rag_search(body: RagQuery) -> dict:
    """Look up course material for a query — called by the ElevenLabs agent
    when it invokes the search_course_material tool. Returns text the agent
    voices back to the lead."""
    result = await search_relevant(body.query)
    return {"result": result}
