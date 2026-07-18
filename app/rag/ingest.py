"""rag/ingest.py — Read course documents, chunk them, embed, store in doc_chunks.

Chunking sizes mirror ai-voice-agent/backend/rag.py's proven values (1000 chars
~= 200-250 words per chunk, 200-char overlap to keep context across borders).
"""

from __future__ import annotations

from pathlib import Path

from app.db import rag_store
from app.rag.embeddings import embed_texts

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
CHUNK_CHARS = 1_000
CHUNK_OVERLAP = 200


def chunk_text(
    text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split into overlapping chunks. Whitespace-only input yields no chunks —
    an empty/blank document contributes nothing to search, not a blank chunk."""
    text = text.strip()
    if not text:
        return []
    if chunk_chars <= overlap:
        raise ValueError("chunk_chars must be greater than overlap")

    chunks: list[str] = []
    n = len(text)
    i = 0
    while i < n:
        end = min(i + chunk_chars, n)
        chunks.append(text[i:end])
        if end == n:
            break
        i = end - overlap
    return chunks


def read_document(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        import pypdf
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if ext == ".docx":
        import docx
        d = docx.Document(str(path))
        return "\n".join(p.text for p in d.paragraphs)
    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported document type: {ext}")


async def ingest_document(path: Path) -> int:
    """Chunk + embed + store one document. Clears any existing chunks for this
    doc_name first, so re-running ingestion doesn't accumulate stale duplicate
    chunks from a previous version of the file. Returns the chunk count."""
    text = read_document(path)
    chunks = chunk_text(text)
    await rag_store.clear_doc(path.name)
    if not chunks:
        return 0

    embeddings = embed_texts(chunks)
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        await rag_store.insert_chunk(
            doc_name=path.name, section=None, content=chunk, embedding=embedding,
        )
    return len(chunks)


async def ingest_directory(directory: Path) -> dict[str, int]:
    """Ingest every supported file in *directory*. Returns {filename: chunk_count}."""
    results: dict[str, int] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            results[path.name] = await ingest_document(path)
    return results
