"""rag/ingest.py — Read course documents, chunk them, embed, store in doc_chunks.

Chunking sizes mirror ai-voice-agent/backend/rag.py's proven values (1000 chars
~= 200-250 words per chunk, 200-char overlap to keep context across borders).
"""

from __future__ import annotations

from pathlib import Path

from app.db import rag_store
from app.rag.embeddings import embed_texts

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
# Formats whose text is EXTRACTED rather than read. For these, "the file has
# bytes but produced no text" is a failure (a scanned PDF, a pypdf version
# that yields nothing), not an empty document — see ingest_document. A .txt
# or .md containing only whitespace really is empty, and must still clear.
_EXTRACTED_FORMATS = {".pdf", ".docx"}
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


class ExtractionFailed(RuntimeError):
    """A file that has content but yielded no text. Raised rather than treated
    as an empty document, because the two need opposite handling — see
    ingest_document."""


async def ingest_document(path: Path) -> int:
    """Chunk + embed + store one document, replacing any existing chunks for this
    doc_name so re-running ingestion doesn't accumulate stale duplicates.
    Returns the chunk count.

    Order matters: the document is EMBEDDED before the store is touched, and the
    old chunks are then replaced ATOMICALLY (see rag_store.replace_doc_chunks).
    An earlier version cleared first and embedded second, so an embed failure
    (rate limit / 500 / rotated key) permanently wiped the doc's corpus with no
    replacement, and the live agent answered 'no relevant material' for it until
    someone noticed."""
    text = read_document(path)
    chunks = chunk_text(text)
    if not chunks:
        # "No chunks" has two very different causes and they must not share a
        # branch. A genuinely empty FILE is a deletion: drop the old chunks.
        # A file with bytes in it that yielded no text is an EXTRACTION
        # FAILURE — a fee sheet re-exported as a scanned PDF, or a pypdf
        # version that returns "" for it — and clearing on that silently
        # deletes the document's whole corpus while reporting success. The
        # agent then denies knowledge of those courses on every call, and
        # nothing anywhere says why. Same reasoning as the embed ordering
        # above, which exists because that exact wipe already happened once.
        if path.suffix.lower() in _EXTRACTED_FORMATS and path.stat().st_size > 0:
            raise ExtractionFailed(
                f"{path.name} is {path.stat().st_size} bytes but no text could "
                "be extracted from it — refusing to clear its existing chunks. "
                "If the document really is retired, delete it; if it is a "
                "scanned PDF, it needs a text layer (OCR) before it can be "
                "ingested."
            )
        # An emptied document: drop its old chunks, nothing to insert.
        await rag_store.clear_doc(path.name)
        return 0

    # Embed BEFORE touching the store: a failure here leaves the prior chunks
    # intact rather than wiping the doc.
    embeddings = embed_texts(chunks)
    rows = [
        (chunk, embedding)
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]
    await rag_store.replace_doc_chunks(path.name, rows)
    return len(chunks)


async def ingest_directory(directory: Path) -> dict[str, int]:
    """Ingest every supported file in *directory*. Returns {filename: chunk_count}."""
    results: dict[str, int] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            results[path.name] = await ingest_document(path)
    return results
