"""rag/ingest.py — Read course documents, chunk them, embed, store in doc_chunks.

Chunking sizes mirror ai-voice-agent/backend/rag.py's proven values (1000 chars
~= 200-250 words per chunk, 200-char overlap to keep context across borders).
"""

from __future__ import annotations

import re
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


_HEADING = re.compile(r"^#{1,6} \S")


def _blocks(text: str) -> list[str]:
    """Paragraph blocks, with a markdown heading glued to what it introduces.

    A heading alone embeds as almost nothing, and the content under it embeds
    without the word that says what it IS. Keeping them together is what makes
    a fee table retrievable by the query "course fee".
    """
    raw = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    out: list[str] = []
    heading: str | None = None
    for block in raw:
        if _HEADING.match(block) and "\n" not in block and len(block) < 120:
            # Two headings in a row: the first introduces a section with no
            # body of its own, so it still belongs with what follows.
            heading = f"{heading}\n{block}" if heading else block
            continue
        out.append(f"{heading}\n\n{block}" if heading else block)
        heading = None
    if heading:
        out.append(heading)
    return out


def _split_long(block: str, chunk_chars: int) -> list[str]:
    """Break a block bigger than one chunk, never mid-word.

    Sentence ends first, then whitespace. A chunk may run slightly over
    chunk_chars rather than cut a word in half: half a word embeds as noise,
    and the few characters saved are worth less than that.
    """
    pieces = re.split(r"(?<=[.!?\u0964])\s+", block)
    parts: list[str] = []
    current = ""
    for piece in pieces:
        while len(piece) > chunk_chars:
            # A single "sentence" longer than a chunk (a table row, a run-on):
            # cut at the last space before the limit.
            cut = piece.rfind(" ", 0, chunk_chars)
            if cut <= 0:
                cut = chunk_chars
            parts.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if current and len(current) + 1 + len(piece) > chunk_chars:
            parts.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        parts.append(current)
    return [p for p in parts if p]


def _tail(chunk: str, overlap: int) -> str:
    """The last whole sentences of *chunk*, up to *overlap* characters.

    Whole sentences, not a character slice: the point of overlap is that a fact
    split across a seam is still readable on one side of it, and half a
    sentence is not.
    """
    if overlap <= 0:
        return ""
    sentences = re.split(r"(?<=[.!?\u0964])\s+", chunk)
    out = ""
    for sentence in reversed(sentences):
        candidate = f"{sentence} {out}".strip()
        if len(candidate) > overlap:
            break
        out = candidate
    if out:
        return out
    # No sentence boundary fits — a table row, or a long unbroken run.
    # Fall back to a character slice cut at a space, because SOME overlap is
    # the point: a fact sitting on the seam has to be readable from one side.
    tail = chunk[-overlap:]
    space = tail.find(" ")
    return tail[space + 1:] if 0 <= space < len(tail) - 1 else tail


def chunk_text(
    text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split into chunks that respect the document's structure.

    This used to slice on raw character offsets, which cut straight through the
    facts a lead actually asks for. Measured on the live corpus: only 2 of 91
    chunks contained a rupee amount, and both began mid-sentence ("ce, build
    confidence..." and "0). Outcome:..."). Their embeddings were dominated by
    whatever else had landed in them, so a query for "course fee" ranked the
    CURRICULUM page above both, and the agent told a prospective student it did
    not have the fee information while the amounts sat in the corpus the whole
    time. That is a sales conversation lost to a chunk boundary.

    Blocks are packed whole, so a heading stays with its table and a price
    stays with the programme it names. Whitespace-only input still yields no
    chunks, and *overlap* still carries context across the seam, but as whole
    trailing sentences rather than an arbitrary number of characters.
    """
    text = text.strip()
    if not text:
        return []
    if chunk_chars <= overlap:
        raise ValueError("chunk_chars must be greater than overlap")

    chunks: list[str] = []
    current = ""
    for block in _blocks(text):
        parts = (_split_long(block, chunk_chars)
                 if len(block) > chunk_chars else [block])
        for part in parts:
            if current and len(current) + 2 + len(part) > chunk_chars:
                chunks.append(current)
                seam = _tail(current, overlap)
                current = f"{seam}\n\n{part}".strip() if seam else part
            else:
                current = f"{current}\n\n{part}".strip() if current else part
    if current:
        chunks.append(current)
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
