
import pytest

from app.rag import ingest


def test_chunk_text_empty_yields_nothing():
    assert ingest.chunk_text("") == []
    assert ingest.chunk_text("   \n\t  ") == []


def test_chunk_text_short_text_is_one_chunk():
    assert ingest.chunk_text("hello world", chunk_chars=1000, overlap=200) == ["hello world"]


def test_chunk_text_splits_and_overlaps():
    text = "x" * 2500
    chunks = ingest.chunk_text(text, chunk_chars=1000, overlap=200)
    # boundaries: [0:1000], [800:1800], [1600:2500] — the last chunk's end hits
    # n=2500 exactly, so the loop stops there rather than emitting a 4th chunk.
    assert len(chunks) == 3
    assert [len(c) for c in chunks] == [1000, 1000, 900]
    # every chunk after the first repeats the prior chunk's last `overlap` chars
    assert chunks[0][-200:] == chunks[1][:200]
    assert chunks[1][-200:] == chunks[2][:200]
    assert "".join(chunks)[:1000] == text[:1000]


def test_chunk_text_rejects_overlap_ge_chunk_chars():
    with pytest.raises(ValueError):
        ingest.chunk_text("hello", chunk_chars=100, overlap=100)


def test_read_document_txt(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("SEO fees: 5000 rupees.", encoding="utf-8")
    assert ingest.read_document(p) == "SEO fees: 5000 rupees."


def test_read_document_rejects_unsupported_extension(tmp_path):
    p = tmp_path / "notes.xyz"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        ingest.read_document(p)


async def test_ingest_document_embeds_and_stores_each_chunk(tmp_path, monkeypatch):
    p = tmp_path / "course.txt"
    p.write_text("x" * 2500, encoding="utf-8")

    replaced = []

    async def fake_replace(doc_name, rows):
        replaced.append((doc_name, rows))

    monkeypatch.setattr(ingest.rag_store, "replace_doc_chunks", fake_replace)
    monkeypatch.setattr(ingest, "embed_texts", lambda texts: [[0.1, 0.2]] * len(texts))

    n = await ingest.ingest_document(p)

    assert n == 3  # matches the chunking test above for the same 2500-char input
    assert len(replaced) == 1
    doc_name, rows = replaced[0]
    assert doc_name == "course.txt"
    assert len(rows) == 3
    assert all(len(embedding) == 2 for _content, embedding in rows)


async def test_ingest_document_clears_but_stores_nothing_for_blank_file(tmp_path, monkeypatch):
    p = tmp_path / "empty.txt"
    p.write_text("   ", encoding="utf-8")

    cleared = []

    async def fake_clear_doc(doc_name):
        cleared.append(doc_name)

    async def boom(*a, **kw):
        raise AssertionError("replace_doc_chunks must not be called for a blank document")

    monkeypatch.setattr(ingest.rag_store, "clear_doc", fake_clear_doc)
    monkeypatch.setattr(ingest.rag_store, "replace_doc_chunks", boom)

    n = await ingest.ingest_document(p)
    assert n == 0
    assert cleared == ["empty.txt"]


async def test_ingest_leaves_chunks_intact_if_embedding_fails(tmp_path, monkeypatch):
    """REGRESSION: re-ingest used to clear a doc's chunks BEFORE embedding, so an
    embed failure (rate limit / 500 / rotated key) permanently wiped the doc's
    corpus with no replacement — the live agent then answered 'no relevant
    material' for that doc until a successful re-ingest. Embedding now happens
    first, so a failure must not touch the store at all."""
    p = tmp_path / "course.txt"
    p.write_text("x" * 2500, encoding="utf-8")

    touched = []

    async def fake_clear(doc_name):
        touched.append(("clear", doc_name))

    async def fake_replace(doc_name, rows):
        touched.append(("replace", doc_name))

    def boom(texts):
        raise RuntimeError("OpenAI rate limit")

    monkeypatch.setattr(ingest.rag_store, "clear_doc", fake_clear)
    monkeypatch.setattr(ingest.rag_store, "replace_doc_chunks", fake_replace)
    monkeypatch.setattr(ingest, "embed_texts", boom)

    with pytest.raises(RuntimeError):
        await ingest.ingest_document(p)

    assert touched == [], "an embed failure must leave the existing chunks untouched"


# ── a failed extraction must not delete the document's corpus ──────────────
#
# `if not chunks: clear_doc(...)` treats "the file is genuinely empty" and "text
# extraction returned nothing" identically. Re-export a fee document as a
# scanned PDF, or hit a pypdf version that yields "" for it, and every existing
# chunk for that document is deleted while the console prints "0 chunk(s)" among
# other lines. The agent then denies knowledge of those courses on every call,
# and nothing anywhere reports it. The module already applies exactly this
# reasoning to embed failures ("cleared first and embedded second... permanently
# wiped the doc's corpus"); extraction deserves the same care.

async def test_a_file_with_bytes_but_no_extractable_text_is_not_wiped(
        tmp_path, monkeypatch):
    from app.rag import ingest

    doc = tmp_path / "fees.pdf"
    doc.write_bytes(b"%PDF-1.7 scanned images, no text layer" * 20)
    cleared = []

    async def fake_clear(name):
        cleared.append(name)

    monkeypatch.setattr(ingest, "read_document", lambda p: "")
    monkeypatch.setattr(ingest.rag_store, "clear_doc", fake_clear)

    with pytest.raises(ingest.ExtractionFailed):
        await ingest.ingest_document(doc)

    assert cleared == [], "the document's chunks were deleted on a failed extract"


async def test_a_genuinely_empty_file_still_clears_its_chunks(tmp_path, monkeypatch):
    """A real deletion must still work — this guard must not strand old chunks."""
    from app.rag import ingest

    doc = tmp_path / "retired.md"
    doc.write_text("", encoding="utf-8")
    cleared = []

    async def fake_clear(name):
        cleared.append(name)

    monkeypatch.setattr(ingest, "read_document", lambda p: "")
    monkeypatch.setattr(ingest.rag_store, "clear_doc", fake_clear)

    assert await ingest.ingest_document(doc) == 0
    assert cleared == ["retired.md"]
