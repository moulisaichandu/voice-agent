
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

    inserted = []
    cleared = []

    async def fake_clear_doc(doc_name):
        cleared.append(doc_name)

    async def fake_insert_chunk(*, doc_name, section, content, embedding):
        inserted.append((doc_name, content, embedding))

    monkeypatch.setattr(ingest.rag_store, "clear_doc", fake_clear_doc)
    monkeypatch.setattr(ingest.rag_store, "insert_chunk", fake_insert_chunk)
    monkeypatch.setattr(ingest, "embed_texts", lambda texts: [[0.1, 0.2]] * len(texts))

    n = await ingest.ingest_document(p)

    assert n == 3  # matches the chunking test above for the same 2500-char input
    assert cleared == ["course.txt"]
    assert len(inserted) == 3
    assert all(doc_name == "course.txt" for doc_name, _, _ in inserted)


async def test_ingest_document_clears_but_stores_nothing_for_blank_file(tmp_path, monkeypatch):
    p = tmp_path / "empty.txt"
    p.write_text("   ", encoding="utf-8")

    cleared = []

    async def fake_clear_doc(doc_name):
        cleared.append(doc_name)

    monkeypatch.setattr(ingest.rag_store, "clear_doc", fake_clear_doc)

    async def boom(*a, **kw):
        raise AssertionError("insert_chunk must not be called for a blank document")

    monkeypatch.setattr(ingest.rag_store, "insert_chunk", boom)

    n = await ingest.ingest_document(p)
    assert n == 0
    assert cleared == ["empty.txt"]
