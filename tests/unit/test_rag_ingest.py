
import pytest

from app.rag import ingest


def test_chunk_text_empty_yields_nothing():
    assert ingest.chunk_text("") == []
    assert ingest.chunk_text("   \n\t  ") == []


def test_chunk_text_short_text_is_one_chunk():
    assert ingest.chunk_text("hello world", chunk_chars=1000, overlap=200) == ["hello world"]


def test_chunk_text_splits_and_overlaps():
    """The contract: bounded chunks, real overlap, nothing lost.

    This used to assert exact character offsets — [1000, 1000, 900] with a
    200-char seam — which pinned the SLICING MECHANISM rather than anything a
    caller depends on. That mechanism was the bug: it cut through fee tables so
    that no chunk embedded as being about fees. The input here ("x" * 2500) has
    no word, sentence or paragraph boundary anywhere, so it exercises the
    fallback path specifically.

    Chunks may now run up to overlap-worth over the target, because a seam is
    prepended to a full-sized block. Carrying the seam is worth more than the
    exactness.
    """
    text = "x" * 2500
    chunks = ingest.chunk_text(text, chunk_chars=1000, overlap=200)

    assert len(chunks) >= 3
    # seam + separator + a full-size block is the true worst case
    assert max(len(c) for c in chunks) <= 1000 + 200 + 2
    assert sum(c.count("x") for c in chunks) >= 2500
    # every chunk after the first carries a seam from the one before it
    for earlier, later in zip(chunks, chunks[1:]):
        assert later[:50] in earlier or earlier[-50:] in later


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


# ── chunks must not cut through the facts a lead asks for ──────────────────
#
# chunk_text sliced on raw character offsets, so a fee table was split across
# a boundary and neither half began with anything about fees. Measured on the
# live corpus: only 2 of 91 chunks contained a rupee amount, and both started
# mid-sentence ("ce, build confidence..." and "0). Outcome:..."). Their
# embeddings were therefore dominated by whatever else landed in them, so a
# query for "course fee" ranked the CURRICULUM page above them and the agent
# told a prospective student it did not have the fee information — while the
# amounts sat in the corpus the whole time.

# Padded so the chunk boundary falls INSIDE the fee list, which is exactly
# what happens in the real corpus at CHUNK_CHARS=1000.
_INTRO = ("We train students in digital marketing with live projects and "
          "placement support. ") * 12

FEE_DOC = f"""## About Digital Brolly

{_INTRO}

## Course Fees

- BDLP - Brolly Digital Learning Program: Rs 25,000 total.
- BDCP - Brolly Digital Career Program: Rs 50,000 total.
- BDDP - Brolly Digital Diploma Program: Rs 1,50,000 total.

## Placements

We support every learner with interview preparation and introductions to hiring
partners across Hyderabad and Bangalore, including mock interviews and reviews.
"""


def test_a_fee_section_stays_with_its_heading():
    chunks = ingest.chunk_text(FEE_DOC, chunk_chars=400, overlap=80)

    fee_chunks = [c for c in chunks if "25,000" in c]
    assert fee_chunks, "the fee amounts vanished entirely"
    assert any("Course Fees" in c for c in fee_chunks), (
        "no chunk contains BOTH the 'Course Fees' heading and its amounts, so "
        "nothing in the corpus embeds as being ABOUT fees"
    )


def test_all_the_fee_amounts_land_in_one_chunk():
    chunks = ingest.chunk_text(FEE_DOC, chunk_chars=400, overlap=80)

    complete = [c for c in chunks
                if all(a in c for a in ("25,000", "50,000", "1,50,000"))]
    assert complete, (
        "the fee list was split across chunks, so a lead asking for fees gets "
        "a partial answer depending on which half ranked higher"
    )


def test_no_chunk_starts_mid_word():
    chunks = ingest.chunk_text(FEE_DOC, chunk_chars=300, overlap=60)

    # A chunk may legitimately begin mid-SENTENCE — that is what an overlap
    # seam is. What it must never do is begin mid-WORD, which is what raw
    # character slicing did and what makes an embedding meaningless. Checked
    # against the source: the opening run must sit on a word boundary there.
    bad = []
    for c in chunks:
        opening = c.lstrip()[:30]
        if not opening:
            continue
        at = FEE_DOC.find(opening)
        if at > 0 and not FEE_DOC[at - 1].isspace():
            bad.append(opening)
    assert not bad, f"chunks beginning mid-word: {bad}"


def test_a_block_larger_than_the_chunk_size_is_still_split():
    """One enormous paragraph must not become one enormous chunk — the
    embedding model has a limit and a huge chunk retrieves poorly."""
    big = "word " * 2000
    chunks = ingest.chunk_text(big, chunk_chars=500, overlap=100)

    assert len(chunks) > 1
    assert max(len(c) for c in chunks) <= 700, "a chunk blew well past the size"
