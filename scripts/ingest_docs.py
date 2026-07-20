#!/usr/bin/env python
"""
ingest_docs.py — Chunk + embed every course document into doc_chunks.

Run this whenever the course docs change (new curriculum, updated pricing,
etc.) — the two-way agent's /rag/search tool only knows what's in doc_chunks,
not what's on disk in documents/. Re-running is safe: ingest_document()
clears each doc's existing chunks before re-inserting, so this never
accumulates stale duplicates from an earlier version of a file.

Usage:
    python scripts/ingest_docs.py                  # ingest ./documents
    python scripts/ingest_docs.py --dir some/path
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATABASE_URL, OPENAI_API_KEY  # noqa: E402
from app.db.pool import close_pool  # noqa: E402
from app.rag.ingest import SUPPORTED_EXTENSIONS, ingest_directory  # noqa: E402

_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "documents"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=str(_DEFAULT_DIR),
                        help="directory of course documents to ingest")
    args = parser.parse_args()

    if not DATABASE_URL:
        print("DATABASE_URL is not set.", file=sys.stderr)
        sys.exit(1)
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY is not set — embeddings cannot be generated.",
              file=sys.stderr)
        sys.exit(1)

    directory = Path(args.dir)
    if not directory.is_dir():
        print(f"{directory} is not a directory.", file=sys.stderr)
        sys.exit(1)

    try:
        results = await ingest_directory(directory)
        if not results:
            exts = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            print(f"No supported documents found in {directory} (supported: {exts})")
            return
        total = 0
        for name, count in results.items():
            print(f"  {name}: {count} chunk(s)")
            total += count
        print(f"\nIngested {len(results)} document(s), {total} chunk(s) total.")
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
