"""Run the indexing pipeline: chunk -> embed -> store in PostgreSQL.

Usage (from the repo root):

    uv run python src/indexing/run_index.py
    uv run python src/indexing/run_index.py --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indexing.chunker import PDFChunker  # noqa: E402
from retrieval.database import Database  # noqa: E402
from retrieval.embedder import EmbeddingModel  # noqa: E402
from retrieval.index_state import IndexState  # noqa: E402
from indexing.indexer import Indexer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Index the source PDF into PostgreSQL.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="write the document even when it has not changed",
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help="path to the PDF (defaults to $SOURCE_PDF)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="drop the chunks table first (needed when the embedding model changes)",
    )
    args = parser.parse_args()

    started = time.perf_counter()

    db = Database()
    print(f"connecting to {db.settings.redacted()}")

    embedder = EmbeddingModel()
    print(f"loading embedding model {embedder.settings.model}")
    embedder.load()
    print(f"embedding dimension: {embedder.dimension}")

    if args.reset:
        db.drop_schema()

    index_state = IndexState(db)
    indexer = Indexer(db, embedder, index_state=index_state)
    indexer.prepare()
    print(f"corpus version: {index_state.generation()}")

    chunker = PDFChunker(args.pdf) if args.pdf else PDFChunker()
    print(f"chunking {chunker.pdf_path}")
    chunks = chunker.chunk()
    print(f"produced {len(chunks)} chunks")

    if args.reset:
        source = chunks[0].metadata.get("source") if chunks else None
        if source:
            index_state.forget_document(source)
            print(f"forgot the fingerprint of {source}")

    written = indexer.index_chunks(chunks, force=args.force)
    print(f"wrote {written} rows")
    print(f"corpus version: {index_state.generation()}")

    print(f"total rows in chunks table: {db.count()}")
    print(f"done in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
