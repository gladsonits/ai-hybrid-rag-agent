"""Run the indexing pipeline: chunk -> embed -> store in PostgreSQL.

Usage (from the repo root):

    uv run python src/indexing/run_index.py
    uv run python src/indexing/run_index.py --replace
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
from indexing.indexer import Indexer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Index the source PDF into PostgreSQL.")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="delete the document's existing rows before indexing",
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

    indexer = Indexer(db, embedder)
    indexer.prepare()

    chunker = PDFChunker(args.pdf) if args.pdf else PDFChunker()
    print(f"chunking {chunker.pdf_path}")
    chunks = chunker.chunk()
    print(f"produced {len(chunks)} chunks")

    written = indexer.index_chunks(chunks) if not args.replace else indexer.index_source(
        chunker, replace=True
    )
    print(f"wrote {written} rows")

    print(f"total rows in chunks table: {db.count()}")
    print(f"done in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
