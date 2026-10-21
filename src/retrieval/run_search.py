"""Query the hybrid index: dense + keyword, fused with RRF.

Usage (from the repo root):

    uv run python src/retrieval/run_search.py "kubernetes docker"
    uv run python src/retrieval/run_search.py "genai chat" --mode dense
    uv run python src/retrieval/run_search.py "azure devops" --mode sparse
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.database import Database  # noqa: E402
from retrieval.embedder import EmbeddingModel  # noqa: E402
from retrieval.retriever import HybridRetriever  # noqa: E402
from retrieval.text_search import TextSearchStore  # noqa: E402
from retrieval.vector_store import VectorStore  # noqa: E402


def _print_hits(hits: list, mode: str) -> None:
    """Print a result list in a compact, readable form."""
    if not hits:
        print(f"  (no {mode} matches)")
        return
    for position, hit in enumerate(hits, start=1):
        text = " ".join(hit.chunk_text.split())
        score = getattr(hit, "score", None)
        rank = getattr(hit, "rank", None)
        shown = score if score is not None else rank
        print(f"  {position}. [p{hit.metadata.get('page')}] "
              f"{hit.metadata.get('section', '')} "
              f"score={shown:.5f}")
        print(f"     {text[:150]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Search the hybrid index.")
    parser.add_argument("query", help="the search query")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "dense", "sparse"],
        default="hybrid",
        help="which index to search (default: hybrid)",
    )
    parser.add_argument("--limit", type=int, default=5, help="results to show")
    parser.add_argument("--candidates", type=int, default=50, help="per-index budget")
    args = parser.parse_args()

    db = Database()
    embedder = EmbeddingModel()

    if args.mode == "hybrid":
        retriever = HybridRetriever(db, embedder)
        hits = retriever.search(args.query, limit=args.limit, candidates=args.candidates)
        print(f"hybrid search for {args.query!r} ({len(hits)} results)")
        for position, hit in enumerate(hits, start=1):
            text = " ".join(hit.chunk_text.split())
            print(f"  {position}. [p{hit.metadata.get('page')}] "
                  f"{hit.metadata.get('section', '')} rrf={hit.score:.5f} "
                  f"(dense #{hit.dense_rank}, sparse #{hit.sparse_rank})")
            print(f"     {text[:150]}")
        return 0

    if args.mode == "dense":
        vector = embedder.embed_one(args.query)
        hits = VectorStore(db).search(vector, limit=args.limit)
        print(f"dense search for {args.query!r} ({len(hits)} results)")
        _print_hits(hits, "dense")
        return 0

    hits = TextSearchStore(db).search(args.query, limit=args.limit)
    print(f"keyword search for {args.query!r} ({len(hits)} results)")
    _print_hits(hits, "keyword")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
