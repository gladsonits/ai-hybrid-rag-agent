"""Hybrid retrieval: fuse the dense and the sparse result sets.

The two indexes score on scales that cannot be compared directly - cosine
similarity runs from -1 to 1, ``ts_rank_cd`` from 0 to about 1 but with a
completely different shape.  Mixing them with a weighted sum needs those
scores normalised first, which is fiddly and fragile.

Reciprocal Rank Fusion avoids the problem: it only looks at *where* a chunk
ranked in each list, not at the score itself.  A chunk that is 1st in either
list gets the same boost, and a chunk that is high in both gets more than
either list alone would give it.

    RRF(d) = sum over lists of  1 / (k + rank(d))

with ``k = 60`` the usual default.  The constant damps the influence of the
very top ranks, so a single list cannot dominate the fused result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from retrieval.database import Database
from retrieval.embedder import EmbeddingModel
from retrieval.text_search import TextSearchStore
from retrieval.vector_store import SearchHit, VectorStore

#: The RRF damping constant.  Larger values flatten the contribution of the
#: top ranks; 60 is the value from the original RRF paper.
RRF_K = 60


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk retrieved by the hybrid search, with its fused score."""

    chunk_id: int
    chunk_text: str
    metadata: dict[str, Any]
    score: float
    #: Where the chunk ranked in each index it appeared in.
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    matched_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_text": self.chunk_text,
            "metadata": self.metadata,
            "score": self.score,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "matched_terms": self.matched_terms,
        }


class HybridRetriever:
    """Searches both indexes and fuses the results with RRF."""

    def __init__(
        self,
        database: Database,
        embedder: EmbeddingModel,
        rrf_k: int = RRF_K,
    ) -> None:
        self.db = database
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.vector_store = VectorStore(database)
        self.text_store = TextSearchStore(database)

    def search(
        self,
        query: str,
        limit: int = 10,
        candidates: int = 50,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Run the hybrid search.

        Each index is asked for ``candidates`` rows (more than the final
        ``limit``), because a chunk that ranks poorly in one index can still
        rank well in the fused list.  ``candidates`` is also the budget each
        index needs to compute its own ranking.
        """
        query_vector = self.embedder.embed_one(query)

        dense_hits = self.vector_store.search(
            query_vector, limit=candidates, filters=filters
        )
        sparse_hits = self.text_store.search(
            query, limit=candidates, filters=filters
        )

        fused = self.fuse(dense_hits, sparse_hits)
        return fused[:limit]

    def fuse(
        self,
        dense_hits: Sequence[SearchHit],
        sparse_hits: Sequence[Any],
    ) -> list[RetrievedChunk]:
        """Fuse two ranked result lists with reciprocal rank fusion."""
        merged: dict[int, RetrievedChunk] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            merged[hit.chunk_id] = RetrievedChunk(
                chunk_id=hit.chunk_id,
                chunk_text=hit.chunk_text,
                metadata=hit.metadata,
                score=0.0,
                dense_rank=rank,
                dense_score=hit.score,
            )

        for rank, hit in enumerate(sparse_hits, start=1):
            existing = merged.get(hit.chunk_id)
            if existing is None:
                merged[hit.chunk_id] = RetrievedChunk(
                    chunk_id=hit.chunk_id,
                    chunk_text=hit.chunk_text,
                    metadata=hit.metadata,
                    score=0.0,
                    sparse_rank=rank,
                    sparse_score=hit.rank,
                    matched_terms=list(hit.matched_terms),
                )
            else:
                merged[hit.chunk_id] = replace(
                    existing,
                    sparse_rank=rank,
                    sparse_score=hit.rank,
                    matched_terms=list(hit.matched_terms),
                )

        scored = [
            replace(chunk, score=self._rrf_score(chunk.dense_rank, chunk.sparse_rank))
            for chunk in merged.values()
        ]

        return sorted(scored, key=lambda c: c.score, reverse=True)

    def _rrf_score(self, dense_rank: int | None, sparse_rank: int | None) -> float:
        """The reciprocal rank fusion score for one chunk."""
        score = 0.0
        if dense_rank is not None:
            score += 1.0 / (self.rrf_k + dense_rank)
        if sparse_rank is not None:
            score += 1.0 / (self.rrf_k + sparse_rank)
        return score
