"""Turn chunks into rows in the database: embed, then store.

This is the stage that sits between the chunker and the two retrieval
stores.  It owns the pipeline - chunk -> embed -> write vector + keyword
index in one row - so that the dense and the sparse index always describe
exactly the same set of chunks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from indexing.chunker import Chunk
from retrieval.database import Database
from retrieval.embedder import EmbeddingModel
from retrieval.vector_store import VectorStore


class ChunkSource(Protocol):
    """Anything that can produce chunks - the chunker, a file, a list."""

    def chunk(self) -> list[Chunk]: ...


class Indexer:
    """Embeds chunks and writes them to PostgreSQL.

    Writes in batches so a large corpus does not need one round trip per
    chunk, and so the embedding model can work on a full batch at a time.
    """

    def __init__(
        self,
        database: Database,
        embedder: EmbeddingModel,
        batch_size: int = 32,
    ) -> None:
        self.db = database
        self.embedder = embedder
        self.batch_size = batch_size
        self.vector_store = VectorStore(database)

    def prepare(self) -> None:
        """Create the database, the extension, and the schema.

        Safe to call repeatedly: every step is guarded by ``IF NOT EXISTS``.
        """
        self.db.ensure_database()
        self.db.create_extension()
        self.db.ensure_schema(self.embedder.dimension)

    def index_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Embed and store a list of chunks, returning how many were written."""
        if not chunks:
            return 0
        rows: list[tuple[str, list[float], dict[str, Any]]] = []
        for chunk in chunks:
            vector = self.embedder.embed_one(chunk.chunk_text)
            rows.append((chunk.chunk_text, vector, chunk.metadata))
        return self.vector_store.upsert_many(rows)

    def index_source(self, source: ChunkSource, replace: bool = False) -> int:
        """Chunk a source and index every chunk it produces.

        With ``replace=True`` the source's existing rows are deleted first,
        so a re-index cannot leave stale chunks behind.
        """
        chunks = source.chunk()
        if replace and chunks:
            existing_sources = {c.metadata.get("source") for c in chunks}
            for existing in existing_sources:
                if existing:
                    self.vector_store.delete_by_source(existing)
        return self.index_chunks(chunks)
