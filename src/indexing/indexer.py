"""Turn chunks into rows in the database: embed, then store.

This is the stage that sits between the chunker and the two retrieval
stores.  It owns the pipeline - chunk -> embed -> write vector + keyword
index in one row - so that the dense and the sparse index always describe
exactly the same set of chunks.

It also owns the **version of the corpus**.  When the corpus is written,
the version moves on, and every answer kept in the memory from the
corpus as it was becomes unreachable.  A rebuild that changes nothing is
skipped, so the memory is not emptied for no reason.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from typing import Any

from indexing.chunker import Chunk
from retrieval.database import Database
from retrieval.embedder import EmbeddingModel
from retrieval.index_state import IndexState
from retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


def document_fingerprint(chunks: Sequence[Chunk], model: str = "") -> str:
    """What a document looks like, as a fingerprint.

    Two runs over the same document produce the same fingerprint, and any
    change to the text or the metadata of any chunk produces a different
    one.  It is what tells the indexer that a document does not need to
    be written again.

    The embedding model is part of the fingerprint: the same text embedded
    by a different model is a different document as far as the index is
    concerned, so a change of model has to write it again rather than
    skip it.
    """
    payload = json.dumps(
        [[chunk.chunk_text, chunk.metadata] for chunk in chunks],
        sort_keys=True,
        default=str,
    )
    digest = f'{model}\n{payload}'
    return hashlib.sha256(digest.encode('utf-8')).hexdigest()


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
        index_state: IndexState | None = None,
    ) -> None:
        self.db = database
        self.embedder = embedder
        self.batch_size = batch_size
        self.vector_store = VectorStore(database)
        self.index_state = index_state

    def prepare(self) -> None:
        """Create the database, the extension, and the schema.

        Safe to call repeatedly: every step is guarded by ``IF NOT EXISTS``.
        """
        self.db.ensure_database()
        self.db.create_extension()
        self.db.ensure_schema(self.embedder.dimension)
        if self.index_state is not None:
            self.index_state.ensure_schema()

    def index_chunks(self, chunks: Sequence[Chunk], force: bool = False) -> int:
        """Embed and store a list of chunks, returning how many were written.

        The write is **one transaction**: the old rows of the document are
        removed, the new rows are written, the fingerprint of the document
        is recorded and the version of the corpus moves on - or none of it
        happened.  There is never a moment at which the rows of the corpus
        and the version that describes them disagree, so an answer can
        never be stored against a corpus that was never there.

        The version moves on **last**, so an answer given while the write
        is in flight belongs to the corpus as it was, which is still the
        one the rows of the moment describe.

        A document that is already in the index exactly as it is now is
        left alone: no rows are written and the version stays where it
        is, so the memory keeps the answers it has.  ``force`` writes it
        anyway, which is what a rebuild after a change of the embedding
        model needs.
        """
        if not chunks:
            return 0

        source = chunks[0].metadata.get("source")
        fingerprint = document_fingerprint(chunks, self._model_name())

        if not force and self.index_state is not None and source:
            if self.index_state.document_fingerprint(source) == fingerprint:
                logger.info(
                    "%s is already indexed exactly as it is - nothing to do",
                    source,
                )
                return 0

        rows: list[tuple[str, list[float], dict[str, Any]]] = []
        for chunk in chunks:
            vector = self.embedder.embed_one(chunk.chunk_text)
            rows.append((chunk.chunk_text, vector, chunk.metadata))

        with self.db.connection() as conn:
            if source:
                self.vector_store.delete_by_source(source, conn=conn)
            written = self.vector_store.upsert_many(rows, conn=conn)
            if self.index_state is not None:
                if source:
                    self.index_state.record_document(
                        source, fingerprint, conn=conn
                    )
                self.index_state.bump(conn=conn)

        return written

    # -- helpers -----------------------------------------------------------

    def _model_name(self) -> str:
        """The name of the embedding model, for the fingerprint."""
        settings = getattr(self.embedder, "settings", None)
        return str(getattr(settings, "model", "") or "")
