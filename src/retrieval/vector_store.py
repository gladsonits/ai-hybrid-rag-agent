"""Dense retrieval: store chunk vectors in pgvector and search them by similarity.

Uses the standard pgvector approach - a ``vector(dim)`` column with an HNSW
index, cosine distance for ranking, and the ``<=>`` operator.  Vectors are
L2-normalised before they are written, so cosine distance and inner product
agree.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from retrieval.database import Database


def as_vector_literal(embedding: Sequence[float]) -> str:
    """Render a vector as the ``[1,2,3]`` literal pgvector parses.

    psycopg has no built-in adapter for the ``vector`` type, so the value is
    written as a string and cast in SQL with ``::vector``.
    """
    return "[" + ",".join(f"{float(value):.8g}" for value in embedding) + "]"


#: The pgvector setting that makes a filtered search return the rows it was
#: asked for.
#:
#: pgvector applies the WHERE filter *after* the approximate scan: the scan
#: visits a bounded number of tuples and the filter throws away the ones
#: that do not match.  A search that filters on a column the HNSW index
#: does not cover can therefore come back with fewer rows than were asked
#: for - including none at all.  That is how a semantic match silently
#: stops matching, right after a rebuild has filled the neighbourhood with
#: the rows of the corpus it has just replaced.
#:
#: The iterative scan keeps the search going until there are enough rows
#: that match, and keeps them in the order the distance puts them in.
ITERATIVE_SCAN_SETTING = "SET hnsw.iterative_scan = strict_order"

#: The statement that writes a chunk - the text, the metadata, the keyword
#: search vector and the embedding - in a single row, so the dense and the
#: sparse index always describe exactly the same set of chunks.
UPSERT_SQL = """
INSERT INTO chunks (
    chunk_text, metadata, source, section, page, end_page,
    chunk_index, search_vector, embedding
)
VALUES (
    %(text)s, %(metadata)s, %(source)s, %(section)s, %(page)s,
    %(end_page)s, %(chunk_index)s,
    to_tsvector('english', %(text)s), %(embedding)s::vector
)
ON CONFLICT (source, chunk_index)
WHERE source IS NOT NULL AND chunk_index IS NOT NULL
DO UPDATE SET
    chunk_text  = EXCLUDED.chunk_text,
    metadata    = EXCLUDED.metadata,
    section     = EXCLUDED.section,
    page        = EXCLUDED.page,
    end_page    = EXCLUDED.end_page,
    search_vector = EXCLUDED.search_vector,
    embedding   = EXCLUDED.embedding,
    updated_at  = now()
"""


def prepare_ann_search(cur: psycopg.Cursor) -> None:
    """Put the session into the mode a filtered search needs.

    It has to be called in the session that runs the search, before the
    search query - the setting is per session, not per database.
    """
    cur.execute(ITERATIVE_SCAN_SETTING)


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One retrieved chunk, with its score and metadata."""

    chunk_id: int
    chunk_text: str
    metadata: dict[str, Any]
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_text": self.chunk_text,
            "metadata": self.metadata,
            "score": self.score,
        }


class VectorStore:
    """Stores chunk embeddings in PostgreSQL and retrieves them by similarity.

    Writing a chunk stores the text, the metadata, the keyword-search
    vector, and the embedding in a single row, so the dense and the sparse
    index are always describing the same set of chunks.
    """

    def __init__(self, database: Database) -> None:
        self.db = database

    # -- writing -----------------------------------------------------------

    def upsert(
        self,
        chunk_text: str,
        embedding: Sequence[float],
        metadata: dict[str, Any],
    ) -> int:
        """Insert or update one chunk, returning its row id.

        The chunk is keyed on ``(source, chunk_index)`` when the metadata
        carries them, so re-indexing a document updates its rows in place
        instead of duplicating them.
        """
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chunks (
                        chunk_text, metadata, source, section, page, end_page,
                        chunk_index, search_vector, embedding
                    )
                    VALUES (
                        %(text)s, %(metadata)s, %(source)s, %(section)s, %(page)s,
                        %(end_page)s, %(chunk_index)s,
                        to_tsvector('english', %(text)s), %(embedding)s::vector
                    )
                    ON CONFLICT (source, chunk_index)
                    WHERE source IS NOT NULL AND chunk_index IS NOT NULL
                    DO UPDATE SET
                        chunk_text  = EXCLUDED.chunk_text,
                        metadata    = EXCLUDED.metadata,
                        section     = EXCLUDED.section,
                        page        = EXCLUDED.page,
                        end_page    = EXCLUDED.end_page,
                        search_vector = EXCLUDED.search_vector,
                        embedding   = EXCLUDED.embedding,
                        updated_at  = now()
                    RETURNING id
                    """,
                    {
                        "text": chunk_text,
                        "metadata": Jsonb(metadata),
                        "source": metadata.get("source"),
                        "section": metadata.get("section"),
                        "page": metadata.get("page"),
                        "end_page": metadata.get("end_page"),
                        "chunk_index": metadata.get("chunk_index"),
                        "embedding": as_vector_literal(embedding),
                    },
                )
                row = cur.fetchone()
                return int(row["id"])

    def upsert_many(
        self,
        chunks: Sequence[tuple[str, Sequence[float], dict[str, Any]]],
        conn: psycopg.Connection | None = None,
    ) -> int:
        """Insert or update many chunks in a single transaction.

        The write joins the transaction of the caller, when one is given.

        Each element is a ``(chunk_text, embedding, metadata)`` triple.  The
        embedding is written as a string literal so PostgreSQL can cast it to
        ``vector`` without psycopg needing a registered vector adapter.
        """
        if not chunks:
            return 0
        rows = [
            {
                "text": text,
                "embedding": as_vector_literal(embedding),
                "metadata": Jsonb(metadata),
                "source": metadata.get("source"),
                "section": metadata.get("section"),
                "page": metadata.get("page"),
                "end_page": metadata.get("end_page"),
                "chunk_index": metadata.get("chunk_index"),
            }
            for text, embedding, metadata in chunks
        ]
        if conn is not None:
            with conn.cursor() as cur:
                cur.executemany(UPSERT_SQL, rows)
        else:
            with self.db.connection() as own:
                with own.cursor() as cur:
                    cur.executemany(UPSERT_SQL, rows)
        return len(rows)

    def delete_by_source(
        self,
        source: str,
        conn: psycopg.Connection | None = None,
    ) -> int:
        """Remove every chunk belonging to one document.

        A connection can be handed in, so the delete joins the transaction of
        the caller: the rows, the fingerprint and the version of the corpus
        are then written together, or not at all.
        """
        if conn is not None:
            with conn.cursor() as cur:
                return self._delete_source(cur, source)
        with self.db.connection() as own:
            with own.cursor() as cur:
                return self._delete_source(cur, source)

    @staticmethod
    def _delete_source(cur: psycopg.Cursor, source: str) -> int:
        """The delete itself, on a cursor the caller owns."""
        cur.execute("DELETE FROM chunks WHERE source = %s", (source,))
        return cur.rowcount

    # -- reading -----------------------------------------------------------

    def search(
        self,
        embedding: Sequence[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Return the chunks most similar to ``embedding``.

        The score is cosine *similarity* (1 minus the distance), so higher is
        better.  ``filters`` may contain ``section``, ``source``, ``project``
        and ``page`` to restrict the search to part of the corpus.
        """
        where, params = self._build_filters(filters)
        params["embedding"] = as_vector_literal(embedding)
        params["limit"] = limit

        with self.db.connection() as conn:
            with conn.cursor() as cur:
                prepare_ann_search(cur)
                cur.execute(
                    f"""
                    SELECT id, chunk_text, metadata,
                           1 - (embedding <=> %(embedding)s::vector) AS similarity
                    FROM chunks
                    WHERE embedding IS NOT NULL {where}
                    ORDER BY embedding <=> %(embedding)s::vector
                    LIMIT %(limit)s
                    """,
                    params,
                )
                return [self._to_hit(row) for row in cur.fetchall()]

    def get(self, chunk_id: int) -> SearchHit | None:
        """Fetch a single chunk by id."""
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, chunk_text, metadata, 0 AS similarity "
                    "FROM chunks WHERE id = %s",
                    (chunk_id,),
                )
                row = cur.fetchone()
                return self._to_hit(row) if row else None

    def count(self) -> int:
        """Number of chunks that carry an embedding."""
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) AS n FROM chunks WHERE embedding IS NOT NULL"
                )
                return int(cur.fetchone()["n"])

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _build_filters(filters: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
        """Turn a filter dict into a SQL ``WHERE`` fragment and its params."""
        if not filters:
            return "", {}
        clauses: list[str] = []
        params: dict[str, Any] = {}
        for key in ("source", "section"):
            if key in filters and filters[key] is not None:
                clauses.append(f"{key} = %({key})s")
                params[key] = filters[key]
        if filters.get("project"):
            clauses.append("metadata->>'project' = %(project)s")
            params["project"] = filters["project"]
        if filters.get("page"):
            clauses.append(
                "%(page)s BETWEEN page AND end_page"
            )
            params["page"] = filters["page"]
        if not clauses:
            return "", {}
        return "AND " + " AND ".join(clauses), params

    @staticmethod
    def _to_hit(row: dict[str, Any]) -> SearchHit:
        """Convert a database row into a :class:`SearchHit`."""
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return SearchHit(
            chunk_id=int(row["id"]),
            chunk_text=row["chunk_text"],
            metadata=metadata or {},
            score=float(row.get("similarity") or 0.0),
        )
