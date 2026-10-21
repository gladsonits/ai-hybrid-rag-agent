"""Sparse retrieval: keyword search over the chunks table using PostgreSQL FTS.

This is the standard Postgres approach - a ``tsvector`` column generated from
the chunk text, a GIN index over it, and a ``tsquery`` built from the user's
query.  The GIN index makes the scan index-only, which is why the keyword
side of a hybrid search is fast even over a large corpus.

The ranking is ``ts_rank_cd`` (cover density), which rewards a term that
appears in a tightly packed region of the document - a better BM25-like
signal than plain term frequency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from retrieval.database import Database, FTS_CONFIG


@dataclass(frozen=True, slots=True)
class TextSearchHit:
    """One chunk matched by keyword search, with its rank and metadata."""

    chunk_id: int
    chunk_text: str
    metadata: dict[str, Any]
    rank: float
    #: The matching terms, so a caller can show why a chunk was retrieved.
    matched_terms: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_text": self.chunk_text,
            "metadata": self.metadata,
            "rank": self.rank,
            "matched_terms": self.matched_terms,
        }


class TextSearchStore:
    """Stores and queries the keyword index in the chunks table."""

    def __init__(self, database: Database) -> None:
        self.db = database

    # -- writing -----------------------------------------------------------

    def upsert(
        self,
        chunk_text: str,
        metadata: dict[str, Any],
        chunk_id: int | None = None,
    ) -> int:
        """Store one chunk for keyword search and return its row id.

        The ``search_vector`` column is computed by PostgreSQL from the text,
        so the keyword index can never drift from the stored text.
        """
        from psycopg.types.json import Jsonb

        with self.db.connection() as conn:
            with conn.cursor() as cur:
                if chunk_id is None:
                    cur.execute(
                        """
                        INSERT INTO chunks (chunk_text, metadata, source, section,
                                            page, end_page, chunk_index, search_vector)
                        VALUES (%(text)s, %(metadata)s, %(source)s, %(section)s,
                                %(page)s, %(end_page)s, %(chunk_index)s,
                                to_tsvector(%(config)s, %(text)s))
                        ON CONFLICT (source, chunk_index)
                        WHERE source IS NOT NULL AND chunk_index IS NOT NULL
                        DO UPDATE SET
                            chunk_text = EXCLUDED.chunk_text,
                            metadata = EXCLUDED.metadata,
                            search_vector = EXCLUDED.search_vector,
                            updated_at = now()
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
                            "config": FTS_CONFIG,
                        },
                    )
                else:
                    cur.execute(
                        """
                        UPDATE chunks
                        SET chunk_text = %s,
                            metadata = %s,
                            search_vector = to_tsvector(%s, %s),
                            updated_at = now()
                        WHERE id = %s
                        RETURNING id
                        """,
                        (chunk_text, Jsonb(metadata), FTS_CONFIG, chunk_text, chunk_id),
                    )
                row = cur.fetchone()
                return int(row["id"])

    # -- reading -----------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[TextSearchHit]:
        """Return the chunks that best match ``query`` by keyword.

        The query is parsed with ``websearch_to_tsquery``, which accepts the
        quoted phrases, ``OR``, ``-`` negation and ``*`` prefix syntax users
        expect from a search box.
        """
        if not query.strip():
            return []
        where, params = self._build_filters(filters)
        params["query"] = query
        params["config"] = FTS_CONFIG
        params["limit"] = limit

        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, chunk_text, metadata,
                           ts_rank_cd(search_vector,
                                      websearch_to_tsquery(%(config)s, %(query)s)) AS rank,
                           ARRAY(
                               SELECT lexeme
                               FROM unnest(tsvector_to_array(search_vector)) AS lexeme
                               WHERE lexeme = ANY (
                                   ARRAY(SELECT unnest(tsvector_to_array(
                                       to_tsvector(%(config)s, %(query)s)))))
                           ) AS matched_terms
                    FROM chunks
                    WHERE search_vector @@ websearch_to_tsquery(%(config)s, %(query)s)
                      {where}
                    ORDER BY rank DESC
                    LIMIT %(limit)s
                    """,
                    params,
                )
                return [self._to_hit(row) for row in cur.fetchall()]

    def count(self) -> int:
        """Number of chunks with a keyword-search vector."""
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) AS n FROM chunks WHERE search_vector IS NOT NULL"
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
            clauses.append("%(page)s BETWEEN page AND end_page")
            params["page"] = filters["page"]
        if not clauses:
            return "", {}
        return "AND " + " AND ".join(clauses), params

    @staticmethod
    def _to_hit(row: dict[str, Any]) -> TextSearchHit:
        """Convert a database row into a :class:`TextSearchHit`."""
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return TextSearchHit(
            chunk_id=int(row["id"]),
            chunk_text=row["chunk_text"],
            metadata=metadata or {},
            rank=float(row.get("rank") or 0.0),
            matched_terms=list(row.get("matched_terms") or []),
        )
