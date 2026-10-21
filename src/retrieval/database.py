"""Schema management and connection handling for the retrieval indexes.

Owns the single ``chunks`` table that both the dense and the sparse index
read from.  The table carries the chunk text, its metadata as JSONB, a
``tsvector`` column for keyword search, and a ``vector`` column for
similarity search, plus the HNSW and GIN indexes that make each search fast.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

from retrieval.config import DatabaseSettings

#: Full-text search configuration used for both indexing and querying.
#: "english" stems and drops stop words, so "running" matches "run".
FTS_CONFIG = "english"

#: Schema for the chunks table.  ``embedding`` is left out here because the
#: column's type depends on the vector dimension, which is only known once a
#: model is chosen - see :meth:`Database.ensure_schema`.
BASE_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    chunk_text  TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    source      TEXT,
    section     TEXT,
    page        INTEGER,
    end_page    INTEGER,
    chunk_index INTEGER,
    search_vector TSVECTOR,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

#: Indexes that do not depend on the vector dimension.
BASE_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS chunks_search_vector_idx "
    "ON chunks USING GIN (search_vector);",
    "CREATE INDEX IF NOT EXISTS chunks_metadata_idx "
    "ON chunks USING GIN (metadata);",
    "CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks (source);",
    "CREATE INDEX IF NOT EXISTS chunks_page_idx ON chunks (page);",
)

#: The upsert target: one row per (document, chunk position).  Re-indexing a
#: document updates its rows in place instead of duplicating them.  A
#: partial unique index is used because ``source`` and ``chunk_index`` are
#: nullable, and PostgreSQL treats NULLs as distinct in a plain unique
#: constraint - which would let duplicate rows in.
CHUNK_IDENTITY_DDL = """
CREATE UNIQUE INDEX IF NOT EXISTS chunks_source_chunk_idx
    ON chunks (source, chunk_index)
    WHERE source IS NOT NULL AND chunk_index IS NOT NULL;
"""


class Database:
    """A thin, explicit wrapper around the PostgreSQL connection.

    Owns the schema lifecycle (create the table, the extension, the indexes)
    and hands out connections.  It deliberately does not know about
    embeddings or queries - those belong to the vector and text stores.
    """

    def __init__(self, settings: DatabaseSettings | None = None) -> None:
        self.settings = settings or DatabaseSettings.from_env()

    # -- connections -------------------------------------------------------

    def connect(self) -> psycopg.Connection:
        """Open a new connection to the configured database."""
        return psycopg.connect(self.settings.conninfo(), row_factory=dict_row)

    @contextmanager
    def connection(self) -> Iterator[psycopg.Connection]:
        """Open a connection that commits on success and rolls back on error."""
        with self.connect() as conn:
            yield conn

    # -- server-level setup ------------------------------------------------

    def ensure_database(self) -> None:
        """Create the target database if it does not exist yet.

        Connects to the maintenance database to do this, because you cannot
        create a database from inside a connection to that same database.
        """
        target = self.settings.dbname
        admin = DatabaseSettings(
            host=self.settings.host,
            port=self.settings.port,
            user=self.settings.user,
            password=self.settings.password,
            dbname="postgres",
        )
        with psycopg.connect(admin.conninfo(), autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target,))
                if cur.fetchone() is None:
                    cur.execute(f'CREATE DATABASE "{target}"')
                    print(f"created database {target!r}")
                else:
                    print(f"database {target!r} already exists")

    def create_extension(self) -> None:
        """Install the ``vector`` extension in the current database."""
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        print("pgvector extension ready")

    # -- schema ------------------------------------------------------------

    def ensure_schema(self, dimension: int) -> None:
        """Create the table, the vector column, and every index.

        The vector column and its HNSW index depend on the embedding
        dimension, so they are created here rather than in the static DDL.
        """
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(BASE_TABLE_DDL)
                for statement in BASE_INDEX_DDL:
                    cur.execute(statement)
                cur.execute(CHUNK_IDENTITY_DDL)
                dimval = int(dimension)
                cur.execute(
                    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding vector("
                    + str(dimval)
                    + ")"
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
                    ON chunks USING hnsw (embedding vector_cosine_ops)
                    """,
                )
        print(f"schema ready (vector({dimension}) + hnsw + gin)")

    def drop_schema(self) -> None:
        """Drop the chunks table. Used to re-index from scratch."""
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS chunks")
        print("dropped chunks table")

    # -- inspection --------------------------------------------------------

    def vector_dimension(self) -> int | None:
        """The dimension of the ``embedding`` column, or None if there is none."""
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT a.atttypmod AS dimension
                    FROM pg_attribute a
                    JOIN pg_class c ON c.oid = a.attrelid
                    WHERE c.relname = 'chunks'
                      AND a.attname = 'embedding'
                      AND a.attnum > 0
                      AND NOT a.attisdropped
                    """
                )
                row = cur.fetchone()
                if not row or row["dimension"] is None:
                    return None
                return int(row["dimension"])

    def count(self) -> int:
        """Number of rows in the chunks table."""
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) AS n FROM chunks")
                return int(cur.fetchone()["n"])
