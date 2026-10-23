"""The version of the indexed corpus.

Every answer kept in the memory was built from a particular version of
the corpus.  When the corpus is rebuilt - the resume is edited and
indexed again - the answers built from the old version are wrong, and
the memory has to be told.

That is what the **generation** is for.  It is a counter that goes up
every time the corpus changes, and it is mixed into every cache key.
After a rebuild the counter is different, so every key the memory holds
is a key nobody asks for again: the memory misses, the model is asked,
and the answer is stored under the new generation.  The old answers are
not deleted - they expire on their own time to live.

The generation is one row in ``index_state``.  Beside it,
``indexed_documents`` holds the fingerprint of every document that has
been indexed, so a rebuild that changes nothing can be skipped: the
corpus is not written again, and the memory is not invalidated for
nothing.

Reading the state fails open: a corpus whose version cannot be read is
reported as version ``0`` - the memory still works, it just cannot tell
that the corpus changed.  Writing it does not: a version that could
not be moved on, or a fingerprint that could not be recorded, is
raised, so a corpus is never written without the version that
describes it.
"""

from __future__ import annotations

import logging

import psycopg

from retrieval.database import Database

logger = logging.getLogger(__name__)

#: The one row that holds the version of the corpus.
INDEX_STATE_DDL = """
CREATE TABLE IF NOT EXISTS index_state (
    id          INTEGER PRIMARY KEY,
    generation  BIGINT NOT NULL DEFAULT 1,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

#: One row per indexed document, holding what it looked like when it was
#: last indexed.  A document whose fingerprint is unchanged does not need
#: to be written again.
INDEXED_DOCUMENTS_DDL = """
CREATE TABLE IF NOT EXISTS indexed_documents (
    source      TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

#: The version reported when the state cannot be read.
UNKNOWN_GENERATION = 0


class IndexState:
    """The version of the corpus, and what each document looked like."""

    def __init__(self, database: Database | None = None) -> None:
        self.db = database or Database()

    # -- the version -------------------------------------------------------

    def generation(self) -> int:
        """The current version of the corpus.

        It goes up every time the corpus is rebuilt.  A state that cannot
        be read is reported as ``0``, so a database that is not there
        stops the memory from working, rather than the other way round.
        """
        try:
            with self.db.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT generation FROM index_state WHERE id = 1"
                    )
                    row = cur.fetchone()
        except Exception:
            logger.warning(
                "the version of the corpus could not be read", exc_info=True
            )
            return UNKNOWN_GENERATION

        if not row:
            return UNKNOWN_GENERATION
        return int(row["generation"])

    def bump(self, conn: psycopg.Connection | None = None) -> int:
        """The corpus has changed - move to the next version.

        This is called *after* the new rows are written, in the same
        transaction, so the version and the rows always describe the same
        corpus: there is no moment at which the version has moved on but
        the rows have not.

        A connection can be handed in, so the move joins the transaction
        of the caller.

        A failure is not swallowed.  A corpus that was written without the
        version moving on would go on serving the answers that were built
        from the corpus as it was, and nothing would say so.
        """
        statement = """
            INSERT INTO index_state (id, generation, updated_at)
            VALUES (1, 1, now())
            ON CONFLICT (id) DO UPDATE SET
                generation = index_state.generation + 1,
                updated_at = now()
            RETURNING generation
        """
        if conn is not None:
            with conn.cursor() as cur:
                cur.execute(statement)
                row = cur.fetchone()
        else:
            with self.db.connection() as own:
                with own.cursor() as cur:
                    cur.execute(statement)
                    row = cur.fetchone()

        return int(row["generation"])

    # -- the documents -----------------------------------------------------

    def document_fingerprint(self, source: str) -> str:
        """What the document looked like when it was last indexed.

        An empty string means it has never been indexed.
        """
        try:
            with self.db.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT fingerprint FROM indexed_documents "
                        "WHERE source = %s",
                        (source,),
                    )
                    row = cur.fetchone()
        except Exception:
            logger.warning(
                "the indexed documents could not be read", exc_info=True
            )
            return ""

        if not row:
            return ""
        return str(row["fingerprint"])

    def record_document(
        self,
        source: str,
        fingerprint: str,
        conn: psycopg.Connection | None = None,
    ) -> None:
        """Remember what the document looks like now that it is indexed.

        A connection can be handed in, so the record joins the transaction
        of the caller: the rows, the fingerprint and the version of the
        corpus are then written together, or not at all.

        A failure is not swallowed.  A document that was written without
        its fingerprint being recorded would be written again on the next
        rebuild, and the version of the corpus would move on for nothing.
        """
        statement = """
            INSERT INTO indexed_documents (source, fingerprint, updated_at)
            VALUES (%s, %s, now())
            ON CONFLICT (source) DO UPDATE SET
                fingerprint = EXCLUDED.fingerprint,
                updated_at = now()
        """
        if conn is not None:
            with conn.cursor() as cur:
                cur.execute(statement, (source, fingerprint))
        else:
            with self.db.connection() as own:
                with own.cursor() as cur:
                    cur.execute(statement, (source, fingerprint))

    def forget_document(
        self,
        source: str,
        conn: psycopg.Connection | None = None,
    ) -> None:
        """The document is no longer part of the corpus.

        A connection can be handed in, so the removal joins the transaction
        of the caller.

        A failure is not swallowed: a fingerprint that is not removed is a
        document that will not be written again when it comes back
        unchanged.
        """
        statement = "DELETE FROM indexed_documents WHERE source = %s"
        if conn is not None:
            with conn.cursor() as cur:
                cur.execute(statement, (source,))
        else:
            with self.db.connection() as own:
                with own.cursor() as cur:
                    cur.execute(statement, (source,))

    # -- the schema --------------------------------------------------------

    def ensure_schema(self) -> None:
        """Create the two tables, if they are not there already."""
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(INDEX_STATE_DDL)
                cur.execute(INDEXED_DOCUMENTS_DDL)
                cur.execute(
                    """
                    INSERT INTO index_state (id, generation, updated_at)
                    VALUES (1, 1, now())
                    ON CONFLICT (id) DO NOTHING
                    """
                )
            conn.commit()
