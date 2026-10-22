"""The FAQ index: the questions that have been asked, in a form that can be
searched by meaning rather than by wording.

The answer cache in Redis is an exact match - the same words in, the same
key.  This is the other half: a table of the questions that have been
asked, each with its embedding, so a question asked in other words can be
recognised as one that has been answered before.

The index is a second, much smaller table beside the chunks, and it uses
the same machinery - a ``vector`` column, an HNSW index and the same
sentence-transformer - because it is the same job on a smaller corpus.  A
row holds the question, the key the question is tracked under, and the key
its answer is stored under in the memory.

Searching it is the same search the retriever does, with one difference
that matters: the score is compared against a threshold rather than
ranked.  A rank says only that one question is nearer than another; a
threshold says the two questions really are the same one, which is what
has to be true before an answer is served without asking the model.
"""

from __future__ import annotations

import logging
from typing import Any

from memory.config import FaqSettings
from memory.faq import FaqCandidate
from retrieval.database import Database
from retrieval.embedder import EmbeddingModel
from retrieval.vector_store import as_vector_literal

logger = logging.getLogger(__name__)

#: The name of the table the questions live in.
FAQ_TABLE = "faq_questions"

#: The DDL for the table.  The vector column is left out, because its
#: type depends on the embedding dimension, which is only known once a
#: model is chosen - see :meth:`PostgresFaqMatcher.ensure_schema`.
FAQ_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS faq_questions (
    id           BIGSERIAL PRIMARY KEY,
    question     TEXT NOT NULL,
    question_key TEXT NOT NULL,
    cache_key    TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

#: The indexes that do not depend on the vector dimension.  The question
#: key is unique, so remembering the same question twice updates the one
#: row rather than adding another.
FAQ_INDEX_DDL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS faq_questions_key_idx "
    "ON faq_questions (question_key);",
    "CREATE INDEX IF NOT EXISTS faq_questions_updated_idx "
    "ON faq_questions (updated_at DESC);",
)

#: How many candidates the search looks at.  Only the nearest one can
#: be the match, so a handful is enough.
CANDIDATE_LIMIT = 5

#: How many questions the index keeps.  The table is a pool of
#: candidates, not the record of what is asked - that is the registry in
#: Redis - so it can be much larger than the FAQ set.
DEFAULT_MAX_ENTRIES = 1000

#: How often the index is trimmed, in writes.  Trimming on every write
#: would spend a delete on every question asked.
PRUNE_EVERY = 50


class PostgresFaqMatcher:
    """Finds the stored question that a new question means the same as.

    The search is the same one the retriever runs - cosine distance over
    a pgvector column - and the embedder is the same one, so the two
    indexes describe the same questions in the same space.  The
    difference is the threshold: a chunk that is merely the nearest
    neighbour is good enough to read, but a question that is only the
    nearest neighbour is not good enough to answer.
    """

    def __init__(
        self,
        database: Database,
        embedder: EmbeddingModel,
        settings: FaqSettings | None = None,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        self.db = database
        self.embedder = embedder
        self.settings = settings or FaqSettings.from_env()
        self.max_entries = max_entries
        self._writes = 0

    # -- schema ------------------------------------------------------------

    def ensure_schema(self) -> None:
        """Create the table, the vector column and the indexes."""
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(FAQ_TABLE_DDL)
                for statement in FAQ_INDEX_DDL:
                    cur.execute(statement)
                dimension = int(self.embedder.dimension)
                cur.execute(
                    "ALTER TABLE faq_questions ADD COLUMN IF NOT EXISTS embedding "
                    "vector(" + str(dimension) + ")"
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS faq_questions_embedding_hnsw_idx
                    ON faq_questions USING hnsw (embedding vector_cosine_ops)
                    """
                )

    # -- writing -----------------------------------------------------------

    def remember(self, question: str, question_key: str, cache_key: str) -> None:
        """Remember a question, so the next one can be matched against it.

        The same question asked again updates the one row rather than
        adding another, and the row is given the key its answer is
        stored under, so a match leads straight to the answer.
        """
        embedding = self.embedder.embed_one(question)
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO faq_questions (
                        question, question_key, cache_key, embedding
                    )
                    VALUES (
                        %(question)s, %(question_key)s, %(cache_key)s,
                        %(embedding)s::vector
                    )
                    ON CONFLICT (question_key) DO UPDATE SET
                        question   = EXCLUDED.question,
                        cache_key  = EXCLUDED.cache_key,
                        embedding  = EXCLUDED.embedding,
                        updated_at = now()
                    """,
                    {
                        "question": question,
                        "question_key": question_key,
                        "cache_key": cache_key,
                        "embedding": as_vector_literal(embedding),
                    },
                )

        self._writes += 1
        if self._writes % PRUNE_EVERY == 0:
            self.prune()

    # -- reading -----------------------------------------------------------

    def match(self, question: str, threshold: float) -> FaqCandidate | None:
        """The stored question this one means the same as, if there is one.

        The nearest question is only returned when it is at least
        ``threshold`` close.  Below that the two questions are merely
        about the same thing, and answering one with the other would be
        a guess.
        """
        embedding = self.embedder.embed_one(question)
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT question, question_key, cache_key,
                           1 - (embedding <=> %(embedding)s::vector) AS similarity
                    FROM faq_questions
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %(embedding)s::vector
                    LIMIT %(limit)s
                    """,
                    {
                        "embedding": as_vector_literal(embedding),
                        "limit": CANDIDATE_LIMIT,
                    },
                )
                rows = cur.fetchall()

        best: FaqCandidate | None = None
        for row in rows:
            similarity = float(row["similarity"] or 0.0)
            if similarity < threshold:
                continue
            if best is not None and similarity <= best.similarity:
                continue
            best = FaqCandidate(
                question=row["question"],
                question_key=row["question_key"],
                cache_key=row["cache_key"],
                similarity=similarity,
            )
        return best

    def prune(self) -> int:
        """Keep only the most recently asked questions.

        The table is a pool of candidates for the semantic search, not
        the record of what has been asked - that is the registry in
        Redis.  The questions nobody has asked lately are of no use to
        the search and are dropped.
        """
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM faq_questions
                    WHERE id NOT IN (
                        SELECT id FROM faq_questions
                        ORDER BY updated_at DESC
                        LIMIT %(limit)s
                    )
                    """,
                    {"limit": self.max_entries},
                )
                return int(cur.rowcount)

    def count(self) -> int:
        """How many questions the index holds."""
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) AS n FROM faq_questions")
                return int(cur.fetchone()["n"])


class NullFaqMatcher:
    """The matcher that finds nothing.

    Used when the FAQ layer is switched off, so the rest of the pipeline
    does not have to know whether there is an index behind it or not.
    """

    def ensure_schema(self) -> None:
        return None

    def match(self, question: str, threshold: float) -> FaqCandidate | None:
        return None

    def remember(self, question: str, question_key: str, cache_key: str) -> None:
        return None

    def prune(self) -> int:
        return 0

    def count(self) -> int:
        return 0
