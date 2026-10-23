"""Tests for a filtered search over an approximate index.

pgvector applies the WHERE filter of a search *after* the approximate
scan.  The scan visits a bounded number of rows and the filter throws away
the ones that do not match, so a search that filters on a column the HNSW
index does not cover can come back with fewer rows than it was asked for -
including none at all.  A semantic match silently stops matching, right
after a rebuild has filled the neighbourhood with the rows of the corpus it
has just replaced.

The corpus in these tests is built so that this shows.  The rows nearest
the query are exactly the rows that do *not* match the filter, and the rows
that do match are further away.  Without the iterative scan the search
finds the nearest rows, the filter throws all of them away, and the answer
is empty.  With the iterative scan the search keeps going until it has the
rows it was asked for, in the order the distance puts them in.

These tests need PostgreSQL, and skip themselves when it is not there.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from psycopg import sql

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from retrieval.config import DatabaseSettings  # noqa: E402
from retrieval.database import Database  # noqa: E402
from retrieval.faq_store import PostgresFaqMatcher  # noqa: E402
from retrieval.vector_store import (  # noqa: E402
    VectorStore,
    as_vector_literal,
)

#: The suffix of the database the live tests use.
TEST_DATABASE_SUFFIX = "_ann_filter_test"

#: The source the matching rows are written under.
TEST_SOURCE = "ann-filter-match"

#: The source the rows nearest the query are written under.
OTHER_SOURCE = "ann-filter-near"

#: The project the matching rows are written under.  The search filters on
#: the project and not on the source, because the source has an index of
#: its own: the planner would then answer the filter from that index, the
#: search would not be an approximate one at all, and the bug this file is
#: about could not show.
TEST_PROJECT = "ann-filter-match"

#: The project the rows nearest the query are written under.
OTHER_PROJECT = "ann-filter-near"

#: How many rows are in the table.
ROW_COUNT = 100

#: How many of the rows nearest the query do not match the filter.
NEAR_COUNT = 60

#: The dimension of the vectors.
DIMENSION = 384

#: The seed of the corpus.  Fixed, so the corpus is the same every time.
SEED = 20241021

#: The generation the matching FAQ questions are written under.
MATCHING_GENERATION = 2

#: The generation the questions nearest the query are written under.
OTHER_GENERATION = 1


class FakeEmbedder:
    """An embedder that hands out the vectors it was given.

    The tests are about what a filtered search returns, not about what the
    vectors mean, so the real model is not needed.
    """

    def __init__(
        self,
        vectors: dict[str, list[float]],
        default: list[float],
        dimension: int = DIMENSION,
    ) -> None:
        self.vectors = vectors
        self.default = default
        self.dimension = dimension

    def embed_one(self, text: str) -> list[float]:
        return self.vectors.get(text, self.default)


def _unit_vector(rng: random.Random) -> list[float]:
    """A random vector of unit length."""
    vector = [rng.gauss(0.0, 1.0) for _ in range(DIMENSION)]
    length = sum(value * value for value in vector) ** 0.5
    return [value / length for value in vector]


def _similarity(left: list[float], right: list[float]) -> float:
    """The cosine similarity of two unit vectors."""
    return sum(a * b for a, b in zip(left, right))


def _ranked(query: list[float], vectors: list[list[float]]) -> list[int]:
    """The rows nearest the query first, counted exactly."""
    return sorted(
        range(len(vectors)),
        key=lambda index: -_similarity(query, vectors[index]),
    )


def _settings_for(dbname: str) -> DatabaseSettings:
    """The settings of the configured database, under another name."""
    settings = DatabaseSettings.from_env()
    return DatabaseSettings(
        host=settings.host,
        port=settings.port,
        user=settings.user,
        password=settings.password,
        dbname=dbname,
    )


def _drop_database(database: Database) -> None:
    """Remove the test database again."""
    name = database.settings.dbname
    admin = _settings_for("postgres")
    try:
        with psycopg.connect(admin.conninfo(), autocommit=True) as conn:
            conn.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity ",
                "WHERE datname = %s",
                (name,),
            )
            conn.execute(
                sql.SQL("DROP DATABASE IF EXISTS {}").format(
                    sql.Identifier(name)
                )
            )
    except Exception:  # pragma: no cover - environment dependent
        pass


@pytest.fixture(scope="module")
def ann_database() -> Iterator[Database]:
    """A database of the tests own, with the planner kept off a table scan.

    The rows are few, so the planner would rather read the table than use
    the index - and a table scan finds the rows whether the filter is
    applied before the search or after it.  The setting makes the search go
    through the HNSW index, which is what a real corpus does.
    """
    settings = DatabaseSettings.from_env()
    database = Database(_settings_for(settings.dbname + TEST_DATABASE_SUFFIX))
    try:
        database.ensure_database()
        admin = _settings_for("postgres")
        with psycopg.connect(admin.conninfo(), autocommit=True) as conn:
            conn.execute(
                sql.SQL("ALTER DATABASE {} SET enable_seqscan = off").format(
                    sql.Identifier(database.settings.dbname)
                )
            )
        with database.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"PostgreSQL not reachable: {exc}")

    yield database

    _drop_database(database)


class Corpus:
    """The corpus the tests search through."""

    def __init__(
        self,
        query: list[float],
        vectors: list[list[float]],
        near: set[int],
    ) -> None:
        self.query = query
        self.vectors = vectors
        self.near = near
        self.ranked = _ranked(query, vectors)

    @property
    def nearest_matching(self) -> list[int]:
        """The matching rows, nearest the query first."""
        return [index for index in self.ranked if index not in self.near]

    def text_of(self, index: int) -> str:
        """The chunk text of a row."""
        return f"row {index}"


@pytest.fixture(scope="module")
def corpus(ann_database: Database) -> Corpus:
    """The corpus: random vectors, the nearest ones behind the filter."""
    rng = random.Random(SEED)
    query = _unit_vector(rng)
    vectors = [_unit_vector(rng) for _ in range(ROW_COUNT)]
    ranked = _ranked(query, vectors)
    near = set(ranked[:NEAR_COUNT])

    database = ann_database
    database.create_extension()
    database.ensure_schema(DIMENSION)
    store = VectorStore(database)
    with database.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks")

    rows = []
    for index in range(ROW_COUNT):
        matching = index not in near
        source = TEST_SOURCE if matching else OTHER_SOURCE
        project = TEST_PROJECT if matching else OTHER_PROJECT
        rows.append(
            (
                f"row {index}",
                vectors[index],
                {
                    "source": source,
                    "project": project,
                    "chunk_index": index,
                    "page": 1,
                },
            )
        )
    store.upsert_many(rows)

    return Corpus(query, vectors, near)


def _faq_embedder(corpus: Corpus) -> FakeEmbedder:
    """An embedder that gives every question the vector of its row."""
    vectors = {
        f"faq {index}": corpus.vectors[index] for index in range(ROW_COUNT)
    }
    vectors["the question"] = corpus.query
    return FakeEmbedder(vectors, corpus.query)


@pytest.fixture(scope="module")
def faq_corpus(ann_database: Database, corpus: Corpus) -> PostgresFaqMatcher:
    """The FAQ index, with the questions nearest the query behind the filter.
    """
    database = ann_database
    matcher = PostgresFaqMatcher(database, _faq_embedder(corpus))
    matcher.ensure_schema()
    with database.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM faq_questions")

    for index in range(ROW_COUNT):
        generation = (
            OTHER_GENERATION if index in corpus.near else MATCHING_GENERATION
        )
        matcher.remember(
            f"faq {index}",
            f"key {index}",
            f"cache {index}",
            generation=generation,
        )

    return matcher

#
# The filtered search over the chunks
#


def test_a_filtered_search_returns_the_rows_that_match(
    corpus: Corpus, ann_database: Database
) -> None:
    """The filter is not allowed to throw the answer away."""
    store = VectorStore(ann_database)

    hits = store.search(corpus.query, limit=5, filters={"project": TEST_PROJECT})

    assert len(hits) == 5
    assert all(hit.metadata["project"] == TEST_PROJECT for hit in hits)


def test_a_filtered_search_returns_the_nearest_matching_rows(
    corpus: Corpus, ann_database: Database
) -> None:
    """The rows that come back are the nearest ones that match."""
    store = VectorStore(ann_database)

    hits = store.search(corpus.query, limit=5, filters={"project": TEST_PROJECT})

    expected = corpus.nearest_matching[:5]
    assert {hit.chunk_text for hit in hits} == {
        corpus.text_of(index) for index in expected
    }


def test_an_unfiltered_search_returns_the_nearest_rows(
    corpus: Corpus, ann_database: Database
) -> None:
    """Without a filter the search returns the nearest rows of all."""
    store = VectorStore(ann_database)

    hits = store.search(corpus.query, limit=5)

    assert {hit.chunk_text for hit in hits} == {
        corpus.text_of(index) for index in corpus.ranked[:5]
    }
    assert all(hit.metadata["source"] == OTHER_SOURCE for hit in hits)


def test_the_filtered_search_goes_through_the_vector_index(
    corpus: Corpus, ann_database: Database
) -> None:
    """The search is an approximate one - otherwise nothing is tested.

    The planner has to reach for the HNSW index, and the filter has to
    be applied to the rows the index returns.  A filter another index
    could answer would hide the bug, which is why the search filters on
    the project.
    """
    with ann_database.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                EXPLAIN (COSTS OFF)
                SELECT id FROM chunks
                WHERE embedding IS NOT NULL
                  AND metadata->>'project' = %(project)s
                ORDER BY embedding <=> %(embedding)s::vector
                LIMIT %(limit)s
                """,
                {
                    "project": TEST_PROJECT,
                    "embedding": as_vector_literal(corpus.query),
                    "limit": 5,
                },
            )
            plan = " ".join(row["QUERY PLAN"] for row in cur.fetchall())

    assert "chunks_embedding_hnsw_idx" in plan


#
# The filtered search over the FAQ index
#


def test_a_filtered_faq_search_returns_a_question_of_this_version(
    corpus: Corpus, faq_corpus: PostgresFaqMatcher
) -> None:
    """The FAQ search finds a question of the current version of the corpus.
    """
    candidate = faq_corpus.match(
        "the question", threshold=-1.0, generation=MATCHING_GENERATION
    )

    assert candidate is not None
    assert candidate.question.startswith("faq ")


def test_a_filtered_faq_search_returns_the_nearest_question_of_this_version(
    corpus: Corpus, faq_corpus: PostgresFaqMatcher
) -> None:
    """The question that comes back is one of the nearest of its version."""
    candidate = faq_corpus.match(
        "the question", threshold=-1.0, generation=MATCHING_GENERATION
    )

    assert candidate is not None
    expected = corpus.nearest_matching[:5]
    assert candidate.question in {f"faq {index}" for index in expected}


def test_a_faq_search_does_not_see_the_questions_of_another_version(
    corpus: Corpus, faq_corpus: PostgresFaqMatcher
) -> None:
    """A question of an older corpus is not a match for this one."""
    candidate = faq_corpus.match(
        "the question", threshold=-1.0, generation=OTHER_GENERATION
    )

    assert candidate is not None
    assert candidate.question in {
        f"faq {index}" for index in corpus.ranked[:NEAR_COUNT]
    }

