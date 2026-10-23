"""Tests for the version of the corpus.

Every answer kept in the memory was built from a particular version of the
corpus.  When the corpus is rebuilt the answers built from the old version
are wrong, and the memory has to be told - which is what the **generation**
is for: a counter that goes up every time the corpus changes, and that is
mixed into every cache key.

The key and the fingerprint need no database.  The state and the indexer do,
in a database of their own, and skip themselves when PostgreSQL is not there.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psycopg
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from indexing.chunker import Chunk  # noqa: E402
from indexing.indexer import Indexer, document_fingerprint  # noqa: E402
from memory.cache import build_cache_key  # noqa: E402
from retrieval.config import DatabaseSettings  # noqa: E402
from retrieval.database import Database  # noqa: E402
from retrieval.index_state import UNKNOWN_GENERATION, IndexState  # noqa: E402
from retrieval.vector_store import VectorStore  # noqa: E402

#: The suffix of the database the live tests use.  They get a database of
#: their own, so the version of the real corpus is never moved by a test.
TEST_DATABASE_SUFFIX = "_index_state_test"

#: The source name the live tests index under.
TEST_SOURCE = "test-index-state"

#: The dimension of the default embedding model.
EMBEDDING_DIMENSION = 384


class FakeEmbedder:
    """An embedder that returns the same unit vector for everything.

    The tests are about what happens to the rows and to the version, not
    about what the vectors mean, so the real model is not needed.
    """

    def __init__(
        self,
        dimension: int = EMBEDDING_DIMENSION,
        model: str = "fake-embedder",
    ) -> None:
        self._dimension = dimension
        self.embedded: list[str] = []
        self.settings = SimpleNamespace(model=model)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_one(self, text: str) -> list[float]:
        self.embedded.append(text)
        vector = [0.0] * self._dimension
        vector[0] = 1.0
        return vector


class RecordingState(IndexState):
    """A state that says what it was asked to do, and in what order."""

    def __init__(self, database: Database, calls: list[str]) -> None:
        super().__init__(database)
        self.calls = calls

    def record_document(
        self,
        source: str,
        fingerprint: str,
        conn: psycopg.Connection | None = None,
    ) -> None:
        self.calls.append("record")
        super().record_document(source, fingerprint, conn=conn)

    def bump(self, conn: psycopg.Connection | None = None) -> int:
        self.calls.append("bump")
        return super().bump(conn=conn)


class FailingState(IndexState):
    """A state whose version cannot be moved on."""

    def bump(self, conn: psycopg.Connection | None = None) -> int:
        raise RuntimeError("the version cannot be moved on")


class RecordingStore(VectorStore):
    """A store that says when it wrote."""

    def __init__(self, database: Database, calls: list[str]) -> None:
        super().__init__(database)
        self.calls = calls

    def upsert_many(
        self,
        chunks: Sequence[tuple[str, Sequence[float], dict[str, Any]]],
        conn: psycopg.Connection | None = None,
    ) -> int:
        self.calls.append("upsert")
        return super().upsert_many(chunks, conn=conn)

#
# The key
#


def test_the_key_carries_the_version_of_the_corpus() -> None:
    """Two versions of the corpus are two different keys."""
    first = build_cache_key("What projects did Rose work on?", "", 1)
    second = build_cache_key("What projects did Rose work on?", "", 2)

    assert first != second


def test_the_same_version_gives_the_same_key() -> None:
    """Nothing about the version makes the key change on its own."""
    first = build_cache_key("What projects did Rose work on?", "", 3)
    second = build_cache_key("What projects did Rose work on?", "", 3)

    assert first == second


def test_a_key_without_a_version_is_the_key_of_version_zero() -> None:
    """A service with no state behind it still has a key."""
    assert build_cache_key("What projects?") == build_cache_key(
        "What projects?", "", 0
    )


#
# The fingerprint
#


def _chunk(text: str, index: int = 0) -> Chunk:
    """A chunk shaped like the ones the chunker produces."""
    return Chunk(
        chunk_text=text,
        metadata={"source": TEST_SOURCE, "chunk_index": index, "page": 1},
    )


def test_the_fingerprint_of_a_document_is_the_same_every_time() -> None:
    """Two runs over the same document produce the same fingerprint."""
    chunks = [_chunk("first chunk"), _chunk("second chunk", 1)]

    assert document_fingerprint(chunks) == document_fingerprint(chunks)


def test_a_changed_text_changes_the_fingerprint() -> None:
    """The text is part of what the document looks like."""
    assert document_fingerprint([_chunk("first")]) != document_fingerprint(
        [_chunk("second")]
    )


def test_a_changed_metadata_changes_the_fingerprint() -> None:
    """The metadata is part of it too - the chunks are not the same."""
    changed = Chunk(
        chunk_text="first",
        metadata={"source": TEST_SOURCE, "chunk_index": 0, "page": 2},
    )

    assert document_fingerprint([_chunk("first")]) != document_fingerprint([changed])


#
# The state, against a live PostgreSQL
#


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
    import psycopg

    name = database.settings.dbname
    admin = _settings_for("postgres")
    try:
        with psycopg.connect(admin.conninfo(), autocommit=True) as conn:
            conn.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s",
                (name,),
            )
            conn.execute(f'DROP DATABASE IF EXISTS "{name}"')
    except Exception:  # pragma: no cover - environment dependent
        pass


@pytest.fixture(scope="module")
def state_database() -> Iterator[Database]:
    """A database of the tests own, if PostgreSQL can be reached."""
    settings = DatabaseSettings.from_env()
    database = Database(_settings_for(settings.dbname + TEST_DATABASE_SUFFIX))
    try:
        database.ensure_database()
        with database.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"PostgreSQL not reachable: {exc}")

    yield database

    _drop_database(database)


@pytest.fixture
def state(state_database: Database) -> Iterator[IndexState]:
    """The state, with the tables emptied before the test."""
    instance = IndexState(state_database)
    instance.ensure_schema()
    with instance.db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM indexed_documents")
            cur.execute("UPDATE index_state SET generation = 1 WHERE id = 1")
    yield instance
    with instance.db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM indexed_documents")
            cur.execute("UPDATE index_state SET generation = 1 WHERE id = 1")


def test_the_version_starts_at_one(state: IndexState) -> None:
    """A corpus that has never been written has the first version."""
    assert state.generation() == 1


def test_the_version_goes_up(state: IndexState) -> None:
    """Every write moves the version on."""
    assert state.bump() == 2
    assert state.generation() == 2
    assert state.bump() == 3
    assert state.generation() == 3


def test_a_state_that_cannot_be_read_is_version_zero() -> None:
    """The state fails open: a database that is not there is version zero."""
    settings = DatabaseSettings.from_env()
    missing = Database(_settings_for(settings.dbname + "_does_not_exist"))

    assert IndexState(missing).generation() == UNKNOWN_GENERATION
    assert IndexState(missing).document_fingerprint("anything") == ""


def test_a_state_that_cannot_be_written_raises() -> None:
    """Writing the state does not fail open: a corpus is never left unversioned."""
    settings = DatabaseSettings.from_env()
    missing = Database(_settings_for(settings.dbname + "_does_not_exist"))

    with pytest.raises(psycopg.Error):
        IndexState(missing).bump()
    with pytest.raises(psycopg.Error):
        IndexState(missing).record_document("anything", "abc123")
    with pytest.raises(psycopg.Error):
        IndexState(missing).forget_document("anything")


def test_the_fingerprint_of_a_document_is_remembered(state: IndexState) -> None:
    """What a document looked like when it was last indexed is kept."""
    assert state.document_fingerprint(TEST_SOURCE) == ""

    state.record_document(TEST_SOURCE, "abc123")

    assert state.document_fingerprint(TEST_SOURCE) == "abc123"


def test_a_document_that_is_gone_is_forgotten(state: IndexState) -> None:
    """A document that is no longer indexed keeps no fingerprint."""
    state.record_document(TEST_SOURCE, "abc123")

    state.forget_document(TEST_SOURCE)

    assert state.document_fingerprint(TEST_SOURCE) == ""


#
# The indexer, against a live PostgreSQL
#


@pytest.fixture
def indexer(state_database: Database, state: IndexState) -> Indexer:
    """An indexer writing into the test database."""
    instance = Indexer(state_database, FakeEmbedder(), index_state=state)
    instance.prepare()
    with instance.db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE source = %s", (TEST_SOURCE,))
    return instance


def test_a_document_that_has_not_changed_is_not_written_again(
    indexer: Indexer, state: IndexState
) -> None:
    """A rebuild that changes nothing is skipped - the memory is kept."""
    chunks = [_chunk("hello")]

    assert indexer.index_chunks(chunks) == 1
    version = state.generation()

    assert indexer.index_chunks(chunks) == 0
    assert state.generation() == version


def test_a_document_that_has_changed_moves_the_version_on(
    indexer: Indexer, state: IndexState
) -> None:
    """A rebuild that changes something invalidates the memory."""
    indexer.index_chunks([_chunk("hello")])
    version = state.generation()

    indexer.index_chunks([_chunk("goodbye")])

    assert state.generation() == version + 1


def test_the_rows_are_written_when_the_document_has_changed(
    indexer: Indexer, state: IndexState
) -> None:
    """The new text is what the index holds."""
    indexer.index_chunks([_chunk("hello")])
    indexer.index_chunks([_chunk("goodbye")])

    with indexer.db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_text FROM chunks WHERE source = %s", (TEST_SOURCE,)
            )
            rows = [row["chunk_text"] for row in cur.fetchall()]

    assert rows == ["goodbye"]


def test_an_indexer_with_no_state_still_writes(
    state_database: Database, state: IndexState
) -> None:
    """The version is optional - the indexer works without it."""
    instance = Indexer(state_database, FakeEmbedder())
    instance.prepare()
    with instance.db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE source = %s", (TEST_SOURCE,))

    assert instance.index_chunks([_chunk("hello")]) == 1


def test_a_document_of_no_chunks_writes_nothing(
    indexer: Indexer, state: IndexState
) -> None:
    """Nothing to index is not a change to the corpus."""
    version = state.generation()

    assert indexer.index_chunks([]) == 0
    assert state.generation() == version


def test_the_fingerprint_is_kept_of_what_was_indexed(
    indexer: Indexer, state: IndexState
) -> None:
    """The state knows what the document looked like when it was indexed."""
    chunks = [_chunk("hello")]

    indexer.index_chunks(chunks)

    assert state.document_fingerprint(TEST_SOURCE) == document_fingerprint(
        chunks, "fake-embedder"
    )


def test_every_chunk_is_embedded_before_it_is_written(
    indexer: Indexer, state: IndexState
) -> None:
    """The embedding happens first, so a failure costs nothing."""
    embedder = indexer.embedder

    indexer.index_chunks([_chunk("first"), _chunk("second", 1)])

    assert isinstance(embedder, FakeEmbedder)
    assert embedder.embedded == ["first", "second"]


#
# The model, the fingerprint and the write
#


def test_the_model_is_part_of_the_fingerprint() -> None:
    """The same text under another model is another document."""
    chunks = [_chunk("hello")]

    assert document_fingerprint(chunks, "model-a") != document_fingerprint(
        chunks, "model-b"
    )


def test_a_document_indexed_with_another_model_is_written_again(
    state_database: Database, state: IndexState
) -> None:
    """A change of the embedding model writes the document again."""
    chunks = [_chunk("hello")]
    first = Indexer(
        state_database, FakeEmbedder(model="model-a"), index_state=state
    )
    first.index_chunks(chunks)
    version = state.generation()

    second = Indexer(
        state_database, FakeEmbedder(model="model-b"), index_state=state
    )
    second.index_chunks(chunks)

    assert state.generation() == version + 1


def test_a_document_that_has_not_changed_is_written_when_forced(
    indexer: Indexer, state: IndexState
) -> None:
    """``force`` writes it anyway, and the version moves on."""
    chunks = [_chunk("hello")]
    indexer.index_chunks(chunks)
    version = state.generation()

    assert indexer.index_chunks(chunks, force=True) == 1
    assert state.generation() == version + 1


#
# The write is one transaction
#


def test_the_version_moves_on_after_the_rows_are_written(
    state_database: Database, state: IndexState
) -> None:
    """The rows are written first, and the version moves on last."""
    calls: list[str] = []
    instance = Indexer(
        state_database,
        FakeEmbedder(),
        index_state=RecordingState(state_database, calls),
    )
    instance.prepare()
    instance.vector_store = RecordingStore(state_database, calls)

    instance.index_chunks([_chunk("hello")])

    assert calls == ["upsert", "record", "bump"]


def test_a_write_that_cannot_move_the_version_leaves_nothing_behind(
    state_database: Database, state: IndexState
) -> None:
    """The write is one transaction - or it is nothing at all."""
    instance = Indexer(
        state_database,
        FakeEmbedder(),
        index_state=FailingState(state_database),
    )
    instance.prepare()
    with state_database.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chunks WHERE source = %s",
                (TEST_SOURCE,),
            )

    with pytest.raises(RuntimeError):
        instance.index_chunks([_chunk("hello")])

    with state_database.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) AS n FROM chunks WHERE source = %s",
                (TEST_SOURCE,),
            )
            assert int(cur.fetchone()["n"]) == 0

    assert state.generation() == 1
    assert state.document_fingerprint(TEST_SOURCE) == ""
