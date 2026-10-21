"""Tests for the PostgreSQL storage and retrieval layer.

These tests need a live PostgreSQL with the ``vector`` extension.  They skip
themselves when the database is not reachable, so the suite still passes on a
machine without PostgreSQL.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from retrieval.config import DatabaseSettings  # noqa: E402
from retrieval.database import Database  # noqa: E402
from retrieval.text_search import TextSearchStore  # noqa: E402
from retrieval.vector_store import VectorStore  # noqa: E402

TEST_SOURCE = "test-storage"

#: The dimension of the default embedding model
#: (sentence-transformers/all-MiniLM-L6-v2).  The live database column must
#: match it, so the suite skips rather than fails when a different model is
#: configured.
EMBEDDING_DIMENSION = 384


def _database() -> Database:
    """Build a Database from the environment."""
    return Database(DatabaseSettings.from_env())


@pytest.fixture(scope="module")
def db() -> Database:
    database = _database()
    try:
        with database.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"PostgreSQL not reachable: {exc}")
    return database


@pytest.fixture(scope="module")
def dimension(db: Database) -> int:
    """The dimension of the live embedding column."""
    db.create_extension()
    db.ensure_schema(EMBEDDING_DIMENSION)
    found = db.vector_dimension()
    if found != EMBEDDING_DIMENSION:
        pytest.skip(
            f"live embedding column is vector({found}), "
            f"tests need vector({EMBEDDING_DIMENSION})"
        )
    return found


@pytest.fixture(scope="module")
def schema(db: Database, dimension: int) -> None:
    """Remove the test rows before the suite runs."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE source = %s", (TEST_SOURCE,))


def _unit_vector(index: int, size: int) -> list[float]:
    """A deterministic unit vector, distinct for each index."""
    vector = [0.0] * size
    vector[index % size] = 1.0
    return vector


@pytest.fixture
def store(db: Database, schema: None, dimension: int) -> VectorStore:
    """A vector store with the test rows removed before each test."""
    vector_store = VectorStore(db)
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE source = %s", (TEST_SOURCE,))
    return vector_store


def test_upsert_stores_text_and_metadata(store: VectorStore):
    chunk_id = store.upsert(
        "Kubernetes orchestration with Docker",
        _unit_vector(0, EMBEDDING_DIMENSION),
        {
            "source": TEST_SOURCE,
            "section": "EXPERIENCE",
            "page": 2,
            "end_page": 2,
            "chunk_index": 0,
            "project": "TruStage",
        },
    )
    assert chunk_id > 0

    hit = store.get(chunk_id)
    assert hit is not None
    assert "Kubernetes" in hit.chunk_text
    assert hit.metadata["project"] == "TruStage"
    assert hit.metadata["page"] == 2


def test_upsert_is_idempotent_per_source_and_index(store: VectorStore):
    """Re-indexing the same chunk must update, not duplicate."""
    metadata = {"source": TEST_SOURCE, "chunk_index": 7, "section": "S"}
    first = store.upsert("first version", _unit_vector(1, EMBEDDING_DIMENSION), metadata)
    second = store.upsert("second version", _unit_vector(1, EMBEDDING_DIMENSION), metadata)

    assert first == second
    with store.db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) AS n FROM chunks WHERE source = %s AND chunk_index = 7",
                (TEST_SOURCE,),
            )
            assert cur.fetchone()["n"] == 1


def test_dense_search_ranks_the_nearest_vector_first(store: VectorStore):
    store.upsert("far away", _unit_vector(0, EMBEDDING_DIMENSION), {"source": TEST_SOURCE, "chunk_index": 0})
    store.upsert("closest", _unit_vector(1, EMBEDDING_DIMENSION), {"source": TEST_SOURCE, "chunk_index": 1})

    hits = store.search(_unit_vector(1, EMBEDDING_DIMENSION), limit=2)
    assert hits
    assert hits[0].chunk_text == "closest"
    assert hits[0].score > hits[1].score


def test_dense_search_respects_filters(store: VectorStore):
    store.upsert(
        "project one",
        _unit_vector(0, EMBEDDING_DIMENSION),
        {"source": TEST_SOURCE, "chunk_index": 0, "project": "Alpha"},
    )
    store.upsert(
        "project two",
        _unit_vector(0, EMBEDDING_DIMENSION),
        {"source": TEST_SOURCE, "chunk_index": 1, "project": "Beta"},
    )

    hits = store.search(_unit_vector(0, EMBEDDING_DIMENSION), limit=5, filters={"project": "Beta"})
    assert len(hits) == 1
    assert hits[0].chunk_text == "project two"


def test_delete_by_source_removes_only_that_source(store: VectorStore):
    store.upsert("mine", _unit_vector(0, EMBEDDING_DIMENSION), {"source": TEST_SOURCE, "chunk_index": 0})
    store.upsert("other", _unit_vector(1, EMBEDDING_DIMENSION), {"source": "some-other-doc", "chunk_index": 0})

    store.delete_by_source(TEST_SOURCE)

    with store.db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM chunks WHERE source = %s", (TEST_SOURCE,))
            assert cur.fetchone()["n"] == 0
            cur.execute("SELECT count(*) AS n FROM chunks WHERE source = %s", ("some-other-doc",))
            assert cur.fetchone()["n"] == 1
    with store.db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE source = %s", ("some-other-doc",))


# -- keyword search --------------------------------------------------------


@pytest.fixture
def text_store(db: Database, schema: None) -> TextSearchStore:
    store = TextSearchStore(db)
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE source = %s", (TEST_SOURCE,))
    return store


def test_keyword_search_matches_terms(text_store: TextSearchStore):
    text_store.upsert(
        "Built GenAI chat assistants with LangChain",
        {"source": TEST_SOURCE, "chunk_index": 0, "section": "EXPERIENCE"},
    )
    text_store.upsert(
        "Managed Kubernetes clusters on Azure",
        {"source": TEST_SOURCE, "chunk_index": 1, "section": "EXPERIENCE"},
    )

    hits = text_store.search("langchain", limit=50)
    mine = [hit for hit in hits if hit.metadata.get("source") == TEST_SOURCE]
    assert len(mine) == 1
    assert "LangChain" in mine[0].chunk_text
    assert mine[0].rank > 0


def test_keyword_search_stems_english(text_store: TextSearchStore):
    """'running' must match 'run' - the english config stems both."""
    text_store.upsert(
        "Engineering teams running the deployment pipeline",
        {"source": TEST_SOURCE, "chunk_index": 0},
    )
    hits = text_store.search("run", limit=5)
    assert any("running the deployment" in hit.chunk_text for hit in hits)


def test_keyword_search_returns_no_match_for_absent_terms(text_store: TextSearchStore):
    text_store.upsert(
        "Built GenAI chat assistants",
        {"source": TEST_SOURCE, "chunk_index": 0},
    )
    hits = text_store.search("kubernetes", limit=5)
    assert not any(hit.metadata.get("source") == TEST_SOURCE for hit in hits)


def test_keyword_search_reports_matched_terms(text_store: TextSearchStore):
    text_store.upsert(
        "Kubernetes and Docker containers",
        {"source": TEST_SOURCE, "chunk_index": 0},
    )
    hits = text_store.search("kubernetes", limit=5)
    assert hits
    # The terms come back stemmed, the way the tsvector stores them.
    assert "kubernet" in hits[0].matched_terms


def test_empty_query_returns_nothing(text_store: TextSearchStore):
    assert text_store.search("   ", limit=5) == []


# -- fusion ----------------------------------------------------------------


def test_rrf_fusion_prefers_chunks_ranked_in_both_indexes():
    from retrieval.retriever import HybridRetriever

    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.rrf_k = 60

    dense = [
        _hit(1, "dense only", 0.9),
        _hit(2, "in both", 0.8),
    ]
    sparse = [
        _sparse_hit(2, "in both", 0.5),
        _sparse_hit(3, "sparse only", 0.4),
    ]

    fused = retriever.fuse(dense, sparse)
    assert fused[0].chunk_id == 2
    assert fused[0].dense_rank == 2
    assert fused[0].sparse_rank == 1
    # A chunk in both lists must beat one in a single list.
    assert fused[0].score > fused[1].score


def test_rrf_score_matches_the_formula():
    from retrieval.retriever import HybridRetriever

    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.rrf_k = 60
    assert retriever._rrf_score(1, 1) == pytest.approx(2 / 61)
    assert retriever._rrf_score(1, None) == pytest.approx(1 / 61)
    assert retriever._rrf_score(None, None) == 0.0


def _hit(chunk_id: int, text: str, score: float):
    from retrieval.vector_store import SearchHit

    return SearchHit(
        chunk_id=chunk_id,
        chunk_text=text,
        metadata={"source": TEST_SOURCE},
        score=score,
    )


def _sparse_hit(chunk_id: int, text: str, rank: float):
    from retrieval.text_search import TextSearchHit

    return TextSearchHit(
        chunk_id=chunk_id,
        chunk_text=text,
        metadata={"source": TEST_SOURCE},
        rank=rank,
        matched_terms=["test"],
    )
