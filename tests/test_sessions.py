"""Tests for the PostgreSQL-backed conversation history.

These tests need a live PostgreSQL.  They skip themselves when the database
is not reachable, so the suite still passes on a machine without one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from api.session_store import SessionStore  # noqa: E402
from retrieval.config import DatabaseSettings  # noqa: E402
from retrieval.database import Database  # noqa: E402

#: A conversation id used only by this suite, so it never collides with a
#: real session.
TEST_CONVERSATION = "test-session-store"

#: A second conversation id, used to check that sessions stay separate.
OTHER_CONVERSATION = "test-session-store-other"


@pytest.fixture(scope="module")
def _session_store() -> SessionStore:
    database = Database(DatabaseSettings.from_env())
    try:
        with database.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"PostgreSQL not reachable: {exc}")

    session_store = SessionStore(database)
    session_store.ensure_schema()
    return session_store


@pytest.fixture
def store(_session_store: SessionStore) -> SessionStore:
    """A store with the test conversations emptied, so tests do not share state."""
    for conversation in (TEST_CONVERSATION, OTHER_CONVERSATION):
        _session_store.clear(conversation)
    yield _session_store
    for conversation in (TEST_CONVERSATION, OTHER_CONVERSATION):
        _session_store.clear(conversation)


def test_a_conversation_is_created_on_the_first_message(store: SessionStore):
    store.append_message(TEST_CONVERSATION, "user", "First question")

    history = store.history(TEST_CONVERSATION)
    assert len(history) == 1
    assert history[0].role == "user"
    assert history[0].content == "First question"


def test_messages_come_back_oldest_first(store: SessionStore):
    store.append_message(TEST_CONVERSATION, "user", "First")
    store.append_message(TEST_CONVERSATION, "assistant", "Answer")
    store.append_message(TEST_CONVERSATION, "user", "Second")

    history = store.history(TEST_CONVERSATION)
    assert [message.content for message in history][-3:] == [
        "First",
        "Answer",
        "Second",
    ]


def test_sources_are_stored_with_an_answer(store: SessionStore):
    sources = [
        {
            "chunk_id": 7,
            "text": "A passage",
            "metadata": {"page": 2, "section": "EXPERIENCE"},
            "score": 0.031,
        }
    ]
    store.append_message(TEST_CONVERSATION, "assistant", "With sources", sources)

    history = store.history(TEST_CONVERSATION)
    latest = history[-1]
    assert latest.sources[0]["chunk_id"] == 7
    assert latest.sources[0]["metadata"]["page"] == 2


def test_the_limit_keeps_the_newest_messages(store: SessionStore):
    store.append_message(TEST_CONVERSATION, "user", "Oldest")
    store.append_message(TEST_CONVERSATION, "user", "Middle")
    store.append_message(TEST_CONVERSATION, "user", "Newest")

    history = store.history(TEST_CONVERSATION, limit=2)
    assert [message.content for message in history] == ["Middle", "Newest"]


def test_conversations_are_isolated_from_each_other(store: SessionStore):
    store.append_message(TEST_CONVERSATION, "user", "In the test session")
    store.append_message(OTHER_CONVERSATION, "user", "In another session")

    assert len(store.history(TEST_CONVERSATION)) == 1
    assert len(store.history(OTHER_CONVERSATION)) == 1
    assert store.history(OTHER_CONVERSATION)[0].content == "In another session"


def test_an_unknown_conversation_has_no_history(store: SessionStore):
    assert store.history("test-session-store-never-used") == []


def test_an_invalid_role_is_rejected(store: SessionStore):
    with pytest.raises(ValueError):
        store.append_message(TEST_CONVERSATION, "system", "Not a role")
