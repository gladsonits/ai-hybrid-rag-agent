"""Tests for the chat API layer.

These tests do not need a database, an embedding model or an LLM.  The
retriever, the LLM and the session store are replaced with stand-ins, so
the whole request path - routing, validation, the SSE encoding, the
conversation memory - is exercised without a network call.
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from api.app import Services, create_app  # noqa: E402
from api.chat_service import ChatService  # noqa: E402
from api.config import ChatSettings  # noqa: E402
from api.prompt import (  # noqa: E402
    build_messages,
    build_retrieval_query,
    format_context,
)
from api.session_store import StoredMessage  # noqa: E402


# --------------------------------------------------------------------------
# Stand-ins
# --------------------------------------------------------------------------


class FakeChunk:
    """A retrieved chunk, shaped like ``retrieval.retriever.RetrievedChunk``."""

    def __init__(
        self,
        chunk_id: int,
        chunk_text: str,
        metadata: dict[str, Any] | None = None,
        score: float = 0.5,
    ) -> None:
        self.chunk_id = chunk_id
        self.chunk_text = chunk_text
        self.metadata = metadata or {}
        self.score = score


class FakeRetriever:
    """Records the queries it is asked for and returns fixed chunks."""

    def __init__(self, chunks: Sequence[FakeChunk] | None = None) -> None:
        self.chunks = list(chunks or [])
        self.queries: list[str] = []

    def search(
        self,
        query: str,
        limit: int = 10,
        candidates: int = 50,
        filters: dict[str, Any] | None = None,
    ) -> list[FakeChunk]:
        self.queries.append(query)
        return self.chunks[:limit]


class FakeLlm:
    """Returns a fixed answer, and records the messages it was given."""

    def __init__(self, tokens: Sequence[str] = ("Hello", " world")) -> None:
        self.tokens = list(tokens)
        self.messages: list[dict[str, str]] = []

    async def stream(self, messages: Sequence[dict[str, str]]) -> AsyncIterator[str]:
        self.messages = list(messages)
        for token in self.tokens:
            yield token


class FakeSessionStore:
    """An in-memory stand-in for ``api.session_store.SessionStore``."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, str, str, list[dict[str, Any]]]] = []
        self.by_conversation: dict[str, list[StoredMessage]] = {}

    def ensure_schema(self) -> None:
        """No schema to create."""

    def history(
        self,
        conversation_id: str,
        limit: int | None = None,
    ) -> list[StoredMessage]:
        stored = self.by_conversation.get(conversation_id, [])
        return stored[-limit:] if limit is not None else list(stored)

    def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> int:
        self.messages.append((conversation_id, role, content, sources or []))
        self.by_conversation.setdefault(conversation_id, []).append(
            StoredMessage(role=role, content=content, sources=list(sources or []))
        )
        return len(self.messages)


def _services(
    retriever: FakeRetriever | None = None,
    llm: FakeLlm | None = None,
    store: FakeSessionStore | None = None,
) -> Services:
    """Build a Services object backed entirely by stand-ins."""
    retriever = retriever or FakeRetriever()
    llm = llm or FakeLlm()
    store = store or FakeSessionStore()
    chat_service = ChatService(
        retriever=retriever,
        llm=llm,
        session_store=store,
        settings=ChatSettings(
            top_k=5,
            history_turns=5,
            query_history_messages=2,
        ),
    )
    return Services(
        chat_service=chat_service,
        session_store=store,  # type: ignore[arg-type]
        database=None,  # type: ignore[arg-type]
        embedder=None,  # type: ignore[arg-type]
        retriever=retriever,  # type: ignore[arg-type]
        llm=llm,
    )


def _client(services: Services) -> TestClient:
    """A test client for an app that does not run the startup lifespan."""
    app = create_app(services)
    # The lifespan would try to create tables and load the embedding model,
    # which the stand-ins do not support.
    return TestClient(app, raise_server_exceptions=True)


def _events(response: Any) -> list[dict[str, Any]]:
    """Parse an SSE response body into a list of event payloads."""
    events: list[dict[str, Any]] = []
    for block in response.text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


# --------------------------------------------------------------------------
# The endpoint
# --------------------------------------------------------------------------


def test_health_reports_ok():
    with _client(_services()) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_streams_an_answer():
    llm = FakeLlm(tokens=["Paris", " is", " the", " capital"])
    client = _client(_services(llm=llm))

    with client.stream(
        "POST",
        "/chat",
        json={"message": "What is the capital of France?", "conversation_id": "s1"},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "Paris is the capital" in body
    events = _events(type("R", (), {"text": body})())
    kinds = [event["type"] for event in events]
    assert "token" in kinds
    assert kinds[-1] == "done"


def test_chat_emits_sources_before_the_tokens():
    chunk = FakeChunk(
        chunk_id=1,
        chunk_text="Designed and implemented GenAI-powered chat assistants",
        metadata={"source": "resume.pdf", "page": 2, "section": "EXPERIENCE"},
        score=0.032,
    )
    services = _services(retriever=FakeRetriever([chunk]), llm=FakeLlm(["Answer"]))
    client = _client(services)

    with client.stream(
        "POST",
        "/chat",
        json={"message": "What did they build?", "conversation_id": "s1"},
    ) as response:
        body = "".join(response.iter_text())

    events = _events(type("R", (), {"text": body})())
    kinds = [event["type"] for event in events]
    assert kinds.index("sources") < kinds.index("token")

    sources_event = next(e for e in events if e["type"] == "sources")
    assert sources_event["sources"][0]["chunk_id"] == 1
    assert sources_event["sources"][0]["metadata"]["page"] == 2


def test_chat_rejects_a_blank_message():
    client = _client(_services())
    response = client.post("/chat", json={"message": "   ", "conversation_id": "s1"})
    assert response.status_code == 422


def test_chat_requires_a_message_and_a_conversation_id():
    client = _client(_services())
    assert client.post("/chat", json={"conversation_id": "s1"}).status_code == 422
    assert client.post("/chat", json={"message": "hi"}).status_code == 422


def test_chat_stores_the_question_and_the_answer():
    store = FakeSessionStore()
    services = _services(store=store, llm=FakeLlm(["Answer"]))
    client = _client(services)

    with client.stream(
        "POST",
        "/chat",
        json={"message": "Tell me about Tristage", "conversation_id": "s1"},
    ) as response:
        "".join(response.iter_text())

    roles = [message[1] for message in store.messages]
    contents = [message[2] for message in store.messages]
    assert roles == ["user", "assistant"]
    assert contents == ["Tell me about Tristage", "Answer"]


def test_the_second_turn_receives_the_first_turn_as_context():
    store = FakeSessionStore()
    llm = FakeLlm(["Second answer"])
    retriever = FakeRetriever()
    services = _services(retriever=retriever, llm=llm, store=store)
    client = _client(services)

    for message in ("First question", "Follow-up question"):
        with client.stream(
            "POST",
            "/chat",
            json={"message": message, "conversation_id": "s1"},
        ) as response:
            "".join(response.iter_text())

    # The second call's messages carry the first exchange.
    roles = [message["role"] for message in llm.messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert llm.messages[1]["content"] == "First question"
    assert llm.messages[3]["content"] == "Follow-up question"


def test_a_follow_up_question_is_made_self_contained_for_retrieval():
    store = FakeSessionStore()
    retriever = FakeRetriever()
    services = _services(retriever=retriever, store=store)
    client = _client(services)

    for message in ("Tell me about the TruStage project", "What was my role?"):
        with client.stream(
            "POST",
            "/chat",
            json={"message": message, "conversation_id": "s1"},
        ) as response:
            "".join(response.iter_text())

    assert len(retriever.queries) == 2
    # The follow-up carries the earlier question, so the index search can
    # find the right chunk even though "What was my role?" stands alone.
    assert "Tell me about the TruStage project" in retriever.queries[1]
    assert "What was my role?" in retriever.queries[1]


def test_separate_conversations_do_not_share_history():
    store = FakeSessionStore()
    llm = FakeLlm(["Answer"])
    services = _services(store=store, llm=llm)
    client = _client(services)

    with client.stream(
        "POST",
        "/chat",
        json={"message": "Question in session A", "conversation_id": "a"},
    ) as response:
        "".join(response.iter_text())

    with client.stream(
        "POST",
        "/chat",
        json={"message": "Question in session B", "conversation_id": "b"},
    ) as response:
        "".join(response.iter_text())

    # The second call only saw its own conversation.
    assert [message["role"] for message in llm.messages] == ["system", "user"]


# --------------------------------------------------------------------------
# Prompt building
# --------------------------------------------------------------------------


def test_format_context_numbers_the_passages():
    chunks = [
        FakeChunk(1, "First passage", {"source": "a.pdf", "page": 1}),
        FakeChunk(2, "Second passage", {"source": "a.pdf", "page": 3}),
    ]
    context = format_context(chunks)
    assert "[1]" in context
    assert "[2]" in context
    assert "a.pdf - page 1" in context


def test_format_context_handles_no_chunks():
    assert "no passages" in format_context([])


def test_build_retrieval_query_without_history_is_the_question():
    assert build_retrieval_query("What is X?", []) == "What is X?"


def test_build_retrieval_query_folds_in_earlier_questions():
    history = [
        StoredMessage(role="user", content="Tell me about Kubernetes", sources=[]),
        StoredMessage(role="assistant", content="It is a container orchestrator", sources=[]),
        StoredMessage(role="user", content="What about Docker?", sources=[]),
    ]
    query = build_retrieval_query("And who uses it?", history, lookback=2)
    assert "What about Docker?" in query
    assert "And who uses it?" in query
    # The assistant's answer is not part of the search query.
    assert "container orchestrator" not in query


def test_build_messages_puts_the_context_in_the_system_turn():
    chunks = [FakeChunk(1, "A passage about Azure", {"source": "a.pdf"})]
    messages = build_messages("What did they use?", chunks)
    assert messages[0]["role"] == "system"
    assert "A passage about Azure" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "What did they use?"}


def test_build_messages_bounds_the_history_window():
    history = []
    for index in range(6):
        history.append(StoredMessage(role="user", content=f"Q{index}", sources=[]))
        history.append(StoredMessage(role="assistant", content=f"A{index}", sources=[]))

    messages = build_messages("New question", [], history, max_history_turns=2)
    roles = [message["role"] for message in messages]
    # Two turns of history, plus the system turn and the new question.
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert messages[1]["content"] == "Q4"
    assert messages[2]["content"] == "A4"
