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
from augmentation.augmenter import Augmenter  # noqa: E402
from augmentation.chat_service import ChatService  # noqa: E402
from augmentation.config import AugmentationSettings  # noqa: E402
from augmentation.history import StoredMessage  # noqa: E402
from generation.responder import Responder  # noqa: E402


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

    async def stream(
        self,
        messages: Sequence[dict[str, str]],
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        self.messages = list(messages)
        for token in self.tokens:
            yield token


class FakeSessionStore:
    """An in-memory stand-in for ``augmentation.session_store.SessionStore``."""

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

    def latest_summary(self, conversation_id: str) -> None:
        """No summaries are stored."""
        return None

    def store_summary(
        self,
        conversation_id: str,
        summary: str,
        covers_up_to_id: int,
    ) -> int:
        """Summaries are not stored."""
        return 0


def _services(
    retriever: FakeRetriever | None = None,
    llm: FakeLlm | None = None,
    store: FakeSessionStore | None = None,
) -> Services:
    """Build a Services object backed entirely by stand-ins."""
    retriever = retriever or FakeRetriever()
    llm = llm or FakeLlm()
    store = store or FakeSessionStore()
    augmenter = Augmenter(
        retriever=retriever,
        llm=llm,
        settings=AugmentationSettings(
            top_k=5,
            history_turns=5,
            query_history_messages=2,
            max_history_tokens=2000,
            summary_max_tokens=400,
        ),
        summary_store=store,
    )
    chat_service = ChatService(
        augmenter=augmenter,
        responder=Responder(llm),
        session_store=store,
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
