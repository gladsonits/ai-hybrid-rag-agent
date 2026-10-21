"""The chat turn: retrieve, prompt, stream, remember.

This is the piece that turns a question into an answer and keeps the
conversation going.  The order matters:

1. Read the conversation's history, so the question can be understood in
   context.
2. Build a search query that stands on its own, by folding the earlier
   questions into it - retrieval does not see the conversation.
3. Pull the relevant chunks out of the index.
4. Ask the model for an answer, streaming it out token by token.
5. Store the question and the answer, so the next turn can read them.

The two dependencies - the retriever and the LLM - are injected, so the
service can be tested against stand-ins without a database or an API key.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

from api.config import ChatSettings
from api.llm import LlmClient
from api.prompt import build_messages, build_retrieval_query
from api.session_store import SessionStore, StoredMessage


class Retriever(Protocol):
    """The part of ``HybridRetriever`` the chat service needs."""

    def search(
        self,
        query: str,
        limit: int = 10,
        candidates: int = 50,
        filters: dict[str, Any] | None = None,
    ) -> list[Any]: ...


@dataclass(frozen=True, slots=True)
class SourcesEvent:
    """The passages a turn was answered from, known before the answer is."""

    sources: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class TokenEvent:
    """One piece of the answer."""

    token: str


class ChatService:
    """Answers a question using the index, and remembers the conversation."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LlmClient,
        session_store: SessionStore,
        settings: ChatSettings | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.session_store = session_store
        self.settings = settings or ChatSettings.from_env()

    async def stream_answer(
        self,
        conversation_id: str,
        question: str,
    ) -> AsyncIterator[SourcesEvent | TokenEvent]:
        """Answer ``question``, yielding the sources and then the answer.

        The sources come first, because they are known before the model
        starts writing - a UI can show what the answer will be based on
        while the answer is still arriving.

        The question and the finished answer are both stored, so the next
        turn in this conversation can read them.
        """
        history = self.session_store.history(conversation_id)

        # The question is stored before the answer is generated, so an
        # interrupted turn still leaves the user's question in the history.
        self.session_store.append_message(conversation_id, "user", question)

        chunks = await self._retrieve(question, history)
        sources = self._sources(chunks)
        yield SourcesEvent(sources=sources)

        messages = build_messages(
            question,
            chunks,
            history,
            max_history_turns=self.settings.history_turns,
        )

        answer_parts: list[str] = []
        async for token in self.llm.stream(messages):
            answer_parts.append(token)
            yield TokenEvent(token=token)

        self.session_store.append_message(
            conversation_id,
            "assistant",
            "".join(answer_parts),
            sources=sources,
        )

    async def _retrieve(
        self,
        question: str,
        history: list[StoredMessage],
    ) -> list[Any]:
        """Pull the chunks that answer the question.

        The search runs in a worker thread, because the retriever is
        synchronous and does blocking work - a database round trip and an
        embedding model forward pass.  Running it on the event loop would
        stall every other request.
        """
        query = build_retrieval_query(
            question,
            history,
            lookback=self.settings.query_history_messages,
        )
        return await self._run_retrieval(query)

    async def _run_retrieval(self, query: str) -> list[Any]:
        """Call the synchronous retriever off the event loop."""
        import anyio.to_thread

        return await anyio.to_thread.run_sync(
            lambda: self.retriever.search(query, limit=self.settings.top_k)
        )

    @staticmethod
    def _sources(chunks: list[Any]) -> list[dict[str, Any]]:
        """Turn retrieved chunks into the citation records stored with an answer."""
        sources: list[dict[str, Any]] = []
        for chunk in chunks:
            sources.append(
                {
                    "chunk_id": int(getattr(chunk, "chunk_id", 0)),
                    "text": " ".join(getattr(chunk, "chunk_text", "").split()),
                    "metadata": getattr(chunk, "metadata", {}) or {},
                    "score": getattr(chunk, "score", None),
                }
            )
        return sources
