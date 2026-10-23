"""Tests for the chat API layer.

These tests do not need a database, an embedding model, an LLM or a Redis
server.  The retriever, the LLM, the session store and the memory are
replaced with stand-ins, so the whole request path - routing, validation,
the SSE encoding, the conversation memory and the answer cache - is
exercised without a network call.
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
from augmentation.history import StoredMessage, build_retrieval_query  # noqa: E402
from generation.responder import Responder  # noqa: E402
from memory.cache import (  # noqa: E402
    CachedAnswer,
    build_cache_key,
    build_question_key,
)
from memory.faq import FaqCandidate, FaqStats  # noqa: E402


# --------------------------------------------------------------------------
# Stand-ins
# --------------------------------------------------------------------------


class FakeChunk:
    """A retrieved chunk, shaped like retrieval.retriever.RetrievedChunk."""

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
        self.calls = 0

    async def stream(
        self,
        messages: Sequence[dict[str, str]],
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        self.messages = list(messages)
        self.calls += 1
        for token in self.tokens:
            yield token


class FakeSessionStore:
    """An in-memory stand-in for augmentation.session_store.SessionStore."""

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


class FakeCache:
    """An in-memory stand-in for memory.cache.RedisAnswerCache."""

    def __init__(self, default_ttl: int = 3600) -> None:
        self.default_ttl = default_ttl
        self.store: dict[str, CachedAnswer] = {}
        self.gets: list[str] = []
        self.sets: list[tuple[str, int]] = []
        self.touches: list[tuple[str, int]] = []

    async def get(self, key: str) -> CachedAnswer | None:
        self.gets.append(key)
        return self.store.get(key)

    async def set(
        self,
        key: str,
        answer: CachedAnswer,
        ttl_seconds: int | None = None,
    ) -> None:
        # The real cache falls back to the time to live in its settings.
        ttl = self.default_ttl if ttl_seconds is None else ttl_seconds
        self.sets.append((key, ttl))
        self.store[key] = answer

    async def touch(self, key: str, ttl_seconds: int) -> None:
        self.touches.append((key, ttl_seconds))

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class FakeMatcher:
    """A stand-in for the FAQ matcher.

    ``matches`` says which stored question a new one means the same as.
    A question that is not in it is not a match.
    """

    def __init__(self) -> None:
        self.matches: dict[str, FaqCandidate] = {}
        self.remembered: list[tuple[str, str, str]] = []
        self.remembered_generations: list[int] = []
        self.searches: list[str] = []
        self.searched_generations: list[int] = []
        self.stored_generations: dict[str, int] = {}

    def match(
        self,
        question: str,
        threshold: float,
        generation: int = 0,
    ) -> FaqCandidate | None:
        self.searches.append(question)
        self.searched_generations.append(generation)
        candidate = self.matches.get(question)
        if candidate is None or candidate.similarity < threshold:
            return None
        if self.stored_generations.get(candidate.question, generation) != generation:
            return None
        return candidate

    def remember(
        self,
        question: str,
        question_key: str,
        cache_key: str,
        generation: int = 0,
    ) -> None:
        self.remembered.append((question, question_key, cache_key))
        self.stored_generations[question] = generation
        self.remembered_generations.append(generation)

    def ensure_schema(self) -> None:
        """No schema to create."""

    def prune(self) -> int:
        return 0

    def count(self) -> int:
        return 0


class FakeMetrics:
    """A stand-in for the FAQ registry."""

    def __init__(self) -> None:
        self.asks: list[dict[str, Any]] = []
        self.reported: list[FaqStats] = []

    async def record_ask(
        self,
        question: str,
        question_key: str,
        cost_ms: int = 0,
        hit: bool = False,
    ) -> FaqStats:
        self.asks.append(
            {
                "question": question,
                "question_key": question_key,
                "cost_ms": cost_ms,
                "hit": hit,
            }
        )
        return FaqStats(question=question)

    async def stats(self, question_key: str) -> FaqStats | None:
        return None

    async def faqs(self, limit: int = 50) -> list[FaqStats]:
        return self.reported[:limit]

    async def is_faq(self, question_key: str) -> bool:
        return False

    async def ttl_for(self, question_key: str) -> int:
        return 3600

    async def close(self) -> None:
        return None


class FakeIndexState:
    """A stand-in for the version of the corpus.

    ``generation`` is the version of the corpus the answers are built
    from.  Changing it is what a rebuild of the corpus does.
    """

    def __init__(self, generation: int = 1) -> None:
        self.current = generation

    def generation(self) -> int:
        return self.current


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _chunk(chunk_id: int = 517) -> FakeChunk:
    return FakeChunk(
        chunk_id=chunk_id,
        chunk_text="Designed and implemented GenAI chat assistants.",
        metadata={"source": "resume.pdf", "page": 2, "project": "TruStage"},
        score=0.8,
    )


def _services(
    retriever: FakeRetriever | None = None,
    llm: FakeLlm | None = None,
    store: FakeSessionStore | None = None,
    cache: FakeCache | None = None,
    matcher: FakeMatcher | None = None,
    metrics: FakeMetrics | None = None,
    index_state: FakeIndexState | None = None,
) -> Services:
    """Build a Services object backed entirely by stand-ins."""
    retriever = retriever or FakeRetriever([_chunk()])
    llm = llm or FakeLlm()
    store = store or FakeSessionStore()
    cache = cache or FakeCache()
    matcher = matcher or FakeMatcher()
    metrics = metrics or FakeMetrics()
    settings = AugmentationSettings(
        top_k=5,
        history_turns=5,
        query_history_messages=2,
        max_history_tokens=2000,
        summary_max_tokens=400,
    )
    augmenter = Augmenter(
        retriever=retriever,
        llm=llm,
        settings=settings,
        summary_store=store,
    )
    chat_service = ChatService(
        augmenter=augmenter,
        responder=Responder(llm),
        session_store=store,
        answer_cache=cache,
        settings=settings,
        faq_matcher=matcher,
        faq_metrics=metrics,
        index_state=index_state,
    )
    return Services(
        chat_service=chat_service,
        session_store=store,  # type: ignore[arg-type]
        database=None,  # type: ignore[arg-type]
        embedder=None,  # type: ignore[arg-type]
        retriever=retriever,  # type: ignore[arg-type]
        llm=llm,
        cache=cache,
        faq_matcher=matcher,
        faq_metrics=metrics,
        index_state=index_state,  # type: ignore[arg-type]
    )


def _client(services: Services) -> TestClient:
    """A test client for an app that does not run the startup lifespan."""
    app = create_app(services)
    # The lifespan would try to create tables, load the embedding model and
    # reach Redis, which the stand-ins do not support.
    return TestClient(app, raise_server_exceptions=True)


def _events(response: Any) -> list[dict[str, Any]]:
    """Parse an SSE response body into a list of event payloads."""
    events: list[dict[str, Any]] = []
    for block in response.text.split(chr(10) + chr(10)):
        block = block.strip()
        if not block:
            continue
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


def _ask(client: TestClient, message: str, conversation_id: str = "s1") -> Any:
    """Ask a question and return the parsed events."""
    response = client.post(
        "/chat",
        json={"message": message, "conversation_id": conversation_id},
    )
    assert response.status_code == 200
    return _events(response)


# --------------------------------------------------------------------------
# The endpoint
# --------------------------------------------------------------------------


def test_chat_streams_the_sources_then_the_answer() -> None:
    """The sources come first, then the answer, then the citations."""
    services = _services()
    client = _client(services)

    events = _ask(client, "What projects did Rose work on?")

    assert [event["type"] for event in events] == [
        "sources",
        "token",
        "token",
        "citations",
        "done",
    ]
    assert events[0]["sources"][0]["chunk_id"] == 517
    assert events[0]["sources"][0]["metadata"]["project"] == "TruStage"
    assert events[-1]["token"] == "Hello world"


def test_chat_reports_the_citations_the_answer_used() -> None:
    """The citations event lists the passages the answer referred to."""
    llm = FakeLlm(["Rose worked on Trustage [1] and Danske [2]."])
    retriever = FakeRetriever([_chunk(517), _chunk(588)])
    services = _services(retriever=retriever, llm=llm)
    client = _client(services)

    events = _ask(client, "What projects did Rose work on?")

    citations = next(event for event in events if event["type"] == "citations")
    assert [citation["marker"] for citation in citations["citations"]] == [1, 2]
    assert citations["citations"][0]["chunk_id"] == 517
    assert citations["citations"][1]["chunk_id"] == 588


def test_chat_keeps_the_conversation() -> None:
    """The second turn sees the first turn, and the earlier question."""
    services = _services()
    client = _client(services)

    _ask(client, "What projects did Rose work on?", "s1")
    _ask(client, "And what did she do there?", "s1")

    # The retriever is given the earlier question, so the follow-up
    # retrieves on a self-contained query.
    assert services.retriever.queries[1] == (
        "What projects did Rose work on? And what did she do there?"
    )
    # The model is given the earlier turn as history.
    history = services.llm.messages[1:3]
    assert [message["role"] for message in history] == ["user", "assistant"]
    assert history[0]["content"] == "What projects did Rose work on?"


def test_chat_keeps_conversations_apart() -> None:
    """Two conversation ids do not see each other."""
    services = _services()
    client = _client(services)

    _ask(client, "First question?", "s1")
    _ask(client, "Second question?", "s2")

    assert services.retriever.queries[1] == "Second question?"


def test_chat_rejects_a_blank_message() -> None:
    """A blank message is a bad request."""
    services = _services()
    client = _client(services)

    response = client.post(
        "/chat", json={"message": "   ", "conversation_id": "s1"}
    )

    assert response.status_code == 422


def test_health_does_not_need_the_database() -> None:
    """The liveness check is cheap."""
    services = _services()
    client = _client(services)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --------------------------------------------------------------------------
# The memory
# --------------------------------------------------------------------------


def test_chat_stores_the_answer_in_the_memory() -> None:
    """A fresh answer is written to the memory, with the time to live."""
    services = _services()
    client = _client(services)

    _ask(client, "What projects did Rose work on?")

    assert len(services.cache.sets) == 1
    key, ttl = services.cache.sets[0]
    assert ttl == 3600
    stored = services.cache.store[key]
    assert stored.answer == "Hello world"
    assert stored.sources[0]["chunk_id"] == 517


def test_chat_answers_from_the_memory_without_calling_the_model() -> None:
    """A question that was answered before does not reach the LLM."""
    llm = FakeLlm(["A brand new answer"])
    services = _services(llm=llm)
    client = _client(services)

    question = "What projects did Rose work on?"
    key = build_cache_key(question, build_retrieval_query(question, [], lookback=2))
    services.cache.store[key] = CachedAnswer(
        answer="Rose worked on Trustage [1].",
        sources=[
            {
                "chunk_id": 517,
                "text": "Designed and implemented GenAI chat assistants.",
                "metadata": {"project": "TruStage"},
                "score": 0.8,
            }
        ],
        citations=[
            {
                "marker": 1,
                "chunk_id": 517,
                "text": "Designed and implemented GenAI chat assistants.",
                "metadata": {"project": "TruStage"},
                "score": 0.8,
            }
        ],
    )

    events = _ask(client, question)

    assert llm.calls == 0
    assert services.retriever.queries == []
    assert [event["type"] for event in events] == [
        "sources",
        "token",
        "citations",
        "done",
    ]
    assert events[0]["cache_hit"] is True
    assert events[1]["token"] == "Rose worked on Trustage [1]."
    citations = next(event for event in events if event["type"] == "citations")
    assert citations["citations"][0]["marker"] == 1
    assert citations["citations"][0]["chunk_id"] == 517
    assert events[-1]["token"] == "Rose worked on Trustage [1]."


def test_a_cached_answer_is_kept_in_the_conversation() -> None:
    """The exchange reads the same way in the history as any other turn."""
    services = _services()
    client = _client(services)

    question = "What projects did Rose work on?"
    key = build_cache_key(question, build_retrieval_query(question, [], lookback=2))
    services.cache.store[key] = CachedAnswer(answer="Rose worked on Trustage.")

    _ask(client, question)

    roles = [message[1] for message in services.session_store.messages]
    assert roles == ["user", "assistant"]
    assert services.session_store.messages[1][2] == "Rose worked on Trustage."


def test_the_memory_is_asked_before_the_model() -> None:
    """The memory is consulted first, and a miss falls through to the model."""
    services = _services()
    client = _client(services)

    _ask(client, "What projects did Rose work on?")

    assert len(services.cache.gets) == 1
    assert services.cache.gets[0] == services.cache.sets[0][0]


def test_a_follow_up_question_is_not_a_cached_question() -> None:
    """A question that depends on the conversation gets its own key."""
    services = _services()
    client = _client(services)

    _ask(client, "What projects did Rose work on?", "s1")
    # The same question in another conversation is answered from the memory.
    _ask(client, "What projects did Rose work on?", "s2")

    assert len(services.cache.sets) == 1
    assert services.llm.calls == 1

    _ask(client, "And what did she do there?", "s1")

    # A follow-up is a different question, because the earlier question is
    # folded into the query it is answered under.
    assert len(services.cache.sets) == 2
    assert services.cache.sets[1][0] != services.cache.sets[0][0]


# --------------------------------------------------------------------------
# The frequently asked questions
# --------------------------------------------------------------------------


def test_a_question_in_other_words_is_answered_from_the_memory() -> None:
    """A question asked in other words does not reach the model."""
    llm = FakeLlm(["A brand new answer"])
    matcher = FakeMatcher()
    services = _services(llm=llm, matcher=matcher)
    client = _client(services)

    question = "What projects did Rose work on?"
    key = build_cache_key(question, build_retrieval_query(question, [], lookback=2))
    services.cache.store[key] = CachedAnswer(answer="Rose worked on Trustage.")

    # The same question, worded differently.
    matcher.matches["Which projects was Rose involved in?"] = FaqCandidate(
        question=question,
        question_key=build_question_key(question),
        cache_key=key,
        similarity=0.93,
    )

    events = _ask(client, "Which projects was Rose involved in?", "s2")

    assert llm.calls == 0
    assert services.retriever.queries == []
    assert events[0]["cache_hit"] is True
    assert events[0]["cache_match"] == "semantic"
    assert events[1]["token"] == "Rose worked on Trustage."


def test_a_question_nothing_matches_is_answered_by_the_model() -> None:
    """A question the memory does not know is answered as usual."""
    llm = FakeLlm(["A brand new answer"])
    services = _services(llm=llm)
    client = _client(services)

    events = _ask(client, "Which projects was Rose involved in?", "s2")

    assert llm.calls == 1
    assert events[0]["cache_hit"] is False
    assert events[0].get("cache_match") is None


def test_a_follow_up_is_not_looked_up_by_meaning() -> None:
    """A follow-up carries the conversation, so it is not looked up."""
    matcher = FakeMatcher()
    services = _services(matcher=matcher)
    client = _client(services)

    _ask(client, "What projects did Rose work on?", "s1")
    _ask(client, "And what did she do there?", "s1")

    # The first question stands on its own and is looked up; the follow-up
    # is not, because its answer depends on what came before it.
    assert matcher.searches == ["What projects did Rose work on?"]


def test_a_question_that_was_answered_is_recorded() -> None:
    """The question is recorded, and the hit with it."""
    metrics = FakeMetrics()
    services = _services(metrics=metrics)
    client = _client(services)

    question = "What projects did Rose work on?"
    _ask(client, question, "s1")

    assert len(metrics.asks) == 1
    assert metrics.asks[0]["question"] == question
    assert metrics.asks[0]["question_key"] == build_question_key(question)
    assert metrics.asks[0]["hit"] is False

    # The second time, the memory answers it and the hit is recorded.
    _ask(client, question, "s2")

    assert len(metrics.asks) == 2
    assert metrics.asks[1]["hit"] is True


def test_a_question_is_put_in_the_index_so_the_next_one_can_be_matched() -> None:
    """A question that stands on its own is indexed for the next one."""
    matcher = FakeMatcher()
    services = _services(matcher=matcher)
    client = _client(services)

    question = "What projects did Rose work on?"
    _ask(client, question, "s1")

    assert len(matcher.remembered) == 1
    remembered_question, question_key, cache_key = matcher.remembered[0]
    assert remembered_question == question
    assert question_key == build_question_key(question)
    assert cache_key == services.cache.sets[0][0]


def test_a_follow_up_is_not_put_in_the_index() -> None:
    """A follow-up is not a question on its own, so it is not indexed."""
    matcher = FakeMatcher()
    services = _services(matcher=matcher)
    client = _client(services)

    _ask(client, "What projects did Rose work on?", "s1")
    _ask(client, "And what did she do there?", "s1")

    assert len(matcher.remembered) == 1


def test_the_frequently_asked_questions_are_listed() -> None:
    """The endpoint reports the questions that are asked often."""
    metrics = FakeMetrics()
    metrics.reported = [
        FaqStats(
            question="What projects did Rose work on?",
            asks=5,
            hits=4,
            saved_ms=9000,
            cost_ms=3000,
            last_asked_at=1000.0,
        )
    ]
    services = _services(metrics=metrics)
    client = _client(services)

    response = client.get("/memory/faqs")

    assert response.status_code == 200
    body = response.json()
    assert body["faqs"][0]["question"] == "What projects did Rose work on?"
    assert body["faqs"][0]["asks"] == 5
    assert body["faqs"][0]["hits"] == 4
    assert body["faqs"][0]["misses"] == 1
    assert body["faqs"][0]["hit_rate"] == 0.8
    assert body["faqs"][0]["saved_ms"] == 9000


def test_a_rebuilt_corpus_invalidates_the_answer() -> None:
    """An answer built from the corpus as it was is not served again."""
    state = FakeIndexState(1)
    services = _services(index_state=state)
    client = _client(services)

    _ask(client, "What projects did Rose work on?", "s1")
    assert services.llm.calls == 1

    # The corpus is rebuilt, so the version moves on.
    state.current = 2
    events = _ask(client, "What projects did Rose work on?", "s2")

    assert events[0]["cache_hit"] is False
    assert services.llm.calls == 2


def test_the_answer_is_stored_under_the_key_of_its_version() -> None:
    """The version of the corpus is part of the key the answer is under."""
    state = FakeIndexState(7)
    services = _services(index_state=state)
    client = _client(services)

    question = "What projects did Rose work on?"
    _ask(client, question, "s1")

    query = build_retrieval_query(question, [], lookback=2)
    assert build_cache_key(question, query, 7) in services.cache.store


def test_the_index_holds_the_questions_of_the_current_corpus() -> None:
    """The version of the corpus goes into the index with the question."""
    state = FakeIndexState(1)
    matcher = FakeMatcher()
    services = _services(matcher=matcher, index_state=state)
    client = _client(services)

    _ask(client, "What projects did Rose work on?", "s1")

    assert matcher.remembered_generations == [1]
    assert matcher.searched_generations == [1]


def test_a_question_of_an_older_corpus_is_not_matched() -> None:
    """A question asked against the old corpus is asked afresh."""
    state = FakeIndexState(1)
    matcher = FakeMatcher()
    services = _services(matcher=matcher, index_state=state)
    client = _client(services)

    question = "What projects did Rose work on?"
    _ask(client, question, "s1")
    old_key = matcher.remembered[0][2]

    # The corpus is rebuilt, and the question is asked in other words.
    state.current = 2
    matcher.matches["Which projects was Rose involved in?"] = FaqCandidate(
        question=question,
        question_key=build_question_key(question),
        cache_key=old_key,
        similarity=0.95,
    )
    events = _ask(client, "Which projects was Rose involved in?", "s2")

    assert events[0]["cache_hit"] is False
    assert services.llm.calls == 2
    # and the index is given the key of the new version.
    assert matcher.remembered[-1][2] != old_key
    assert matcher.remembered_generations[-1] == 2
