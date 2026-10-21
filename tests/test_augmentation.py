"""Tests for the augmentation step and the context it builds.

The augmentation step decides what the model sees: the passages the hybrid
search retrieved, the conversation so far, and - when the conversation has
grown too long to replay - a summary of its earlier part.

These tests do not need a database, an embedding model or an LLM.  The
retriever and the model are replaced with stand-ins, so the whole step -
the retrieval query, the history window, the summarization - is exercised
without a network call.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from augmentation.augmenter import Augmenter  # noqa: E402
from augmentation.config import AugmentationSettings  # noqa: E402
from augmentation.context import (  # noqa: E402
    AugmentedContext,
    build_messages,
    format_context,
    format_summary,
)
from augmentation.history import (  # noqa: E402
    StoredMessage,
    build_retrieval_query,
    estimate_tokens,
    select_window,
    split_turns,
)
from generation.citations import extract_citations  # noqa: E402
from generation.responder import AnswerEvent, Responder, TokenEvent  # noqa: E402


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
    """Returns fixed tokens, and records the messages it was given."""

    def __init__(self, tokens: Sequence[str] = ("Hello", " world")) -> None:
        self.tokens = list(tokens)
        self.messages: list[dict[str, str]] = []
        self.max_tokens: list[int | None] = []

    async def stream(
        self,
        messages: Sequence[dict[str, str]],
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        self.messages = list(messages)
        self.max_tokens.append(max_tokens)
        for token in self.tokens:
            yield token


class FakeSummaryStore:
    """An in-memory stand-in for the summary store."""

    def __init__(self) -> None:
        self.stored: dict[str, tuple[str, int]] = {}

    def latest_summary(self, conversation_id: str) -> Any | None:
        entry = self.stored.get(conversation_id)
        if entry is None:
            return None
        summary, covers_up_to_id = entry
        return _StoredSummary(summary, covers_up_to_id)

    def store_summary(
        self,
        conversation_id: str,
        summary: str,
        covers_up_to_id: int,
    ) -> int:
        self.stored[conversation_id] = (summary, covers_up_to_id)
        return len(self.stored)


class _StoredSummary:
    """The shape the augmenter expects back from the store."""

    def __init__(self, summary: str, covers_up_to_id: int) -> None:
        self.summary = summary
        self.covers_up_to_id = covers_up_to_id


def _settings(**overrides: Any) -> AugmentationSettings:
    """The settings for a test, with the defaults overridden as needed."""
    defaults: dict[str, Any] = {
        "top_k": 5,
        "history_turns": 5,
        "query_history_messages": 2,
        "max_history_tokens": 2000,
        "summary_max_tokens": 400,
    }
    defaults.update(overrides)
    return AugmentationSettings(**defaults)


# --------------------------------------------------------------------------
# The history window
# --------------------------------------------------------------------------


def _turns(count: int, start_id: int = 1) -> list[StoredMessage]:
    """A history of ``count`` user/assistant pairs, with row ids."""
    history: list[StoredMessage] = []
    message_id = start_id
    for index in range(count):
        history.append(
            StoredMessage(
                role="user",
                content=f"Q{index}",
                sources=[],
                id=message_id,
            )
        )
        message_id += 1
        history.append(
            StoredMessage(
                role="assistant",
                content=f"A{index}",
                sources=[],
                id=message_id,
            )
        )
        message_id += 1
    return history


def test_split_turns_keeps_whole_turns():
    history = _turns(3)

    old, recent = split_turns(history, turns=1)

    assert [message.content for message in recent] == ["Q2", "A2"]
    assert [message.content for message in old] == ["Q0", "A0", "Q1", "A1"]


def test_split_turns_with_no_user_message_treats_all_as_old():
    history = [StoredMessage(role="assistant", content="A", sources=[])]

    old, recent = split_turns(history, turns=3)

    assert len(old) == 1
    assert recent == []


def test_estimate_tokens_is_rough_but_never_zero():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 40) == 10


def test_select_window_keeps_the_newest_turn_when_nothing_fits():
    history = _turns(2)

    recent, old = select_window(history, turns=5, max_tokens=1)

    # The newest turn is kept even though it blows the budget - dropping it
    # would leave the model with no conversation at all.
    assert [message.content for message in recent] == ["Q1", "A1"]
    assert [message.content for message in old] == ["Q0", "A0"]


def test_select_window_spends_the_budget_on_the_newest_turns():
    history = _turns(3)

    # Each turn costs two tokens (one per message), so a budget of four
    # covers the two newest turns.
    recent, old = select_window(history, turns=5, max_tokens=4)

    assert [message.content for message in recent] == ["Q1", "A1", "Q2", "A2"]
    assert [message.content for message in old] == ["Q0", "A0"]


def test_select_window_never_splits_a_turn():
    history = _turns(3)

    # A budget of two covers exactly one turn, and it has to be a whole
    # one - an answer without its question is not a turn.
    recent, old = select_window(history, turns=5, max_tokens=2)

    assert [message.content for message in recent] == ["Q2", "A2"]
    assert [message.content for message in old] == ["Q0", "A0", "Q1", "A1"]


def test_select_window_respects_the_turn_limit():
    history = _turns(4)

    recent, old = select_window(history, turns=2, max_tokens=10000)

    assert [message.content for message in recent] == ["Q2", "A2", "Q3", "A3"]
    assert len(old) == 4


# --------------------------------------------------------------------------
# The retrieval query
# --------------------------------------------------------------------------


def test_build_retrieval_query_without_history_is_the_question():
    assert build_retrieval_query("What is X?", []) == "What is X?"


def test_build_retrieval_query_folds_in_earlier_questions():
    history = [
        StoredMessage(role="user", content="Tell me about Kubernetes", sources=[]),
        StoredMessage(
            role="assistant",
            content="It is a container orchestrator",
            sources=[],
        ),
        StoredMessage(role="user", content="What about Docker?", sources=[]),
    ]

    query = build_retrieval_query("And who uses it?", history, lookback=2)

    assert "What about Docker?" in query
    assert "And who uses it?" in query
    # The assistant's answer is not part of the search query.
    assert "container orchestrator" not in query


def test_build_retrieval_query_drops_a_repeated_question():
    history = [StoredMessage(role="user", content="What is X?", sources=[])]

    assert build_retrieval_query("What is X?", history) == "What is X?"


# --------------------------------------------------------------------------
# The context
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


def test_format_summary_is_labelled():
    assert "SUMMARY" in format_summary("The user asked about Azure.")


def test_build_messages_puts_the_context_in_the_system_turn():
    chunks = [FakeChunk(1, "A passage about Azure", {"source": "a.pdf"})]

    messages = build_messages(AugmentedContext(question="What did they use?", passages=chunks))

    assert messages[0]["role"] == "system"
    assert "A passage about Azure" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "What did they use?"}


def test_build_messages_includes_the_summary_in_the_system_turn():
    context = AugmentedContext(
        question="And what about the second one?",
        passages=[FakeChunk(1, "A passage")],
        summary="The user asked about the first project.",
    )

    messages = build_messages(context)

    assert "The user asked about the first project." in messages[0]["content"]
    # The summary is context, not a turn of its own.
    assert [message["role"] for message in messages] == ["system", "user"]


def test_build_messages_replays_the_history_it_is_given():
    context = AugmentedContext(
        question="New question",
        passages=[],
        history=_turns(2),
    )

    messages = build_messages(context)

    roles = [message["role"] for message in messages]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert messages[1]["content"] == "Q0"
    assert messages[2]["content"] == "A0"


def test_build_messages_skips_an_empty_turn():
    context = AugmentedContext(
        question="New question",
        passages=[],
        history=[StoredMessage(role="user", content="   ", sources=[])],
    )

    messages = build_messages(context)

    assert [message["role"] for message in messages] == ["system", "user"]


# --------------------------------------------------------------------------
# The augmenter
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_augmenter_retrieves_with_the_question():
    retriever = FakeRetriever([FakeChunk(1, "A passage")])
    augmenter = Augmenter(retriever=retriever, llm=FakeLlm(), settings=_settings())

    context = await augmenter.augment("What is Azure?")

    assert retriever.queries == ["What is Azure?"]
    assert len(context.passages) == 1
    assert context.question == "What is Azure?"


@pytest.mark.asyncio
async def test_the_augmenter_asks_for_only_the_top_k_passages():
    retriever = FakeRetriever([FakeChunk(index, "A passage") for index in range(10)])
    augmenter = Augmenter(
        retriever=retriever,
        llm=FakeLlm(),
        settings=_settings(top_k=3),
    )

    context = await augmenter.augment("What is Azure?")

    assert len(context.passages) == 3


@pytest.mark.asyncio
async def test_the_augmenter_folds_the_earlier_question_into_the_query():
    retriever = FakeRetriever()
    augmenter = Augmenter(retriever=retriever, llm=FakeLlm(), settings=_settings())
    history = [StoredMessage(role="user", content="Tell me about TruStage", sources=[])]

    await augmenter.augment("What was my role?", history)

    assert "Tell me about TruStage" in retriever.queries[0]
    assert "What was my role?" in retriever.queries[0]


@pytest.mark.asyncio
async def test_a_long_history_is_summarized_and_the_summary_is_used():
    llm = FakeLlm(["The user asked about Azure."])
    augmenter = Augmenter(
        retriever=FakeRetriever(),
        llm=llm,
        settings=_settings(history_turns=1, max_history_tokens=10),
    )
    history = _turns(4)

    context = await augmenter.augment("And what else?", history)

    assert context.was_summarized
    assert context.summary == "The user asked about Azure."
    # The recent turn is still replayed verbatim.
    assert [message.content for message in context.history] == ["Q3", "A3"]


@pytest.mark.asyncio
async def test_a_short_history_is_not_summarized():
    llm = FakeLlm(["Should not be called."])
    augmenter = Augmenter(retriever=FakeRetriever(), llm=llm, settings=_settings())

    context = await augmenter.augment("What is Azure?", _turns(1))

    assert not context.was_summarized
    assert context.summary is None
    assert llm.messages == []


@pytest.mark.asyncio
async def test_the_summary_is_stored_and_reused_on_the_next_turn():
    llm = FakeLlm(["The user asked about Azure."])
    store = FakeSummaryStore()
    augmenter = Augmenter(
        retriever=FakeRetriever(),
        llm=llm,
        settings=_settings(history_turns=1, max_history_tokens=10),
        summary_store=store,
    )
    history = _turns(4)

    context = await augmenter.augment("And what else?", history, conversation_id="c1")

    assert context.summary == "The user asked about Azure."
    assert store.stored["c1"][0] == "The user asked about Azure."
    # The summary reaches up to the last message it covers.
    assert store.stored["c1"][1] == history[5].id


@pytest.mark.asyncio
async def test_a_stored_summary_is_not_summarized_again():
    store = FakeSummaryStore()
    store.stored["c1"] = ("An earlier summary.", 2)
    llm = FakeLlm(["A new summary."])
    augmenter = Augmenter(
        retriever=FakeRetriever(),
        llm=llm,
        settings=_settings(history_turns=1, max_history_tokens=10),
        summary_store=store,
    )
    # The messages up to id 2 are already covered by the stored summary.
    history = _turns(4)

    context = await augmenter.augment("And what else?", history, conversation_id="c1")

    # The earlier summary is kept and the new one is added to it.
    assert context.summary.startswith("An earlier summary.")
    assert "A new summary." in context.summary


# --------------------------------------------------------------------------
# The generation step
# --------------------------------------------------------------------------


def test_citations_are_resolved_to_the_passages_they_point_at():
    passages = [
        FakeChunk(1, "First passage", {"source": "a.pdf", "page": 1}),
        FakeChunk(2, "Second passage", {"source": "a.pdf", "page": 2}),
    ]

    citations = extract_citations("The first thing [1] and the second [2].", passages)

    assert [citation.marker for citation in citations] == [1, 2]
    assert citations[0].chunk_id == 1
    assert citations[0].metadata["page"] == 1
    assert citations[1].chunk_id == 2


def test_a_passage_is_cited_once_however_many_times_it_is_marked():
    passages = [FakeChunk(1, "First passage")]

    citations = extract_citations("It matters [1]. Really [1].", passages)

    assert len(citations) == 1


def test_a_marker_outside_the_passages_is_dropped():
    passages = [FakeChunk(1, "First passage")]

    citations = extract_citations("An invented source [7].", passages)

    assert citations == []


def test_the_fullwidth_brackets_are_understood():
    passages = [FakeChunk(1, "First passage")]

    citations = extract_citations("A citation 【1】.", passages)

    assert [citation.marker for citation in citations] == [1]


def test_an_answer_with_no_markers_has_no_citations():
    passages = [FakeChunk(1, "First passage")]

    assert extract_citations("No markers here.", passages) == []


@pytest.mark.asyncio
async def test_the_responder_streams_the_tokens_then_the_answer():
    llm = FakeLlm(["They built ", "a RAG agent [1]."])
    responder = Responder(llm)
    passages = [FakeChunk(1, "Built a RAG agent", {"source": "a.pdf"})]

    events = [event async for event in responder.respond([], passages)]

    tokens = [event.token for event in events if isinstance(event, TokenEvent)]
    assert tokens == ["They built ", "a RAG agent [1]."]

    final = events[-1]
    assert isinstance(final, AnswerEvent)
    assert final.answer == "They built a RAG agent [1]."
    assert [citation.marker for citation in final.citations] == [1]
    assert final.citations[0].chunk_id == 1
