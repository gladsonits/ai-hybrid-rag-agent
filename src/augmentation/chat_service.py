"""The chat turn: the business logic of answering a question.

This is the step that turns a question into an answer and keeps the
conversation going.  The order matters:

1. Read the conversation, so the question can be understood in context.
2. **Ask the memory.**  The memory is asked twice, in two ways.  First
   the exact question - the same words in, the same key.  If that misses,
   the *meaning* of the question, so a question asked in other words is
   recognised as one that has been answered before.  Either way the
   answer comes straight back with the sources and the citations it was
   built from, and neither the retrieval nor the model runs.
3. **Augment** the context - retrieve the relevant passages, choose which
   turns to replay, and summarize the ones that did not fit.
4. **Generate** the answer from that context, streaming it out token by
   token, and resolve the citations it used.
5. Store the question and the answer, so the next turn can read them,
   and store the answer in the memory, so the next time this question is
   asked it does not have to be answered again.  Record how the question
   was answered, because that is what decides whether it counts as a
   frequently asked question.

The retrieval, the history windowing and the summarization all live in
augmentation.augmenter; the call to the model and the citation handling
live in generation; the memory lives in memory.  This module is the turn
that uses them.

The dependencies - the retriever, the LLM, the session store, the
memory, the FAQ index and the FAQ statistics - are injected, so the
service can be tested against stand-ins without a database, an API key
or a Redis server.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from augmentation.augmenter import Augmenter
from augmentation.config import AugmentationSettings
from augmentation.context import build_messages
from augmentation.history import StoredMessage, build_retrieval_query
from augmentation.session_store import SessionStore
from generation.citations import (
    Citation,
    build_source_records,
    citation_records,
    citations_from_records,
)
from generation.responder import AnswerEvent, Responder, TokenEvent
from memory.cache import (
    AnswerCache,
    CachedAnswer,
    IndexGeneration,
    NullAnswerCache,
    build_cache_key,
    build_question_key,
)
from memory.config import FaqSettings
from memory.faq import FaqCandidate, FaqMatcher, FaqMetrics, NullFaqRegistry

logger = logging.getLogger(__name__)

#: The ways the memory can answer a question.
MATCH_EXACT = "exact"
MATCH_SEMANTIC = "semantic"


@dataclass(frozen=True, slots=True)
class SourcesEvent:
    """The passages a turn was answered from, known before the answer is."""

    sources: list[dict[str, Any]]
    #: Whether the turn was answered from the memory rather than by the model.
    cached: bool = False
    #: How the memory answered it - "exact" for the same words,
    #: "semantic" for the same question in other words.
    match: str | None = None


@dataclass(frozen=True, slots=True)
class ChatTokenEvent:
    """One piece of the answer."""

    token: str


@dataclass(frozen=True, slots=True)
class AnswerCompleteEvent:
    """The finished answer, with the passages it cited."""

    answer: str
    citations: list[Citation]
    #: Whether the turn was answered from the memory rather than by the model.
    cached: bool = False
    #: How the memory answered it - "exact" for the same words,
    #: "semantic" for the same question in other words.
    match: str | None = None


class ChatService:
    """Answers a question using the index, and remembers the conversation."""

    def __init__(
        self,
        augmenter: Augmenter,
        responder: Responder,
        session_store: SessionStore,
        answer_cache: AnswerCache | None = None,
        settings: AugmentationSettings | None = None,
        faq_matcher: FaqMatcher | None = None,
        faq_metrics: FaqMetrics | None = None,
        faq_settings: FaqSettings | None = None,
        index_state: IndexGeneration | None = None,
    ) -> None:
        self.augmenter = augmenter
        self.responder = responder
        self.session_store = session_store
        self.answer_cache = answer_cache or NullAnswerCache()
        self.settings = settings or augmenter.settings
        self.faq_matcher = faq_matcher
        self.faq_metrics = faq_metrics or NullFaqRegistry()
        self.faq_settings = faq_settings or FaqSettings.from_env()
        self.index_state = index_state

    async def stream_answer(
        self,
        conversation_id: str,
        question: str,
    ) -> AsyncIterator[SourcesEvent | ChatTokenEvent | AnswerCompleteEvent]:
        """Answer the question, yielding the sources and then the answer.

        The sources come first, because they are known before the model
        starts writing - a UI can show what the answer will be based on
        while the answer is still arriving.

        A question that has been answered before is answered from the
        memory instead, and comes back with the same sources and the same
        citations it had the first time.  "Before" means either the same
        question, or a question that means the same thing.

        The question and the finished answer are both stored, so the next
        turn in this conversation can read them.
        """
        history = self.session_store.history(conversation_id)

        # The question is stored before the answer is generated, so an
        # interrupted turn still leaves the question in the history.
        self.session_store.append_message(conversation_id, "user", question)

        query = build_retrieval_query(
            question,
            history,
            lookback=self.settings.query_history_messages,
        )
        generation = await self._generation()
        cache_key = build_cache_key(question, query, generation)
        question_key = build_question_key(question)

        # The memory, asked with the exact question.
        cached = await self.answer_cache.get(cache_key)
        if cached is not None:
            await self._record(question, question_key, hit=True)
            await self._refresh(cache_key, question_key)
            async for event in self._replay_cached(
                conversation_id, cached, MATCH_EXACT
            ):
                yield event
            return

        # The memory, asked with the meaning of the question.
        candidate = await self._semantic_candidate(question, query, generation)
        if candidate is not None:
            semantic = await self.answer_cache.get(candidate.cache_key)
            if semantic is not None:
                await self._record(
                    candidate.question, candidate.question_key, hit=True
                )
                await self._refresh(candidate.cache_key, candidate.question_key)
                async for event in self._replay_cached(
                    conversation_id, semantic, MATCH_SEMANTIC
                ):
                    yield event
                return

        context = await self.augmenter.augment(
            question, history, conversation_id=conversation_id
        )
        sources = build_source_records(context.passages)
        yield SourcesEvent(sources=sources)

        messages = build_messages(context)

        answer_parts: list[str] = []
        citations: list[Citation] = []
        started = time.perf_counter()
        async for event in self.responder.respond(messages, context.passages):
            if isinstance(event, TokenEvent):
                answer_parts.append(event.token)
                yield ChatTokenEvent(token=event.token)
            elif isinstance(event, AnswerEvent):
                answer_parts = [event.answer]
                citations = event.citations
                yield AnswerCompleteEvent(
                    answer=event.answer,
                    citations=event.citations,
                )

        answer = "".join(answer_parts)
        cost_ms = int((time.perf_counter() - started) * 1000)

        self.session_store.append_message(
            conversation_id,
            "assistant",
            answer,
            sources=sources,
        )

        if answer.strip():
            await self._store(
                question=question,
                question_key=question_key,
                cache_key=cache_key,
                query=query,
                answer=answer,
                sources=sources,
                citations=citations,
                cost_ms=cost_ms,
                generation=generation,
            )
        else:
            await self._record(question, question_key, cost_ms=cost_ms)

    # -- the memory --------------------------------------------------------

    async def _generation(self) -> int:
        """The version of the corpus this turn is answered from.

        The version is part of the key an answer is stored under, so a
        corpus that has been rebuilt since an answer was given is never
        served again: the key is a different one, and the memory misses.
        """
        if self.index_state is None:
            return 0

        try:
            import anyio.to_thread

            return int(
                await anyio.to_thread.run_sync(self.index_state.generation)
            )
        except Exception:
            logger.warning(
                "the version of the corpus could not be read", exc_info=True
            )
            return 0

    async def _semantic_candidate(
        self,
        question: str,
        query: str,
        generation: int,
    ) -> FaqCandidate | None:
        """The stored question this one means the same as, if there is one.

        Only a question that stands on its own is looked up.  A follow-up
        is answered from the conversation it was asked in, and a question
        in other words carries none of that context, so the answer to the
        stored question would not be the answer to this one.

        Only the questions of this version of the corpus are looked at:
        an answer built from the corpus as it was is not the answer to
        this question any more.
        """
        if self.faq_matcher is None:
            return None
        if query.strip() != question.strip():
            return None

        try:
            import anyio.to_thread

            return await anyio.to_thread.run_sync(
                lambda: self.faq_matcher.match(
                    question,
                    self.faq_settings.similarity_threshold,
                    generation,
                )
            )
        except Exception:
            logger.warning(
                "the frequently asked questions could not be searched",
                exc_info=True,
            )
            return None

    async def _store(
        self,
        question: str,
        question_key: str,
        cache_key: str,
        query: str,
        answer: str,
        sources: list[dict[str, Any]],
        citations: list[Citation],
        cost_ms: int,
        generation: int,
    ) -> None:
        """Keep the answer in the memory, and record what the question cost.

        The time to live is the one the question deserves: an hour for a
        question asked once, longer once it counts as frequently asked.
        """
        ttl = await self.faq_metrics.ttl_for(question_key)
        await self.answer_cache.set(
            cache_key,
            CachedAnswer(
                answer=answer,
                sources=sources,
                citations=citation_records(citations),
            ),
            ttl_seconds=ttl,
        )
        await self._record(question, question_key, cost_ms=cost_ms)
        await self._remember(question, question_key, cache_key, query, generation)

    async def _remember(
        self,
        question: str,
        question_key: str,
        cache_key: str,
        query: str,
        generation: int,
    ) -> None:
        """Put the question in the index, so the next one can be matched.

        Only a question that stands on its own goes in the index.  A
        follow-up is answered from the conversation it was asked in, and
        another question in other words is not that question.

        The version of the corpus goes with it, so a question that was
        asked against an older corpus is not matched against a question
        asked against this one.
        """
        if self.faq_matcher is None:
            return
        if query.strip() != question.strip():
            return

        try:
            import anyio.to_thread

            await anyio.to_thread.run_sync(
                lambda: self.faq_matcher.remember(
                    question, question_key, cache_key, generation
                )
            )
        except Exception:
            logger.warning("the question could not be indexed", exc_info=True)

    # -- the statistics ----------------------------------------------------

    async def _record(
        self,
        question: str,
        question_key: str,
        cost_ms: int = 0,
        hit: bool = False,
    ) -> None:
        """Note that the question was asked, and how it was answered.

        This is what the FAQ rule is made of: how often a question is
        asked, and how often the memory served it.
        """
        try:
            await self.faq_metrics.record_ask(
                question, question_key, cost_ms=cost_ms, hit=hit
            )
        except Exception:
            logger.warning("the question could not be recorded", exc_info=True)

    async def _refresh(self, key: str, question_key: str) -> None:
        """Give the stored answer the time to live it now deserves.

        A question that has become a frequently asked question is kept
        for much longer, so the answer it already has is given more time.
        The time to live is only ever extended, never cut short.
        """
        try:
            ttl = await self.faq_metrics.ttl_for(question_key)
            await self.answer_cache.touch(key, ttl)
        except Exception:
            logger.warning("the answer could not be kept alive", exc_info=True)

    # -- replaying ---------------------------------------------------------

    async def _replay_cached(
        self,
        conversation_id: str,
        cached: CachedAnswer,
        match: str | None = None,
    ) -> AsyncIterator[SourcesEvent | ChatTokenEvent | AnswerCompleteEvent]:
        """Return an answer that was already given, without calling the model.

        The answer is still stored in the conversation, so the exchange
        reads the same way in the history as any other turn.
        """
        yield SourcesEvent(sources=cached.sources, cached=True, match=match)
        yield ChatTokenEvent(token=cached.answer)
        yield AnswerCompleteEvent(
            answer=cached.answer,
            citations=citations_from_records(cached.citations),
            cached=True,
            match=match,
        )
        self.session_store.append_message(
            conversation_id,
            "assistant",
            cached.answer,
            sources=cached.sources,
        )
