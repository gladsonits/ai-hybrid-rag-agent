"""The chat turn: the business logic of answering a question.

This is the step that turns a question into an answer and keeps the
conversation going.  The order matters:

1. Read the conversation, so the question can be understood in context.
2. **Augment** the context - retrieve the relevant passages, choose which
   turns to replay, and summarize the ones that did not fit.
3. **Generate** the answer from that context, streaming it out token by
   token, and resolve the citations it used.
4. Store the question and the answer, so the next turn can read them.

The retrieval, the history windowing and the summarization all live in
:mod:`augmentation.augmenter`; the call to the model and the citation
handling live in :mod:`generation`.  This module is the turn that uses
them.

The dependencies - the retriever, the LLM and the session store - are
injected, so the service can be tested against stand-ins without a
database or an API key.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from augmentation.augmenter import Augmenter
from augmentation.context import AugmentedContext, build_messages
from augmentation.session_store import SessionStore
from generation.citations import Citation, build_source_records
from generation.responder import AnswerEvent, Responder, TokenEvent


@dataclass(frozen=True, slots=True)
class SourcesEvent:
    """The passages a turn was answered from, known before the answer is."""

    sources: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ChatTokenEvent:
    """One piece of the answer."""

    token: str


@dataclass(frozen=True, slots=True)
class AnswerCompleteEvent:
    """The finished answer, with the passages it cited."""

    answer: str
    citations: list[Citation]


class ChatService:
    """Answers a question using the index, and remembers the conversation."""

    def __init__(
        self,
        augmenter: Augmenter,
        responder: Responder,
        session_store: SessionStore,
    ) -> None:
        self.augmenter = augmenter
        self.responder = responder
        self.session_store = session_store

    async def stream_answer(
        self,
        conversation_id: str,
        question: str,
    ) -> AsyncIterator[SourcesEvent | ChatTokenEvent | AnswerCompleteEvent]:
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

        context = await self.augmenter.augment(
            question, history, conversation_id=conversation_id
        )
        sources = build_source_records(context.passages)
        yield SourcesEvent(sources=sources)

        messages = build_messages(context)

        answer_parts: list[str] = []
        async for event in self.responder.respond(messages, context.passages):
            if isinstance(event, TokenEvent):
                answer_parts.append(event.token)
                yield ChatTokenEvent(token=event.token)
            elif isinstance(event, AnswerEvent):
                answer_parts = [event.answer]
                yield AnswerCompleteEvent(
                    answer=event.answer,
                    citations=event.citations,
                )

        self.session_store.append_message(
            conversation_id,
            "assistant",
            "".join(answer_parts),
            sources=sources,
        )
