"""The augmentation step: assembling the model's context for one turn.

This is the step between "the user asked something" and "the model
answers".  It does three things, in order:

1. **Retrieves** the passages that bear on the question.  The question is
   first made self-contained by folding in the earlier questions, because
   the index does not see the conversation.
2. **Chooses** how much of the conversation to replay.  The recent turns
   are kept verbatim, up to a token budget.
3. **Summarizes** whatever did not fit, and puts the summary in the
   context in place of the original turns.

The result is an :class:`AugmentedContext`, which the caller turns into
messages.  Nothing here knows about HTTP, or about how the answer is
streamed back.

A summary is not regenerated on every turn.  Once a summary has been
written for a conversation it is stored against the id of the last message
it covers, and only the turns *after* that id are considered for
summarizing on the next turn.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from augmentation.config import AugmentationSettings
from augmentation.context import AugmentedContext
from augmentation.history import (
    StoredMessage,
    build_retrieval_query,
    select_window,
)
from generation.llm import LlmClient
from generation.summarizer import Summarizer


class Retriever(Protocol):
    """The part of ``HybridRetriever`` the augmentation step needs."""

    def search(
        self,
        query: str,
        limit: int = 10,
        candidates: int = 50,
        filters: dict[str, Any] | None = None,
    ) -> list[Any]: ...


class SummaryStore(Protocol):
    """The part of the session store the augmentation step needs.

    Keeping it as a protocol means the augmenter does not depend on
    PostgreSQL - it only needs somewhere to keep a summary.
    """

    def latest_summary(self, conversation_id: str) -> Any | None: ...

    def store_summary(
        self,
        conversation_id: str,
        summary: str,
        covers_up_to_id: int,
    ) -> int: ...


class Augmenter:
    """Builds the context for one turn."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LlmClient,
        settings: AugmentationSettings | None = None,
        summary_store: SummaryStore | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.settings = settings or AugmentationSettings.from_env()
        self.summary_store = summary_store
        self.summarizer = Summarizer(
            llm, max_tokens=self.settings.summary_max_tokens
        )

    async def augment(
        self,
        question: str,
        history: Sequence[StoredMessage] = (),
        conversation_id: str | None = None,
    ) -> AugmentedContext:
        """Assemble the context for ``question``."""
        query = build_retrieval_query(
            question,
            history,
            lookback=self.settings.query_history_messages,
        )

        passages = await self._retrieve(query)

        recent, old, summary = await self._select_history(
            history, conversation_id
        )

        return AugmentedContext(
            question=question,
            passages=passages,
            history=recent,
            summary=summary,
            retrieval_query=query,
        )

    async def _retrieve(self, query: str) -> list[Any]:
        """Call the synchronous retriever off the event loop.

        The retriever does blocking work - a database round trip and an
        embedding forward pass - so running it inline would stall every
        other request.
        """
        import anyio.to_thread

        return await anyio.to_thread.run_sync(
            lambda: self.retriever.search(query, limit=self.settings.top_k)
        )

    async def _select_history(
        self,
        history: Sequence[StoredMessage],
        conversation_id: str | None,
    ) -> tuple[list[StoredMessage], list[StoredMessage], str | None]:
        """Work out the replay window, and the summary of the rest.

        Returns ``(recent, old, summary)``.  ``old`` is the run of
        messages that did not fit and, when there is a summary store, is
        summarized and saved.  It is returned so the caller can see what
        was dropped even when no summary was written.
        """
        stored = self._stored_summary(conversation_id)
        summary_text = stored.summary if stored else None

        # Messages already folded into a stored summary are not
        # considered again - re-summarizing a summary would only lose
        # detail.
        unsummarized = list(history)
        if stored is not None:
            unsummarized = [
                message
                for message in history
                if message.id is None or message.id > stored.covers_up_to_id
            ]

        recent, old = select_window(
            unsummarized,
            turns=self.settings.history_turns,
            max_tokens=self.settings.max_history_tokens,
        )

        if not old:
            return recent, old, summary_text

        # A stored summary already covers the older messages, so only the
        # newly-dropped ones need summarizing.
        new_summary = await self.summarizer.summarize(old)
        if not new_summary:
            return recent, old, summary_text

        if summary_text:
            new_summary = f"{summary_text}\n\n{new_summary}"
        summary_text = new_summary

        self._store_summary(conversation_id, summary_text, old)
        return recent, old, summary_text

    def _stored_summary(self, conversation_id: str | None) -> Any | None:
        """Read the stored summary, if there is one."""
        if self.summary_store is None or conversation_id is None:
            return None
        return self.summary_store.latest_summary(conversation_id)

    def _store_summary(
        self,
        conversation_id: str | None,
        summary: str,
        old: Sequence[StoredMessage],
    ) -> None:
        """Save the summary, marked with how far it reaches."""
        if self.summary_store is None or conversation_id is None:
            return
        ids = [message.id for message in old if message.id is not None]
        if not ids:
            return
        self.summary_store.store_summary(conversation_id, summary, max(ids))
