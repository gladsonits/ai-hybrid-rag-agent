"""The FastAPI application: one endpoint, ``POST /chat``.

This module is only the transport.  The work - retrieving, choosing what
to replay, summarizing, answering, remembering - happens in
:mod:`augmentation` and :mod:`generation`, and the endpoint just turns an
HTTP request into a call on the chat service and the service events back
into an HTTP response.

The endpoint receives the user message, answers it from the hybrid index
and streams the answer back as server-sent events.  The conversation
history lives in PostgreSQL, keyed by a client-supplied
``conversation_id``, so the model has the context of the earlier questions
when it answers a follow-up.

The event stream carries five kinds of event:

- ``sources``   - the passages the answer will be based on, sent first so
  the UI can show them while the answer is still arriving.
- ``token``     - a piece of the answer, as it is produced.
- ``citations`` - the passages the answer actually cited, once the answer
  is complete.
- ``done``      - the end of the stream, with the full answer.
- ``error``     - a failure; the stream ends after it.

The heavy pieces - the embedding model, the database, the LLM client - are
built once at startup and shared, because loading the embedding model takes
seconds and it is safe to reuse.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.schemas import ChatEvent, ChatRequest, Citation, Source
from augmentation.augmenter import Augmenter
from augmentation.chat_service import (
    AnswerCompleteEvent,
    ChatService,
    ChatTokenEvent,
    SourcesEvent,
)
from augmentation.config import AugmentationSettings
from augmentation.session_store import SessionStore
from generation.llm import GroqLlmClient
from generation.responder import Responder
from retrieval.database import Database
from retrieval.embedder import EmbeddingModel
from retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)

#: The media type for a server-sent event stream.
SSE_MEDIA_TYPE = "text/event-stream"


@dataclass
class Services:
    """The long-lived objects the endpoint needs.

    They are built once and shared across requests: the embedding model is
    expensive to load, and the LLM client keeps a connection pool.
    """

    chat_service: ChatService
    session_store: SessionStore
    database: Database | None
    embedder: EmbeddingModel | None
    retriever: HybridRetriever
    llm: Any


def build_services() -> Services:
    """Build the application services from the environment."""
    database = Database()
    embedder = EmbeddingModel()
    retriever = HybridRetriever(database, embedder)
    session_store = SessionStore(database)
    llm = GroqLlmClient()
    augmenter = Augmenter(
        retriever=retriever,
        llm=llm,
        settings=AugmentationSettings.from_env(),
        summary_store=session_store,
    )
    chat_service = ChatService(
        augmenter=augmenter,
        responder=Responder(llm),
        session_store=session_store,
    )
    return Services(
        chat_service=chat_service,
        session_store=session_store,
        database=database,
        embedder=embedder,
        retriever=retriever,
        llm=llm,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepare the database and warm the embedding model up at startup.

    Warming up here means the first real request does not pay for loading
    the model, which takes several seconds.  A failure to load the model is
    logged rather than raised, so the server still starts and the error
    surfaces on the request that needs it.
    """
    services: Services = app.state.services
    services.session_store.ensure_schema()

    if services.embedder is not None:
        try:
            services.embedder.load()
            logger.info(
                "embedding model ready (%s)", services.embedder.settings.model
            )
        except Exception:
            logger.exception(
                "could not load the embedding model %s",
                services.embedder.settings.model,
            )

    yield


def create_app(services: Services | None = None) -> FastAPI:
    """Build the FastAPI application.

    ``services`` is injectable so a test can supply stand-ins for the
    database and the LLM.
    """
    app = FastAPI(
        title="AI Hybrid RAG Agent",
        description=(
            "A chat endpoint over a document corpus, answered with a hybrid "
            "dense + keyword index and grounded in the passages it retrieves."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.services = services if services is not None else build_services()

    @app.get("/health")
    def health() -> dict[str, Any]:
        """A cheap liveness check that does not touch the database."""
        return {"status": "ok"}

    @app.post("/chat")
    async def chat(
        request: ChatRequest,
        services: Services = Depends(_get_services),
    ) -> StreamingResponse:
        """Answer a message, streaming the answer back as SSE events."""
        question = request.message.strip()
        if not question:
            raise HTTPException(status_code=422, detail="message must not be blank")

        async def event_stream() -> AsyncIterator[bytes]:
            async for event in _chat_events(
                services.chat_service, request.conversation_id, question
            ):
                yield _encode_sse(event)

        return StreamingResponse(
            event_stream(),
            media_type=SSE_MEDIA_TYPE,
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def _get_services(request: Request) -> Services:
    """FastAPI dependency that hands the endpoint the shared services."""
    return request.app.state.services


async def _chat_events(
    service: ChatService,
    conversation_id: str,
    question: str,
) -> AsyncIterator[ChatEvent]:
    """Run one chat turn and emit the events for it."""
    try:
        answer_parts: list[str] = []
        async for event in service.stream_answer(conversation_id, question):
            if isinstance(event, SourcesEvent):
                yield ChatEvent(
                    type="sources",
                    sources=[
                        Source(
                            chunk_id=int(source.get("chunk_id", 0)),
                            text=source.get("text", ""),
                            metadata=source.get("metadata", {}) or {},
                            score=source.get("score"),
                        )
                        for source in event.sources
                    ],
                )
            elif isinstance(event, ChatTokenEvent):
                answer_parts.append(event.token)
                yield ChatEvent(type="token", token=event.token)
            elif isinstance(event, AnswerCompleteEvent):
                answer_parts = [event.answer]
                yield ChatEvent(
                    type="citations",
                    citations=[
                        Citation(
                            marker=citation.marker,
                            chunk_id=citation.chunk_id,
                            text=citation.text,
                            metadata=citation.metadata,
                            score=citation.score,
                        )
                        for citation in event.citations
                    ],
                )

        yield ChatEvent(
            type="done",
            conversation_id=conversation_id,
            token="".join(answer_parts),
        )
    except Exception as exc:
        logger.exception("chat turn failed")
        yield ChatEvent(type="error", error=str(exc))


def _encode_sse(event: ChatEvent) -> bytes:
    """Encode a chat event as a server-sent event."""
    payload = event.model_dump(exclude_none=True)
    data = json.dumps(payload)
    return f"event: {event.type}\ndata: {data}\n\n".encode("utf-8")


#: The application instance uvicorn serves.
app = create_app()
