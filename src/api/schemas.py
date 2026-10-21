"""The request and response shapes of the chat endpoint."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """What the UI sends.

    ``conversation_id`` is chosen by the client and identifies the session.
    The client keeps the same id for the whole conversation, which is how
    the model knows the earlier turns.
    """

    message: str = Field(
        ...,
        min_length=1,
        description="The user's question.",
    )
    conversation_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Client-supplied session id; reuse it to keep the history.",
    )


class Source(BaseModel):
    """A chunk the answer was built from."""

    chunk_id: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None


class Citation(BaseModel):
    """A passage the answer actually referred to.

    ``marker`` is the number the model wrote, e.g. 1 for "[1]", and points
    at the passage of that number in the sources list.
    """

    marker: int
    chunk_id: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None


class ChatEvent(BaseModel):
    """One server-sent event.

    ``type`` says what the event carries: ``token`` for a piece of the
    answer, ``sources`` for the passages the answer was built from,
    ``citations`` for the passages it actually cited, ``done`` for the end
    of the stream, and ``error`` for a failure.
    """

    type: str
    token: str | None = None
    sources: list[Source] | None = None
    citations: list[Citation] | None = None
    conversation_id: str | None = None
    error: str | None = None
