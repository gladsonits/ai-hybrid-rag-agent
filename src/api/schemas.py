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

    ``cache_hit`` says whether the turn was answered from the memory
    rather than by the model, and ``cache_match`` says how the memory
    answered it - ``exact`` for the same words, ``semantic`` for the same
    question in other words.
    """

    type: str
    token: str | None = None
    sources: list[Source] | None = None
    citations: list[Citation] | None = None
    conversation_id: str | None = None
    error: str | None = None
    cache_hit: bool | None = None
    cache_match: str | None = None


class FaqEntry(BaseModel):
    """One frequently asked question, and what is known about it.

    ``asks`` is how many times the question has been asked, ``hits`` how
    many of those the memory served, and ``misses`` how many had to be
    answered by the model.  ``saved_ms`` is the model time the memory
    has saved on it, and ``cost_ms`` is what the last answer cost.
    """

    question: str
    asks: int
    hits: int
    misses: int
    hit_rate: float
    saved_ms: int
    cost_ms: int
    last_asked_at: float
