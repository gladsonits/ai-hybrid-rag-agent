"""Producing the answer: the prompt goes in, the text and the citations come out.

This is the generation step.  It owns two things:

- the **call to the model** - the prompt is sent and the answer is streamed
  back, token by token, so the user sees the answer as it is written;
- the **citations** - the markers the model used are resolved back to the
  passages they point at, so the response can say where each claim came
  from.

The citations can only be resolved once the whole answer is in, because a
marker may appear anywhere in it.  So the responder yields the tokens as
they arrive and the finished answer at the end.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from generation.citations import Citation, extract_citations
from generation.llm import LlmClient


@dataclass(frozen=True, slots=True)
class TokenEvent:
    """One piece of the answer."""

    token: str


@dataclass(frozen=True, slots=True)
class AnswerEvent:
    """The finished answer, with the passages it cited."""

    answer: str
    citations: list[Citation]


class Responder:
    """Turns a prompt into an answer.

    The dependencies are injected, so the responder can be tested against a
    stand-in for the model without a network call and without an API key.
    """

    def __init__(self, llm: LlmClient) -> None:
        self.llm = llm

    async def respond(
        self,
        messages: Sequence[dict[str, str]],
        passages: Sequence[Any] = (),
    ) -> AsyncIterator[TokenEvent | AnswerEvent]:
        """Stream the answer, then report it with its citations.

        ``passages`` are the ones the prompt was built from, in the order
        they were numbered there - the citation markers refer to them by
        that number.
        """
        parts: list[str] = []
        async for token in self.llm.stream(messages):
            parts.append(token)
            yield TokenEvent(token=token)

        answer = "".join(parts)
        yield AnswerEvent(
            answer=answer,
            citations=extract_citations(answer, passages),
        )
