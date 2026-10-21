"""Communication with the LLM: the interface, and the Groq client.

The client is wrapped behind a small protocol, so the rest of the code can
be tested without a network call and without an API key - the tests supply
a stand-in that returns a fixed stream.

Only one method matters: :meth:`GroqLlmClient.stream`, which yields the
answer's text token by token.  The summarizer in the augmentation step
uses the same interface, because summarizing a conversation is a
generation task too.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from generation.config import LlmSettings


class LlmClient(Protocol):
    """Anything that can turn chat messages into a stream of text."""

    def stream(
        self,
        messages: Sequence[dict[str, str]],
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield the answer's text, piece by piece.

        ``max_tokens`` caps the length of the answer; a caller that does
        not care leaves it unset.
        """
        ...


class GroqLlmClient:
    """Streams a chat completion from Groq.

    The API key and model come from the environment.  A missing key is
    reported when the client is built, not when the first token is
    requested, so a misconfigured deployment fails loudly at startup.
    """

    def __init__(self, settings: LlmSettings | None = None) -> None:
        self.settings = settings or LlmSettings.from_env()
        if not self.settings.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set - add it to .env before starting the API"
            )
        self._client = None

    def _get_client(self) -> object:
        """Build the AsyncGroq client on first use."""
        if self._client is None:
            from groq import AsyncGroq

            self._client = AsyncGroq(api_key=self.settings.api_key)
        return self._client

    async def stream(
        self,
        messages: Sequence[dict[str, str]],
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield the model's answer as it is produced.

        ``max_tokens`` caps the answer; without it the configured
        ``LLM_MAX_TOKENS`` applies.  The summarizer uses it to keep a
        summary short.
        """
        client = self._get_client()
        stream = await client.chat.completions.create(
            messages=list(messages),
            model=self.settings.model,
            temperature=self.settings.temperature,
            max_completion_tokens=max_tokens or self.settings.max_completion_tokens,
            stream=True,
        )
        async for chunk in stream:
            text = chunk.choices[0].delta.content
            if text:
                yield text
