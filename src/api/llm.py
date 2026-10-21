"""The LLM call: Groq, streamed.

The client is wrapped behind a tiny interface so the chat service can be
tested without a network call and without an API key - the tests supply a
stand-in that returns a fixed stream.

Only one method matters: :meth:`LlmClient.stream`, which yields the answer
token by token.  The interface is deliberately small so a different provider
could be dropped in behind it later.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol

from api.config import LlmSettings


class LlmClient(Protocol):
    """Anything that can turn a list of chat messages into a stream of text."""

    def stream(self, messages: Sequence[dict[str, str]]) -> AsyncIterator[str]:
        """Yield the answer's text, piece by piece."""
        ...


class GroqLlmClient:
    """Streams a chat completion from Groq.

    The API key and model come from the environment.  A missing key is
    reported when the client is built, not when the first token is requested,
    so a misconfigured deployment fails loudly at startup.
    """

    def __init__(self, settings: LlmSettings | None = None) -> None:
        self.settings = settings or LlmSettings.from_env()
        if not self.settings.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set - add it to .env before starting the API"
            )
        self._client = None

    def _get_client(self) -> Any:
        """Build the AsyncGroq client on first use."""
        if self._client is None:
            from groq import AsyncGroq

            self._client = AsyncGroq(api_key=self.settings.api_key)
        return self._client

    async def stream(self, messages: Sequence[dict[str, str]]) -> AsyncIterator[str]:
        """Yield the model's answer as it is produced."""
        client = self._get_client()
        stream = await client.chat.completions.create(
            messages=list(messages),
            model=self.settings.model,
            temperature=self.settings.temperature,
            max_completion_tokens=self.settings.max_completion_tokens,
            stream=True,
        )
        async for chunk in stream:
            text = chunk.choices[0].delta.content
            if text:
                yield text
