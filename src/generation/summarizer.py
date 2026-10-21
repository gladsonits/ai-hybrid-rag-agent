"""Summarizing text with the model.

The augmentation step uses this to compress a run of conversation turns
that has grown too long to replay.  It lives here rather than there
because asking the model to write text is a generation task, and this
package owns every call to the model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from generation.llm import LlmClient

#: The instruction for the summarizer.
SUMMARY_PROMPT = """You are summarizing part of a conversation so it can take the place of the
original messages.

Write a compact summary that keeps:
- the facts the user established (names, roles, projects, preferences),
- any question the user has already asked and the answer given,
- anything the user referred back to later.

Drop the pleasantries and the filler. Do not add anything that was not
said. Do not answer anything - only report what was said.
"""


class Summarizer:
    """Summarizes a run of conversation turns.

    The turns are taken as anything with ``role`` and ``content``
    attributes, so this package does not have to know what the
    augmentation step calls a message.
    """

    def __init__(self, llm: LlmClient, max_tokens: int = 400) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    async def summarize(self, messages: Sequence[Any]) -> str:
        """Summarize ``messages`` into a single block of text.

        An empty run summarizes to an empty string, so the caller does not
        have to special-case it.
        """
        if not messages:
            return ""

        transcript = "\n".join(
            f"{message.role}: {message.content}" for message in messages
        )
        prompt = (
            f"{SUMMARY_PROMPT}\n\n"
            f"Summarize this part of the conversation:\n\n"
            f"{transcript}"
        )

        parts: list[str] = []
        async for token in self.llm.stream(
            [{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
        ):
            parts.append(token)
        return "".join(parts).strip()
