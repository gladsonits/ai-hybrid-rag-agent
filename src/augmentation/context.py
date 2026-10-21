"""The assembled context: the retrieved passages, the history, the question.

This module builds what the model actually reads.  The pieces are:

- the **passages** the hybrid search returned, numbered so the model can
  cite them,
- the **summary** of the part of the conversation that was too long to
  replay,
- the **recent turns**, replayed verbatim,
- the **question** itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from augmentation.history import StoredMessage

#: The instruction that tells the model what it may rely on and how to cite.
SYSTEM_PROMPT = """You are a helpful assistant that answers questions about a document corpus.

Answer only from the CONTEXT below. The context is extracted from the source
documents and is the only material you may rely on - do not use outside
knowledge to fill gaps, and do not guess.

Rules:
- Cite the context passages you use with bracketed markers, like [1] or [2].
  A marker refers to the numbered passage in the CONTEXT section.
- If the context does not contain the answer, say so plainly and say what
  is missing. Do not invent details.
- If the question refers to something from earlier in the conversation, use
  the earlier context to work out what it refers to, then answer from the
  CONTEXT.
- Answer in clear prose. Keep it to the length the question needs.
"""

#: The header of the retrieved-passages block.
CONTEXT_HEADER = "CONTEXT"

#: The header of the conversation-summary block.
SUMMARY_HEADER = "SUMMARY OF THE EARLIER CONVERSATION"

#: The separator between the blocks of the system message.
BLOCK_SEPARATOR = "\n\n"


@dataclass(frozen=True, slots=True)
class AugmentedContext:
    """Everything the model is given for one turn."""

    #: The question being answered.
    question: str
    #: The chunks the hybrid search returned.
    passages: list[Any] = field(default_factory=list)
    #: The recent turns, replayed verbatim.
    history: list[StoredMessage] = field(default_factory=list)
    #: A summary of the older part of the conversation, when there was one.
    summary: str | None = None
    #: The search query that was actually used, for logging and debugging.
    retrieval_query: str = ""

    @property
    def was_summarized(self) -> bool:
        """Whether any part of the history had to be summarized."""
        return bool(self.summary)


def format_context(passages: Sequence[Any]) -> str:
    """Render the retrieved passages as a numbered, citable block.

    Each passage is numbered from 1 so the model can cite it, and carries
    the metadata a reader needs to find it in the original document.
    """
    if not passages:
        return f"{CONTEXT_HEADER}{BLOCK_SEPARATOR}(no passages were retrieved)"

    parts = [CONTEXT_HEADER]
    for position, passage in enumerate(passages, start=1):
        metadata = getattr(passage, "metadata", {}) or {}
        citation = _citation_line(metadata)
        text = " ".join(getattr(passage, "chunk_text", "").split())
        parts.append(f"\n[{position}] {citation}\n{text}")
    return "\n".join(parts)


def _citation_line(metadata: dict[str, Any]) -> str:
    """A short, human-readable pointer to where a passage came from."""
    bits: list[str] = []
    source = metadata.get("source")
    if source:
        bits.append(str(source))
    page = metadata.get("page")
    if page is not None:
        bits.append(f"page {page}")
    section = metadata.get("section")
    if section:
        bits.append(str(section))
    project = metadata.get("project")
    if project:
        bits.append(f"project: {project}")
    return " - ".join(bits) if bits else "document"


def format_summary(summary: str) -> str:
    """Render the conversation summary as a block."""
    return f"{SUMMARY_HEADER}{BLOCK_SEPARATOR}{summary}"


def build_messages(context: AugmentedContext) -> list[dict[str, str]]:
    """Build the message list to send to the model.

    The order is the one every chat API expects: the system instruction,
    then the previous turns, then the new question.  The retrieved
    passages and the summary go into the system message, so they are always
    in force, rather than being a user turn the model might treat as a
    question.
    """
    system_parts = [SYSTEM_PROMPT, format_context(context.passages)]
    if context.summary:
        system_parts.append(format_summary(context.summary))

    messages: list[dict[str, str]] = [
        {"role": "system", "content": BLOCK_SEPARATOR.join(system_parts)}
    ]

    messages.extend(
        {"role": message.role, "content": message.content}
        for message in context.history
        if message.role in ("user", "assistant") and message.content.strip()
    )
    messages.append({"role": "user", "content": context.question})
    return messages
