"""Turning a question plus retrieved chunks into messages for the model.

Two things happen here, and both matter for a chat UI:

1. **The retrieved context is grounded.**  The chunks are numbered and given
   to the model as the only source it may cite, so the answer can point back
   at a specific passage.
2. **A follow-up question is made self-contained.**  "And what about the
   second one?" means nothing on its own - it only makes sense next to the
   earlier turns.  The model sees the conversation, but the *retrieval* step
   does not, so the earlier user messages are folded into the search query.

The prompt is built here rather than inline in the service so it can be
tested on its own, without a database or a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from api.session_store import StoredMessage

#: The instruction that tells the model what it may rely on and how to cite.
SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions about a document corpus.

Answer only from the CONTEXT below. The context is extracted from the source
documents and is the only material you may rely on - do not use outside
knowledge to fill gaps, and do not guess.

Rules:
- Cite the context passages you use with bracketed markers, like [1] or [2].
  A marker refers to the numbered passage in the CONTEXT section.
- If the context does not contain the answer, say so plainly and say what
  is missing. Do not invent details.
- If the question refers to something from earlier in the conversation, use
  that earlier context to work out what it refers to, then answer from the
  CONTEXT.
- Answer in clear prose. Keep it to the length the question needs.
"""

#: The header of the context block.
CONTEXT_HEADER = "CONTEXT"


def format_context(chunks: Sequence[Any]) -> str:
    """Render the retrieved chunks as a numbered, citable block.

    Each chunk is numbered from 1 so the model can cite it, and carries the
    metadata a reader needs to find the passage in the original document.
    """
    if not chunks:
        return "CONTEXT\n\n(no passages were retrieved)"

    parts = [CONTEXT_HEADER]
    for position, chunk in enumerate(chunks, start=1):
        metadata = getattr(chunk, "metadata", {}) or {}
        citation = _citation_line(metadata)
        text = " ".join(getattr(chunk, "chunk_text", "").split())
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


def build_retrieval_query(
    question: str,
    history: Sequence[StoredMessage],
    lookback: int = 2,
) -> str:
    """Fold earlier user messages into the search query.

    Only the user's own earlier messages are used, and only the last
    ``lookback`` of them, so the query stays a question rather than turning
    into a transcript.  The current question always comes last, so it
    carries the most weight in both the dense and the keyword index.
    """
    earlier = [
        message.content.strip()
        for message in history
        if message.role == "user" and message.content.strip()
    ]
    earlier = earlier[-lookback:] if lookback > 0 else []

    if not earlier:
        return question

    # The most recent earlier question is dropped when it is the same as the
    # current one - repeating it adds nothing to the query.
    if earlier and earlier[-1].lower() == question.strip().lower():
        earlier = earlier[:-1]

    if not earlier:
        return question

    return " ".join([*earlier, question])


def build_messages(
    question: str,
    chunks: Sequence[Any],
    history: Sequence[StoredMessage] = (),
    max_history_turns: int = 5,
) -> list[dict[str, str]]:
    """Build the message list to send to the model.

    The order is the one every chat API expects: the system instruction, then
    the previous turns, then the new question.  The retrieved context goes
    into the system message so it is always in force, rather than being a
    user turn that the model might treat as a question.
    """
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": f"{SYSTEM_PROMPT}\n\n{format_context(chunks)}",
        }
    ]

    messages.extend(_history_messages(history, max_history_turns))
    messages.append({"role": "user", "content": question})
    return messages


def _history_messages(
    history: Sequence[StoredMessage],
    max_history_turns: int,
) -> list[dict[str, str]]:
    """Render the previous turns as chat messages.

    ``max_history_turns`` counts turns, and a turn is a user message plus
    the assistant answer that came back for it, so the window is taken from
    the end of the history and always starts on a user message.
    """
    if max_history_turns <= 0 or not history:
        return []

    # Keep whole trailing turns: walk back to the start of the last
    # ``max_history_turns`` user messages.
    kept: list[StoredMessage] = []
    turns = 0
    for message in reversed(history):
        if message.role == "user":
            if turns == max_history_turns:
                break
            turns += 1
        kept.append(message)
    kept.reverse()

    # The walk stops on a user message, so it can pick up the assistant
    # answer that came *before* the window.  Drop it - a turn that starts
    # with an answer and has no question is not a turn.
    while kept and kept[0].role != "user":
        kept.pop(0)

    return [
        {"role": message.role, "content": message.content}
        for message in kept
        if message.role in ("user", "assistant") and message.content.strip()
    ]
