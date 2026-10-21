"""The conversation history, and how much of it to keep.

Two decisions live here:

- **Which turns to replay.**  The most recent ones are kept verbatim; the
  older ones are dropped or summarized, depending on the budget.
- **What to search for.**  The retrieval step does not see the
  conversation, so a follow-up question has to be made self-contained
  before it reaches the index.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class StoredMessage:
    """One message in a conversation."""

    role: str
    content: str
    #: The chunks cited by an assistant message.  Empty for a user message.
    sources: list[dict[str, Any]] = field(default_factory=list)
    #: The row id, used to mark how far a stored summary reaches.
    id: int | None = None


def split_turns(
    history: Sequence[StoredMessage],
    turns: int,
) -> tuple[list[StoredMessage], list[StoredMessage]]:
    """Split a history into the old part and the recent part.

    A *turn* is a user message and the assistant answer that came back for
    it.  The split is at a turn boundary, so the recent part never starts
    with a half turn.

    The recent part holds the last ``turns`` turns.  If the history has
    no user message at all, the whole thing is treated as old.
    """
    if not history:
        return [], []

    # Find where the last ``turns`` user messages start.
    user_positions = [
        index for index, message in enumerate(history) if message.role == "user"
    ]
    if not user_positions:
        return list(history), []

    start = user_positions[-turns] if len(user_positions) >= turns else 0
    return list(history[:start]), list(history[start:])


def estimate_tokens(text: str) -> int:
    """A rough token count for a piece of text.

    Deliberately approximate - a real tokenizer would cost more than it
    is worth here, and the budget only needs to be roughly right.
    """
    return max(1, len(text) // 4)


def group_turns(
    messages: Sequence[StoredMessage],
) -> list[list[StoredMessage]]:
    """Group messages into whole turns.

    A turn starts at a user message and runs to the next one, so it holds
    the question and the answer that came back for it.  Anything before
    the first user message is its own group - there is no question to
    attach it to.
    """
    groups: list[list[StoredMessage]] = []
    for message in messages:
        if message.role == "user" or not groups:
            groups.append([message])
        else:
            groups[-1].append(message)
    return groups


def select_window(
    history: Sequence[StoredMessage],
    turns: int,
    max_tokens: int,
) -> tuple[list[StoredMessage], list[StoredMessage]]:
    """Choose the turns to replay, and the ones left over.

    Returns ``(recent, old)``.  ``recent`` is the tail of the history
    that fits the budget; ``old`` is everything before it, which the
    caller may summarize.

    The budget is spent on whole turns, newest first.  A turn is a
    question and the answer it produced, and replaying one without the
    other gives the model an answer to a question it never saw - so a
    turn is either kept or dropped as a unit.

    The most recent turn is always kept, even when it does not fit:
    dropping it would leave the model with no conversation at all.
    """
    old, recent = split_turns(history, turns=turns)
    if not recent:
        return recent, old

    groups = group_turns(recent)

    kept_groups: list[list[StoredMessage]] = []
    used = 0
    for group in reversed(groups):
        cost = sum(estimate_tokens(message.content) for message in group)
        if kept_groups and used + cost > max_tokens:
            break
        kept_groups.append(group)
        used += cost
        if used >= max_tokens:
            break

    kept_groups.reverse()
    kept = [message for group in kept_groups for message in group]
    dropped = recent[: len(recent) - len(kept)]
    return kept, old + dropped


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

    # The most recent earlier question is dropped when it is the same as
    # the current one - repeating it adds nothing to the query.
    if earlier[-1].lower() == question.strip().lower():
        earlier = earlier[:-1]

    if not earlier:
        return question

    return " ".join([*earlier, question])
