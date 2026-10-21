"""Settings for the augmentation step.

The augmentation step decides what the model sees: the chunks the hybrid
search retrieved, the conversation so far, and - when the conversation has
grown too long to replay - a summary of its earlier part.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

#: Rough characters per token, used to size the conversation history
#: against the model's context window without a model-specific tokenizer.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True, slots=True)
class AugmentationSettings:
    """How much context to gather, and how much of it to keep verbatim."""

    #: How many chunks the hybrid search returns.
    top_k: int
    #: How many of the most recent turns are replayed verbatim.
    history_turns: int
    #: How many earlier questions are folded into the search query.
    query_history_messages: int
    #: The token budget for the replayed history.  Past it, the older
    #: turns are summarized instead of replayed.
    max_history_tokens: int
    #: The cap on the length of the summary.
    summary_max_tokens: int

    @classmethod
    def from_env(cls) -> "AugmentationSettings":
        return cls(
            top_k=int(os.getenv("AUGMENT_TOP_K", "5")),
            history_turns=int(os.getenv("AUGMENT_HISTORY_TURNS", "5")),
            query_history_messages=int(
                os.getenv("AUGMENT_QUERY_HISTORY_MESSAGES", "2")
            ),
            max_history_tokens=int(os.getenv("AUGMENT_MAX_HISTORY_TOKENS", "2000")),
            summary_max_tokens=int(os.getenv("AUGMENT_SUMMARY_MAX_TOKENS", "400")),
        )
