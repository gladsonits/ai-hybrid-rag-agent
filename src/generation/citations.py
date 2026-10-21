"""Resolving the citation markers the model used back to the passages.

The model is told to cite the passages it used with bracketed markers -
``[1]``, ``[2]`` - and the context block numbers the passages from 1, so
a marker is a 1-based index into the passages that were given to it.

A marker that points at a passage that was not retrieved is dropped.  The
model must not be able to invent a source, so a marker outside the range
of the passages is treated as noise rather than as a citation.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

#: The citation markers.  The fullwidth brackets are there because models
#: often emit 【1】 instead of [1].
CITATION_PATTERN = re.compile(r"[\[【]\s*(\d+)\s*[\]】]")


@dataclass(frozen=True, slots=True)
class Citation:
    """One passage the answer referred to."""

    #: The marker as it appears in the answer, e.g. 1 for "[1]".
    marker: int
    #: The id of the chunk in the index.
    chunk_id: int
    #: The text of the passage.
    text: str
    #: Where the passage came from.
    metadata: dict[str, Any]
    #: The retrieval score of the passage.
    score: float | None = None


def extract_citations(answer: str, passages: Sequence[Any]) -> list[Citation]:
    """Find the passages the answer cited.

    The markers are returned in the order they first appear in the answer,
    and each passage is cited once however many times it is marked.
    """
    seen: set[int] = set()
    citations: list[Citation] = []

    for match in CITATION_PATTERN.finditer(answer):
        marker = int(match.group(1))
        if marker in seen:
            continue
        seen.add(marker)
        if not 1 <= marker <= len(passages):
            continue

        passage = passages[marker - 1]
        citations.append(
            Citation(
                marker=marker,
                chunk_id=int(getattr(passage, "chunk_id", 0)),
                text=" ".join(getattr(passage, "chunk_text", "").split()),
                metadata=getattr(passage, "metadata", {}) or {},
                score=getattr(passage, "score", None),
            )
        )

    return citations


def build_source_records(passages: Sequence[Any]) -> list[dict[str, Any]]:
    """The source records for the passages, in the order they were numbered.

    These are the passages the answer was *built from*.  They are known
    before the answer exists, so a UI can show them while the answer is
    still streaming.
    """
    return [
        {
            "chunk_id": int(getattr(passage, "chunk_id", 0)),
            "text": " ".join(getattr(passage, "chunk_text", "").split()),
            "metadata": getattr(passage, "metadata", {}) or {},
            "score": getattr(passage, "score", None),
        }
        for passage in passages
    ]
