"""The distributed memory: answers that are asked often, kept in Redis.

The chat turn asks the memory before it does anything expensive.  A
question that has been answered before comes straight back out of Redis,
with the sources and the citations it was built from, and neither the
retrieval nor the model is called.

A Redis outage must not take the API down with it, so every call fails
open: a lookup that cannot be made is a miss, and a write that cannot be
made is dropped.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from memory.config import CacheSettings

logger = logging.getLogger(__name__)

#: The namespace every answer sits under, so the memory can share a
#: Redis database with other things.
CACHE_NAMESPACE = "rag:faq:answer"


@dataclass(frozen=True, slots=True)
class CachedAnswer:
    """One stored answer, with the sources and the citations it used.

    The sources and the citations are plain dictionaries, because they
    have to survive a round trip through JSON.
    """

    #: The answer itself.
    answer: str = ""
    #: The passages the answer was built from.
    sources: list[dict[str, Any]] = field(default_factory=list)
    #: The passages the answer actually referred to.
    citations: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        """The stored form of the answer."""
        return json.dumps(
            {
                "answer": self.answer,
                "sources": self.sources,
                "citations": self.citations,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> "CachedAnswer":
        """Read an answer back out of its stored form."""
        payload = json.loads(raw)
        return cls(
            answer=str(payload.get("answer", "")),
            sources=list(payload.get("sources") or []),
            citations=list(payload.get("citations") or []),
        )


class IndexGeneration(Protocol):
    """Where the version of the indexed corpus is read from.

    The corpus is versioned so that an answer built from an older
    version of it is never served again.  The interface is synchronous
    and blocking, like the FAQ matcher - the caller runs it in a worker
    thread.
    """

    def generation(self) -> int:
        """The current version of the corpus."""
        ...


class AnswerCache(Protocol):
    """Where an answer is looked up, and where it is stored."""

    async def get(self, key: str) -> CachedAnswer | None:
        """The answer stored under ``key``, or ``None`` when there is none."""
        ...

    async def set(
        self,
        key: str,
        answer: CachedAnswer,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store ``answer`` under ``key``, for ``ttl_seconds``."""
        ...

    async def touch(self, key: str, ttl_seconds: int) -> None:
        """Give a stored answer a new time to live, without rewriting it."""
        ...

    async def ping(self) -> bool:
        """Whether the memory is reachable."""
        ...

    async def close(self) -> None:
        """Release whatever the memory holds open."""
        ...


class NullAnswerCache:
    """The memory that remembers nothing.

    Used when the cache is switched off, so the rest of the pipeline does
    not have to know whether there is a memory behind it or not.
    """

    async def get(self, key: str) -> CachedAnswer | None:
        return None

    async def set(
        self,
        key: str,
        answer: CachedAnswer,
        ttl_seconds: int | None = None,
    ) -> None:
        return None

    async def touch(self, key: str, ttl_seconds: int) -> None:
        return None

    async def ping(self) -> bool:
        return False

    async def close(self) -> None:
        return None


class RedisAnswerCache:
    """The distributed memory, in Redis."""

    def __init__(
        self,
        settings: CacheSettings | None = None,
        client: Any | None = None,
    ) -> None:
        self.settings = settings or CacheSettings.from_env()
        self.ttl_seconds = self.settings.ttl_seconds
        self._client = client

    @property
    def client(self) -> Any:
        """The Redis client, built the first time it is needed."""
        if self._client is None:
            from redis.asyncio import Redis

            self._client = Redis.from_url(
                self.settings.url, decode_responses=True
            )
        return self._client

    async def get(self, key: str) -> CachedAnswer | None:
        """The answer stored under ``key``, or ``None``."""
        try:
            raw = await self.client.get(key)
        except Exception:
            logger.warning("the memory could not be read", exc_info=True)
            return None

        if not raw:
            return None

        try:
            return CachedAnswer.from_json(raw)
        except (ValueError, TypeError):
            logger.warning(
                "the memory held something that is not an answer", exc_info=True
            )
            return None

    async def set(
        self,
        key: str,
        answer: CachedAnswer,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store ``answer`` under ``key``, for an hour by default."""
        ttl = self.ttl_seconds if ttl_seconds is None else ttl_seconds
        try:
            await self.client.set(key, answer.to_json(), ex=ttl)
        except Exception:
            logger.warning("the memory could not be written", exc_info=True)

    async def touch(self, key: str, ttl_seconds: int) -> None:
        """Give a stored answer a new time to live, without rewriting it.

        This is what keeps a frequently asked question alive: every time
        it is asked again, the answer it already has is given more time
        to live.
        """
        try:
            # GT means the time to live is only ever extended, never cut
            # short - a question that is no longer asked often should not
            # lose the time it had left.
            await self.client.expire(key, ttl_seconds, gt=True)
        except Exception:
            logger.warning("the memory could not be extended", exc_info=True)

    async def ping(self) -> bool:
        """Whether the memory is reachable."""
        try:
            await self.client.ping()
        except Exception:
            return False
        return True

    async def close(self) -> None:
        """Close the connection to the memory."""
        if self._client is None:
            return
        try:
            await self._client.aclose()
        except Exception:
            logger.warning(
                "the memory connection could not be closed", exc_info=True
            )


def normalize_question(question: str) -> str:
    """The question as it is compared: lower case, with the spacing collapsed.

    A difference in case or in spacing does not make a new question, so
    it does not make a new key.
    """
    return " ".join(question.lower().split())


def build_question_key(question: str) -> str:
    """The identifier of a question, whatever it is worded like.

    This identifies the question itself, with no conversation attached.
    It is what the statistics of the question are kept under, and what
    the FAQ index looks a question up by.  It is not a Redis key.
    """
    return hashlib.sha256(
        normalize_question(question).encode("utf-8")
    ).hexdigest()


def build_cache_key(
    question: str,
    context: str = "",
    generation: int = 0,
) -> str:
    """The key an answer is stored under.

    The question is normalized, so a difference in case or in spacing
    does not make a new key.  The context is mixed in as well: the same
    question asked in a different conversation is a different question,
    and the answer to it would be different too.

    The **generation** is the version of the corpus the answer was built
    from, and it is part of the key.  That is what makes a rebuild of
    the corpus invalidate the memory: an answer from the corpus as it
    was is simply never asked for again, because the key it is stored
    under is not the key the question produces any more.  The memory
    misses, the question is answered afresh, and the old answer expires
    on its own time to live.
    """
    digest = hashlib.sha256()
    digest.update(normalize_question(question).encode("utf-8"))
    digest.update(b"|")
    digest.update(context.strip().encode("utf-8"))
    digest.update(b"|")
    digest.update(str(int(generation)).encode("utf-8"))
    return f"{CACHE_NAMESPACE}:{digest.hexdigest()}"
