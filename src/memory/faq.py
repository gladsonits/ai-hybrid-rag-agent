"""What makes a question a frequently asked question.

Two things live here.

**The statistics.**  Every question the memory is asked about gets a
small record: how many times it has been asked, how many times the
memory served it, how much model time that saved, and when it was last
asked.  The record is kept in Redis, so the counters are shared by every
worker and survive a restart.

**The rule.**  A question becomes a *frequently asked question* - an
FAQ - when it has been asked often enough, when the memory has actually
served it, and when it is still being asked.  That is the whole
definition, and it decides two things: how long the answer is kept (an
FAQ is kept for a day, a one-off for an hour), and what the memory
reports as its frequently asked questions.

The module also holds the two small types the rest of the project shares
with the FAQ layer: :class:`FaqStats`, the statistics of one question,
and :class:`FaqCandidate`, the stored question that a new one turns out
to mean the same as.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from memory.config import CacheSettings, FaqSettings

logger = logging.getLogger(__name__)

#: The namespace the statistics of each question live under.
STAT_NAMESPACE = "rag:faq:stat"

#: The sorted set that ranks the tracked questions by how often they are
#: asked.  The score is the number of asks, the member is the key of the
#: question.  It is what the eviction works on.
RANKED_KEY = "rag:faq:ranked"


@dataclass(frozen=True, slots=True)
class FaqStats:
    """What is known about one question."""

    #: The question, as it was asked.
    question: str = ""
    #: How many times it has been asked.
    asks: int = 0
    #: How many times the memory served it.
    hits: int = 0
    #: How much model time the memory has saved on it, in milliseconds.
    saved_ms: int = 0
    #: How long the model took to answer it the last time it had to, in
    #: milliseconds.  This is what every hit on it saves.
    cost_ms: int = 0
    #: When it was last asked, as a Unix timestamp.
    last_asked_at: float = 0.0

    @property
    def misses(self) -> int:
        """How many times it had to be answered by the model."""
        return max(0, self.asks - self.hits)

    @property
    def hit_rate(self) -> float:
        """The share of the asks the memory served."""
        return self.hits / self.asks if self.asks else 0.0

    def to_dict(self) -> dict[str, Any]:
        """The statistics as plain data, for the API and the reports."""
        return {
            "question": self.question,
            "asks": self.asks,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "saved_ms": self.saved_ms,
            "cost_ms": self.cost_ms,
            "last_asked_at": self.last_asked_at,
        }


class FaqMetrics(Protocol):
    """Where the statistics of the frequently asked questions are kept."""

    async def record_ask(
        self,
        question: str,
        question_key: str,
        cost_ms: int = 0,
        hit: bool = False,
    ) -> FaqStats:
        """Record that a question was asked, and how it was answered.

        ``cost_ms`` is how long the model took to answer it, which is
        what a later hit saves.  ``hit`` says the memory served it.
        """
        ...

    async def stats(self, question_key: str) -> FaqStats | None:
        """What is known about one question, if anything."""
        ...

    async def faqs(self, limit: int = 50) -> list[FaqStats]:
        """The questions that count as frequently asked, most asked first."""
        ...

    async def is_faq(self, question_key: str) -> bool:
        """Whether the question counts as a frequently asked question."""
        ...

    async def ttl_for(self, question_key: str) -> int:
        """How long the answer to this question should be kept."""
        ...


@dataclass(frozen=True, slots=True)
class FaqCandidate:
    """A stored question that a new one turns out to mean the same as."""

    #: The stored question, as it was first asked.
    question: str
    #: The key of the stored question.
    question_key: str
    #: The key its answer is stored under in the memory.
    cache_key: str
    #: How close the two questions are, from 0 to 1.
    similarity: float


class FaqMatcher(Protocol):
    """Finds the stored question that a new question means the same as."""

    def match(self, question: str, threshold: float) -> FaqCandidate | None:
        """The stored question this one means the same as, if there is one."""
        ...

    def remember(self, question: str, question_key: str, cache_key: str) -> None:
        """Remember a question, so the next one can be matched against it."""
        ...


class FaqRegistry:
    """The statistics of the frequently asked questions, kept in Redis.

    The counters live in Redis rather than in the process, so they are
    shared by every worker and survive a restart.  As with the answer
    cache, every call fails open: a registry that cannot be reached
    reports nothing and never raises, because a question that cannot be
    counted is still a question that has to be answered.
    """

    def __init__(
        self,
        faq_settings: FaqSettings | None = None,
        cache_settings: CacheSettings | None = None,
        client: Any | None = None,
    ) -> None:
        self.faq_settings = faq_settings or FaqSettings.from_env()
        self.cache_settings = cache_settings or CacheSettings.from_env()
        self._client = client

    @property
    def client(self) -> Any:
        """The Redis client, built the first time it is needed."""
        if self._client is None:
            from redis.asyncio import Redis

            self._client = Redis.from_url(
                self.cache_settings.url, decode_responses=True
            )
        return self._client

    # -- recording ---------------------------------------------------------

    async def record_ask(
        self,
        question: str,
        question_key: str,
        cost_ms: int = 0,
        hit: bool = False,
    ) -> FaqStats:
        """Record that a question was asked.

        The cost is only written when the model actually answered the
        question - a hit must not overwrite the cost of the original
        answer, because that cost is what the memory saves.
        """
        try:
            key = self._stat_key(question_key)
            fields: dict[str, Any] = {
                "question": question,
                "last_asked_at": time.time(),
            }
            if cost_ms:
                fields["cost_ms"] = int(cost_ms)

            await self.client.hset(key, mapping=fields)
            await self.client.hincrby(key, "asks", 1)
            await self.client.zincrby(RANKED_KEY, 1, question_key)

            if hit:
                await self.client.hincrby(key, "hits", 1)
                saved = await self.client.hget(key, "cost_ms")
                if saved:
                    await self.client.hincrby(key, "saved_ms", int(saved))

            await self._evict()
        except Exception:
            logger.warning(
                "the frequently asked questions could not be recorded",
                exc_info=True,
            )

        return await self.stats(question_key) or FaqStats(question=question)

    # -- reading -----------------------------------------------------------

    async def stats(self, question_key: str) -> FaqStats | None:
        """What is known about one question, if anything."""
        try:
            raw = await self.client.hgetall(self._stat_key(question_key))
        except Exception:
            logger.warning(
                "the frequently asked questions could not be read", exc_info=True
            )
            return None

        if not raw:
            return None
        return self._to_stats(raw)

    async def faqs(self, limit: int = 50) -> list[FaqStats]:
        """The questions that count as frequently asked, most asked first.

        The ranking is by how often the question is asked, and the
        questions that do not meet the rule are left out - the report is
        of the FAQs, not of every question that was ever seen.
        """
        try:
            ranked = await self.client.zrange(
                RANKED_KEY, 0, -1, desc=True, withscores=True
            )
        except Exception:
            logger.warning(
                "the frequently asked questions could not be listed", exc_info=True
            )
            return []

        found: list[FaqStats] = []
        for member, _score in ranked:
            try:
                raw = await self.client.hgetall(self._stat_key(member))
            except Exception:
                logger.warning(
                    "the frequently asked questions could not be read", exc_info=True
                )
                return found

            if not raw:
                continue
            stats = self._to_stats(raw)
            if self._is_faq(stats):
                found.append(stats)
            if len(found) >= limit:
                break

        return found

    async def is_faq(self, question_key: str) -> bool:
        """Whether the question counts as a frequently asked question."""
        stats = await self.stats(question_key)
        return stats is not None and self._is_faq(stats)

    async def ttl_for(self, question_key: str) -> int:
        """How long the answer to this question should be kept.

        A frequently asked question is kept for much longer than a
        one-off, because it is the one that will be asked again.
        """
        if await self.is_faq(question_key):
            return self.cache_settings.faq_ttl_seconds
        return self.cache_settings.ttl_seconds

    # -- lifecycle ---------------------------------------------------------

    async def close(self) -> None:
        """Close the connection to the registry."""
        if self._client is None:
            return
        try:
            await self._client.aclose()
        except Exception:
            logger.warning(
                "the registry connection could not be closed", exc_info=True
            )

    # -- helpers -----------------------------------------------------------

    def _is_faq(self, stats: FaqStats) -> bool:
        """Whether the statistics meet the rule for a frequently asked question.

        Three conditions, all of them needed: asked often enough, served
        from the memory at least once, and still fresh.
        """
        if stats.asks < self.faq_settings.promote_asks:
            return False
        if stats.hits < self.faq_settings.promote_hits:
            return False
        age = time.time() - stats.last_asked_at
        return age <= self.faq_settings.fresh_seconds

    async def _evict(self) -> None:
        """Drop the least asked questions once there are too many tracked.

        The questions are ranked by how often they are asked, and the ones
        nobody asks go first.  Their answers are left to expire on their
        own time to live - the eviction is about what the memory keeps
        *track of*, not about what it keeps.
        """
        limit = self.faq_settings.max_entries
        if limit <= 0:
            return

        try:
            count = int(await self.client.zcard(RANKED_KEY))
            excess = count - limit
            if excess <= 0:
                return

            victims = await self.client.zrange(RANKED_KEY, 0, excess - 1)
            if not victims:
                return
            for victim in victims:
                await self.client.delete(self._stat_key(victim))
            await self.client.zrem(RANKED_KEY, *victims)
        except Exception:
            logger.warning(
                "the frequently asked questions could not be trimmed", exc_info=True
            )

    @staticmethod
    def _stat_key(question_key: str) -> str:
        """The key the statistics of one question are kept under."""
        return f"{STAT_NAMESPACE}:{question_key}"

    @staticmethod
    def _to_stats(raw: dict[str, str]) -> FaqStats:
        """Read the statistics back out of their stored form."""
        return FaqStats(
            question=str(raw.get("question", "")),
            asks=int(raw.get("asks", 0) or 0),
            hits=int(raw.get("hits", 0) or 0),
            saved_ms=int(raw.get("saved_ms", 0) or 0),
            cost_ms=int(raw.get("cost_ms", 0) or 0),
            last_asked_at=float(raw.get("last_asked_at", 0.0) or 0.0),
        )


class NullFaqRegistry:
    """The registry that counts nothing.

    Used when the FAQ layer is switched off, so the rest of the pipeline
    does not have to know whether anything is being counted.
    """

    def __init__(self, cache_settings: CacheSettings | None = None) -> None:
        self.cache_settings = cache_settings or CacheSettings.from_env()

    async def record_ask(
        self,
        question: str,
        question_key: str,
        cost_ms: int = 0,
        hit: bool = False,
    ) -> FaqStats:
        return FaqStats(question=question)

    async def stats(self, question_key: str) -> FaqStats | None:
        return None

    async def faqs(self, limit: int = 50) -> list[FaqStats]:
        return []

    async def is_faq(self, question_key: str) -> bool:
        return False

    async def ttl_for(self, question_key: str) -> int:
        return self.cache_settings.ttl_seconds

    async def close(self) -> None:
        return None
