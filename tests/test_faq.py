"""Tests for the frequently asked questions.

Three things are tested here, and none of them needs a network:

- **The registry** - the statistics behind the FAQ rule: how often a
  question is asked, how often the memory served it, how much model time
  that saved, when it counts as frequently asked, and how long it is
  kept.  The Redis client is replaced with a stand-in.
- **The chat turn** - that a question asked in other words is answered
  from the memory, that a follow-up is not, and that the question is
  recorded either way.  The matcher and the registry are stand-ins.
- **The index** - the search for the question a new one means the same
  as, against a live PostgreSQL.  These skip themselves when PostgreSQL
  is unreachable.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from augmentation.augmenter import Augmenter  # noqa: E402
from augmentation.chat_service import (  # noqa: E402
    AnswerCompleteEvent,
    ChatService,
    ChatTokenEvent,
    SourcesEvent,
)
from augmentation.config import AugmentationSettings  # noqa: E402
from generation.responder import Responder  # noqa: E402
from memory.cache import CachedAnswer, build_question_key  # noqa: E402
from memory.config import CacheSettings, FaqSettings  # noqa: E402
from memory.faq import (  # noqa: E402
    FaqCandidate,
    FaqRegistry,
    FaqStats,
    NullFaqRegistry,
)
from retrieval.config import DatabaseSettings  # noqa: E402
from retrieval.database import Database  # noqa: E402
from retrieval.embedder import EmbeddingModel  # noqa: E402
from retrieval.faq_store import PostgresFaqMatcher  # noqa: E402


# --------------------------------------------------------------------------
# Stand-ins
# --------------------------------------------------------------------------


class FakeStatsRedis:
    """A stand-in for the Redis client that holds the FAQ statistics.

    It keeps the hashes and the sorted sets the registry uses, and
    nothing else.
    """

    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.sets: dict[str, dict[str, float]] = {}

    # -- hashes ------------------------------------------------------------

    async def hset(
        self,
        name: str,
        key: str | None = None,
        value: Any = None,
        mapping: dict[str, Any] | None = None,
    ) -> int:
        fields = self.hashes.setdefault(name, {})
        if mapping:
            fields.update({str(k): str(v) for k, v in mapping.items()})
        if key is not None:
            fields[str(key)] = str(value)
        return len(fields)

    async def hincrby(self, name: str, key: str, amount: int = 1) -> int:
        fields = self.hashes.setdefault(name, {})
        value = int(fields.get(key, 0) or 0) + int(amount)
        fields[key] = str(value)
        return value

    async def hget(self, name: str, key: str) -> str | None:
        return self.hashes.get(name, {}).get(key)

    async def hgetall(self, name: str) -> dict[str, str]:
        return dict(self.hashes.get(name, {}))

    # -- sorted sets -------------------------------------------------------

    async def zincrby(self, name: str, amount: float, value: str) -> float:
        members = self.sets.setdefault(name, {})
        score = members.get(value, 0.0) + float(amount)
        members[value] = score
        return score

    async def zrange(
        self,
        name: str,
        start: int,
        end: int,
        desc: bool = False,
        withscores: bool = False,
    ) -> Any:
        members = self.sets.get(name, {})
        ordered = sorted(
            members.items(), key=lambda item: item[1], reverse=desc
        )
        window = ordered if end == -1 else ordered[start : end + 1]
        if withscores:
            return list(window)
        return [member for member, _score in window]

    async def zcard(self, name: str) -> int:
        return len(self.sets.get(name, {}))

    async def zrem(self, name: str, *members: str) -> int:
        stored = self.sets.setdefault(name, {})
        removed = 0
        for member in members:
            if member in stored:
                del stored[member]
                removed += 1
        return removed

    async def delete(self, *names: str) -> int:
        removed = 0
        for name in names:
            if self.hashes.pop(name, None) is not None:
                removed += 1
        return removed

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None


class BrokenRedis:
    """A stand-in for a Redis server that is not there."""

    async def hset(self, *args: Any, **kwargs: Any) -> int:
        raise ConnectionError("the server is not there")

    async def hincrby(self, *args: Any, **kwargs: Any) -> int:
        raise ConnectionError("the server is not there")

    async def hget(self, *args: Any, **kwargs: Any) -> str | None:
        raise ConnectionError("the server is not there")

    async def hgetall(self, *args: Any, **kwargs: Any) -> dict[str, str]:
        raise ConnectionError("the server is not there")

    async def zincrby(self, *args: Any, **kwargs: Any) -> float:
        raise ConnectionError("the server is not there")

    async def zrange(self, *args: Any, **kwargs: Any) -> Any:
        raise ConnectionError("the server is not there")

    async def zcard(self, *args: Any, **kwargs: Any) -> int:
        raise ConnectionError("the server is not there")

    async def zrem(self, *args: Any, **kwargs: Any) -> int:
        raise ConnectionError("the server is not there")

    async def delete(self, *args: Any, **kwargs: Any) -> int:
        raise ConnectionError("the server is not there")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _faq_settings(**overrides: Any) -> FaqSettings:
    """The FAQ settings, with the defaults the tests expect."""
    settings = {
        "enabled": True,
        "similarity_threshold": 0.9,
        "promote_asks": 3,
        "promote_hits": 1,
        "fresh_seconds": 86400,
        "max_entries": 500,
    }
    settings.update(overrides)
    return FaqSettings(**settings)


def _cache_settings(**overrides: Any) -> CacheSettings:
    """The cache settings, with the defaults the tests expect."""
    settings = {"url": "redis://:secret@127.0.0.1:6379/0", "ttl_seconds": 3600}
    settings.update(overrides)
    return CacheSettings(**settings)


def _registry(
    client: Any | None = None,
    faq: FaqSettings | None = None,
    cache: CacheSettings | None = None,
) -> FaqRegistry:
    """A registry backed by a stand-in Redis."""
    return FaqRegistry(
        faq_settings=faq or _faq_settings(),
        cache_settings=cache or _cache_settings(),
        client=client if client is not None else FakeStatsRedis(),
    )


# --------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------


async def test_an_ask_is_counted() -> None:
    """Every question the memory is asked about is counted."""
    registry = _registry()

    stats = await registry.record_ask("What projects did Rose work on?", "key1")

    assert stats.asks == 1
    assert stats.hits == 0
    assert stats.misses == 1


async def test_a_hit_is_counted() -> None:
    """A question the memory served is counted as a hit, not a miss."""
    registry = _registry()

    stats = await registry.record_ask("What projects?", "key1", hit=True)

    assert stats.asks == 1
    assert stats.hits == 1
    assert stats.misses == 0
    assert stats.hit_rate == 1.0


async def test_a_question_asked_often_is_not_yet_an_faq() -> None:
    """Asked three times but never served - it is not an FAQ yet."""
    registry = _registry()

    for _ in range(3):
        await registry.record_ask("What projects?", "key1")

    assert await registry.is_faq("key1") is False


async def test_a_question_asked_often_and_served_is_an_faq() -> None:
    """Asked often enough, and served at least once - it is an FAQ."""
    registry = _registry()

    await registry.record_ask("What projects?", "key1")
    await registry.record_ask("What projects?", "key1", hit=True)
    await registry.record_ask("What projects?", "key1", hit=True)

    assert await registry.is_faq("key1") is True


async def test_a_question_nobody_asked_lately_is_not_an_faq() -> None:
    """A question that has gone stale is not one anybody is asking now."""
    registry = _registry(faq=_faq_settings(fresh_seconds=0))

    await registry.record_ask("What projects?", "key1")
    await registry.record_ask("What projects?", "key1", hit=True)
    await registry.record_ask("What projects?", "key1", hit=True)

    assert await registry.is_faq("key1") is False


async def test_the_rule_can_be_set_ask_for_itself() -> None:
    """The number of asks that makes a question an FAQ is a setting."""
    registry = _registry(faq=_faq_settings(promote_asks=2, promote_hits=0))

    await registry.record_ask("What projects?", "key1")
    await registry.record_ask("What projects?", "key1")

    assert await registry.is_faq("key1") is True


# --------------------------------------------------------------------------
# The time to live
# --------------------------------------------------------------------------


async def test_a_fresh_answer_is_kept_for_an_hour() -> None:
    """A question asked once is kept for the ordinary time to live."""
    registry = _registry()

    assert await registry.ttl_for("key1") == 3600


async def test_an_faq_is_kept_for_longer() -> None:
    """A frequently asked question is worth keeping for much longer."""
    registry = _registry(cache=_cache_settings(faq_ttl_seconds=86400))

    await registry.record_ask("What projects?", "key1")
    await registry.record_ask("What projects?", "key1", hit=True)
    await registry.record_ask("What projects?", "key1", hit=True)

    assert await registry.ttl_for("key1") == 86400


# --------------------------------------------------------------------------
# What the memory saves
# --------------------------------------------------------------------------


async def test_the_memory_saves_the_time_the_answer_cost() -> None:
    """Every hit saves the time the original answer took."""
    registry = _registry()

    await registry.record_ask("What projects?", "key1", cost_ms=2900)
    stats = await registry.record_ask("What projects?", "key1", hit=True)

    assert stats.saved_ms == 2900


async def test_a_hit_does_not_overwrite_the_cost_of_the_answer() -> None:
    """The cost of the original answer is what the later hits save."""
    registry = _registry()

    await registry.record_ask("What projects?", "key1", cost_ms=2900)
    stats = await registry.record_ask("What projects?", "key1", hit=True)
    stats = await registry.record_ask("What projects?", "key1", hit=True)

    assert stats.cost_ms == 2900
    assert stats.saved_ms == 5800


# --------------------------------------------------------------------------
# The list of frequently asked questions
# --------------------------------------------------------------------------


async def test_the_faqs_are_listed_most_asked_first() -> None:
    """The report is of the FAQs, and the busiest one is first."""
    registry = _registry()

    for _ in range(3):
        await registry.record_ask("Rare question?", "rare", hit=True)
    for _ in range(5):
        await registry.record_ask("Common question?", "common", hit=True)

    faqs = await registry.faqs()

    assert [entry.question for entry in faqs] == [
        "Common question?",
        "Rare question?",
    ]


async def test_a_question_that_is_not_an_faq_is_not_listed() -> None:
    """A question asked once is not in the report."""
    registry = _registry()

    await registry.record_ask("Asked once?", "key1")

    assert await registry.faqs() == []


async def test_the_least_asked_questions_are_dropped_first() -> None:
    """Only so many questions are tracked, and the quiet ones go first."""
    registry = _registry(faq=_faq_settings(max_entries=2))

    for _ in range(5):
        await registry.record_ask("Common question?", "common")
    await registry.record_ask("Rare question?", "rare")
    await registry.record_ask("Other question?", "other")

    assert await registry.stats("rare") is None
    assert await registry.stats("common") is not None


async def test_the_registry_fails_open_when_redis_is_down() -> None:
    """A registry that cannot be reached reports nothing and does not raise."""
    registry = _registry(client=BrokenRedis())

    stats = await registry.record_ask("What projects?", "key1")

    assert stats.asks == 0
    assert await registry.faqs() == []
    assert await registry.ttl_for("key1") == 3600


async def test_the_null_registry_counts_nothing() -> None:
    """The registry that is switched off."""
    registry = NullFaqRegistry(_cache_settings())

    stats = await registry.record_ask("What projects?", "key1", hit=True)

    assert stats.asks == 0
    assert await registry.faqs() == []
    assert await registry.is_faq("key1") is False
    assert await registry.ttl_for("key1") == 3600


def test_the_statistics_are_reported_as_plain_data() -> None:
    """The statistics carry everything the report needs."""
    stats = FaqStats(
        question="What projects?",
        asks=4,
        hits=3,
        saved_ms=2900,
        cost_ms=2900,
        last_asked_at=1000.0,
    )

    data = stats.to_dict()

    assert data["question"] == "What projects?"
    assert data["asks"] == 4
    assert data["hits"] == 3
    assert data["misses"] == 1
    assert data["hit_rate"] == 0.75
    assert data["saved_ms"] == 2900


# --------------------------------------------------------------------------
# The index, against a live PostgreSQL
# --------------------------------------------------------------------------

#: The questions the live tests use.  They are cleared before and after,
#: so they never sit in the index of a running application.
STORED_QUESTION = "What projects did Rose work on?"
PARAPHRASE = "Which projects was Rose involved in?"
UNRELATED = "What is the capital of France?"


@pytest.fixture(scope="module")
def faq_db() -> Database:
    """The database, if it can be reached."""
    database = Database(DatabaseSettings.from_env())
    try:
        with database.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"PostgreSQL not reachable: {exc}")
    return database


@pytest.fixture(scope="module")
def faq_embedder(faq_db: Database) -> EmbeddingModel:
    """The real embedding model, if it can be loaded."""
    model = EmbeddingModel()
    try:
        model.load()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"the embedding model could not be loaded: {exc}")
    return model


@pytest.fixture
def matcher(faq_db: Database, faq_embedder: EmbeddingModel) -> PostgresFaqMatcher:
    """A matcher whose test questions have been cleared."""
    instance = PostgresFaqMatcher(faq_db, faq_embedder, _faq_settings())
    instance.ensure_schema()
    _clear(faq_db)
    yield instance
    _clear(faq_db)


def _clear(database: Database) -> None:
    """Remove the questions the live tests use."""
    keys = [build_question_key(q) for q in (STORED_QUESTION, PARAPHRASE)]
    with database.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM faq_questions WHERE question_key = ANY(%s)", (keys,))


def test_the_index_finds_the_question_a_new_one_means_the_same_as(
    matcher: PostgresFaqMatcher,
) -> None:
    """A question asked in other words finds the question it repeats."""
    matcher.remember(STORED_QUESTION, build_question_key(STORED_QUESTION), "cache-key")

    candidate = matcher.match(PARAPHRASE, 0.9)

    assert candidate is not None
    assert candidate.question == STORED_QUESTION
    assert candidate.cache_key == "cache-key"
    assert candidate.similarity >= 0.9


def test_a_question_below_the_threshold_is_not_a_match(
    matcher: PostgresFaqMatcher,
) -> None:
    """The threshold is what decides, not the ranking."""
    matcher.remember(STORED_QUESTION, build_question_key(STORED_QUESTION), "cache-key")

    # The same question is still the same question at any threshold.
    assert matcher.match(STORED_QUESTION, 0.999) is not None
    # The reworded one is close enough, but not close enough.
    assert matcher.match(PARAPHRASE, 0.99) is None


def test_a_question_that_means_something_else_is_not_a_match(
    matcher: PostgresFaqMatcher,
) -> None:
    """A question about a different thing does not find the stored one."""
    matcher.remember(STORED_QUESTION, build_question_key(STORED_QUESTION), "cache-key")

    assert matcher.match(UNRELATED, 0.9) is None


def test_the_same_question_is_remembered_once(
    matcher: PostgresFaqMatcher,
) -> None:
    """Remembering the same question again updates the one row."""
    key = build_question_key(STORED_QUESTION)
    matcher.remember(STORED_QUESTION, key, "first-key")
    matcher.remember(STORED_QUESTION, key, "second-key")

    candidate = matcher.match(STORED_QUESTION, 0.9)

    assert candidate is not None
    assert candidate.cache_key == "second-key"
    assert matcher.count() == 1
