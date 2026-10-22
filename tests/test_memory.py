"""Tests for the distributed memory.

These tests do not need a Redis server.  The client is replaced with a
stand-in, so the key derivation, the stored form of an answer, the time
to live and the behaviour when Redis is down are all exercised without a
network call.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from memory.cache import (  # noqa: E402
    CACHE_NAMESPACE,
    CachedAnswer,
    NullAnswerCache,
    RedisAnswerCache,
    build_cache_key,
)
from memory.config import (  # noqa: E402
    DEFAULT_TTL_SECONDS,
    CacheSettings,
)


class FakeRedis:
    """A stand-in for the Redis client."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.closed = False

    async def get(self, key: str) -> str | None:
        return self.data.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.data[key] = value
        if ex is not None:
            self.ttls[key] = ex

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        self.closed = True


class BrokenRedis:
    """A stand-in for a Redis server that is not there."""

    async def get(self, key: str) -> str | None:
        raise ConnectionError("the server is not there")

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        raise ConnectionError("the server is not there")

    async def ping(self) -> bool:
        raise ConnectionError("the server is not there")

    async def aclose(self) -> None:
        return None


# --------------------------------------------------------------------------
# The key
# --------------------------------------------------------------------------


def test_the_key_ignores_case_and_spacing() -> None:
    """The same question is the same key, however it is typed."""
    first = build_cache_key("What projects did Rose work on?")
    second = build_cache_key("  what PROJECTS   did rose work on?  ")

    assert first == second


def test_the_key_is_namespaced() -> None:
    """The keys sit under a namespace, so the memory can be shared."""
    key = build_cache_key("What projects did Rose work on?")

    assert key.startswith(CACHE_NAMESPACE + ":")


def test_the_key_tells_different_contexts_apart() -> None:
    """The same question in a different conversation is a different key."""
    first = build_cache_key("What was my role?", "earlier question")
    second = build_cache_key("What was my role?", "another earlier question")

    assert first != second


def test_a_different_question_is_a_different_key() -> None:
    """Two questions never share a key."""
    first = build_cache_key("What projects did Rose work on?")
    second = build_cache_key("What projects did she work on?")

    assert first != second


# --------------------------------------------------------------------------
# The stored answer
# --------------------------------------------------------------------------


def test_an_answer_survives_the_round_trip() -> None:
    """What goes in comes back out."""
    answer = CachedAnswer(
        answer="Rose worked on Trustage [1].",
        sources=[{"chunk_id": 517, "metadata": {"project": "TruStage"}}],
        citations=[{"marker": 1, "chunk_id": 517}],
    )

    restored = CachedAnswer.from_json(answer.to_json())

    assert restored == answer


def test_an_empty_answer_still_round_trips() -> None:
    """An answer with nothing in it is still an answer."""
    restored = CachedAnswer.from_json(CachedAnswer().to_json())

    assert restored.answer == ""
    assert restored.sources == []


# --------------------------------------------------------------------------
# The cache
# --------------------------------------------------------------------------


async def test_the_cache_stores_an_answer_for_an_hour() -> None:
    """The answer is written with the time to live from the settings."""
    redis = FakeRedis()
    settings = CacheSettings(url="redis://:secret@127.0.0.1:6379/0", ttl_seconds=3600)
    cache = RedisAnswerCache(settings, client=redis)
    answer = CachedAnswer(answer="Hello world")

    await cache.set("rag:faq:abc", answer)

    assert redis.ttls["rag:faq:abc"] == 3600
    assert await cache.get("rag:faq:abc") == answer


async def test_the_cache_misses_when_the_key_is_unknown() -> None:
    """A question that was never asked has no answer."""
    cache = RedisAnswerCache(
        CacheSettings(url="redis://:secret@127.0.0.1:6379/0"), client=FakeRedis()
    )

    assert await cache.get("rag:faq:missing") is None


async def test_the_cache_fails_open_when_redis_is_down() -> None:
    """A memory that cannot be reached is a miss, not an error."""
    cache = RedisAnswerCache(
        CacheSettings(url="redis://:secret@127.0.0.1:6379/0"), client=BrokenRedis()
    )

    assert await cache.get("rag:faq:abc") is None
    assert await cache.ping() is False

    # A write that cannot be made is dropped, and does not raise.
    await cache.set("rag:faq:abc", CachedAnswer(answer="Hello"))


async def test_the_cache_ignores_something_that_is_not_an_answer() -> None:
    """A value that is not an answer is treated as a miss."""
    redis = FakeRedis()
    redis.data["rag:faq:abc"] = "not json"
    cache = RedisAnswerCache(
        CacheSettings(url="redis://:secret@127.0.0.1:6379/0"), client=redis
    )

    assert await cache.get("rag:faq:abc") is None


async def test_the_cache_closes_its_connection() -> None:
    """The connection is released when the application stops."""
    redis = FakeRedis()
    cache = RedisAnswerCache(
        CacheSettings(url="redis://:secret@127.0.0.1:6379/0"), client=redis
    )

    await cache.close()

    assert redis.closed is True


async def test_the_null_cache_remembers_nothing() -> None:
    """The cache that is switched off."""
    cache = NullAnswerCache()

    await cache.set("rag:faq:abc", CachedAnswer(answer="Hello"))

    assert await cache.get("rag:faq:abc") is None
    assert await cache.ping() is False


# --------------------------------------------------------------------------
# The settings
# --------------------------------------------------------------------------


def test_the_settings_come_from_the_environment(monkeypatch: Any) -> None:
    """The location and the password are read from the environment."""
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("REDIS_HOST", "cache.internal")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_PASSWORD", "s3cret")
    monkeypatch.setenv("REDIS_DB", "2")

    settings = CacheSettings.from_env()

    assert settings.url == "redis://:s3cret@cache.internal:6380/2"


def test_an_explicit_url_wins(monkeypatch: Any) -> None:
    """REDIS_URL overrides the individual variables."""
    monkeypatch.setenv("REDIS_URL", "redis://elsewhere:6379/1")
    monkeypatch.setenv("REDIS_HOST", "ignored")

    settings = CacheSettings.from_env()

    assert settings.url == "redis://elsewhere:6379/1"


def test_the_answer_is_kept_for_an_hour(monkeypatch: Any) -> None:
    """The time to live is an hour by default."""
    monkeypatch.delenv("CACHE_TTL_SECONDS", raising=False)

    settings = CacheSettings.from_env()

    assert settings.ttl_seconds == DEFAULT_TTL_SECONDS
    assert settings.ttl_seconds == 3600


def test_the_password_is_hidden_when_the_settings_are_logged() -> None:
    """The settings can be logged without leaking the password."""
    settings = CacheSettings(url="redis://:s3cret@127.0.0.1:6379/0", ttl_seconds=3600)

    assert "s3cret" not in settings.redacted()
    assert "127.0.0.1" in settings.redacted()
