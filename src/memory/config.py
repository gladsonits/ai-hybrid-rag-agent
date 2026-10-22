"""Settings for the distributed memory.

The memory is a Redis server - the one the Docker Compose file starts.
Everything is read from the environment, so the location of the server
and its credentials never live in the source tree.

Either set ``REDIS_URL`` and the whole thing collapses to that, or set
the individual ``REDIS_*`` variables.  This mirrors how the PostgreSQL
settings work.

There are two sets of settings, because there are two things to decide:

- :class:`CacheSettings` - where the memory lives, and how long it keeps
  an answer.  A fresh answer is kept for an hour; a question that turns
  out to be asked often is kept for longer, because it is worth keeping.
- :class:`FaqSettings` - what makes a question a *frequently asked
  question*.  This is the rule the whole FAQ feature turns on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

#: The default port a Redis server listens on.
DEFAULT_REDIS_PORT = 6379

#: The default database number.
DEFAULT_REDIS_DB = 0

#: The default password.  The Docker Compose file falls back to the same
#: value, so the two agree whether or not the variable is set.
DEFAULT_REDIS_PASSWORD = "ragredis"

#: How long a fresh answer is kept, in seconds.  One hour.
DEFAULT_TTL_SECONDS = 3600

#: How long an answer for a frequently asked question is kept, in
#: seconds.  A day - a question that keeps being asked is worth keeping
#: around far longer than one that was asked once.
DEFAULT_FAQ_TTL_SECONDS = 86400

#: The cosine similarity two questions have to reach before they are
#: treated as the same question.  High on purpose: a wrong answer to a
#: question that only looks similar is worse than a slow answer.
DEFAULT_SIMILARITY_THRESHOLD = 0.90

#: How many times a question has to be asked before it counts as
#: frequently asked.
DEFAULT_PROMOTE_ASKS = 3

#: How many times the memory has to have served it.  A question that was
#: asked three times and answered three times by the model is a question
#: nobody has asked twice, and is not an FAQ yet.
DEFAULT_PROMOTE_HITS = 1

#: How long a question stays fresh, in seconds.  A day.  A question that
#: has not been asked in a week is not one anybody is asking now.
DEFAULT_FRESH_SECONDS = 86400

#: How many questions the registry keeps track of.
DEFAULT_MAX_ENTRIES = 500


def _as_bool(value: str, default: bool) -> bool:
    """Read a boolean out of the environment."""
    text = value.strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class CacheSettings:
    """Where the memory lives, and how long it keeps an answer."""

    #: The connection URL for the Redis client.
    url: str
    #: How long a fresh answer is kept, in seconds.
    ttl_seconds: int = DEFAULT_TTL_SECONDS
    #: How long the answer to a frequently asked question is kept.
    faq_ttl_seconds: int = DEFAULT_FAQ_TTL_SECONDS

    @classmethod
    def from_env(cls) -> "CacheSettings":
        """Build the settings from ``REDIS_URL`` or the ``REDIS_*`` vars."""
        url = os.getenv("REDIS_URL", "").strip()
        if not url:
            password = (
                os.getenv("REDIS_PASSWORD", "").strip() or DEFAULT_REDIS_PASSWORD
            )
            host = os.getenv("REDIS_HOST", "127.0.0.1")
            port = int(os.getenv("REDIS_PORT", str(DEFAULT_REDIS_PORT)))
            db = int(os.getenv("REDIS_DB", str(DEFAULT_REDIS_DB)))
            url = f"redis://:{password}@{host}:{port}/{db}"

        return cls(
            url=url,
            ttl_seconds=int(
                os.getenv("CACHE_TTL_SECONDS", str(DEFAULT_TTL_SECONDS))
            ),
            faq_ttl_seconds=int(
                os.getenv("FAQ_TTL_SECONDS", str(DEFAULT_FAQ_TTL_SECONDS))
            ),
        )

    def redacted(self) -> str:
        """A form with the password hidden, safe to log."""
        if "@" not in self.url:
            return self.url
        scheme, _, rest = self.url.partition("://")
        _, _, host = rest.rpartition("@")
        return f"{scheme}://:***@{host}"


@dataclass(frozen=True, slots=True)
class FaqSettings:
    """What makes a question a frequently asked question.

    A question is an FAQ when it has been asked often enough, when the
    memory has actually served it, and when it is still being asked.  The
    three conditions together are the whole rule:

    - **asked often enough** - one question asked twice is a coincidence,
      not a pattern.
    - **served from the memory** - the point of remembering a question is
      that the same question comes back.  A question that never repeats
      was never a FAQ.
    - **still fresh** - a question nobody has asked in a week is not one
      anybody is asking now, however often it used to be asked.
    """

    #: Whether the FAQ layer is on at all.
    enabled: bool = True
    #: The cosine similarity two questions have to reach to be the same.
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    #: How many times it has to be asked.
    promote_asks: int = DEFAULT_PROMOTE_ASKS
    #: How many times the memory has to have served it.
    promote_hits: int = DEFAULT_PROMOTE_HITS
    #: How long it stays fresh, in seconds.
    fresh_seconds: int = DEFAULT_FRESH_SECONDS
    #: How many questions are tracked at once.
    max_entries: int = DEFAULT_MAX_ENTRIES

    @classmethod
    def from_env(cls) -> "FaqSettings":
        """Build the settings from the ``FAQ_*`` variables."""
        return cls(
            enabled=_as_bool(os.getenv("FAQ_ENABLED", ""), True),
            similarity_threshold=float(
                os.getenv(
                    "FAQ_SIMILARITY_THRESHOLD",
                    str(DEFAULT_SIMILARITY_THRESHOLD),
                )
            ),
            promote_asks=int(
                os.getenv("FAQ_PROMOTE_ASKS", str(DEFAULT_PROMOTE_ASKS))
            ),
            promote_hits=int(
                os.getenv("FAQ_PROMOTE_HITS", str(DEFAULT_PROMOTE_HITS))
            ),
            fresh_seconds=int(
                os.getenv("FAQ_FRESH_SECONDS", str(DEFAULT_FRESH_SECONDS))
            ),
            max_entries=int(
                os.getenv("FAQ_MAX_ENTRIES", str(DEFAULT_MAX_ENTRIES))
            ),
        )
