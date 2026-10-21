"""Runtime configuration for the PostgreSQL-backed indexes.

Every value is read from the environment (see ``.env``) so that credentials
never live in the source tree.  The module exposes one frozen
:class:`Settings` object per concern rather than a grab-bag of module-level
constants, so a caller can pass an explicit object into the stores instead of
relying on ambient state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

#: Default database name created for this project.
DEFAULT_DB_NAME = "rag_agent"

#: Default embedding model. 384 dimensions, small enough to run on CPU.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True, slots=True)
class DatabaseSettings:
    """Everything needed to open a connection to PostgreSQL.

    Either set ``DATABASE_URL`` and the whole thing collapses to that, or set
    the individual ``POSTGRES_*`` variables.
    """

    host: str
    port: int
    user: str
    password: str
    dbname: str

    @classmethod
    def from_env(cls) -> "DatabaseSettings":
        """Build settings from ``DATABASE_URL`` or the ``POSTGRES_*`` vars."""
        url = os.getenv("DATABASE_URL", "").strip()
        if url:
            return cls._from_url(url)
        return cls(
            host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
            dbname=os.getenv("POSTGRES_DB", DEFAULT_DB_NAME),
        )

    @staticmethod
    def _from_url(url: str) -> "DatabaseSettings":
        """Parse a ``postgresql://user:pass@host:port/db`` URL."""
        from urllib.parse import urlparse

        parsed = urlparse(url)
        return DatabaseSettings(
            host=parsed.hostname or "127.0.0.1",
            port=parsed.port or 5432,
            user=parsed.username or "postgres",
            password=parsed.password or "",
            dbname=(parsed.path or "/").lstrip("/") or DEFAULT_DB_NAME,
        )

    def conninfo(self) -> str:
        """Render the settings as a libpq ``conninfo`` string."""
        return (
            f"host={self.host} port={self.port} "
            f"dbname={self.dbname} user={self.user} password={self.password}"
        )

    def redacted(self) -> str:
        """A human-readable form with the password hidden, safe to log."""
        return f"postgresql://{self.user}@:{self.port}/{self.dbname}"


@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    """Which embedding model to use, and how to reach it."""

    model: str
    device: str
    batch_size: int
    normalize: bool
    max_seq_length: int
    hf_token: str

    @classmethod
    def from_env(cls) -> "EmbeddingSettings":
        return cls(
            model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            device=os.getenv("EMBEDDING_DEVICE", "cpu"),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
            normalize=os.getenv("EMBEDDING_NORMALIZE", "true").lower()
            in {"1", "true", "yes"},
            max_seq_length=int(os.getenv("EMBEDDING_MAX_SEQ_LENGTH", "512")),
            hf_token=os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_TOKEN", "")),
        )
