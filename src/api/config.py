"""Settings for the chat API itself.

The settings that shape the model's context - how many chunks to
retrieve, how much history to replay - live in
:mod:`augmentation.config`, because they belong to the augmentation step
rather than to the transport.  The LLM settings live in
:mod:`generation.config`, because they belong to the generation step.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

#: The origins the API answers a browser from when nothing else is set.
#: The Angular development server is the one that matters.
DEFAULT_CORS_ORIGINS = "http://localhost:4201, http://localhost:4200, httpL://127.0.0.1:*"


@dataclass(frozen=True, slots=True)
class ApiSettings:
    """Where the API listens, and who may call it from a browser."""

    host: str
    port: int
    cors_origins: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "ApiSettings":
        return cls(
            host=os.getenv("API_HOST", "127.0.0.1"),
            port=int(os.getenv("API_PORT", "8000")),
            cors_origins=parse_origins(
                os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
            ),
        )


def parse_origins(value: str) -> tuple[str, ...]:
    """Read a comma separated list of origins.

    An origin the API does not know is a browser that is refused, so
    this is a list and not a wildcard.
    """
    return tuple(
        origin.strip() for origin in value.split(",") if origin.strip() != ""
    )
