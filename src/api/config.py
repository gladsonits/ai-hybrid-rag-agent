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


@dataclass(frozen=True, slots=True)
class ApiSettings:
    """Where the API listens."""

    host: str
    port: int

    @classmethod
    def from_env(cls) -> "ApiSettings":
        return cls(
            host=os.getenv("API_HOST", "127.0.0.1"),
            port=int(os.getenv("API_PORT", "8000")),
        )
