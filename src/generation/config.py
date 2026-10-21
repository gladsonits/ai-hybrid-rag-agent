"""Settings for the generation step.

The generation step talks to the LLM: which provider, which model, and how
long an answer may be.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

#: Default Groq model.  Fast and cheap, which is what a chat UI wants.
DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"


@dataclass(frozen=True, slots=True)
class LlmSettings:
    """Which LLM to call, and how to reach it."""

    api_key: str
    model: str
    temperature: float
    max_completion_tokens: int

    @classmethod
    def from_env(cls) -> "LlmSettings":
        return cls(
            api_key=os.getenv("GROQ_API_KEY", "").strip(),
            model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            max_completion_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
        )

    def redacted(self) -> str:
        """A loggable form with the key hidden."""
        return f"groq:{self.model}"
