"""Runtime configuration for the chat API.

Read from the environment (see ``.env``) so that the API key never lives in
the source tree.  The settings are grouped by concern, matching the style of
``retrieval.config``.
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


@dataclass(frozen=True, slots=True)
class ChatSettings:
    """How the retrieval-and-answer loop behaves."""

    #: How many chunks to pull from the index and put in front of the model.
    top_k: int
    #: How many past turns of the conversation to replay to the model.  The
    #: window is counted in turns, so one turn is a user message and the
    #: answer that came back for it.
    history_turns: int
    #: How many past user messages to fold into the retrieval query, so a
    #: follow-up like "and the second one?" still finds the right chunks.
    query_history_messages: int

    @classmethod
    def from_env(cls) -> "ChatSettings":
        return cls(
            top_k=int(os.getenv("CHAT_TOP_K", "5")),
            history_turns=int(os.getenv("CHAT_HISTORY_TURNS", "5")),
            query_history_messages=int(os.getenv(
                "CHAT_QUERY_HISTORY_MESSAGES", "2"
            )),
        )
