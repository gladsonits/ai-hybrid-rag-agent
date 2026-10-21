"""Per-session conversation history, stored in PostgreSQL.

The chat endpoint has to remember what was said earlier in a session so the
model can answer follow-up questions.  The history is kept in the same
database as the chunk index, in two tables:

- ``conversations`` - one row per session, keyed by a client-supplied id.
- ``messages`` - one row per turn, pointing back at its conversation.

Storing history in PostgreSQL rather than in the process means the history
survives a restart of the API and works unchanged when more than one worker
serves the traffic, which an in-process dict cannot do.

The store owns its own schema - it does not touch the ``chunks`` table or the
retrieval schema, and ``retrieval`` knows nothing about it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from retrieval.database import Database

#: Schema for the conversation tables.  Both are created together because
#: ``messages`` has a foreign key into ``conversations``.
SESSION_SCHEMA_DDL = (
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id          TEXT PRIMARY KEY,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id              BIGSERIAL PRIMARY KEY,
        conversation_id TEXT NOT NULL
                        REFERENCES conversations (id) ON DELETE CASCADE,
        role            TEXT NOT NULL,
        content         TEXT NOT NULL,
        sources         JSONB NOT NULL DEFAULT '[]'::jsonb,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS messages_conversation_idx "
    "ON messages (conversation_id, id);",
)

#: The roles a stored message can have.
ROLES = ("user", "assistant")


@dataclass(frozen=True, slots=True)
class StoredMessage:
    """One message in a conversation."""

    role: str
    content: str
    #: The chunks cited by an assistant message.  Empty for a user message.
    sources: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content, "sources": self.sources}


class SessionStore:
    """Reads and writes conversation history."""

    def __init__(self, database: Database) -> None:
        self.db = database

    # -- schema ------------------------------------------------------------

    def ensure_schema(self) -> None:
        """Create the conversation tables. Safe to call repeatedly."""
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                for statement in SESSION_SCHEMA_DDL:
                    cur.execute(statement)

    # -- writing -----------------------------------------------------------

    def ensure_conversation(self, conversation_id: str) -> None:
        """Make sure a conversation row exists for ``conversation_id``."""
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversations (id)
                    VALUES (%(id)s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    {"id": conversation_id},
                )

    def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> int:
        """Append one message to a conversation, returning its row id.

        The conversation row is created on demand, so a caller can start
        talking without a separate "create session" call.
        """
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversations (id)
                    VALUES (%(id)s)
                    ON CONFLICT (id) DO UPDATE SET updated_at = now()
                    """,
                    {"id": conversation_id},
                )
                cur.execute(
                    """
                    INSERT INTO messages (conversation_id, role, content, sources)
                    VALUES (%(conversation_id)s, %(role)s, %(content)s, %(sources)s)
                    RETURNING id
                    """,
                    {
                        "conversation_id": conversation_id,
                        "role": role,
                        "content": content,
                        "sources": json.dumps(sources or []),
                    },
                )
                row = cur.fetchone()
                cur.execute(
                    "UPDATE conversations SET updated_at = now() WHERE id = %(id)s",
                    {"id": conversation_id},
                )
                return int(row["id"])

    # -- reading -----------------------------------------------------------

    def history(
        self,
        conversation_id: str,
        limit: int | None = None,
    ) -> list[StoredMessage]:
        """Return the conversation's messages, oldest first.

        ``limit`` keeps only the most recent ``limit`` messages, which is
        what the prompt builder uses to bound the replay window.
        """
        query = """
            SELECT role, content, sources
            FROM messages
            WHERE conversation_id = %(conversation_id)s
            ORDER BY id DESC
        """
        params: dict[str, Any] = {"conversation_id": conversation_id}
        if limit is not None:
            query += " LIMIT %(limit)s"
            params["limit"] = limit

        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

        # The query sorts newest-first so a LIMIT keeps the newest messages;
        # flip it back to chronological order for the prompt.
        messages = [self._to_message(row) for row in rows]
        messages.reverse()
        return messages

    def clear(self, conversation_id: str) -> None:
        """Delete a conversation and all of its messages."""
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM messages WHERE conversation_id = %s",
                    (conversation_id,),
                )
                cur.execute(
                    "DELETE FROM conversations WHERE id = %s", (conversation_id,)
                )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _to_message(row: dict[str, Any]) -> StoredMessage:
        """Convert a database row into a :class:`StoredMessage`."""
        sources = row.get("sources")
        if isinstance(sources, str):
            sources = json.loads(sources)
        return StoredMessage(
            role=row["role"],
            content=row["content"],
            sources=list(sources or []),
        )
