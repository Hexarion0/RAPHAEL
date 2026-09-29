"""SQLite persistence engine for short-term conversation context and long-term memories."""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from raphael.logging import get_logger
from raphael.memory.models import ConversationTurn, MemoryItem, MemoryType

logger = get_logger("memory.store")


class MemoryStore:
    """Thread-safe SQLite storage for long-term facts, preferences, and session history."""

    def __init__(self, db_path: str | Path = "data/raphael.db") -> None:
        self.db_path_str = str(db_path)
        self._is_in_memory = self.db_path_str == ":memory:"

        if not self._is_in_memory:
            path_obj = Path(self.db_path_str)
            path_obj.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection with WAL mode and row factory."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                self.db_path_str,
                check_same_thread=False,
                timeout=10.0,
            )
            conn.row_factory = sqlite3.Row
            if not self._is_in_memory:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.execute("PRAGMA busy_timeout=5000;")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        """Create necessary tables and indices if they do not exist."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content TEXT NOT NULL,
                        memory_type TEXT NOT NULL,
                        source TEXT NOT NULL DEFAULT 'user_explicit',
                        confidence REAL NOT NULL DEFAULT 1.0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}'
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_turns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL DEFAULT 'default',
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        provider TEXT NOT NULL DEFAULT '',
                        model TEXT NOT NULL DEFAULT '',
                        tokens INTEGER NOT NULL DEFAULT 0,
                        latency REAL NOT NULL DEFAULT 0.0
                    );
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_conv_session ON conversation_turns(session_id, timestamp);"
                )
            logger.debug("Memory database initialized at %s", self.db_path_str)

    # -------------------------------------------------------------------------
    # Long-Term Memory Operations
    # -------------------------------------------------------------------------

    def save_memory(self, item: MemoryItem) -> int:
        """Insert or update a memory item. Returns the assigned memory ID."""
        with self._lock:
            conn = self._get_connection()
            now_iso = datetime.now(timezone.utc).isoformat()
            type_val = (
                item.memory_type.value
                if isinstance(item.memory_type, MemoryType)
                else str(item.memory_type)
            )
            meta_str = json.dumps(item.metadata)

            with conn:
                if item.id is not None:
                    # Update existing
                    cursor = conn.execute(
                        """
                        UPDATE memories
                        SET content = ?, memory_type = ?, source = ?, confidence = ?,
                            updated_at = ?, metadata_json = ?
                        WHERE id = ?;
                        """,
                        (
                            item.content,
                            type_val,
                            item.source,
                            item.confidence,
                            now_iso,
                            meta_str,
                            item.id,
                        ),
                    )
                    if cursor.rowcount > 0:
                        return item.id

                # Insert new
                cursor = conn.execute(
                    """
                    INSERT INTO memories (
                        content, memory_type, source, confidence,
                        created_at, updated_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        item.content,
                        type_val,
                        item.source,
                        item.confidence,
                        item.created_at.isoformat() if item.created_at else now_iso,
                        now_iso,
                        meta_str,
                    ),
                )
                item.id = cursor.lastrowid
                return item.id

    def get_memory(self, memory_id: int) -> MemoryItem | None:
        """Retrieve a specific memory by ID."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute("SELECT * FROM memories WHERE id = ?;", (memory_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_memory(row)

    def search_memories(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        """Search memory content using case-insensitive substring / keyword matching."""
        with self._lock:
            conn = self._get_connection()
            words = [w.strip() for w in query.split() if w.strip()]
            if not words:
                return self.list_memories(memory_type=memory_type, limit=limit)

            sql = "SELECT * FROM memories WHERE "
            conditions: list[str] = []
            params: list[Any] = []

            # Match any keyword in content
            content_clauses = ["content LIKE ?" for _ in words]
            conditions.append(f"({' OR '.join(content_clauses)})")
            params.extend([f"%{w}%" for w in words])

            if memory_type is not None:
                type_val = (
                    memory_type.value
                    if isinstance(memory_type, MemoryType)
                    else str(memory_type)
                )
                conditions.append("memory_type = ?")
                params.append(type_val)

            sql += " AND ".join(conditions)
            sql += " ORDER BY updated_at DESC LIMIT ?;"
            params.append(limit)

            cursor = conn.execute(sql, params)
            return [self._row_to_memory(r) for r in cursor.fetchall()]

    def list_memories(
        self,
        memory_type: MemoryType | None = None,
        limit: int = 50,
    ) -> list[MemoryItem]:
        """List the most recent memory items."""
        with self._lock:
            conn = self._get_connection()
            if memory_type is not None:
                type_val = (
                    memory_type.value
                    if isinstance(memory_type, MemoryType)
                    else str(memory_type)
                )
                cursor = conn.execute(
                    "SELECT * FROM memories WHERE memory_type = ? ORDER BY updated_at DESC LIMIT ?;",
                    (type_val, limit),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?;",
                    (limit,),
                )
            return [self._row_to_memory(r) for r in cursor.fetchall()]

    def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory item by ID."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                cursor = conn.execute("DELETE FROM memories WHERE id = ?;", (memory_id,))
                return cursor.rowcount > 0

    def delete_by_pattern(self, pattern: str) -> int:
        """Delete memories matching a search pattern (case-insensitive). Returns count deleted."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                cursor = conn.execute(
                    "DELETE FROM memories WHERE content LIKE ?;",
                    (f"%{pattern}%",),
                )
                return cursor.rowcount

    def count_memories(self, memory_type: MemoryType | None = None) -> int:
        """Return total count of memories stored."""
        with self._lock:
            conn = self._get_connection()
            if memory_type is not None:
                type_val = (
                    memory_type.value
                    if isinstance(memory_type, MemoryType)
                    else str(memory_type)
                )
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE memory_type = ?;",
                    (type_val,),
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM memories;")
            return cursor.fetchone()[0]

    # -------------------------------------------------------------------------
    # Short-Term Conversation History Operations
    # -------------------------------------------------------------------------

    def save_turn(self, turn: ConversationTurn) -> int:
        """Save a single conversation turn to history."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO conversation_turns (
                        session_id, role, content, timestamp,
                        provider, model, tokens, latency
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        turn.session_id,
                        turn.role,
                        turn.content,
                        turn.timestamp.isoformat()
                        if turn.timestamp
                        else datetime.now(timezone.utc).isoformat(),
                        turn.provider,
                        turn.model,
                        turn.tokens,
                        turn.latency,
                    ),
                )
                turn.id = cursor.lastrowid
                return turn.id

    def get_recent_turns(
        self,
        session_id: str = "default",
        limit: int = 10,
    ) -> list[ConversationTurn]:
        """Retrieve the N most recent conversation turns for a session in chronological order."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(
                """
                SELECT * FROM (
                    SELECT * FROM conversation_turns
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC;
                """,
                (session_id, limit),
            )
            return [self._row_to_turn(r) for r in cursor.fetchall()]

    def clear_session(self, session_id: str) -> int:
        """Clear all conversation turns for a given session ID."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                cursor = conn.execute(
                    "DELETE FROM conversation_turns WHERE session_id = ?;",
                    (session_id,),
                )
                return cursor.rowcount

    # -------------------------------------------------------------------------
    # Helper / Conversion Methods
    # -------------------------------------------------------------------------

    def _row_to_memory(self, row: sqlite3.Row) -> MemoryItem:
        meta = {}
        try:
            meta = json.loads(row["metadata_json"])
        except Exception:
            pass

        created_dt = datetime.now(timezone.utc)
        try:
            created_dt = datetime.fromisoformat(row["created_at"])
        except Exception:
            pass

        updated_dt = datetime.now(timezone.utc)
        try:
            updated_dt = datetime.fromisoformat(row["updated_at"])
        except Exception:
            pass

        return MemoryItem(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            source=row["source"],
            confidence=float(row["confidence"]),
            created_at=created_dt,
            updated_at=updated_dt,
            metadata=meta,
        )

    def _row_to_turn(self, row: sqlite3.Row) -> ConversationTurn:
        ts = datetime.now(timezone.utc)
        try:
            ts = datetime.fromisoformat(row["timestamp"])
        except Exception:
            pass

        return ConversationTurn(
            id=row["id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            timestamp=ts,
            provider=row["provider"],
            model=row["model"],
            tokens=row["tokens"],
            latency=float(row["latency"]),
        )

    def close(self) -> None:
        """Close thread-local database connection if active."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None
