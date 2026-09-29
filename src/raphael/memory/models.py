"""Data models for RAPHAEL's memory and conversation persistence system."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    """Classification of stored memory items."""

    FACT = "fact"
    PREFERENCE = "preference"
    PROJECT = "project"
    CONVERSATION = "conversation"
    REMINDER = "reminder"


@dataclass
class MemoryItem:
    """A single persistent memory record stored in SQLite."""

    content: str
    memory_type: MemoryType = MemoryType.FACT
    source: str = "user_explicit"
    confidence: float = 1.0
    id: int | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert memory item to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value if isinstance(self.memory_type, MemoryType) else str(self.memory_type),
            "source": self.source,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ConversationTurn:
    """A single message turn in a multi-turn conversation session."""

    role: str
    content: str
    session_id: str = "default"
    id: int | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    provider: str = ""
    model: str = ""
    tokens: int = 0
    latency: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert conversation turn to dictionary representation."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "provider": self.provider,
            "model": self.model,
            "tokens": self.tokens,
            "latency": self.latency,
        }
