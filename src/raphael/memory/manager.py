"""Conversation session manager, sliding context window, and rolling summarizer for RAPHAEL."""

import threading
from typing import Any

from raphael.config import get_settings
from raphael.logging import get_logger
from raphael.memory.models import ConversationTurn, MemoryItem, MemoryType
from raphael.memory.store import MemoryStore
from raphael.providers.base import ChatMessage

logger = get_logger("memory.manager")

SUMMARY_BATCH_TURNS = 32
SUMMARY_TURN_CHARS = 1000
SUMMARY_MAX_CHARS = 2000


class ConversationManager:
    """Manage persisted sessions, sliding context, and rolling summaries."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        session_id: str = "default",
        max_turns: int | None = None,
        auto_summarize_threshold: int = 14,
    ) -> None:
        settings = get_settings()
        self.store = store or MemoryStore(db_path=settings.memory.db_path)
        self.session_id = session_id
        self.max_turns = (
            max_turns if max_turns is not None else settings.memory.max_short_term_turns
        )
        self.auto_summarize_threshold = auto_summarize_threshold
        self._lock = threading.RLock()
        self._summary_lock = threading.Lock()
        self._generation = 0
        self._cached_summary: str | None = None
        self._summary_thread: threading.Thread | None = None

    def schedule_summary(self, router_or_provider: Any = None) -> bool:
        """Schedule occasional summaries without delaying voice replies."""
        with self._lock:
            if self._summary_thread and self._summary_thread.is_alive():
                return False
            count = self.get_total_turn_count()
            if count <= self.auto_summarize_threshold:
                return False
            summary = self.store.get_session_summary(self.session_id)
            after_id = summary.metadata.get("last_turn_id", 0) if summary else 0
            if not self.store.get_summary_batch(self.session_id, after_id, self.max_turns, 1):
                return False

            def summarize() -> None:
                try:
                    self.summarize_older_turns(router_or_provider)
                except Exception as err:
                    logger.warning("Background summary failed: %s", err)
                finally:
                    if not self.store._is_in_memory:
                        self.store.close()

            self._summary_thread = threading.Thread(target=summarize, daemon=True)
            self._summary_thread.start()
            return True

    def add_turn(
        self,
        role: str,
        content: str,
        provider: str = "",
        model: str = "",
        latency: float = 0.0,
        tokens: int = 0,
    ) -> int:
        """Persist a conversation turn to SQLite and return the turn ID."""
        turn = ConversationTurn(
            session_id=self.session_id,
            role=role,
            content=content.strip(),
            provider=provider,
            model=model,
            latency=latency,
            tokens=tokens,
        )
        turn_id = self.store.save_turn(turn)
        logger.debug(
            "Saved %s turn #%d to session '%s' (%d chars)",
            role,
            turn_id,
            self.session_id,
            len(content),
        )
        return turn_id

    def get_recent_turns(self, limit: int | None = None) -> list[ConversationTurn]:
        """Retrieve recent conversation turns from SQLite in chronological order."""
        lim = limit or self.max_turns
        return self.store.get_recent_turns(session_id=self.session_id, limit=lim)

    def get_total_turn_count(self) -> int:
        """Return total number of recorded turns for the current session."""
        with self.store._lock:
            conn = self.store._get_connection()
            cursor = conn.execute(
                "SELECT COUNT(*) FROM conversation_turns WHERE session_id = ?;",
                (self.session_id,),
            )
            return cursor.fetchone()[0]

    def get_summary(self) -> str | None:
        """Retrieve the latest running summary for the active session if available."""
        with self._lock:
            summary = self.store.get_session_summary(self.session_id)
            self._cached_summary = summary.content if summary else None
            return self._cached_summary

    def get_active_messages(
        self,
        system_prompt: str,
        limit: int | None = None,
    ) -> list[ChatMessage]:
        """Build messages with system persona, summary, and recent turns."""
        messages: list[ChatMessage] = [ChatMessage(role="system", content=system_prompt)]

        # Inject conversation summary of older turns if available
        summary = self.get_summary()
        if summary:
            messages.append(
                ChatMessage(
                    role="system",
                    content=f"PREVIOUS CONVERSATION CONTEXT (Earlier turns summarized):\n{summary}",
                )
            )

        # Retrieve sliding window of recent turns
        recent_turns = self.get_recent_turns(limit=limit)
        for t in recent_turns:
            messages.append(ChatMessage(role=t.role, content=t.content))

        return messages

    def summarize_older_turns(self, router_or_provider: Any = None) -> str | None:
        """Merge a bounded batch of new older turns into the persisted running summary."""
        with self._summary_lock:
            return self._summarize_batch(router_or_provider)

    def _summarize_batch(self, router_or_provider: Any) -> str | None:
        """Serialize summarizers while allowing replies and clearing during inference."""
        with self._lock:
            if self.get_total_turn_count() <= self.auto_summarize_threshold:
                return None
            generation = self._generation
            previous = self.store.get_session_summary(self.session_id)
            after_id = previous.metadata.get("last_turn_id", 0) if previous else 0
            older_turns = self.store.get_summary_batch(
                self.session_id, after_id, self.max_turns, SUMMARY_BATCH_TURNS
            )
        if not older_turns:
            return None

        previous_text = previous.content[:SUMMARY_MAX_CHARS] if previous else ""
        transcript = "\n".join(
            f"{turn.role.capitalize()}: {turn.content[:SUMMARY_TURN_CHARS]}" for turn in older_turns
        )
        summary_text = ""
        if router_or_provider is not None:
            try:
                response = router_or_provider.send(
                    [
                        ChatMessage(
                            role="system",
                            content=(
                                "Update the previous conversation summary with the new dialogue. "
                                "Keep key topics, facts, and decisions in 1 to 2 clear sentences. "
                                "Provide only the concise factual summary."
                            ),
                        ),
                        ChatMessage(
                            role="user",
                            content=(
                                f"Previous summary:\n{previous_text}\nNew dialogue:\n{transcript}"
                            ),
                        ),
                    ],
                    temperature=0.3,
                    max_tokens=150,
                )
                summary_text = response.content.strip()[:SUMMARY_MAX_CHARS]
            except Exception as err:
                logger.warning("LLM summarization failed (%s); using fallback.", err)
        if not summary_text:
            topics = "; ".join(turn.content[:100] for turn in older_turns[:3])
            summary_text = f"{previous_text} Previously discussed: {topics}.".strip()
            summary_text = summary_text[-SUMMARY_MAX_CHARS:]

        with self._lock:
            if generation != self._generation:
                return None
            self.store.save_memory(
                MemoryItem(
                    id=previous.id if previous else None,
                    content=summary_text,
                    memory_type=MemoryType.CONVERSATION,
                    source="auto_summarizer",
                    metadata={
                        "session_id": self.session_id,
                        "last_turn_id": older_turns[-1].id,
                        "older_turns_count": (
                            previous.metadata.get("older_turns_count", 0) if previous else 0
                        )
                        + len(older_turns),
                    },
                )
            )
            self._cached_summary = summary_text
        logger.info(
            "Summarized %d new older turns for session '%s'", len(older_turns), self.session_id
        )
        return summary_text

    def clear_session(self) -> int:
        """Clear persisted context and invalidate any summary currently being generated."""
        with self._lock:
            cleared = self.store.clear_session(self.session_id)
            self._generation += 1
            self._cached_summary = None
        logger.info("Cleared %d turns for session '%s'", cleared, self.session_id)
        return cleared
