"""Conversation session manager, sliding context window, and rolling summarizer for RAPHAEL."""

import threading
from typing import Any

from raphael.config import get_settings
from raphael.logging import get_logger
from raphael.memory.models import ConversationTurn, MemoryItem, MemoryType
from raphael.memory.store import MemoryStore
from raphael.providers.base import ChatMessage

logger = get_logger("memory.manager")


class ConversationManager:
    """Manages active conversation sessions with SQLite persistence and rolling context summarization."""

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
            max_turns
            if max_turns is not None
            else settings.memory.max_short_term_turns
        )
        self.auto_summarize_threshold = auto_summarize_threshold
        self._lock = threading.Lock()
        self._cached_summary: str | None = None

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
        if self._cached_summary:
            return self._cached_summary

        summaries = self.store.search_memories(
            query=self.session_id,
            memory_type=MemoryType.CONVERSATION,
            limit=1,
        )
        if summaries:
            self._cached_summary = summaries[0].content
            return self._cached_summary
        return None

    def get_active_messages(
        self,
        system_prompt: str,
        limit: int | None = None,
    ) -> list[ChatMessage]:
        """Build the full ChatMessage list for the LLM including system persona, summary, and recent turns."""
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=system_prompt)
        ]

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
        """Summarize older turns when turn count exceeds threshold to maintain compact token usage."""
        total_turns = self.get_total_turn_count()
        if total_turns <= self.auto_summarize_threshold:
            return None

        # Retrieve older turns that will be pruned from the active window
        all_turns = self.store.get_recent_turns(
            session_id=self.session_id,
            limit=total_turns,
        )
        older_turns = all_turns[: -self.max_turns]
        if not older_turns:
            return None

        # Format transcript of older turns
        transcript_lines = [f"{t.role.capitalize()}: {t.content}" for t in older_turns]
        transcript_text = "\n".join(transcript_lines)

        summary_text = ""
        if router_or_provider is not None:
            try:
                summary_prompt = [
                    ChatMessage(
                        role="system",
                        content=(
                            "You are a concise summarizer. In 1 to 2 clear sentences, summarize the key topics, "
                            "decisions, and context discussed in the following dialogue transcript. "
                            "Do not use markdown, lists, or filler. Provide only the concise factual summary."
                        ),
                    ),
                    ChatMessage(
                        role="user",
                        content=f"Transcript to summarize:\n{transcript_text}",
                    ),
                ]
                resp = router_or_provider.send(
                    summary_prompt,
                    temperature=0.3,
                    max_tokens=150,
                )
                summary_text = resp.content.strip()
            except Exception as err:
                logger.warning("LLM summarization failed (%s) — using rule-based fallback.", err)

        if not summary_text:
            # Rule-based fallback summary if router/provider is offline or not passed
            topics = [t.content[:40] + "..." for t in older_turns[:3]]
            summary_text = f"Previously discussed topics included: {'; '.join(topics)}."

        # Save summary to SQLite
        self.store.save_memory(
            MemoryItem(
                content=summary_text,
                memory_type=MemoryType.CONVERSATION,
                source="auto_summarizer",
                metadata={"session_id": self.session_id, "older_turns_count": len(older_turns)},
            )
        )
        self._cached_summary = summary_text
        logger.info(
            "Summarized %d older turns for session '%s': '%s'",
            len(older_turns),
            self.session_id,
            summary_text,
        )
        return summary_text

    def clear_session(self) -> int:
        """Clear all turns and cached summary for the current session."""
        cleared = self.store.clear_session(self.session_id)
        self._cached_summary = None
        logger.info("Cleared %d turns for session '%s'", cleared, self.session_id)
        return cleared
