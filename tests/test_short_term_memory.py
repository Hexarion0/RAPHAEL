"""Tests for RAPHAEL short-term memory, sliding context window, and rolling summarization."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from raphael.memory import ConversationManager, MemoryStore
from raphael.providers.base import LLMResponse


@pytest.fixture
def temp_store():
    """Create an isolated SQLite memory store backed by a temp database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = tmp.name

    store = MemoryStore(db_path=tmp_path)
    yield store
    store.close()
    try:
        Path(tmp_path).unlink(missing_ok=True)
    except Exception:
        pass


def test_cross_restart_conversation_persistence(temp_store: MemoryStore):
    """Verify turns saved in session A are preserved and restored upon restarting manager."""
    # 1. First session run
    manager1 = ConversationManager(store=temp_store, session_id="test_session", max_turns=5)
    manager1.add_turn(role="user", content="Hello Raphael, my name is Hexarion")
    manager1.add_turn(role="assistant", content="Hey Hexarion! Great to meet you.")
    manager1.add_turn(role="user", content="What GPU do I have?")
    manager1.add_turn(role="assistant", content="You have a GTX 1660 SUPER.")

    # 2. Simulate assistant restart (new manager instance on same DB)
    manager2 = ConversationManager(store=temp_store, session_id="test_session", max_turns=5)
    turns = manager2.get_recent_turns()
    assert len(turns) == 4
    assert turns[0].role == "user"
    assert turns[0].content == "Hello Raphael, my name is Hexarion"
    assert turns[3].content == "You have a GTX 1660 SUPER."

    # Verify active messages structure
    messages = manager2.get_active_messages(system_prompt="System Persona")
    assert len(messages) == 5  # 1 system + 4 turns
    assert messages[0].role == "system"
    assert messages[0].content == "System Persona"
    assert messages[1].content == "Hello Raphael, my name is Hexarion"


def test_sliding_context_window_trimming(temp_store: MemoryStore):
    """Verify that when turn count exceeds max_turns, only the latest N turns are returned."""
    manager = ConversationManager(store=temp_store, session_id="overflow_session", max_turns=4)

    for i in range(10):
        manager.add_turn(role="user", content=f"User turn {i}")
        manager.add_turn(role="assistant", content=f"Assistant turn {i}")

    # Total 20 turns recorded in SQLite
    assert manager.get_total_turn_count() == 20

    # Active messages should only contain the system prompt + latest 4 turns
    messages = manager.get_active_messages(system_prompt="Base System")
    assert len(messages) == 5  # 1 system + 4 turns
    assert messages[1].content == "User turn 8"
    assert messages[2].content == "Assistant turn 8"
    assert messages[3].content == "User turn 9"
    assert messages[4].content == "Assistant turn 9"


def test_rolling_summarization(temp_store: MemoryStore):
    """Verify older turns are summarized and injected as background context."""
    manager = ConversationManager(
        store=temp_store,
        session_id="summarize_session",
        max_turns=2,
        auto_summarize_threshold=4,
    )

    # Add 6 turns (exceeds threshold of 4)
    manager.add_turn(role="user", content="We are planning to build a desktop assistant in Python.")
    manager.add_turn(role="assistant", content="Awesome, I recommend using faster-whisper and sounddevice.")
    manager.add_turn(role="user", content="Let's make sure it runs on Arch Linux with Hyprland.")
    manager.add_turn(role="assistant", content="Got it, we will configure Linux audio backend.")
    manager.add_turn(role="user", content="What is our current task?")
    manager.add_turn(role="assistant", content="We are implementing the short term memory module.")

    # Mock LLM router for summarizer
    mock_router = MagicMock()
    mock_router.send.return_value = LLMResponse(
        content="User and assistant planned a Python voice assistant on Arch Linux with Hyprland.",
        model="test_model",
        provider="nim",
    )

    summary = manager.summarize_older_turns(router_or_provider=mock_router)
    assert summary is not None
    assert "Arch Linux" in summary

    # Verify summary is injected into active messages
    active_msgs = manager.get_active_messages(system_prompt="Base Persona")
    # Expected: [System Persona, Summary Context, Turn 5, Turn 6]
    assert len(active_msgs) == 4
    assert active_msgs[0].content == "Base Persona"
    assert "PREVIOUS CONVERSATION CONTEXT" in active_msgs[1].content
    assert "Arch Linux" in active_msgs[1].content
    assert active_msgs[2].content == "What is our current task?"
    assert active_msgs[3].content == "We are implementing the short term memory module."


def test_clear_session(temp_store: MemoryStore):
    """Verify clearing session removes turns and summary."""
    manager = ConversationManager(store=temp_store, session_id="clear_me", max_turns=5)
    manager.add_turn(role="user", content="Some test text")
    manager.add_turn(role="assistant", content="Some reply")
    assert manager.get_total_turn_count() == 2

    cleared = manager.clear_session()
    assert cleared == 2
    assert manager.get_total_turn_count() == 0
    assert len(manager.get_active_messages(system_prompt="Base")) == 1
