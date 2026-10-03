"""Tests for RAPHAEL SQLite memory architecture and persistence engine."""

import tempfile
from pathlib import Path

import pytest

from raphael.memory import ConversationTurn, MemoryItem, MemoryStore, MemoryType


@pytest.fixture
def temp_store():
    """Create a temporary MemoryStore backed by an isolated SQLite database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = tmp.name

    store = MemoryStore(db_path=tmp_path)
    yield store
    store.close()
    try:
        Path(tmp_path).unlink(missing_ok=True)
    except Exception:
        pass


def test_memory_crud_operations(temp_store: MemoryStore):
    """Verify inserting, retrieving, updating, and deleting memories."""
    # 1. Insert Fact
    item = MemoryItem(
        content="User's favorite game is Counter-Strike 2",
        memory_type=MemoryType.PREFERENCE,
        source="user_explicit",
        confidence=1.0,
        metadata={"category": "gaming"},
    )
    mem_id = temp_store.save_memory(item)
    assert mem_id is not None
    assert mem_id > 0
    assert temp_store.count_memories() == 1

    # 2. Retrieve
    retrieved = temp_store.get_memory(mem_id)
    assert retrieved is not None
    assert retrieved.content == "User's favorite game is Counter-Strike 2"
    assert retrieved.memory_type == MemoryType.PREFERENCE
    assert retrieved.metadata == {"category": "gaming"}

    # 3. Update
    retrieved.content = "User's favorite game is CS2 and Valorant"
    temp_store.save_memory(retrieved)
    updated = temp_store.get_memory(mem_id)
    assert updated.content == "User's favorite game is CS2 and Valorant"

    # 4. Delete
    deleted = temp_store.delete_memory(mem_id)
    assert deleted is True
    assert temp_store.get_memory(mem_id) is None
    assert temp_store.count_memories() == 0


def test_memory_search_and_filtering(temp_store: MemoryStore):
    """Verify searching memory by keyword and filtering by type."""
    temp_store.save_memory(
        MemoryItem(
            content="User prefers dark mode on all interfaces",
            memory_type=MemoryType.PREFERENCE,
        )
    )
    temp_store.save_memory(
        MemoryItem(
            content="Project RAPHAEL is a desktop voice assistant",
            memory_type=MemoryType.PROJECT,
        )
    )
    temp_store.save_memory(
        MemoryItem(
            content="User is running an NVIDIA GTX 1660 SUPER GPU",
            memory_type=MemoryType.FACT,
        )
    )

    # Search keyword across all types
    gpu_results = temp_store.search_memories("GPU")
    assert len(gpu_results) == 1
    assert "GTX 1660" in gpu_results[0].content

    # Search with type filter
    pref_results = temp_store.search_memories("mode", memory_type=MemoryType.PREFERENCE)
    assert len(pref_results) == 1
    assert "dark mode" in pref_results[0].content

    # Search without results
    empty_results = temp_store.search_memories("nonexistent_keyword_xyz")
    assert len(empty_results) == 0


def test_delete_by_pattern(temp_store: MemoryStore):
    """Verify deleting memories matching a pattern."""
    temp_store.save_memory(MemoryItem(content="Remember to buy milk"))
    temp_store.save_memory(MemoryItem(content="Remember to buy eggs"))
    temp_store.save_memory(MemoryItem(content="User name is Hexarion"))

    deleted_count = temp_store.delete_by_pattern("buy")
    assert deleted_count == 2
    assert temp_store.count_memories() == 1


@pytest.mark.parametrize("query", ["you", ""])
def test_recall_excludes_conversation_summaries_before_limit(temp_store, query):
    """A newer broad-match summary must not displace the user's saved fact."""
    fact = MemoryItem(content="You prefer patient explanations", memory_type=MemoryType.FACT)
    temp_store.save_memory(fact)
    archived = MemoryItem(
        content="You requested a cold commanding tone", memory_type=MemoryType.CONVERSATION,
        source="auto_summarizer", metadata={"session_id": "desktop_session"},
    )
    temp_store.save_memory(archived)

    recalled = temp_store.search_memories(
        query, limit=1, exclude_types=(MemoryType.CONVERSATION,),
    )

    assert [memory.id for memory in recalled] == [fact.id]
    assert temp_store.get_memory(archived.id).content == archived.content


def test_conversation_turns_persistence(temp_store: MemoryStore):
    """Verify storing and retrieving multi-turn conversation history."""
    session_a = "session_123"
    session_b = "session_456"

    # Add turns to session A
    temp_store.save_turn(
        ConversationTurn(session_id=session_a, role="user", content="Hello Raphael")
    )
    temp_store.save_turn(
        ConversationTurn(
            session_id=session_a,
            role="assistant",
            content="Hey there! How can I help?",
            provider="nim",
            model="nemotron",
            latency=0.5,
        )
    )
    temp_store.save_turn(
        ConversationTurn(session_id=session_a, role="user", content="What is my GPU?")
    )

    # Add turn to session B
    temp_store.save_turn(
        ConversationTurn(session_id=session_b, role="user", content="Testing other session")
    )

    # Retrieve session A turns (ordered chronologically)
    turns = temp_store.get_recent_turns(session_id=session_a, limit=10)
    assert len(turns) == 3
    assert turns[0].role == "user"
    assert turns[0].content == "Hello Raphael"
    assert turns[1].role == "assistant"
    assert turns[1].provider == "nim"
    assert turns[2].content == "What is my GPU?"

    # Verify limit parameter
    limited = temp_store.get_recent_turns(session_id=session_a, limit=2)
    assert len(limited) == 2
    assert limited[0].role == "assistant"
    assert limited[1].role == "user"

    # Clear session
    cleared = temp_store.clear_session(session_a)
    assert cleared == 3
    assert len(temp_store.get_recent_turns(session_id=session_a)) == 0
    assert len(temp_store.get_recent_turns(session_id=session_b)) == 1
