"""End-to-end local facts, corrections, recall, and forgetting without provider calls."""

from datetime import datetime

import pytest

from raphael.memory import ConversationManager, MemoryItem, MemoryStore, MemoryType
from raphael.memory.context import recall_context_memories
from raphael.memory.recall import rank_memories
from raphael.memory.service import MemoryService


@pytest.fixture
def store():
    memory = MemoryStore(":memory:")
    yield memory
    memory.close()


def test_natural_fact_and_correction_replace_one_current_record(store):
    service = MemoryService(store)
    assert "saved" in service.handle("My favorite game is CS2.")
    first = store.get_fact("user:favorite_game")
    assert service.handle("What's my favorite game?") == "Your favorite game is CS2."
    assert "updated" in service.handle("No, it's Valorant.")
    current = store.get_fact("user:favorite_game")
    assert current.id == first.id
    assert current.metadata["value"] == "Valorant"
    assert store.count_memories() == 1
    assert current.metadata["revisions"][0]["value"] == "CS2"
    assert "CS2" not in "\n".join(recall_context_memories(store, "games"))


def test_correction_and_profile_survive_restart(tmp_path):
    path = tmp_path / "memory.db"
    store = MemoryStore(path)
    service = MemoryService(store)
    service.handle("My favorite food is pizza.")
    service.handle("Actually, my favorite food is pasta.")
    service.handle("Call me hexarion.")
    store.close()
    reopened = MemoryStore(path)
    try:
        assert reopened.get_fact("user:favorite_food").metadata["value"] == "pasta"
        assert reopened.get_fact("user:preferred_name").metadata["value"] == "hexarion"
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "text",
    [
        "What if my favorite game is CS2?",
        "She said my favorite game is CS2.",
        "Maybe my favorite game is CS2",
        "Call me tomorrow",
        "I feel tired today",
    ],
)
def test_questions_other_people_and_ambiguous_statements_are_not_facts(store, text):
    assert MemoryService(store).handle(text) is None
    assert store.count_memories() == 0


def test_explicit_notes_are_deduplicated(store):
    service = MemoryService(store)
    service.handle("Remember that I own a red keyboard.")
    service.handle("Remember that I own a red keyboard.")
    assert store.count_memories() == 1


def test_start_date_correction_is_separate_from_first_commit(store):
    service = MemoryService(store)
    now = datetime(2026, 10, 3)
    service.handle("The project started September 28th.", now)
    service.handle("The first commit was September 29th.", now)
    assert store.get_fact("raphael:project_start_date").metadata["value"] == "2026-09-28"
    assert store.get_fact("raphael:first_commit_date").metadata["value"] == "2026-09-29"
    service.handle("What development day is it?", now)
    service.handle("No, September 27th.", now)
    assert store.get_fact("raphael:project_start_date").metadata["value"] == "2026-09-27"


def test_alias_recall_ranks_relevant_fact_above_newer_generic_notes(store):
    service = MemoryService(store)
    service.handle("My favorite game is CS2.")
    service.handle("Remember that we worked on code today.")
    ranked = rank_memories(store, "What do I enjoy playing?")
    assert ranked[0][1].metadata["value"] == "CS2"


def test_forget_redacts_history_and_summary_but_keeps_archive(store):
    service = MemoryService(store)
    service.handle("My favorite game is CS2.")
    service.handle("Actually, my favorite game is Valorant.")
    manager = ConversationManager(store=store, session_id="test")
    manager.add_turn("user", "My favorite game is CS2.")
    manager.add_turn("assistant", "You enjoy Valorant.")
    store.save_memory(
        MemoryItem(
            content="The user plays Valorant.",
            memory_type=MemoryType.CONVERSATION,
            metadata={"session_id": "test"},
        )
    )
    assert "forgotten" in service.handle("Forget my favorite game.")
    assert store.get_fact("user:favorite_game") is None
    model_context = "\n".join(m.content for m in manager.get_active_messages("Persona"))
    assert "CS2" not in model_context
    assert "Valorant" not in model_context
    assert "CS2" in store.get_recent_turns("test")[0].content
    service.handle("My favorite game is Valorant.")
    assert store.redact_forgotten("Valorant") == "Valorant"


def test_unknown_forget_does_not_delete_previous_focus(store):
    service = MemoryService(store)
    service.handle("My favorite game is CS2.")
    reply = service.handle("Forget my passport number.")
    assert "couldn't find" in reply
    assert store.get_fact("user:favorite_game") is not None


def test_forget_that_uses_last_saved_fact_and_zero_match_is_honest(store):
    service = MemoryService(store)
    service.handle("Remember that I have a red keyboard.")
    assert "forgotten" in service.handle("Forget that.")
    assert "couldn't find" in service.handle("Forget my red keyboard.")


def test_existing_structured_anchor_is_updated_in_place(tmp_path):
    path = tmp_path / "legacy.db"
    store = MemoryStore(path)
    memory_id = store.save_memory(
        MemoryItem(
            content="RAPHAEL project began on 2026-09-28.",
            memory_type=MemoryType.PROJECT,
            metadata={
                "project": "raphael",
                "fact_key": "project_start_date",
                "value": "2026-09-28",
            },
        )
    )
    store.close()
    store = MemoryStore(path)
    try:
        MemoryService(store).handle(
            "Actually, the project started September 27.", datetime(2026, 10, 3)
        )
        assert store.get_fact("raphael:project_start_date").id == memory_id
        assert store.count_memories() == 1
    finally:
        store.close()


def test_unrelated_question_clears_ambiguous_correction_focus(store):
    service = MemoryService(store)
    service.handle("My favorite game is CS2.")
    service.handle("What's the time right now?")
    assert service.handle("No, it's Valorant.") is None
    assert store.get_fact("user:favorite_game").metadata["value"] == "CS2"


def test_forgetting_persists_across_restart(tmp_path):
    path = tmp_path / "forgotten.db"
    store = MemoryStore(path)
    service = MemoryService(store)
    service.handle("My favorite food is pizza.")
    service.handle("Forget my favorite food.")
    store.close()
    store = MemoryStore(path)
    try:
        assert store.get_fact("user:favorite_food") is None
        assert "pizza" not in store.redact_forgotten("Your favorite food is pizza.")
    finally:
        store.close()


def test_note_object_is_redacted_in_paraphrased_summary(store):
    service = MemoryService(store)
    service.handle("Remember that I own a red keyboard.")
    service.handle("Forget that.")
    assert "red keyboard" not in store.redact_forgotten("The user owns a red keyboard.")


def test_unknown_profile_query_does_not_create_a_fact(store):
    service = MemoryService(store)
    assert service.handle("What's my favorite game?") is None
    assert store.count_memories() == 0


def test_parallel_key_updates_do_not_create_conflicting_current_records(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "parallel.db"
    first = MemoryStore(path)
    second = MemoryStore(path)

    def save(store, value):
        return store.upsert_fact(
            "user:favorite_game",
            MemoryItem(content=f"Your favorite game is {value}.", metadata={"value": value}),
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            ids = list(
                executor.map(lambda args: save(*args), [(first, "CS2"), (second, "Valorant")])
            )
        assert ids[0] == ids[1]
        assert first.count_memories() == 1
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize(
    "correction", ["No, wait.", "No, it's incorrect.", "Actually, never mind."]
)
def test_vague_objection_does_not_replace_favorite(store, correction):
    service = MemoryService(store)
    service.handle("My favorite game is CS2.")
    assert service.handle(correction) is None
    assert store.get_fact("user:favorite_game").metadata["value"] == "CS2"


def test_curly_apostrophe_correction(store):
    service = MemoryService(store)
    service.handle("My favorite game is CS2.")
    assert "updated" in service.handle("No, it’s Valorant.")
    assert store.get_fact("user:favorite_game").metadata["value"] == "Valorant"
