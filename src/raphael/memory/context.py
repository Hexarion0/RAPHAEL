"""Build dated, bounded memory context without replaying archived conversation summaries."""

from datetime import date, datetime

from raphael.memory.models import MemoryItem, MemoryType
from raphael.memory.recall import rank_memories
from raphael.memory.store import MemoryStore


def recall_context_memories(
    store: MemoryStore, query: str = "", now: datetime | None = None,
) -> list[str]:
    """Recall durable facts with timestamps and authoritative project date arithmetic."""
    current_date = (now or datetime.now()).date()
    recalled: list[MemoryItem] = []
    if query:
        recalled.extend(item for _, item in rank_memories(store, query))
    # Project anchors must survive informal questions that don't match their exact wording.
    recalled.extend(store.list_memories(memory_type=MemoryType.PROJECT, limit=3))
    start = store.get_fact("raphael:project_start_date")
    if start is not None:
        recalled.append(start)
    recalled.extend(store.list_memories(memory_type=MemoryType.PREFERENCE, limit=3))

    context: list[str] = []
    seen: set[int | str] = set()
    for memory in recalled:
        identity = memory.id if memory.id is not None else memory.content
        if identity in seen:
            continue
        seen.add(identity)
        recorded_at = memory.metadata.get("observed_at") or (
            memory.created_at.astimezone().isoformat(timespec="seconds")
        )
        context.append(f"Recorded at {recorded_at}: {memory.content}")
        if (
            memory.memory_type == MemoryType.PROJECT
            and memory.metadata.get("project") == "raphael"
            and memory.metadata.get("fact_key") == "project_start_date"
        ):
            try:
                start_date = date.fromisoformat(str(memory.metadata["value"]))
            except (KeyError, ValueError):
                continue
            elapsed = (current_date - start_date).days
            if elapsed >= 0:
                context.append(
                    f"RAPHAEL timeline as of {current_date.isoformat()}: confirmed project "
                    f"start date {start_date.isoformat()}; {elapsed} elapsed calendar days; "
                    f"development day {elapsed + 1} counting the start date as day one. "
                    "Use this dated calculation instead of old relative day-count statements."
                )
    return context
