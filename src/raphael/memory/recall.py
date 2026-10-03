"""Local ranked recall with word normalization and small, explicit vocabulary groups."""

import math
import re
from collections import Counter

from raphael.memory.models import MemoryItem, MemoryType
from raphael.memory.store import MemoryStore

STOP_WORDS = set(
    "a an the is are was were do does did i you my your me we our it this that "
    "what which who when how about please can could would should tell know remember "
    "forget of for to in on at and or actually no yes right now".split()
)
GROUPS = (
    {"favorite", "favourite", "preferred", "prefer", "likes", "like", "love", "enjoy"},
    {"game", "games", "gaming", "play", "playing"},
    {"start", "started", "begin", "began", "beginning", "origin"},
    {"development", "developing", "build", "building", "project"},
    {"date", "day", "days", "timeline"},
    {"name", "named", "call", "called", "nickname"},
)
ALIASES = {word: next(iter(sorted(group))) for group in GROUPS for word in group}


def terms(text: str) -> set[str]:
    """Normalize meaningful Unicode words without treating punctuation as a wildcard."""
    words = re.findall(r"[^\W_]+", text.casefold())
    return {ALIASES.get(word, word) for word in words if word not in STOP_WORDS}


def rank_memories(
    store: MemoryStore,
    query: str,
    limit: int = 4,
) -> list[tuple[float, MemoryItem]]:
    """Rank a bounded durable-memory set by distinctive shared terms and full phrases."""
    query_terms = terms(query)
    if not query_terms:
        return []
    candidates = store.search_memories(
        "",
        limit=2000,
        exclude_types=(MemoryType.CONVERSATION,),
    )
    documents = [
        terms(item.content + " " + str(item.metadata.get("fact_key", "")).replace("_", " "))
        for item in candidates
    ]
    frequencies = Counter(term for document in documents for term in document)
    ranked: list[tuple[float, MemoryItem]] = []
    for item, document in zip(candidates, documents):
        overlap = query_terms & document
        if not overlap:
            continue
        score = sum(math.log(1 + len(candidates) / frequencies[word]) for word in overlap)
        score *= len(overlap) / len(query_terms)
        ranked.append((score, item))
    ranked.sort(key=lambda pair: (pair[0], pair[1].updated_at), reverse=True)
    return ranked[:limit]
