"""Conservative local fact recognition, keyed updates, and explicit forgetting."""

import hashlib
import re
from datetime import datetime

from raphael.memory.models import MemoryItem, MemoryType
from raphael.memory.recall import rank_memories
from raphael.memory.store import MemoryStore

MONTHS = {
    name: number
    for number, names in enumerate(
        (
            "jan january",
            "feb february",
            "mar march",
            "apr april",
            "may",
            "jun june",
            "jul july",
            "aug august",
            "sep sept september",
            "oct october",
            "nov november",
            "dec december",
        ),
        1,
    )
    for name in names.split()
}


def parse_project_date(text: str, now: datetime) -> str | None:
    """Resolve explicit month/day dates and ISO dates; never infer a date from 'day three'."""
    iso = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if iso:
        try:
            date = datetime.strptime(iso.group(1), "%Y-%m-%d").date()
            return date.isoformat() if date <= now.date() else None
        except ValueError:
            return None
    match = re.search(
        r"\b(" + "|".join(MONTHS) + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?"
        r"(?:,?\s+(\d{4}))?\b",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    try:
        date = datetime(
            int(match.group(3) or now.year),
            MONTHS[match.group(1).lower()],
            int(match.group(2)),
        ).date()
        return date.isoformat() if date <= now.date() else None
    except ValueError:
        return None


class MemoryService:
    """Persist direct user statements; unknown or ambiguous speech stays conversational."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self.focus_key: str | None = None

    def observe_topic(self, text: str) -> None:
        """Track an explicit topic for a subsequent short correction."""
        self.focus_key = None
        lowered = text.casefold()
        if re.search(r"\b(?:project|development|developing)\b", lowered):
            self.focus_key = "raphael:project_start_date"
        if "first commit" in lowered:
            self.focus_key = "raphael:first_commit_date"
        match = re.search(
            r"\bfavou?rite\s+(game|food|color|colour|drink|movie|language)\b", lowered
        )
        if match:
            self.focus_key = "user:favorite_" + match.group(1).replace("colour", "color")

    def handle(self, text: str, now: datetime | None = None) -> str | None:
        """Save recognized facts or commands and return an honest local acknowledgement."""
        now = now or datetime.now()
        original = text.strip().replace("’", "'")
        explicit = re.match(
            r"^(?:please\s+)?(?:remember(?:\s+that)?|note\s+that)\s+(.+)$",
            original,
            re.IGNORECASE,
        )
        forget = re.match(
            r"^(?:please\s+)?forget(?:\s+that|\s+about)?\s+(.+)$",
            original,
            re.IGNORECASE,
        )
        if forget:
            return self._forget(forget.group(1))
        if re.fullmatch(r"(?:please\s+)?forget(?:\s+(?:that|it))?[.!]?", original, re.I):
            return self._forget("that")
        statement = explicit.group(1) if explicit else original
        correction = bool(
            re.match(r"^(?:no[,.\s]|actually\b|i meant\b|correction\b)", statement, re.I)
        )
        clean = (
            re.sub(
                r"^(?:(?:no|actually|correction)[,.\s]+|i meant\s+)+",
                "",
                statement,
                flags=re.I,
            )
            .strip()
            .rstrip(".!")
        )
        # Questions, quotations and speculative statements aren't facts.
        if not explicit and (
            "?" in clean
            or re.match(r"^(what|why|when|how|do|does|can|if|maybe|she|he)\b", clean, re.I)
        ):
            self.observe_topic(original)
            simple_recall = re.fullmatch(
                r"(?:what(?:'s| is)|which is|do you (?:remember|know))\s+"
                r"my favou?rite (?:game|food|color|colour|drink|movie|language)[?.!]*",
                original,
                re.I,
            )
            if simple_recall and self.focus_key:
                remembered = self.store.get_fact(self.focus_key)
                if remembered is not None:
                    return remembered.content
            if re.fullmatch(
                r"(?:what(?:'s| is) my name|what do you call me)[?.!]*",
                original,
                re.I,
            ):
                remembered = self.store.get_fact("user:preferred_name")
                if remembered is not None:
                    return remembered.content
            return None
        key = None
        value = None
        kind = MemoryType.FACT
        favorite = re.fullmatch(
            r"my\s+favou?rite\s+(game|food|color|colour|drink|movie|language)\s+"
            r"(?:is(?: now)?|has changed to)\s+(.+)",
            clean,
            re.I,
        )
        name = re.fullmatch(r"(?:my name is|call me|i prefer to be called)\s+(.+)", clean, re.I)
        if favorite:
            topic = favorite.group(1).lower().replace("colour", "color")
            key, value = "user:favorite_" + topic, favorite.group(2).strip()
            clean = f"Your favorite {topic} is {value}."
            kind = MemoryType.PREFERENCE
        elif name:
            key, value = "user:preferred_name", name.group(1).strip()
            if len(value.split()) > 4 or re.match(
                r"^(?:later|tomorrow|back|when|if|at|after|on)\b",
                value,
                re.I,
            ):
                self.focus_key = None
                return None
            clean = f"Your preferred name is {value}."
            kind = MemoryType.PREFERENCE
        elif re.search(
            r"\b(?:project|development|developing|building|first commit|start date)\b",
            clean,
            re.I,
        ) or self.focus_key in ("raphael:project_start_date", "raphael:first_commit_date"):
            value = parse_project_date(clean, now)
            if value and (
                re.search(r"\b(?:start|started|began|begin|from|commit|meant)\b", clean, re.I)
                or (correction and self.focus_key and self.focus_key.startswith("raphael:"))
            ):
                key = (
                    "raphael:first_commit_date"
                    if "first commit" in clean.lower()
                    or (correction and self.focus_key == "raphael:first_commit_date")
                    else "raphael:project_start_date"
                )
                kind = MemoryType.PROJECT
                clean = (
                    f"RAPHAEL's first commit was on {value}."
                    if key.endswith("first_commit_date")
                    else f"RAPHAEL project development began on {value}."
                )
        if key is None and correction and self.focus_key:
            short = re.fullmatch(r"(?:it(?:'s| is)\s+)?(.{1,80})", clean, re.I)
            if (
                short
                and self.focus_key.startswith("user:favorite_")
                and not re.match(
                    r"^(?:i|you|that|this|wrong|incorrect|stop|wait|hold|what|huh|"
                    r"sorry|never|not|don'?t)\b",
                    short.group(1),
                    re.I,
                )
            ):
                key, value = self.focus_key, short.group(1).strip()
                topic = key.split("favorite_", 1)[1]
                clean = f"Your favorite {topic} is {value}."
                kind = MemoryType.PREFERENCE
        if key is None:
            if not explicit:
                self.observe_topic(original)
                return None
            digest = hashlib.sha256(re.sub(r"\W+", " ", clean.casefold()).encode()).hexdigest()[:20]
            key = "user:note_" + digest
        if (
            key.startswith(("user:favorite_", "user:preferred_name"))
            and value
            and re.match(
                r"^(?:not|unknown|unsure|maybe)\b",
                value,
                re.I,
            )
        ):
            return None
        old = self.store.get_fact(key)
        self.store.upsert_fact(
            key,
            MemoryItem(
                content=clean,
                memory_type=kind,
                source="user_explicit",
                metadata={"value": value} if value is not None else {},
            ),
        )
        self.focus_key = key
        action = "updated" if old and old.content != clean else "saved"
        return f"I've {action} that. {clean}"

    def _forget(self, topic: str) -> str:
        """Resolve specific saved topics, refusing an ambiguous broad deletion."""
        topic = topic.strip().rstrip(".!?")
        if topic.strip().casefold() not in ("that", "it", "the last fact"):
            self.focus_key = None
            self.observe_topic(topic)
        item = self.store.get_fact(self.focus_key) if self.focus_key else None
        matches = [
            memory
            for memory in self.store.search_memories(
                "",
                limit=2000,
                exclude_types=(MemoryType.CONVERSATION,),
            )
            if topic.casefold() in memory.content.casefold()
        ]
        if item is not None:
            matches = [item]
        if not matches:
            ranked = rank_memories(self.store, topic, limit=2)
            if ranked and (len(ranked) == 1 or ranked[0][0] > ranked[1][0] * 1.5):
                matches = [ranked[0][1]]
        if not matches:
            return "I couldn't find a saved fact matching that. Which fact should I forget?"
        if len(matches) > 1:
            return "That matches several saved facts. Could you name the specific one to forget?"
        deleted = self.store.forget_memories(matches)
        self.focus_key = None
        return "I've forgotten that saved fact." if deleted else "That fact was already removed."
