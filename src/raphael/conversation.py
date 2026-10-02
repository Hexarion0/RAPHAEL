"""Small intent checks shared by the voice conversation handler and its tests."""

import re

_FAREWELL = re.compile(
    r"(?:(?:ok(?:ay)?|alright|thanks|thank you)[,!.\s]+)?"
    r"(?:bye(?:[ -]bye)?|goodbye|good\s*night|see you(?: (?:later|soon|around|tomorrow))?"
    r"|take care|farewell)"
    r"(?:[,\s]+(?:raphael|rafael))?[.!\s]*",
    re.IGNORECASE,
)


def is_farewell(text: str) -> bool:
    """Recognize a short, deliberate sign-off, rather than words inside a question."""
    return _FAREWELL.fullmatch(text.strip()) is not None


def strip_wake_phrase(text: str, wake_phrase: str = "hey raphael") -> str:
    """Remove leading wake phrases and their punctuation without changing the query."""
    configured = r"\s+".join(re.escape(word) for word in wake_phrase.split())
    variants = [configured] if configured else []
    if re.search(r"\b(?:raphael|rafael|raphel|rafeal)\b", wake_phrase, re.IGNORECASE):
        variants.append(
            r"(?:(?:hey|hi|yo|okay|ok)\s+)?"
            r"(?:raphael|rafael|raphel|rafeal|raffael|refael|raph|ralph|raffaele)"
        )
    if not variants:
        return text.strip()
    prefix = re.compile(r"^(?:" + "|".join(variants) + r")\b[,.!?;:\s—-]*", re.IGNORECASE)
    query = text.strip()
    while match := prefix.match(query):
        query = query[match.end() :].lstrip()
    return query
