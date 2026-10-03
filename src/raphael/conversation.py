"""Small intent checks shared by the voice conversation handler and its tests."""

import re

_RAPHAEL_NAMES = r"(?:raphael|rafael|raphel|rafeal|raffael|refael|raph|ralph|raffaele)"
_ADDRESS_FILLERS = re.compile(
    r"^(?:(?:so|well|uh|um|oh|okay|ok|alright|and)[,\s]+){1,3}", re.IGNORECASE
)

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
    configured = r"[\s,]+".join(re.escape(word) for word in wake_phrase.split())
    variants = [configured] if configured else []
    if re.search(r"\b(?:raphael|rafael|raphel|rafeal)\b", wake_phrase, re.IGNORECASE):
        variants.append(
            r"(?:(?:hey|hi|hello|yo|okay|ok)[\s,]+)?" + _RAPHAEL_NAMES
        )
    if not variants:
        return text.strip()
    prefix = re.compile(r"^(?:" + "|".join(variants) + r")\b[,.!?;:\s—-]*", re.IGNORECASE)
    query = text.strip()
    while True:
        candidate = query
        match = prefix.match(candidate)
        if match is None:
            candidate = _ADDRESS_FILLERS.sub("", query, count=1)
            match = prefix.match(candidate)
        if match is None:
            break
        query = candidate[match.end() :].lstrip()
    return query


def is_direct_address(text: str, wake_phrase: str = "hey raphael") -> bool:
    """Recognize leading addresses and short questions/greetings ending in the name."""
    text = text.strip()
    candidate = _ADDRESS_FILLERS.sub("", text, count=1)
    if re.match(r"^" + _RAPHAEL_NAMES + r"\s+(?:is|was|has|said|told)\b", candidate, re.I):
        return False
    if strip_wake_phrase(text, wake_phrase) != text:
        return True
    if not re.search(r"\b(?:raphael|rafael|raphel|rafeal)\b", wake_phrase, re.I):
        return False
    return bool(
        re.match(
            r"^(?:what|how|why|when|where|can|could|would|will|do|tell|help|please|hey|hi|hello)\b",
            candidate, re.I,
        )
        and re.search(r"[,\s]+" + _RAPHAEL_NAMES + r"[.!?]*$", candidate, re.I)
        # A name as the object of a sentence is not a vocative address.
        and not re.search(r"\b(?:about|of|with|to|named|called|is|was)\s+" + _RAPHAEL_NAMES,
                          candidate, re.I)
    )
