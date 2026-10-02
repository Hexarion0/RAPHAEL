import pytest

from raphael.conversation import is_farewell


@pytest.mark.parametrize(
    "text",
    [
        "bye",
        "Goodbye!",
        "good night",
        "thanks, bye Raphael!",
        "See you tomorrow.",
        "take care",
        "bye-bye",
    ],
)
def test_deliberate_sign_off(text):
    assert is_farewell(text)


@pytest.mark.parametrize(
    "text",
    [
        "How do you say goodbye in Spanish?",
        "Explain Goodbye Yellow Brick Road",
        "Say goodbye",
        "Don't say goodbye yet",
        "What does take care mean?",
        'The answer is "goodbye".',
        "Good night shifts pay more",
    ],
)
def test_farewell_words_in_questions_do_not_end_session(text):
    assert not is_farewell(text)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hey Raphael.", ""),
        ("Hey Raphael!", ""),
        ("Raphael?", ""),
        ("Hey Raphael, are you good?", "are you good?"),
        ("hey raphael. Hey Raphael, what's the time?", "what's the time?"),
        ("Raphaelite art", "Raphaelite art"),
        ("Tell me about Raphael.", "Tell me about Raphael."),
    ],
)
def test_wake_phrase_is_removed_with_its_punctuation(text, expected):
    from raphael.conversation import strip_wake_phrase

    assert strip_wake_phrase(text) == expected


def test_custom_wake_phrase_is_removed():
    from raphael.conversation import strip_wake_phrase

    assert strip_wake_phrase("Hey Jarvis. Open the browser.", "hey jarvis") == "Open the browser."
