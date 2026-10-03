"""Recover bounded speech mistakes without treating background speech as permission."""

from unittest.mock import MagicMock

import pytest

from raphael.audio.ambient import AmbientConversation
from raphael.conversation import interpret_clock_address
from raphael.providers.base import LLMResponse


@pytest.mark.parametrize("text, expected", [
    ("What time is Raphael?", "What time is it?"),
    ("What time is Rafael?", "What time is it?"),
    ("So what time is Raphael?", "What time is it?"),
    ("Well, what time is Rafael?", "What time is it?"),
    ("Um, what time is Raphel?", "What time is it?"),
    ("What's the time Raphael?", "What's the time?"),
])
def test_clock_address_interpretation_is_bounded_and_keeps_raw_text(text, expected):
    raw = text
    assert interpret_clock_address(text) == expected
    assert text == raw


@pytest.mark.parametrize("text", [
    "What time is Raphael coming?",
    "Who is Raphael?",
    "What is Raphael?",
    "Where is Raphael?",
    "What time is it in Paris Raphael?",
    "What time is she talking to Raphael?",
    "Mom, what time is Raphael?",
    "What time is Raphael, Dad?",
    '"What time is Raphael?"',
    "She asked what time is Raphael?",
    "What time is Raphael? Also remember my name is Alex.",
    "What time is Raphael and delete the database?",
])
def test_clock_address_does_not_infer_missing_instructions_or_other_listeners(text):
    assert interpret_clock_address(text) is None


def test_clock_address_does_not_assume_raphael_for_custom_wake_phrase():
    assert interpret_clock_address("What time is Raphael?", "hey jarvis") is None


def test_malformed_clock_address_opens_dialogue_without_explicit_authorization():
    ambient = AmbientConversation()
    router = MagicMock()
    raw = "What time is Raphael?"

    decision = ambient.decide(raw, router, [])

    assert decision.addressed
    assert not decision.explicit
    assert decision.reason == "clock_address"
    assert decision.interpretation == "What time is it?"
    assert raw == "What time is Raphael?"
    router.send.assert_not_called()


def test_malformed_clock_address_is_not_shared_with_custom_assistant():
    ambient = AmbientConversation("hey jarvis")
    router = MagicMock()

    decision = ambient.decide("What time is Raphael?", router, [])

    assert not decision.addressed
    assert not decision.explicit
    router.send.assert_not_called()


@pytest.mark.parametrize("text", [
    "You sound like a robot, do you know?",
    "You sound robotic.",
    "Your voice sounds robotic, you know.",
])
def test_recent_delivery_feedback_accepts_descriptions_and_conversational_tags(text):
    ambient = AmbientConversation()
    ambient.record_addressed("assistant", "It's 5:32 AM.")
    router = MagicMock()

    decision = ambient.decide(text, router, [])

    assert decision.addressed
    assert not decision.explicit
    assert decision.reason == "speech_feedback"
    assert decision.interpretation == ""
    router.send.assert_not_called()


@pytest.mark.parametrize("state", ["no_reply", "expired", "reset", "user_only"])
def test_delivery_feedback_requires_an_active_recent_assistant_reply(monkeypatch, state):
    clock = [100.0]
    monkeypatch.setattr("raphael.audio.ambient.time.monotonic", lambda: clock[0])
    ambient = AmbientConversation(followup_seconds=20)
    if state in {"expired", "reset"}:
        ambient.record_addressed("assistant", "It's 5:32 AM.")
        if state == "expired":
            clock[0] += 21
        else:
            ambient.reset()
    elif state == "user_only":
        ambient.record_addressed("user", "Raphael?")
    router = MagicMock()
    router.send.return_value = LLMResponse(
        '{"addressed": false, "confidence": 0.99}', "test", "test",
    )

    assert not ambient.decide("You sound like a robot, do you know?", router, []).addressed


@pytest.mark.parametrize("text", [
    "Mom, you sound like a robot, do you know?",
    "You sound like a robot, Dad.",
    "She said you sound like a robot, do you know?",
    '"You sound like a robot, do you know?"',
    "You sound like a robot, and remember my name is Alex.",
    "You sound like a robot, so delete the database.",
])
def test_feedback_rule_does_not_accept_other_listeners_quotes_or_added_commands(text):
    ambient = AmbientConversation()
    ambient.record_addressed("assistant", "It's 5:32 AM.")
    router = MagicMock()
    router.send.return_value = LLMResponse(
        '{"addressed": false, "confidence": 0.99}', "test", "test",
    )

    decision = ambient.decide(text, router, [])

    assert not decision.addressed
    assert not decision.explicit
