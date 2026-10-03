"""Ambient reply decisions, local VAD framing, and persistence boundaries."""

import json
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from raphael.audio.ambient import AmbientConversation
from raphael.providers.base import ChatMessage, LLMResponse


@pytest.mark.parametrize(
    "text", [
        "Hey, Raphael.", "Raphael, help me.", "Can you help me, Raphael?",
        "What's up Raphael?", "What's up, Raphel?", "Hi Ralph!", "Hello, Raphael.",
        "So what's good Raphael?", "So Raphael, what's on your mind?",
        "Well, hey Raphael, what's the time?", "Uh, what's up, Rafael?",
    ]
)
def test_direct_address_needs_no_cloud_judgment(text):
    router = MagicMock()
    decision = AmbientConversation().decide(text, router, [])
    assert decision.addressed and decision.explicit
    router.send.assert_not_called()


def test_unaddressed_speech_is_temporary_and_expires(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("raphael.audio.ambient.time.monotonic", lambda: clock[0])
    ambient = AmbientConversation()
    router = MagicMock()
    assert not ambient.decide("I hate Raphael.", router, []).addressed
    assert not ambient.decide("Raphael is an assistant.", router, []).addressed
    assert not ambient.decide("What do you think about Raphael?", router, []).addressed
    assert not ambient.decide("Who is Raphael?", router, []).addressed
    assert not ambient.decide("Dad, what's for dinner?", router, []).addressed
    assert "Dad" in ambient.context_note()
    router.send.assert_not_called()
    clock[0] += 91
    assert ambient.context_note() == ""


@pytest.mark.parametrize("text", [
    "So Raphael is an assistant.", "Well, Raphael said hello.",
    "So what do you think about Raphael?", "Um, who is Raphael?",
    "So I hate Raphael.", "So what's good?", "Well, Mom, what's for dinner?",
])
def test_fillers_do_not_turn_background_mentions_into_addresses(text):
    router = MagicMock()
    assert not AmbientConversation().decide(text, router, []).addressed
    router.send.assert_not_called()


def test_independent_wake_confirms_address_even_when_stt_misses_name():
    router = MagicMock()
    decision = AmbientConversation().decide("What's up?", router, [], verified_wake=True)
    assert decision.addressed and decision.explicit and decision.reason == "verified_wake"
    router.send.assert_not_called()


@pytest.mark.parametrize(
    "result",
    [
        "not json", "[]", '{"addressed": true, "confidence": 0.7}',
        '{"addressed": true, "confidence": "0.99"}',
        '{"addressed": true, "confidence": true}',
        '{"addressed": true, "confidence": 2}',
        '{"addressed": false, "confidence": 0.99}',
    ],
)
def test_uncertain_or_malformed_judgment_means_silence(result):
    ambient = AmbientConversation()
    ambient.replied()
    router = MagicMock()
    router.send.return_value = LLMResponse(result, "test", "test")
    assert not ambient.decide("What do you mean?", router, []).addressed
    assert ambient.deadline == 0


def test_followup_can_supply_hint_without_replacing_original():
    ambient = AmbientConversation()
    ambient.replied()
    router = MagicMock()
    router.send.return_value = LLMResponse(
        '{"addressed": true, "confidence": 0.95, "interpretation": "A new feature"}',
        "test", "test",
    )
    original = "What future should we add?"
    decision = ambient.decide(original, router, [ChatMessage("assistant", "Let's plan features.")])
    assert decision.addressed and not decision.explicit
    assert decision.interpretation == "A new feature"
    request = router.send.call_args
    assert request.kwargs["purpose"] == "speech_gate"
    assert json.loads(request.args[0][1].content)["transcript"] == original


def test_fenced_json_judgment_is_accepted():
    ambient = AmbientConversation()
    ambient.replied()
    router = MagicMock()
    router.send.return_value = LLMResponse(
        '```json\n{"addressed": true, "confidence": 0.95}\n```', "test", "test"
    )
    decision = ambient.decide("And what does that mean?", router, [])
    assert decision.addressed and decision.reason == "clear_followup"


def test_silent_decisions_explain_the_reason():
    ambient = AmbientConversation()
    router = MagicMock()
    assert ambient.decide("Dinner is ready.", router, []).reason == "outside_followup_window"
    assert ambient.decide("", router, []).reason == "no_transcript"
    assert ambient.decide("Mom, pass that.", router, []).reason == "addressed_to_someone_else"


def test_provider_failure_defaults_to_silence():
    ambient = AmbientConversation()
    ambient.replied()
    router = MagicMock()
    router.send.side_effect = RuntimeError("offline")
    assert not ambient.decide("Continue?", router, []).addressed
    ambient.reset()
    assert ambient.context_note() == ""


def test_long_followup_is_judged_using_when_speech_started(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("raphael.audio.ambient.time.monotonic", lambda: clock[0])
    ambient = AmbientConversation(followup_seconds=20)
    ambient.replied()
    clock[0] = 130.0  # Speech began during the window but took time to finish.
    router = MagicMock()
    router.send.return_value = LLMResponse(
        '{"addressed": true, "confidence": 0.95}', "test", "test"
    )
    assert ambient.decide("Explain that a little more.", router, [], started_at=110).addressed


def test_vad_preserves_partial_chunks_and_converts_integer_audio(monkeypatch):
    from raphael.audio.activity import SpeechActivity

    model = MagicMock()
    model.chunk_samples.return_value = 512
    model.process_array.side_effect = [0.1, 0.8]
    monkeypatch.setattr("pysilero_vad.SileroVoiceActivityDetector", lambda: model)
    vad = SpeechActivity()
    assert not vad(np.full(256, 16384, dtype=np.int16))
    assert vad(np.full(768, 16384, dtype=np.int16))
    assert model.process_array.call_count == 2
    np.testing.assert_allclose(model.process_array.call_args.args[0], 0.5)
    assert vad.pending.size == 0
    vad.reset()
    model.reset.assert_called_once()


def run_callbacks(
    tmp_path, monkeypatch, utterances, *, ambient=True, router=None, observed=None
):
    """Run real CLI callbacks using an isolated DB and no audio hardware."""
    import sys

    from raphael import __main__, audio, config, platform, providers
    from raphael.memory import MemoryStore
    from raphael.persona import PERSONA_CONTEXT_VERSION

    settings = config.Settings(_env_file=None, memory_db_path=str(tmp_path / "memory.db"))
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(platform, "get_audio_backend", MagicMock())
    router = router or MagicMock()
    monkeypatch.setattr(providers, "get_model_router", lambda: router)
    tts = MagicMock()
    for name in ("WakeWordDetector", "SpeechToText", "VoiceRecorder"):
        monkeypatch.setattr(audio, name, MagicMock())
    monkeypatch.setattr(audio, "TextToSpeech", MagicMock(return_value=tts))
    loop = SimpleNamespace(is_running=True, stop=MagicMock(), set_ambient=MagicMock())

    def listener(**kwargs):
        def start():
            for text, info in observed or []:
                kwargs["on_transcript_observed"](text, info)
            for text, info in utterances:
                kwargs["on_transcription"](text, info, None)

        loop.start = start
        return loop

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(audio, "WakeListenerLoop", listener)
    monkeypatch.setattr(__main__.time, "sleep", interrupt)
    monkeypatch.setattr(sys, "argv", ["raphael", "--ambient" if ambient else "--listen"])
    assert __main__.main() == 0
    store = MemoryStore(settings.memory.db_path)
    turns = store.get_recent_turns(f"desktop_session:{PERSONA_CONTEXT_VERSION}")
    return router, tts, loop, store, turns


def test_overheard_facts_are_not_saved_or_sent_to_provider(tmp_path, monkeypatch):
    router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("My favorite game is CS2.", {"ambient": True})]
    )
    try:
        assert not turns
        assert store.get_fact("user:favorite_game") is None
        router.send.assert_not_called()
        tts.speak.assert_not_called()
    finally:
        store.close()


def test_voice_mode_controls_and_background_silence(tmp_path, monkeypatch):
    router, tts, loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [
            ("Raphael, listen continuously.", {}),
            ("Mom, pass me that.", {"ambient": True}),
            ("Raphael, stop listening.", {"ambient": True}),
        ],
        ambient=False,
    )
    try:
        assert [call.args[0] for call in loop.set_ambient.call_args_list] == [True, False]
        assert tts.speak.call_count == 2
        router.send.assert_not_called()
        assert not turns
    finally:
        store.close()


def test_uncertain_recognition_does_not_auto_save_a_fact(tmp_path, monkeypatch):
    router = MagicMock()
    router.send.return_value = LLMResponse("Did you say CS2?", "test", "test")
    _router, _tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("My favorite game is CS2.", {"stt_confidence": 0.45})],
        ambient=False, router=router,
    )
    try:
        assert store.get_fact("user:favorite_game") is None
        assert turns[0].content == "My favorite game is CS2."
        assert "Preserve names" in router.send.call_args.args[0][0].content
    finally:
        store.close()


def test_resumed_speech_suppresses_old_answer_and_assistant_history(tmp_path, monkeypatch):
    cancelled = Event()
    router = MagicMock()

    def interrupted(*_args, **_kwargs):
        cancelled.set()
        return LLMResponse("An old answer", "test", "test")

    router.send.side_effect = interrupted
    _router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Explain that idea.", {"cancel_event": cancelled})],
        ambient=False, router=router,
    )
    try:
        tts.speak.assert_not_called()
        assert len(turns) == 1
        assert turns[0].role == "user"
    finally:
        store.close()


def test_ambient_local_greeting_opens_followup_with_actual_context(tmp_path, monkeypatch):
    router = MagicMock()
    router.send.side_effect = [
        LLMResponse('{"addressed": true, "confidence": 0.95}', "test", "test"),
        LLMResponse("I can help you build things.", "test", "test"),
    ]
    _router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Raphael.", {}), ("What are you good at?", {"ambient": True})],
        router=router,
    )
    try:
        assert tts.speak.call_count == 2
        gate_request = router.send.call_args_list[0]
        payload = json.loads(gate_request.args[0][1].content)
        assert payload["recent_dialogue"][0]["content"] == "Raphael."
        assert payload["recent_dialogue"][1]["role"] == "assistant"
        assert [turn.role for turn in turns] == ["user", "assistant"]
    finally:
        store.close()


def test_inferred_ambient_followup_never_auto_saves_personal_facts(tmp_path, monkeypatch):
    router = MagicMock()
    router.send.side_effect = [
        LLMResponse('{"addressed": true, "confidence": 0.99}', "test", "test"),
        LLMResponse("You can tell me your preference directly.", "test", "test"),
    ]
    _router, _tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Raphael.", {}), ("My favorite game is CS2.", {"ambient": True})],
        router=router,
    )
    try:
        assert store.get_fact("user:favorite_game") is None
        assert turns[0].content == "My favorite game is CS2."
    finally:
        store.close()


def test_superseded_direct_address_opens_temporary_followup_without_saving(tmp_path, monkeypatch):
    router = MagicMock()
    router.send.side_effect = [
        LLMResponse('{"addressed": true, "confidence": 0.99}', "test", "test"),
        LLMResponse("I'm listening.", "test", "test"),
    ]
    _router, _tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("What are you good at?", {"ambient": True})],
        router=router, observed=[("Hey Raphael.", {"superseded": True})],
    )
    try:
        gate = json.loads(router.send.call_args_list[0].args[0][1].content)
        assert gate["recent_dialogue"][0]["content"] == "Hey Raphael."
        assert len(turns) == 2
        assert turns[0].content == "What are you good at?"
    finally:
        store.close()


def test_unclear_direct_speech_gets_repeat_without_saving(tmp_path, monkeypatch):
    router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("", {"stt_raw_text": "Hey Raphael.", "stt_needs_repeat": True})],
    )
    try:
        router.send.assert_not_called()
        assert not turns
        tts.speak.assert_called_once_with(
            "I didn't catch that clearly. Could you say it again?", block=True
        )
    finally:
        store.close()


def test_verified_wake_with_unclear_command_asks_repeat_without_guessing(tmp_path, monkeypatch):
    router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("", {"wake_verified": True, "stt_raw_text": "file", "stt_needs_repeat": True})],
    )
    try:
        router.send.assert_not_called()
        assert not turns
        tts.speak.assert_called_once_with(
            "I didn't catch that clearly. Could you say it again?", block=True
        )
    finally:
        store.close()


def test_verified_wake_allows_local_time_without_name_in_stt(tmp_path, monkeypatch):
    router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("What's the time?", {"wake_verified": True})],
    )
    try:
        router.send.assert_not_called()
        tts.speak.assert_called_once()
        assert ":" in tts.speak.call_args.args[0]
        assert [turn.role for turn in turns] == ["user", "assistant"]
        assert turns[0].content == "What's the time?"
    finally:
        store.close()


@pytest.mark.parametrize("text, query", [
    ("So what's good Raphael?", "So what's good Raphael?"),
    ("So Raphael, what's on your mind?", "what's on your mind?"),
])
def test_live_log_addresses_reply_without_speech_gate(tmp_path, monkeypatch, text, query):
    router = MagicMock()
    router.send.return_value = LLMResponse("Hey hexarion! How's your day going?", "test", "test")
    router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, [(text, {"stt_confidence": 0.66})], router=router,
    )
    try:
        assert router.send.call_count == 1
        assert router.send.call_args.kwargs.get("purpose", "conversation") == "conversation"
        assert router.send.call_args.args[0][-1].content == query
        tts.speak.assert_called_once()
        assert turns[0].content == query
    finally:
        store.close()
