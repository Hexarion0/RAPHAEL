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
    tmp_path, monkeypatch, utterances, *, ambient=True, router=None, observed=None,
    tts=None, interrupted=None,
    streaming=False, clock=None,
):
    """Run real CLI callbacks using an isolated DB and no audio hardware."""
    import sys

    from raphael import __main__, audio, config, platform, providers
    from raphael.memory import MemoryStore
    from raphael.persona import PERSONA_CONTEXT_VERSION

    settings = config.Settings(
        _env_file=None, memory_db_path=str(tmp_path / "memory.db"), tts_streaming=streaming,
    )
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(platform, "get_audio_backend", MagicMock())
    router = router or MagicMock()
    monkeypatch.setattr(providers, "get_model_router", lambda: router)
    tts = tts or MagicMock()
    for name in ("WakeWordDetector", "SpeechToText", "VoiceRecorder"):
        monkeypatch.setattr(audio, name, MagicMock())
    monkeypatch.setattr(audio, "TextToSpeech", MagicMock(return_value=tts))
    loop = SimpleNamespace(is_running=True, stop=MagicMock(), set_ambient=MagicMock())

    def listener(**kwargs):
        def start():
            if interrupted:
                kwargs['on_interruption'](interrupted)
            for text, info in observed or []:
                kwargs["on_transcript_observed"](text, info)
            for text, info in utterances:
                kwargs["on_transcription"](text, info, None)

        loop.start = start
        return loop

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(audio, "WakeListenerLoop", listener)
    # Patch only the CLI clock: background provider threads use real time.sleep.
    monkeypatch.setattr(__main__, "time", SimpleNamespace(
        sleep=interrupt, monotonic=clock or __main__.time.monotonic,
    ))
    monkeypatch.setattr(sys, "argv", ["raphael", "--ambient" if ambient else "--listen"])
    assert __main__.main() == 0
    store = MemoryStore(settings.memory.db_path)
    turns = store.get_recent_turns(f"desktop_session:{PERSONA_CONTEXT_VERSION}")
    return router, tts, loop, store, turns


def test_followup_window_begins_after_long_playback(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr('raphael.audio.ambient.time.monotonic', lambda: clock[0])
    tts = MagicMock()

    def speak(*_args, **_kwargs):
        clock[0] += 30  # Longer than the default 20-second follow-up window.
        return True

    tts.speak.side_effect = speak
    router = MagicMock()
    router.send.side_effect = [
        LLMResponse('{"addressed": true, "confidence": 0.95}', 'test', 'test'),
        LLMResponse('Let us continue.', 'test', 'test'),
    ]
    _router, tts, _loop, store, _turns = run_callbacks(
        tmp_path, monkeypatch, [('Hey Raphael.', {}), ('Continue?', {})],
        router=router, tts=tts,
    )
    try:
        assert tts.speak.call_count == 2
        assert router.send.call_count == 2
    finally:
        store.close()


def test_live_stream_speaks_sentences_once_and_persists_one_assistant_turn(tmp_path, monkeypatch):
    from raphael.providers.base import LLMStreamChunk

    router, tts = MagicMock(), MagicMock()
    router.stream.side_effect = lambda *args, **kwargs: iter([
        LLMStreamChunk("First sentence. ", "stream-model", "nim"),
        LLMStreamChunk("Second sentence.", "stream-model", "nim"),
    ])

    def speak(text, **kwargs):
        kwargs["on_start"]()
        return True

    tts.speak.side_effect = speak
    _router, _tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Raphael, explain streaming.", {})],
        router=router, tts=tts, streaming=True,
    )
    try:
        router.send.assert_not_called()
        assert [call.args[0] for call in tts.speak.call_args_list] == [
            "First sentence.", "Second sentence.",
        ]
        assert [(turn.role, turn.content) for turn in turns] == [
            ("user", "explain streaming."), ("assistant", "First sentence. Second sentence."),
        ]
        assert turns[-1].model == "stream-model"
    finally:
        store.close()


def test_canceled_stream_and_new_addition_preserve_combined_request(tmp_path, monkeypatch):
    from raphael.providers.base import LLMStreamChunk

    canceled, replacement = Event(), Event()
    router = MagicMock()

    def generate(messages, **kwargs):
        if router.stream.call_count == 1:
            canceled.set()
            yield LLMStreamChunk("Discard this.", "stream-model", "nim")
        else:
            yield LLMStreamChunk("A combined answer.", "stream-model", "nim")

    router.stream.side_effect = generate
    _router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Raphael, explain streaming.", {"cancel_event": canceled}),
         ("And include interruptions.", {
             "cancel_event": replacement, "supersedes_cancel_event": canceled,
         })], router=router, streaming=True,
    )
    try:
        router.send.assert_not_called()
        assert [turn.role for turn in turns] == ["user", "user", "assistant"]
        messages = router.stream.call_args.args[0]
        assert [message.content for message in messages if message.role == "user"] == [
            "explain streaming.\nAnd include interruptions.",
        ]
        tts.speak.assert_called_once()
    finally:
        store.close()


def test_live_stream_interrupted_after_audio_starts_archives_partial_reply(tmp_path, monkeypatch):
    from raphael.providers.base import LLMStreamChunk

    cancel = Event()
    router, tts = MagicMock(), MagicMock()
    router.stream.side_effect = lambda *args, **kwargs: iter([
        LLMStreamChunk("First sentence. Second sentence. ", "stream-model", "nim"),
    ])

    def speak(text, **kwargs):
        kwargs["on_start"]()
        cancel.set()
        return False

    tts.speak.side_effect = speak
    _router, _tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Raphael, explain streaming.", {"cancel_event": cancel}),
         ("", {"supersedes_cancel_event": cancel})],
        router=router, tts=tts, streaming=True,
    )
    try:
        assert [turn.role for turn in turns] == ["user", "assistant"]
        router.stream.assert_called_once()  # Empty noise must not restart partial playback.
        assert turns[-1].content.endswith("[Playback was interrupted.]")
        assert "First sentence." in turns[-1].content
        assert [call.args[0] for call in tts.speak.call_args_list] == ["First sentence."]
    finally:
        store.close()


def test_live_partial_stream_failure_speaks_notice_without_repeating_answer(tmp_path, monkeypatch):
    from raphael.providers.base import LLMStreamChunk

    router, tts = MagicMock(), MagicMock()

    def generate(*args, **kwargs):
        yield LLMStreamChunk("First sentence. ", "stream-model", "nim")
        raise RuntimeError("connection lost")

    def speak(text, **kwargs):
        if "on_start" in kwargs:
            kwargs["on_start"]()
        return True

    router.stream.side_effect = generate
    tts.speak.side_effect = speak
    _router, _tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Raphael, explain streaming.", {})],
        router=router, tts=tts, streaming=True,
    )
    try:
        assert [call.args[0] for call in tts.speak.call_args_list] == [
            "First sentence.", "My connection cut out before I finished. Ask me to continue.",
        ]
        assert turns[-1].content.endswith("[Generation ended early because the connection failed.]")
        router.send.assert_not_called()
    finally:
        store.close()


def test_interruption_during_long_reply_can_be_judged_after_deadline(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr('raphael.audio.ambient.time.monotonic', lambda: clock[0])
    ambient = AmbientConversation()
    ambient.record_addressed('assistant', 'Let me explain the next steps.')
    clock[0] = 130.0
    router = MagicMock()
    router.send.return_value = LLMResponse(
        '{"addressed": true, "confidence": 0.95}', 'test', 'test'
    )
    assert ambient.decide('Wait, also include this.', router, [], during_reply=True).addressed
    router.send.assert_called_once()
    ambient.reset()
    router.reset_mock()
    assert not ambient.decide('Dinner is ready.', router, [], during_reply=True).addressed
    router.send.assert_not_called()


def test_superseded_request_and_added_words_reach_one_response(tmp_path, monkeypatch):
    router = MagicMock()
    router.send.side_effect = [
        LLMResponse('{"addressed": true, "confidence": 0.95}', 'test', 'test'),
        LLMResponse('Here is the explanation, with headphones in mind.', 'test', 'test'),
    ]
    _router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, [('And include headphone support.', {})], router=router,
        observed=[('Raphael, explain always listening.', {'superseded': True})],
    )
    try:
        messages = router.send.call_args.args[0]
        users = [message.content for message in messages if message.role == 'user']
        assert users == ['explain always listening.\nAnd include headphone support.']
        assert 'same request' in messages[0].content
        tts.speak.assert_called_once()
        assert [turn.role for turn in turns] == ['user', 'user', 'assistant']
    finally:
        store.close()


def test_live_canceled_request_and_short_addition_are_merged(tmp_path, monkeypatch):
    canceled, replacement = Event(), Event()
    router = MagicMock()

    def respond(messages, **kwargs):
        assert kwargs.get('purpose', 'conversation') == 'conversation'
        if router.send.call_count == 1:
            canceled.set()  # Speech resumed while the first route was in flight.
            return LLMResponse('Discard this stale response.', 'test', 'test')
        return LLMResponse('Here is what I am good at.', 'test', 'test')

    router.send.side_effect = respond
    _router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [
            ('Hey Raphael, what is it that you are special?', {'cancel_event': canceled}),
            ('And good.', {
                'cancel_event': replacement, 'supersedes_cancel_event': canceled,
                'response_pending': True, 'during_reply': True, 'stt_confidence': 0.54,
            }),
        ], router=router,
    )
    try:
        assert router.send.call_count == 2
        user_messages = [
            message.content for message in router.send.call_args.args[0] if message.role == 'user'
        ]
        assert user_messages == ['what is it that you are special?\nAnd good.']
        assert [turn.content for turn in turns if turn.role == 'user'] == [
            'what is it that you are special?', 'And good.',
        ]
        assert [turn.content for turn in turns if turn.role == 'assistant'] == [
            'Here is what I am good at.'
        ]
        tts.speak.assert_called_once_with(
            'Here is what I am good at.', block=True, cancel_event=replacement,
        )
    finally:
        store.close()


def test_linked_superseded_stt_request_is_merged_before_reply_gate(tmp_path, monkeypatch):
    canceled, replacement = Event(), Event()
    canceled.set()
    router = MagicMock()
    router.send.return_value = LLMResponse('Together, this is the full answer.', 'test', 'test')
    _router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [('Also with headphones.', {
            'cancel_event': replacement, 'supersedes_cancel_event': canceled,
            'response_pending': True,
        })],
        router=router,
        observed=[('Raphael, explain always listening.', {
            'superseded': True, 'cancel_event': canceled,
        })],
    )
    try:
        router.send.assert_called_once()
        assert router.send.call_args.args[0][-1].content == (
            'explain always listening.\nAlso with headphones.'
        )
        assert [turn.role for turn in turns] == ['user', 'user', 'assistant']
        tts.speak.assert_called_once()
    finally:
        store.close()


def test_repeated_additions_keep_the_original_request(tmp_path, monkeypatch):
    first, second, third = Event(), Event(), Event()
    router = MagicMock()

    def respond(_messages, **kwargs):
        assert kwargs.get('purpose', 'conversation') == 'conversation'
        if router.send.call_count == 1:
            first.set()
        elif router.send.call_count == 2:
            second.set()
        return LLMResponse('Here is the combined answer.', 'test', 'test')

    router.send.side_effect = respond
    _router, tts, _loop, store, _turns = run_callbacks(
        tmp_path, monkeypatch,
        [('Raphael, explain this in five steps.', {'cancel_event': first}),
         ('And include examples.', {'cancel_event': second, 'supersedes_cancel_event': first}),
         ('No, three steps, not five.', {
             'cancel_event': third, 'supersedes_cancel_event': second,
         })], router=router,
    )
    try:
        assert router.send.call_count == 3
        users = [
            message.content for message in router.send.call_args.args[0] if message.role == 'user'
        ]
        assert users == [
            'explain this in five steps.\nAnd include examples.\nNo, three steps, not five.'
        ]
        tts.speak.assert_called_once()
    finally:
        store.close()


def test_unrelated_recording_cannot_inherit_canceled_request(tmp_path, monkeypatch):
    canceled, other_token = Event(), Event()
    router = MagicMock()

    def respond(_messages, **kwargs):
        if kwargs.get('purpose') == 'speech_gate':
            return LLMResponse('{"addressed": false, "confidence": 0.95}', 'test', 'test')
        canceled.set()
        return LLMResponse('Discard this.', 'test', 'test')

    router.send.side_effect = respond
    _router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [('Raphael, explain this.', {'cancel_event': canceled}), ('And good.', {
            'supersedes_cancel_event': other_token, 'response_pending': True,
        })], router=router,
    )
    try:
        assert router.send.call_args.kwargs['purpose'] == 'speech_gate'
        assert [turn.role for turn in turns] == ['user']
        tts.speak.assert_not_called()
    finally:
        store.close()


def test_merge_permission_does_not_override_other_listener_or_save_facts(tmp_path, monkeypatch):
    canceled = Event()
    canceled.set()
    router = MagicMock()
    router.send.return_value = LLMResponse('I can take that into account.', 'test', 'test')
    _router, tts, _loop, store, _turns = run_callbacks(
        tmp_path, monkeypatch,
        [('Actually, Mom, pass me that.', {'supersedes_cancel_event': canceled}),
         ('Actually my favorite game is CS2.', {'supersedes_cancel_event': canceled})],
        router=router, observed=[('Raphael, explain this.', {
            'superseded': True, 'cancel_event': canceled,
        })],
    )
    try:
        assert store.get_fact('user:favorite_game') is None
        tts.speak.assert_called_once()
        router.send.assert_called_once()
    finally:
        store.close()


def test_continue_receives_interrupted_playback_context(tmp_path, monkeypatch):
    router = MagicMock()
    router.send.return_value = LLMResponse('And the second step is to test it.', 'test', 'test')
    context = {
        'full_text': 'First, configure it. Second, test it.',
        'estimated_spoken_text': 'First, configure it.',
        'remaining_text': 'Second, test it.',
        'played_seconds': 2.0, 'duration_seconds': 4.0,
    }
    _router, tts, _loop, store, _turns = run_callbacks(
        tmp_path, monkeypatch, [('Raphael, continue.', {})], router=router, interrupted=context,
    )
    try:
        system = router.send.call_args.args[0][0].content
        assert 'previous reply was interrupted' in system
        assert 'Second, test it.' in system
        assert 'word boundaries are estimated' in system
        tts.speak.assert_called_once()
    finally:
        store.close()


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


@pytest.mark.parametrize("ambient", [True, False])
def test_empty_interruption_restores_superseded_question(tmp_path, monkeypatch, ambient):
    canceled, replacement = Event(), Event()
    canceled.set()
    router = MagicMock()
    router.send.return_value = LLMResponse('Here is what I can do.', 'test', 'test')
    _router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [('', {'cancel_event': replacement, 'supersedes_cancel_event': canceled})],
        ambient=ambient, router=router,
        observed=[('Raphael, what are you good at?', {
            'superseded': True, 'cancel_event': canceled,
        })],
    )
    try:
        router.send.assert_called_once()
        assert router.send.call_args.args[0][-1].content == 'what are you good at?'
        assert [(turn.role, turn.content) for turn in turns] == [
            ('user', 'what are you good at?'), ('assistant', 'Here is what I can do.'),
        ]
        tts.speak.assert_called_once()
    finally:
        store.close()


def test_empty_interruption_retries_without_duplicate_history(tmp_path, monkeypatch):
    canceled, replacement = Event(), Event()
    router = MagicMock()

    def respond(messages, **kwargs):
        if router.send.call_count == 1:
            canceled.set()
            return LLMResponse('Discard me.', 'test', 'test')
        return LLMResponse('The answer.', 'test', 'test')

    router.send.side_effect = respond
    _router, tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [('Raphael, explain this.', {'cancel_event': canceled}),
         ('', {'cancel_event': replacement, 'supersedes_cancel_event': canceled})],
        router=router,
    )
    try:
        assert router.send.call_count == 2
        assert [m.content for m in router.send.call_args.args[0] if m.role == 'user'] == [
            'explain this.',
        ]
        assert [(turn.role, turn.content) for turn in turns] == [
            ('user', 'explain this.'), ('assistant', 'The answer.'),
        ]
        tts.speak.assert_called_once()
    finally:
        store.close()


@pytest.mark.parametrize('case', ['unlinked', 'expired', 'unclear'])
def test_empty_audio_does_not_restore_unrelated_or_unclear_requests(tmp_path, monkeypatch, case):
    canceled = Event()
    canceled.set()
    info = {'supersedes_cancel_event': canceled}
    clock = None
    if case == 'unlinked':
        info['supersedes_cancel_event'] = Event()
    elif case == 'unclear':
        info.update(stt_raw_text='possibly stop', stt_needs_repeat=True)
    else:
        # observe_transcript records at time zero; restoration occurs after expiry.
        times = iter([0.0, 1000.0])
        def clock():
            return next(times)
    router = MagicMock()
    router.send.return_value = LLMResponse('{"addressed": false}', 'test', 'test')
    _router, _tts, _loop, store, turns = run_callbacks(
        tmp_path, monkeypatch, [('', info)], router=router, clock=clock,
        observed=[('Raphael, explain this.', {'superseded': True, 'cancel_event': canceled})],
    )
    try:
        assert all(
            call.kwargs.get('purpose') == 'speech_gate' for call in router.send.call_args_list
        )
        assert turns == []
    finally:
        store.close()


@pytest.mark.parametrize('ambient', [True, False])
def test_empty_recovery_cannot_execute_a_memory_write(tmp_path, monkeypatch, ambient):
    canceled = Event()
    canceled.set()
    handler = MagicMock(return_value=None)
    monkeypatch.setattr('raphael.memory.service.MemoryService.handle', handler)
    router = MagicMock()
    router.send.return_value = LLMResponse('I heard your request.', 'test', 'test')
    _router, _tts, _loop, store, _turns = run_callbacks(
        tmp_path, monkeypatch, [('', {'supersedes_cancel_event': canceled})],
        ambient=ambient, router=router,
        observed=[('Raphael, remember my name is Alex.', {
            'superseded': True, 'cancel_event': canceled,
        })],
    )
    try:
        handler.assert_not_called()
        assert store.get_fact('user:preferred_name') is None
    finally:
        store.close()
