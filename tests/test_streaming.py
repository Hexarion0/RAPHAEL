"""Concurrency regressions for first-sentence playback and canceled replies."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from raphael.audio.streaming import SentenceBuffer, VisibleText, stream_reply
from raphael.audio.tts import TextToSpeech
from raphael.providers.base import ChatMessage, LLMStreamChunk


@pytest.fixture
def speech(monkeypatch):
    tts = TextToSpeech(enabled=False)
    tts.enabled = True
    played = []
    monkeypatch.setattr(tts, "synthesize", lambda text: (
        np.ones(160, dtype=np.float32), 16000,
    ))
    monkeypatch.setattr("raphael.audio.tts.sd", SimpleNamespace(
        play=lambda *args, **kwargs: played.append(1),
        get_stream=lambda: SimpleNamespace(active=False), stop=lambda: None,
    ))
    return tts, played


def chunk(text, final=False):
    return LLMStreamChunk(text, "test-model", "test-provider", final)


def test_delimiters_split_across_chunks_never_speak_thinking_or_code():
    visible = VisibleText()
    pieces = ["<thi", "nk>private. <reflection>secret</ref", "lection></think>Answer. ",
              "``", "`python\nprint('private')\n`", "`` Done."]
    output = "".join(visible.feed(piece) for piece in pieces) + visible.feed("", final=True)
    assert output == "Answer.  [code omitted]  Done."
    assert "private" not in output and "secret" not in output
    hidden = VisibleText()
    assert hidden.feed("<reasoning>unclosed secret.", final=True) == ""


def test_sentence_boundaries_keep_decimal_and_abbreviation():
    buffer = SentenceBuffer()
    assert buffer.feed("Dr. Smith measured 3.") == []
    assert buffer.feed("14 volts. Next") == ["Dr. Smith measured 3.14 volts."]
    assert buffer.feed(" sentence", final=True) == ["Next sentence"]
    assert buffer.feed("1. First item\n2. Second item", final=True) == [
        "1. First item", "2. Second item",
    ]


def test_long_unpunctuated_speech_is_bounded_without_losing_words():
    text = " ".join(f"word{index}" for index in range(100))
    parts = SentenceBuffer(max_chars=80).feed(text, final=True)
    assert len(parts) > 1 and max(map(len, parts)) <= 80
    assert " ".join(parts) == text


def test_first_sentence_plays_before_provider_finishes(speech, monkeypatch):
    tts, played = speech
    first_audio, generation_finished = threading.Event(), threading.Event()

    def play(*args, **kwargs):
        assert not generation_finished.is_set() if not played else True
        played.append(1)
        first_audio.set()

    monkeypatch.setattr("raphael.audio.tts.sd.play", play)

    def generate(*args, **kwargs):
        yield chunk("First sentence. ")
        assert first_audio.wait(1), "Playback waited for full generation"
        generation_finished.set()
        yield chunk("Second sentence.")
        yield chunk("", final=True)

    result = stream_reply(SimpleNamespace(stream=generate), [ChatMessage("user", "Explain.")],
                          tts, lambda: True)
    assert result.response.content == "First sentence. Second sentence."
    assert result.response.provider == "test-provider"
    assert result.spoken and not result.canceled and len(played) == 2
    assert result.first_token_seconds is not None and result.first_audio_seconds is not None
    assert tts.stop() is None


def test_sentence_callback_arrives_at_playback_before_generation_finishes(speech):
    tts, played = speech
    first_sentence, generation_finished = threading.Event(), threading.Event()
    observed = []

    def sentence_started(text):
        assert len(played) == len(observed) + 1, "Transcript preceded audio playback"
        if not observed:
            assert not generation_finished.is_set(), "Transcript waited for full generation"
        observed.append(text)
        first_sentence.set()

    def generate(*args, **kwargs):
        yield chunk("First sentence. ")
        assert first_sentence.wait(1), "No first-sentence transcript during generation"
        generation_finished.set()
        yield chunk("Second sentence.")
        yield chunk("", final=True)

    result = stream_reply(
        SimpleNamespace(stream=generate), [], tts, lambda: True,
        on_sentence_start=sentence_started,
    )
    assert observed == ["First sentence.", "Second sentence."]
    assert result.spoken and not result.canceled
    assert generation_finished.is_set()


@pytest.mark.parametrize("cancel_sentence", ["First.", "Second."])
def test_canceled_synthesis_does_not_report_canceled_or_queued_sentences(
    speech, monkeypatch, cancel_sentence,
):
    tts, played = speech
    entered, release, cancel, finished, synthesized = (threading.Event() for _ in range(5))
    observed, results = [], []

    def synthesize(text):
        if text == cancel_sentence:
            entered.set()
            try:
                assert release.wait(2)
            finally:
                synthesized.set()
        return np.ones(160, dtype=np.float32), 16000

    monkeypatch.setattr(tts, "synthesize", synthesize)
    router = SimpleNamespace(stream=lambda *a, **k: iter([chunk("First. Second. Third. ")]))

    def run():
        results.append(stream_reply(
            router, [], tts, lambda: not cancel.is_set(), cancel_event=cancel,
            on_sentence_start=observed.append,
        ))
        finished.set()

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(1)
        cancel.set()
        tts.stop()
        assert finished.wait(0.5), "Canceled synthesis delayed the new utterance"
    finally:
        release.set()
        worker.join(2)
        assert synthesized.wait(1)
    assert results[0].canceled
    assert observed == ([] if cancel_sentence == "First." else ["First."])
    assert len(played) == len(observed)


def test_cancel_during_blocked_provider_returns_without_waiting(speech):
    tts, played = speech
    entered, release, closed, done, cancel = (threading.Event() for _ in range(5))

    def generate(*args, **kwargs):
        entered.set()
        try:
            assert release.wait(2)
            yield chunk("This stale reply must not play.")
        finally:
            closed.set()

    results = []

    def run():
        results.append(stream_reply(SimpleNamespace(stream=generate), [], tts,
                                    lambda: not cancel.is_set(), cancel_event=cancel))
        done.set()

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(1)
        cancel.set()
        assert done.wait(0.5), "Canceled network read blocked the next utterance"
        assert results[0].canceled and not results[0].response.content
        assert played == []
    finally:
        release.set()
        worker.join(2)
        assert closed.wait(1)


def test_cancel_during_synthesis_discards_queued_sentences(speech, monkeypatch):
    tts, played = speech
    entered, release, cancel, finished = (threading.Event() for _ in range(4))

    def synthesize(text):
        entered.set()
        assert release.wait(2)
        return np.ones(160), 16000

    monkeypatch.setattr(tts, "synthesize", synthesize)
    results = []
    router = SimpleNamespace(stream=lambda *a, **k: iter([chunk("First. Second. ")]))
    def run():
        results.append(stream_reply(
            router, [], tts, lambda: not cancel.is_set(), cancel_event=cancel,
        ))
        finished.set()

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(1)
        cancel.set()
        tts.stop()
        assert finished.wait(0.5), "Canceled synthesis blocked the next utterance"
    finally:
        release.set()
        worker.join(2)
    assert results[0].canceled and played == []


def test_cancel_during_playback_stops_rest_of_reply(speech, monkeypatch):
    tts, played = speech
    cancel = threading.Event()

    def play(*args, **kwargs):
        played.append(1)
        cancel.set()

    monkeypatch.setattr("raphael.audio.tts.sd.play", play)
    monkeypatch.setattr("raphael.audio.tts.sd.get_stream", lambda: SimpleNamespace(active=True))
    router = SimpleNamespace(stream=lambda *a, **k: iter([chunk("First. Second. ")]))
    result = stream_reply(router, [], tts, lambda: not cancel.is_set(), cancel_event=cancel)
    assert result.canceled and len(played) == 1
    assert result.first_audio_seconds is not None
    assert not tts.is_speaking()


def test_partial_failure_keeps_answer_and_does_not_restart(speech):
    tts, played = speech

    def generate(*args, **kwargs):
        yield chunk("First sentence. ")
        raise RuntimeError("Connection dropped")

    result = stream_reply(SimpleNamespace(stream=generate), [], tts, lambda: True)
    assert result.response.content == "First sentence."
    assert isinstance(result.error, RuntimeError) and len(played) == 1


def test_code_stays_in_history_but_is_omitted_from_spoken_reply(speech, monkeypatch):
    tts, _played = speech
    synthesized = []
    original = tts.synthesize

    def synthesize(text):
        synthesized.append(text)
        return original(text)

    monkeypatch.setattr(tts, "synthesize", synthesize)
    text = "Here is the code.\n```python\nx = 1\n```\nThat is all."
    router = SimpleNamespace(stream=lambda *a, **k: iter([chunk(text)]))
    result = stream_reply(router, [], tts, lambda: True)
    assert result.response.content == text
    spoken = " ".join(synthesized)
    assert "x = 1" not in spoken and "python" not in spoken
    assert "[code omitted]" in spoken and "That is all." in spoken


def test_sentence_transcripts_filter_reasoning_and_code_across_chunk_boundaries(speech):
    tts, _played = speech
    observed = []
    pieces = [
        "<thi", "nk>Private reasoning. <reflection>Secret.</ref",
        "lection></think>Here is the code.\n", "``", "`python\nprivate_value = 1\n`",
        "``\nThat is all.",
    ]
    router = SimpleNamespace(stream=lambda *a, **k: (chunk(text) for text in pieces))
    result = stream_reply(router, [], tts, lambda: True, on_sentence_start=observed.append)
    assert observed == ["Here is the code.", "[code omitted]", "That is all."]
    transcripts = " ".join(observed)
    assert "Private" not in transcripts and "Secret" not in transcripts
    assert "private_value" not in transcripts and "python" not in transcripts
    assert "private_value = 1" in result.response.content
    assert "Private" not in result.response.content and "Secret" not in result.response.content


def test_empty_or_reasoning_only_reply_never_plays(speech):
    tts, played = speech
    router = SimpleNamespace(stream=lambda *a, **k: iter([chunk("<think>Secret.</think>")]))
    with pytest.raises(RuntimeError, match="without a visible reply"):
        stream_reply(router, [], tts, lambda: True)
    assert played == []


def test_stream_interruption_reports_completed_and_current_sentences(speech, monkeypatch):
    tts, _played = speech
    clock = [100.0]
    monkeypatch.setattr("raphael.audio.tts.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(tts, "synthesize", lambda text: (np.ones(100), 10))
    generation = tts.begin_stream()
    tts.update_stream(generation, "First complete sentence. One two three four. Last sentence.")
    assert tts.speak("First complete sentence.", generation=generation)
    monkeypatch.setattr("raphael.audio.tts.sd.get_stream", lambda: SimpleNamespace(active=True))
    assert tts.speak("One two three four.", generation=generation, block=False)
    clock[0] += 5
    interrupted = tts.stop()
    assert interrupted["estimated_spoken_text"] == "First complete sentence. One two"
    assert interrupted["remaining_text"] == "three four. Last sentence."
    assert interrupted["played_seconds"] == 15
    assert tts.stop() is None
    assert not tts.speak("Last sentence.", generation=generation)


def test_disabled_speech_still_consumes_and_returns_reply():
    tts = TextToSpeech(enabled=False)
    router = SimpleNamespace(stream=lambda *a, **k: iter([chunk("A text-only reply.")]))
    result = stream_reply(router, [], tts, lambda: True)
    assert result.response.content == "A text-only reply." and not result.spoken


def test_interruption_between_sentences_does_not_count_previous_audio_twice(speech, monkeypatch):
    tts, _played = speech
    clock = [100.0]
    monkeypatch.setattr("raphael.audio.tts.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(tts, "synthesize", lambda text: (np.ones(100), 10))
    generation = tts.begin_stream()
    tts.update_stream(generation, "First sentence. Second sentence.")
    assert tts.speak("First sentence.", generation=generation)
    clock[0] += 15  # Finished audio plus a pause while the provider generates more.
    interrupted = tts.stop()
    assert interrupted["estimated_spoken_text"] == "First sentence."
    assert interrupted["remaining_text"] == "Second sentence."
    assert interrupted["played_seconds"] == interrupted["duration_seconds"] == 10
