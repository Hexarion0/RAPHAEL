"""Unit tests for Text-to-Speech (TTS) engine."""

from unittest.mock import patch

import numpy as np
import pytest

from raphael.audio.tts import TextToSpeech


def test_clean_text_for_speech():
    raw_text = """
    <think>Internal reasoning here that should be stripped</think>
    # Header 1
    Here is some `code inline` and a link [OpenAI](https://openai.com).
    ```python
    def test():
        pass
    ```
    * Bullet 1
    1. Numbered item
    **Bold text** and *italic* and emojis 🚀🔥
    """
    cleaned = TextToSpeech.clean_text_for_speech(raw_text)
    assert "Internal reasoning" not in cleaned
    assert "#" not in cleaned
    assert "https://openai.com" not in cleaned
    assert "🚀" not in cleaned
    assert "🔥" not in cleaned
    assert "code inline" in cleaned
    assert "Bold text" in cleaned


def test_tts_engine_detection():
    # mommy voice auto-selects fish_speech
    tts_mommy = TextToSpeech(voice_name="mommy", enabled=False, engine="auto")
    assert tts_mommy.engine == "fish_speech"

    # edge_tts voice auto-detects
    tts_edge = TextToSpeech(voice_name="en-US-AvaNeural", enabled=False, engine="auto")
    assert tts_edge.engine == "edge_tts"

    # piper voice auto-detects
    tts_piper = TextToSpeech(voice_name="en_GB-alan-medium", enabled=False, engine="auto")
    assert tts_piper.engine == "piper"


@pytest.mark.integration
def test_piper_initialization_and_synthesis():
    tts = TextToSpeech(voice_name="en_GB-alan-medium", engine="piper", enabled=True)
    assert tts.enabled is True
    assert tts.voice_name == "en_GB-alan-medium"

    result = tts.synthesize("Hello sir, RAPHAEL system is online.")
    assert result is not None
    audio, sample_rate = result
    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1
    assert audio.size > 0
    assert sample_rate == 22050


def test_fish_speech_fallback_when_offline():
    # Fish speech with invalid port should gracefully fall back to Edge-TTS or Piper
    tts = TextToSpeech(
        voice_name="mommy",
        engine="fish_speech",
        fish_speech_url="http://127.0.0.1:9999/v1/tts",
        enabled=True,
    )
    # Mock fallback to avoid external network dependencies during unit tests
    dummy_audio = (np.zeros(16000, dtype=np.float32), 16000)
    with (
        patch("raphael.audio.tts.requests.post", side_effect=ConnectionError("offline")),
        patch.object(tts, "_fallback_synthesize", return_value=dummy_audio) as mock_fb,
    ):
        result = tts.synthesize("Testing fallback mechanism.")
        assert result is not None
        mock_fb.assert_called_once()


def test_tts_disabled():
    tts = TextToSpeech(enabled=False)
    assert tts.synthesize("Hello") is None
    assert tts.speak("Hello") is False


def test_stop_during_synthesis_prevents_late_playback(monkeypatch):
    import threading
    from types import SimpleNamespace

    entered, release = threading.Event(), threading.Event()
    tts = TextToSpeech(enabled=False)
    tts.enabled = True
    played = []

    def synthesize(_text):
        entered.set()
        assert release.wait(2)
        return np.zeros(16000, dtype=np.float32), 16000

    monkeypatch.setattr(tts, "synthesize", synthesize)
    monkeypatch.setattr(
        "raphael.audio.tts.sd", SimpleNamespace(play=lambda *_a, **_k: played.append(1))
    )
    results = []
    worker = threading.Thread(target=lambda: results.append(tts.speak("hello")))
    worker.start()
    try:
        assert entered.wait(1)
        interruption = tts.stop()
    finally:
        release.set()
        worker.join(timeout=2)
    assert results == [False]
    assert played == []
    assert interruption['remaining_text'] == 'hello'
    assert interruption['estimated_spoken_text'] == ''
    assert interruption['played_seconds'] == 0


def test_interrupted_playback_reports_estimated_unheard_text(monkeypatch):
    from types import SimpleNamespace

    clock = [100.0]
    monkeypatch.setattr('raphael.audio.tts.time.monotonic', lambda: clock[0])
    monkeypatch.setattr('raphael.audio.tts.sd', SimpleNamespace(
        play=lambda *_args, **_kwargs: None,
        get_stream=lambda: SimpleNamespace(active=True), stop=lambda: None,
    ))
    tts = TextToSpeech(enabled=False)
    tts.enabled = True
    monkeypatch.setattr(tts, 'synthesize', lambda _: (np.ones(80000), 8000))
    assert tts.speak('one two three four five six seven eight nine ten', block=False)
    clock[0] += 5
    interruption = tts.stop()
    assert interruption['estimated_spoken_text'] == 'one two three four five'
    assert interruption['remaining_text'] == 'six seven eight nine ten'
    assert interruption['played_seconds'] == 5
    assert interruption['duration_seconds'] == 10
    assert not tts.is_speaking()
    assert tts.stop() is None  # Do not replay an old interruption on another stop.


def test_completed_playback_does_not_report_an_interruption(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr('raphael.audio.tts.sd', SimpleNamespace(
        play=lambda *_args, **_kwargs: None, get_stream=lambda: SimpleNamespace(active=False),
        stop=lambda: None,
    ))
    tts = TextToSpeech(enabled=False)
    tts.enabled = True
    monkeypatch.setattr(tts, 'synthesize', lambda _: (np.ones(8000), 8000))
    assert tts.speak('This reply finished.')
    assert tts.stop() is None


def test_omitted_engine_uses_selected_piper_voice(monkeypatch):
    from raphael.config import Settings

    settings = Settings(_env_file=None, tts_voice="en_US-amy-medium", tts_engine="auto")
    monkeypatch.setattr("raphael.audio.tts.get_settings", lambda: settings)
    tts = TextToSpeech(
        voice_name=settings.audio.tts_voice, engine=settings.audio.tts_engine, enabled=False
    )
    tts.enabled = True
    expected = np.ones(100, dtype=np.float32), 22050
    with (
        patch.object(tts, "_synthesize_piper", return_value=expected) as local,
        patch.object(tts, "_synthesize_fish_speech") as fish,
        patch.object(tts, "_synthesize_edge_tts") as edge,
    ):
        assert tts.synthesize("I'm here.") is expected
    local.assert_called_once()
    fish.assert_not_called()
    edge.assert_not_called()


def test_local_fallback_is_used_and_cached_before_network(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    (tmp_path / "en_GB-alan-medium.onnx").write_bytes(b"model")
    (tmp_path / "en_GB-alan-medium.onnx.json").write_text("{}")
    tts = TextToSpeech(voice_name="mommy", engine="fish_speech", models_dir=tmp_path, enabled=False)
    tts.enabled = True
    voice = MagicMock()
    audio = np.ones(100, dtype=np.float32), 22050
    with (
        patch("raphael.audio.tts.PiperVoice.load", return_value=voice) as load,
        patch.object(tts, "_synthesize_piper", return_value=audio) as local,
        patch.object(tts, "_synthesize_edge_tts") as edge,
    ):
        assert tts._fallback_synthesize("first") is audio
        assert tts._fallback_synthesize("second") is audio
    assert load.call_count == 1
    assert local.call_count == 2
    edge.assert_not_called()
    assert tts.enabled


@pytest.mark.parametrize(
    "voice, engine, expected",
    [
        ("en_US-amy-medium", "auto", "piper"),
        ("custom_voice", "auto", "piper"),
        ("en-US-AvaNeural", "auto", "edge_tts"),
        ("mommy", "auto", "fish_speech"),
        ("en_US-amy-medium", "fish_speech", "fish_speech"),
        ("mommy", "piper", "piper"),
    ],
)
def test_engine_resolution_respects_explicit_selection(voice, engine, expected):
    from raphael.audio.tts import resolve_tts_engine

    assert resolve_tts_engine(voice, engine) == expected


def test_explicit_output_device_numeric_string_and_name():
    for value, expected in [("3", 3), ("USB Speaker", "USB Speaker")]:
        tts = TextToSpeech(enabled=False, output_device=value)
        assert tts.output_device == expected
