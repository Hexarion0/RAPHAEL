"""Unit tests for Text-to-Speech (TTS) engine."""

from unittest.mock import MagicMock, patch
import numpy as np

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
    tts_mommy = TextToSpeech(voice_name="mommy", enabled=False)
    assert tts_mommy.engine == "fish_speech"

    # edge_tts voice auto-detects
    tts_edge = TextToSpeech(voice_name="en-US-AvaNeural", enabled=False)
    assert tts_edge.engine == "edge_tts"

    # piper voice auto-detects
    tts_piper = TextToSpeech(voice_name="en_GB-alan-medium", enabled=False)
    assert tts_piper.engine == "piper"


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
    with patch.object(tts, "_fallback_synthesize", return_value=dummy_audio) as mock_fb:
        result = tts.synthesize("Testing fallback mechanism.")
        assert result is not None
        mock_fb.assert_called_once()


def test_tts_disabled():
    tts = TextToSpeech(enabled=False)
    assert tts.synthesize("Hello") is None
    assert tts.speak("Hello") is False
