"""Unit tests for Text-to-Speech (TTS) engine."""

import numpy as np

from raphael.audio.tts import TextToSpeech


def test_clean_text_for_speech():
    raw_text = """
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
    assert "#" not in cleaned
    assert "https://openai.com" not in cleaned
    assert "🚀" not in cleaned
    assert "🔥" not in cleaned
    assert "code inline" in cleaned
    assert "Bold text" in cleaned


def test_tts_initialization_and_synthesis():
    tts = TextToSpeech(voice_name="en_GB-alan-medium", enabled=True)
    assert tts.enabled is True
    assert tts.voice_name == "en_GB-alan-medium"

    result = tts.synthesize("Hello sir, RAPHAEL system is online.")
    assert result is not None
    audio, sample_rate = result
    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1
    assert audio.size > 0
    assert sample_rate == 22050


def test_tts_disabled():
    tts = TextToSpeech(enabled=False)
    assert tts.synthesize("Hello") is None
    assert tts.speak("Hello") is False
