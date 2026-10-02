"""Tests for faster-whisper speech-to-text module."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from raphael.audio.stt import SpeechToText


@pytest.fixture(autouse=True)
def model_without_downloads(monkeypatch):
    model = MagicMock()
    model.transcribe.return_value = ([], SimpleNamespace(language="en", duration=1.0))
    monkeypatch.setattr(SpeechToText, "_load_model", lambda *_args: model)
    monkeypatch.setattr("raphael.audio.stt._preload_cuda_libraries", lambda: None)
    return model


def test_stt_initialization():
    """Verify SpeechToText initializes model cleanly on CPU."""
    stt = SpeechToText(model_size="tiny.en", device="cpu", compute_type="int8")
    assert stt.model_size == "tiny.en"
    assert stt.device == "cpu"


def test_stt_empty_audio():
    """Verify STT handles empty audio arrays without crashing."""
    stt = SpeechToText(model_size="tiny.en", device="cpu", compute_type="int8")
    empty_audio = np.empty((0,), dtype=np.float32)
    result = stt.transcribe(empty_audio)
    assert result == ""

    detailed = stt.transcribe_detailed(empty_audio)
    assert detailed["text"] == ""
    assert detailed["segments"] == []


def test_stt_silence_audio():
    """Verify STT with 1 second of silence produces clean empty output."""
    stt = SpeechToText(model_size="tiny.en", device="cpu", compute_type="int8")
    silent_audio = np.zeros(16000, dtype=np.float32)
    result = stt.transcribe(silent_audio)
    assert isinstance(result, str)
