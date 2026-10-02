import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

SOURCE = str(Path(__file__).resolve().parents[1] / "src")


def test_listen_wake_only_acknowledges_without_provider_request(tmp_path, monkeypatch):
    from raphael import __main__, audio, config, platform, providers

    settings = config.Settings(_env_file=None, memory_db_path=str(tmp_path / "memory.db"))
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(platform, "get_audio_backend", MagicMock())
    router = MagicMock()
    monkeypatch.setattr(providers, "get_model_router", lambda: router)
    tts = MagicMock()
    for component in ("WakeWordDetector", "SpeechToText", "VoiceRecorder"):
        monkeypatch.setattr(audio, component, MagicMock())
    monkeypatch.setattr(audio, "TextToSpeech", MagicMock(return_value=tts))
    followup = []

    def listener(**kwargs):
        loop = SimpleNamespace(is_running=True, stop=MagicMock())
        loop.start = lambda: followup.append(kwargs["on_transcription"]("Hey Raphael.", {}, None))
        return loop

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(audio, "WakeListenerLoop", listener)
    monkeypatch.setattr(__main__.time, "sleep", interrupt)
    monkeypatch.setattr(sys, "argv", ["raphael", "--listen"])
    assert __main__.main() == 0
    assert followup == [True]
    router.send.assert_not_called()
    tts.speak.assert_called_once_with("Hey! What's on your mind?", block=True)


def test_help_does_not_import_audio_or_training(tmp_path):
    script = """
import runpy, sys
sys.argv = ["raphael", "--help"]
try:
    runpy.run_module("raphael", run_name="__main__")
except SystemExit as error:
    assert error.code == 0
assert "sounddevice" not in sys.modules
assert "raphael.audio.trainer" not in sys.modules
assert "faster_whisper" not in sys.modules
assert "onnx" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": SOURCE},
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert "--listen" in result.stdout


def test_train_voice_bad_dataset_reports_error_without_audio(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "raphael", "train-voice", "--output", str(tmp_path)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": SOURCE},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert "metadata.csv" in result.stdout
    assert "Traceback" not in result.stderr


def test_training_imports_are_optional_and_missing_extras_are_actionable(tmp_path):
    script = """
import importlib.abc, sys
class NoTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"onnx", "sklearn"}:
            raise ImportError("training extra not installed")
sys.meta_path.insert(0, NoTraining())
from raphael.audio import VoiceRecorder
from raphael.audio.trainer import train_custom_wakeword
assert "sounddevice" not in sys.modules
try:
    train_custom_wakeword()
except RuntimeError as error:
    assert 'pip install -e ".[train]"' in str(error)
else:
    raise AssertionError("Missing dependency was not reported")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": SOURCE},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
