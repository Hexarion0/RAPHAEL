import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

SOURCE = str(Path(__file__).resolve().parents[1] / "src")


@pytest.mark.parametrize('arguments, enabled', [([], True), (
    ['--no-ambient', '--no-show-transcripts'], False,
)])
def test_env_listening_defaults_and_cli_overrides(tmp_path, monkeypatch, arguments, enabled):
    from raphael import __main__, audio, config, platform, providers

    settings = config.Settings(
        _env_file=None, memory_db_path=str(tmp_path / 'memory.db'),
        ambient_listening=True, show_transcripts=True, barge_in_mode='speech',
    )
    monkeypatch.setattr(config, 'get_settings', lambda: settings)
    monkeypatch.setattr(platform, 'get_audio_backend', MagicMock())
    monkeypatch.setattr(providers, 'get_model_router', MagicMock())
    for component in ('WakeWordDetector', 'SpeechToText', 'VoiceRecorder', 'TextToSpeech'):
        monkeypatch.setattr(audio, component, MagicMock())
    constructor = MagicMock(return_value=SimpleNamespace(
        is_running=True, start=lambda: None, stop=lambda: None,
    ))
    monkeypatch.setattr(audio, 'WakeListenerLoop', constructor)

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(__main__, 'time', SimpleNamespace(
        sleep=interrupt, monotonic=__main__.time.monotonic,
    ))
    monkeypatch.setattr(sys, 'argv', ['raphael', *arguments])
    assert __main__.main() == 0
    assert constructor.call_args.kwargs['ambient'] is enabled
    assert constructor.call_args.kwargs['show_transcripts'] is enabled
    assert constructor.call_args.kwargs['barge_in_mode'] == 'speech'


@pytest.mark.parametrize(
    "text, wake_info, expected_reply",
    [
        ("Hey Raphael.", {}, "Hey! What's on your mind?"),
        ("", {"stt_needs_repeat": True}, "I didn't catch that clearly. Could you say it again?"),
        ("So what's the time right now?", {}, "It's 3:16 AM."),
        ("My favorite game is CS2.", {}, "I've saved that. Your favorite game is CS2."),
    ],
)
def test_listen_wake_or_uncertain_speech_does_not_call_provider(
    tmp_path, monkeypatch, text, wake_info, expected_reply
):
    from datetime import datetime

    from raphael import __main__, audio, config, platform, providers
    from raphael.providers import intents

    clock_answer = intents.answer_clock_query
    monkeypatch.setattr(
        intents,
        "answer_clock_query",
        lambda text: clock_answer(text, datetime(2026, 10, 3, 3, 16)),
    )

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
        loop.start = lambda: followup.append(kwargs["on_transcription"](text, wake_info, None))
        return loop

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(audio, "WakeListenerLoop", listener)
    monkeypatch.setattr(__main__, "time", SimpleNamespace(
        sleep=interrupt, monotonic=__main__.time.monotonic,
    ))
    monkeypatch.setattr(sys, "argv", ["raphael", "--listen"])
    assert __main__.main() == 0
    assert followup == [True]
    router.send.assert_not_called()
    tts.speak.assert_called_once_with(expected_reply, block=True)


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


def test_listen_archives_legacy_style_but_keeps_saved_facts(tmp_path, monkeypatch):
    """Exercise the provider request after switching away from the legacy persona."""
    from raphael import __main__, audio, config, platform, providers
    from raphael.memory import ConversationManager, MemoryItem, MemoryStore, MemoryType
    from raphael.persona import PERSONA_CONTEXT_VERSION
    from raphael.providers.base import LLMResponse

    settings = config.Settings(
        _env_file=None, memory_db_path=str(tmp_path / "memory.db"), tts_streaming=False,
    )
    store = MemoryStore(settings.memory.db_path)
    legacy = ConversationManager(store=store, session_id="desktop_session")
    legacy_answer = "I follow protocol. Speak clearly. I am waiting."
    legacy.add_turn(role="user", content="What do you do?")
    legacy.add_turn(role="assistant", content=legacy_answer)
    fact = "You prefer patient explanations."
    store.save_memory(MemoryItem(content=fact, memory_type=MemoryType.FACT))
    store.save_memory(
        MemoryItem(
            content="You requested direct, cold replies. " + legacy_answer,
            memory_type=MemoryType.CONVERSATION,
            source="auto_summarizer",
            metadata={"session_id": "desktop_session", "last_turn_id": 2},
        )
    )
    store.close()

    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(platform, "get_audio_backend", MagicMock())
    router = MagicMock()
    router.send.return_value = LLMResponse(
        content="I'm here to help.",
        model="test",
        provider="test",
    )
    monkeypatch.setattr(providers, "get_model_router", lambda: router)
    for component in ("WakeWordDetector", "SpeechToText", "VoiceRecorder", "TextToSpeech"):
        monkeypatch.setattr(audio, component, MagicMock())

    def listener(**kwargs):
        loop = SimpleNamespace(is_running=True, stop=MagicMock())
        loop.start = lambda: kwargs["on_transcription"](
            "What do you remember about patient explanations?",
            {},
            None,
        )
        return loop

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(audio, "WakeListenerLoop", listener)
    monkeypatch.setattr(__main__, "time", SimpleNamespace(
        sleep=interrupt, monotonic=__main__.time.monotonic,
    ))
    monkeypatch.setattr(sys, "argv", ["raphael", "--listen"])

    assert __main__.main() == 0
    messages = router.send.call_args.args[0]
    assert [message.role for message in messages] == ["system", "user"]
    assert fact in messages[0].content
    assert legacy_answer not in messages[0].content
    assert messages[1].content == "What do you remember about patient explanations?"
    restored = MemoryStore(settings.memory.db_path)
    try:
        assert len(restored.get_recent_turns("desktop_session")) == 2
        current = f"desktop_session:{PERSONA_CONTEXT_VERSION}"
        assert len(restored.get_recent_turns(current)) == 2
        assert restored.get_session_summary("desktop_session") is not None
    finally:
        restored.close()


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
