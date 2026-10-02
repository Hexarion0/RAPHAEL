"""Audio processing, wake word detection, recording, and speech-to-text subsystem."""

from importlib import import_module

_EXPORTS = {
    "ListenerState": "listener",
    "WakeListenerLoop": "listener",
    "WakeWordDetector": "wake",
    "VoiceRecorder": "recorder",
    "SpeechToText": "stt",
    "TextToSpeech": "tts",
    "resolve_piper_voice_paths": "tts",
    "PiperTrainError": "piper_train",
    "PiperTrainPlan": "piper_train",
    "build_piper_train_plan": "piper_train",
    "install_trained_voice": "piper_train",
    "VoiceDatasetReport": "voice_dataset",
    "RAW_VOICE_DIR": "voice_dataset",
    "PREPARED_DATASET_DIR": "voice_dataset",
    "prepare_voice_dataset": "voice_dataset",
    "record_voice_samples": "trainer",
    "train_custom_wakeword": "trainer",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Load audio components only when they are requested."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
