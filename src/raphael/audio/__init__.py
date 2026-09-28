"""Audio processing, wake word detection, recording, and speech-to-text subsystem."""

from raphael.audio.listener import ListenerState, WakeListenerLoop
from raphael.audio.recorder import VoiceRecorder
from raphael.audio.stt import SpeechToText
from raphael.audio.trainer import record_voice_samples, train_custom_wakeword
from raphael.audio.tts import TextToSpeech
from raphael.audio.wake import WakeWordDetector

__all__ = [
    "ListenerState",
    "WakeListenerLoop",
    "WakeWordDetector",
    "VoiceRecorder",
    "SpeechToText",
    "TextToSpeech",
    "record_voice_samples",
    "train_custom_wakeword",
]
