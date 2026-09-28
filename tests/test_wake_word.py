"""Tests for wake word detection, voice recording, and listener loop."""

import numpy as np

from raphael.audio.listener import ListenerState, WakeListenerLoop
from raphael.audio.recorder import VoiceRecorder
from raphael.audio.wake import WakeWordDetector
from raphael.platform.base import AudioBackend, AudioDeviceInfo


class MockAudioBackend(AudioBackend):
    """Mock audio backend for simulating hardware streams in tests."""

    def __init__(self) -> None:
        self._streaming = False
        self.callback = None

    def list_devices(self) -> list[AudioDeviceInfo]:
        return [
            AudioDeviceInfo(
                index=0,
                name="Mock Mic",
                hostapi="Mock",
                max_input_channels=1,
                max_output_channels=0,
                default_samplerate=16000.0,
                is_default_input=True,
            )
        ]

    def get_default_input_device(self) -> AudioDeviceInfo | None:
        return self.list_devices()[0]

    def get_default_output_device(self) -> AudioDeviceInfo | None:
        return None

    def resolve_device(self, device: int | str | None, is_input: bool = True) -> int | None:
        return 0

    def record(
        self,
        duration: float,
        sample_rate: int = 16000,
        channels: int = 1,
        device: int | str | None = None,
    ) -> np.ndarray:
        return np.zeros((int(duration * sample_rate), channels), dtype=np.float32)

    def save_wav(self, audio_data: np.ndarray, file_path, sample_rate: int = 16000):
        return file_path

    def start_stream(
        self,
        callback,
        sample_rate: int = 16000,
        channels: int = 1,
        device: int | str | None = None,
        blocksize: int = 1024,
    ) -> None:
        self._streaming = True
        self.callback = callback

    def stop_stream(self) -> None:
        self._streaming = False

    def is_streaming(self) -> bool:
        return self._streaming


def test_wake_detector_initialization():
    """Verify openWakeWord detector initializes and exposes models."""
    detector = WakeWordDetector(models=["hey_jarvis", "alexa"], threshold=0.6, cooldown_seconds=1.5)
    assert detector.threshold == 0.6
    assert detector.cooldown_seconds == 1.5
    assert not detector.is_in_cooldown()
    assert len(detector.available_models) > 0


def test_wake_detector_silent_frame():
    """Verify detector handles silent frames cleanly without false positive triggers."""
    detector = WakeWordDetector(threshold=0.5)
    silent_frame = np.zeros(1280, dtype=np.float32)
    result = detector.process_frame(silent_frame)
    assert result is None


def test_voice_recorder():
    """Verify VoiceRecorder accumulates frames, detects speech, and detects trailing silence."""
    recorder = VoiceRecorder(
        sample_rate=16000,
        silence_threshold_rms=0.01,
        silence_duration_seconds=0.1,
        min_speech_duration_seconds=0.05,
        max_duration_seconds=2.0,
    )
    recorder.start()
    assert recorder.is_recording()

    # Feed speech chunk (sine tone)
    t = np.linspace(0, 0.1, 1600, endpoint=False)
    speech_frame = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # 2 speech frames (~0.2s)
    cont = recorder.add_frame(speech_frame)
    assert cont is True
    cont = recorder.add_frame(speech_frame)
    assert cont is True

    # Feed silent frames to trigger silence conclusion
    silent_frame = np.zeros(1600, dtype=np.float32)
    recorder.add_frame(silent_frame)
    recorder.add_frame(silent_frame)

    audio = recorder.get_audio()
    assert len(audio) > 0


def test_wake_listener_loop_state_transitions():
    """Verify WakeListenerLoop lifecycle and mock streaming integration."""
    backend = MockAudioBackend()
    states_observed: list[ListenerState] = []

    def on_state(s: ListenerState):
        states_observed.append(s)

    loop = WakeListenerLoop(
        audio_backend=backend,
        on_state_change=on_state,
    )
    assert loop.state == ListenerState.IDLE

    loop.start()
    assert loop.state == ListenerState.LISTENING_WAKE
    assert backend.is_streaming()

    loop.stop()
    assert loop.state == ListenerState.IDLE
    assert not backend.is_streaming()


def test_wake_listener_barge_in_interruption():
    """Verify that speech during TTS playback triggers immediate barge-in stop."""
    backend = MockAudioBackend()

    class MockTTS:
        def __init__(self):
            self._speaking = True
            self.stopped = False

        def is_speaking(self) -> bool:
            return self._speaking

        def stop(self) -> None:
            self._speaking = False
            self.stopped = True

    mock_tts = MockTTS()
    loop = WakeListenerLoop(
        audio_backend=backend,
        tts=mock_tts,
        barge_in=True,
        barge_in_threshold_rms=0.01,
    )
    loop.start()

    # Create high-energy speech frame to trigger barge-in
    t = np.linspace(0, 0.08, 1280, endpoint=False)
    speech_frame = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # Trigger audio callback directly
    backend.callback(speech_frame, 1280, None, None)

    # Verify TTS was stopped and loop switched to recording
    assert mock_tts.stopped is True
    assert loop.state == ListenerState.RECORDING

    loop.stop()
