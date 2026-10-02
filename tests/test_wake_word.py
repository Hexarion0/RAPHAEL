"""Tests for wake word detection, voice recording, and listener loop."""

import time
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

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


def test_wake_detector_initialization(monkeypatch):
    """Verify openWakeWord detector initializes and exposes models."""
    monkeypatch.setattr(
        WakeWordDetector, "_resolve_model_paths", staticmethod(lambda _models: ["model.onnx"])
    )
    monkeypatch.setattr(
        WakeWordDetector,
        "_load_wake_model",
        staticmethod(lambda _paths: SimpleNamespace(models={"hey_jarvis": object()})),
    )
    detector = WakeWordDetector(
        models=["hey_jarvis"], threshold=0.6, cooldown_seconds=1.5, enable_whisper_spotter=False
    )
    assert detector.threshold == 0.6
    assert detector.cooldown_seconds == 1.5
    assert not detector.is_in_cooldown()
    assert len(detector.available_models) > 0


def test_wake_detector_silent_frame():
    """Verify detector handles silent frames cleanly without false positive triggers."""
    detector = WakeWordDetector(threshold=0.5, enable_whisper_spotter=False)
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
        detector=WakeWordDetector(enable_whisper_spotter=False),
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
        detector=WakeWordDetector(enable_whisper_spotter=False),
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
    wait_for(lambda: mock_tts.stopped)
    assert mock_tts.stopped is True
    assert loop.state == ListenerState.RECORDING

    loop.stop()


def wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("Timed out waiting for audio worker")
        time.sleep(0.005)


def test_keyword_inference_does_not_block_frames_and_reset_discards_results(monkeypatch):
    entered, release = Event(), Event()

    class SlowSpotter:
        def transcribe(self, _audio, **_kwargs):
            entered.set()
            assert release.wait(2)
            return [SimpleNamespace(text="Hey Raphael")], None

    monkeypatch.setattr(WakeWordDetector, "_load_spotter_model", lambda _self: SlowSpotter())
    detector = WakeWordDetector()
    try:
        wait_for(lambda: detector._spotter_model is not None)
        frame = np.ones(1280, dtype=np.float32) * 0.1
        for _ in range(7):
            detector.process_frame(frame)
        assert entered.wait(1)
        assert detector.process_frame(frame) is None
        detector.reset()
        release.set()
        wait_for(lambda: detector._spotter_requests.unfinished_tasks == 0)
        assert detector.process_frame(frame) is None
    finally:
        release.set()
        detector.stop()


def test_keyword_result_triggers_wake_on_next_frame(monkeypatch):
    class Spotter:
        def transcribe(self, _audio, **_kwargs):
            return [SimpleNamespace(text="Hey Raphael")], None

    monkeypatch.setattr(WakeWordDetector, "_load_spotter_model", lambda _self: Spotter())
    detector = WakeWordDetector()
    try:
        wait_for(lambda: detector._spotter_model is not None)
        frame = np.ones(1280, dtype=np.float32) * 0.1
        for _ in range(7):
            detector.process_frame(frame)
        wait_for(lambda: detector._spotter_result is not None)
        assert detector.process_frame(frame)["model"] == "whisper_keyword"
        assert detector.process_frame(frame) is None
    finally:
        detector.stop()


def test_callback_keeps_latest_audio_when_detection_is_slow():
    entered, release = Event(), Event()

    class SlowDetector:
        def process_frame(self, _frame):
            entered.set()
            assert release.wait(2)

        def reset(self, **_kwargs):
            pass

    backend = MockAudioBackend()
    loop = WakeListenerLoop(backend, detector=SlowDetector())
    loop.start()
    try:
        frame = np.zeros(1280, dtype=np.float32)
        backend.callback(frame, 1280, None, None)
        assert entered.wait(1)
        for _ in range(100):
            backend.callback(frame, 1280, None, None)
        assert loop._frame_queue.qsize() == 8
        assert loop.dropped_frames >= 92
        frame[:] = 1
        assert not np.any(loop._frame_queue.queue[-1])  # Callback owns a copy.
    finally:
        release.set()
        loop.stop()


def test_slow_wake_notification_does_not_delay_recording():
    entered, release = Event(), Event()

    class Trigger:
        def process_frame(self, _frame):
            return {"model": "mock"}

        def reset(self, **_kwargs):
            pass

    def on_wake(_info):
        entered.set()
        assert release.wait(2)

    backend = MockAudioBackend()
    loop = WakeListenerLoop(backend, detector=Trigger(), on_wake=on_wake)
    loop.start()
    try:
        frame = np.ones(1280, dtype=np.float32) * 0.1
        backend.callback(frame, 1280, None, None)
        assert entered.wait(1)
        backend.callback(frame, 1280, None, None)
        wait_for(lambda: len(loop.recorder._buffer) == 2)
        assert loop.state == ListenerState.RECORDING
    finally:
        release.set()
        loop.stop()


def test_listener_restart_and_failed_stream_start_are_clean():
    backend = MockAudioBackend()
    loop = WakeListenerLoop(backend, detector=WakeWordDetector(enable_whisper_spotter=False))
    for _ in range(2):
        loop.start()
        assert loop.is_running
        loop.stop()
        assert not loop.is_running
        assert all(not thread.is_alive() for thread in loop._threads)

    def fail(**_kwargs):
        raise RuntimeError("No microphone")

    backend.start_stream = fail
    with pytest.raises(RuntimeError, match="No microphone"):
        loop.start()
    assert loop.state == ListenerState.IDLE
    assert all(not thread.is_alive() for thread in loop._threads)


def test_old_response_cannot_cancel_barge_in():
    started, finish = Event(), Event()
    backend = MockAudioBackend()
    detector = WakeWordDetector(enable_whisper_spotter=False)

    def response(*_args):
        started.set()
        assert finish.wait(2)
        return False

    loop = WakeListenerLoop(backend, detector=detector, on_transcription=response)
    loop.start()
    try:
        with loop._lock:
            loop._start_recording()
            loop._state = ListenerState.PROCESSING
            loop._processing_queue.put((np.ones(1280), {}, loop._recording_generation))
        assert started.wait(1)
        with loop._lock:
            loop._start_recording()  # New recording started while old callback is running.
        finish.set()
        wait_for(lambda: loop._processing_queue.unfinished_tasks == 0)
        assert loop.state == ListenerState.RECORDING
    finally:
        finish.set()
        loop.stop()


def test_recorder_uses_audio_duration_instead_of_processing_speed():
    recorder = VoiceRecorder(silence_duration_seconds=0.2, min_speech_duration_seconds=0.1)
    recorder.start()
    for _ in range(4):
        assert recorder.add_frame(np.ones(1280, dtype=np.float32) * 0.1)
    assert recorder.add_frame(np.zeros(1280, dtype=np.float32))
    assert recorder.add_frame(np.zeros(1280, dtype=np.float32))
    assert not recorder.add_frame(np.zeros(1280, dtype=np.float32))
