"""High-level wake-word listening loop and event pipeline."""

import threading
from collections.abc import Callable
from enum import Enum
from typing import Any

import numpy as np

from raphael.audio.recorder import VoiceRecorder
from raphael.audio.stt import SpeechToText
from raphael.audio.wake import WakeWordDetector
from raphael.logging import get_logger
from raphael.platform.base import AudioBackend

logger = get_logger("audio.listener")


class ListenerState(str, Enum):
    """Lifecycle states of the voice assistant listener."""

    IDLE = "idle"
    LISTENING_WAKE = "listening_wake"
    WAKE_DETECTED = "wake_detected"
    RECORDING = "recording"
    PROCESSING = "processing"


class WakeListenerLoop:
    """Coordinates audio backend streaming, wake word detection, utterance recording, and STT."""

    def __init__(
        self,
        audio_backend: AudioBackend,
        detector: WakeWordDetector | None = None,
        recorder: VoiceRecorder | None = None,
        stt: SpeechToText | None = None,
        on_wake: Callable[[dict[str, Any]], None] | None = None,
        on_utterance: Callable[[np.ndarray, dict[str, Any]], None] | None = None,
        on_transcription: Callable[[str, dict[str, Any], np.ndarray], None] | None = None,
        on_state_change: Callable[[ListenerState], None] | None = None,
        sample_rate: int = 16000,
        device: int | str | None = None,
    ) -> None:
        self.backend = audio_backend
        self.detector = detector or WakeWordDetector()
        self.recorder = recorder or VoiceRecorder(sample_rate=sample_rate)
        self.stt = stt
        self.on_wake = on_wake
        self.on_utterance = on_utterance
        self.on_transcription = on_transcription
        self.on_state_change = on_state_change
        self.sample_rate = sample_rate
        self.device = device

        self._state = ListenerState.IDLE
        self._last_wake_info: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._running = False

    @property
    def state(self) -> ListenerState:
        """Current listener state."""
        return self._state

    def _set_state(self, new_state: ListenerState) -> None:
        if self._state != new_state:
            self._state = new_state
            logger.debug("Listener state changed to: %s", new_state.value)
            if self.on_state_change:
                try:
                    self.on_state_change(new_state)
                except Exception as err:
                    logger.error("Error in state change callback: %s", err)

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """Audio stream callback triggered for every audio buffer chunk."""
        if not self._running:
            return

        with self._lock:
            if self._state == ListenerState.LISTENING_WAKE:
                trigger = self.detector.process_frame(indata)
                if trigger:
                    self._last_wake_info = trigger
                    self._set_state(ListenerState.WAKE_DETECTED)
                    if self.on_wake:
                        try:
                            self.on_wake(trigger)
                        except Exception as err:
                            logger.error("Error in on_wake callback: %s", err)

                    # Transition immediately to recording utterance
                    self.recorder.start()
                    self._set_state(ListenerState.RECORDING)

            elif self._state == ListenerState.RECORDING:
                continue_recording = self.recorder.add_frame(indata)
                if not continue_recording:
                    self._set_state(ListenerState.PROCESSING)
                    audio_data = self.recorder.get_audio()
                    wake_info = self._last_wake_info.copy()

                    if self.on_utterance:
                        try:
                            self.on_utterance(audio_data, wake_info)
                        except Exception as err:
                            logger.error("Error in on_utterance callback: %s", err)

                    # Perform STT transcription if STT engine is attached
                    if self.stt and len(audio_data) > 0:
                        try:
                            logger.info("Transcribing speech with Whisper...")
                            text = self.stt.transcribe(audio_data)
                            logger.info("🗣️ Transcribed: '%s'", text)
                            if self.on_transcription:
                                self.on_transcription(text, wake_info, audio_data)
                        except Exception as err:
                            logger.error("STT transcription error: %s", err)

                    # Reset detector and resume listening for wake word
                    self.detector.reset()
                    self._set_state(ListenerState.LISTENING_WAKE)

    def start(self) -> None:
        """Start continuous wake word listening loop."""
        if self._running:
            return

        self._running = True
        self._set_state(ListenerState.LISTENING_WAKE)
        # Blocksize = 1280 frames (80ms at 16kHz), optimal for openWakeWord
        self.backend.start_stream(
            callback=self._audio_callback,
            sample_rate=self.sample_rate,
            channels=1,
            device=self.device,
            blocksize=1280,
        )
        logger.info("WakeListenerLoop is active. Awaiting wake word...")

    def stop(self) -> None:
        """Stop listening loop."""
        self._running = False
        self.backend.stop_stream()
        self._set_state(ListenerState.IDLE)
        logger.info("WakeListenerLoop stopped.")
