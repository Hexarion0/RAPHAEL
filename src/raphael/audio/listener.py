"""High-level wake-word listening loop, event pipeline, and barge-in interruption."""

import queue
import threading
from collections.abc import Callable
from enum import Enum
from typing import Any

import numpy as np

from raphael.audio.recorder import VoiceRecorder
from raphael.audio.stt import SpeechToText
from raphael.audio.tts import TextToSpeech
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
        tts: TextToSpeech | None = None,
        on_wake: Callable[[dict[str, Any]], None] | None = None,
        on_utterance: Callable[[np.ndarray, dict[str, Any]], None] | None = None,
        on_transcription: Callable[[str, dict[str, Any], np.ndarray], bool | None] | None = None,
        on_state_change: Callable[[ListenerState], None] | None = None,
        sample_rate: int = 16000,
        device: int | str | None = None,
        barge_in: bool = True,
        barge_in_threshold_rms: float = 0.030,
    ) -> None:
        self.backend = audio_backend
        self.detector = detector or WakeWordDetector()
        self.recorder = recorder or VoiceRecorder(sample_rate=sample_rate)
        self.stt = stt
        self.tts = tts
        self.on_wake = on_wake
        self.on_utterance = on_utterance
        self.on_transcription = on_transcription
        self.on_state_change = on_state_change
        self.sample_rate = sample_rate
        self.device = device
        self.barge_in = barge_in
        self.barge_in_threshold_rms = barge_in_threshold_rms

        self._state = ListenerState.IDLE
        self._last_wake_info: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._running = False
        self._processing_queue: queue.Queue[tuple[np.ndarray, dict[str, Any]] | None] = (
            queue.Queue()
        )
        self._worker_thread: threading.Thread | None = None

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

    def _process_worker(self) -> None:
        """Background worker thread to execute STT transcription and on_transcription."""
        while self._running:
            try:
                item = self._processing_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                break

            audio_data, wake_info = item
            try:
                text = ""
                if self.stt and len(audio_data) > 0:
                    try:
                        logger.info("Transcribing speech with Whisper...")
                        text = self.stt.transcribe(audio_data)
                        logger.info("🗣️ Transcribed: '%s'", text)
                    except Exception as err:
                        logger.error("STT transcription error: %s", err)

                keep_listening = False
                if self.on_transcription:
                    try:
                        result = self.on_transcription(text, wake_info, audio_data)
                        keep_listening = bool(result) if result is not None else False
                    except Exception as err:
                        logger.error("Error in on_transcription callback: %s", err)

                with self._lock:
                    if self._state == ListenerState.PROCESSING:
                        if keep_listening:
                            logger.info(
                                "👂 Follow-up mode active — speak next request or say 'Goodbye'."
                            )
                            self.recorder.start()
                            self._set_state(ListenerState.RECORDING)
                        else:
                            self.detector.reset(set_cooldown=True)
                            self._set_state(ListenerState.LISTENING_WAKE)
            finally:
                self._processing_queue.task_done()

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
            # 1. Real-time Barge-In Interruption Check
            if self.barge_in and self.tts and self.tts.is_speaking():
                rms = VoiceRecorder.calculate_rms(indata)
                # If speech energy threshold exceeded or wake word detected while speaking
                if rms >= self.barge_in_threshold_rms or self.detector.process_frame(indata):
                    logger.info("🛑 Barge-in! Stopping speech and listening to user...")
                    self.tts.stop()
                    self.recorder.start()
                    self._set_state(ListenerState.RECORDING)
                    return

            # 2. State machine handling
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

                    # Dispatch to worker thread so audio stream is never blocked
                    self._processing_queue.put((audio_data, wake_info))

    def start(self) -> None:
        """Start continuous wake word listening loop and worker thread."""
        if self._running:
            return

        self._running = True
        self._set_state(ListenerState.LISTENING_WAKE)

        # Start worker thread for background processing
        self._worker_thread = threading.Thread(target=self._process_worker, daemon=True)
        self._worker_thread.start()

        # Blocksize = 1280 frames (80ms at 16kHz), optimal for low-latency detection
        self.backend.start_stream(
            callback=self._audio_callback,
            sample_rate=self.sample_rate,
            channels=1,
            device=self.device,
            blocksize=1280,
        )
        logger.info("WakeListenerLoop is active. Awaiting wake word...")

    def stop(self) -> None:
        """Stop listening loop and terminate worker thread."""
        self._running = False
        self.backend.stop_stream()
        self._processing_queue.put(None)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        self._set_state(ListenerState.IDLE)
        logger.info("WakeListenerLoop stopped.")
