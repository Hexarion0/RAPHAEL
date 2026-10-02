"""Bounded audio ingestion, wake detection, recording, and conversation workers."""

from __future__ import annotations

import queue
import threading
from collections import deque
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from raphael.audio.recorder import VoiceRecorder
from raphael.audio.wake import WakeWordDetector
from raphael.logging import get_logger

if TYPE_CHECKING:
    from raphael.audio.stt import SpeechToText
    from raphael.audio.tts import TextToSpeech
    from raphael.platform.base import AudioBackend

logger = get_logger("audio.listener")


class ListenerState(str, Enum):
    IDLE = "idle"
    LISTENING_WAKE = "listening_wake"
    WAKE_DETECTED = "wake_detected"
    RECORDING = "recording"
    PROCESSING = "processing"


class WakeListenerLoop:
    """Keep microphone callbacks independent of inference and user callbacks."""

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
        on_barge_in: Callable[[], None] | None = None,
        on_state_change: Callable[[ListenerState], None] | None = None,
        sample_rate: int = 16000,
        device: int | str | None = None,
        barge_in: bool = True,
        barge_in_threshold_rms: float = 0.030,
        stt_beam_size: int = 3,
    ) -> None:
        self.backend = audio_backend
        self.detector = detector or WakeWordDetector()
        self.recorder = recorder or VoiceRecorder(sample_rate=sample_rate)
        self.stt, self.tts = stt, tts
        self.on_wake, self.on_utterance = on_wake, on_utterance
        self.on_transcription, self.on_barge_in = on_transcription, on_barge_in
        self.on_state_change = on_state_change
        self.sample_rate, self.device = sample_rate, device
        self.barge_in, self.barge_in_threshold_rms = barge_in, barge_in_threshold_rms
        self.stt_beam_size = stt_beam_size
        self._state = ListenerState.IDLE
        self._last_wake_info: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._running = False
        self._stop_event = threading.Event()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=8)
        self._processing_queue: queue.Queue = queue.Queue(maxsize=4)
        self._notifications: queue.Queue = queue.Queue(maxsize=32)
        self._threads: list[threading.Thread] = []
        self._recent_audio: deque = deque(maxlen=max(1, round(sample_rate * 1.5 / 1280)))
        self._recording_generation = 0
        self.dropped_frames = 0

    @property
    def state(self) -> ListenerState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._running

    @staticmethod
    def _offer(target: queue.Queue, item: Any) -> bool:
        """Never block producers; replace the oldest item if the consumer falls behind."""
        try:
            target.put_nowait(item)
            return False
        except queue.Full:
            try:
                target.get_nowait()
                target.task_done()
            except queue.Empty:
                pass
            try:
                target.put_nowait(item)
            except queue.Full:
                pass
            return True

    def _notify(self, callback: Callable | None, *args: Any) -> None:
        if callback:
            self._offer(self._notifications, (callback, args))

    def _set_state(self, state: ListenerState) -> None:
        if self._state != state:
            self._state = state
            self._notify(self.on_state_change, state)

    def _notification_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                callback, args = self._notifications.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if not self._stop_event.is_set():
                    callback(*args)
            except Exception:
                logger.exception("Listener notification failed")
            finally:
                self._notifications.task_done()

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Copy borrowed audio and enqueue it; never run inference or take the state lock."""
        if self._running:
            if self._offer(self._frame_queue, indata.copy()):
                self.dropped_frames += 1

    def _start_recording(self) -> None:
        self._recording_generation += 1
        self.recorder.start()
        self._set_state(ListenerState.RECORDING)

    def _frame_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                audio = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                with self._lock:
                    if not self._running:
                        continue
                    self._handle_frame(audio)
            except Exception:
                logger.exception("Audio frame processing failed")
                with self._lock:
                    if self._running:
                        self.detector.reset(set_cooldown=False)
                        self._set_state(ListenerState.LISTENING_WAKE)
            finally:
                self._frame_queue.task_done()

    def _handle_frame(self, audio: np.ndarray) -> None:
        if (
            self.barge_in
            and self.tts
            and self.tts.is_speaking()
            and self._state != ListenerState.RECORDING
        ):
            if VoiceRecorder.calculate_rms(
                audio
            ) >= self.barge_in_threshold_rms or self.detector.process_frame(audio):
                self.tts.stop()
                self._start_recording()
                self.recorder.add_frame(audio)
                self._notify(self.on_barge_in)
                return
        if self._state == ListenerState.LISTENING_WAKE:
            self._recent_audio.append(audio)
            trigger = self.detector.process_frame(audio)
            if trigger:
                self._last_wake_info = trigger
                self._set_state(ListenerState.WAKE_DETECTED)
                self._start_recording()
                # Async wake inference can finish after the user starts the command.
                for recent in self._recent_audio:
                    self.recorder.add_frame(recent)
                self._recent_audio.clear()
                self._notify(self.on_wake, trigger)
        elif self._state == ListenerState.RECORDING:
            if not self.recorder._speech_started:
                trigger = self.detector.process_frame(audio)
                if trigger:
                    self._last_wake_info = trigger
                    self._start_recording()
                    self.recorder.add_frame(audio)
                    self._notify(self.on_wake, trigger)
                    return
            if not self.recorder.add_frame(audio):
                recorded = self.recorder.get_audio()
                info = self._last_wake_info.copy()
                self._offer(self._processing_queue, (recorded, info, self._recording_generation))
                self._set_state(ListenerState.PROCESSING)
                self._notify(self.on_utterance, recorded, info)

    def _process_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                audio, info, generation = self._processing_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if not self._running or generation != self._recording_generation:
                    continue
                text = (
                    self.stt.transcribe(audio, beam_size=self.stt_beam_size)
                    if (self.stt and audio.size)
                    else ""
                )
                if not self._running:
                    continue
                keep_listening = (
                    bool(self.on_transcription(text, info, audio))
                    if (self.on_transcription)
                    else False
                )
                with self._lock:
                    # A completed old response cannot cancel a new barge-in recording.
                    if not self._running or generation != self._recording_generation:
                        continue
                    if keep_listening:
                        self._start_recording()
                    else:
                        self._recent_audio.clear()
                        self.detector.reset(set_cooldown=True)
                        self._set_state(ListenerState.LISTENING_WAKE)
            except Exception:
                logger.exception("Utterance processing failed")
                with self._lock:
                    if self._running and generation == self._recording_generation:
                        self.detector.reset(set_cooldown=False)
                        self._set_state(ListenerState.LISTENING_WAKE)
            finally:
                self._processing_queue.task_done()

    def start(self) -> None:
        if self._running:
            return
        if any(thread.is_alive() for thread in self._threads):
            raise RuntimeError("Previous listener workers are still stopping")
        self._frame_queue = queue.Queue(maxsize=8)
        self._processing_queue = queue.Queue(maxsize=4)
        self._notifications = queue.Queue(maxsize=32)
        self._recent_audio.clear()
        self.dropped_frames = 0
        self._stop_event = threading.Event()
        start_detector = getattr(self.detector, "start", None)
        if start_detector:
            start_detector()
        self._running = True
        self._set_state(ListenerState.LISTENING_WAKE)
        self._threads = [
            threading.Thread(target=target, daemon=True)
            for target in (
                self._frame_worker,
                self._process_worker,
                self._notification_worker,
            )
        ]
        for thread in self._threads:
            thread.start()
        try:
            self.backend.start_stream(
                callback=self._audio_callback,
                sample_rate=self.sample_rate,
                channels=1,
                device=self.device,
                blocksize=1280,
            )
        except BaseException:
            self.stop()
            raise
        logger.info("Wake listener active; audio and inference run on separate workers.")

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        try:
            self.backend.stop_stream()
        finally:
            if self.tts:
                self.tts.stop()
            stop_detector = getattr(self.detector, "stop", None)
            if stop_detector:
                stop_detector()
            for thread in self._threads:
                if thread is not threading.current_thread():
                    thread.join(timeout=1.0)
            with self._lock:
                self._state = ListenerState.IDLE
            logger.info("Wake listener stopped (%d dropped audio frames).", self.dropped_frames)
