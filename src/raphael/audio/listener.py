"""Bounded audio ingestion, wake detection, recording, and conversation workers."""

from __future__ import annotations

import queue
import threading
import time
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
        ambient: bool = False,
        speech_detector: Callable[[np.ndarray], bool] | None = None,
        monitor_resumed_speech: bool = False,
        on_transcript_observed: Callable[[str, dict[str, Any]], None] | None = None,
        show_transcripts: bool = False,
        barge_in_mode: str = "wake",
        barge_in_speech_seconds: float = 0.24,
        on_interruption: Callable[[dict[str, Any]], None] | None = None,
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
        self.ambient = ambient
        self._speech_detector = speech_detector
        self.monitor_resumed_speech = monitor_resumed_speech
        self.on_transcript_observed = on_transcript_observed
        self.show_transcripts = show_transcripts
        if barge_in_mode not in {"speech", "wake"}:
            raise ValueError("Barge-in mode must be 'speech' or 'wake'")
        self.barge_in_mode = barge_in_mode
        self.barge_in_speech_seconds = barge_in_speech_seconds
        self.on_interruption = on_interruption
        self._interrupt_speech_samples = 0
        self._speech_streak = 0
        self._response_cancel: threading.Event | None = None
        self._state = ListenerState.IDLE
        self._last_wake_info: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._running = False
        self._stop_event = threading.Event()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=8)
        self._processing_queue: queue.Queue = queue.Queue(maxsize=4)
        self._notifications: queue.Queue = queue.Queue(maxsize=32)
        self._threads: list[threading.Thread] = []
        self._recent_audio: deque = deque(maxlen=max(1, round(sample_rate * 3.0 / 1280)))
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
        if self._response_cancel is not None:
            self._response_cancel.set()
            # Also invalidate TTS still synthesizing: is_speaking() is false then.
            self._stop_output()
        self._response_cancel = threading.Event()
        self._recording_generation += 1
        self.recorder.start()
        self._set_state(ListenerState.RECORDING)

    def _stop_output(self) -> None:
        if self.tts:
            info = self.tts.stop()
            if isinstance(info, dict):
                self._notify(self.on_interruption, info)

    def set_ambient(self, enabled: bool) -> None:
        """Switch capture mode; background speech is never treated as a wake trigger."""
        if enabled and self._speech_detector is None:
            if self.sample_rate != 16000:
                raise ValueError("Ambient speech detection requires 16000 Hz audio")
            from raphael.audio.activity import SpeechActivity

            self._speech_detector = SpeechActivity()
        with self._lock:
            self.ambient = enabled
            if self._speech_detector is not None:
                self.recorder.speech_detector = self._speech_detector
            self._speech_streak = 0
            self._recent_audio.clear()

    def _start_speech_recording(self) -> None:
        """Retain a short onset pre-roll when speech starts without a wake phrase."""
        # Neural VAD may recognize onset after the first word has already begun.
        # Keep 800 ms rather than just the two positive frames and their predecessor.
        preroll_frames = max(3, round(self.sample_rate * 0.8 / 1280))
        recent = list(self._recent_audio)[-preroll_frames:]
        resumed = self._state == ListenerState.PROCESSING
        if self.ambient and resumed:
            # A late keyword result from the previous utterance cannot authorize
            # this new one. Re-seed the detector with this recording's onset only.
            self.detector.reset(set_cooldown=False)
        speaking = bool(self.tts and self.tts.is_speaking())
        during_reply = speaking or bool(
            resumed and self._response_cancel is not None and not self._response_cancel.is_set()
        )
        if speaking:
            self._stop_output()
        self._last_wake_info = {
            "ambient": self.ambient,
            "resumed_speech": True,
            "speech_started_at": time.monotonic(),
            "during_reply": during_reply,
            "response_pending": resumed,
        }
        self._start_recording()
        for frame in recent:
            self.recorder.add_frame(frame)
            if self.ambient and resumed:
                self._check_ambient_wake(frame)
        self._recent_audio.clear()
        self._speech_streak = 0

    def _check_ambient_wake(self, audio: np.ndarray) -> dict[str, Any] | None:
        """Bind independent wake evidence to the current utterance without restarting it."""
        if self._last_wake_info.get("wake_verified"):
            return None
        trigger = self.detector.process_frame(audio)
        if trigger:
            self._confirm_ambient_wake(trigger)
        return trigger

    def _confirm_ambient_wake(self, trigger: dict[str, Any]) -> None:
        self._last_wake_info.update(
            wake_verified=True,
            wake_phrase=trigger.get(
                "wake_phrase", getattr(self.detector, "wake_phrase", "hey raphael")
            ),
        )
        self._notify(self.on_wake, trigger)

    def _seed_wake_audio(self, trigger: dict[str, Any]) -> None:
        """Keep post-greeting audio using the keyword spotter's original word timing."""
        recent = np.concatenate(list(self._recent_audio)) if self._recent_audio else np.empty(0)
        tail = trigger.get("wake_tail_samples")
        if tail is not None:
            tail = min(recent.shape[0], max(0, int(tail)))
            recent = recent[-tail:] if tail else recent[:0]
            self._last_wake_info["wake_prefix_trimmed"] = True
        for start in range(0, recent.shape[0], 1280):
            if not self.recorder.add_frame(recent[start : start + 1280]):
                break
        self._recent_audio.clear()

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
        self._recent_audio.append(audio)
        if (
            self.barge_in
            and self.tts
            and self.tts.is_speaking()
            and self._state != ListenerState.RECORDING
        ):
            trigger = self.detector.process_frame(audio)
            speech_interrupt = False
            if self.barge_in_mode == "speech" and self._speech_detector is not None:
                speech = self._speech_detector(audio)
                self._interrupt_speech_samples = (
                    self._interrupt_speech_samples + audio.size if speech else 0
                )
                speech_interrupt = (
                    self._interrupt_speech_samples >=
                    round(self.sample_rate * self.barge_in_speech_seconds)
                )
            loud_interrupt = (
                not self.ambient
                and self.barge_in_mode == "wake"
                and VoiceRecorder.calculate_rms(audio) >= self.barge_in_threshold_rms
            )
            if speech_interrupt and not trigger:
                self._start_speech_recording()
                self._notify(self.on_barge_in)
                self._interrupt_speech_samples = 0
                return
            if loud_interrupt or trigger:
                self._stop_output()
                self._last_wake_info = dict(trigger) if trigger else {}
                self._last_wake_info["ambient"] = self.ambient
                self._last_wake_info["during_reply"] = True
                self._last_wake_info["speech_started_at"] = time.monotonic()
                if trigger and self.ambient:
                    self._last_wake_info["wake_verified"] = True
                self._start_recording()
                if trigger:
                    self._seed_wake_audio(trigger)
                else:
                    self.recorder.add_frame(audio)
                self._notify(self.on_barge_in)
                return
            # With speakers select wake mode so loudspeaker speech isn't treated
            # as a user interruption. Speech mode is intended for headphones.
            return
        self._interrupt_speech_samples = 0
        if self.ambient and self._state == ListenerState.LISTENING_WAKE:
            trigger = self.detector.process_frame(audio)
            if trigger:
                self._last_wake_info = dict(trigger, ambient=True, wake_verified=True)
                self._start_recording()
                self._seed_wake_audio(trigger)
                self._notify(self.on_wake, trigger)
                return
        capture_speech = (
            self.ambient and self._state == ListenerState.LISTENING_WAKE
        ) or (self.monitor_resumed_speech and self._state == ListenerState.PROCESSING)
        if capture_speech and self._speech_detector is not None:
            speech = self._speech_detector(audio)
            self._speech_streak = self._speech_streak + 1 if speech else 0
            if self._speech_streak >= 2:
                self._start_speech_recording()
                return
        if self._state == ListenerState.LISTENING_WAKE:
            if self.ambient:
                return
            trigger = self.detector.process_frame(audio)
            if trigger:
                self._last_wake_info = trigger
                self._set_state(ListenerState.WAKE_DETECTED)
                self._start_recording()
                # Async wake inference can finish after the user starts the command.
                self._seed_wake_audio(trigger)
                self._notify(self.on_wake, trigger)
        elif self._state == ListenerState.RECORDING:
            if self.ambient:
                self._check_ambient_wake(audio)
            if not self.recorder._speech_started and not self.ambient:
                trigger = self.detector.process_frame(audio)
                if trigger:
                    self._last_wake_info = trigger
                    self._start_recording()
                    self._seed_wake_audio(trigger)
                    self._notify(self.on_wake, trigger)
                    return
            if not self.recorder.add_frame(audio):
                recorded = self.recorder.get_audio()
                info = self._last_wake_info.copy()
                info["cancel_event"] = self._response_cancel
                if info.get("wake_prefix_trimmed"):
                    info["post_wake_speech"] = self.recorder._speech_started
                # The greeting belongs only to this utterance, never to a follow-up.
                self._last_wake_info.pop("wake_prefix_trimmed", None)
                if self.ambient:
                    # The keyword worker can finish during STT. Keep its evidence
                    # attached to this queued utterance, never a subsequent one.
                    self._last_wake_info = info
                self._offer(self._processing_queue, (recorded, info, self._recording_generation))
                self._set_state(ListenerState.PROCESSING)
                self._speech_streak = 0
                reset_activity = getattr(self._speech_detector, "reset", None)
                if reset_activity:
                    reset_activity()
                self._notify(self.on_utterance, recorded, info)
        elif self.ambient and self._state == ListenerState.PROCESSING:
            self._check_ambient_wake(audio)

    def _process_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                audio, info, generation = self._processing_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if not self._running:
                    continue
                if generation != self._recording_generation:
                    logger.info("Decoding superseded speech for temporary conversation context.")
                text = ""
                has_command_audio = not (
                    info.get("wake_prefix_trimmed") and not info.get("post_wake_speech", True)
                )
                if self.stt and audio.size and has_command_audio:
                    logger.info(
                        "Transcribing %.2fs of recorded speech...", audio.size / self.sample_rate
                    )
                    detailed = getattr(self.stt, "transcribe_detailed", None)
                    if callable(detailed):
                        result = detailed(audio, beam_size=self.stt_beam_size)
                        text = result["text"]
                        info["stt_needs_repeat"] = result.get("needs_repeat", False)
                        info["stt_confidence"] = result.get("confidence", 0.0)
                        info["stt_raw_text"] = result.get("raw_text", text)
                    else:
                        text = self.stt.transcribe(audio, beam_size=self.stt_beam_size)
                    logger.info(
                        "STT finished: %d words, confidence=%.2f, repeat=%s.",
                        len(text.split()), info.get("stt_confidence", 0.0),
                        info.get("stt_needs_repeat", False),
                    )
                    logger.debug("STT candidate: %r", info.get("stt_raw_text", text))
                    if self.show_transcripts:
                        logger.info(
                            "STT candidate (diagnostic): %r", info.get("stt_raw_text", text)
                        )
                if info.get("wake_prefix_trimmed") and not info.get("stt_needs_repeat"):
                    greeting = info.get("wake_phrase", "hey raphael")
                    text = f"{greeting}, {text}" if text else greeting
                if not self._running:
                    continue
                with self._lock:
                    if self.ambient and generation == self._recording_generation:
                        poll = getattr(self.detector, "poll", None)
                        trigger = poll() if callable(poll) else None
                        if trigger:
                            self._confirm_ambient_wake(trigger)
                            info["wake_verified"] = True
                        # Discard late results from this recording before playback;
                        # its own greeting must not become a fresh barge-in trigger.
                        self.detector.reset(set_cooldown=False)
                superseded = generation != self._recording_generation
                info["superseded"] = superseded
                if self.on_transcript_observed:
                    self.on_transcript_observed(text, info)
                if superseded:
                    logger.info("Speech resumed during STT; suppressed the old reply.")
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
                    if keep_listening and not self.ambient:
                        self._start_recording()
                    else:
                        self._recent_audio.clear()
                        self.detector.reset(set_cooldown=not self.ambient)
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
        if (
            self.ambient or self.monitor_resumed_speech or self.barge_in_mode == "speech"
        ) and self._speech_detector is None:
            if self.sample_rate != 16000:
                raise ValueError("Ambient speech detection requires 16000 Hz audio")
            from raphael.audio.activity import SpeechActivity

            self._speech_detector = SpeechActivity()
        reset_activity = getattr(self._speech_detector, "reset", None)
        if reset_activity:
            reset_activity()
        if self._speech_detector is not None:
            self.recorder.speech_detector = self._speech_detector
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
        if self._response_cancel is not None:
            self._response_cancel.set()
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
