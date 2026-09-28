"""Continuous wake word detection engine supporting openWakeWord and 'Hey Raphael'."""

import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import openwakeword
from faster_whisper import WhisperModel
from openwakeword.model import Model

from raphael.logging import get_logger

logger = get_logger("audio.wake")


class WakeWordDetector:
    """Real-time streaming wake word detection supporting openWakeWord and keyword spotting."""

    # Phonetic and spelling variants for Raphael
    RAPHAEL_VARIANTS = [
        "raphael",
        "rafael",
        "hey raphael",
        "hey rafael",
        "rafeal",
        "raphel",
        "raffaele",
        "refael",
        "hey rapha",
        "rafa",
    ]

    def __init__(
        self,
        wake_phrase: str = "hey raphael",
        models: list[str] | None = None,
        threshold: float = 0.5,
        cooldown_seconds: float = 2.0,
        enable_whisper_spotter: bool = True,
    ) -> None:
        self.wake_phrase = wake_phrase.lower()
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.last_trigger_time = 0.0
        self.enable_whisper_spotter = enable_whisper_spotter

        # Build search keywords
        clean_phrase = re.sub(r"[^\w\s]", "", self.wake_phrase)
        combined_keywords = [
            clean_phrase,
            clean_phrase.replace("ph", "f"),
            "raphael",
            "rafael",
            *self.RAPHAEL_VARIANTS,
        ]
        self.keywords = list(set(combined_keywords))

        model_paths = self._resolve_model_paths(models)

        logger.info("Initializing openWakeWord detector with models: %s", models)
        try:
            self.model = Model(wakeword_model_paths=model_paths)
            self.available_models = list(self.model.models.keys())
            logger.info("Loaded openWakeWord models: %s", self.available_models)
        except Exception as err:
            logger.warning(
                "Failed to load openWakeWord model (%s). Relying on keyword spotter.",
                err,
            )
            self.model = None
            self.available_models = []

        # Sliding audio buffer for keyword spotting (1.2 seconds at 16kHz = 19,200 samples)
        self._sliding_buffer: deque = deque(maxlen=19200)
        self._speech_frames_count = 0
        self._last_spotter_check = 0.0
        self._ambient_rms = 0.01
        self._spotter_model: WhisperModel | None = None

        if self.enable_whisper_spotter:
            try:
                self._spotter_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
                logger.info("Initialized keyword wake spotter for target: '%s'", self.wake_phrase)
            except Exception as err:
                logger.warning("Failed to initialize keyword spotter: %s", err)
                self._spotter_model = None

    @staticmethod
    def _resolve_model_paths(models: list[str] | None) -> list[str]:
        """Resolve model names or filepaths to exact model paths."""
        pretrained_paths = openwakeword.get_pretrained_model_paths()
        if not models:
            return pretrained_paths

        resolved: list[str] = []
        for requested in models:
            req_path = Path(requested)
            if req_path.is_file() and requested.endswith(".onnx"):
                resolved.append(str(req_path.resolve()))
                continue

            matched = False
            for p in pretrained_paths:
                base_name = os.path.basename(p).lower()
                if requested.lower() in base_name:
                    resolved.append(p)
                    matched = True
                    break

            if not matched:
                logger.warning(
                    "Wake model '%s' not in pretrained models. Using defaults.",
                    requested,
                )

        return resolved or pretrained_paths

    def is_in_cooldown(self) -> bool:
        """Check whether the detector is currently within cooldown protection."""
        return (time.time() - self.last_trigger_time) < self.cooldown_seconds

    def process_frame(self, audio_chunk: np.ndarray) -> dict[str, Any] | None:
        """Feed a mono 16kHz audio frame and check for wake word triggers.

        Args:
            audio_chunk: Audio array of shape (N,) or (N, 1). Float32 (-1..1) or Int16.

        Returns:
            Dictionary with trigger details if wake word is detected, otherwise None.
        """
        if audio_chunk.size == 0:
            return None

        # Format chunk to 1D float32 and 1D int16
        if np.issubdtype(audio_chunk.dtype, np.floating):
            float_chunk = audio_chunk.squeeze().astype(np.float32)
            pcm16_chunk = (np.clip(float_chunk, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            pcm16_chunk = audio_chunk.squeeze().astype(np.int16)
            float_chunk = (pcm16_chunk.astype(np.float32) / 32768.0).astype(np.float32)

        now = time.time()

        # 1. Check openWakeWord models (e.g. hey_jarvis, alexa)
        if self.model is not None:
            predictions = self.model.predict(pcm16_chunk)
            for model_name, score in predictions.items():
                if score >= self.threshold:
                    if self.is_in_cooldown():
                        return None

                    self.last_trigger_time = now
                    logger.info(
                        "🎯 Wake word detected via openWakeWord! Model: '%s' (Confidence: %.2f)",
                        model_name,
                        score,
                    )
                    self._sliding_buffer.clear()
                    self._speech_frames_count = 0
                    return {
                        "model": model_name,
                        "score": float(score),
                        "timestamp": now,
                    }

        # 2. Check keyword spotter for "Raphael" / "Hey Raphael"
        if self._spotter_model is not None:
            self._sliding_buffer.extend(float_chunk)

            chunk_rms = float(np.sqrt(np.mean(float_chunk**2))) if float_chunk.size > 0 else 0.0

            # Update rolling ambient noise estimate
            if chunk_rms < 0.03:
                self._ambient_rms = 0.95 * self._ambient_rms + 0.05 * chunk_rms

            # Detect active speech chunk
            is_speech = chunk_rms > max(0.025, self._ambient_rms * 2.0)
            if is_speech:
                self._speech_frames_count += 1
            else:
                self._speech_frames_count = max(0, self._speech_frames_count - 1)

            # Trigger Whisper check only when speech is sustained
            if (
                self._speech_frames_count >= 5
                and len(self._sliding_buffer) >= 12800
                and (now - self._last_spotter_check >= 0.4)
            ):
                self._last_spotter_check = now
                buffer_array = np.array(self._sliding_buffer, dtype=np.float32)

                try:
                    segments, _ = self._spotter_model.transcribe(
                        buffer_array,
                        language="en",
                        beam_size=1,
                        without_timestamps=True,
                    )
                    transcription = " ".join(s.text.strip().lower() for s in segments)
                    cleaned = re.sub(r"[^\w\s]", "", transcription).strip()

                    for kw in self.keywords:
                        if kw in cleaned:
                            if self.is_in_cooldown():
                                return None

                            self.last_trigger_time = now
                            logger.info(
                                "🎯 Wake word '%s' detected! (Transcription: '%s')",
                                self.wake_phrase,
                                transcription,
                            )
                            self._sliding_buffer.clear()
                            self._speech_frames_count = 0
                            return {
                                "model": f"keyword_{self.wake_phrase.replace(' ', '_')}",
                                "score": 0.95,
                                "timestamp": now,
                            }
                except Exception as err:
                    logger.debug("Keyword spotter check error: %s", err)

        return None

    def reset(self) -> None:
        """Reset internal prediction buffers."""
        if self.model is not None:
            self.model.reset()
        self._sliding_buffer.clear()
        self._speech_frames_count = 0
        self.last_trigger_time = 0.0
