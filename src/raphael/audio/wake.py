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

    # Regex patterns to match wake phrases and natural phonetic variations
    WAKE_PATTERNS = [
        re.compile(
            r"\b(hey|hi|yo|a|okay|ok)?\s*(raphael|rafael|raphel|rafeal|raffael|refael|raph|ralph|raffaele)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(hey\s+)?(raf|raph)\b", re.IGNORECASE),
    ]

    def __init__(
        self,
        wake_phrase: str = "hey raphael",
        models: list[str] | None = None,
        threshold: float = 0.5,
        cooldown_seconds: float = 1.0,
        enable_whisper_spotter: bool = True,
    ) -> None:
        self.wake_phrase = wake_phrase.lower()
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.last_trigger_time = 0.0
        self.enable_whisper_spotter = enable_whisper_spotter

        model_paths = self._resolve_model_paths(models)

        if model_paths:
            logger.info("Initializing openWakeWord detector with models: %s", model_paths)
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
        else:
            logger.info("No openWakeWord models configured — using Whisper keyword spotter only.")
            self.model = None
            self.available_models = []

        # Sliding audio buffer for keyword spotting (1.5 seconds at 16kHz = 24,000 samples)
        self._sliding_buffer: deque = deque(maxlen=24000)
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
        """Resolve model names to full paths. Returns [] when no models are requested."""
        if not models:
            return []  # No models → Whisper-only mode, do NOT load all pretrained models

        pretrained_paths = openwakeword.get_pretrained_model_paths()
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
                    "Wake model '%s' not found in pretrained models — skipping.",
                    requested,
                )

        return resolved

    def is_in_cooldown(self) -> bool:
        """Check whether the detector is currently within cooldown protection."""
        return (time.time() - self.last_trigger_time) < self.cooldown_seconds

    def process_frame(self, audio_chunk: np.ndarray) -> dict[str, Any] | None:
        """Feed a mono 16kHz audio frame and check for wake word triggers."""
        if audio_chunk.size == 0 or self.is_in_cooldown():
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
                    self.last_trigger_time = now
                    logger.info(
                        "🎯 Wake word detected via openWakeWord! Model: '%s' (Confidence: %.2f)",
                        model_name,
                        score,
                    )
                    self.reset(set_cooldown=True)
                    return {
                        "model": model_name,
                        "score": float(score),
                        "timestamp": now,
                    }

        # 2. Check keyword spotter for "Raphael" / "Hey Raphael"
        if self._spotter_model is not None:
            self._sliding_buffer.extend(float_chunk)

            chunk_rms = float(np.sqrt(np.mean(float_chunk**2))) if float_chunk.size > 0 else 0.0

            # Dynamic noise estimate
            if chunk_rms < 0.03:
                self._ambient_rms = 0.95 * self._ambient_rms + 0.05 * chunk_rms

            # Detect voice energy burst with adaptive threshold
            is_speech = chunk_rms > max(0.012, self._ambient_rms * 1.5)
            if is_speech:
                self._speech_frames_count += 1
            else:
                self._speech_frames_count = max(0, self._speech_frames_count - 1)

            # Check Whisper when speech has accumulated (~0.2s to 0.5s)
            if (
                self._speech_frames_count >= 2
                and len(self._sliding_buffer) >= 8000
                and (now - self._last_spotter_check >= 0.25)
            ):
                self._last_spotter_check = now
                buffer_array = np.array(self._sliding_buffer, dtype=np.float32)

                try:
                    segments, _ = self._spotter_model.transcribe(
                        buffer_array,
                        language="en",
                        beam_size=1,
                        initial_prompt="Hey Raphael, Rafael",
                        without_timestamps=True,
                    )
                    transcription = " ".join(s.text.strip() for s in segments)

                    for pattern in self.WAKE_PATTERNS:
                        if pattern.search(transcription):
                            self.last_trigger_time = now
                            logger.info(
                                "🎯 Wake word '%s' detected! (Transcription: '%s')",
                                self.wake_phrase,
                                transcription,
                            )
                            self.reset(set_cooldown=True)
                            return {
                                "model": f"keyword_{self.wake_phrase.replace(' ', '_')}",
                                "score": 0.95,
                                "timestamp": now,
                            }
                except Exception as err:
                    logger.debug("Keyword spotter error: %s", err)

        return None

    def reset(self, set_cooldown: bool = False) -> None:
        """Reset internal prediction buffers and optionally engage cooldown."""
        if self.model is not None:
            self.model.reset()
            # Flush internal openwakeword buffers with zeros
            silence = np.zeros(1280, dtype=np.int16)
            for _ in range(8):
                self.model.predict(silence)

        self._sliding_buffer.clear()
        self._speech_frames_count = 0
        self._last_spotter_check = time.time()
        if set_cooldown:
            self.last_trigger_time = time.time()
