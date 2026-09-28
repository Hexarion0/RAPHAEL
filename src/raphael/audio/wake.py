"""Continuous wake word detection engine using openWakeWord."""

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import openwakeword
from openwakeword.model import Model

from raphael.logging import get_logger

logger = get_logger("audio.wake")


class WakeWordDetector:
    """Real-time streaming wake word detection with thresholding and cooldown protection."""

    def __init__(
        self,
        models: list[str] | None = None,
        threshold: float = 0.5,
        cooldown_seconds: float = 2.0,
    ) -> None:
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.last_trigger_time = 0.0

        model_paths = self._resolve_model_paths(models)

        logger.info("Initializing openWakeWord detector with models: %s", models)
        try:
            self.model = Model(wakeword_model_paths=model_paths)
            self.available_models = list(self.model.models.keys())
            logger.info("Loaded wake word models: %s", self.available_models)
        except Exception as err:
            logger.error("Failed to load wake word model: %s", err)
            raise

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

            # Match against pretrained model base names
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
        """Feed a mono 16kHz audio frame into the model and check for wake word triggers.

        Args:
            audio_chunk: Audio array of shape (N,) or (N, 1). Float32 (-1..1) or Int16.

        Returns:
            Dictionary with trigger details if wake word is detected, otherwise None.
        """
        if audio_chunk.size == 0:
            return None

        # Convert to 1D int16 array for openWakeWord
        if np.issubdtype(audio_chunk.dtype, np.floating):
            # Scale float32 (-1.0 to 1.0) to int16
            pcm16_chunk = (np.clip(audio_chunk.squeeze(), -1.0, 1.0) * 32767).astype(np.int16)
        else:
            pcm16_chunk = audio_chunk.squeeze().astype(np.int16)

        # Feed frame into openWakeWord model
        predictions = self.model.predict(pcm16_chunk)
        now = time.time()

        for model_name, score in predictions.items():
            if score >= self.threshold:
                if self.is_in_cooldown():
                    logger.debug(
                        "Wake word '%s' detected (score=%.3f) but suppressed due to cooldown.",
                        model_name,
                        score,
                    )
                    return None

                self.last_trigger_time = now
                logger.info(
                    "🎯 Wake word detected! Model: '%s' (Confidence: %.2f)",
                    model_name,
                    score,
                )
                return {
                    "model": model_name,
                    "score": float(score),
                    "timestamp": now,
                }

        return None

    def reset(self) -> None:
        """Reset internal prediction buffers."""
        self.model.reset()
        self.last_trigger_time = 0.0
