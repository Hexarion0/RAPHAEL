"""Speech-to-text conversion engine using faster-whisper."""

import time
from pathlib import Path
from typing import Any

import numpy as np
from faster_whisper import WhisperModel

from raphael.logging import get_logger

logger = get_logger("audio.stt")


class SpeechToText:
    """Fast, local speech-to-text transcriber powered by faster-whisper / CTranslate2."""

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "auto",
        compute_type: str = "default",
        language: str = "en",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language

        logger.info(
            "Loading faster-whisper STT model: '%s' (device=%s, compute_type=%s)",
            model_size,
            device,
            compute_type,
        )

        try:
            self.model = WhisperModel(
                model_size_or_path=model_size,
                device=device,
                compute_type=compute_type,
            )
            logger.info("faster-whisper model '%s' initialized successfully.", model_size)
        except Exception as err:
            logger.warning(
                "Whisper initialization failed (device=%s, type=%s): %s. Falling back to CPU/int8.",
                device,
                compute_type,
                err,
            )
            self.model = WhisperModel(
                model_size_or_path=model_size,
                device="cpu",
                compute_type="int8",
            )
            logger.info("faster-whisper loaded on CPU/int8 fallback.")

    def transcribe(
        self,
        audio: np.ndarray | str | Path,
        language: str | None = None,
        beam_size: int = 5,
    ) -> str:
        """Transcribe an audio numpy array or file path to plain text."""
        result = self.transcribe_detailed(
            audio=audio,
            language=language or self.language,
            beam_size=beam_size,
        )
        return result["text"]

    def transcribe_detailed(
        self,
        audio: np.ndarray | str | Path,
        language: str | None = None,
        beam_size: int = 5,
    ) -> dict[str, Any]:
        """Transcribe audio and return full segment details and timing statistics."""
        # Convert numpy array to 1D float32 normalized
        if isinstance(audio, np.ndarray):
            if audio.size == 0:
                return {
                    "text": "",
                    "language": language or self.language,
                    "segments": [],
                    "duration": 0.0,
                    "latency": 0.0,
                }
            audio_input = audio.squeeze().astype(np.float32)
        else:
            audio_input = str(Path(audio).resolve())

        target_lang = language or self.language
        start_t = time.time()

        segments_generator, info = self.model.transcribe(
            audio_input,
            language=target_lang,
            beam_size=beam_size,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        segments_list = []
        text_parts = []
        for segment in segments_generator:
            cleaned_text = segment.text.strip()
            if cleaned_text:
                text_parts.append(cleaned_text)
                segments_list.append(
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "text": cleaned_text,
                        "avg_logprob": segment.avg_logprob,
                        "no_speech_prob": segment.no_speech_prob,
                    }
                )

        full_text = " ".join(text_parts).strip()
        elapsed = time.time() - start_t

        logger.debug(
            "Transcribed %.1fs audio in %.2fs: '%s'",
            info.duration if hasattr(info, "duration") else 0.0,
            elapsed,
            full_text,
        )

        return {
            "text": full_text,
            "language": info.language if hasattr(info, "language") else target_lang,
            "segments": segments_list,
            "duration": getattr(info, "duration", 0.0),
            "latency": elapsed,
        }
