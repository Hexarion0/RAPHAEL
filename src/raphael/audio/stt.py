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
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
        initial_prompt: str = "Hey Raphael, questions, commands, and conversational dialogue.",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.initial_prompt = initial_prompt

        logger.info(
            "Loading faster-whisper STT model: '%s' (device=%s, compute_type=%s)",
            model_size,
            device,
            compute_type,
        )
        self.model = self._load_model(model_size, device, compute_type)

    def _load_model(self, model_size: str, device: str, compute_type: str) -> WhisperModel:
        try:
            return WhisperModel(
                model_size_or_path=model_size,
                device=device,
                compute_type=compute_type,
            )
        except Exception as err:
            logger.warning(
                "Whisper initialization failed (device=%s, type=%s): %s. Falling back to CPU/int8.",
                device,
                compute_type,
                err,
            )
            self.device = "cpu"
            self.compute_type = "int8"
            return WhisperModel(
                model_size_or_path=model_size,
                device="cpu",
                compute_type="int8",
            )

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

        try:
            segments_generator, info = self.model.transcribe(
                audio_input,
                language=target_lang,
                beam_size=beam_size,
                initial_prompt=self.initial_prompt,
                vad_filter=False,
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
        except Exception as err:
            logger.warning(
                "Runtime error during Whisper transcription (%s). Retrying on CPU fallback.",
                err,
            )
            self.model = WhisperModel(
                model_size_or_path=self.model_size,
                device="cpu",
                compute_type="int8",
            )
            self.device = "cpu"
            self.compute_type = "int8"
            segments_generator, info = self.model.transcribe(
                audio_input,
                language=target_lang,
                beam_size=beam_size,
                initial_prompt=self.initial_prompt,
                vad_filter=False,
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
