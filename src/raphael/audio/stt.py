"""Speech-to-text conversion engine using faster-whisper with VAD and anti-hallucination."""

import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from faster_whisper import WhisperModel

from raphael.logging import get_logger

logger = get_logger("audio.stt")

# Common Whisper hallucination patterns on silence or ambient noise
_HALLUCINATION_PATTERNS = [
    re.compile(r"^\s*(thank\s+you(\s+very\s+much|\s+for\s+watching)?|thanks\s+for\s+watching)[.?!]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(subtitles?\s+by|subscribe|like\s+and\s+subscribe)[.?!]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(\[[^\]]+\]|\([^\)]+\))\s*$"),  # [Music], (bell rings), etc.
    re.compile(r"^\s*(\.|\?|!|,|-|_)+\s*$"),         # lone punctuation
    re.compile(r"^\s*(you|bye|okay|oh)\.?\s*$", re.IGNORECASE),  # single phantom syllables on noise
]


def _preload_cuda_libraries() -> None:
    """Preload NVIDIA CUDA runtime libraries (cublas, cudnn, nvrtc) into global symbol table."""
    try:
        import ctypes
        import sys

        site_pkgs = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia"
        if site_pkgs.is_dir():
            for so_file in sorted(site_pkgs.glob("*/lib/*.so*")):
                if so_file.is_file() and not so_file.name.endswith(".a"):
                    try:
                        ctypes.CDLL(str(so_file), mode=ctypes.RTLD_GLOBAL)
                    except Exception:
                        pass
    except Exception:
        pass


class SpeechToText:
    """Fast, local speech-to-text transcriber powered by faster-whisper / CTranslate2."""

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "auto",
        compute_type: str = "default",
        language: str = "en",
        initial_prompt: str = (
            "Hey Raphael. Conversational voice commands and questions. "
            "Words like: open, close, search, play, pause, stop, set, remind, timer, weather, time, "
            "what, how, why, when, where, who, tell me, can you, could you, please, thanks."
        ),
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.initial_prompt = initial_prompt

        self.model: WhisperModel | None = None
        self._ready = threading.Event()
        self._load_error: Exception | None = None

        _preload_cuda_libraries()
        # Load model in background thread — startup continues immediately
        self._loader = threading.Thread(target=self._load_model_bg, daemon=True)
        self._loader.start()

    def _load_model_bg(self) -> None:
        """Background thread: load model then set the ready event."""
        try:
            self.model = self._load_model(self.model_size, self.device, self.compute_type)
        except Exception as err:
            self._load_error = err
            logger.error("STT model failed to load: %s", err)
        finally:
            self._ready.set()

    def is_ready(self) -> bool:
        """Return True if the model has finished loading."""
        return self._ready.is_set()

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Block until model is loaded (or timeout seconds). Returns True if loaded."""
        return self._ready.wait(timeout=timeout)

    def _load_model(self, model_size: str, device: str, compute_type: str) -> WhisperModel:
        # Resolve 'auto' device
        resolved_device = device
        resolved_compute = compute_type

        if device == "auto":
            try:
                import ctranslate2
                if ctranslate2.get_cuda_device_count() > 0:
                    resolved_device = "cuda"
                    resolved_compute = "float16" if compute_type in ("default", "auto") else compute_type
                else:
                    resolved_device = "cpu"
                    resolved_compute = "int8" if compute_type in ("default", "auto") else compute_type
            except Exception:
                resolved_device = "cpu"
                resolved_compute = "int8"
        elif compute_type in ("default", "auto"):
            resolved_compute = "float16" if resolved_device == "cuda" else "int8"

        logger.info(
            "Loading faster-whisper STT model: '%s' (device=%s, compute_type=%s)",
            model_size,
            resolved_device,
            resolved_compute,
        )

        try:
            model = WhisperModel(
                model_size_or_path=model_size,
                device=resolved_device,
                compute_type=resolved_compute,
            )
            self.device = resolved_device
            self.compute_type = resolved_compute
            return model
        except Exception as err:
            logger.warning(
                "Whisper initialization on %s/%s failed (%s). Falling back to CPU/int8.",
                resolved_device,
                resolved_compute,
                err,
            )
            self.device = "cpu"
            self.compute_type = "int8"
            return WhisperModel(
                model_size_or_path=model_size,
                device="cpu",
                compute_type="int8",
            )

    @staticmethod
    def _is_hallucination(text: str, no_speech_prob: float, avg_logprob: float) -> bool:
        """Check if a segment appears to be a Whisper hallucination on noise/silence."""
        if not text:
            return True
        # High probability of silence / no speech
        if no_speech_prob > 0.65:
            return True
        # Very low confidence decoding
        if avg_logprob < -1.3:
            return True
        # Matches common hallucination regexes
        for pattern in _HALLUCINATION_PATTERNS:
            if pattern.match(text):
                return True
        return False

    def transcribe(
        self,
        audio: np.ndarray | str | Path,
        language: str | None = None,
        beam_size: int = 8,
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
        beam_size: int = 8,
    ) -> dict[str, Any]:
        """Transcribe audio with Silero VAD filtering and anti-hallucination safeguards."""
        # Wait for background model load if it hasn't finished yet
        if not self._ready.is_set():
            logger.info("⏳ STT model still loading — waiting...")
            self._ready.wait()
        if self._load_error or self.model is None:
            raise RuntimeError(f"STT model failed to load: {self._load_error}")

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

        # Advanced decode options — deterministic greedy with beam search for maximum accuracy
        decode_kwargs: dict[str, Any] = {
            "language": target_lang,
            "beam_size": beam_size,
            "temperature": 0,                        # Greedy decoding — no random word sampling
            "best_of": 1,                            # With temperature=0, only one candidate needed
            "initial_prompt": self.initial_prompt,
            "condition_on_previous_text": False,     # Prevents hallucinations cascading across segments
            "vad_filter": True,                      # Silero VAD strips silence before inference
            "vad_parameters": dict(
                min_silence_duration_ms=200,
                speech_pad_ms=400,                   # Extra padding so word edges aren't clipped
            ),
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 3,
            "compression_ratio_threshold": 2.2,      # Tighter: discard garbled/repetition-heavy output
            "log_prob_threshold": -0.7,              # Tighter: drop low-confidence segments
            "no_speech_threshold": 0.55,             # Slightly tighter silence filter
        }

        try:
            segments_generator, info = self.model.transcribe(audio_input, **decode_kwargs)
            segments_list = []
            text_parts = []
            for segment in segments_generator:
                cleaned_text = segment.text.strip()
                if not cleaned_text:
                    continue
                if self._is_hallucination(cleaned_text, segment.no_speech_prob, segment.avg_logprob):
                    logger.debug(
                        "Filtered out hallucination: '%s' (no_speech_prob=%.2f, logprob=%.2f)",
                        cleaned_text,
                        segment.no_speech_prob,
                        segment.avg_logprob,
                    )
                    continue

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
            segments_generator, info = self.model.transcribe(audio_input, **decode_kwargs)
            segments_list = []
            text_parts = []
            for segment in segments_generator:
                cleaned_text = segment.text.strip()
                if not cleaned_text:
                    continue
                if self._is_hallucination(cleaned_text, segment.no_speech_prob, segment.avg_logprob):
                    continue
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
            getattr(info, "duration", 0.0),
            elapsed,
            full_text,
        )

        return {
            "text": full_text,
            "language": getattr(info, "language", target_lang),
            "segments": segments_list,
            "duration": getattr(info, "duration", 0.0),
            "latency": elapsed,
        }

