"""Text-to-Speech (TTS) engine using Piper neural voice synthesis and sounddevice playback."""

import re
from pathlib import Path

import numpy as np
import sounddevice as sd
from piper import PiperVoice
from piper.config import SynthesisConfig
from piper.download_voices import download_voice

from raphael.logging import get_logger

logger = get_logger("audio.tts")


class TextToSpeech:
    """Neural Text-to-Speech engine supporting Piper voices with interruptible playback."""

    def __init__(
        self,
        voice_name: str = "en_GB-alan-medium",
        models_dir: str | Path = "models/tts",
        speed: float = 1.0,
        output_device: int | str | None = None,
        enabled: bool = True,
    ) -> None:
        self.voice_name = voice_name
        self.models_dir = Path(models_dir)
        self.speed = speed
        self.output_device = output_device
        self.enabled = enabled
        self._voice: PiperVoice | None = None
        self._is_playing = False

        if self.enabled:
            self._load_voice()

    def _ensure_model_files(self) -> tuple[Path, Path]:
        """Ensure voice model .onnx and .onnx.json files exist locally, downloading if necessary."""
        # 1. Check if direct file path was provided
        direct_path = Path(self.voice_name)
        if direct_path.is_file() and direct_path.suffix == ".onnx":
            json_file = direct_path.with_suffix(".onnx.json")
            if not json_file.is_file():
                json_file = direct_path.with_name(f"{direct_path.stem}.json")
            return direct_path, json_file

        # 2. Check in models directory
        self.models_dir.mkdir(parents=True, exist_ok=True)
        onnx_file = self.models_dir / f"{self.voice_name}.onnx"
        json_file = self.models_dir / f"{self.voice_name}.onnx.json"

        if not onnx_file.is_file() or not json_file.is_file():
            logger.info("Downloading Piper TTS voice model '%s'...", self.voice_name)
            try:
                download_voice(self.voice_name, self.models_dir)
                logger.info("Successfully downloaded TTS voice '%s'.", self.voice_name)
            except Exception as err:
                logger.error("Failed to download voice model '%s': %s", self.voice_name, err)
                raise

        return onnx_file, json_file

    def _load_voice(self) -> None:
        """Load the Piper neural voice model."""
        try:
            onnx_path, json_path = self._ensure_model_files()
            logger.info("Loading Piper TTS voice: %s", self.voice_name)
            self._voice = PiperVoice.load(
                model_path=str(onnx_path),
                config_path=str(json_path),
                use_cuda=False,
            )
            logger.info("Piper TTS engine initialized successfully.")
        except Exception as err:
            logger.warning("Failed to initialize Piper TTS: %s. Audio output disabled.", err)
            self._voice = None
            self.enabled = False

    @staticmethod
    def clean_text_for_speech(text: str) -> str:
        """Clean markdown, code blocks, reasoning tags, emojis, and symbols for spoken speech."""
        # Strip reasoning / thinking tags <think>...</think> and <thought>...</thought>
        clean = re.sub(r"<(think|thought)>[\s\S]*?</\1>", "", text, flags=re.IGNORECASE)
        clean = re.sub(r"^<(think|thought)>[\s\S]*", "", clean, flags=re.IGNORECASE)
        # Remove code blocks
        clean = re.sub(r"```[\s\S]*?```", " [code omitted] ", clean)
        # Remove inline backticks
        clean = re.sub(r"`([^`]+)`", r"\1", clean)
        # Remove markdown links [text](url) -> text
        clean = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", clean)
        # Remove markdown headers (#, ##, etc.)
        clean = re.sub(r"^\s*#{1,6}\s+", "", clean, flags=re.MULTILINE)
        # Remove markdown bold/italics
        clean = re.sub(r"[*_~]{1,2}([^*_~]+)[*_~]{1,2}", r"\1", clean)
        # Remove remaining stray asterisks/underscores
        clean = re.sub(r"[*_~]", "", clean)
        # Remove bullets / list markers
        clean = re.sub(r"^\s*[-*+]\s+", "", clean, flags=re.MULTILINE)
        clean = re.sub(r"^\s*\d+\.\s+", "", clean, flags=re.MULTILINE)
        # Remove emojis (surrogate pairs and emoji blocks)
        clean = re.sub(r"[\U00010000-\U0010ffff]", "", clean)
        # Collapse multiple spaces / newlines
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    def synthesize(self, text: str) -> tuple[np.ndarray, int] | None:
        """Synthesize text to a 1D float32 audio numpy array and sample rate."""
        if not self.enabled or self._voice is None:
            return None

        clean_text = self.clean_text_for_speech(text)
        if not clean_text:
            return None

        syn_config = SynthesisConfig(
            length_scale=1.0 / max(0.2, min(self.speed, 3.0)),
        )

        audio_chunks: list[np.ndarray] = []
        sample_rate = 22050

        for chunk in self._voice.synthesize(clean_text, syn_config=syn_config):
            sample_rate = chunk.sample_rate
            if chunk.audio_float_array is not None and chunk.audio_float_array.size > 0:
                audio_chunks.append(chunk.audio_float_array)

        if not audio_chunks:
            return None

        combined_audio = np.concatenate(audio_chunks)
        return combined_audio, sample_rate

    def speak(self, text: str, block: bool = True) -> bool:
        """Synthesize and play speech audio through speakers."""
        if not self.enabled:
            return False

        synth_result = self.synthesize(text)
        if synth_result is None:
            return False

        audio, sample_rate = synth_result

        try:
            self._is_playing = True
            sd.play(audio, samplerate=sample_rate, device=self.output_device)
            if block:
                sd.wait()
                self._is_playing = False
            return True
        except Exception as err:
            logger.error("Error playing TTS audio: %s", err)
            self._is_playing = False
            return False

    def stop(self) -> None:
        """Immediately stop audio playback (barge-in support)."""
        try:
            sd.stop()
        except Exception as err:
            logger.debug("Error stopping sounddevice: %s", err)
        finally:
            self._is_playing = False

    def is_speaking(self) -> bool:
        """Check whether audio is currently playing."""
        return self._is_playing
