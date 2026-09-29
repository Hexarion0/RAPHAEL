import argparse
import asyncio
import io
import re
import threading
from pathlib import Path

import edge_tts
import msgpack
import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from piper import PiperVoice
from piper.config import SynthesisConfig
from piper.download_voices import download_voice

from raphael.config import get_settings
from raphael.logging import get_logger

logger = get_logger("audio.tts")


class TextToSpeech:
    """Neural Text-to-Speech engine supporting Fish Speech (zero-shot local cloning), Microsoft Edge-TTS, and Piper ONNX."""

    def __init__(
        self,
        voice_name: str | None = None,
        engine: str | None = None,
        models_dir: str | Path = "models/tts",
        speed: float | None = None,
        output_device: int | str | None = None,
        enabled: bool = True,
        fish_speech_url: str | None = None,
        fish_ref_audio: str | Path | None = None,
        fish_ref_text: str | None = None,
        fish_temperature: float | None = None,
        fish_top_p: float | None = None,
        fish_repetition_penalty: float | None = None,
        fish_chunk_length: int | None = None,
        fish_max_new_tokens: int | None = None,
    ) -> None:
        settings = get_settings().audio

        self.voice_name = voice_name if voice_name is not None else settings.tts_voice
        raw_engine = engine if engine is not None else settings.tts_engine
        self.engine = raw_engine.lower()

        # Auto-detect engine from voice name if engine not explicitly specified
        if engine is None:
            if self.voice_name.lower() in ("mommy", "fish_speech", "fish"):
                self.engine = "fish_speech"
            elif "neural" in self.voice_name.lower():
                self.engine = "edge_tts"
            elif self.voice_name.endswith(".onnx") or "medium" in self.voice_name.lower() or self.voice_name == "custom_voice":
                self.engine = "piper"

        self.models_dir = Path(models_dir)
        self.speed = speed if speed is not None else settings.tts_speed
        self.output_device = output_device if output_device is not None else settings.output_device
        self.enabled = enabled

        # Fish Speech zero-shot parameters
        self.fish_speech_url = fish_speech_url or settings.fish_speech_url
        self.fish_ref_audio = fish_ref_audio or settings.fish_ref_audio
        self.fish_ref_text = fish_ref_text or settings.fish_ref_text
        self.fish_temperature = fish_temperature if fish_temperature is not None else settings.fish_temperature
        self.fish_top_p = fish_top_p if fish_top_p is not None else settings.fish_top_p
        self.fish_repetition_penalty = fish_repetition_penalty if fish_repetition_penalty is not None else settings.fish_repetition_penalty
        self.fish_chunk_length = fish_chunk_length if fish_chunk_length is not None else settings.fish_chunk_length
        self.fish_max_new_tokens = fish_max_new_tokens if fish_max_new_tokens is not None else settings.fish_max_new_tokens

        self._cached_ref_audio_bytes: bytes | None = None
        self._voice: PiperVoice | None = None
        self._is_playing = False
        self._stop_event = threading.Event()  # signals stop() to unblock speak(block=True)

        if self.enabled:
            logger.info("Initializing TTS engine '%s' with voice '%s'...", self.engine, self.voice_name)
            if self.engine == "piper":
                self._load_voice()
            elif self.engine in ("fish_speech", "fish"):
                logger.info("Fish Speech zero-shot engine ready (URL: %s, Voice: %s).", self.fish_speech_url, self.voice_name)
            else:
                logger.info("Edge-TTS engine ready (voice: %s).", self.voice_name)

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

    def _get_ref_audio_bytes(self) -> bytes | None:
        """Load and cache reference audio bytes for zero-shot voice cloning."""
        if self._cached_ref_audio_bytes is not None:
            return self._cached_ref_audio_bytes

        ref_path = Path(self.fish_ref_audio)
        if not ref_path.is_file():
            # Check relative to project root / current working dir
            for candidate in [
                Path.cwd() / self.fish_ref_audio,
                Path(__file__).resolve().parent.parent.parent.parent / self.fish_ref_audio,
                Path("/home/hexarion/Raphael") / self.fish_ref_audio,
            ]:
                if candidate.is_file():
                    ref_path = candidate
                    break

        if ref_path.is_file():
            try:
                self._cached_ref_audio_bytes = ref_path.read_bytes()
                logger.debug("Loaded reference audio (%d bytes) from %s", len(self._cached_ref_audio_bytes), ref_path)
                return self._cached_ref_audio_bytes
            except Exception as err:
                logger.warning("Failed to read reference audio '%s': %s", ref_path, err)
        else:
            logger.warning("Reference audio file not found at '%s'", self.fish_ref_audio)
        return None

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

    def _synthesize_fish_speech(self, text: str) -> tuple[np.ndarray, int] | None:
        """Synthesize using local Fish Speech zero-shot TTS API server."""
        ref_bytes = self._get_ref_audio_bytes()
        references = []
        if ref_bytes is not None and self.fish_ref_text:
            references.append({
                "audio": ref_bytes,
                "text": self.fish_ref_text,
            })

        payload = {
            "text": text,
            "references": references,
            "reference_id": None,
            "max_new_tokens": self.fish_max_new_tokens,
            "chunk_length": self.fish_chunk_length,
            "top_p": self.fish_top_p,
            "repetition_penalty": self.fish_repetition_penalty,
            "temperature": self.fish_temperature,
            "format": "wav",
            "streaming": False,
            "use_memory_cache": "on",
            "seed": None,
            "normalize": True,
        }

        try:
            packed_data = msgpack.packb(payload, use_bin_type=True)
            response = requests.post(
                self.fish_speech_url,
                data=packed_data,
                headers={"Content-Type": "application/msgpack"},
                timeout=40.0,
            )
            if response.status_code != 200:
                logger.warning(
                    "Fish Speech server returned HTTP %d: %s. Falling back...",
                    response.status_code,
                    response.text[:200],
                )
                return self._fallback_synthesize(text)

            audio_data, sample_rate = sf.read(io.BytesIO(response.content))
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            return audio_data.astype(np.float32), sample_rate

        except Exception as err:
            logger.warning("Fish Speech synthesis failed (%s). Falling back...", err)
            return self._fallback_synthesize(text)

    def _fallback_synthesize(self, text: str) -> tuple[np.ndarray, int] | None:
        """Gracefully fall back to Edge-TTS or Piper when primary engine fails."""
        try:
            res = self._synthesize_edge_tts(text)
            if res is not None:
                return res
        except Exception:
            pass
        return self._synthesize_piper(text)

    def _synthesize_edge_tts(self, text: str) -> tuple[np.ndarray, int] | None:
        """Synthesize using Microsoft Edge-TTS neural speech."""
        rate_str = "+0%"
        if abs(self.speed - 1.0) > 0.05:
            pct = int((self.speed - 1.0) * 100)
            rate_str = f"+{pct}%" if pct >= 0 else f"{pct}%"

        edge_voice = self.voice_name
        if "neural" not in edge_voice.lower():
            edge_voice = "en-US-AvaNeural"

        async def _run_edge():
            communicate = edge_tts.Communicate(
                text=text,
                voice=edge_voice,
                rate=rate_str,
            )
            raw_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    raw_data += chunk["data"]
            return raw_data

        try:
            # Run in isolated event loop to support multi-threaded caller
            raw_audio = asyncio.run(_run_edge())
            if not raw_audio:
                return None
            audio_array, sample_rate = sf.read(io.BytesIO(raw_audio))
            # Convert to float32 mono array
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)
            return audio_array.astype(np.float32), sample_rate
        except Exception as err:
            logger.warning("Edge-TTS synthesis failed (%s). Falling back to Piper...", err)
            return self._synthesize_piper(text)

    def _synthesize_piper(self, text: str) -> tuple[np.ndarray, int] | None:
        """Synthesize using local Piper ONNX model."""
        if self._voice is None:
            self._load_voice()
        if self._voice is None:
            return None

        syn_config = SynthesisConfig(
            length_scale=1.0 / max(0.2, min(self.speed, 3.0)),
        )

        audio_chunks: list[np.ndarray] = []
        sample_rate = 22050

        for chunk in self._voice.synthesize(text, syn_config=syn_config):
            sample_rate = chunk.sample_rate
            if chunk.audio_float_array is not None and chunk.audio_float_array.size > 0:
                audio_chunks.append(chunk.audio_float_array)

        if not audio_chunks:
            return None

        combined_audio = np.concatenate(audio_chunks)
        return combined_audio, sample_rate

    def synthesize(self, text: str) -> tuple[np.ndarray, int] | None:
        """Synthesize text to a 1D float32 audio numpy array and sample rate."""
        if not self.enabled:
            return None

        clean_text = self.clean_text_for_speech(text)
        if not clean_text:
            return None

        if self.engine in ("fish_speech", "fish") or self.voice_name.lower() == "mommy":
            return self._synthesize_fish_speech(clean_text)
        elif self.engine == "edge_tts" or "neural" in self.voice_name.lower():
            return self._synthesize_edge_tts(clean_text)
        return self._synthesize_piper(clean_text)

    def speak(self, text: str, block: bool = True) -> bool:
        """Synthesize and play speech audio through speakers."""
        if not self.enabled:
            return False

        synth_result = self.synthesize(text)
        if synth_result is None:
            return False

        audio, sample_rate = synth_result

        try:
            self._stop_event.clear()
            self._is_playing = True
            sd.play(audio, samplerate=sample_rate, device=self.output_device)
            if block:
                # Poll our stop_event instead of calling sd.wait() directly.
                # On Linux, sd.stop() does not reliably unblock a concurrent sd.wait()
                # due to a PortAudio race condition, which permanently stalls the worker.
                while not self._stop_event.wait(timeout=0.02):
                    try:
                        if not sd.get_stream().active:
                            break
                    except Exception:
                        break  # stream gone — playback finished
                self._is_playing = False
            else:
                # Spawn a daemon thread to clear _is_playing once audio finishes or stop is requested
                def _wait_done() -> None:
                    try:
                        while not self._stop_event.wait(timeout=0.02):
                            try:
                                if not sd.get_stream().active:
                                    break
                            except Exception:
                                break
                    finally:
                        self._is_playing = False

                threading.Thread(target=_wait_done, daemon=True).start()
            return True
        except Exception as err:
            logger.error("Error playing TTS audio: %s", err)
            self._is_playing = False
            return False

    def stop(self) -> None:
        """Immediately stop audio playback (barge-in support)."""
        self._stop_event.set()  # unblocks any speak(block=True) polling loop
        try:
            sd.stop()
        except Exception as err:
            logger.debug("Error stopping sounddevice: %s", err)
        finally:
            self._is_playing = False

    def is_speaking(self) -> bool:
        """Check whether audio is currently playing."""
        return self._is_playing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RAPHAEL Text-to-Speech")
    parser.add_argument("text", nargs="?", default="Hello! I am Raphael, your desktop companion.", help="Text to speak")
    parser.add_argument("--engine", default=None, help="TTS engine (fish_speech, edge_tts, piper)")
    parser.add_argument("--voice", default=None, help="TTS voice name")
    args = parser.parse_args()

    app_settings = get_settings()
    selected_engine = args.engine or app_settings.audio.tts_engine
    selected_voice = args.voice or app_settings.audio.tts_voice

    tts = TextToSpeech(
        voice_name=selected_voice,
        engine=selected_engine,
        speed=app_settings.audio.tts_speed,
        output_device=app_settings.audio.output_device,
        fish_speech_url=app_settings.audio.fish_speech_url,
        fish_ref_audio=app_settings.audio.fish_ref_audio,
        fish_ref_text=app_settings.audio.fish_ref_text,
    )
    print(f"Synthesizing: '{args.text}' [Engine: {tts.engine}, Voice: {tts.voice_name}]")
    success = tts.speak(args.text, block=True)
    print(f"Speech finished (success={success}).")
