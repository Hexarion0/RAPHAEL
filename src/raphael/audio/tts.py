import argparse
import asyncio
import io
import queue
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path

import edge_tts
import msgpack
import numpy as np
import requests
import soundfile as sf
from piper import PiperVoice
from piper.config import SynthesisConfig
from piper.download_voices import download_voice

from raphael.audio.native import sd
from raphael.config import get_settings, normalize_audio_device
from raphael.logging import get_logger

logger = get_logger("audio.tts")

_CUSTOM_PIPER_VOICES = {"custom_voice", "custom"}


def resolve_tts_engine(voice_name: str, engine: str = "auto") -> str:
    """An explicit engine wins; otherwise choose the engine for the voice."""
    selected = engine.lower()
    if selected == "fish":
        selected = "fish_speech"
    if selected == "auto":
        if voice_name.lower() in {"mommy", "fish_speech", "fish"}:
            return "fish_speech"
        if "neural" in voice_name.lower():
            return "edge_tts"
        return "piper"
    if selected not in {"piper", "fish_speech", "edge_tts"}:
        raise ValueError(f"Unknown TTS engine: {engine}")
    return selected


def resolve_piper_voice_paths(voice_name: str, models_dir: str | Path) -> tuple[Path, Path]:
    """Resolve a Piper ONNX + config pair without starting any TTS server."""
    models = Path(models_dir)
    direct_path = Path(voice_name)
    if direct_path.is_file() and direct_path.suffix == ".onnx":
        json_file = direct_path.with_suffix(".onnx.json")
        if not json_file.is_file():
            json_file = direct_path.with_name(f"{direct_path.stem}.json")
        return direct_path, json_file

    models.mkdir(parents=True, exist_ok=True)
    onnx_file = models / f"{voice_name}.onnx"
    json_file = models / f"{voice_name}.onnx.json"
    if not json_file.is_file():
        alt_json = models / f"{voice_name}.json"
        if alt_json.is_file():
            json_file = alt_json

    if voice_name.lower() in _CUSTOM_PIPER_VOICES or onnx_file.is_file():
        if not onnx_file.is_file() or not json_file.is_file():
            raise FileNotFoundError(
                f"Custom Piper voice '{voice_name}' not found in {models}. "
                "Run `python -m raphael prepare-voice` then `./scripts/train_piper_voice.sh`."
            )
        return onnx_file, json_file

    if not onnx_file.is_file() or not json_file.is_file():
        logger.info("Downloading Piper TTS voice model '%s'...", voice_name)
        download_voice(voice_name, models)
        logger.info("Successfully downloaded TTS voice '%s'.", voice_name)
    return onnx_file, json_file


class TextToSpeech:
    """Neural speech using local Fish Speech, Microsoft Edge-TTS, or Piper ONNX."""

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
        self.engine = resolve_tts_engine(self.voice_name, raw_engine)

        self.models_dir = Path(models_dir)
        self.speed = speed if speed is not None else settings.tts_speed
        self.output_device = normalize_audio_device(
            output_device if output_device is not None else settings.output_device
        )
        self.enabled = enabled

        # Fish Speech zero-shot parameters
        self.fish_speech_url = fish_speech_url or settings.fish_speech_url
        self.fish_ref_audio = fish_ref_audio or settings.fish_ref_audio
        self.fish_ref_text = fish_ref_text or settings.fish_ref_text
        self.fish_temperature = (
            fish_temperature if fish_temperature is not None else settings.fish_temperature
        )
        self.fish_top_p = fish_top_p if fish_top_p is not None else settings.fish_top_p
        self.fish_repetition_penalty = (
            fish_repetition_penalty
            if fish_repetition_penalty is not None
            else settings.fish_repetition_penalty
        )
        self.fish_chunk_length = (
            fish_chunk_length if fish_chunk_length is not None else settings.fish_chunk_length
        )
        self.fish_max_new_tokens = (
            fish_max_new_tokens if fish_max_new_tokens is not None else settings.fish_max_new_tokens
        )

        self._cached_ref_audio_bytes: bytes | None = None
        self._voice: PiperVoice | None = None
        self._fallback_voice: PiperVoice | None = None
        self._is_playing = False
        self._playback_lock = threading.Lock()
        self._playback_generation = 0
        self._stop_event = threading.Event()  # signals stop() to unblock speak(block=True)
        self._pending_text = ""
        self._play_started_at = 0.0
        self._play_duration = 0.0
        self._stream_text = ""
        self._stream_spoken: list[str] | None = None
        self._stream_played_seconds = 0.0
        self._synthesis_lock = threading.Lock()
        self._synthesis_slots = threading.BoundedSemaphore(2)

        if self.enabled:
            logger.info(
                "Initializing TTS engine '%s' with voice '%s'...", self.engine, self.voice_name
            )
            if self.engine == "piper":
                self._load_voice()
            elif self.engine in ("fish_speech", "fish"):
                logger.info(
                    "Fish Speech zero-shot engine ready (URL: %s, Voice: %s).",
                    self.fish_speech_url,
                    self.voice_name,
                )
            else:
                logger.info("Edge-TTS engine ready (voice: %s).", self.voice_name)

    def _ensure_model_files(self) -> tuple[Path, Path]:
        """Ensure voice model .onnx and .onnx.json files exist locally, downloading if necessary."""
        return resolve_piper_voice_paths(self.voice_name, self.models_dir)

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
                logger.debug(
                    "Loaded reference audio (%d bytes) from %s",
                    len(self._cached_ref_audio_bytes),
                    ref_path,
                )
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
            references.append(
                {
                    "audio": ref_bytes,
                    "text": self.fish_ref_text,
                }
            )

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
        """Use installed local voices before attempting a network-based fallback."""
        if self._fallback_voice is not None:
            return self._synthesize_piper(text, voice=self._fallback_voice)
        for name in dict.fromkeys([self.voice_name, "en_GB-alan-medium", "en_US-amy-medium"]):
            onnx = self.models_dir / f"{name}.onnx"
            config = onnx.with_suffix(".onnx.json")
            if not config.is_file():
                config = onnx.with_suffix(".json")
            if not onnx.is_file() or not config.is_file():
                continue
            try:
                self._fallback_voice = PiperVoice.load(
                    model_path=str(onnx),
                    config_path=str(config),
                    use_cuda=False,
                )
                logger.info("Using local Piper fallback voice '%s'.", name)
                return self._synthesize_piper(text, voice=self._fallback_voice)
            except Exception as err:
                self._fallback_voice = None
                logger.warning("Piper fallback '%s' failed: %s", name, err)
        return self._synthesize_edge_tts(text)

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

    def _synthesize_piper(
        self,
        text: str,
        voice: PiperVoice | None = None,
    ) -> tuple[np.ndarray, int] | None:
        """Synthesize using local Piper ONNX model."""
        if voice is None:
            if self._voice is None:
                self._load_voice()
            voice = self._voice
        if voice is None:
            return None

        syn_config = SynthesisConfig(
            length_scale=1.0 / max(0.2, min(self.speed, 3.0)),
        )

        audio_chunks: list[np.ndarray] = []
        sample_rate = 22050

        for chunk in voice.synthesize(text, syn_config=syn_config):
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

        if self.engine == "fish_speech":
            return self._synthesize_fish_speech(clean_text)
        elif self.engine == "edge_tts":
            return self._synthesize_edge_tts(clean_text)
        return self._synthesize_piper(clean_text)

    def begin_stream(self) -> int:
        """Start cumulative playback tracking for a sentence-streamed reply."""
        with self._playback_lock:
            self._playback_generation += 1
            self._pending_text = ""
            self._play_started_at = 0.0
            self._play_duration = 0.0
            self._stream_text = ""
            self._stream_spoken = []
            self._stream_played_seconds = 0.0
            return self._playback_generation

    def update_stream(self, generation: int, text: str) -> None:
        """Track known generated text without reviving a canceled stream."""
        with self._playback_lock:
            if generation == self._playback_generation:
                self._stream_text = self.clean_text_for_speech(text)

    def end_stream(self, generation: int) -> None:
        """Clear completed stream state; an old worker cannot clear a newer reply."""
        with self._playback_lock:
            if generation == self._playback_generation:
                self._stream_text = ""
                self._stream_spoken = None
                self._stream_played_seconds = 0.0

    def _synthesize_cancelable(
        self, text: str, cancel_event: threading.Event, generation: int,
    ) -> tuple[np.ndarray, int] | None:
        """Let a canceled caller leave while a bounded synthesis job unwinds.

        Voice inference is serialized: an old Piper/Fish job cannot race a new
        job against the same voice object. Its late audio never reaches playback.
        """
        def current() -> bool:
            return not cancel_event.is_set() and generation == self._playback_generation

        while not self._synthesis_slots.acquire(timeout=0.02):
            if not current():
                return None
        result: queue.Queue = queue.Queue(maxsize=1)

        def synthesize() -> None:
            acquired = False
            try:
                while current():
                    acquired = self._synthesis_lock.acquire(timeout=0.02)
                    if acquired:
                        break
                if acquired and current():
                    result.put(self.synthesize(text))
                else:
                    result.put(None)
            except Exception as err:
                result.put(err)
            finally:
                if acquired:
                    self._synthesis_lock.release()
                self._synthesis_slots.release()

        worker = threading.Thread(target=synthesize, name="raphael-synthesis", daemon=True)
        try:
            worker.start()
        except Exception:
            self._synthesis_slots.release()
            raise
        while current():
            try:
                audio = result.get(timeout=0.02)
                if isinstance(audio, Exception):
                    raise audio
                return audio
            except queue.Empty:
                continue
        return None

    def speak(
        self,
        text: str,
        block: bool = True,
        *,
        cancel_event: threading.Event | None = None,
        generation: int | None = None,
        on_start: Callable[[], None] | None = None,
    ) -> bool:
        """Synthesize and play speech audio through speakers."""
        if not self.enabled:
            return False

        with self._playback_lock:
            if generation is None:
                generation = self._playback_generation
            if generation != self._playback_generation or (cancel_event and cancel_event.is_set()):
                return False
            self._pending_text = self.clean_text_for_speech(text)
            self._play_started_at = 0.0
            self._play_duration = 0.0
        if cancel_event is not None:
            synth_result = self._synthesize_cancelable(text, cancel_event, generation)
        else:
            with self._synthesis_lock:
                synth_result = self.synthesize(text)
        if synth_result is None:
            with self._playback_lock:
                if generation == self._playback_generation:
                    self._pending_text = ""
            return False

        audio, sample_rate = synth_result

        try:
            with self._playback_lock:
                if (
                    generation != self._playback_generation
                    or (cancel_event and cancel_event.is_set())
                ):
                    return False
                self._stop_event.clear()
                self._is_playing = True
                self._play_started_at = time.monotonic()
                self._play_duration = audio.size / sample_rate
                sd.play(audio, samplerate=sample_rate, device=self.output_device)
            if on_start:
                on_start()
            if block:
                # Poll our stop_event instead of calling sd.wait() directly.
                # On Linux, sd.stop() does not reliably unblock a concurrent sd.wait()
                # due to a PortAudio race condition, which permanently stalls the worker.
                while not self._stop_event.wait(timeout=0.02):
                    if cancel_event and cancel_event.is_set():
                        self.stop()
                        break
                    try:
                        if not sd.get_stream().active:
                            break
                    except Exception:
                        break  # stream gone — playback finished
                with self._playback_lock:
                    if generation == self._playback_generation:
                        self._is_playing = False
                        if self._stream_spoken is not None:
                            self._stream_spoken.append(self._pending_text)
                            self._stream_played_seconds += self._play_duration
                        self._pending_text = ""
            else:
                # Clear playback state when audio finishes or stop is requested.
                def _wait_done() -> None:
                    try:
                        while not self._stop_event.wait(timeout=0.02):
                            try:
                                if not sd.get_stream().active:
                                    break
                            except Exception:
                                break
                    finally:
                        with self._playback_lock:
                            if generation == self._playback_generation:
                                self._is_playing = False
                                self._pending_text = ""

                threading.Thread(target=_wait_done, daemon=True).start()
            return generation == self._playback_generation
        except Exception as err:
            logger.error("Error playing TTS audio: %s", err)
            self._is_playing = False
            return False

    def stop(self) -> dict | None:
        """Immediately stop audio playback (barge-in support)."""
        with self._playback_lock:
            interruption = None
            if self._pending_text or self._stream_text:
                elapsed = (
                    max(0.0, time.monotonic() - self._play_started_at)
                    if self._pending_text and self._play_started_at else 0.0
                )
                duration = self._play_duration if self._pending_text else 0.0
                fraction = min(1.0, elapsed / duration) if duration else 0.0
                words = self._pending_text.split()
                heard = int(len(words) * fraction)
                spoken = " ".join([*(self._stream_spoken or []), " ".join(words[:heard])]).strip()
                full = self._stream_text or self._pending_text
                remaining = " ".join(full.split()[len(spoken.split()):])
                interruption = {
                    "full_text": full[:4000],
                    "estimated_spoken_text": spoken[:4000],
                    "remaining_text": remaining[:4000],
                    "played_seconds": round(self._stream_played_seconds + elapsed, 2),
                    "duration_seconds": round(self._stream_played_seconds + duration, 2),
                }
            self._pending_text = ""
            self._stream_text = ""
            self._stream_spoken = None
            self._stream_played_seconds = 0.0
            self._playback_generation += 1
            self._stop_event.set()
            was_playing = self._is_playing
            self._is_playing = False
        if not was_playing:
            return interruption
        try:
            sd.stop()
        except Exception as err:
            logger.debug("Error stopping sounddevice: %s", err)
        finally:
            self._is_playing = False
        return interruption

    def is_speaking(self) -> bool:
        """Check whether audio is currently playing."""
        return self._is_playing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RAPHAEL Text-to-Speech")
    parser.add_argument(
        "text",
        nargs="?",
        default="Hello! I am Raphael, your desktop companion.",
        help="Text to speak",
    )
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
