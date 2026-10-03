"""Centralized configuration and secrets management for RAPHAEL."""

from functools import lru_cache

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def normalize_audio_device(value: int | str | None) -> int | str | None:
    """Normalize environment indices while retaining device name queries."""
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        if value.isdecimal():
            return int(value)
    return value


class AppConfig(BaseModel):
    """General application configuration."""

    env: str = Field(
        default="development",
        description="Application environment (development, production, test)",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    debug: bool = Field(default=False, description="Enable debug mode")


class MemoryConfig(BaseModel):
    """Memory and persistence configuration."""

    db_path: str = Field(
        default="data/raphael.db",
        description="Path to SQLite memory database file",
    )
    max_short_term_turns: int = Field(
        default=10,
        description="Maximum turns of conversation to keep in short-term context",
    )


class ProviderConfig(BaseModel):
    """AI Provider API keys and endpoints."""

    nim_api_key: SecretStr | None = Field(
        default=None,
        description="NVIDIA NIM API key",
    )
    nim_model: str = Field(
        default="nvidia/nemotron-3-super-120b-a12b",
        description="Default NVIDIA NIM model name",
    )
    nim_complex_model: str = Field(
        default="nvidia/nemotron-3-ultra-550b-a55b",
        description="NVIDIA NIM model for complex reasoning and coding tasks",
    )
    nim_fallback_model: str = Field(
        default="meta/llama-3.2-90b-vision-instruct",
        description="NVIDIA NIM fallback model for failover / rate-limiting",
    )
    openrouter_api_key: SecretStr | None = Field(
        default=None,
        description="OpenRouter API key",
    )
    groq_api_key: SecretStr | None = Field(
        default=None,
        description="Groq API key",
    )
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama local server endpoint",
    )


class AudioConfig(BaseModel):
    """Audio, wake word, and speech-to-text settings."""

    wake_word: str = Field(default="hey raphael", description="Wake word trigger phrase")
    wake_threshold: float = Field(
        default=0.5,
        description="Wake word detection threshold (0.0 - 1.0)",
    )
    wake_cooldown: float = Field(
        default=2.0,
        description="Cooldown seconds after wake trigger",
    )
    wake_stt_model: str = Field(default="base.en")
    wake_min_rms: float = Field(default=0.006, gt=0.0, le=0.05)
    wake_window_seconds: float = Field(default=3.0, ge=1.5, le=5.0)
    ambient_listening: bool = Field(default=False)
    ambient_followup_seconds: float = Field(default=20.0, ge=2, le=120)
    wake_models: list[str] = Field(
        default_factory=list,
        description="openWakeWord model names/paths (empty = Whisper-only keyword spotter)",
    )
    stt_model: str = Field(
        default="base.en",
        description="faster-whisper model size (e.g. tiny.en, base.en, small.en)",
    )
    stt_device: str = Field(
        default="auto",
        description="Inference device for whisper (auto, cuda, cpu)",
    )
    stt_compute_type: str = Field(
        default="default",
        description="Compute precision (default, int8, float16, float32)",
    )
    stt_language: str = Field(
        default="en",
        description="Primary language code for STT transcription",
    )
    stt_beam_size: int = Field(default=3, ge=1, le=10)
    stt_min_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    stt_retry_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    stt_retry_model: str | None = Field(default=None)
    utterance_silence_seconds: float = Field(default=1.0, ge=0.3, le=5.0)
    utterance_pause_grace_seconds: float = Field(default=0.8, ge=0, le=3)
    tts_engine: str = Field(
        default="auto",
        description=("TTS engine: auto (follow voice), piper, edge_tts, or fish_speech"),
    )
    tts_voice: str = Field(
        default="mommy",
        description="TTS voice name (custom_voice, en_GB-alan-medium, en-US-AvaNeural, mommy)",
    )
    tts_speed: float = Field(
        default=1.0,
        description="Speech synthesis speed multiplier (1.0 = normal)",
    )
    tts_enabled: bool = Field(
        default=True,
        description="Enable speech synthesis voice output",
    )
    fish_speech_url: str = Field(
        default="http://127.0.0.1:8080/v1/tts",
        description="Local Fish Speech API server endpoint",
    )
    fish_ref_audio: str = Field(
        default="data/voices/mommy/ref.wav",
        description="Path to reference audio for zero-shot voice cloning",
    )
    fish_ref_text: str = Field(
        default=(
            "Oh my god, did I like break your ribs or something? "
            "It's not my fault that you're fragile."
        ),
        description="Transcript of reference audio for zero-shot voice cloning",
    )
    fish_temperature: float = Field(default=0.7, description="Fish Speech sampling temperature")
    fish_top_p: float = Field(default=0.7, description="Fish Speech top_p sampling")
    fish_repetition_penalty: float = Field(
        default=1.2, description="Fish Speech repetition penalty"
    )
    fish_chunk_length: int = Field(
        default=200, description="Fish Speech chunk length for synthesis"
    )
    fish_max_new_tokens: int = Field(default=1024, description="Fish Speech max new tokens")
    sample_rate: int = Field(default=16000, description="Audio sample rate in Hz")
    channels: int = Field(default=1, description="Audio channel count (1 for mono)")
    input_device: int | str | None = Field(
        default=None,
        description="Microphone device index or substring name",
    )
    output_device: int | str | None = Field(
        default=None,
        description="Speaker device index or substring name",
    )

    @field_validator("input_device", "output_device", mode="before")
    @classmethod
    def parse_device(cls, value: int | str | None) -> int | str | None:
        """Accept indices and names for direct audio configuration."""
        return normalize_audio_device(value)


class Settings(BaseSettings):
    """Centralized settings for RAPHAEL loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # General app settings
    raphael_env: str = Field(default="development")
    raphael_log_level: str = Field(default="INFO")
    raphael_debug: bool = Field(default=False)
    raphael_preferred_name: str = Field(default="")
    raphael_persona_file: str = Field(
        default="persona.txt",
        description="Optional UTF-8 personality preferences file; empty disables it",
    )

    # Provider settings
    nim_api_key: SecretStr | None = Field(default=None)
    nim_model: str = Field(default="nvidia/nemotron-3-super-120b-a12b")
    nim_complex_model: str = Field(default="nvidia/nemotron-3-ultra-550b-a55b")
    nim_fallback_model: str = Field(default="meta/llama-3.2-90b-vision-instruct")
    openrouter_api_key: SecretStr | None = Field(default=None)
    groq_api_key: SecretStr | None = Field(default=None)
    ollama_host: str = Field(default="http://localhost:11434")

    # Memory & Persistence settings
    memory_db_path: str = Field(default="data/raphael.db")
    memory_max_short_term_turns: int = Field(default=10)

    # Audio & Voice settings
    wake_word: str = Field(default="hey raphael")
    wake_threshold: float = Field(default=0.5)
    wake_cooldown: float = Field(default=2.0)
    wake_stt_model: str = Field(default="base.en")
    wake_min_rms: float = Field(default=0.006, gt=0.0, le=0.05)
    wake_window_seconds: float = Field(default=3.0, ge=1.5, le=5.0)
    ambient_listening: bool = Field(default=False)
    ambient_followup_seconds: float = Field(default=20.0, ge=2, le=120)
    stt_model: str = Field(default="base.en")
    stt_device: str = Field(default="cpu")
    stt_compute_type: str = Field(default="int8")
    stt_language: str = Field(default="en")
    stt_beam_size: int = Field(default=3, ge=1, le=10)
    stt_min_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    stt_retry_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    stt_retry_model: str | None = Field(default=None)
    utterance_silence_seconds: float = Field(default=1.0, ge=0.3, le=5.0)
    utterance_pause_grace_seconds: float = Field(default=0.8, ge=0, le=3)
    tts_engine: str = Field(default="auto")
    tts_voice: str = Field(default="mommy")
    tts_speed: float = Field(default=1.0)
    tts_enabled: bool = Field(default=True)
    fish_speech_url: str = Field(default="http://127.0.0.1:8080/v1/tts")
    fish_ref_audio: str = Field(default="data/voices/mommy/ref.wav")
    fish_ref_text: str = Field(
        default=(
            "Oh my god, did I like break your ribs or something? "
            "It's not my fault that you're fragile."
        )
    )
    fish_temperature: float = Field(default=0.7)
    fish_top_p: float = Field(default=0.7)
    fish_repetition_penalty: float = Field(default=1.2)
    fish_chunk_length: int = Field(default=200)
    fish_max_new_tokens: int = Field(default=1024)
    audio_sample_rate: int = Field(default=16000)
    audio_channels: int = Field(default=1)
    audio_input_device: int | str | None = Field(default=None)
    audio_output_device: int | str | None = Field(default=None)

    @field_validator("audio_input_device", "audio_output_device", mode="before")
    @classmethod
    def parse_device(cls, value: int | str | None) -> int | str | None:
        """Parse device selections loaded from environment variables."""
        return normalize_audio_device(value)

    @property
    def app(self) -> AppConfig:
        """Structured application configuration."""
        return AppConfig(
            env=self.raphael_env,
            log_level=self.raphael_log_level.upper(),
            debug=self.raphael_debug or (self.raphael_log_level.upper() == "DEBUG"),
        )

    @property
    def memory(self) -> MemoryConfig:
        """Structured memory and persistence configuration."""
        return MemoryConfig(
            db_path=self.memory_db_path,
            max_short_term_turns=self.memory_max_short_term_turns,
        )

    @property
    def providers(self) -> ProviderConfig:
        """Structured provider configuration."""
        return ProviderConfig(
            nim_api_key=self.nim_api_key,
            nim_model=self.nim_model,
            nim_complex_model=self.nim_complex_model,
            nim_fallback_model=self.nim_fallback_model,
            openrouter_api_key=self.openrouter_api_key,
            groq_api_key=self.groq_api_key,
            ollama_host=self.ollama_host,
        )

    @property
    def audio(self) -> AudioConfig:
        """Structured audio configuration."""
        return AudioConfig(
            wake_word=self.wake_word,
            wake_threshold=self.wake_threshold,
            wake_cooldown=self.wake_cooldown,
            wake_stt_model=self.wake_stt_model,
            wake_min_rms=self.wake_min_rms,
            wake_window_seconds=self.wake_window_seconds,
            ambient_listening=self.ambient_listening,
            ambient_followup_seconds=self.ambient_followup_seconds,
            stt_model=self.stt_model,
            stt_device=self.stt_device,
            stt_compute_type=self.stt_compute_type,
            stt_language=self.stt_language,
            stt_beam_size=self.stt_beam_size,
            stt_min_confidence=self.stt_min_confidence,
            stt_retry_confidence=self.stt_retry_confidence,
            stt_retry_model=self.stt_retry_model,
            utterance_silence_seconds=self.utterance_silence_seconds,
            utterance_pause_grace_seconds=self.utterance_pause_grace_seconds,
            tts_engine=self.tts_engine,
            tts_voice=self.tts_voice,
            tts_speed=self.tts_speed,
            tts_enabled=self.tts_enabled,
            fish_speech_url=self.fish_speech_url,
            fish_ref_audio=self.fish_ref_audio,
            fish_ref_text=self.fish_ref_text,
            fish_temperature=self.fish_temperature,
            fish_top_p=self.fish_top_p,
            fish_repetition_penalty=self.fish_repetition_penalty,
            fish_chunk_length=self.fish_chunk_length,
            fish_max_new_tokens=self.fish_max_new_tokens,
            sample_rate=self.audio_sample_rate,
            channels=self.audio_channels,
            input_device=self.audio_input_device,
            output_device=self.audio_output_device,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()
