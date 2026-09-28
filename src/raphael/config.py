"""Centralized configuration and secrets management for RAPHAEL."""

from functools import lru_cache

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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


class ProviderConfig(BaseModel):
    """AI Provider API keys and endpoints."""

    nim_api_key: SecretStr | None = Field(
        default=None,
        description="NVIDIA NIM API key",
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
    """Audio and voice loop settings."""

    wake_word: str = Field(default="raphael", description="Wake word trigger phrase")
    wake_threshold: float = Field(
        default=0.5,
        description="Wake word detection threshold (0.0 - 1.0)",
    )
    wake_cooldown: float = Field(
        default=2.0,
        description="Cooldown seconds after wake trigger",
    )
    wake_models: list[str] = Field(
        default_factory=lambda: ["hey_jarvis", "alexa"],
        description="List of openWakeWord model names or custom model paths",
    )
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

    # Provider settings
    nim_api_key: SecretStr | None = Field(default=None)
    openrouter_api_key: SecretStr | None = Field(default=None)
    groq_api_key: SecretStr | None = Field(default=None)
    ollama_host: str = Field(default="http://localhost:11434")

    # Audio settings
    wake_word: str = Field(default="raphael")
    wake_threshold: float = Field(default=0.5)
    wake_cooldown: float = Field(default=2.0)
    audio_sample_rate: int = Field(default=16000)
    audio_channels: int = Field(default=1)
    audio_input_device: int | str | None = Field(default=None)
    audio_output_device: int | str | None = Field(default=None)

    @property
    def app(self) -> AppConfig:
        """Structured application configuration."""
        return AppConfig(
            env=self.raphael_env,
            log_level=self.raphael_log_level.upper(),
            debug=self.raphael_debug or (self.raphael_log_level.upper() == "DEBUG"),
        )

    @property
    def providers(self) -> ProviderConfig:
        """Structured provider configuration."""
        return ProviderConfig(
            nim_api_key=self.nim_api_key,
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
            sample_rate=self.audio_sample_rate,
            channels=self.audio_channels,
            input_device=self.audio_input_device,
            output_device=self.audio_output_device,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()
