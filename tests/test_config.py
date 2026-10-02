"""Tests for centralized configuration and secret handling."""

import os
from unittest.mock import patch

from raphael.config import Settings


def test_default_settings():
    """Verify default settings values when no environment variables are set."""
    settings = Settings(
        _env_file=None,
        raphael_env="development",
        raphael_log_level="INFO",
        wake_word="raphael",
    )
    assert settings.app.env == "development"
    assert settings.app.log_level == "INFO"
    assert settings.audio.wake_word == "raphael"
    assert settings.audio.sample_rate == 16000
    assert settings.providers.nim_api_key is None


def test_environment_override():
    """Verify settings pick up environment variables."""
    env_vars = {
        "RAPHAEL_ENV": "production",
        "RAPHAEL_LOG_LEVEL": "DEBUG",
        "NIM_API_KEY": "nvapi-test-key-1234567890",
        "WAKE_WORD": "jarvis",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        settings = Settings(_env_file=None)
        assert settings.app.env == "production"
        assert settings.app.log_level == "DEBUG"
        assert settings.app.debug is True
        assert settings.audio.wake_word == "jarvis"
        assert settings.providers.nim_api_key is not None
        # Verify SecretStr masks the secret in str/repr
        assert "nvapi-test-key-1234567890" not in repr(settings.providers.nim_api_key)
        assert settings.providers.nim_api_key.get_secret_value() == "nvapi-test-key-1234567890"


def test_engine_defaults_to_voice_detection():
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings(_env_file=None, tts_voice="en_US-amy-medium")
    assert settings.audio.tts_engine == "auto"


def test_audio_device_indices_and_names_from_env(tmp_path):
    from raphael.config import AudioConfig

    env_path = tmp_path / ".env"
    env_path.write_text('AUDIO_INPUT_DEVICE=3\nAUDIO_OUTPUT_DEVICE="USB Speaker"\n')
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings(_env_file=env_path)
    assert settings.audio_input_device == 3
    assert settings.audio.input_device == 3
    assert settings.audio.output_device == "USB Speaker"
    direct = AudioConfig(input_device=" 4 ", output_device="")
    assert direct.input_device == 4
    assert direct.output_device is None
