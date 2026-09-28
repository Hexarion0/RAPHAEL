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
