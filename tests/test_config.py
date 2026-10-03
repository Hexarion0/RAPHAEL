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


def test_stt_quality_thresholds_are_exposed_and_validated():
    import pytest
    from pydantic import ValidationError

    settings = Settings(_env_file=None, stt_min_confidence=0.3, stt_retry_confidence=0.6)
    assert settings.audio.stt_min_confidence == 0.3
    assert settings.audio.stt_retry_confidence == 0.6
    with pytest.raises(ValidationError):
        Settings(_env_file=None, stt_min_confidence=1.1)


def test_persona_file_can_be_configured_or_disabled():
    with patch.dict(os.environ, {}, clear=True):
        assert Settings(_env_file=None).raphael_persona_file == "persona.txt"
    for path in ("data/custom-persona.txt", ""):
        with patch.dict(os.environ, {"RAPHAEL_PERSONA_FILE": path}, clear=True):
            assert Settings(_env_file=None).raphael_persona_file == path


def test_wake_sensitivity_settings_reach_audio_config():
    with patch.dict(
        os.environ, {"WAKE_MIN_RMS": "0.009", "WAKE_WINDOW_SECONDS": "4"}, clear=True
    ):
        settings = Settings(_env_file=None)
    assert settings.audio.wake_min_rms == 0.009
    assert settings.audio.wake_window_seconds == 4.0


def test_ambient_and_pause_settings_reach_audio_config():
    with patch.dict(
        os.environ,
        {
            "AMBIENT_LISTENING": "true", "AMBIENT_FOLLOWUP_SECONDS": "30",
            "UTTERANCE_PAUSE_GRACE_SECONDS": "1.2",
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
    assert settings.audio.ambient_listening
    assert settings.audio.ambient_followup_seconds == 30
    assert settings.audio.utterance_pause_grace_seconds == 1.2


def test_listening_defaults_can_be_selected_in_env(tmp_path):
    env_file = tmp_path / '.env'
    env_file.write_text(
        'AMBIENT_LISTENING=true\nSHOW_TRANSCRIPTS=true\n'
        'BARGE_IN_MODE=speech\nBARGE_IN_SPEECH_SECONDS=0.32\n'
    )
    with patch.dict(os.environ, {}, clear=True):
        audio = Settings(_env_file=env_file).audio
    assert audio.ambient_listening and audio.show_transcripts
    assert audio.barge_in_mode == 'speech'
    assert audio.barge_in_speech_seconds == 0.32


def test_invalid_barge_in_mode_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        Settings(_env_file=None, barge_in_mode='anything')


def test_followup_policy_is_configurable_and_validated():
    import pytest
    from pydantic import ValidationError

    with patch.dict(os.environ, {}, clear=True):
        assert Settings(_env_file=None).audio.ambient_followup_policy == 'conversation'
    with patch.dict(os.environ, {'AMBIENT_FOLLOWUP_POLICY': 'strict'}, clear=True):
        assert Settings(_env_file=None).audio.ambient_followup_policy == 'strict'
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ambient_followup_policy='anything')
