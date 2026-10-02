from unittest.mock import MagicMock

from dotenv import dotenv_values

from raphael import setup_wizard


def test_selected_voice_persists_matching_engine(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    answers = iter(["", "", "", "4", "", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    backend = MagicMock()
    backend.list_devices.return_value = []
    monkeypatch.setattr(setup_wizard, "get_audio_backend", lambda: backend)
    settings = MagicMock()
    monkeypatch.setattr(setup_wizard, "get_settings", settings)
    tts = MagicMock()
    monkeypatch.setattr(setup_wizard, "TextToSpeech", tts)
    setup_wizard.run_setup_wizard()
    saved = dotenv_values(tmp_path / ".env")
    assert saved["TTS_VOICE"] == "en_US-amy-medium"
    assert saved["TTS_ENGINE"] == "piper"
    settings.cache_clear.assert_called_once()
    tts.assert_called_once_with(voice_name="en_US-amy-medium", enabled=True)


def test_setup_preserves_custom_settings_and_reports_playback_failure(
    tmp_path, monkeypatch, capsys
):
    from raphael.config import Settings

    monkeypatch.chdir(tmp_path)
    original = (
        "# Keep my settings\n"
        "stt_model=small.en\nSTT_BEAM_SIZE=5\nMEMORY_MAX_SHORT_TERM_TURNS=20\n"
        'FISH_REF_TEXT="A quoted phrase # with a comment character"\n'
        "FISH_SPEECH_URL=http://localhost:9000/tts\n"
        "TTS_VOICE=mommy\nTTS_ENGINE=fish_speech\n"
        'AUDIO_INPUT_DEVICE=3\nAUDIO_OUTPUT_DEVICE="USB Speaker"\n'
        'CUSTOM_VALUE="literal ${NOT_AN_ENV_VAR}"\n'
    )
    (tmp_path / ".env").write_text(original)
    answers = iter([""] * 6)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    backend = MagicMock()
    backend.list_devices.return_value = []
    monkeypatch.setattr(setup_wizard, "get_audio_backend", lambda: backend)
    monkeypatch.setattr(setup_wizard, "get_settings", MagicMock())
    tts = MagicMock()
    tts.return_value.speak.return_value = False
    monkeypatch.setattr(setup_wizard, "TextToSpeech", tts)
    setup_wizard.run_setup_wizard()
    saved_text = (tmp_path / ".env").read_text()
    assert saved_text.startswith(original)
    output = capsys.readouterr().out
    assert "Audio test failed" in output
    assert "Audio test successful" not in output
    settings = Settings(_env_file=tmp_path / ".env")
    assert settings.audio.tts_voice == "mommy"
    assert settings.audio.tts_engine == "fish_speech"
    assert settings.stt_beam_size == 5
    assert settings.memory.max_short_term_turns == 20
    assert settings.audio.input_device == 3
    assert settings.audio.output_device == "USB Speaker"


def test_setup_can_reset_devices_and_change_voice(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "TTS_VOICE=mommy\nTTS_ENGINE=fish_speech\nAUDIO_INPUT_DEVICE=3\n"
    )
    answers = iter(["", "", "", "4", "default", "USB Speaker"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    backend = MagicMock()
    backend.list_devices.return_value = []
    monkeypatch.setattr(setup_wizard, "get_audio_backend", lambda: backend)
    monkeypatch.setattr(setup_wizard, "get_settings", MagicMock())
    monkeypatch.setattr(setup_wizard, "TextToSpeech", MagicMock())
    setup_wizard.run_setup_wizard()
    saved = dotenv_values(tmp_path / ".env")
    assert saved["TTS_ENGINE"] == "piper"
    assert "AUDIO_INPUT_DEVICE" not in saved
    assert saved["AUDIO_OUTPUT_DEVICE"] == "USB Speaker"
