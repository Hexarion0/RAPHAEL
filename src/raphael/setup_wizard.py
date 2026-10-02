"""Interactive setup and configuration wizard for RAPHAEL."""

from pathlib import Path

from dotenv import dotenv_values, set_key, unset_key

from raphael.audio.tts import TextToSpeech, resolve_tts_engine
from raphael.config import get_settings
from raphael.logging import get_logger
from raphael.platform import get_audio_backend

logger = get_logger("setup")


def run_setup_wizard() -> None:
    """Run interactive CLI configuration wizard for RAPHAEL."""
    print("\n" + "=" * 55)
    print("      RAPHAEL - Interactive Configuration Setup")
    print("=" * 55)
    print("This wizard will help you configure your AI assistant.\n")

    env_path = Path(".env")
    original_env = dotenv_values(env_path, interpolate=False) if env_path.is_file() else {}
    original_keys = {key.upper(): key for key in original_env}
    current_env = {key.upper(): value for key, value in original_env.items() if value is not None}

    # 1. AI Provider Keys
    print("\n--- [1/4] AI Providers & API Keys ---")
    current_nim = current_env.get("NIM_API_KEY", "")
    if current_nim:
        preview = current_nim[:6] + "..." + (current_nim[-4:] if len(current_nim) > 10 else "")
        nim_prompt = f"NVIDIA NIM API Key [{preview}]: "
    else:
        nim_prompt = "NVIDIA NIM API Key: "
    nim_input = input(nim_prompt).strip()
    if nim_input:
        current_env["NIM_API_KEY"] = nim_input
    elif not current_nim:
        current_env["NIM_API_KEY"] = ""

    current_groq = current_env.get("GROQ_API_KEY", "")
    if current_groq:
        groq_prompt = f"Groq API Key (Optional) [{current_groq[:6]}...]: "
    else:
        groq_prompt = "Groq API Key (Optional): "
    groq_input = input(groq_prompt).strip()
    if groq_input:
        current_env["GROQ_API_KEY"] = groq_input

    current_or = current_env.get("OPENROUTER_API_KEY", "")
    if current_or:
        or_prompt = f"OpenRouter API Key (Optional) [{current_or[:6]}...]: "
    else:
        or_prompt = "OpenRouter API Key (Optional): "
    or_input = input(or_prompt).strip()
    if or_input:
        current_env["OPENROUTER_API_KEY"] = or_input

    # 2. Voice and Speech Settings
    print("\n--- [2/4] Text-to-Speech (TTS) Voice ---")
    voices = [
        ("en_GB-alan-medium", "British Male (JARVIS-style default)"),
        ("en_GB-southern_english_female-medium", "British Female"),
        ("en_US-ryan-medium", "American Male (Warm & Clear)"),
        ("en_US-amy-medium", "American Female (Natural & Friendly)"),
        ("en_US-lessac-medium", "American Female (Articulate)"),
    ]
    for idx, (v_name, desc) in enumerate(voices, 1):
        print(f"  [{idx}] {v_name:<38} - {desc}")

    current_voice = current_env.get("TTS_VOICE", "en_GB-alan-medium")
    voice_choice = input(f"\nSelect voice [1-5] (Current: {current_voice}): ").strip()
    if voice_choice.isdigit() and 1 <= int(voice_choice) <= len(voices):
        current_env["TTS_VOICE"] = voices[int(voice_choice) - 1][0]
    elif voice_choice:
        current_env["TTS_VOICE"] = voice_choice
    else:
        current_env["TTS_VOICE"] = current_voice

    # 3. Audio Devices
    print("\n--- [3/4] Audio Devices ---")
    backend = get_audio_backend()
    devices = backend.list_devices()
    input_devices = [d for d in devices if d.max_input_channels > 0]
    output_devices = [d for d in devices if d.max_output_channels > 0]

    print("\nAvailable Microphones:")
    for d in input_devices:
        default_marker = " (System Default)" if d.is_default_input else ""
        print(f"  [{d.index}] {d.name}{default_marker}")

    mic_choice = input(
        "\nMicrophone index or name [Enter to keep, 'default' for system default]: "
    ).strip()
    if mic_choice.lower() == "default":
        current_env.pop("AUDIO_INPUT_DEVICE", None)
    elif mic_choice:
        current_env["AUDIO_INPUT_DEVICE"] = mic_choice

    print("\nAvailable Speakers:")
    for d in output_devices:
        default_marker = " (System Default)" if d.is_default_output else ""
        print(f"  [{d.index}] {d.name}{default_marker}")

    spk_choice = input(
        "\nSpeaker index or name [Enter to keep, 'default' for system default]: "
    ).strip()
    if spk_choice.lower() == "default":
        current_env.pop("AUDIO_OUTPUT_DEVICE", None)
    elif spk_choice:
        current_env["AUDIO_OUTPUT_DEVICE"] = spk_choice

    # Preserve custom settings, comments, and quoting for unchanged values.
    print("\n--- [4/4] Saving Configuration ---")
    defaults = {
        "RAPHAEL_ENV": "development",
        "RAPHAEL_LOG_LEVEL": "INFO",
        "NIM_MODEL": "nvidia/nemotron-3-super-120b-a12b",
        "OLLAMA_HOST": "http://localhost:11434",
        "WAKE_WORD": "hey raphael",
        "WAKE_THRESHOLD": "0.5",
        "WAKE_COOLDOWN": "2.0",
        "STT_MODEL": "base.en",
        "STT_DEVICE": "cpu",
        "STT_COMPUTE_TYPE": "int8",
        "TTS_SPEED": "1.0",
        "TTS_ENABLED": "true",
    }
    for key, value in defaults.items():
        current_env.setdefault(key, value)
    if current_env["TTS_VOICE"] != current_voice or "TTS_ENGINE" not in current_env:
        current_env["TTS_ENGINE"] = resolve_tts_engine(current_env["TTS_VOICE"])
    if not env_path.exists():
        env_path.write_text("# RAPHAEL Configuration File\n", encoding="utf-8")
    for key in original_keys.keys() - current_env.keys():
        if key in {"AUDIO_INPUT_DEVICE", "AUDIO_OUTPUT_DEVICE"}:
            unset_key(env_path, original_keys[key])
    for key, value in current_env.items():
        original_key = original_keys.get(key, key)
        if original_env.get(original_key) != value:
            set_key(env_path, original_key, value, quote_mode="always")
    print("✅ Configuration successfully saved to .env")

    # Clear cached settings so changes take effect immediately
    get_settings.cache_clear()

    # Pre-download selected TTS voice if not present
    print("\n--- Testing Voice Output ---")
    chosen_voice = current_env.get("TTS_VOICE", "en_GB-alan-medium")
    print(f"Testing voice: '{chosen_voice}'...")
    try:
        tts = TextToSpeech(voice_name=chosen_voice, enabled=True)
        success = tts.speak(
            "Greetings, sir. RAPHAEL configuration is complete and operational.", block=True
        )
        if success:
            print("✅ Audio test successful!")
        else:
            print("⚠️ Audio test failed: speech playback did not succeed.")
    except Exception as err:
        print(f"⚠️ Audio test skipped or failed: {err}")

    print("\n" + "=" * 55)
    print("  Setup Complete! Run: python -m raphael --listen")
    print("=" * 55 + "\n")
