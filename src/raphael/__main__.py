"""Entry point for running RAPHAEL via `python -m raphael`."""

import argparse
import sys
import time

from raphael.audio import (
    SpeechToText,
    WakeListenerLoop,
    WakeWordDetector,
    record_voice_samples,
    train_custom_wakeword,
)
from raphael.config import get_settings
from raphael.logging import setup_logging
from raphael.platform import get_audio_backend


def main() -> int:
    """Initialize core settings, start logging, and launch RAPHAEL."""
    parser = argparse.ArgumentParser(description="RAPHAEL Desktop AI Assistant")
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run", "listen", "record-samples", "train-wake"],
        help="Command to run: 'run' (default), 'listen', 'record-samples', 'train-wake'",
    )
    parser.add_argument(
        "--listen",
        action="store_true",
        help="Start the continuous wake-word listener loop with STT transcription",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=8,
        help="Number of voice samples to record for training (default: 8)",
    )
    parser.add_argument(
        "--phrase",
        type=str,
        default="Hey Raphael",
        help="Target wake phrase to train (default: 'Hey Raphael')",
    )
    args = parser.parse_args()

    # If user ran `python -m raphael record-samples`
    if args.command == "record-samples":
        record_voice_samples(count=args.count, phrase=args.phrase)
        return 0

    # If user ran `python -m raphael train-wake`
    if args.command == "train-wake":
        try:
            train_custom_wakeword()
        except Exception as err:
            print(f"\n❌ Training failed: {err}")
            return 1
        return 0

    settings = get_settings()
    logger = setup_logging(settings.app.log_level)

    logger.info("==========================================")
    logger.info("   RAPHAEL - Desktop AI Assistant v0.1.0  ")
    logger.info("==========================================")
    logger.info("Environment: %s | Log Level: %s", settings.app.env, settings.app.log_level)

    # Initialize and report audio backend status
    audio_backend = get_audio_backend()
    default_input = audio_backend.get_default_input_device()
    default_output = audio_backend.get_default_output_device()
    input_name = default_input.name if default_input else "None found"
    output_name = default_output.name if default_output else "None found"

    logger.info(
        "Audio Backend: Input='%s' | Output='%s' | Rate=%d Hz",
        input_name,
        output_name,
        settings.audio.sample_rate,
    )
    logger.info(
        "Wake Word: '%s' | STT Model: '%s' (Device: %s)",
        settings.audio.wake_word,
        settings.audio.stt_model,
        settings.audio.stt_device,
    )

    # Report provider configuration status safely (boolean status only, no keys printed)
    nim_status = "configured" if settings.providers.nim_api_key else "not set"
    openrouter_status = "configured" if settings.providers.openrouter_api_key else "not set"
    groq_status = "configured" if settings.providers.groq_api_key else "not set"
    logger.info(
        "AI Providers: NIM (%s), OpenRouter (%s), Groq (%s), Ollama (%s)",
        nim_status,
        openrouter_status,
        groq_status,
        settings.providers.ollama_host,
    )

    if args.listen or args.command == "listen":
        logger.info("Initializing wake-word and Whisper STT engine...")
        detector = WakeWordDetector(
            wake_phrase=settings.audio.wake_word,
            models=settings.audio.wake_models,
            threshold=settings.audio.wake_threshold,
            cooldown_seconds=settings.audio.wake_cooldown,
        )
        stt = SpeechToText(
            model_size=settings.audio.stt_model,
            device=settings.audio.stt_device,
            compute_type=settings.audio.stt_compute_type,
            language=settings.audio.stt_language,
        )

        def on_wake(info: dict):
            logger.info("🎯 Wake detected! Details: %s", info)

        def on_transcription(text: str, wake_info: dict, audio_data):
            if text.strip():
                logger.info('🗣️ You: "%s"', text.strip())
            else:
                logger.info("🗣️ (No clear speech detected in recording)")

        loop = WakeListenerLoop(
            audio_backend=audio_backend,
            detector=detector,
            stt=stt,
            on_wake=on_wake,
            on_transcription=on_transcription,
            sample_rate=settings.audio.sample_rate,
            device=settings.audio.input_device,
        )

        loop.start()
        logger.info(
            "Awaiting wake word... Say '%s' followed by your question.",
            settings.audio.wake_word.title(),
        )
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            loop.stop()
            logger.info("Wake listener terminated cleanly.")
            return 0

    logger.info("Ready. Use 'python -m raphael --listen' for live voice listening.")
    logger.info("Use 'python -m raphael record-samples' to record voice samples.")
    logger.info("Use 'python -m raphael train-wake' to train a personalized wake model.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[RAPHAEL] Shutting down cleanly...")
        sys.exit(0)
