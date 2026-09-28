"""Entry point for running RAPHAEL via `python -m raphael`."""

import argparse
import sys
import time

from raphael.audio import WakeListenerLoop, WakeWordDetector
from raphael.config import get_settings
from raphael.logging import setup_logging
from raphael.platform import get_audio_backend


def main() -> int:
    """Initialize core settings, start logging, and launch RAPHAEL."""
    parser = argparse.ArgumentParser(description="RAPHAEL Desktop AI Assistant")
    parser.add_argument(
        "--listen",
        action="store_true",
        help="Start the continuous wake-word listener loop",
    )
    args = parser.parse_args()

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
    logger.info("Wake Word configured: '%s'", settings.audio.wake_word)

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

    if args.listen:
        logger.info("Starting live wake-word listener... (Say 'Hey Jarvis' or 'Alexa')")
        detector = WakeWordDetector(
            models=settings.audio.wake_models,
            threshold=settings.audio.wake_threshold,
            cooldown_seconds=settings.audio.wake_cooldown,
        )

        def on_wake(info: dict):
            logger.info("🎯 Wake detected! Details: %s", info)

        def on_utterance(audio_data, wake_info):
            logger.info(
                "🎙️ Recorded utterance: %d samples (%.2fs). Ready for STT.",
                len(audio_data),
                len(audio_data) / settings.audio.sample_rate,
            )

        loop = WakeListenerLoop(
            audio_backend=audio_backend,
            detector=detector,
            on_wake=on_wake,
            on_utterance=on_utterance,
            sample_rate=settings.audio.sample_rate,
            device=settings.audio.input_device,
        )

        loop.start()
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            loop.stop()
            logger.info("Wake listener terminated cleanly.")
            return 0

    logger.info("Ready. Use '--listen' to start live wake listening. (Milestone 1.2: Wake Word)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[RAPHAEL] Shutting down cleanly...")
        sys.exit(0)
