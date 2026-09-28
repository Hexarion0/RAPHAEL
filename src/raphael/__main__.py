"""Entry point for running RAPHAEL via `python -m raphael`."""

import argparse
import re
import sys
import time
from datetime import datetime

from raphael.audio import (
    SpeechToText,
    TextToSpeech,
    WakeListenerLoop,
    WakeWordDetector,
    record_voice_samples,
    train_custom_wakeword,
)
from raphael.config import get_settings
from raphael.logging import setup_logging
from raphael.platform import get_audio_backend
from raphael.providers import ChatMessage, get_model_router


def main() -> int:
    """Initialize core settings, start logging, and launch RAPHAEL."""
    parser = argparse.ArgumentParser(description="RAPHAEL Desktop AI Assistant")
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run", "listen", "setup", "record-samples", "train-wake"],
        help="Command to run: 'run' (default), 'listen', 'setup', 'record-samples', 'train-wake'",
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

    # If user ran `python -m raphael setup`
    if args.command == "setup":
        from raphael.setup_wizard import run_setup_wizard

        run_setup_wizard()
        return 0

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
    logger.info("   RAPHAEL - Desktop AI Assistant v0.2.0  ")
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
        logger.info("Initializing wake-word, Whisper STT, and Piper TTS engines...")
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
        tts = TextToSpeech(
            voice_name=settings.audio.tts_voice,
            speed=settings.audio.tts_speed,
            output_device=settings.audio.output_device,
            enabled=settings.audio.tts_enabled,
        )
        router = get_model_router()

        def get_system_prompt() -> str:
            now_str = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")
            return (
                "You are RAPHAEL, a sophisticated, concise, and loyal AI desktop assistant. "
                f"Current local system time: {now_str}. "
                "Respond conversationally and concisely (1 to 3 sentences maximum unless the "
                "user explicitly asks for detailed explanations or code). "
                "Be helpful, witty, and direct. "
                "Provide only your direct spoken answer without any internal "
                "monologue, thinking process, or preambles. "
                "Do not use markdown headers, bullet lists, or bold symbols in spoken responses."
            )

        conversation_history: list[ChatMessage] = [
            ChatMessage(role="system", content=get_system_prompt())
        ]

        _farewell_re = re.compile(
            r"\b(bye|goodbye|good\s+night|goodnight|see\s+you|take\s+care|farewell|later|night)\b",
            re.IGNORECASE,
        )
        _wait_re = re.compile(
            r"^(wait|hold\s+on|hang\s+on|one\s+sec(ond)?|pause)[.?!]*$",
            re.IGNORECASE,
        )
        _stop_re = re.compile(
            r"^(stop|be\s+quiet|shut\s+up|never\s*mind|cancel)[.?!]*$",
            re.IGNORECASE,
        )
        _in_followup = [False]  # mutable flag shared across calls

        def on_wake(info: dict):
            logger.info("🎯 Wake detected! Details: %s", info)
            _in_followup[0] = False
            tts.speak("Yes?", block=False)

        def on_transcription(text: str, wake_info: dict, audio_data) -> bool:
            user_text = text.strip()
            if not user_text:
                logger.info("🗣️ (No speech detected after wake — returning to standby.)")
                _in_followup[0] = False
                return False

            logger.info('🗣️ You: "%s"', user_text)

            # Clean wake phrase from user query
            cleaned_query = re.sub(
                r"^(hey\s+)?(raphael|rafael|raphel|rafeal)[,\s]*",
                "",
                user_text,
                flags=re.IGNORECASE,
            ).strip()

            if not cleaned_query:
                # User just said the wake word with no follow-up
                reply = "Yes, sir? How can I assist you?"
                logger.info('🤖 RAPHAEL: "%s"', reply)
                tts.speak(reply, block=True)
                _in_followup[0] = True
                return True

            # Check if user requested pause / wait
            if _wait_re.search(cleaned_query):
                logger.info("⏸️ Pause requested ('%s') — saying 'Hmm?'", cleaned_query)
                tts.speak("Hmm?", block=True)
                _in_followup[0] = True
                return True

            # Check if user requested immediate stop / cancel
            if _stop_re.search(cleaned_query):
                logger.info("🛑 Stop requested ('%s') — returning to standby.", cleaned_query)
                tts.speak("Understood.", block=True)
                _in_followup[0] = False
                return False

            # Check for farewell in the user's query before calling the AI
            if _farewell_re.search(cleaned_query):
                logger.info("👋 Farewell detected in user query — ending session.")
                farewell_reply = "Goodnight, sir. I'll be here when you need me."
                logger.info('🤖 RAPHAEL: "%s"', farewell_reply)
                tts.speak(farewell_reply, block=True)
                _in_followup[0] = False
                return False

            # Update system prompt with fresh timestamp
            conversation_history[0] = ChatMessage(role="system", content=get_system_prompt())
            # Append user query and generate response
            conversation_history.append(ChatMessage(role="user", content=cleaned_query))
            # Keep history manageable (last 10 turns)
            if len(conversation_history) > 12:
                conversation_history[:] = [conversation_history[0]] + conversation_history[-10:]

            try:
                response = router.send(conversation_history, temperature=0.7, max_tokens=256)
                reply_text = response.content.strip()
                logger.info(
                    '🤖 RAPHAEL: "%s" [%s/%s]',
                    reply_text,
                    response.provider,
                    response.model,
                )
                conversation_history.append(ChatMessage(role="assistant", content=reply_text))

                # Speak response out loud
                tts.speak(reply_text, block=True)

                # Check if the AI's reply signals session end
                if _farewell_re.search(reply_text):
                    logger.info("👋 Farewell detected in AI reply — ending session.")
                    _in_followup[0] = False
                    return False

                # Stay in follow-up conversation mode
                _in_followup[0] = True
                return True

            except Exception as err:
                logger.error("Error generating or speaking AI response: %s", err)
                error_msg = "Apologies, sir. I encountered an error processing that request."
                tts.speak(error_msg, block=True)
                return True  # Stay in conversation despite transient error

        def on_barge_in():
            logger.info("🛑 Barge-in triggered: audio stopped, listening...")
            _in_followup[0] = True

        loop = WakeListenerLoop(
            audio_backend=audio_backend,
            detector=detector,
            stt=stt,
            tts=tts,
            on_wake=on_wake,
            on_transcription=on_transcription,
            on_barge_in=on_barge_in,
            sample_rate=settings.audio.sample_rate,
            device=settings.audio.input_device,
            barge_in=True,
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
