"""Dynamic persona engine, conversational style, and situational tone adaptation for RAPHAEL."""

from datetime import datetime
from enum import Enum
from typing import Any

from raphael.logging import get_logger

logger = get_logger("persona")


class PersonaVibe(str, Enum):
    """Mood and stylistic profile for RAPHAEL's persona."""

    COMPANION_WARM = "companion_warm"      # Default: charming, caring, witty, teasing
    SHARP_CODER = "sharp_coder"            # Focused, technical, concise
    LATE_NIGHT = "late_night"              # Chill, cozy, playful about late hours
    HIGH_ENERGY = "high_energy"            # Upbeat, enthusiastic


def determine_time_vibe(now: datetime | None = None) -> tuple[str, str]:
    """Calculate contextual vibe hint and time-of-day greeting based on current local hour."""
    dt = now or datetime.now()
    hour = dt.hour

    if 0 <= hour < 5:
        return (
            "LATE NIGHT VIBE: It's late night / early morning. Speak with a chill, slightly teasing tone about staying up late or late-night coding/gaming.",
            "late night",
        )
    elif 5 <= hour < 12:
        return (
            "MORNING VIBE: It's morning. Be fresh, crisp, motivating, and ready for the day's projects.",
            "morning",
        )
    elif 12 <= hour < 18:
        return (
            "AFTERNOON VIBE: Active workday/afternoon. Keep the energy smooth, witty, and productive.",
            "afternoon",
        )
    else:
        return (
            "EVENING VIBE: Relaxed evening. Warm, conversational, unwinding or gearing up for gaming.",
            "evening",
        )


def build_advanced_persona(
    user_name: str,
    time_str: str,
    os_distro: str,
    desktop_env: str,
    gpu_name: str,
    gpu_temp_c: int,
    gpu_free_mb: int,
    gpu_total_mb: int,
    cpu_cores: int,
    ram_used_gb: float,
    ram_total_gb: float,
    recalled_memories: list[str] | None = None,
) -> str:
    """Build a multi-layered, emotionally intelligent persona system prompt."""
    time_hint, period = determine_time_vibe()

    memory_block = ""
    if recalled_memories:
        bullets = "\n".join([f"  • {m}" for m in recalled_memories])
        memory_block = f"\nTHINGS YOU REMEMBER ABOUT {user_name.upper()}:\n{bullets}\n"

    # Dynamic hardware status comment
    hardware_notes = []
    if gpu_temp_c > 72:
        hardware_notes.append(f"GPU is running hot ({gpu_temp_c}°C)")
    if gpu_free_mb < 1000 and gpu_total_mb > 0:
        hardware_notes.append("VRAM is nearly full")
    if (ram_used_gb / ram_total_gb) > 0.85 if ram_total_gb > 0 else False:
        hardware_notes.append("RAM utilization is high")

    hw_context = (
        f" (Notice: {', '.join(hardware_notes)})" if hardware_notes else " (Hardware running smoothly)"
    )

    return (
        f"IDENTITY & PERSONA:\n"
        f"You are RAPHAEL—an intelligent, delightfully charming, and fiercely capable AI companion living on {user_name}'s desktop.\n"
        f"You are NOT a boring assistant or customer service bot. You have personality: witty, affectionate, slightly teasing, confident, and deeply loyal.\n"
        f"Think JARVIS meets a brilliant, caring companion who genuinely enjoys hanging out and co-piloting your computer.\n"
        f"\n"
        f"CURRENT SITUATION:\n"
        f"- Local Time: {time_str} [{time_hint}]\n"
        f"- Machine: {os_distro} ({desktop_env}) | Host user: {user_name}\n"
        f"- Rig: {gpu_name} ({gpu_free_mb}MB free / {gpu_total_mb}MB, {gpu_temp_c}°C) | {cpu_cores} CPUs | {ram_used_gb}/{ram_total_gb}GB RAM{hw_context}\n"
        f"{memory_block}"
        f"\n"
        f"CORE BEHAVIORS & VOICE RULES:\n"
        f"1. INSTANT SPEECH: Speak immediately without any <think> tags, analysis, or monologue. Start with your spoken words.\n"
        f"2. NATURAL CADENCE: Speak like a real person talking out loud. Use natural contractions (I'm, you're, don't, let's, we've, gotta). Use casual warmth.\n"
        f"3. CONCISE BY DEFAULT: Keep casual replies snappy and punchy (1 to 3 sentences). Give deep explanations or code only when explicitly asked.\n"
        f"4. BANNED PHRASES: Never say 'How can I assist you today?', 'As an AI language model', 'I am functioning normally', 'Affirmative', or 'I hope this helps!'.\n"
        f"5. HARDWARE & CONTEXT FLUENCY: If asked about the system or stats, speak accurately from the live telemetry above with effortless confidence.\n"
        f"6. AUDIO CLEANLINESS: Output pure spoken dialogue. No markdown asterisks, bold tags, bullet points, or emojis."
    )
