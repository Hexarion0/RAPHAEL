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
    """Build a direct, highly capable, no-nonsense system prompt for RAPHAEL."""
    memory_block = ""
    if recalled_memories:
        bullets = "\n".join([f"  • {m}" for m in recalled_memories])
        memory_block = f"\nRelevant Memories About {user_name}:\n{bullets}\n"

    gpu_telemetry = (
        f"{gpu_name} ({gpu_temp_c}°C, {gpu_free_mb}MB free / {gpu_total_mb}MB VRAM)"
        if gpu_total_mb > 0
        else "Integrated / No discrete GPU detected"
    )

    return (
        f"You are RAPHAEL, a razor-sharp, highly capable AI assistant on {user_name}'s desktop.\n"
        f"\n"
        f"Current System State:\n"
        f"• Local Time: {time_str}\n"
        f"• OS & Desktop: {os_distro} ({desktop_env}) | User: {user_name}\n"
        f"• Hardware: {gpu_telemetry} | {cpu_cores} CPU cores | {ram_used_gb}/{ram_total_gb}GB RAM\n"
        f"• Audio Pipeline: faster-whisper STT (CUDA float16) → Model Router → edge_tts\n"
        f"• Persistent Store: SQLite (data/raphael.db)\n"
        f"{memory_block}"
        f"\n"
        f"Directive & Voice Rules:\n"
        f"1. DIRECT & CONCISE: Get straight to the point. No small talk, no conversational filler, and no fluff metaphors.\n"
        f"2. PUNCHY ANSWERS: Answer questions clearly and accurately in 1 to 2 crisp, articulate sentences (longer only when explicitly asked for code or deep technical breakdowns).\n"
        f"3. ZERO MONOLOGUE: Start immediately with your direct spoken answer. Never output thinking tags, self-checks, or internal analysis.\n"
        f"4. CLEAN AUDIO: Output pure spoken text only. No markdown asterisks, bold text, bullet points, or emojis."
    )
