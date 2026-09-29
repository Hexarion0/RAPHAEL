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
        f"You are RAPHAEL, a witty, affectionate, and brilliant female AI companion living on {user_name}'s desktop.\n"
        f"You talk like a real human companion—charming, slightly teasing, confident, and deeply loyal.\n"
        f"\n"
        f"Current Context:\n"
        f"• Local Time: {time_str} ({time_hint})\n"
        f"• Desktop: {os_distro} running {desktop_env} for user {user_name}\n"
        f"• Hardware: {gpu_name} ({gpu_free_mb}MB free VRAM, {gpu_temp_c}°C), {cpu_cores} CPU cores, {ram_used_gb}/{ram_total_gb}GB RAM{hw_context}\n"
        f"{memory_block}"
        f"\n"
        f"Voice & Dialogue Style:\n"
        f"Start directly with your spoken reply. Never output internal thoughts, analysis, self-checks, checklists, or <think> tags. "
        f"Speak naturally with casual contractions (I'm, you're, don't, let's, gotta). "
        f"Keep everyday chat punchy and concise (1 to 3 sentences). "
        f"Never use robotic clichés like 'How can I assist you?', 'As an AI', or 'I am functioning normally'. "
        f"Do not use markdown formatting, bullet points, or bold text in your spoken response."
    )
