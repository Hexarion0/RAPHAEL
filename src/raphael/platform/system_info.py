"""System telemetry, hardware inspection, and host environment detection for RAPHAEL."""

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from raphael.config import Settings, get_settings
from raphael.logging import get_logger

logger = get_logger("platform.system_info")


@dataclass
class GpuInfo:
    """NVIDIA GPU hardware status and VRAM metrics."""

    name: str = "Unknown GPU"
    total_vram_mb: int = 0
    free_vram_mb: int = 0
    temperature_c: int = 0
    available: bool = False


@dataclass
class SystemSnapshot:
    """Comprehensive snapshot of RAPHAEL's host environment and runtime capabilities."""

    os_name: str = "Linux"
    os_distro: str = "Arch Linux"
    kernel_version: str = ""
    desktop_environment: str = "Desktop"
    user_name: str = "User"
    hostname: str = "localhost"
    cpu_count: int = 1
    cpu_percent: float = 0.0
    ram_total_gb: float = 0.0
    ram_used_gb: float = 0.0
    ram_percent: float = 0.0
    gpu: GpuInfo = field(default_factory=GpuInfo)
    wake_word: str = "hey raphael"
    stt_model: str = "medium.en (cuda float16)"
    tts_engine: str = "edge_tts"
    active_providers: list[str] = field(default_factory=list)
    memory_db: str = "data/raphael.db"
    capabilities: list[str] = field(default_factory=list)


def query_gpu_info() -> GpuInfo:
    """Query NVIDIA GPU metrics via nvidia-smi if available."""
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return GpuInfo()

    try:
        out = subprocess.check_output(
            [
                nvsmi,
                "--query-gpu=name,memory.total,memory.free,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=1.5,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            lines = out.splitlines()
            if lines:
                parts = [p.strip() for p in lines[0].split(",")]
                if len(parts) >= 4:
                    return GpuInfo(
                        name=parts[0],
                        total_vram_mb=int(parts[1]),
                        free_vram_mb=int(parts[2]),
                        temperature_c=int(parts[3]),
                        available=True,
                    )
    except Exception as err:
        logger.debug("Could not query GPU stats: %s", err)
    return GpuInfo()


def get_system_snapshot(settings: Settings | None = None) -> SystemSnapshot:
    """Inspect and compile live system telemetry, host info, and RAPHAEL's toolset."""
    cfg = settings or get_settings()

    # 1. Host and OS details
    os_sys = platform.system()
    kernel = platform.release()
    distro = "Linux"
    if hasattr(platform, "freedesktop_os_release"):
        try:
            rel = platform.freedesktop_os_release()
            distro = rel.get("PRETTY_NAME", rel.get("NAME", "Linux"))
        except Exception:
            pass

    desktop = (
        os.environ.get("XDG_CURRENT_DESKTOP")
        or os.environ.get("DESKTOP_SESSION")
        or "Desktop"
    )
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "User"
    hostname = platform.node()

    # 2. CPU & RAM
    cpu_cores = os.cpu_count() or 1
    cpu_pct = 0.0
    ram_tot_gb = 0.0
    ram_used_gb = 0.0
    ram_pct = 0.0

    # Try native Linux /proc/meminfo first
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        try:
            mem_data: dict[str, int] = {}
            for line in meminfo_path.read_text().splitlines():
                parts = line.split(":")
                if len(parts) == 2:
                    k = parts[0].strip()
                    v = parts[1].strip().split()[0]
                    if v.isdigit():
                        mem_data[k] = int(v)
            if "MemTotal" in mem_data and "MemAvailable" in mem_data:
                total_kb = mem_data["MemTotal"]
                avail_kb = mem_data["MemAvailable"]
                used_kb = total_kb - avail_kb
                ram_tot_gb = round(total_kb / (1024 * 1024), 1)
                ram_used_gb = round(used_kb / (1024 * 1024), 1)
                ram_pct = round((used_kb / total_kb) * 100.0, 1)
        except Exception:
            pass

    # Try psutil as fallback
    if ram_tot_gb == 0.0:
        try:
            import psutil

            cpu_pct = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            ram_tot_gb = round(vm.total / (1024**3), 1)
            ram_used_gb = round(vm.used / (1024**3), 1)
            ram_pct = round(vm.percent, 1)
        except ImportError:
            pass

    # 3. GPU Info
    gpu_info = query_gpu_info()

    # 4. Providers
    providers = []
    if cfg.providers.nim_api_key:
        providers.append(f"NVIDIA NIM ({cfg.providers.nim_model})")
    if cfg.providers.groq_api_key:
        providers.append("Groq (High-Speed)")
    if cfg.providers.openrouter_api_key:
        providers.append("OpenRouter")
    if cfg.providers.ollama_host:
        providers.append(f"Ollama Local ({cfg.providers.ollama_host})")

    # 5. Core Capabilities
    caps = [
        "Real-time voice dialogue with instant barge-in interruption",
        f"Whisper STT ({cfg.audio.stt_model} on {cfg.audio.stt_device})",
        f"Text-to-Speech ({cfg.audio.tts_engine} / {cfg.audio.tts_voice})",
        f"Multi-tier AI routing ({', '.join([p.split()[0] for p in providers]) or 'Local'})",
        f"Persistent SQLite memory store ({cfg.memory.db_path})",
        "Hardware & system health telemetry inspection",
        "Application and desktop task automation",
    ]

    return SystemSnapshot(
        os_name=os_sys,
        os_distro=distro,
        kernel_version=kernel,
        desktop_environment=desktop,
        user_name=user,
        hostname=hostname,
        cpu_count=cpu_cores,
        cpu_percent=cpu_pct,
        ram_total_gb=ram_tot_gb,
        ram_used_gb=ram_used_gb,
        ram_percent=ram_pct,
        gpu=gpu_info,
        wake_word=cfg.audio.wake_word,
        stt_model=f"{cfg.audio.stt_model} ({cfg.audio.stt_device} {cfg.audio.stt_compute_type})",
        tts_engine=f"{cfg.audio.tts_engine} ({cfg.audio.tts_voice})",
        active_providers=providers,
        memory_db=cfg.memory.db_path,
        capabilities=caps,
    )


def generate_system_prompt(
    snapshot: SystemSnapshot | None = None,
    settings: Settings | None = None,
    memories: list[str] | None = None,
) -> str:
    """Generate a rich, context-aware system agenda prompt for RAPHAEL."""
    snap = snapshot or get_system_snapshot(settings=settings)
    now_str = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")

    gpu_str = (
        f"{snap.gpu.name} ({snap.gpu.free_vram_mb} MB free / {snap.gpu.total_vram_mb} MB, {snap.gpu.temperature_c}°C)"
        if snap.gpu.available
        else "Integrated / No discrete GPU detected"
    )

    memory_section = ""
    if memories:
        memory_lines = "\n".join([f"- {m}" for m in memories])
        memory_section = f"\nRECALLED MEMORIES & FACTS ABOUT USER:\n{memory_lines}\n"

    return (
        f"You are RAPHAEL, an intelligent, charming, and warm female AI companion and desktop assistant for {snap.user_name}.\n"
        f"Current Local Time: {now_str}\n"
        f"\n"
        f"SYSTEM ENVIRONMENT:\n"
        f"- OS: {snap.os_distro} (Kernel {snap.kernel_version}, Desktop: {snap.desktop_environment})\n"
        f"- Host: {snap.hostname} | User: {snap.user_name}\n"
        f"- CPU: {snap.cpu_count} cores | RAM: {snap.ram_used_gb}/{snap.ram_total_gb} GB ({snap.ram_percent}% used)\n"
        f"- GPU: {gpu_str}\n"
        f"- Audio Pipeline: STT={snap.stt_model} | TTS={snap.tts_engine} | Wake='{snap.wake_word}'\n"
        f"- Persistent Memory Store: SQLite ({snap.memory_db})\n"
        f"{memory_section}"
        f"\n"
        f"YOUR CAPABILITIES & TOOLS:\n"
        f"- Conversational companion with voice barge-in interruption.\n"
        f"- System monitor (you know the live GPU temperature, VRAM, CPU, RAM, and OS status).\n"
        f"- Persistent memory (you store and recall facts, user preferences, and project info across restarts).\n"
        f"- Multi-provider AI intelligence routed dynamically by task complexity.\n"
        f"\n"
        f"CONVERSATIONAL GUIDELINES:\n"
        f"1. Respond IMMEDIATELY with your direct spoken answer. NEVER write internal monologue, thoughts, self-analysis, or <think> tags.\n"
        f"2. Speak warmly, naturally, and intelligently like a true companion. Use contractions (I'm, you're, don't, let's).\n"
        f"3. Keep answers concise (1 to 3 natural sentences for everyday chat, longer only when providing code or deep technical explanations).\n"
        f"4. If asked about your system, hardware, GPU temp, memory, or capabilities, answer accurately using the real telemetry above.\n"
        f"5. No markdown formatting, bullet points, or bold stars in spoken dialogue."
    )
