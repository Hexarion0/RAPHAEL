# RAPHAEL

> A JARVIS-style, always-listening desktop AI assistant — its own voice, multi-provider AI routing, memory, and messaging integration, built one feature at a time.

**Status:** 🚧 Early development — see [`docs/roadmap.md`](docs/roadmap.md) for current progress.

---

## What is RAPHAEL?

RAPHAEL is a personal desk assistant that listens for a wake word, talks back in its own voice, remembers context across sessions, and reaches you over chat apps when you're away from your desk. It runs on both Linux and Windows, and is designed to eventually notice when you're actually in the room.

## Features

**Core**
- 🎙️ Wake word detection + speech-to-text (local, via Whisper)
- 🧠 Multi-provider AI backend (NVIDIA NIM, OpenRouter, Groq, Ollama) with a fast/complex-task router
- 🔊 Text-to-speech with its own voice
- 💾 Persistent memory (short-term + long-term)

**Extending it**
- 🖐️ Pluggable skills/actions (open apps, reminders, web search, system info)
- 🖥️ Web dashboard for live status, conversation history, and routing decisions
- 💬 Messaging integration — Telegram, Discord (WhatsApp planned as a stretch goal)
- 🎭 Mood/tone detection from voice, not just words

**Planned**
- 🪟 Full Windows parity
- 👁️ Webcam-based presence detection with proactive, presence-triggered reminders

See the [full roadmap](docs/roadmap.md) for the complete, step-by-step build plan.

## Tech Stack

| Layer | Tools |
|---|---|
| Core | Python |
| Server | FastAPI + WebSocket |
| Dashboard | HTML/JS (React/TS planned later) |
| Speech-to-text | faster-whisper |
| Text-to-speech | Piper |
| Wake word | openWakeWord / Porcupine |
| AI providers | NVIDIA NIM, OpenRouter, Groq, Ollama |
| Messaging | Telegram Bot API, Discord API |

## Project Structure

```text
raphael/
├── core/           → brain, router, memory, events
├── audio/          → wake word, STT, TTS
├── providers/      → AI provider clients
├── actions/        → skills (one file per skill)
├── platform/       → OS-specific backends (linux/windows)
├── integrations/   → Telegram, Discord, WhatsApp
├── presence/       → webcam/motion detection
└── server/         → FastAPI + WebSocket dashboard backend
```

## Getting Started

> RAPHAEL is early in development — setup will get simpler as the project matures.

```bash
git clone https://github.com/Hexarion0/RAPHAEL.git
cd raphael
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your API keys
python -m raphael
```

### Configuration

Set these in `.env`:

```text
NIM_API_KEY=
OPENROUTER_API_KEY=
GROQ_API_KEY=
```

Ollama runs locally and needs no key — it's the free offline fallback.

## Roadmap

RAPHAEL is being built one feature at a time, fully refined before moving to the next.

| Version | Milestone | Status |
|---|---|---|
| `v0.1` | Project foundation | ✅ |
| `v0.2` | Core AI (wake word → STT → router → LLM → TTS) | ⬜ |
| `v0.3` | Persistent memory | ⬜ |
| `v0.4` | Skills and actions | ⬜ |
| `v0.5` | Web dashboard | ⬜ |
| `v0.6` | Telegram + Discord | ⬜ |
| `v0.7` | Voice tone awareness | ⬜ |
| `v0.8` | Windows + desktop integration | ⬜ |
| `v0.9` | Presence detection | ⬜ |
| `v1.0` | Fully integrated, proactive assistant | ⬜ |

Full breakdown with sub-steps: [`docs/roadmap.md`](docs/roadmap.md)

## License

[MIT](LICENSE)

---

*Built as a personal project — one feature, fully refined, at a time.*
