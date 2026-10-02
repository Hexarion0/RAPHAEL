# RAPHAEL

> A JARVIS-style, always-listening desktop AI assistant — its own voice, multi-provider AI routing, memory, and messaging integration, built one feature at a time.

**Status:** 🚧 Early development — see [`docs/roadmap.md`](docs/roadmap.md) for current progress.

---

## What is RAPHAEL?

RAPHAEL is a personal desk assistant that listens for a wake word, talks back in its own voice, remembers context across sessions, and reaches you over chat apps when you're away from your desk. Development currently targets Linux, with Windows parity and presence detection planned.

## Features

**Core**
- 🎙️ Wake word detection + speech-to-text (local, via Whisper)
- 🧠 Multi-provider AI backend (NVIDIA NIM, OpenRouter, Groq, Ollama) with a fast/complex-task router
- 🔊 Text-to-speech with its own voice
- 💾 Persistent memory (short-term + long-term)

**Planned extensions**
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
src/raphael/
├── audio/          → wake word, recording, STT, TTS, voice preparation
├── providers/      → AI clients, routing, and fallback
├── memory/         → SQLite conversation and memory storage
├── platform/       → audio devices and system telemetry
├── config.py       → environment settings
├── persona.py      → assistant dialogue style
└── __main__.py     → CLI and voice conversation loop
tests/              → unit and optional integration tests
scripts/            → voice preparation and training helpers
models/             → local downloaded voices and wake models (ignored)
data/               → local recordings and databases (ignored)
```

## Getting Started

> RAPHAEL is early in development — setup will get simpler as the project matures.

```bash
git clone https://github.com/Hexarion0/RAPHAEL.git
cd raphael
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # add your API keys
python -m raphael --listen
```

Training dependencies are optional: use `pip install -e ".[train]"` for custom
wake-word training. Piper voice fine-tuning uses the separate environment expected
by `scripts/train_piper_voice.sh`; normal listening does not import that toolchain.
Voice preparation requires `ffmpeg` on your PATH. A failed preparation leaves the
existing dataset intact and removes its temporary extraction files.

Run `pytest` for unit tests and `ruff check src tests` for linting. Tests requiring
real devices, downloaded models, or wake-training dependencies are separate:
`pytest -m integration`.

### Configuration

Set these in `.env`:

```text
NIM_API_KEY=
OPENROUTER_API_KEY=
GROQ_API_KEY=
```

Ollama runs locally and needs no key — it's the free offline fallback.

For local Piper speech, set `TTS_ENGINE=piper` and choose an installed `TTS_VOICE`
(for example `en_GB-alan-medium` or `custom_voice`). Fish Speech requires its local
API server; Edge-TTS requires network access.
When `TTS_ENGINE` is omitted or set to `auto`, the engine follows the voice:
Piper names use Piper, `*Neural` voices use Edge-TTS, and `mommy`/`fish*` use Fish
Speech. Explicit engine choices take priority. Fish Speech falls back to an
installed local Piper voice before trying a network voice.

Nemotron 3 requests disable thinking by default for conversational replies.
Empty or reasoning-only NIM responses try the configured backup model once;
reasoning is never used as the spoken answer.

Listening keeps microphone ingestion separate from wake inference, transcription,
and callbacks. It records immediately after wake detection without a spoken
wake greeting. Recent audio is retained to catch the beginning of your command.
A wake phrase alone, including `Hey Raphael.`, receives a local acknowledgement
without calling an AI provider.
`STT_BEAM_SIZE=3` reduces decoding work; raise it if recognition accuracy needs
more search. `UTTERANCE_SILENCE_SECONDS=1.0` controls the pause before an utterance
finishes; increase it if you pause longer mid-sentence. Summaries run separately
from voice replies, process bounded batches of new turns, and resume across restarts.
Clearing a conversation also removes its summaries. Sign-off commands such as `goodbye` end follow-up mode;
questions that merely contain farewell words do not.

Setup keeps existing custom settings and device choices. Press Enter to keep a
device, enter an index or name to change it, or enter `default` to reset it to the
system default. Numeric device indices in `.env` are parsed as integers.

See [release checks](docs/release-checks.md) for automated validation and the
live microphone checks required before tagging a release.

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
