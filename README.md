# RAPHAEL

> A JARVIS-style, always-listening desktop AI assistant — its own voice, multi-provider AI routing, memory, and messaging integration, built one feature at a time.

**Status:** v0.3.0 development — persistent memory and streamed voice conversations.
Live microphone acceptance remains in [`docs/release-checks.md`](docs/release-checks.md).

---

## What is RAPHAEL?

RAPHAEL is a personal desk assistant that listens for a wake word, talks back in its own voice, remembers context across sessions, and reaches you over chat apps when you're away from your desk. Development currently targets Linux, with Windows parity and presence detection planned.

## Features

**Core**
- 🎙️ Wake word detection + speech-to-text (local, via Whisper)
- 🧠 Multi-provider AI backend (NVIDIA NIM, OpenRouter, Groq, Ollama) with a fast/complex-task router
- 🔊 Text-to-speech with its own voice
- Completed sentences play while the AI is still generating; interruptions cancel queued speech
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
Standalone time and date questions are answered from the local clock without an
AI request. Background summaries use the economical route (Groq when configured,
otherwise the default NIM model), with separate `summary` routing logs. Technical
words or long messages alone do not select the complex model; implementation,
debugging, architecture, and explicit `/strong` requests do. `/fast` and `/local`
remain available, and continuation requests retain the preceding task's routing.

The voice persona is a warm, mature, confident feminine companion: playful in
casual conversation, focused during tasks, and patient when you're frustrated.
She uses expressive, natural conversational phrasing, with concise answers that
stay warm and follow-ups when useful. Human-like delivery does not require invented
personal experiences or repeated AI disclaimers in ordinary small talk.
Set `RAPHAEL_PREFERRED_NAME` to the name you want her to use. English is the
default language. Edit `persona.txt` in the project folder to customize her tone,
humor, affection, language, and reply length. It is reloaded before every AI reply;
restart once after installing this feature, then edits take effect while running.
Copy `persona.example.txt` to `persona.txt` for a fresh template. Your personal
file is excluded from Git. `RAPHAEL_PERSONA_FILE` selects another path (relative
to the working directory), or an empty value disables custom preferences.
Missing, unreadable, invalid UTF-8, or oversized files fall back to the built-in
personality. Keep the file under 32 KiB; `#` comment lines are ignored. These
preferences shape AI replies; they do not change TTS voices, STT models, local
clock/memory acknowledgements, or saved facts.
The prompt uses configured audio settings and supplied
telemetry; unavailable GPU telemetry does not establish which hardware is present.
Historical replies and summaries are context, rather than verified facts or style
instructions. Generated replies cannot run commands or update saved facts:
the application handles common direct facts (preferred name, favorite items, and
project dates), their corrections, and explicit `remember ...` / `forget ...`
commands. Recognized facts update one keyed record with a bounded revision
history. Other facts can be saved with `remember ...`. Saved name and favorite
queries can be answered locally. Recall ranks related words and a small set of
aliases; it does not use embeddings or an additional model.
For example: `My favorite game is CS2`, then `Actually, my favorite game is
Valorant`, then `Forget my favorite game`. Forgetting removes the durable fact
and masks its known wording and previous values in history sent to the model,
including after a restart. Archived conversation text remains in SQLite;
paraphrases outside the known wording can still require explicit cleanup.
Conversation context is scoped to the persona version. The companion persona
starts a separate context while the legacy desktop chat stays archived in SQLite.
Saved facts and preferences remain shared; conversation summaries are excluded
from general memory recall and only the active session's summary is replayed.
Recalled facts include their recorded timestamp so relative statements such as
"this is day three" don't become claims about today. Saved project anchors are
included even for informal questions, with elapsed calendar days and inclusive
development-day counts calculated from a confirmed start date.

For CUDA speech recognition, set `STT_DEVICE=cuda` and
`STT_COMPUTE_TYPE=int8_float16`, then start with
`./scripts/launch_raphael_gpu.sh --listen`. The launcher adds existing NVIDIA
library directories from the application or Piper training environment before
starting Python. It does not install libraries or start training. CUDA 12 cuBLAS
and cuDNN 9 are required by
[faster-whisper](https://github.com/SYSTRAN/faster-whisper#gpu).
Keep the smaller command model when voice training shares limited GPU memory.
Both launch commands run the same application. The script selects `.venv/bin/python`,
switches to the project folder, and prepares `LD_LIBRARY_PATH`. Running
`python -m raphael --listen` uses the current Python and working directory without
that preparation; activate `.venv` first. The script does not force GPU settings:
`STT_DEVICE` and `STT_COMPUTE_TYPE` still control transcription.

Listening keeps microphone ingestion separate from wake inference, transcription,
and callbacks. It records immediately after wake detection without a spoken
wake greeting. Recent audio is retained to catch the beginning of your command.
A wake phrase alone, including `Hey Raphael.`, receives a local acknowledgement
without calling an AI provider.
Whisper wake detection retains three seconds of audio by default and submits a
complete snapshot after a short trailing pause, including for slower greetings.
`WAKE_MIN_RMS=0.006` controls the minimum audio level considered for a wake check;
raise it if noise causes excessive checks. `WAKE_WINDOW_SECONDS=3.0` controls
retained greeting audio. Whisper's VAD and transcript confidence checks still
filter wake candidates. `WAKE_THRESHOLD` applies to openWakeWord models, not the
Whisper keyword spotter. Wait for the `Wake keyword spotter ready` log on startup.
`STT ready` reports that command transcription has loaded. INFO logs show audio
being transcribed, STT word count/confidence, quality retries, superseded speech,
and each ambient reply/silence reason. DEBUG logs include raw candidates, including
background speech. For a single diagnostic run, add `--show-transcripts` to show
the recognized words at INFO level, including rejected ambient speech. Set
`SHOW_TRANSCRIPTS=true` in `.env` to make this the default, or override it for a run
with `--no-show-transcripts`. Recognizable but low-confidence
direct speech can prompt a repeat without executing or saving the uncertain text.
Bare wake phrases meeting the acceptance threshold skip the larger quality retry.
If a quality retry runs out of memory, its cached model is released and further
quality retries on that device are disabled for this process. Primary transcription
continues. A CPU fallback discards any cached GPU retry model.
`STT_BEAM_SIZE=3` reduces decoding work; raise it if recognition accuracy needs
more search. `UTTERANCE_SILENCE_SECONDS=1.0` plus
`UTTERANCE_PAUSE_GRACE_SECONDS=0.8` gives 1.8 seconds before an utterance finishes;
increase the grace if you pause longer mid-sentence. Recording uses local VAD
instead of relying only on volume, so quiet speech and louder steady noise can
be distinguished. Speech resumed during STT or
reply generation invalidates the old response, retains the new speech onset, and
suppresses stale playback/history. Addressed speech superseded during STT is kept
temporarily and included with the next utterance so additions do not lose the
original request. An in-flight provider request can still finish before the next
queued utterance is processed; its canceled reply is discarded. Linked additions
such as `and ...`, `also ...`, and `no, I meant ...` inherit reply permission from
the exact unfinished request they canceled. Pending fragments are sent as one user
message, while their original wording remains separate in chat history. Unrelated
recordings do not inherit this permission, and inherited permission does not
automatically authorize saving personal facts.

Summaries run separately
from voice replies, wait for at least six new older messages, process bounded
batches, and resume across restarts. Context keeps a bounded tail of messages
awaiting a summary so batching does not immediately lose the preceding exchange.
Clearing a conversation also removes its summaries. Sign-off commands such as `goodbye` end follow-up mode;
questions that merely contain farewell words do not.

Start optional ambient listening with `./scripts/launch_raphael_gpu.sh --ambient`,
or set `AMBIENT_LISTENING=true` in `.env`. With no arguments the GPU launcher starts
listening using `.env` settings, so `./scripts/launch_raphael_gpu.sh` is sufficient.
`--no-ambient` selects wake-word mode for one run. Local Silero VAD captures
speech without a wake phrase at 16 kHz. Direct addresses are handled locally;
possible follow-ups within `AMBIENT_FOLLOWUP_SECONDS` (40 in the example `.env`)
use an economical
`speech_gate` request before a reply. Unclear intent, malformed judgments, provider
failure, and ordinary background conversation default to silence. This estimates
the intended listener from text and context; it does not identify speakers.
Ambient onset keeps 800 ms of preceding audio to preserve the greeting while VAD
decides speech has begun. The independent keyword spotter also checks ambient audio;
its confirmed address permits a reply even if command STT misses the name. Greetings
such as `What's up Raphael?`, `So what's good Raphael?`, and `So Raphael, what's on
your mind?` are recognized locally, including known name spellings and short leading
fillers such as "so" or "well". A name
at the end of a question does not trim away the preceding question. Keyword evidence
belongs to one utterance and does not restart its recording or authorize the next
background conversation. If new speech supersedes an in-flight STT result, a clear
direct address from that result can still enter temporary context. Its canceled
reply is not executed or spoken; the original user fragment enters history only
when a subsequent addressed utterance is handled.

Say `Raphael, stop listening` to return to wake-word mode, or `Hey Raphael, listen
continuously` to turn ambient capture back on. This switches modes rather than
closing the microphone. With headphones, set `BARGE_IN_MODE=speech`: local VAD
stops playback after `BARGE_IN_SPEECH_SECONDS=0.24` of sustained speech and records
the interruption with onset pre-roll. Saying her name still works. With speakers,
select `BARGE_IN_MODE=wake` to avoid treating RAPHAEL's loudspeaker output as your
voice. Speech mode does not provide acoustic echo cancellation or identify speakers.

The follow-up window starts after playback finishes. Interruptions during a long
reply can still be judged as follow-ups even if that window would otherwise have
expired. A canceled synthesis cannot play later. The next AI reply receives the
previous reply's text and estimated playback progress; asking `continue` can resume
the unfinished explanation. The resume point is estimated from elapsed audio time,
not exact word timestamps.

Up to six short background excerpts are held in RAM for temporary context and
expire after 90 seconds. Raw background turns are not written to chat or saved
as personal facts. Relevant excerpts and candidate follow-ups may be sent to the
configured AI provider for interpretation. Ambient capture uses more STT work,
and judging follow-ups can add cloud latency. Inferred follow-ups do not modify
durable memories: address RAPHAEL directly for memory changes.

The AI uses recent dialogue to interpret small recognition mistakes while retaining
the original transcript. Names, dates, numbers, negation, and commands stay uncertain
when recognition is unclear; the assistant asks rather than inventing missing
details. Low-confidence transcripts do not automatically save personal facts.

Setup keeps existing custom settings and device choices. Press Enter to keep a
device, enter an index or name to change it, or enter `default` to reset it to the
system default. Numeric device indices in `.env` are parsed as integers.

See [release checks](docs/release-checks.md) for automated validation and the
live microphone checks required before tagging a release.

### Streaming replies and shared GPU use

`TTS_STREAMING=true` (the default) connects provider tokens to sentence-sized speech.
Generation continues during playback, so the first sentence can play before the
whole response finishes. Set `TTS_STREAMING=false` to restore batch replies.
Reasoning tags are filtered across token boundaries. Code remains in conversation
history and is omitted from spoken output. Provider fallback happens before text
arrives; a connection failure after a partial answer is reported rather than
splicing a second provider's answer onto it.

New speech cancels pending network reads, synthesis, and queued sentences. An old
synthesis job may finish internally, but its audio cannot play; canceled callers
return promptly. Voice inference is serialized, and worker queues are bounded.
Interruption context includes completed sentences and the estimated position in
the current sentence. An interrupted generated reply is archived only if playback
started, with an interruption note. Logs separate first text, first audio, provider
generation time, and total reply time; these are measurements, not latency guarantees.

While voice training shares the GPU, use `small.en` with `int8_float16` for command
STT (the example configuration). Before allocating a separate CUDA quality-retry
model, RAPHAEL checks free VRAM on its default GPU. `STT_RETRY_MIN_FREE_MB=2048`
sets the minimum; if memory is lower or cannot be checked, the already loaded
primary model receives a stronger decode instead. Temporary CUDA retry models
release their allocation after use. CPU retry models remain cached. Set the
threshold to `0` to disable the probe. This reduces avoidable allocations; another
process can still change free memory between the check and inference.

## Roadmap

RAPHAEL is being built one feature at a time, fully refined before moving to the next.

| Version | Milestone | Status |
|---|---|---|
| `v0.1` | Project foundation | ✅ |
| `v0.2` | Core AI (wake word → STT → router → LLM → TTS) | Tagged |
| `v0.3` | Persistent memory + streamed voice | Development; live acceptance pending |
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

### Speech recognition accuracy

If short phrases are misheard, try `STT_MODEL=small.en` and `STT_BEAM_SIZE=5`.
The larger model needs more CPU time; compare recognition on your microphone
before choosing the faster `base.en` setting. Use an explicit microphone index
or name if the system default is ambiguous.

The wake keyword spotter applies speech filtering and ignores low-confidence
segments. Utterance decoding retains natural repetitions such as "no, no" and
confident short replies such as "okay" or "bye". These safeguards reduce specific
failure modes; they do not identify the speaker or guarantee accuracy in fan
noise. A close microphone and sensible input gain still matter.

Wake recognition now supplies word timestamps to the listener. The detected
wake greeting is removed from the recorded audio before command transcription,
then supplied as known context for that first utterance. Bare wakes receive the
local acknowledgement; follow-up speech and genuine negative statements are
not rewritten. Three seconds of recent audio are retained to cover delayed wake
inference. These timestamps come from the recognizer and may be imperfect.

For ambiguous speech up to 12 seconds long, STT makes at most one quality retry
with a wider beam. `STT_RETRY_MODEL=medium.en` uses a separately loaded model for
that retry; an empty value retries the primary model. The retry model is loaded
only when needed and retained afterward, increasing memory use. A failed retry
keeps the primary result and quality check. Longer unclear recordings request a
repeat without a second inference pass. Scores below `STT_MIN_CONFIDENCE=0.4`
ask the speaker to repeat instead of sending a guessed command to the provider.
`STT_RETRY_CONFIDENCE=0.55` controls when to try the second pass. Scores combine
word and segment likelihoods; they are diagnostic scores rather than calibrated
probabilities that a sentence is correct.

`WAKE_STT_MODEL=base.en` controls the separate CPU/int8 keyword spotter. It now
uses base instead of tiny and requires a greeting at the beginning of its
transcript, rather than triggering on every mention of Raphael. Change this
setting independently of `STT_MODEL` if wake recognition needs tuning.

When a keyword snapshot contains only the greeting, the entire recognized
snapshot is consumed; its trailing syllables are not decoded as a new command.
Audio arriving afterward is still retained for the question. If the snapshot
contains both greeting and command, word timing separates them instead.
