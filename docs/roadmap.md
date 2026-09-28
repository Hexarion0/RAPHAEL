# RAPHAEL — Development Roadmap

> **Goal:** Build RAPHAEL from a simple local voice assistant into a persistent, multi-provider, cross-platform AI assistant with skills, messaging, tone awareness, and eventually presence detection.
>
> **Development rule:** Complete one feature completely before starting the next. Every sub-step should be implemented → tested → refined → committed.

---

## 0. Project Foundation

### 0.1 Repository

```text
RAPHAEL/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
└── src/
    └── raphael/
```

### 0.2 Python Environment

- [x] 0.2.1 Create Python virtual environment
- [x] 0.2.2 Configure package/dependency management
- [x] 0.2.3 Add formatting/linting
- [x] 0.2.4 Add basic test framework
- [x] 0.2.5 Create `python -m raphael` entry point

### 0.3 Configuration

- [x] 0.3.1 Create centralized configuration system
- [x] 0.3.2 Load secrets from environment variables
- [x] 0.3.3 Keep API keys and credentials out of Git

Example:

```text
.env
```

```text
NIM_API_KEY=
OPENROUTER_API_KEY=
GROQ_API_KEY=
```

### 0.4 Logging

- [x] 0.4.1 Create structured logging
- [x] 0.4.2 Add log levels
- [x] 0.4.3 Log provider/router/audio events
- [x] 0.4.4 Never log secrets or private data

### 0.5 Git Workflow

- [ ] 0.5.1 Commit each meaningful sub-step
- [ ] 0.5.2 Write useful commit messages
- [ ] 0.5.3 Tag stable milestones
- [ ] 0.5.4 Commit at every completed checkbox where practical — frequent small commits build a stronger GitHub contribution graph than rare large ones
- [ ] 0.5.5 Keep the repo public from day one so progress is visible

Example:

```bash
git commit -m "feat: add linux microphone capture"
git tag v0.1
git push origin v0.1
```

### 0.6 Budget & Provider Strategy

Starting broke, on the free tier — the whole provider layer (**1.4**) and router (**1.5**) are built around this from day one, not bolted on later.

| Provider | Cost | Role |
|---|---|---|
| **NVIDIA NIM** | Free tier | Default provider, current starting point |
| **OpenRouter** | Free-tier models + pay-as-you-go | Unified access to many models (OpenAI, Grok, etc.) without writing separate clients for each |
| **Groq** | Free tier | Extremely fast inference — the "fast" side of the router |
| **Ollama** (local) | Free, unlimited | Offline fallback, zero cost |
| OpenAI / xAI (Grok) direct | Paid | Add later once budget allows, or reach them via OpenRouter |

- [ ] 0.6.1 Default to free-tier providers (NIM, Groq, Ollama) while budget is tight
- [ ] 0.6.2 Track token/request usage per provider to catch anything approaching a paid limit
- [ ] 0.6.3 Treat Ollama as the zero-cost fallback of last resort, not the primary model

---

## 1. Core AI

The foundation of RAPHAEL.

Final pipeline:

```text
🎤 Microphone
      ↓
  Wake Word
      ↓
      STT
      ↓
    Router
      ↓
      LLM
      ↓
      TTS
      ↓
🔊 Speaker
```

---

### 1.1 Audio Backend

#### 1.1.1 Define Interface

Create:

```text
platform/base.py
```

Define an abstraction such as:

```text
AudioBackend
├── record()
├── start_stream()
├── stop_stream()
└── list_devices()
```

#### 1.1.2 Linux Implementation

Create:

```text
platform/linux.py
```

Use:

- PipeWire/PulseAudio
- `sounddevice`

#### 1.1.3 Device Discovery

- [x] List available microphones
- [x] Select default device
- [x] Allow manual device selection

#### 1.1.4 Recording

Implement:

```text
microphone → WAV
```

#### 1.1.5 Test

- [x] Microphone works
- [x] Correct sample rate
- [x] Correct channels
- [x] Recording is not clipped
- [x] Recording is not silent
- [x] Saved audio plays correctly

#### 1.1.6 Refine

- [x] Handle missing microphone
- [x] Handle microphone disconnect
- [x] Handle incorrect sample rate
- [x] Handle permission errors
- [x] Handle background noise
- [x] Handle device switching

**Checkpoint:** RAPHAEL can reliably record your voice.

---

### 1.2 Wake Word

RAPHAEL should remain idle until you say:

> **"Raphael"**

#### 1.2.1 Integration

- [x] Integrate openWakeWord, or
- [ ] Integrate Porcupine

#### 1.2.2 Detection

```text
Listening...

[Wake word detected]
```

- [x] Detect the wake word reliably

#### 1.2.3 Connect to Recording

```text
Wake word
    ↓
Start recording
```

- [x] Wire wake detection to start recording

#### 1.2.4 Cooldown Protection

- [x] Add cooldown protection after a trigger

#### 1.2.5 Refine

- [x] Test in a quiet room
- [x] Test with music playing
- [x] Test with game audio
- [x] Test with people talking
- [x] Test at different distances
- [x] Test at different speaking volumes
- [x] Tune false positives and false negatives

---

### 1.3 Speech-to-Text

Use **faster-whisper**.

#### 1.3.1 Setup

- [x] Install and load the model

#### 1.3.2 Conversion

- [x] Convert audio → text

#### 1.3.3 Pipeline Connection

```text
Wake word
    ↓
Record
    ↓
Whisper
    ↓
Text
```

- [x] Connect wake word → record → whisper → text

#### 1.3.4 Silence Detection

- [x] Add silence detection

#### 1.3.5 Refine

- [x] Test different accents
- [x] Test fast speech
- [x] Test quiet speech
- [x] Test background noise
- [x] Test gaming audio
- [x] Test short commands
- [x] Test long questions

#### 1.3.6 Transcription Mode

- [x] Start with batch transcription
- [x] Only add streaming if latency becomes a real problem

**Checkpoint:**

```text
You:
"Raphael, what's my GPU temperature?"

RAPHAEL:
"what's my GPU temperature?"
```

---

### 1.4 AI Provider Layer

Do not make `brain.py` directly depend on one provider. Cost profile for each is in **0.6**.

Create:

```text
providers/
├── base.py
├── nim.py
├── openrouter.py
├── groq.py
└── ollama.py
```

Common interface:

```text
LLMProvider
├── send()
├── stream()
└── health_check()
```

#### 1.4.1 NVIDIA NIM

- [x] Implement NIM client
- [x] Send prompt
- [x] Receive response
- [x] Handle authentication

#### 1.4.2 OpenRouter

- [x] Implement OpenRouter client
- [x] Support model selection
- [x] Handle API errors

#### 1.4.3 Groq

- [x] Implement Groq client
- [x] Support fast models
- [x] Handle rate limits

#### 1.4.4 Ollama

- [x] Implement Ollama client
- [x] Detect local availability
- [x] Support offline fallback

#### 1.4.5 Provider Manager

Track provider state:

```text
NIM        → available
Groq       → available
OpenRouter → unavailable
Ollama     → available
```

- [x] Track provider availability state

#### 1.4.6 Failure Handling

```text
NIM
 ↓ failed
Groq
 ↓ failed
Ollama
 ↓
response
```

- [x] Implement the NIM → Groq → Ollama fallback chain

#### 1.4.7 Refine

- [x] Timeouts
- [x] Retries
- [x] Exponential backoff
- [x] Rate-limit handling
- [x] Provider health state
- [x] Clean error messages

---

### 1.5 Model Router

RAPHAEL decides which model/provider should handle a request.

#### 1.5.1 Define Complexity

Example:

```text
Simple:
"What time is it?"

Medium:
"Explain how DNS works."

Complex:
"Analyze this Python architecture and redesign it."
```

- [ ] Define simple/medium/complex examples for calibration

#### 1.5.2 Create Router

```text
User request
     ↓
Complexity classifier
     ↓
┌───────────────┐
│ simple        │ → fast model
│ medium        │ → normal model
│ complex       │ → strong model
└───────────────┘
```

- [ ] Build the complexity classifier
- [ ] Route simple → fast model, medium → normal model, complex → strong model

#### 1.5.3 Manual Override

- [ ] Support internal overrides such as `/fast`, `/strong`, `/local`

#### 1.5.4 Routing Logs

Record:

```text
Request: "Explain this code"
Complexity: complex
Provider: NIM
Model: <model>
Reason: code analysis
```

- [ ] Log request, complexity, provider, model, and reasoning for each routed call

#### 1.5.5 Refine

- [ ] Measure latency
- [ ] Measure response quality
- [ ] Measure cost
- [ ] Measure failure rate
- [ ] Measure routing accuracy
- [ ] Optimize using real usage data

---

### 1.6 Text-to-Speech

Use **Piper** initially.

#### 1.6.1 Setup

- [ ] Install Piper

#### 1.6.2 Generation

```text
"text"
   ↓
 Piper
   ↓
 ".wav"
```

- [ ] Generate speech from text via Piper

#### 1.6.3 Playback

- [ ] Play responses automatically

#### 1.6.4 Voice Selection

- [ ] Select RAPHAEL's voice

#### 1.6.5 Tuning

- [ ] Tune speed
- [ ] Tune volume
- [ ] Tune pitch where supported
- [ ] Tune pauses

#### 1.6.6 Streaming Refine

```text
AI response
    ↓
stream chunks
    ↓
TTS
    ↓
speaker
```

- [ ] Stream response chunks to TTS instead of waiting for the full reply

#### 1.6.7 Barge-In

```text
RAPHAEL:
"The answer is—"

You:
"Raphael, stop."

RAPHAEL:
[Stops speaking]
```

- [ ] Allow the user to interrupt RAPHAEL mid-speech

---

### 1.7 Full Voice Loop

Connect everything:

```text
┌──────────────────┐
│   Microphone     │
└────────┬─────────┘
         ↓
    Wake Word
         ↓
      Record
         ↓
      Whisper
         ↓
    Model Router
         ↓
    AI Provider
         ↓
      Response
         ↓
       Piper
         ↓
      Speaker
```

#### 1.7.1 Run

- [ ] Run the complete loop end-to-end

#### 1.7.2 Context

- [ ] Add conversation context

#### 1.7.3 Interruption

- [ ] Add interruption handling

#### 1.7.4 Failure Recovery

- [ ] Add failure recovery

#### 1.7.5 Stress Test

- [ ] Stress test extended sessions

#### 1.7.6 Final Refinement

- [ ] Target low latency
- [ ] Reliable wake detection
- [ ] Clean transcription
- [ ] Stable AI responses
- [ ] Natural speech
- [ ] Graceful failures

---

## v0.2 — Core AI Complete

```bash
git tag v0.2
git push origin v0.2
```

RAPHAEL can now:

```text
Hear
 ↓
Understand
 ↓
Think
 ↓
Speak
```

---

## 2. Memory

Give RAPHAEL continuity across conversations and restarts.

### 2.1 Memory Architecture

Start with SQLite. Do not introduce a vector database until it is actually needed.

Possible schema:

```text
memories
──────────────
id
content
type
created_at
updated_at
source
confidence
```

Potential types:

```text
fact
preference
project
conversation
reminder
```

---

### 2.2 Short-Term Memory

#### 2.2.1
- [ ] Store the current conversation

#### 2.2.2
- [ ] Pass relevant conversation to the model

#### 2.2.3
- [ ] Limit context size

#### 2.2.4
- [ ] Summarize older conversation when necessary

---

### 2.3 Long-Term Memory

#### 2.3.1
- [ ] Store explicit facts

#### 2.3.2
- [ ] Allow RAPHAEL to propose memories rather than save them silently

Example:

> "My favorite game is CS2."

RAPHAEL proposes:

```text
favorite game = CS2
```

#### 2.3.3
- [ ] Retrieve relevant memories

#### 2.3.4
- [ ] Inject relevant memories into context

#### 2.3.5
- [ ] Handle conflicting memories

Example:

```text
Old:
favorite game = X

New:
favorite game = Y
```

#### 2.3.6
- [ ] Add memory deletion ("Forget that.")

---

### 2.4 Restart Test

- [ ] Teach RAPHAEL a fact
- [ ] Close RAPHAEL
- [ ] Restart RAPHAEL
- [ ] Ask about the fact
- [ ] Confirm it remembers

---

## v0.3 — Memory Complete

```bash
git tag v0.3
git push origin v0.3
```

---

## 3. Skills / Actions

Give RAPHAEL the ability to perform actions.

```text
actions/
├── open_app.py
├── system_info.py
├── web_search.py
├── reminder.py
└── ...
```

---

### 3.1 Skill Loader

- [ ] Automatically discover skills — adding `actions/weather.py` shouldn't require touching the core

---

### 3.2 Core Skills

#### 3.2.1 Open Applications
- [ ] Implement "open app" skill (e.g. "Open Discord")

#### 3.2.2 System Information
- [ ] Implement system info skill (e.g. "How much RAM am I using?")

#### 3.2.3 Web Search
- [ ] Implement web search skill (e.g. "Search for the latest NVIDIA driver")

---

### 3.3 Reminders

- [ ] Support relative reminders ("Remind me in 30 minutes")
- [ ] Support absolute reminders ("Remind me tomorrow at 8")
- [ ] Support recurring reminders ("Remind me every Monday")

---

### 3.4 Function Calling

AI chooses:

```text
skill
+
arguments
```

Example:

```json
{
  "skill": "open_app",
  "app": "vesktop"
}
```

- [ ] AI selects skill + arguments via function calling
- [ ] RAPHAEL validates and executes the skill

---

### 3.5 Safety

- [ ] Require confirmation for potentially destructive actions

Example:

```text
You:
"Delete this file."

RAPHAEL:
"This permanently deletes X.
Do you want me to continue?"
```

---

### 3.6 Refine

- [ ] Handle ambiguous requests
- [ ] Handle invalid arguments
- [ ] Handle unavailable applications
- [ ] Handle skill failures
- [ ] Handle timeouts
- [ ] Handle permissions
- [ ] Handle destructive operations

---

## v0.4 — Skills Complete

```bash
git tag v0.4
git push origin v0.4
```

---

## 4. Dashboard

Give RAPHAEL a visual interface.

Use:

- FastAPI
- WebSocket
- HTML/CSS/JavaScript initially
- TypeScript/React later if needed

Example:

```text
┌───────────────────────────────────┐
│             RAPHAEL               │
│                                   │
│             ● IDLE                │
│                                   │
│       ╭──────────────╮            │
│       │  WAVEFORM    │            │
│       ╰──────────────╯            │
│                                   │
│ User: What's using my GPU?        │
│                                   │
│ RAPHAEL: NVIDIA GPU is at 37%.   │
│                                   │
│ Provider: Groq                    │
│ Model: ...                        │
│ Latency: 0.8s                     │
└───────────────────────────────────┘
```

### 4.1
- [ ] Create the FastAPI server

### 4.2
- [ ] Create the WebSocket connection

### 4.3
- [ ] Display listening/idle state

### 4.4
- [ ] Display conversation history

### 4.5
- [ ] Add waveform visualization

### 4.6
- [ ] Display provider/model information

### 4.7
- [ ] Display router decisions

### 4.8
- [ ] Display errors and system status

### 4.9
- [ ] Perform a full UI/UX polish pass

---

## v0.5 — Dashboard Complete

```bash
git tag v0.5
git push origin v0.5
```

---

## 5. Messaging

Make RAPHAEL available outside the microphone.

```text
Telegram ──┐
Discord ───┼──→ RAPHAEL CORE
Voice ─────┤
Dashboard ─┘
```

All interfaces should use the same brain and memory.

---

### 5.1 Telegram

#### 5.1.1
- [ ] Text message → RAPHAEL → reply

#### 5.1.2
- [ ] Voice message → STT → RAPHAEL → reply

#### 5.1.3
- [ ] Optional TTS voice reply

---

### 5.2 Discord

- [ ] Text message → RAPHAEL → reply
- [ ] *(Later)* Discord voice
- [ ] *(Later)* Channel restrictions
- [ ] *(Later)* Permissions
- [ ] *(Later)* Server-specific configuration

---

### 5.3 Shared Memory

Introduce channel/user identity:

```text
User
 ├── Voice
 ├── Telegram
 └── Discord
```

- [ ] Introduce channel/user identity so memory doesn't mix unrelated users

---

### 5.4 Rate Limiting

- [ ] Protect against spam
- [ ] Protect against API exhaustion
- [ ] Protect against loops
- [ ] Protect against accidental runaway requests

---

### 5.5 WhatsApp

Keep this last — unofficial integrations can break.

- [ ] *(Stretch)* Prototype a basic WhatsApp text bridge, only if still worth maintaining

---

## v0.6 — Messaging Complete

```bash
git tag v0.6
git push origin v0.6
```

---

## 6. Mood / Tone

Teach RAPHAEL to pay attention to how something is said, not just what was said.

---

### 6.1 Audio Features

- [ ] Extract pitch
- [ ] Extract energy
- [ ] Extract speech rate
- [ ] Extract pauses
- [ ] Extract volume

---

### 6.2 Tone Classifier

Potential output:

```text
neutral
excited
frustrated
sad
calm
```

- [ ] Build/integrate a tone classifier
- [ ] Treat outputs as estimates, not facts

---

### 6.3 Context Integration

Instead of:

```text
User mood = angry
```

Prefer:

```text
Possible tone:
frustrated

Confidence:
0.72
```

- [ ] Pass tone as a confidence estimate, not a hard label
- [ ] Let the LLM decide whether the signal is useful

---

### 6.4 Refine

- [ ] Test different microphones
- [ ] Test background noise
- [ ] Test different speakers
- [ ] Test different speaking speeds
- [ ] Test sarcasm
- [ ] Test gaming environments
- [ ] Avoid overreacting to every small change in voice

---

## v0.7 — Tone Complete

```bash
git tag v0.7
git push origin v0.7
```

---

## 7. System Integration + Windows

Make RAPHAEL feel like a real desktop application.

---

### 7.1 Linux Service

- [ ] Create `raphael.service`
- [ ] Start RAPHAEL automatically with systemd

---

### 7.2 Notifications

- [ ] Support Linux notifications
- [ ] Support Windows notifications
- [ ] Use a platform abstraction rather than hardcoding Linux behavior into the core

---

### 7.3 Global Hotkey

Example: `SUPER + R`

- [ ] Add a manual trigger hotkey — useful when wake word fails, the room is noisy, or you don't want to rely on it

---

### 7.4 Windows Backend

Implement `platform/windows.py`:

- [ ] Handle microphone
- [ ] Handle audio playback
- [ ] Handle notifications
- [ ] Handle startup
- [ ] Handle application launching
- [ ] Handle hotkeys

---

### 7.5 Windows Parity

- [ ] Test wake word on Windows
- [ ] Test STT on Windows
- [ ] Test AI/router on Windows
- [ ] Test TTS on Windows
- [ ] Test memory on Windows
- [ ] Test skills on Windows
- [ ] Test dashboard on Windows
- [ ] Test messaging on Windows

---

## v0.8 — Cross-Platform Complete

```bash
git tag v0.8
git push origin v0.8
```

---

## 8. Proactive Intelligence + Presence

Make RAPHAEL capable of reacting to events instead of only waiting for commands.

---

### 8.1 Webcam

Start simple — do not immediately implement face recognition.

- [ ] Detect "nobody" vs. "presence"
- [ ] Detect motion

---

### 8.2 Event Engine

Build a general trigger system:

```text
Event
 ↓
Trigger
 ↓
Condition
 ↓
Action
```

Potential event sources:

```text
Time
Presence
Motion
Application state
Message
System event
```

- [ ] Build the Event → Trigger → Condition → Action engine

---

### 8.3 First Presence Feature

```text
Nobody detected

        ↓

Person enters room

        ↓

RAPHAEL

"Welcome back."
```

- [ ] Implement a simple "welcome back" greeting on room entry

---

### 8.4 Presence-Based Reminders

Example:

> "When I get back to my desk, remind me to finish RAPHAEL."

```text
Presence detected
       ↓
Check reminders
       ↓
Matching reminder
       ↓
RAPHAEL speaks
```

- [ ] Extend reminders to support presence-based triggers, not just time-based ones

---

### 8.5 Quiet Hours / DND

- [ ] Support Do Not Disturb
- [ ] Support quiet hours
- [ ] Support disabling camera triggers
- [ ] Support disabling voice notifications

---

### 8.6 False-Trigger Protection

Avoid triggering from:

- [ ] Shadows
- [ ] Pets
- [ ] Monitor changes
- [ ] Camera noise
- [ ] Random movement
- [ ] Use confidence thresholds and cooldowns

---

### 8.7 Optional WhatsApp Completion

- [ ] *(Stretch)* Finish WhatsApp integration, only if still desired

---

## v0.9 — Presence Complete

```bash
git tag v0.9
git push origin v0.9
```

---

## Final Integration & Polish

Before tagging v1.0, do one pass across the whole system rather than each feature in isolation:

- [ ] Run the full pipeline end-to-end with every feature active at once (voice, memory, skills, messaging, tone, presence)
- [ ] Re-test Windows parity now that every feature is in
- [ ] Review logs for silent failures across providers
- [ ] Clean up dead code / unused config left over from earlier iterations
- [ ] Update README with final architecture, feature list, and a demo clip

---

## v1.0 — RAPHAEL Complete

```bash
git tag v1.0
git push origin v1.0
```

At this point RAPHAEL is a personal desktop assistant platform rather than just a chatbot with a microphone.

---

## Final Architecture

```text
RAPHAEL/
│
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
│
├── src/
│   └── raphael/
│       │
│       ├── core/
│       │   ├── brain.py
│       │   ├── router.py
│       │   ├── memory.py
│       │   ├── context.py
│       │   └── events.py
│       │
│       ├── audio/
│       │   ├── wakeword.py
│       │   ├── stt.py
│       │   └── tts.py
│       │
│       ├── providers/
│       │   ├── base.py
│       │   ├── nim.py
│       │   ├── openrouter.py
│       │   ├── groq.py
│       │   └── ollama.py
│       │
│       ├── actions/
│       │   ├── open_app.py
│       │   ├── system_info.py
│       │   ├── web_search.py
│       │   └── reminder.py
│       │
│       ├── platform/
│       │   ├── base.py
│       │   ├── linux.py
│       │   └── windows.py
│       │
│       ├── integrations/
│       │   ├── telegram.py
│       │   ├── discord.py
│       │   └── whatsapp.py
│       │
│       ├── presence/
│       │   ├── camera.py
│       │   └── detection.py
│       │
│       └── server/
│           ├── api.py
│           └── websocket.py
│
├── web/
│   └── dashboard/
│
├── tests/
│
├── scripts/
│
└── docs/
    └── roadmap.md
```

---

## Language Strategy

Use **Python as the primary language**.

Python handles:

- AI orchestration
- Audio processing
- Whisper
- TTS
- Memory
- Provider APIs
- Skills
- Automation
- Webcam processing
- System integration

For the dashboard, add TypeScript/React later if a more advanced frontend is needed:

```text
Python
   ↓
RAPHAEL Core
   ↓
FastAPI
   ↓
WebSocket
   ↓
TypeScript/React Dashboard
```

Do not make RAPHAEL a Python/Node hybrid from day one.

---

## GitHub Growth Strategy

RAPHAEL is also meant to build up your GitHub profile, not just work as software:

- [ ] Commit at each completed sub-step where practical, not just at each version tag — frequent small commits build a stronger contribution graph than rare large ones
- [ ] Keep the repo public from day one
- [ ] Update the README at each version tag, with a short changelog and (once there's something to see) a demo gif or clip
- [ ] Use this roadmap file itself (`docs/roadmap.md`) as a live checklist — checked-off boxes double as a visible changelog for anyone browsing the repo
- [ ] Consider a short devlog (issues, discussions, or a `CHANGELOG.md`) noting what changed at each tag — extra visible activity beyond just commits

---

## Version Milestones

| Version | Capability |
| --- | --- |
| `v0.1` | 🧱 Project foundation |
| `v0.2` | 🎙️ Wake → STT → Router → AI → TTS |
| `v0.3` | 🧠 Persistent memory |
| `v0.4` | 🖐️ Skills and actions |
| `v0.5` | 🖥️ Web dashboard |
| `v0.6` | 💬 Telegram + Discord |
| `v0.7` | 🎭 Voice tone awareness |
| `v0.8` | 🪟 Windows + desktop integration |
| `v0.9` | 👁️ Presence detection |
| `v1.0` | ✅ Fully integrated, polished, proactive assistant |

---

## Development Philosophy

> **Don't build JARVIS all at once. Build one reliable component at a time.**

Start with:

```text
Microphone
   ↓
Record WAV
```

Then:

```text
Microphone
   ↓
Wake word
   ↓
Record
```

Then:

```text
Wake word
   ↓
Record
   ↓
Whisper
```

Then:

```text
Wake word
   ↓
Whisper
   ↓
AI
```

Then:

```text
Wake word
   ↓
Whisper
   ↓
AI
   ↓
Piper
```

Then:

```text
🎤 → RAPHAEL → 🔊
```

**That is the first real milestone.**

---

## RAPHAEL Long-Term Architecture

```text
                 ┌─────────────┐
                 │   Voice     │
                 └──────┬──────┘
                        │
┌─────────────┐         │        ┌─────────────┐
│  Telegram   │─────────┤        │   Discord   │
└─────────────┘         │        └─────────────┘
                        ↓
               ┌─────────────────┐
               │     RAPHAEL     │
               │                 │
               │ Brain           │
               │ Memory          │
               │ Router          │
               │ Skills          │
               │ Events          │
               └───────┬─────────┘
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
       NIM          Groq       OpenRouter
          │
          ↓
       Ollama
   (local fallback)

                       │
              ┌────────┴────────┐
              ↓                 ↓
          Dashboard          Speaker
                                │
                              🔊

                       Future
                          ↓
                     ┌─────────┐
                     │ Webcam  │
                     └────┬────┘
                          ↓
                      Presence
                          ↓
                       Events
                          ↓
                       Skills
```

**RAPHAEL starts as a voice pipeline and gradually becomes the assistant.**
