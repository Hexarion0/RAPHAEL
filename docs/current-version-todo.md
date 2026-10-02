# Current Version Stabilization

Scope: polish the existing v0.2.0 assistant before adding features. Items below
come from the current source, focused tests, and local reproductions.

## Fix First

- [x] **Make installation and CLI startup self-contained.**
  `audio/tts.py` imports `edge_tts`, `msgpack`, and `requests`; `audio/trainer.py`
  imports `onnx`. These packages are not declared directly in `pyproject.toml`
  or `requirements.txt`. `onnx` is absent from the inspected environment, and
  `audio/__init__.py` imports the trainer even for ordinary runtime commands.
  Declare required dependencies and load training dependencies only for training
  commands. Verify a clean installation can run `python -m raphael --help` and
  use Piper without installing the training toolchain.

- [x] **Restore and clear summaries by session identity.**
  `memory/manager.py:get_summary()` searches summary *content* for the session
  ID, although the ID is saved in metadata. A newly created manager failed to
  restore a saved summary in a local reproduction. `clear_session()` deletes
  turns but leaves stored summaries. Retrieve summaries by exact session ID and
  remove that session's summaries when clearing it. Test restart restoration,
  isolation between sessions, and clearing followed by a restart.

- [x] **Preserve configuration when running setup.**
  `setup_wizard.py` rewrites `.env` with fixed defaults and drops custom STT,
  memory, and Fish Speech settings. The selected voice now saves its matching
  engine, and an omitted engine follows the voice on normal startup.
  Preserve existing values and report
  voice-test success only when `tts.speak()` succeeds. Verify setup followed by
  a normal launch uses the same voice and settings.

- [x] **Parse audio device indices consistently.**
  Setup writes numeric device choices to `.env`, but `int | str` settings retain
  values such as `"3"` as strings. `platform/linux.py:resolve_device()` treats
  strings as name searches; TTS passes the string directly to sounddevice.
  Normalize numeric strings to integers while preserving device-name queries.
  Test microphone and speaker selection by both index and name.

## Reliability and Response Quality

- [x] **Keep slow work outside the microphone callback.**
  `audio/listener.py:_audio_callback()` runs wake detection and invokes `on_wake`
  while holding the listener lock. Wake detection can run Whisper inference;
  `__main__.py:on_wake()` synthesizes an acknowledgement before starting playback.
  Queue this work outside the audio callback and measure callback latency against
  the 80 ms frame interval. Test wake acknowledgement, barge-in, failed stream
  startup, shutdown during processing, and restarting the listener.

- [x] **Bound summarization work and avoid repeated requests.**
  After the threshold is exceeded, `summarize_older_turns()` rereads and summarizes
  all older turns on each call, saving another summary. The call in `__main__.py`
  is synchronous despite its background-work comment. Track the last summarized
  turn and update the summary incrementally. Verify unchanged history causes no
  additional request and long conversations keep summary input bounded.

- [x] **Make failed voice preparation clean and retryable.**
  `audio/voice_dataset.py` removes `_scratch` only after extraction and
  transcription succeed. Exceptions leave temporary audio and partial clips;
  reruns start numbering at `0001` in the same output folder. Use guaranteed
  scratch cleanup and prepare into a staging directory before publishing the
  dataset. Test extraction/transcription failures and a retry against an existing
  dataset; existing valid metadata and clips must remain consistent.

- [x] **Recognize farewell intent without matching ordinary questions.**
  `__main__.py` searches for farewell words anywhere in user input and AI replies.
  The existing regex matches `How do you say goodbye in Spanish?`, causing a
  canned farewell instead of answering. Restrict session-ending detection to
  deliberate sign-offs. Test actual farewells, quoted words, translation requests,
  and explanatory replies mentioning goodbye.

## Verification Before Moving On

- [x] Separate deterministic unit tests from audio/model integration tests;
  ordinary tests should not require a microphone, model downloads, or services.
- [x] Resolve the original 62 Ruff findings: 54 long lines, six unused imports, one
  unnecessary f-string, and one import-order issue.
- [x] Update README installation instructions to include the editable package
  installation and describe the actual `src/raphael/` layout and supported engines.
- [ ] Run the unit suite, Ruff, a clean-install CLI check, and a manual microphone
  session covering wake, follow-up, interruption, silence, provider failure, and
  exit. Treat live audio checks separately from unit-test results.

## Original Audit Evidence

- Focused configuration, logging, provider, router, and memory suite: **24 passed**.
- Local checks reproduced missing summary restoration, retained summaries after
  clearing, numeric device strings, and farewell false positives.
- The audio-inclusive test run stalled; an isolated import trace stopped during
  sounddevice initialization. It was stopped, so audio behavior is unverified here.
- This checklist proposes fixes; application code was not changed during the audit.


## Selected Fixes Completed

The dependency/startup, failed-preparation cleanup, farewell detection, and
microphone callback items above are implemented. Voice preparation stages clips
and metadata before replacing the destination, restores the previous dataset on
publication failure, and retains a recoverable backup if restoration itself fails.

The listener now uses bounded audio ingestion, asynchronous wake inference,
separate notification and conversation workers, and recent audio for commands
spoken immediately after wake detection. Old responses cannot cancel new
barge-in recordings. Wake greetings no longer delay recording. Silence timing
uses audio samples; defaults are a one-second pause and STT beam size three,
configurable through `UTTERANCE_SILENCE_SECONDS` and `STT_BEAM_SIZE`.

Summary requests run separately from replies. The remaining memory and setup
items were completed in the stabilization pass below.

Validation: **118 unit tests passed**, with three integration tests excluded by
default. The local Piper synthesis integration test also passed. Ruff passes for
changed modules and tests. A wheel built successfully;
its isolated installation starts CLI help without importing native audio,
Whisper, or training dependencies. In a synthetic stress check with detection
blocked, callback latency was 0.0036 ms median and 0.0224 ms maximum, with at most
eight queued frames. This measures ingestion overhead, not live recognition or
end-to-end reply latency. Live microphone checks still require the target hardware.

The reported listen failures are also covered: punctuation after a bare wake
phrase no longer triggers a provider request; automatic TTS engine selection
uses the selected Piper voice; Fish Speech tries installed local fallback voices
before network speech. NIM handles null content, requests non-thinking Nemotron 3
answers, and checks the backup response rather than returning empty text.

Live service verification: a short NIM request to the configured Nemotron 3
model returned a visible answer in 0.95 seconds. Loading the configured
`en_US-amy-medium` voice and synthesizing a short sentence locally took 1.33
seconds. These checks did not record microphone audio or play speech. The
non-thinking request option follows
[NVIDIA's Nemotron 3 documentation](https://docs.nvidia.com/nim/large-language-models/2.0.4/turbo/get-started-nemotron-3-super-120b-a12b.html#reasoning).

## Stabilization Pass — 2026-10-02

- Summaries restore by exact session metadata, including summaries from older
  releases. Clearing deletes the session's summaries and turns atomically and
  invalidates in-flight summary results. Other sessions remain intact.
- Summarization updates one stored summary from at most 32 new older turns per
  request, with at most 1,000 characters per turn and 2,000 characters of prior
  context/output. The persisted turn ID resumes progress across restarts;
  unchanged history sends no request. This bounds input by truncating very long
  turns, so details beyond those limits may be omitted.
- Background summaries also work with an in-memory SQLite database.
- Setup preserves existing configuration, comments, and unchanged quoting,
  including STT, Fish Speech, and memory settings. Enter keeps device selections;
  `default` resets them. A changed voice saves its matching engine, and failed
  playback no longer reports success.
- Numeric audio device strings become integer indices; name queries remain names.
  Regression tests cover both settings and direct backend/TTS calls.
- The entire source and test tree passes Ruff. Unit engine-selection tests are
  independent of the local `.env` engine choice.

Validation: 128 unit tests passed; three integration tests were excluded by
selection. The local Piper synthesis integration test passed separately. Wheel
build and fresh dependency installation succeeded with Python 3.14.7. The isolated
wheel starts CLI help without importing PortAudio, Whisper, or training modules,
and synthesizes Piper speech with the existing local model and no `onnx` training
package. `pip check` verifies the installed dependency set.

Live microphone and speaker checks remain outstanding: this execution environment
has no `/dev/snd`. Follow [the manual release checks](release-checks.md) on the
assistant's target machine before tagging a release. Earlier recorded service and
callback timings above are historical audit evidence, not measurements from this
pass.
