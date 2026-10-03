# v0.3 Release Checks

Run from the repository root with the development environment active.

## Automated validation

```bash
pytest
ruff check src tests
pytest -m integration tests/test_tts.py -k piper_initialization_and_synthesis
pip wheel . --no-deps -w dist
```

Unit tests use mocked devices/providers. The Piper integration test requires its
local voice model; synthesis does not prove speaker playback works. Device and
wake-training integrations have their own hardware/model requirements and can be
run with `pytest -m integration` on an appropriately configured machine.

Install the wheel in a fresh virtual environment outside the checkout. From a
separate directory, run both `python -m raphael --help` and `raphael --help`, then
`pip check`. Verify normal startup does not require optional training packages.
Select an existing local Piper model to check synthesis without playback.

## Live microphone acceptance

Use the target desktop, microphone, and speakers with the normal `.env` and
installed speech models. Record device names/indices, STT model/device/compute
type, TTS voice/engine, silence duration, and beam size with the results. Keep keys
and private audio out of reports.

1. Run `python -m raphael setup`. Keep custom settings, select your devices and
   local voice, and confirm the voice test actually plays. Restart normally with
   `python -m raphael --listen` and verify the same devices and voice are used.
2. Say “Hey Raphael.” Expect an acknowledgement, then ask a short question.
   Repeat with “Hey Raphael, what time is it?” in one utterance. Verify the start
   of the command is retained and one reply plays.
   Wait for `Wake keyword spotter ready` before starting. Test ten wake attempts
   at normal volume, ten quieter attempts, and several slower greetings. Record
   successful attempts and wake latency, along with `WAKE_MIN_RMS` and
   `WAKE_WINDOW_SECONDS`. Leave the PC humming without speaking for one minute
   and record any false wakes; increase the minimum RMS if needed.
3. Ask a follow-up without the wake phrase. Check that the reply uses the earlier
   context. Pause briefly mid-sentence and verify the recorder does not cut you
   off; adjust `UTTERANCE_SILENCE_SECONDS` if needed.
4. Interrupt a longer spoken reply with a new command. Expect playback to stop,
   the new command to be recorded, and the new reply to complete.
5. Remain silent after wake/follow-up. Expect recording to time out and return
   to standby without invented questions, repeated replies, or a stuck listener.
6. In a cloud-provider session, temporarily disconnect the network. Ask a
   question and verify configured fallback or an error acknowledgement; restore
   connectivity and check the next question succeeds. Also run the mocked
   provider regression tests to cover null answers and failed fallback responses.
7. Ask “How do you say goodbye in Spanish?” and confirm it receives an answer.
   Say “Goodbye” and verify follow-up ends. Say the wake phrase again, then press
   Ctrl+C. Confirm the process exits and the microphone can be opened again on a
   fresh launch.
8. Launch with `./scripts/launch_raphael_gpu.sh --ambient`. Address RAPHAEL directly,
   then ask a related follow-up without her name. Talk to someone else and verify
   she stays quiet and no background turn or personal fact appears in SQLite.
   Check that `Raphael, stop listening` returns to wake-word mode and `Hey Raphael,
   listen continuously` enables ambient mode again. Interrupt playback by name.
   With headphones, also set `BARGE_IN_MODE=speech` and interrupt with ordinary
   speech. Confirm audio stops, the first word is retained, and steady PC hum does
   not interrupt. Pause a long reply, then say `Raphael, continue`; verify the AI
   receives interruption context and resumes instead of starting over. Estimated
   progress may repeat or skip a few words; it is not exact word alignment.
   Add words while STT or routing is pending and confirm one combined response.
   In particular, ask a question by name and add `And good` while routing is
   pending: expect `merged_continuation` and a single combined request. Repeat
   with successive additions and a correction; verify the original question and
   negation survive. Speech addressed to `Mom` must still be rejected.
   Check `AMBIENT_LISTENING` and `SHOW_TRANSCRIPTS` defaults with a no-argument GPU
   launch, then test `--no-ambient --no-show-transcripts` overrides.
9. Pause for 1.2 seconds mid-sentence, then finish the thought; expect one complete
   turn. Resume speaking during reply generation and verify the old reply is
   suppressed. Check small wording mistakes against recent context; uncertain
   names, dates, quantities, and negations should get a clarification, not a saved
   inferred correction. Repeat with the actual PC hum and record missed/false wakes.

10. With `TTS_STREAMING=true`, ask for a multi-sentence explanation. Confirm the
    first sentence plays before the entire reply arrives and history contains one
    assistant turn. Compare logged first-text and first-audio times with batch mode
    (`TTS_STREAMING=false`) using the same model, voice, and question. Interrupt
    during the first sentence, synthesis, and a gap between sentences: later queued
    audio must not play. Say `Raphael, continue` and check completed sentences are
    represented in interruption context. Ask for code and verify it remains in
    history while spoken output omits the code block. Disconnect after a sentence
    starts: expect a connection-cutoff notice, without a fallback answer being
    appended to the partially spoken reply.
11. While voice training is running, ask a short unclear question that triggers
    STT retry. Below `STT_RETRY_MIN_FREE_MB`, expect a primary-model retry without
    allocating `medium.en`. After training releases VRAM, a stronger retry can be
    used again and its temporary CUDA model must be released afterward. Record
    GPU memory and transcript quality; this does not prove hum rejection by itself.

## Recorded validation — 2026-10-04 (v0.3.4 patch)

- 488 unit tests passed; three integrations excluded. Ruff, whitespace checks,
  and the runtime version command passed.
- Before the change, a live NIM gate probe with a synthetic recent exchange
  rejected “What do you want to talk about?” with addressed=false/confidence=0.6.
  After the change it selected listener=assistant/confidence=0.95 and the
  application accepted it as `active_conversation`.
- A second live ordinary follow-up, “Can you suggest something fun?”, was also
  accepted. “Would you like some tea, Mom?” selected listener=other/confidence=0.95
  and remained silent. These are intent checks, not microphone acceptance.
- Configured the local conversation policy with the existing 40-second window.
  During recent dialogue valid uncertainty favors continuation; strict mode
  requires high certainty. Provider failures and malformed judgments remain
  silent in both policies. Inferred turns remain non-explicit and cannot save
  personal facts through the memory-command handler.
- Regressions cover unseen topic changes, other listeners, legacy gate JSON,
  strict mode, expired/reset/user-only contexts, onset-based time windows,
  malformed confidence, and structured NIM JSON containing cleanup markers.
- Reproduce using the ambient GPU launcher: ask a question by name, wait for
  playback, then ask “What do you want to talk about?” without her name. Expect
  `active_conversation` with logged intent confidence. Address Mom and expect
  silence; test again after the window expires and in strict mode.

## Recorded validation — 2026-10-04 (v0.3.4 patch)

- 489 unit tests passed; three integrations excluded. Ruff, whitespace checks,
  and the runtime version command passed.
- Before the change, a live NIM gate probe with a synthetic recent exchange
  rejected “What do you want to talk about?” with addressed=false/confidence=0.6.
  After the change it selected listener=assistant/confidence=0.95 and the
  application accepted it as `active_conversation`.
- A second live ordinary follow-up, “Can you suggest something fun?”, was also
  accepted. “Would you like some tea, Mom?” selected listener=other/confidence=0.95
  and remained silent. These are intent checks, not microphone acceptance.
- Configured the local conversation policy with the existing 40-second window.
  During recent dialogue valid uncertainty favors continuation; strict mode
  requires high certainty. Provider failures and malformed judgments remain
  silent in both policies. Inferred turns remain non-explicit and cannot save
  personal facts through the memory-command handler.
- Regressions cover unseen topic changes, other listeners, legacy gate JSON,
  strict mode, expired/reset/user-only contexts, onset-based time windows,
  stale assistant turns after idle, malformed confidence, and structured NIM
  JSON containing cleanup markers.
- Reproduce using the ambient GPU launcher: ask a question by name, wait for
  playback, then ask “What do you want to talk about?” without her name. Expect
  `active_conversation` with logged intent confidence. Address Mom and expect
  silence; test again after the window expires and in strict mode.

## Recorded validation — 2026-10-04 (v0.3.3 patch)

- 428 unit tests passed; three integrations excluded. Ruff, whitespace checks,
  and the runtime version command passed.
- Replayed the reported “What time is Raphael?” followed by “You sound like a
  robot, do you know?” through the real CLI callbacks with mocked audio and an
  isolated database. Both receive replies: local clock first, then one model
  answer for the feedback, without cloud speech-gate calls.
- Original transcripts are retained in history. Clock address inferred from a
  missing “it” is non-explicit and does not authorize memory writes. Regression
  coverage rejects quoted speech, other listeners, geographic/compound clock
  requests, custom-assistant mismatches, and appended instructions.
- Feedback descriptions and brief tags require recent assistant speech within
  the original follow-up window. Expired and reset contexts still reject them.
- The reported hardware run already shows `small.en` on CUDA without fallback;
  microphone acceptance of these new reply-gate rules remains pending. Reproduce
  using the GPU launcher in ambient mode, ask the clock question by name, and
  give the voice feedback after the local reply.

## Recorded validation — 2026-10-04 (v0.3.2 patch)

- 391 unit tests passed; three integrations excluded. Ruff, whitespace checks,
  and the runtime version command passed.
- Replayed the reported transcript “Why are you talking so fast right now?”
  after an assistant reply: accepted locally as `speech_feedback`, without
  another wake phrase or cloud intent request. Tests retain room-conversation
  rejection, deadline expiry, and context after uncertain fragments.
- Installed CUDA 12 cuBLAS and cuDNN 9 into the application `.venv` and tested
  the cached `small.en` model on the GTX 1660 SUPER using the launcher's library
  search paths. Actual inference on synthetic silence completed successfully
  with `cuda/int8_float16`; loading plus inference took 3.11 seconds.
- Startup regressions force a lazy CUDA-library failure and verify CPU fallback
  occurs before the ready event. The new `gpu` extra documents reproducible
  installation independent of the voice-training environment.
- Live follow-up/wake acceptance with the user's microphone remains pending.
  Restart the launcher, ask a question by name, then ask “Why are you talking
  so fast right now?” after the reply; expect `speech_feedback` and a response.
  This change does not automatically alter `TTS_SPEED`.

## Recorded validation — 2026-10-03 (v0.3.1 patch)

- 375 unit tests passed; three hardware/model integrations excluded. Ruff,
  `git diff --check`, and `python -m raphael --version` passed.
- Authenticated NIM requests confirmed the old default model returned HTTP 410
  with retirement time 2026-10-03 09:00 UTC (16:00 Asia/Hovd).
- Updated the local `NIM_MODEL` and package defaults to
  `nvidia/nemotron-3.5-lightning-30b-a3b`. A synthetic greeting through the real
  streaming provider produced its first text in 0.98s and completed in 1.02s.
  This measures provider text, not microphone-to-speaker latency.
- Regressions cover missing/retired models, transient retries, authentication
  failures, cancellation, single backup attempts, empty streams, terminal SSE
  events, and refusing fallback after partial output.
- Empty-interruption recovery is tested before STT delivery and during generation,
  without duplicate history, memory writes, expired-request replay, or restarting
  partial playback. Live microphone acceptance remains pending.
- Reproduce the reported audio case with headphones, `small.en` on CUDA with
  `int8_float16`, Piper, and ambient speech barge-in: ask a question by name, then
  let a brief non-speech sound trigger recording before the answer starts. When
  the new STT result is empty, expect `resumed_after_empty_audio` and one answer
  without repeating the question. Meaningful added speech must still merge.

## Recorded validation — 2026-10-03 (v0.3 development)

- 350 unit tests passed; three hardware/model integrations excluded.
- Regression checks cover speech before generation finishes, cancellation during
  network reads, synthesis and playback, merged additions, partial failures,
  cumulative interruption context, and STT retries under simulated VRAM pressure.
- Ruff and `git diff --check` passed.
- Built `raphael-0.3.0-py3-none-any.whl` and installed it into an isolated target
  directory using the existing environment's dependencies. Verified installed
  package imports, version metadata, module CLI and entry-point CLI with `--help`
  and `--version` from outside the checkout.
- Live microphone, cloud latency, GPU training-load and voice restart acceptance
  remain pending. No stable v0.3 release tag was created.

## Recorded validation — 2026-10-02 (v0.2)

- 128 unit tests passed; three integrations excluded by default.
- Ruff and whitespace checks passed across the repository changes.
- Local Piper synthesis integration passed.
- Release wheel built and dependencies installed into a fresh Python 3.14.7 venv.
- Isolated installed CLI and Piper synthesis checked without the ONNX training
  package; installed dependencies checked with `pip check`.
- Live microphone/speaker acceptance is pending because the execution environment
  has no `/dev/snd`. Do not mark hardware acceptance complete or tag the release
  based only on the automated results.
