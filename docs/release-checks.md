# v0.2 Release Checks

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

## Recorded validation — 2026-10-02

- 128 unit tests passed; three integrations excluded by default.
- Ruff and whitespace checks passed across the repository changes.
- Local Piper synthesis integration passed.
- Release wheel built and dependencies installed into a fresh Python 3.14.7 venv.
- Isolated installed CLI and Piper synthesis checked without the ONNX training
  package; installed dependencies checked with `pip check`.
- Live microphone/speaker acceptance is pending because the execution environment
  has no `/dev/snd`. Do not mark hardware acceptance complete or tag the release
  based only on the automated results.
