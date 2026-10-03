# Changelog

## 0.3.2 — 2026-10-04 (development)

- Accept recent feedback about spoken pace and delivery without a cloud speech
  gate or another wake phrase. Keep the original conversation window after an
  uncertain fragment; confirmed speech to someone else still ends it.
- Include speech feedback and answers to the assistant's last question in the
  follow-up classifier instructions.
- Add a Linux `gpu` installation extra for CUDA 12 cuBLAS and cuDNN 9, so command
  STT does not depend on a separate training environment's libraries.
- Run actual CUDA inference during background model loading before reporting
  STT ready; missing lazy-loaded libraries trigger the startup CPU fallback.

## 0.3.1 — 2026-10-03 (development)

- Replace the retired default NIM model with Nemotron 3.5 Lightning and keep
  thinking disabled for conversational replies.
- Try the configured NIM backup immediately on HTTP 404/410 in both batch and
  streaming modes. Preserve transient retries and cancellation without restarting
  a partially emitted answer. Handle usage-only and terminal SSE events.
- Resume an unanswered, directly linked question when a false interruption
  produces empty STT output. Keep original history once, reject expired or
  unrelated requests, and avoid replaying memory writes or partial playback.

## 0.3.0 — 2026-10-03 (development)

- Speak completed sentences while provider generation continues. Filter reasoning
  across token boundaries and retain code in history without reading it aloud.
- Cancel network reads, queued speech, and pending synthesis on interruption.
  Preserve completed sentences and estimated partial playback for continuation.
- Keep interrupted questions and additions together before the ambient reply
  decision. Record original user fragments separately and one assistant response.
- Budget additional CUDA STT retry models against available VRAM; reuse the
  primary model under training load and release temporary CUDA retry models.
- Retain SQLite facts, corrections, forgetting, and restart recall. Batch older
  conversation summaries in the background using economical routing.
- Select ambient listening, transcript diagnostics, and headphone speech barge-in
  through `.env`. The GPU launcher works without flags.
- Align package metadata, runtime version, and startup banner at 0.3.0.

Live microphone, training-load, and restart acceptance remain pending in
[`docs/release-checks.md`](docs/release-checks.md). A stable v0.3 tag has not been created.
