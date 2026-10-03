# Changelog

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
