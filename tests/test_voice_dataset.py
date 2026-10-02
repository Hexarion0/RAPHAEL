"""Tests for Piper voice-dataset preparation from local video/audio."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from raphael.audio.tts import resolve_piper_voice_paths
from raphael.audio.voice_dataset import (
    MEDIA_EXTENSIONS,
    RAW_VOICE_DIR,
    _default_transcriber,
    collect_media_files,
    prepare_voice_dataset,
    segment_speech_regions,
    segment_speech_regions_vad,
    write_piper_metadata,
)


def test_raw_voice_dir_is_training_raw():
    assert RAW_VOICE_DIR.name == "raw"
    assert RAW_VOICE_DIR.parent.name == "training"


def test_collect_media_files_filters_supported_extensions(tmp_path: Path):
    (tmp_path / "talk.mp4").write_bytes(b"x")
    (tmp_path / "clip.wav").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    nested = tmp_path / "batch"
    nested.mkdir()
    (nested / "more.mkv").write_bytes(b"x")

    found = collect_media_files(tmp_path)
    names = {path.name for path in found}
    assert names == {"talk.mp4", "clip.wav", "more.mkv"}
    assert ".mp4" in MEDIA_EXTENSIONS


def test_segment_speech_regions_finds_isolated_utterances():
    sample_rate = 22050
    silence = np.zeros(sample_rate, dtype=np.float32)
    t = np.linspace(0, 1.2, int(sample_rate * 1.2), endpoint=False)
    speech_a = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    speech_b = (0.2 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    audio = np.concatenate([silence, speech_a, silence * 2, speech_b, silence])

    regions = segment_speech_regions(audio, sample_rate)
    assert len(regions) == 2
    durations = [(end - start) / sample_rate for start, end in regions]
    assert all(0.8 < duration < 2.0 for duration in durations)


def test_write_piper_metadata(tmp_path: Path):
    clips = tmp_path / "wavs"
    clips.mkdir()
    wav_path = clips / "0001.wav"
    sf.write(wav_path, np.zeros(22050, dtype=np.float32), 22050)
    metadata_path = tmp_path / "metadata.csv"

    write_piper_metadata(
        metadata_path,
        [(wav_path, "Hello from Raphael.")],
    )

    line = metadata_path.read_text(encoding="utf-8").strip()
    assert line == "wavs/0001.wav|Hello from Raphael."


def test_segment_speech_regions_keeps_continuous_speech():
    sample_rate = 22050
    t = np.linspace(0, 2.0, sample_rate * 2, endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    regions = segment_speech_regions(audio, sample_rate)
    assert len(regions) == 1
    start, end = regions[0]
    assert (end - start) / sample_rate >= 1.0


def test_segment_speech_regions_keeps_quiet_continuous_speech():
    sample_rate = 22050
    t = np.linspace(0, 2.0, sample_rate * 2, endpoint=False)
    audio = (0.01 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)

    regions = segment_speech_regions(audio, sample_rate)

    assert len(regions) == 1
    start, end = regions[0]
    assert (end - start) / sample_rate >= 1.0


@pytest.mark.parametrize("segmenter", [segment_speech_regions, segment_speech_regions_vad])
def test_segment_speech_regions_rejects_nonpositive_sample_rate(segmenter):
    with pytest.raises(ValueError, match="sample_rate must be positive"):
        segmenter(np.ones(100, dtype=np.float32), 0)


@pytest.mark.parametrize("segmenter", [segment_speech_regions, segment_speech_regions_vad])
def test_segment_speech_regions_rejects_nonpositive_clip_limit(segmenter):
    with pytest.raises(ValueError, match="clip limits"):
        segmenter(np.ones(100, dtype=np.float32), 22050, max_clip_seconds=0)


def test_prepare_voice_dataset_builds_piper_layout(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    video = source / "episode.mp4"
    video.write_bytes(b"fake")
    output = tmp_path / "dataset"

    sample_rate = 22050
    t = np.linspace(0, 2.0, sample_rate * 2, endpoint=False)
    extracted = (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)

    def fake_extract(_media: Path, dest: Path, _sample_rate: int) -> Path:
        sf.write(dest, extracted, sample_rate)
        return dest

    report = prepare_voice_dataset(
        input_path=source,
        output_dir=output,
        transcribe_clip=lambda _audio, _sr: "licensed speech clip",
        extract_audio_fn=fake_extract,
    )

    assert report.clip_count == 1
    assert report.skipped_count == 0
    assert (output / "metadata.csv").is_file()
    assert (output / "wavs" / "0001.wav").is_file()
    assert "licensed speech clip" in (output / "metadata.csv").read_text(encoding="utf-8")


def test_prepare_voice_dataset_rejects_unexpected_sample_rate(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "episode.mp4").write_bytes(b"fake")

    def wrong_rate_extract(_media: Path, dest: Path, _sample_rate: int) -> Path:
        sf.write(dest, np.zeros(22050, dtype=np.float32), 16000)
        return dest

    with pytest.raises(ValueError, match="Unexpected sample rate 16000 Hz"):
        prepare_voice_dataset(
            input_path=source,
            output_dir=tmp_path / "dataset",
            transcribe_clip=lambda _audio, _sr: "unused",
            extract_audio_fn=wrong_rate_extract,
        )


def test_prepare_voice_dataset_rejects_empty_transcriptions(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "episode.mp4").write_bytes(b"fake")

    def fake_extract(_media: Path, dest: Path, sample_rate: int) -> Path:
        sf.write(dest, np.ones(sample_rate, dtype=np.float32) * 0.1, sample_rate)
        return dest

    monkeypatch.setattr(
        "raphael.audio.voice_dataset.segment_speech_regions_vad",
        lambda _audio, sample_rate: [(0, sample_rate)],
    )
    output = tmp_path / "dataset"

    with pytest.raises(ValueError, match="No usable speech clips"):
        prepare_voice_dataset(
            input_path=source,
            output_dir=output,
            transcribe_clip=lambda _audio, _sr: "",
            extract_audio_fn=fake_extract,
        )

    assert not (output / "_scratch").exists()


def test_default_transcriber_resamples_to_whisper_rate(monkeypatch):
    captured = {}

    class StubTranscriber:
        def wait_ready(self):
            return None

        def transcribe(self, audio, language):
            captured["audio"] = audio
            captured["language"] = language
            return "spoken words"

    audio_settings = SimpleNamespace(
        stt_model="base.en",
        stt_device="cpu",
        stt_compute_type="int8",
        stt_language="en",
    )
    monkeypatch.setattr("raphael.audio.stt.SpeechToText", lambda **_kwargs: StubTranscriber())
    monkeypatch.setattr(
        "raphael.config.get_settings",
        lambda: SimpleNamespace(audio=audio_settings),
    )

    transcribe = _default_transcriber("en")
    result = transcribe(np.zeros(22050, dtype=np.float32), 22050)

    assert result == "spoken words"
    assert captured["audio"].shape == (16000,)
    assert captured["language"] == "en"


def test_resolve_custom_piper_voice_uses_local_onnx(tmp_path: Path):
    onnx_path = tmp_path / "custom_voice.onnx"
    json_path = tmp_path / "custom_voice.onnx.json"
    onnx_path.write_bytes(b"onnx")
    json_path.write_text("{}", encoding="utf-8")

    with patch("raphael.audio.tts.download_voice") as mock_download:
        resolved_onnx, resolved_json = resolve_piper_voice_paths("custom_voice", tmp_path)

    assert resolved_onnx == onnx_path
    assert resolved_json == json_path
    mock_download.assert_not_called()


@pytest.mark.parametrize("failure", ["extract", "transcribe", "publish"])
def test_failed_preparation_preserves_existing_dataset_and_cleans_staging(
    tmp_path: Path,
    monkeypatch,
    failure,
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "episode.wav").write_bytes(b"input")
    output = tmp_path / "dataset"
    (output / "wavs").mkdir(parents=True)
    (output / "wavs" / "0001.wav").write_bytes(b"previous audio")
    (output / "metadata.csv").write_text("wavs/0001.wav|previous text\n")
    before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}

    def extract(_source, destination, sample_rate):
        sf.write(destination, np.ones(sample_rate * 2, dtype=np.float32) * 0.1, sample_rate)
        if failure == "extract":
            raise RuntimeError("extraction failed")

    transcribed = 0

    def transcribe(_audio, _rate):
        nonlocal transcribed
        transcribed += 1
        if failure == "transcribe" and transcribed == 2:
            raise RuntimeError("transcription failed")
        return "prepared words"

    monkeypatch.setattr(
        "raphael.audio.voice_dataset.segment_speech_regions_vad",
        lambda _audio, rate: [(0, rate), (rate, rate * 2)],
    )
    rename = Path.rename

    def publish(path, target):
        if failure == "publish" and path.parent.name.startswith(".dataset-prepare-"):
            raise OSError("publication failed")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", publish)
    with pytest.raises((RuntimeError, OSError), match="failed"):
        prepare_voice_dataset(source, output, transcribe_clip=transcribe, extract_audio_fn=extract)
    after = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    assert after == before
    assert not list(tmp_path.glob(".dataset-prepare-*"))
    assert not (output / "_scratch").exists()


def test_successful_retry_replaces_clips_and_preserves_notes(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.wav"
    sf.write(source, np.ones(22050, dtype=np.float32) * 0.1, 22050)
    output = tmp_path / "dataset"
    (output / "wavs").mkdir(parents=True)
    (output / "wavs" / "9999.wav").write_bytes(b"old clip")
    (output / "notes.txt").write_text("keep these notes")
    (output / "_scratch").mkdir()
    (output / "_scratch" / "old.wav").write_bytes(b"stale extraction")
    monkeypatch.setattr(
        "raphael.audio.voice_dataset.segment_speech_regions_vad", lambda _audio, rate: [(0, rate)]
    )

    def extract(_source, destination, _rate):
        destination.write_bytes(source.read_bytes())

    report = prepare_voice_dataset(
        source, output, transcribe_clip=lambda *_args: "new words", extract_audio_fn=extract
    )
    assert report.clip_count == 1
    assert report.metadata_path == output / "metadata.csv"
    assert (output / "metadata.csv").read_text() == "wavs/0001.wav|new words\n"
    assert (output / "notes.txt").read_text() == "keep these notes"
    assert not (output / "wavs" / "9999.wav").exists()
    assert not (output / "_scratch").exists()
    assert not list(tmp_path.glob(".dataset-prepare-*"))


def test_preparation_rejects_output_that_contains_original_audio(tmp_path):
    source = tmp_path / "dataset" / "raw"
    source.mkdir(parents=True)
    original = source / "voice.wav"
    original.write_bytes(b"preserve original")
    with pytest.raises(ValueError, match="must not contain"):
        prepare_voice_dataset(source, source.parent, transcribe_clip=lambda *_args: "text")
    assert original.read_bytes() == b"preserve original"


def test_publication_and_recovery_failure_keeps_previous_dataset(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    sf.write(source, np.ones(22050, dtype=np.float32) * 0.1, 22050)
    output = tmp_path / "dataset"
    (output / "wavs").mkdir(parents=True)
    (output / "wavs" / "old.wav").write_bytes(b"original audio")
    (output / "metadata.csv").write_text("wavs/old.wav|original words\n")
    monkeypatch.setattr(
        "raphael.audio.voice_dataset.segment_speech_regions_vad", lambda _audio, rate: [(0, rate)]
    )

    def extract(_source, destination, _rate):
        destination.write_bytes(source.read_bytes())

    rename = Path.rename

    def fail_after_backup(path, target):
        if path == output:
            return rename(path, target)
        raise OSError("destination unavailable")

    monkeypatch.setattr(Path, "rename", fail_after_backup)
    with pytest.raises(RuntimeError, match="previous dataset is safe at"):
        prepare_voice_dataset(
            source, output, transcribe_clip=lambda *_args: "text", extract_audio_fn=extract
        )
    backups = list(tmp_path.glob(".dataset-backup-*/dataset"))
    assert len(backups) == 1
    assert (backups[0] / "wavs" / "old.wav").read_bytes() == b"original audio"
    assert (backups[0] / "metadata.csv").read_text() == "wavs/old.wav|original words\n"
    assert not list(tmp_path.glob(".dataset-prepare-*"))
