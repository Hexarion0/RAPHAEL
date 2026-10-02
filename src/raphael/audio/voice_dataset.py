"""Prepare a Piper TTS dataset from local video/audio the user has rights to use.

Runtime synthesis stays a small in-process ONNX model. This module only builds
training data: extract audio, split on speech energy, transcribe, write metadata.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from raphael.logging import get_logger

logger = get_logger("audio.voice_dataset")

MEDIA_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".webm",
    ".mov",
    ".avi",
    ".mp3",
    ".wav",
    ".m4a",
    ".ogg",
    ".flac",
    ".opus",
}

PIPER_SAMPLE_RATE = 22050
_MIN_CLIP_SECONDS = 1.0
_MAX_CLIP_SECONDS = 12.0
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_VOICE_DIR = PROJECT_ROOT / "training" / "raw"
PREPARED_DATASET_DIR = PROJECT_ROOT / "training" / "dataset"


@dataclass(frozen=True)
class VoiceDatasetReport:
    """Summary of a prepared Piper dataset."""

    output_dir: Path
    source_count: int
    clip_count: int
    skipped_count: int
    metadata_path: Path


def collect_media_files(input_path: Path) -> list[Path]:
    """Return supported media files from a file or directory tree."""
    root = Path(input_path)
    if root.is_file():
        if root.suffix.lower() not in MEDIA_EXTENSIONS:
            raise ValueError(f"Unsupported media type: {root.suffix}")
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"Media path not found: {root}")

    files = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
    ]
    if not files:
        raise FileNotFoundError(f"No supported media files under {root}")
    return files


def segment_speech_regions(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: int = 30,
    min_clip_seconds: float = _MIN_CLIP_SECONDS,
    max_clip_seconds: float = _MAX_CLIP_SECONDS,
    pad_ms: int = 120,
) -> list[tuple[int, int]]:
    """Split mono audio into speech regions using RMS energy."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if min_clip_seconds <= 0 or max_clip_seconds < min_clip_seconds or frame_ms <= 0 or pad_ms < 0:
        raise ValueError(
            "clip limits and frame duration must be positive; padding cannot be negative"
        )

    samples = np.asarray(audio, dtype=np.float32).squeeze()
    if samples.ndim != 1 or samples.size == 0:
        return []

    frame_size = max(1, int(sample_rate * frame_ms / 1000))
    rms = np.array(
        [
            float(np.sqrt(np.mean(np.square(samples[i : i + frame_size]))))
            for i in range(0, samples.size, frame_size)
        ],
        dtype=np.float32,
    )
    if rms.size == 0:
        return []

    peak = float(np.max(rms))
    if peak < 0.002:
        return []

    noise_floor = float(np.percentile(rms, 15))
    spread = float(np.percentile(rms, 85) - noise_floor)
    if spread < peak * 0.2:
        # Continuous speech (no quiet baseline) — keep the whole take.
        speech_frames = rms >= max(0.002, peak * 0.03)
    else:
        threshold = max(0.002, noise_floor + 0.2 * (peak - noise_floor))
        speech_frames = rms >= threshold

    regions: list[tuple[int, int]] = []
    start_frame: int | None = None
    for index, is_speech in enumerate(speech_frames):
        if is_speech and start_frame is None:
            start_frame = index
        elif not is_speech and start_frame is not None:
            regions.append(_frames_to_samples(start_frame, index, frame_size, samples.size))
            start_frame = None
    if start_frame is not None:
        regions.append(
            _frames_to_samples(start_frame, len(speech_frames), frame_size, samples.size)
        )

    pad = int(sample_rate * pad_ms / 1000)
    min_samples = int(sample_rate * min_clip_seconds)
    max_samples = int(sample_rate * max_clip_seconds)
    clipped: list[tuple[int, int]] = []
    for start, end in regions:
        start = max(0, start - pad)
        end = min(samples.size, end + pad)
        if end - start < min_samples:
            continue
        cursor = start
        while end - cursor > max_samples:
            clipped.append((cursor, cursor + max_samples))
            cursor += max_samples
        if end - cursor >= min_samples:
            clipped.append((cursor, end))
    return clipped


def segment_speech_regions_vad(
    audio: np.ndarray,
    sample_rate: int,
    min_clip_seconds: float = _MIN_CLIP_SECONDS,
    max_clip_seconds: float = _MAX_CLIP_SECONDS,
    pad_ms: int = 120,
    merge_silence_ms: int = 400,
    threshold: float = 0.35,
) -> list[tuple[int, int]]:
    """Find quiet speech with Silero VAD and return regions in source samples."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if (
        min_clip_seconds <= 0
        or max_clip_seconds < min_clip_seconds
        or pad_ms < 0
        or merge_silence_ms < 0
    ):
        raise ValueError(
            "clip limits must be positive and ordered; padding and silence cannot be negative"
        )

    try:
        from pysilero_vad import SileroVoiceActivityDetector
    except ImportError:
        logger.warning("Silero VAD is unavailable; falling back to RMS segmentation")
        return segment_speech_regions(
            audio,
            sample_rate,
            min_clip_seconds=min_clip_seconds,
            max_clip_seconds=max_clip_seconds,
            pad_ms=pad_ms,
        )

    vad_rate = 16000
    chunk_size = SileroVoiceActivityDetector.chunk_samples()
    samples = np.asarray(audio, dtype=np.float32).squeeze()
    if samples.ndim != 1 or samples.size == 0:
        return []

    target_size = round(samples.size * vad_rate / sample_rate)
    detector = SileroVoiceActivityDetector()
    chunk_offsets = np.arange(chunk_size, dtype=np.float64)
    speech_frames: list[bool] = []
    for target_start in range(0, target_size, chunk_size):
        target_positions = target_start + chunk_offsets
        source_positions = target_positions * sample_rate / vad_rate
        valid = target_positions < target_size
        left = np.floor(source_positions[valid]).astype(np.intp)
        fractions = (source_positions[valid] - left).astype(np.float32)
        right = np.minimum(left + 1, samples.size - 1)
        chunk = np.zeros(chunk_size, dtype=np.float32)
        chunk[valid] = samples[left] * (1.0 - fractions) + samples[right] * fractions
        speech_frames.append(detector.process_array(chunk) >= threshold)
    speech_frames = np.asarray(speech_frames, dtype=np.bool_)

    changes = np.diff(np.pad(speech_frames.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    max_gap_frames = round(merge_silence_ms * vad_rate / (1000 * chunk_size))
    pad_samples = int(vad_rate * pad_ms / 1000)
    min_samples = int(vad_rate * min_clip_seconds)
    max_samples = int(vad_rate * max_clip_seconds)

    merged: list[tuple[int, int]] = []
    for start_frame, end_frame in zip(starts, ends):
        start_sample = min(int(start_frame) * chunk_size, target_size)
        end_sample = min(int(end_frame) * chunk_size, target_size)
        if merged and start_sample - merged[-1][1] <= max_gap_frames * chunk_size:
            merged[-1] = (merged[-1][0], end_sample)
        else:
            merged.append((start_sample, end_sample))

    regions: list[tuple[int, int]] = []
    for start, end in merged:
        start = max(0, start - pad_samples)
        end = min(target_size, end + pad_samples)
        if end - start < min_samples:
            continue
        while end - start > max_samples:
            next_end = start + max_samples
            regions.append(
                (round(start * sample_rate / vad_rate), round(next_end * sample_rate / vad_rate))
            )
            start = next_end
        if end - start >= min_samples:
            regions.append(
                (
                    round(start * sample_rate / vad_rate),
                    min(samples.size, round(end * sample_rate / vad_rate)),
                )
            )
    return regions


def write_piper_metadata(metadata_path: Path, clips: list[tuple[Path, str]]) -> None:
    """Write Piper `|`-delimited metadata relative to the dataset root."""
    dataset_root = metadata_path.parent
    lines: list[str] = []
    for wav_path, text in clips:
        relative = wav_path.resolve().relative_to(dataset_root.resolve()).as_posix()
        cleaned = " ".join(text.split())
        lines.append(f"{relative}|{cleaned}")
    metadata_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def extract_audio_with_ffmpeg(media_path: Path, dest_wav: Path, sample_rate: int) -> Path:
    """Extract mono WAV audio with ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to extract audio from video files.")

    dest_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "s16",
        str(dest_wav),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not dest_wav.is_file():
        raise RuntimeError(f"ffmpeg failed on {media_path.name}: {result.stderr[-500:]}")
    return dest_wav


def prepare_voice_dataset(
    input_path: Path,
    output_dir: Path,
    transcribe_clip: Callable[[np.ndarray, int], str] | None = None,
    extract_audio_fn: Callable[[Path, Path, int], Path] = extract_audio_with_ffmpeg,
    sample_rate: int = PIPER_SAMPLE_RATE,
    language: str = "en",
) -> VoiceDatasetReport:
    """Build a Piper dataset (`wavs/` + `metadata.csv`) from local media files."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    source = Path(input_path).resolve()
    output = Path(output_dir).absolute()
    if source.is_relative_to(output.resolve()):
        raise ValueError("Dataset output must not contain the source audio")
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("Dataset output must be a directory, not a file or symlink")
    media_files = collect_media_files(source)
    if transcribe_clip is None:
        transcribe_clip = _default_transcriber(language)
    output.parent.mkdir(parents=True, exist_ok=True)
    skipped = 0

    # Everything is prepared beside the destination on the same filesystem.
    # The previous dataset is untouched until clips and metadata are complete.
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-prepare-", dir=output.parent) as tmp:
        work = Path(tmp)
        staging = work / "dataset"
        wavs_dir = staging / "wavs"
        wavs_dir.mkdir(parents=True)
        scratch = work / "scratch"
        scratch.mkdir()
        clips: list[tuple[Path, str]] = []
        for index, media in enumerate(media_files):
            extracted = scratch / f"{index:04d}.wav"
            logger.info("Extracting audio from %s", media.name)
            extract_audio_fn(media, extracted, sample_rate)
            audio, file_rate = sf.read(extracted, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if file_rate != sample_rate:
                raise ValueError(
                    f"Unexpected sample rate {file_rate} Hz for {media.name}; "
                    f"expected {sample_rate} Hz."
                )
            regions = segment_speech_regions_vad(audio, sample_rate)
            if not regions:
                regions = segment_speech_regions(audio, sample_rate)
            for start, end in regions:
                text = transcribe_clip(audio[start:end], sample_rate).strip()
                if not text:
                    skipped += 1
                    continue
                wav_path = wavs_dir / f"{len(clips) + 1:04d}.wav"
                sf.write(wav_path, audio[start:end], sample_rate, subtype="PCM_16")
                clips.append((wav_path, text))
        if not clips:
            raise ValueError(
                "No usable speech clips were prepared; "
                "check source audio and transcription settings."
            )
        write_piper_metadata(staging / "metadata.csv", clips)
        backup = None
        if output.exists():
            # Preserve notes and other files that are not generated dataset artifacts.
            for extra in output.iterdir():
                if extra.name in {"wavs", "metadata.csv", "_scratch"}:
                    continue
                destination = staging / extra.name
                if extra.is_dir() and not extra.is_symlink():
                    shutil.copytree(extra, destination, symlinks=True)
                else:
                    shutil.copy2(extra, destination, follow_symlinks=False)
            backup_root = Path(
                tempfile.mkdtemp(prefix=f".{output.name}-backup-", dir=output.parent)
            )
            backup = backup_root / "dataset"
            try:
                output.rename(backup)
            except BaseException:
                backup_root.rmdir()
                raise
        try:
            staging.rename(output)
        except BaseException:
            if backup is not None:
                try:
                    backup.rename(output)
                except OSError as err:
                    raise RuntimeError(
                        f"Dataset recovery failed; previous dataset is safe at {backup}"
                    ) from err
                backup.parent.rmdir()
            raise
        if backup is not None:
            shutil.rmtree(backup.parent)
        clip_count = len(clips)

    report = VoiceDatasetReport(
        output_dir=output,
        source_count=len(media_files),
        clip_count=clip_count,
        skipped_count=skipped,
        metadata_path=output / "metadata.csv",
    )
    logger.info(
        "Prepared Piper dataset: %d clips from %d sources at %s",
        clip_count,
        len(media_files),
        output,
    )
    return report


def _frames_to_samples(
    start_frame: int,
    end_frame: int,
    frame_size: int,
    total: int,
) -> tuple[int, int]:
    start = start_frame * frame_size
    end = min(total, end_frame * frame_size)
    return start, end


def _default_transcriber(language: str) -> Callable[[np.ndarray, int], str]:
    from raphael.audio.stt import SpeechToText
    from raphael.config import get_settings

    settings = get_settings().audio
    stt = SpeechToText(
        model_size=settings.stt_model,
        device=settings.stt_device,
        compute_type=settings.stt_compute_type,
        language=language or settings.stt_language,
        initial_prompt="Clear spoken dialogue for text-to-speech training.",
    )
    stt.wait_ready()

    def transcribe(audio: np.ndarray, sample_rate: int) -> str:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        samples = np.asarray(audio, dtype=np.float32).squeeze()
        if sample_rate != 16000:
            target_size = round(samples.size * 16000 / sample_rate)
            source_positions = np.arange(samples.size, dtype=np.float32)
            target_positions = np.arange(target_size, dtype=np.float32) * (sample_rate / 16000)
            samples = np.interp(target_positions, source_positions, samples).astype(np.float32)
        return stt.transcribe(samples, language=language)

    return transcribe
