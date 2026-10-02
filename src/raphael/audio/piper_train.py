"""Train a custom Piper ONNX voice from a prepared dataset.

Training may use the GPU. Inference stays in-process: RAPHAEL loads the
exported `.onnx` with piper-tts in the same Python process.
"""

from __future__ import annotations

import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path

from raphael.audio.voice_dataset import PROJECT_ROOT
from raphael.logging import get_logger

logger = get_logger("audio.piper_train")


class PiperTrainError(RuntimeError):
    """Raised when a Piper training dataset or export is invalid."""


@dataclass(frozen=True)
class PiperTrainPlan:
    """Command plan for preprocess → train → export → install."""

    dataset_dir: Path
    work_dir: Path
    output_onnx: Path
    voice_name: str
    quality: str
    max_epochs: int
    steps: list[str]


def build_piper_train_plan(
    dataset_dir: Path,
    voice_name: str = "custom_voice",
    work_dir: Path | None = None,
    models_dir: Path | None = None,
    quality: str = "medium",
    max_epochs: int = 500,
) -> PiperTrainPlan:
    """Validate a prepared dataset and describe the Piper training steps."""
    dataset = Path(dataset_dir)
    metadata = dataset / "metadata.csv"
    wavs = dataset / "wavs"
    if max_epochs <= 0:
        raise PiperTrainError("max_epochs must be positive")
    if quality != "medium":
        raise PiperTrainError("Only medium-quality Piper checkpoints are currently supported")
    if not metadata.is_file():
        raise PiperTrainError(f"Piper dataset missing metadata.csv in {dataset}")
    if not wavs.is_dir() or not any(wavs.glob("*.wav")):
        raise PiperTrainError(f"Piper dataset missing wavs/*.wav in {dataset}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", voice_name):
        raise PiperTrainError("voice_name may contain only letters, numbers, '_' and '-'")

    work = Path(work_dir) if work_dir is not None else PROJECT_ROOT / "training" / "piper"
    models = Path(models_dir) if models_dir is not None else PROJECT_ROOT / "models" / "tts"
    output_onnx = models / f"{voice_name}.onnx"
    steps = [
        shlex.join(
            [
                str(PROJECT_ROOT / "scripts" / "train_piper_voice.sh"),
                str(dataset),
                voice_name,
                str(max_epochs),
            ]
        )
    ]
    return PiperTrainPlan(
        dataset_dir=dataset,
        work_dir=work,
        output_onnx=output_onnx,
        voice_name=voice_name,
        quality=quality,
        max_epochs=max_epochs,
        steps=steps,
    )


def install_trained_voice(
    source_onnx: Path,
    voice_name: str = "custom_voice",
    models_dir: Path | None = None,
) -> Path:
    """Copy a trained Piper ONNX pair into RAPHAEL's in-process models directory."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", voice_name):
        raise PiperTrainError("voice_name may contain only letters, numbers, '_' and '-'")

    onnx_path = Path(source_onnx)
    if not onnx_path.is_file():
        raise PiperTrainError(f"Trained ONNX not found: {onnx_path}")

    json_path = onnx_path.with_suffix(".onnx.json")
    if not json_path.is_file():
        json_path = onnx_path.with_name(f"{onnx_path.stem}.json")
    if not json_path.is_file():
        raise PiperTrainError(f"Piper config JSON not found next to {onnx_path}")

    dest_dir = Path(models_dir) if models_dir is not None else PROJECT_ROOT / "models" / "tts"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_onnx = dest_dir / f"{voice_name}.onnx"
    dest_json = dest_dir / f"{voice_name}.onnx.json"
    shutil.copy2(onnx_path, dest_onnx)
    shutil.copy2(json_path, dest_json)
    logger.info("Installed custom Piper voice at %s", dest_onnx)
    return dest_onnx
