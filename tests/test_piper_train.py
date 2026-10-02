"""Tests for Piper training command construction (runtime stays in-process ONNX)."""

from pathlib import Path

import pytest

from raphael.audio.piper_train import (
    PiperTrainError,
    build_piper_train_plan,
    install_trained_voice,
)


def test_build_piper_train_plan_requires_metadata(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    with pytest.raises(PiperTrainError, match="metadata.csv"):
        build_piper_train_plan(dataset_dir=dataset, voice_name="custom_voice")


def test_build_piper_train_plan_from_prepared_dataset(tmp_path: Path):
    dataset = tmp_path / "dataset"
    wavs = dataset / "wavs"
    wavs.mkdir(parents=True)
    (wavs / "0001.wav").write_bytes(b"RIFF")
    (dataset / "metadata.csv").write_text("wavs/0001.wav|hello\n", encoding="utf-8")

    plan = build_piper_train_plan(
        dataset_dir=dataset,
        voice_name="custom_voice",
        work_dir=tmp_path / "training",
        models_dir=tmp_path / "models" / "tts",
        quality="medium",
        max_epochs=2000,
    )

    assert plan.dataset_dir == dataset
    assert plan.output_onnx.name == "custom_voice.onnx"
    assert plan.quality == "medium"
    assert plan.max_epochs == 2000
    assert "train_piper_voice.sh" in plan.steps[0]
    assert plan.steps[0].startswith("/")
    assert "custom_voice" in plan.steps[0]
    assert "2000" in plan.steps[0]


def test_build_piper_train_plan_rejects_path_like_voice_name(tmp_path: Path):
    dataset = tmp_path / "dataset"
    wavs = dataset / "wavs"
    wavs.mkdir(parents=True)
    (wavs / "0001.wav").write_bytes(b"RIFF")
    (dataset / "metadata.csv").write_text("wavs/0001.wav|hello\n", encoding="utf-8")

    with pytest.raises(PiperTrainError, match="voice_name"):
        build_piper_train_plan(dataset, voice_name="../escape")


def test_build_piper_train_plan_rejects_unsupported_quality(tmp_path: Path):
    dataset = tmp_path / "dataset"
    wavs = dataset / "wavs"
    wavs.mkdir(parents=True)
    (wavs / "0001.wav").write_bytes(b"RIFF")
    (dataset / "metadata.csv").write_text("wavs/0001.wav|hello\n", encoding="utf-8")

    with pytest.raises(PiperTrainError, match="Only medium-quality"):
        build_piper_train_plan(dataset, quality="high")


def test_install_trained_voice_copies_onnx_pair(tmp_path: Path):
    source_onnx = tmp_path / "export" / "voice.onnx"
    source_json = tmp_path / "export" / "voice.onnx.json"
    source_onnx.parent.mkdir()
    source_onnx.write_bytes(b"onnx")
    source_json.write_text("{}", encoding="utf-8")
    dest_dir = tmp_path / "models" / "tts"

    installed = install_trained_voice(
        source_onnx=source_onnx,
        voice_name="custom_voice",
        models_dir=dest_dir,
    )

    assert installed == dest_dir / "custom_voice.onnx"
    assert installed.is_file()
    assert (dest_dir / "custom_voice.onnx.json").is_file()


def test_install_trained_voice_rejects_path_like_voice_name(tmp_path: Path):
    source_onnx = tmp_path / "voice.onnx"
    source_json = tmp_path / "voice.onnx.json"
    source_onnx.write_bytes(b"onnx")
    source_json.write_text("{}", encoding="utf-8")

    with pytest.raises(PiperTrainError, match="voice_name"):
        install_trained_voice(source_onnx, voice_name="../escape", models_dir=tmp_path / "models")
