#!/usr/bin/env bash
# Train a custom Piper voice with a dedicated GPU venv, then install the ONNX
# into RAPHAEL. Inference does not start this script — RAPHAEL loads the .onnx
# in-process via piper-tts.
#
# Usage:
#   ./scripts/train_piper_voice.sh [dataset_dir] [voice_name] [max_epochs]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="${1:-${ROOT}/training/dataset}"
VOICE_NAME="${2:-custom_voice}"
MAX_EPOCHS="${3:-500}"
TRAIN_DIR="${ROOT}/training/piper"
VENV_DIR="${TRAIN_DIR}/.venv"
EXPORT_DIR="${TRAIN_DIR}/export"
RUN_DIR="${TRAIN_DIR}/runs/${VOICE_NAME}"
CHECKPOINT_DIR="${TRAIN_DIR}/checkpoints"
MODELS_DIR="${ROOT}/models/tts"
PIPER_CHECKPOINT="${PIPER_CHECKPOINT:-${CHECKPOINT_DIR}/en_US-amy-medium.ckpt}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
BATCH_SIZE="${PIPER_BATCH_SIZE:-2}"

if [[ ! "${VOICE_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
  echo "Voice name may contain only letters, numbers, underscores, and hyphens."
  exit 1
fi

if [[ ! -s "${DATASET_DIR}/metadata.csv" ]]; then
  echo "Missing ${DATASET_DIR}/metadata.csv"
  echo "Run: python -m raphael prepare-voice"
  exit 1
fi

shopt -s nullglob
AUDIO_FILES=("${DATASET_DIR}"/wavs/*.wav)
shopt -u nullglob
if [[ "${#AUDIO_FILES[@]}" -eq 0 ]]; then
  echo "No WAV clips found in ${DATASET_DIR}/wavs"
  exit 1
fi
DATASET_REAL="$(realpath -- "${DATASET_DIR}")"
while IFS='|' read -r relative_audio _; do
  [[ -z "${relative_audio}" ]] && continue
  audio_path="$(realpath -m -- "${DATASET_DIR}/${relative_audio}")"
  if [[ "${audio_path}" != "${DATASET_REAL}"/* ]]; then
    echo "Dataset metadata references audio outside the dataset: ${relative_audio}"
    exit 1
  fi
  if [[ ! -f "${audio_path}" ]]; then
    echo "Dataset metadata references missing audio: ${audio_path}"
    exit 1
  fi
done < "${DATASET_DIR}/metadata.csv"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Missing Piper training environment: ${VENV_DIR}"
  exit 1
fi

mkdir -p "${EXPORT_DIR}" "${RUN_DIR}" "${CHECKPOINT_DIR}" "${MODELS_DIR}"
source "${VENV_DIR}/bin/activate"

if [[ ! -f "${PIPER_CHECKPOINT}" ]]; then
  echo "Missing Piper fine-tuning checkpoint: ${PIPER_CHECKPOINT}"
  echo "Download the official en_US-amy medium checkpoint before training."
  exit 1
fi
if [[ ! "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PIPER_BATCH_SIZE must be a positive integer."
  exit 1
fi
if [[ -n "${RESUME_CHECKPOINT}" && ! -f "${RESUME_CHECKPOINT}" ]]; then
  echo "Resume checkpoint not found: ${RESUME_CHECKPOINT}"
  exit 1
fi

TRAIN_ARGS=(
  --data.voice_name "${VOICE_NAME}"
  --data.csv_path "${DATASET_DIR}/metadata.csv"
  --data.audio_dir "${DATASET_DIR}"
  --data.cache_dir "${TRAIN_DIR}/cache/${VOICE_NAME}"
  --data.config_path "${EXPORT_DIR}/${VOICE_NAME}.onnx.json"
  --data.espeak_voice en-us
  --data.batch_size "${BATCH_SIZE}"
  --data.num_workers 2
  --model.sample_rate 22050
  --trainer.accelerator gpu
  --trainer.devices 1
  --trainer.max_epochs "${MAX_EPOCHS}"
  --trainer.limit_val_batches 2
  --trainer.default_root_dir "${RUN_DIR}"
)
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  TRAIN_ARGS+=(--ckpt_path "${RESUME_CHECKPOINT}")
else
  TRAIN_ARGS+=(--model.warmstart_ckpt "${PIPER_CHECKPOINT}")
fi

echo "Fine-tuning ${VOICE_NAME} for up to ${MAX_EPOCHS} epochs on the local GPU (batch size ${BATCH_SIZE})..."
python -m piper.train fit "${TRAIN_ARGS[@]}"

CHECKPOINT="$(find "${RUN_DIR}" -type f -name '*.ckpt' -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
if [[ -z "${CHECKPOINT}" ]]; then
  echo "No checkpoint found after training."
  exit 1
fi

echo "Exporting ONNX from ${CHECKPOINT}..."
python -m piper.train.export_onnx \
  --checkpoint "${CHECKPOINT}" \
  --output-file "${EXPORT_DIR}/${VOICE_NAME}.onnx"

cp "${EXPORT_DIR}/${VOICE_NAME}.onnx" "${MODELS_DIR}/${VOICE_NAME}.onnx"
cp "${EXPORT_DIR}/${VOICE_NAME}.onnx.json" "${MODELS_DIR}/${VOICE_NAME}.onnx.json"

echo "Installed in-process voice: ${MODELS_DIR}/${VOICE_NAME}.onnx"
echo "Set TTS_ENGINE=piper and TTS_VOICE=${VOICE_NAME} then run python -m raphael"
