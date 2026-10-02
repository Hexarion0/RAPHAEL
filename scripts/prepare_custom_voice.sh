#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_NAME="${1:-custom_voice}"
shift || true

if [[ ! "${VOICE_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
  echo "Voice name may contain only letters, numbers, underscores, and hyphens."
  exit 1
fi

RAW_DIR="${ROOT}/training/raw/${VOICE_NAME}"
DATASET_DIR="${ROOT}/training/dataset/${VOICE_NAME}"
mkdir -p "${RAW_DIR}" "${DATASET_DIR}"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <voice_name> <youtube_url_1> [youtube_url_2 ...]"
  exit 1
fi

# Download the highest-quality available audio track for each supplied URL.
for url in "$@"; do
  echo "Downloading audio from: $url"
  . "${ROOT}/.venv/bin/activate"
  yt-dlp -x --audio-format wav --audio-quality 0 --restrict-filenames \
    -o "${RAW_DIR}/%(title)s.%(ext)s" "$url"
done

# Prepare the dataset for Piper training.
. "${ROOT}/.venv/bin/activate"
python -m raphael prepare-voice --input "${RAW_DIR}" --output "${DATASET_DIR}" --voice-name "${VOICE_NAME}"

echo "Voice dataset ready at: ${DATASET_DIR}"
echo "Next heavy step: ./scripts/train_piper_voice.sh ${DATASET_DIR} ${VOICE_NAME}"
