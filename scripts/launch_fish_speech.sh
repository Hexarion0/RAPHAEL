#!/usr/bin/env bash
# Launch local Fish Speech TTS API server for RAPHAEL on GPU with half-precision

set -e

FISH_DIR="/home/hexarion/fish-speech"
PORT="8080"
HOST="127.0.0.1"

if [ ! -d "$FISH_DIR" ]; then
    echo "Fish speech directory not found at $FISH_DIR"
    exit 1
fi

echo "Starting local Fish Speech TTS server at http://${HOST}:${PORT}..."
cd "$FISH_DIR"
source .venv/bin/activate
exec python tools/api_server.py \
    --listen "${HOST}:${PORT}" \
    --device cuda \
    --half \
    --llama-checkpoint-path checkpoints/openaudio-s1-mini \
    --decoder-checkpoint-path checkpoints/openaudio-s1-mini/codec.pth \
    --decoder-config-name modded_dac_vq
