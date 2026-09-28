#!/usr/bin/env bash
# ==============================================================================
#  RAPHAEL — Automated Installation & Configuration Setup Script
# ==============================================================================

set -e

# Color helpers
BOLD='\033[1m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}${BOLD}"
echo "  ================================================================"
echo "      RAPHAEL — JARVIS-Style Desktop AI Assistant Installer       "
echo "  ================================================================"
echo -e "${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Check Python
echo -e "${BOLD}[1/5] Checking Python environment...${NC}"
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}❌ Python 3 is not installed. Please install Python 3.10 or newer.${NC}"
    exit 1
fi

PY_VER=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "   Found Python ${GREEN}v$PY_VER${NC} ($($PYTHON_CMD --version))"

# 2. Setup Virtual Environment
echo -e "\n${BOLD}[2/5] Setting up virtual environment (.venv)...${NC}"
if [ ! -d ".venv" ]; then
    echo "   Creating virtual environment in $SCRIPT_DIR/.venv..."
    $PYTHON_CMD -m venv .venv
else
    echo "   Existing virtual environment detected."
fi

# Activate virtual environment
source .venv/bin/activate
echo -e "   Active Python: ${GREEN}$(which python)${NC}"

# 3. Install Dependencies
echo -e "\n${BOLD}[3/5] Installing dependencies and packages...${NC}"
pip install --upgrade pip --quiet
pip install -r requirements.txt
pip install -e . --no-deps --quiet
echo -e "   ${GREEN}✅ Dependencies successfully installed.${NC}"

# 4. Pre-download Default Models (Whisper STT & Piper TTS)
echo -e "\n${BOLD}[4/5] Pre-downloading voice and AI models...${NC}"
python -c "
from pathlib import Path
import os
print('   Checking Piper TTS voice model...')
from piper.download_voices import download_voice
tts_dir = Path('models/tts')
tts_dir.mkdir(parents=True, exist_ok=True)
if not (tts_dir / 'en_GB-alan-medium.onnx').is_file():
    print('   Downloading en_GB-alan-medium voice model (~60MB)...')
    download_voice('en_GB-alan-medium', tts_dir)
print('   Piper voice ready.')

print('   Checking faster-whisper base.en model...')
from faster_whisper import WhisperModel
model = WhisperModel('base.en', device='cpu', compute_type='int8')
print('   Whisper model ready.')
"
echo -e "   ${GREEN}✅ Neural voice and speech models ready.${NC}"

# 5. Configuration & Setup Wizard
echo -e "\n${BOLD}[5/5] Checking configuration (.env)...${NC}"
if [ ! -f ".env" ]; then
    echo -e "   ${YELLOW}No .env file found. Launching configuration wizard...${NC}"
    python -m raphael setup
else
    echo -e "   ${GREEN}Found existing .env configuration.${NC}"
    read -p "   Do you want to reconfigure settings with the interactive wizard? [y/N]: " reconf
    if [[ "$reconf" =~ ^[Yy]$ ]]; then
        python -m raphael setup
    fi
fi

echo -e "\n${CYAN}${BOLD}"
echo "  ================================================================"
echo "      RAPHAEL Installation & Setup Completed Successfully!       "
echo "  ================================================================"
echo -e "${NC}"
echo -e "To start listening:"
echo -e "   ${GREEN}source .venv/bin/activate${NC}"
echo -e "   ${GREEN}python -m raphael --listen${NC}\n"
