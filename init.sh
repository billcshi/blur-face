#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] Python 3 not found." >&2
    exit 1
fi

echo "[1/3] Creating isolated environment..."
python3 -m venv .venv
VENV_PYTHON="$(pwd)/.venv/bin/python"

echo "[2/3] Installing tested dependencies..."
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r requirements.lock
"$VENV_PYTHON" -m pip install --no-deps .

echo "[3/3] Downloading and verifying models..."
"$VENV_PYTHON" scripts/download_models.py

echo
echo "Setup complete."
echo "Run: .venv/bin/blur-face input.mov -o output.mp4"
