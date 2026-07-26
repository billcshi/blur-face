#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] Python 3 not found." >&2
    exit 1
fi

echo "[1/4] Creating isolated environment..."
python3 -m venv .venv
VENV_PYTHON="$(pwd)/.venv/bin/python"

echo "[2/4] Selecting PyTorch compute backend..."
"$VENV_PYTHON" -m pip install --upgrade pip
if command -v nvidia-smi >/dev/null 2>&1 && [[ "${BLUR_FACE_CPU_ONLY:-0}" != "1" ]]; then
    if ! "$VENV_PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
        echo "[GPU] NVIDIA driver detected; installing the tested CUDA 12.6 PyTorch wheel..."
        "$VENV_PYTHON" -m pip install --upgrade --force-reinstall -r requirements.cuda126.lock
    fi
    "$VENV_PYTHON" -c "import torch, sys; print('[GPU] torch', torch.__version__, '| runtime CUDA', torch.version.cuda, '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'UNAVAILABLE'); sys.exit(0 if torch.cuda.is_available() else 1)"
else
    echo "[CPU] NVIDIA driver not selected; installing the standard PyTorch dependency."
fi

echo "[3/4] Installing tested runtime dependencies..."
"$VENV_PYTHON" -m pip install -r requirements.lock
"$VENV_PYTHON" -m pip install --no-deps --editable .

echo "[4/4] Downloading and verifying models..."
"$VENV_PYTHON" scripts/download_models.py

echo
echo "Setup complete."
echo "Run: .venv/bin/blur-face input.mov -o output.mp4"
