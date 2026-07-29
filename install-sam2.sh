#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SETUP_START=$SECONDS

VENV_PYTHON="$(pwd)/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "[ERROR] Project environment not found. Run ./init.sh first." >&2
    exit 1
fi

STAGE_START=$SECONDS
if command -v nvidia-smi >/dev/null 2>&1 && [[ "${BLUR_FACE_CPU_ONLY:-0}" != "1" ]]; then
    echo "[1/2] Installing the tested CUDA 12.6 PyTorch runtime..."
    "$VENV_PYTHON" -m pip install --upgrade --force-reinstall -r requirements.cuda126.lock
else
    echo "[1/2] Installing the tested CPU PyTorch runtime..."
    "$VENV_PYTHON" -m pip install --upgrade --force-reinstall -r requirements.cpu.lock
fi
echo "[TIME] PyTorch: $((SECONDS - STAGE_START))s"

STAGE_START=$SECONDS
echo "[2/2] Installing optional SAM 2.1 support..."
"$VENV_PYTHON" -m pip install -r requirements.sam2.lock
"$VENV_PYTHON" -c "import torch; from transformers import Sam2Model, Sam2Processor, Sam2VideoModel, Sam2VideoProcessor; print('[OK] SAM 2.1 support is ready | torch', torch.__version__, '| device', 'cuda' if torch.cuda.is_available() else 'cpu')"
echo "[TIME] SAM dependencies: $((SECONDS - STAGE_START))s"
echo "SAM setup complete in $((SECONDS - SETUP_START))s. The selected checkpoint downloads on first use unless offline mode is enabled."
echo "Run ./start-ui.sh and choose SAM 2.1 under Mask engine."
