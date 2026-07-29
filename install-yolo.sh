#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

VENV_PYTHON="$(pwd)/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "[ERROR] Project environment not found. Run ./init.sh first." >&2
    exit 1
fi

if [[ "${BLUR_FACE_YOLO_LICENSE_ACCEPTED:-0}" != "1" ]]; then
    echo
    echo "Optional YOLO license notice:"
    echo "- Ultralytics is AGPL-3.0, or requires an Ultralytics Enterprise license."
    echo "- akanametov/yolo-face 1.0.0 states GPL-3.0 or Enterprise terms for"
    echo "  yolov11m-face.pt; its training recipe uses WIDER FACE, whose data"
    echo "  terms also apply."
    echo "- Neither component is part of blur-face's default MIT/Apache runtime."
    echo "See THIRD_PARTY_NOTICES.md before continuing."
    if [[ ! -t 0 ]]; then
        echo "[ERROR] Interactive license acceptance is required." >&2
        echo "Run this script in a terminal, or explicitly set BLUR_FACE_YOLO_LICENSE_ACCEPTED=1." >&2
        exit 1
    fi
    read -r -p "Accept these upstream terms and install optional YOLO? [y/N] " answer
    if [[ ! "$answer" =~ ^[Yy]$ ]]; then
        echo "[NOTICE] Optional YOLO installation cancelled."
        exit 0
    fi
fi

if command -v nvidia-smi >/dev/null 2>&1 && [[ "${BLUR_FACE_CPU_ONLY:-0}" != "1" ]]; then
    echo "[1/3] Installing the tested CUDA 12.6 PyTorch runtime..."
    "$VENV_PYTHON" -m pip install --upgrade --force-reinstall -r requirements.cuda126.lock
else
    echo "[1/3] Installing the tested CPU PyTorch runtime..."
    "$VENV_PYTHON" -m pip install --upgrade --force-reinstall -r requirements.cpu.lock
fi

echo "[2/3] Installing the optional AGPL-3.0 Ultralytics backend..."
"$VENV_PYTHON" -m pip install -r requirements.yolo.lock

echo "[3/3] Downloading and SHA-256 verifying the optional face weight..."
"$VENV_PYTHON" scripts/download_models.py --yolo
"$VENV_PYTHON" -c "import torch, ultralytics; print('[OK] Optional YOLO is ready | ultralytics', ultralytics.__version__, '| torch', torch.__version__, '| device', 'cuda' if torch.cuda.is_available() else 'cpu')"
echo "Choose YOLO in the UI, or use: --detector yolo --model yolov11m-face.pt"
