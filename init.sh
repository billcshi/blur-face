#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SETUP_START=$SECONDS

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] Python 3 not found." >&2
    exit 1
fi

STAGE_START=$SECONDS
echo "[1/4] Removing the previous isolated environment and generated metadata..."
python3 scripts/uninstall.py
echo "[TIME] Cleanup: $((SECONDS - STAGE_START))s"

STAGE_START=$SECONDS
echo "[2/4] Creating isolated environment..."
python3 -m venv .venv
VENV_PYTHON="$(pwd)/.venv/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip
echo "[TIME] Environment: $((SECONDS - STAGE_START))s"

STAGE_START=$SECONDS
echo "[3/4] Installing tested base runtime dependencies..."
"$VENV_PYTHON" -m pip install -r requirements.lock
"$VENV_PYTHON" -m pip install --no-deps --editable .
echo "[TIME] Runtime dependencies: $((SECONDS - STAGE_START))s"

STAGE_START=$SECONDS
echo "[4/4] Downloading and verifying the MIT-licensed OpenCV YuNet model..."
"$VENV_PYTHON" scripts/download_models.py
echo "[TIME] YuNet model: $((SECONDS - STAGE_START))s"

if command -v ffmpeg >/dev/null 2>&1; then
    echo "[OK] System FFmpeg: $(command -v ffmpeg)"
else
    echo "[WARN] FFmpeg is not on PATH and is required for video output." >&2
    echo "       Install it with your operating-system package manager, then" >&2
    echo "       open a new shell and run: ffmpeg -version" >&2
fi

echo
echo "Optional detector:"
echo "YuNet is already installed and remains the MIT-licensed default."
echo "YOLO may improve difficult-face recall, but Ultralytics is AGPL-3.0"
echo "unless you have an Enterprise license. The akanametov/yolo-face 1.0.0"
echo "weight is GPL-3.0 or requires an applicable Enterprise license; its"
echo "training recipe uses WIDER FACE, whose dataset terms also apply."
echo "See THIRD_PARTY_NOTICES.md for exact upstream links."
INSTALL_YOLO=0
if [[ "${CI:-false}" == "true" || "${BLUR_FACE_SKIP_YOLO_INSTALL:-0}" == "1" ]]; then
    :
elif [[ "${BLUR_FACE_INSTALL_YOLO:-0}" == "1" ]]; then
    INSTALL_YOLO=1
elif [[ -t 0 ]]; then
    read -r -p "Install the optional YOLO backend and face weight? [y/N] " answer
    [[ "$answer" =~ ^[Yy]$ ]] && INSTALL_YOLO=1
fi
if [[ "$INSTALL_YOLO" == "1" ]]; then
    BLUR_FACE_YOLO_LICENSE_ACCEPTED=1 ./install-yolo.sh
else
    echo "[NOTICE] Optional YOLO skipped; YuNet remains selected."
fi

echo
echo "Setup complete in $((SECONDS - SETUP_START))s."
echo "Geometric mode with YuNet is ready without PyTorch."
echo "Run ./install-sam2.sh for SAM, or ./install-yolo.sh to add YOLO later."
echo "Dependency and model licenses: THIRD_PARTY_NOTICES.md"
echo "Run: .venv/bin/blur-face input.mov -o output.mp4"
