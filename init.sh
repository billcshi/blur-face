#!/usr/bin/env bash
# ==============================================
#  blur-face — init script (Linux / macOS)
# ==============================================
set -e
cd "$(dirname "$0")"

echo
echo "============================================"
echo "  blur-face — Setup"
echo "============================================"
echo

# --- Check Python ---
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3 not found."
    exit 1
fi
echo "[OK] Python found: $(python3 --version)"

# --- Install dependencies ---
echo
echo "[1/3] Installing Python packages..."
pip3 install ultralytics opencv-python imageio-ffmpeg numpy

# --- Check CUDA ---
echo
echo "[2/3] Checking CUDA (optional)..."
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())" 2>/dev/null || true

# --- Download models ---
echo
echo "[3/3] Downloading face detection models..."
MODELS_DIR="$(pwd)/models"
mkdir -p "$MODELS_DIR"

# yolov11m-face.pt (accurate, 38.6 MB)
MEDIUM="$MODELS_DIR/yolov11m-face.pt"
if [ ! -f "$MEDIUM" ]; then
    echo "  - yolov11m-face.pt (accurate, ~38.6 MB) ..."
    curl -L -o "$MEDIUM" "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11m-face.pt" 2>/dev/null && echo "    [OK]" || echo "    [SKIP] download failed"
else
    echo "  [OK] yolov11m-face.pt exists"
fi

# yolo26n-face.pt (fast, 5.6 MB)
NANO="$MODELS_DIR/yolo26n-face.pt"
if [ ! -f "$NANO" ]; then
    echo "  - yolo26n-face.pt (fast, ~5.6 MB) ..."
    curl -L -o "$NANO" "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolo26n-face.pt" 2>/dev/null && echo "    [OK]" || echo "    [SKIP] download failed"
else
    echo "  [OK] yolo26n-face.pt exists"
fi

echo
echo "============================================"
echo "  Setup complete!"
echo
echo "  Usage:"
echo "    python3 blur-face.py input.mov --debug"
echo "    python3 blur-face.py input.mov -o output.mp4"
echo
echo "  To use local models:"
echo "    python3 blur-face.py input.mov --model models/yolov11m-face.pt"
echo "============================================"
