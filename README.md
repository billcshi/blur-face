# blur-face

**YOLO + custom tracker — face detection, tracking, and blurring with prediction through occlusions.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-supported-green.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

blur-face automatically detects and blurs faces in videos. Unlike most tools that process each frame independently, it **tracks faces across frames** and **predicts positions through occlusions** — no flickering, no gaps, no missed frames.

## Why This Project Exists

I created this tool to anonymize my own intimate videos before sharing them with partners. This means **every design decision prioritizes privacy and local execution**:

- **100% local** — no cloud uploads, no network calls, no telemetry. Everything runs on your machine.
- **Open source (MIT)** — you can read every line, audit it, and build it yourself.
- **Zero external tracker deps** — no supervision, no ultralytics tracker that could change silently.
- **iPhone-compatible output** — videos stay in your control, not in a cloud service.

If you're blurring sensitive content, you need a tool you can trust completely. Not a SaaS. Not a black-box app. Something verifiable, offline, and transparent.

https://github.com/user-attachments/assets/placeholder

## Why not deface / others?

| | deface | blur-face |
|---|---|---|
| Per-frame detection | ✅ | ✅ (YOLO) |
| Cross-frame tracking | ❌ | ✅ custom tracker |
| Prediction through occlusion | ❌ | ✅ up to N frames |
| Smooth coordinates | ❌ | ✅ exponential EMA |
| Review mode (colored IDs) | ❌ | ✅ `--debug` |
| Selective exclusion | ❌ | ✅ `--exclude-ids` |
| Time-segmented thresholds | ❌ | ✅ `--time-thresh` |
| iPhone-compatible output | ⚠️ | ✅ H.264+AAC+faststart |
| External tracker deps | — | **zero** (custom built) |

## Quick Start

### 1. Install

```bash
# Windows
init.bat

# Linux / macOS
chmod +x init.sh && ./init.sh
```

This installs Python dependencies and downloads face detection models to `models/`.

### 2. Run

```bash
# Review faces with colored boxes
python blur-face.py video.mov --debug

# Blur all faces
python blur-face.py video.mov -o output.mp4

# Exclude certain faces (keep them unblurred)
python blur-face.py video.mov --exclude-ids 2,5

# Different sensitivity per time range
python blur-face.py video.mov --time-thresh "0:0.15,120:0.3"
```

## Performance

| GPU | Model | Video | Speed |
|-----|-------|-------|-------|
| RTX 3080 Ti | yolov11m-face.pt | 1080p, 275s | ~1.1× slower than real-time |
| RTX 3080 Ti | yolo26n-face.pt | 1080p, 275s | ~1.4× faster than real-time |
| CPU (i7-11700K) | yolov11m-face.pt | 1080p | ~5× slower than real-time |

Processing speed scales with model size. Available models from [yolo-face releases](https://github.com/akanametov/yolo-face/releases):

| Model | Size | Speed | Best for |
|-------|------|-------|----------|
| `yolo26n-face.pt` | 5.6 MB | Fastest | Well-lit, frontal faces |
| `yolov10n-face.pt` | 5.5 MB | Fast | Quick preview |
| `yolov10s-face.pt` | 15.7 MB | Medium | Balanced |
| `yolov11m-face.pt` | 38.6 MB | Slower | Edge faces, low light (default) |
| `yolov11l-face.pt` | 48.8 MB | Slowest | Maximum detection rate |

## How it works

```
Frame → YOLO face detection → Custom Tracker → Blur / Debug draw → ffmpeg H.264
                                  │
                    ┌─────────────┼─────────────┐
                    │  Matches detections to     │
                    │  existing tracks by        │
                    │  centroid distance.        │
                    │                            │
                    │  Smooths with EMA.         │
                    │                            │
                    │  When detection misses:    │
                    │  → holds last position     │
                    │  → marks as "predicted"    │
                    │  → drops after N frames    │
                    └────────────────────────────┘
```

**Key insight:** The tracker is called **every frame**, even when YOLO finds nothing. This is what enables prediction — other tools skip the tracker on empty detections and lose their tracks.

## Options

```
--model PATH         Model file (default: yolov11m-face.pt)
--thresh FLOAT       Detection confidence threshold (0-1, default 0.2)
--time-thresh STR    Per-segment thresholds: "sec:thresh,sec:thresh"
--mask-scale FLOAT   Blur region expansion (default 1.15)
--blur-kernel INT    Gaussian kernel size, odd, larger = heavier blur (default 51)
--device cuda/cpu    Default cuda (auto-fallback to cpu)
--debug              Review mode: colored boxes + IDs, no blur applied
--profile            Show per-phase timing breakdown
--exclude-ids STR    Comma-separated track IDs to leave unblurred
--lost-buffer INT    Frames to predict after detection lost (default 60 ≈ 2s)
--smooth FLOAT       Coordinate smoothing factor 0-1 (default 0.7)
-o, --output PATH    Output video path
```

## Project structure

```
blur-face/
├── blur-face.py          Main entry point (orchestration)
├── init.bat / init.sh    One-click setup scripts
├── blur-batch.bat        Batch process all videos in a folder
├── README.md
├── LICENSE
└── blurface/             Core package
    ├── cli.py            Argument parsing & config
    ├── detector.py       YOLO face detection wrapper
    ├── tracker.py        Custom multi-face tracker
    ├── renderer.py       Blur & debug-draw operations
    ├── encoder.py        ffmpeg H.264 encoding pipe
    └── profiler.py       Per-phase timing utilities
```

Each module is self-contained and independently testable. The tracker has **zero external dependencies** beyond NumPy — no supervision, no ultralytics tracker, no ByteTrack config files that break between versions.

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (works on CPU, just slower)
- `pip install ultralytics opencv-python imageio-ffmpeg numpy`
- GPU detection uses PyTorch CUDA and GPU encoding uses NVENC. GPU blur rendering additionally requires an OpenCV build with CUDA; the standard `opencv-python` wheel falls back to CPU rendering.

## License

MIT © 2025 Jiechang Shi

中文说明: [README.zh.md](README.zh.md)
