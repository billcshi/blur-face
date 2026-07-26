# Blur Face Local

**A local-first face anonymization studio for video — now with a browser UI.**

[中文说明](README.zh.md) · [Version 1.0.0 notes](CHANGELOG.md)

Blur Face detects, tracks, and obscures faces without sending video frames to a
remote service. Version 1.0 makes the local web interface the easiest way to
use the project, while keeping the complete CLI for automation and advanced
workflows.

![Blur Face Local Studio UI](docs/blur-face-local-studio.png)

## What you get in 1.0

- A polished local web UI with native input, output, and model file pickers.
- Automatic English/Chinese selection from browser preferences, plus
  `Auto / EN / 中文` manual controls.
- Local model discovery: every `.pt` file in the configured `models`
  directories appears in the UI.
- Helpful `?` explanations for every processing option.
- Standard, Strong, and Strict Privacy presets with editable advanced settings.
- Live frame progress, logs, and safe cancellation.
- Conservative multi-face tracking with optical-flow support during short
  detector gaps.
- Confidence-adaptive blur, strong privacy defaults, and three mask shapes.
- Atomic output replacement: an incomplete encode never replaces a good file.
- NVIDIA CUDA detection and optional NVENC encoding with verified fallbacks.

## Quick start

Python 3.10 or newer is required.

### Windows

```powershell
git clone https://github.com/billcshi/blur-face.git
cd blur-face
.\init.bat
.\start-ui.bat
```

### Linux / macOS

```bash
git clone https://github.com/billcshi/blur-face.git
cd blur-face
chmod +x init.sh start-ui.sh
./init.sh
./start-ui.sh
```

The setup script creates an isolated `.venv`, installs pinned dependencies,
installs the project in editable mode, and downloads the verified release
models. After that, a normal `git pull` updates the code used by the launchers.

## Local Studio

`start-ui.bat` or `start-ui.sh` starts a small server bound exclusively to
`127.0.0.1` and opens a random-token-protected page in your browser.

The UI provides:

- native local file selection — the browser does not upload the video;
- a model dropdown populated from `models/` and `BLUR_FACE_MODEL_DIR`;
- an additional picker for any other local `.pt` model;
- automatic browser-language detection and persistent manual language choice;
- bilingual hover/focus help for every option;
- privacy presets and full access to mask, blur, threshold, size, tracking,
  overwrite, offline, and encoding controls;
- current frame progress, processing logs, completion state, and cancellation.

Closing the browser page does not cancel a running job. Use the Cancel button
to stop the child process safely and discard its incomplete temporary output.

If port 8765 is busy:

```powershell
.\start-ui.bat --port 8766
```

## Privacy boundary

- Video decoding, detection, tracking, masking, and encoding happen locally.
- The frame-processing loop does not make network calls or upload video.
- If a selected model is missing, model initialization may download a known
  model **before** the first video frame is decoded.
- Enable **Require offline model** in the UI, or pass `--offline`, to reject a
  missing model instead of downloading it.
- The UI listens only on loopback, requires a random token for API calls, has no
  upload control, and loads no remote assets.
- No automatic detector can guarantee perfect recall. Review sensitive output
  before publishing it.

## Privacy presets and masks

| Preset | Shape | Blur | Coverage |
|---|---|---|---:|
| Standard | Rounded rectangle | Adaptive 101–251 | 1.5× |
| Strong | Rounded rectangle | Adaptive 151–301 | 1.65× |
| Strict Privacy | Full rectangle | Fixed 301 | 1.5× |

The default rounded rectangle rounds only the margin outside the detector box.
The complete detected rectangle remains opaque. If the expanded region reaches
a frame boundary, it becomes square rather than cutting into the face.

- `rounded-rect`: safe default with less visual bulk.
- `rectangle`: strictest full expanded-box coverage.
- `ellipse`: legacy appearance; less conservative around corners.

Current raw detections and smoothed/predicted tracking regions are rendered
separately. Smoothing therefore cannot replace or lag behind the current
detector result. The default minimum face size is 30 pixels to avoid turning
tiny noise into visible masks.

## GPU behavior

The project has three independent acceleration paths:

1. **Detection:** YOLO uses PyTorch CUDA when available and prints the GPU,
   PyTorch, and CUDA runtime versions.
2. **Mask rendering:** the standard OpenCV wheel performs blur composition on
   CPU. Seeing `Render: CPU` does not mean face detection is on CPU.
3. **Encoding:** NVENC is used only when the selected FFmpeg passes a real
   runtime probe; otherwise encoding falls back to `libx264`.

On Windows and Linux, `init` checks `nvidia-smi`. NVIDIA systems receive the
tested PyTorch 2.12.1 CUDA 12.6 wheel and setup verifies
`torch.cuda.is_available()`. Set `BLUR_FACE_CPU_ONLY=1` before setup only when
CPU-only installation is intentional.

## CLI

The UI and CLI use the same validated processing pipeline.

```powershell
# Windows
.\.venv\Scripts\blur-face.exe input.mov -o output.mp4 --overwrite

# Select a local model and strict rectangular coverage
.\.venv\Scripts\blur-face.exe input.mov -o output.mp4 `
  --model .\models\yolov11m-face.pt `
  --mask-shape rectangle `
  --blur-strategy fixed `
  --blur-kernel 301 `
  --overwrite
```

```bash
# Linux / macOS
.venv/bin/blur-face input.mov -o output.mp4 --overwrite

# Strictly offline model initialization
.venv/bin/blur-face input.mov -o output.mp4 --offline
```

Useful defaults:

| Option | Default | Purpose |
|---|---:|---|
| `--model` | `yolov11m-face.pt` | Local path or downloadable model name |
| `--thresh` | `0.3` | Detector confidence threshold |
| `--mask-shape` | `rounded-rect` | Coverage geometry |
| `--mask-scale` | `1.5` | Expansion around each coverage region |
| `--blur-strategy` | `adaptive` | Confidence-adaptive or fixed blur |
| `--blur-kernel-min` | `101` | Minimum adaptive blur kernel |
| `--blur-kernel` | `251` | Maximum adaptive or fixed kernel |
| `--min-face-size` | `30` | Minimum detection width and height |
| `--lost-buffer` | `180` | Maximum retained tracking lifetime |
| `--preset` | `quality` | Tracking/optical-flow cost policy |
| `--offline` | off | Prohibit missing-model download |

Run `blur-face --help` for the complete authoritative option list.

Track exclusions are intentionally guarded. IDs are temporary motion tracks,
not biometric identities, so `--exclude-ids` requires the explicit
`--allow-unsafe-exclusions` acknowledgement.

## Safe output behavior

Encoding first writes a hidden temporary file beside the target. The requested
output path is replaced only after FFmpeg exits successfully and the result is
non-empty. Failed or interrupted runs remove the temporary output and return a
non-zero exit status. Existing files require `--overwrite`, and even then remain
untouched until the replacement is complete.

## Models and network access

The installer places verified release models in `models/` using pinned SHA-256
digests and atomic downloads. Model lookup checks:

1. an explicit path;
2. `BLUR_FACE_MODEL_DIR`;
3. the current project's `models/` directory;
4. the installed package project's `models/` directory.

The UI lists local `.pt` files from these model directories. Known missing
release models may be downloaded during initialization unless offline mode is
enabled. No model download occurs after video processing begins.

## Update and uninstall

```powershell
# Update
git pull

# Remove environment and generated caches; keep models and user files
.\uninstall.bat

# Also remove model files
.\uninstall.bat --remove-models
```

```bash
git pull
./uninstall.sh
./uninstall.sh --remove-models
```

## Architecture

```text
Local UI / CLI → AppConfig validation → model initialization
                                           ↓
video → detector → global tracking + optical flow → privacy regions
                                                        ↓
                                                mask renderer
                                                        ↓
                                      atomic FFmpeg encoder → output
```

The package is split into focused modules for configuration, model storage,
video input, detection, tracking, rendering, encoding, pipeline ownership,
console compatibility, and local UI process control.

## Development and CI

```bash
python -m unittest discover -s tests -v
python -m compileall -q blurface tests scripts
```

GitHub Actions tests Python 3.10 and 3.12 on both Windows and Ubuntu. The suite
covers privacy-mask invariants, tracking behavior, model resolution, local UI
request validation, Windows console compatibility, installation scripts, and
atomic encoder failures.

## License

MIT © 2025 Jiechang Shi
