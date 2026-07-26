# Changelog

## 1.0.0 — First stable release

Blur Face Local 1.0 is the first production-ready release. It turns the early
single-script beta into a modular, tested local application.

### Local Studio

- Added a token-protected browser UI bound only to `127.0.0.1`.
- Added automatic browser-language selection and manual `Auto / EN / 中文`
  controls.
- Added local model discovery, a model dropdown, and a native custom-model
  picker.
- Added bilingual hover/focus explanations for every processing option.
- Added privacy presets, live progress, logs, completion state, and safe
  cancellation.

### Privacy and output safety

- Made video processing local-only after model initialization.
- Added explicit offline mode for installations that must never download a
  missing model.
- Added conservative raw-detection and tracked-region coverage.
- Replaced the old default ellipse with a full-face-safe rounded rectangle;
  retained strict rectangle and legacy ellipse options.
- Added stronger confidence-adaptive blur with 101–251 defaults and a fixed
  strategy.
- Restored a conservative 30-pixel minimum face size.
- Added atomic output commit and reliable FFmpeg failure handling.

### Tracking and performance

- Split detection, tracking, rendering, encoding, configuration, model storage,
  and UI process control into focused modules.
- Improved global multi-face association, lost-track retention, and optical-flow
  validation.
- Added verified CUDA PyTorch installation for supported NVIDIA systems.
- Added real NVENC probing with safe `libx264` fallback.

### Installation and quality

- Added Windows and Unix init, UI launcher, and safe uninstall scripts.
- Added verified model downloads with pinned SHA-256 digests.
- Added editable installation so a normal `git pull` updates the application.
- Added Windows/Ubuntu CI across Python 3.10 and 3.12.
- Added tests for privacy coverage, tracking, models, UI, installation,
  encoding, and Windows console behavior.

## 0.1.0-beta — Initial prototype

- Initial YOLO-based face detection and video blurring prototype.
- Early tracking, interpolation, GPU rendering, and FFmpeg encoding support.
