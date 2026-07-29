# Changelog

## 1.1.0 beta 1 — Offline SAM masks

- Replaced Ultralytics/YOLO with the MIT-licensed OpenCV YuNet model and
  removed face parsing and its non-commercial checkpoint path.
- Restored Ultralytics YOLO as an explicitly selected optional detector only:
  base setup remains YuNet-only, init displays the AGPL/Enterprise boundary,
  optional dependencies and a verified face weight use separate installers,
  and video processing never downloads or auto-selects a YOLO weight.
- Documented the pinned `akanametov/yolo-face` 1.0.0 weight's upstream
  GPL-3.0/Enterprise boundary and WIDER FACE training-data source consistently
  across the notices, bilingual README/UI, and both optional installers.
- Show two precomposed YuNet masking modes by default and add two equivalent
  YOLO modes only after a complete optional install. Automatic discovery is
  restricted to the verified face weight, while runtime metadata rejects
  generic/non-detection YOLO models without an explicit `face` class.
- Kept the geometric engine as the default low-overhead single-pass path.
- Simplified Local Studio to Fast geometric and SAM 2.1 high quality modes,
  with engine-specific controls under one expandable advanced section.
- Added union/intersection selection plus a segmentation-only `mask-only`
  policy; failures in all policies retain geometric privacy fallback.
- Added bilingual Local Studio controls and explanations for SAM.
- Added an optional SAM 2.1 high-quality engine using detector-box prompts,
  safety-core union, and geometric fallback.
- Kept SAM 2.1 dependencies out of the default installation and added explicit
  optional installers for evaluation.
- Added short-window SAM 2 video-memory propagation with periodic YuNet
  correction, stable tracker IDs, and geometric fallback.
- Added offline two-pass temporal stabilization for SAM:
  same-scene reverse backfill, motion-aligned hysteresis, scene-cut resets,
  bounded disk-backed masks, and fail-closed geometric rendering.
- Added a bilingual mask-diagnostic video mode that renders the exact final
  coverage in blue on black without retaining source pixels or audio.
- Added a scene-level, cross-Track final-mask contour pass after combination,
  while retaining identity isolation in the per-track/SAM stages.
- Removed numeric Track ID exclusions; a future exclusion workflow requires
  visual track thumbnails instead of temporary ID guessing.
- Added sparse-mask validity gates, automatic CPU fallback, job-owned forced
  cancellation cleanup, actual/peak temporary-storage accounting, monotonic
  multi-stage progress, installation timing, and SAM quality summaries.
- Removed the bundled FFmpeg fallback; encoding now uses an explicitly selected
  or system-provided FFmpeg executable.
- Hardened the post-audit SAM gate against centered one-dimensional and
  under-covering masks, and reject missing or non-finite object scores.
- Added synchronous Web UI shutdown cleanup, full SQLite-inclusive temporary
  storage accounting, clean environment rebuilds, and opt-in Windows FFmpeg
  installation with an external-license notice.
- Throttled SAM progress to roughly two overall-percentage-point milestones
  with timed heartbeats, added whole-job ETA, and labeled the reverse/forward
  Track and final-mask passes. Logs also include rates, active/detected/
  predicted tracks, corrections, prompts, mask validity, and temporary storage.
- Centralized streaming SAM combination in the pipeline so unprompted
  propagation frames apply `intersection` against current Track geometry
  instead of retaining an out-of-Track contour.
- Corrected explicit `cuda:N` YOLO logging to identify the selected GPU and
  made the Windows SAM installer use the project virtual-environment Python
  for all setup timing commands.

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
