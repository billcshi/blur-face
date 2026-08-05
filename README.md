# Blur Face Local

**Offline face anonymization for video and images, with a local browser UI and CLI.**

[中文说明](README.zh.md) · [Version notes](CHANGELOG.md) ·
[Temporal design](docs/temporal-stabilization.md)

Blur Face decodes, analyzes, masks, and writes video or images on the local
computer. The browser page connects only to `127.0.0.1`; media is never uploaded.

## Masking modes

- **Fast geometric** is the default. The selected detector finds faces, the local
  tracker bridges short gaps, and the existing geometric masks render in one
  pass with low startup cost.
- **SAM 2.1 high quality** uses detector boxes for discovery and correction, then
  SAM video memory produces face-shaped masks. It uses an offline two-pass
  pipeline so later detections can repair earlier frames and the final
  cross-track mask can be motion-aligned and stabilized before rendering.

Face parsing/BiSeNet is not a dependency. The default
`face_detection_yunet_2023mar.onnx` detector comes from OpenCV Zoo and is MIT
licensed. OpenCV 4.5+ and official SAM 2 code/checkpoints use Apache-2.0.
Ultralytics YOLO is available only as an explicit optional detector:
Ultralytics is AGPL-3.0 unless covered by an Enterprise license, and the
`akanametov/yolo-face` 1.0.0 weight is offered upstream under GPL-3.0 or an
applicable Enterprise license. Its training recipe identifies WIDER FACE, whose
dataset terms also remain applicable. YOLO is not installed by the base runtime
and is never selected from a `.pt` file automatically.

## Still images and batches

Version 1.2.0 applies the same detection, geometric coverage, blur, preview, and
optional SAM policies to JPEG, PNG, WebP, BMP, and TIFF images. The CLI accepts
one image, multiple image paths, or one non-recursive image directory. Local
Studio has a bilingual **Video / Images** selector and native multi-image and
output-folder pickers. A batch loads the detector and optional SAM model once
and retains only the current image working set.

Each completed image is committed atomically. Without `--overwrite`, a file
that appears during processing is not replaced. Output images intentionally do
not copy EXIF or other source metadata, removing embedded location and camera
details as well. PNG, WebP, and TIFF transparency is preserved; choosing an
opaque output format for an alpha-bearing image fails without committing it.

## Quick start

Python 3.10 or newer and a system FFmpeg executable are required.

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

The setup output reports the duration of environment creation, dependency
installation, the small YuNet download, and the total. Geometric mode does not
install PyTorch. Setup rebuilds the isolated `.venv`, while preserving models
and user files, so dependencies removed by an upgrade cannot remain active.
On Windows, if FFmpeg is missing, `init.bat` discloses that Gyan's full build
is a large external GPL package and asks before invoking WinGet. Unix setup
only reports the system-package prerequisite.

During Windows initialization, setup explicitly asks whether to install the
optional YOLO detector and SAM 2.1 mask engine. Choosing **No** leaves the lean
MIT/Apache-default YuNet geometric runtime. CI skips both prompts. Either choice
can be made later by running:

```powershell
.\install-yolo.bat
.\install-sam2.bat
```

```bash
./install-yolo.sh
```

The optional YOLO installer asks for license acceptance, installs its PyTorch and
Ultralytics dependencies, and downloads the SHA-256-pinned
`yolov11m-face.pt`. Before acceptance it identifies the weight's upstream
GPL-3.0/Enterprise boundary and WIDER FACE training-data source. CI and
non-interactive initialization skip YOLO by default. See
`THIRD_PARTY_NOTICES.md` for exact upstream links.

On Linux/macOS, or to enable SAM later after declining the Windows prompt:

```powershell
.\install-sam2.bat
```

```bash
./install-sam2.sh
```

The SAM installer selects the tested CUDA runtime when NVIDIA is available and
otherwise installs the CPU runtime. A named SAM checkpoint may download during
model initialization, before analysis starts. Later jobs try the complete local
cache first, so they do not contact the Hub merely to check for updates. Use
offline mode to require cached files, or select a local model directory. Each
UI job is a separate process and therefore loads the weights once per job, while
an image batch reuses that one loaded model for every image.

Public Hugging Face checkpoints do not require an account. An optional
`.venv\Scripts\hf.exe auth login` on Windows (or `.venv/bin/hf auth login` on
Linux/macOS) removes the unauthenticated first-download notice and raises Hub
rate limits; credentials are handled by the Hugging Face client, not this app.

A normal `git pull` does not require reinitialization unless dependency lock
files or the setup scripts changed.

## Local Studio

The main form exposes media type, input, output, masking mode, and privacy
preset. Image mode accepts one or many local paths and an output folder, while
video mode retains its single input/output workflow. Without optional YOLO it
shows two YuNet modes: Fast geometric
and SAM high quality. After a complete YOLO installation, the same selector
adds Fast geometric · YOLO and SAM · YOLO. **Advanced settings** expands the
controls that apply to the selected engine:

- the model matching the detector already named by the selected mode;
- common blur, tracking, output, and diagnostic settings;
- geometric/fallback shape and coverage;
- SAM model, device, correction interval, mask combination, temporal window,
  scene-cut sensitivity, and temporary-storage limit.

Privacy presets never change the selected masking engine. The UI supports
automatic English/Chinese selection and manual `Auto / EN / 中文` switching.
Automatic YOLO discovery is intentionally limited to the verified
`yolov11m-face.pt`; arbitrary local YOLO weights must also pass a runtime
`detect` task and explicit `face` class check.

Enable **Mask diagnostic video / 遮挡区域测试视频** to produce a black video
whose blue pixels are the exact final region that normal output would blur.
Source pixels, audio, boxes, IDs, and labels are excluded.

Progress is one monotonic percentage across analysis, per-track stabilization,
final scene-mask stabilization, and render/encode. SAM logs include model
initialization time, device, video metadata, scenes, corrections, prompts,
accepted/fallback masks, reverse backfill, memory resets, temporary-storage
peak, pass-specific work counters, whole-job ETA, and per-phase timing.

## SAM mask semantics

Temporal stabilization happens before tracks are merged, then again on the
identity-free final scene mask:

1. analyze frames and persist bounded proxy-resolution masks;
2. repair and stabilize each `(scene_id, track_id)`;
3. apply the selected SAM/geometry combination policy;
4. merge tracks and stabilize the final scene mask;
5. reopen the source and render the completed mask sequence.

Combination modes:

- `union` adds current tracker geometry to the SAM contour and is recommended;
- `intersection` clips the SAM contour to current tracker geometry;
- `mask-only` renders a valid SAM contour without adding the box.

Sparse, one-dimensional, under-covering, fragmented, empty, missing/non-finite
score, failed, or drifted SAM masks use the configured geometric fallback in
every combination mode. Scene cuts clear the
tracker, SAM memory, reverse cache, and temporal history. Ambiguous crossings
are not used to seed identity-owned SAM memory.

By default, a newly reliable face can backfill up to 10 frames in the same
continuous shot. Reverse propagation stops at cuts, real edge entry, implausible
motion or scale, weak appearance agreement, or overlap with another person.
Temporary proxy frames, masks, and SQLite metadata have a 4096 MiB hard limit
by default and are removed on success, failure, or UI cancellation.

## CLI

```powershell
.\.venv\Scripts\blur-face.exe input.mov -o output.mp4 --overwrite

.\.venv\Scripts\blur-face.exe input.mov -o output.mp4 `
  --mask-engine sam2.1 `
  --detector yolo `
  --model yolov11m-face.pt `
  --segmentation-combine union `
  --device auto `
  --overwrite
```

```bash
.venv/bin/blur-face input.mov -o output.mp4 --overwrite

.venv/bin/blur-face portrait.jpg -o portrait_blurred.jpg
.venv/bin/blur-face first.jpg second.png -o blurred-images/
.venv/bin/blur-face photos/ -o blurred-images/
```

Useful defaults:

| Option | Default | Purpose |
|---|---:|---|
| `--detector` | `yunet` | `yunet` or explicitly installed `yolo` |
| `--model` | `face_detection_yunet_2023mar.onnx` | Local YuNet ONNX model |
| `--thresh` | `0.3` | Detector confidence threshold |
| `--mask-engine` | `geometric` | `geometric` or `sam2.1` |
| `--mask-shape` | `rounded-rect` | Geometry and SAM fallback shape |
| `--mask-scale` | `1.5` | Geometry and fallback expansion |
| `--segmentation-combine` | `union` | `union`, `intersection`, or `mask-only` |
| `--sam-mask-expansion` | `0.12` | SAM contour expansion |
| `--sam2-model` | `facebook/sam2.1-hiera-base-plus` | SAM directory or model ID |
| `--sam2-refresh-interval` | `15` | Selected-detector correction interval |
| `--device` | `auto` | SAM/YOLO device: auto, CPU, CUDA, or MPS |
| `--[no-]temporal-stabilization` | on | Offline two-pass stabilization |
| `--backfill-frames` | `10` | Same-scene reverse repair window |
| `--release-hold-frames` | `5` | Aligned contour transition window |
| `--scene-cut-sensitivity` | `0.55` | Scene reset sensitivity |
| `--temporal-storage-limit-mb` | `4096` | Temporary storage hard limit |
| `--mask-preview` | off | Black/blue final-mask diagnostic video |

Run `blur-face --help` for all tracking, blur, encoding, and offline options.

## Privacy and output safety

- No automatic detector guarantees perfect recall; review sensitive output.
- Model initialization completes before the first frame is processed.
- Incomplete encoding never replaces the destination. Output is written to a
  job-owned partial path and atomically moved into place only after FFmpeg
  succeeds and produces a non-empty file.
- Image output follows the same no-partial-output rule per file. A failed or
  cancelled batch keeps completed images but never commits the current
  incomplete image.
- The UI owns a unique job directory on the destination filesystem (inside an
  image output folder). Normal exceptions, cancellation, and forced
  process-tree termination all trigger cleanup.
- FFmpeg is an external system prerequisite and is not bundled by this
  project. Its license depends on the build selected by the user.

## Development and CI

```bash
python -m unittest discover -s tests -v
python -m compileall -q blurface tests scripts
git diff --check
```

GitHub Actions runs unittest discovery and compile checks on Ubuntu and Windows
with Python 3.10 and 3.12. New `tests/test_*.py` files are included
automatically. CI explicitly provisions an external system FFmpeg before the
video integration tests. The default matrix remains model-free. Maintainers
can additionally set `BLUR_FACE_REAL_SAM2_MODEL` to a local checkpoint
directory when running unittest discovery to exercise the installed
Transformers processor/model/session contract on CPU.

## License

Project-owned source code is MIT © 2025 Jiechang Shi. Dependencies and model
assets keep their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
