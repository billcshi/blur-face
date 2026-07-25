# blur-face

Privacy-oriented local video face detection, tracking, and blurring.

blur-face uses a YOLO face detector on every frame, conservative multi-face
tracking, optical flow during short detection gaps, and an atomic FFmpeg output
pipeline. It is designed to fail visibly instead of reporting success for a
missing or incomplete output.

## Privacy and network behavior

- Video frames are decoded, detected, blurred, and encoded locally. The
  processing loop does not upload video or make network requests.
- Model initialization may download a missing YOLO model before the first video
  frame is decoded.
- Use `--offline` to require an already installed local model and prohibit that
  automatic download path.
- No automatic detector can guarantee that every face is found. Review the
  result before sharing sensitive material.

## Install

Python 3.10 or newer is required.

```bash
# Linux / macOS
chmod +x init.sh
./init.sh

# Windows
init.bat
```

The setup scripts create an isolated `.venv`, install the versions in
`requirements.lock`, and download two release models. Downloads are written
atomically and verified against pinned SHA-256 digests.

## Use

```bash
# Linux / macOS
.venv/bin/blur-face input.mov -o output.mp4

# Windows
.venv\Scripts\blur-face.exe input.mov -o output.mp4

# Strictly local model initialization
.venv/bin/blur-face input.mov -o output.mp4 --offline

# Use a specific Windows FFmpeg build with NVENC
.venv\Scripts\blur-face.exe input.mov -o output.mp4 --ffmpeg C:\ffmpeg\bin\ffmpeg.exe

# Review conservative tracking coverage without applying blur
.venv/bin/blur-face input.mov --debug -o review.mp4

# Use different confidence thresholds over time
.venv/bin/blur-face input.mov --time-thresh "0:0.15,120:0.3"
```

The legacy `python blur-face.py ...` launcher remains available when the
dependencies are active in the current interpreter.

## Important behavior

The render region for a detected face is the union of its current raw detection
and its smoothed track. Smoothing can therefore add coverage but cannot replace
or lag behind the current detection. Close-up faces are accepted by default,
and filtered detections are limited to invalid, tiny, or extremely distorted
boxes.

`--exclude-ids` is intended for carefully reviewed material only. Track IDs are
temporary motion tracks, not biometric identities, and can change when people
cross or are re-detected. It is rejected unless
`--allow-unsafe-exclusions` explicitly acknowledges that risk, and the program
still prints a warning whenever exclusions are used.

Outputs are first encoded to a hidden temporary file in the target directory.
The target path is replaced only after FFmpeg exits successfully and the output
is non-empty. Interrupted or failed runs remove the temporary output and return
a non-zero exit status. Existing outputs are refused unless `--overwrite` is
explicitly supplied; even then the old file remains intact until the new one is
complete.

## Options

Run `blur-face --help` for the authoritative list. Key defaults:

| Option | Default | Meaning |
|---|---:|---|
| `--model` | `yolov11m-face.pt` | Local path or downloadable model name |
| `--thresh` | `0.3` | YOLO confidence threshold, 0–1 |
| `--mask-scale` | `1.35` | Expansion around the conservative track region |
| `--blur-kernel` | `51` | Positive Gaussian kernel; even values are normalized |
| `--lost-buffer` | `180` | Frames retained after a missed detection |
| `--smooth` | `0.7` | Detection weight used for EMA smoothing |
| `--min-face-size` | `8` | Minimum detection width and height in pixels |
| `--max-face-height-ratio` | `1.0` | Maximum detection height relative to frame |
| `--preset` | `quality` | Optical-flow cost policy |
| `--offline` | off | Reject a missing local model instead of downloading |
| `--ffmpeg` | system/bundled | Explicit FFmpeg executable |

All safety-sensitive numeric options are validated before a model is loaded or
an output process is started.

## Architecture

```text
CLI → AppConfig validation → VideoSource probe
                              ↓
                    model initialization
                              ↓
frame → detector → global association + optical flow → privacy coverage
                              ↓
                    CPU/CUDA renderer
                              ↓
                atomic FFmpeg encoder → output
```

- `config.py`: typed configuration and invariants
- `video.py`: validated OpenCV input lifecycle
- `detector.py`: local-first YOLO model initialization
- `tracker.py`: global assignment, track state, conservative coverage
- `renderer.py`: validated CPU/CUDA blur backends
- `encoder.py`: runtime NVENC probe, checked FFmpeg, atomic commit
- `pipeline.py`: resource ownership and frame orchestration
- `app.py`: process-level errors and exit codes

## Development

```bash
python -m unittest discover -s tests -v
python -m compileall -q blurface tests
```

The test suite covers privacy coverage invariants, close-up detections,
order-independent assignment, stale optical-flow invalidation, configuration
validation, and FFmpeg failure handling.

## License

MIT © 2025 Jiechang Shi

中文说明：[README.zh.md](README.zh.md)
