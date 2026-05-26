# Release Notes

## v1.0.0

First standalone release.

### Downloads

| Version | Size | GPU Required? |
|---------|------|---------------|
| `blur-face-v1.0.0-win64.zip` | ~2-3 GB | NVIDIA GPU recommended |
| `blur-face-v1.0.0-win64-cpu.zip` | ~300 MB | No |

### What's New

- **Standalone executable** — no Python required, just unzip and run
- **Drag & drop** — drag a video file onto `blur-face.exe` to blur all faces
- **GPU-accelerated** YOLO face detection (NVIDIA GPU with updated drivers)
- **NVENC hardware encoding** for fast video output
- **Optical flow tracking** — faces stay blurred even when YOLO misses
- **Two presets**: `--preset quality` (default) and `--preset fast`
- **CPU fallback** — runs without GPU (slower but works everywhere)

### How to Use

1. Download the zip file for your system
2. Unzip anywhere
3. Drag a video file onto `blur-face.exe`
4. Find the blurred video as `output_blur.mp4`

### Advanced Usage

```
blur-face.exe input.mp4 -o output.mp4
blur-face.exe input.mp4 --preset fast        # faster, less tracking overhead
blur-face.exe input.mp4 --debug               # show boxes without blurring
blur-face.exe input.mp4 --no-flow             # disable optical flow
blur-face.exe input.mp4 --model models\yolo26n-face.pt  # nano model
blur-face.exe --help                          # all options
```

### System Requirements

**GPU Edition:**
- Windows 10/11 64-bit
- NVIDIA GPU with updated drivers (GTX 10xx+)
- NVENC-capable GPU for hardware encoding

**CPU Edition:**
- Windows 10/11 64-bit
- No special hardware required
- Expect 3-10× slower processing vs GPU edition

### Notes

- The first run may trigger Windows SmartScreen; click "More info" → "Run anyway"
- Some antivirus software may flag PyInstaller executables; this is a false positive
- For best results, use videos with clearly visible faces and good lighting
