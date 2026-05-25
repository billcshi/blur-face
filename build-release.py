#!/usr/bin/env python3
"""
build-release.py — Build a standalone blur-face release with PyInstaller.

Usage:
  python build-release.py              # CUDA build (uses current env's torch)
  python build-release.py --cpu        # CPU-only build (auto-creates venv)
  python build-release.py --no-zip     # skip zip step
  python build-release.py --clean      # clean dist/ and rebuild

Requires: pyinstaller (pip install pyinstaller), or let --cpu auto-setup
"""
import os
import sys
import shutil
import zipfile
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"
SPEC = ROOT / "blur-face.spec"
MODELS_DIR = ROOT / "models"
VERSION = "1.0.0"

# Default model to bundle in the release
DEFAULT_MODEL = "yolov11m-face.pt"


def run(cmd, **kwargs):
    pretty = " ".join(str(c) for c in cmd)
    print(f"  $ {pretty}")
    subprocess.run(cmd, check=True, **kwargs)


def ensure_model():
    """Ensure the default model exists."""
    model_path = MODELS_DIR / DEFAULT_MODEL
    if model_path.exists():
        print(f"[OK] Model found: {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")
        return model_path
    print(f"[ERROR] Model not found: {model_path}")
    print(f"  Please download {DEFAULT_MODEL} (38.6 MB) from:")
    print(f"  https://github.com/akanametov/yolo-face/releases")
    print(f"  and place it in {MODELS_DIR}/")
    sys.exit(1)


def clean():
    """Remove previous build artifacts."""
    for d in [DIST, BUILD]:
        if d.exists():
            print(f"[CLEAN] Removing {d}")
            shutil.rmtree(d)
    for f in ROOT.glob("blur-face-v*-win64*.zip"):
        print(f"[CLEAN] Removing {f}")


def build_pyinstaller(python_exe=None):
    """Run PyInstaller with the spec file.

    Args:
        python_exe: Path to python executable (for venv builds).
    """
    py = python_exe or sys.executable
    print("\n[BUILD] Running PyInstaller...")
    run([py, "-m", "PyInstaller", str(SPEC)], cwd=str(ROOT))


def create_release_dir(suffix=""):
    """Set up the release folder with model and README.

    Args:
        suffix: Appended to the release folder name (e.g. '-cpu').
    """
    src_dir = DIST / "blur-face"
    release_dir = DIST / f"blur-face{suffix}"

    if not src_dir.exists():
        print(f"[ERROR] PyInstaller output not found: {src_dir}")
        sys.exit(1)

    # PyInstaller always outputs to the same name; rename it
    if release_dir.exists():
        shutil.rmtree(release_dir)
    src_dir.rename(release_dir)

    # Copy model into release folder
    model_src = MODELS_DIR / DEFAULT_MODEL
    model_dst = release_dir / "models" / DEFAULT_MODEL
    model_dst.parent.mkdir(exist_ok=True)
    if not model_dst.exists():
        print(f"[COPY] {model_src} → {model_dst}")
        shutil.copy2(model_src, model_dst)

    # Also copy nano model if available
    nano_model = MODELS_DIR / "yolo26n-face.pt"
    if nano_model.exists():
        nano_dst = release_dir / "models" / "yolo26n-face.pt"
        if not nano_dst.exists():
            print(f"[COPY] {nano_model} → {nano_dst}")
            shutil.copy2(nano_model, nano_dst)

    # Create README
    is_cpu = suffix == "-cpu"
    readme = release_dir / "README.txt"
    readme_text = f"""blur-face v{VERSION} — Face Blur Tool{" (CPU Edition)" if is_cpu else ""}
====================================={'=' * (18 if is_cpu else 0)}

HOW TO USE:
  Drag and drop a video file onto blur-face.exe
  The blurred video will be saved as output_blur.mp4

ADVANCED (Command Prompt / PowerShell):
  blur-face.exe input.mp4 -o output.mp4
  blur-face.exe input.mp4 --preset fast     (faster, slightly less tracking)
  blur-face.exe input.mp4 --debug            (show boxes, no blur)
  blur-face.exe input.mp4 --model models\\yolo26n-face.pt  (use nano model)

MODELS:
  models\\yolov11m-face.pt  — medium model (best quality, included)
  models\\yolo26n-face.pt   — nano model (faster, lower quality)
"""
    if is_cpu:
        readme_text += """
PERFORMANCE NOTE (CPU Edition):
  Processing runs entirely on CPU — expect ~3-10x slower than GPU.
  For best speed: use --preset fast and the nano model.
  NVIDIA GPU users: download the standard edition instead.
"""
    else:
        readme_text += """
REQUIREMENTS:
  - NVIDIA GPU with updated drivers for YOLO detection (auto-falls back to CPU)
  - NVENC-capable GPU for hardware encoding (auto-detected)
"""
    readme_text += "\nFor more options: blur-face.exe --help\n"
    readme.write_text(readme_text, encoding="utf-8")

    return release_dir


def create_zip(release_dir, suffix=""):
    """Package release folder into a zip."""
    zip_name = f"blur-face-v{VERSION}-win64{suffix}.zip"
    zip_path = ROOT / zip_name

    print(f"\n[ZIP] Creating {zip_name}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(release_dir):
            for file in files:
                full = Path(root) / file
                arcname = full.relative_to(DIST)
                zf.write(full, arcname)

    size_mb = zip_path.stat().st_size / 1e6
    print(f"[ZIP] Created {zip_name} ({size_mb:.0f} MB)")
    return zip_path


# ── CPU build via temporary venv ──────────────────────────────


def _build_cpu():
    """Build a CPU-only release using a temporary venv.

    Creates a venv, installs CPU PyTorch + deps, runs PyInstaller,
    then cleans up. The user's main environment is never touched.
    """
    print("\n[CPU] Setting up temporary venv for CPU-only build...")
    venv_dir = Path(tempfile.mkdtemp(prefix="blur-face-cpu-"))
    venv_python = venv_dir / "Scripts" / "python.exe"

    try:
        # Create venv
        run([sys.executable, "-m", "venv", str(venv_dir)])
        print(f"  [OK] Created venv: {venv_dir}")

        # Install deps
        print("\n[CPU] Installing CPU PyTorch... (~200 MB, one-time)")
        run([str(venv_python), "-m", "pip", "install", "--quiet",
             "torch", "torchvision",
             "--index-url", "https://download.pytorch.org/whl/cpu"])

        print("\n[CPU] Installing packages...")
        # Install remaining deps (torch/torchvision already installed)
        cpu_deps = [
            "pyinstaller",
            "opencv-python",
            "numpy",
            "imageio-ffmpeg",
            "ultralytics",
        ]
        for pkg in cpu_deps:
            run([str(venv_python), "-m", "pip", "install", "--quiet", pkg])

        # Build
        build_pyinstaller(python_exe=str(venv_python))

        # Package
        release_dir = create_release_dir(suffix="-cpu")
        return release_dir

    finally:
        print(f"\n[CPU] Cleaning up venv: {venv_dir}")
        shutil.rmtree(venv_dir, ignore_errors=True)


# ── Main ─────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(description="Build blur-face release")
    p.add_argument("--cpu", action="store_true",
                   help="Build CPU-only release (auto-creates temp venv)")
    p.add_argument("--no-zip", action="store_true", help="Skip zip step")
    p.add_argument("--clean", action="store_true", help="Clean before build")
    args = p.parse_args()

    print("=" * 50)
    print(f"  blur-face Release Builder v{VERSION}")
    print(f"  {'CPU-only' if args.cpu else 'CUDA'} build")
    print("=" * 50)

    if args.clean:
        clean()

    ensure_model()

    suffix = ""

    if args.cpu:
        release_dir = _build_cpu()
        suffix = "-cpu"
    else:
        # Verify CUDA is available for GPU build
        try:
            import torch
            if not torch.cuda.is_available():
                print("[WARN] CUDA not available in current environment!")
                print("       Use --cpu for a CPU-only build instead.")
                sys.exit(1)
        except ImportError:
            print("[ERROR] torch not installed. Use --cpu for auto-setup.")
            sys.exit(1)
        build_pyinstaller()
        release_dir = create_release_dir()

    if not args.no_zip:
        zip_path = create_zip(release_dir, suffix)
        size_mb = zip_path.stat().st_size / 1e6
        print(f"\n[DONE] Release built: {zip_path}")
        print(f"       Size: {size_mb:.0f} MB")
    else:
        print(f"\n[DONE] Release folder: {release_dir}")

    print("\nTo test:")
    print(f'  "{release_dir / "blur-face.exe"}" --help')


if __name__ == "__main__":
    main()
