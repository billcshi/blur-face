"""Local-first model resolution with verified downloads for supported models."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.request import urlopen

OPTIONAL_YOLO_MODELS = frozenset({"yolov11m-face.pt"})

MODEL_SPECS = {
    "face_detection_yunet_2023mar.onnx": (
        (
            "https://github.com/opencv/opencv_zoo/raw/"
            "f12e12798e8314f7c074a6656816c048dcc95b7a/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    # Optional only. This third-party weight comes from a fork of Ultralytics
    # and is not part of the MIT default runtime. It is downloaded only by the
    # explicit YOLO installer after its upstream license notice is accepted.
    "yolov11m-face.pt": (
        (
            "https://github.com/akanametov/yolo-face/releases/download/1.0.0/"
            "yolov11m-face.pt"
        ),
        "6ccbe920c1fac95ed84de570519e89fbe24d326d466a7aae297960b3ecc6c661",
    ),
}


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def model_directories() -> tuple[Path, ...]:
    directories: list[Path] = []
    configured = os.environ.get("BLUR_FACE_MODEL_DIR")
    if configured:
        directories.append(Path(configured).expanduser())
    directories.append(Path.cwd() / "models")
    directories.append(Path(__file__).resolve().parent.parent / "models")
    return tuple(dict.fromkeys(path.resolve() for path in directories))


def download_model(name: str, destination: Path) -> Path:
    try:
        url, expected_digest = MODEL_SPECS[name]
    except KeyError as exc:
        raise ValueError(
            f"no verified download is configured for model: {name}"
        ) from exc
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / name
    if target.is_file() and file_digest(target) == expected_digest:
        return target

    partial = target.with_suffix(target.suffix + ".part")
    partial.unlink(missing_ok=True)
    print(f"[Model] Downloading {name} before video processing...")
    try:
        with urlopen(url, timeout=30) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        actual_digest = file_digest(partial)
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"SHA-256 mismatch for {name}: "
                f"expected {expected_digest}, got {actual_digest}"
            )
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return target


def resolve_model(
    path: str | Path,
    allow_download: bool = True,
    search_directories: tuple[Path, ...] | None = None,
) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())

    name = candidate.name
    directories = search_directories or model_directories()
    for directory in directories:
        stored = directory / name
        if stored.is_file():
            return str(stored.resolve())

    if allow_download and name in MODEL_SPECS:
        destination = directories[0] if directories else Path.cwd() / "models"
        return str(download_model(name, destination))
    if allow_download:
        raise ValueError(f"no verified download is configured for model: {name}")
    raise FileNotFoundError(
        f"model not found locally: {path} (remove --offline to allow download)"
    )
