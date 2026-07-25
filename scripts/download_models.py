#!/usr/bin/env python3
"""Download release models with pinned SHA-256 verification."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from urllib.request import urlopen

MODELS = {
    "yolo26n-face.pt": (
        (
            "https://github.com/akanametov/yolo-face/releases/download/1.0.0/"
            "yolo26n-face.pt"
        ),
        "c6a5405127a2e351292315a6a8084ea3e790dbec25b9d16a8e80d1e3f866efe1",
    ),
    "yolov11m-face.pt": (
        (
            "https://github.com/akanametov/yolo-face/releases/download/1.0.0/"
            "yolov11m-face.pt"
        ),
        "6ccbe920c1fac95ed84de570519e89fbe24d326d466a7aae297960b3ecc6c661",
    ),
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download(name: str, url: str, expected: str, destination: Path) -> None:
    target = destination / name
    if target.is_file() and digest(target) == expected:
        print(f"[OK] {name}")
        return
    partial = target.with_suffix(target.suffix + ".part")
    partial.unlink(missing_ok=True)
    print(f"[GET] {name}")
    try:
        with urlopen(url, timeout=30) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        actual = digest(partial)
        if actual != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for {name}: expected {expected}, got {actual}"
            )
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def main() -> int:
    destination = Path(__file__).resolve().parent.parent / "models"
    destination.mkdir(exist_ok=True)
    try:
        for name, (url, expected) in MODELS.items():
            download(name, url, expected, destination)
    except Exception as exc:  # noqa: BLE001 - command boundary reports all failures
        print(f"[ERROR] model download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
