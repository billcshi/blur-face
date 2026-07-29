#!/usr/bin/env python3
"""Download release models with pinned SHA-256 verification."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

from blurface.model_store import download_model


DEFAULT_MODELS = ("face_detection_yunet_2023mar.onnx",)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="download the optional third-party YOLO face weight",
    )
    args = parser.parse_args()
    destination = Path(__file__).resolve().parent.parent / "models"
    models = ("yolov11m-face.pt",) if args.yolo else DEFAULT_MODELS
    try:
        for name in models:
            target = download_model(name, destination)
            print(f"[OK] {target.name}")
    except Exception as exc:  # noqa: BLE001 - command boundary reports all failures
        print(f"[ERROR] model download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
