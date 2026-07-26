#!/usr/bin/env python3
"""Download release models with pinned SHA-256 verification."""

from __future__ import annotations

import sys
from pathlib import Path

from blurface.model_store import MODEL_SPECS, download_model


def main() -> int:
    destination = Path(__file__).resolve().parent.parent / "models"
    try:
        for name in MODEL_SPECS:
            target = download_model(name, destination)
            print(f"[OK] {target.name}")
    except Exception as exc:  # noqa: BLE001 - command boundary reports all failures
        print(f"[ERROR] model download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
