#!/usr/bin/env python3
"""Safely remove blur-face's isolated environment and generated caches."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


class UninstallError(RuntimeError):
    """Raised when the requested project root cannot be safely verified."""


def _remove(path: Path, root: Path, removed: list[Path]) -> None:
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise UninstallError(f"refusing to remove path outside project: {path}")
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)
    else:
        return
    removed.append(path)


def uninstall(project_root: Path, remove_models: bool = False) -> list[Path]:
    root = project_root.resolve()
    if not (root / "pyproject.toml").is_file() or not (root / "blur-face.py").is_file():
        raise UninstallError(f"not a verified blur-face project root: {root}")

    removed: list[Path] = []
    for name in (".venv", "build", ".pytest_cache"):
        _remove(root / name, root, removed)
    for path in root.glob("*.egg-info"):
        _remove(path, root, removed)
    for path in sorted(
        root.rglob("__pycache__"), key=lambda item: len(item.parts), reverse=True
    ):
        if ".git" not in path.parts:
            _remove(path, root, removed)
    for path in root.rglob("*.pyc"):
        if ".git" not in path.parts:
            _remove(path, root, removed)
    if remove_models:
        _remove(root / "models", root, removed)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove blur-face's isolated environment and generated caches."
    )
    parser.add_argument(
        "--remove-models",
        action="store_true",
        help="Also permanently remove the downloaded and custom models directory",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    try:
        removed = uninstall(root, remove_models=args.remove_models)
    except (OSError, UninstallError) as exc:
        print(f"[ERROR] uninstall failed: {exc}", file=sys.stderr)
        return 1
    if removed:
        for path in removed:
            print(f"Removed: {path.relative_to(root)}")
    else:
        print("Nothing to remove.")
    print("User videos and output files were not touched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
