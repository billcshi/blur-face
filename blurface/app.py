"""Application entry point."""

from __future__ import annotations

import sys

from .cli import parse_args


def main() -> int:
    try:
        config = parse_args()
        # Keep argument help and validation available before heavy video/ML
        # dependencies are imported.
        from .pipeline import VideoProcessor

        VideoProcessor(config).run()
        return 0
    except KeyboardInterrupt:
        print("\nCancelled; incomplete output was not committed.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - process boundary reports all failures
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
