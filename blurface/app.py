"""Application entry point."""

from __future__ import annotations

import sys
import traceback

from .cli import parse_args


def _configure_console_stream(stream) -> None:
    """Keep Windows legacy consoles from crashing on paths they cannot encode."""
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass


def main() -> int:
    _configure_console_stream(sys.stdout)
    _configure_console_stream(sys.stderr)
    try:
        config = parse_args()
        # Keep argument help and validation available before heavy media/ML
        # dependencies are imported.
        from .config import ImageBatchConfig

        if isinstance(config, ImageBatchConfig):
            from .image_pipeline import ImageProcessor

            ImageProcessor(config).run()
        else:
            from .pipeline import VideoProcessor

            VideoProcessor(config).run()
        return 0
    except KeyboardInterrupt:
        print("\nCancelled; incomplete output was not committed.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - process boundary reports all failures
        print(f"[ERROR] {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1
