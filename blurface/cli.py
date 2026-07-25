"""Command-line parsing for blur-face."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .config import AppConfig, ConfigError


def parse_time_thresh(raw: str) -> tuple[tuple[float, float], ...]:
    """Parse ``sec:threshold,sec:threshold`` into a sorted tuple."""
    if not raw:
        return ()
    segments: list[tuple[float, float]] = []
    try:
        for part in raw.split(","):
            fields = part.strip().split(":")
            if len(fields) != 2:
                raise ValueError
            segments.append((float(fields[0]), float(fields[1])))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            'expected "sec:threshold,sec:threshold"'
        ) from exc
    return tuple(sorted(segments))


def parse_exclude_ids(raw: str) -> frozenset[int]:
    if not raw:
        return frozenset()
    try:
        return frozenset(
            int(value.strip()) for value in raw.split(",") if value.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integer IDs"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blur-face",
        description="Local video face blur with detection, tracking, and optical flow.",
    )
    parser.add_argument("input", type=Path, help="Input video path")
    parser.add_argument("-o", "--output", type=Path, default=Path("output_blur.mp4"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing output after successful encoding",
    )
    parser.add_argument("--model", default="yolov11m-face.pt")
    parser.add_argument("--thresh", type=float, default=0.3)
    parser.add_argument("--time-thresh", type=parse_time_thresh, default=())
    parser.add_argument("--mask-scale", type=float, default=1.35)
    parser.add_argument("--blur-kernel", type=int, default=51)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        help="FFmpeg executable (default: system FFmpeg, then bundled fallback)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Draw track coverage; do not blur"
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--exclude-ids", type=parse_exclude_ids, default=frozenset())
    parser.add_argument(
        "--allow-unsafe-exclusions",
        action="store_true",
        help="Acknowledge that temporary track IDs can switch between people",
    )
    parser.add_argument("--lost-buffer", type=int, default=180)
    parser.add_argument("--smooth", type=float, default=0.7)
    parser.add_argument("--preset", choices=("quality", "fast"), default="quality")
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=8,
        help="Minimum detection width/height in pixels (default: 8)",
    )
    parser.add_argument(
        "--max-face-height-ratio",
        type=float,
        default=1.0,
        help="Maximum face height as a fraction of frame height (default: 1.0)",
    )
    parser.add_argument("--flow-max-points", type=int, default=50)
    parser.add_argument("--flow-min-confirmations", type=int, default=3)
    parser.add_argument("--flow-max-missed", type=int, default=0)
    parser.add_argument("--no-flow", action="store_true")
    parser.add_argument("--no-nvenc", action="store_true")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require a local model; never ask YOLO to download one",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> AppConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    flow_max_points = args.flow_max_points
    flow_max_missed = args.flow_max_missed
    if args.preset == "fast":
        if flow_max_points == 50:
            flow_max_points = 20
        if flow_max_missed == 0:
            flow_max_missed = 45
    config = AppConfig(
        input=args.input,
        output=args.output,
        overwrite=args.overwrite,
        model=args.model,
        threshold=args.thresh,
        time_thresholds=args.time_thresh,
        mask_scale=args.mask_scale,
        blur_kernel=args.blur_kernel,
        device=args.device,
        ffmpeg=args.ffmpeg,
        debug=args.debug,
        profile=args.profile,
        exclude_ids=args.exclude_ids,
        allow_unsafe_exclusions=args.allow_unsafe_exclusions,
        lost_buffer=args.lost_buffer,
        smooth=args.smooth,
        preset=args.preset,
        min_face_size=args.min_face_size,
        max_face_height_ratio=args.max_face_height_ratio,
        flow_max_points=flow_max_points,
        flow_min_confirmations=args.flow_min_confirmations,
        flow_max_missed=flow_max_missed,
        flow_enabled=not args.no_flow,
        use_nvenc=not args.no_nvenc,
        offline=args.offline,
    )
    try:
        return config.validate()
    except ConfigError as exc:
        parser.error(str(exc))


# Compatibility helpers for callers of the original module API.
def get_thresh(args, frame_idx: int, fps: float) -> float:
    if isinstance(args, AppConfig):
        return args.threshold_for(frame_idx, fps)
    if not args.time_thresh:
        return args.thresh
    if fps <= 0:
        raise ConfigError("video FPS must be positive when --time-thresh is used")
    threshold = args.thresh
    for second, candidate in args.time_thresh:
        if frame_idx / fps >= second:
            threshold = candidate
    return threshold
