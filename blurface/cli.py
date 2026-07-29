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
    parser.add_argument(
        "--detector",
        choices=("yunet", "yolo"),
        default="yunet",
        help=(
            "Face detector backend: bundled OpenCV YuNet, or explicitly "
            "installed optional Ultralytics YOLO"
        ),
    )
    parser.add_argument(
        "--model",
        default="face_detection_yunet_2023mar.onnx",
        help="Local detector model (.onnx for YuNet, .pt for YOLO)",
    )
    parser.add_argument("--thresh", type=float, default=0.3)
    parser.add_argument("--time-thresh", type=parse_time_thresh, default=())
    parser.add_argument("--mask-scale", type=float, default=1.5)
    parser.add_argument(
        "--mask-shape",
        choices=("rounded-rect", "rectangle", "ellipse"),
        default="rounded-rect",
        help="Coverage shape (default: rounded-rect)",
    )
    parser.add_argument(
        "--mask-engine",
        choices=("geometric", "sam2.1"),
        default="geometric",
        help="Fast geometric coverage or high-quality SAM 2.1",
    )
    parser.add_argument(
        "--sam-mask-expansion",
        type=float,
        default=0.12,
        help="Face-width fraction added around a SAM mask (default: 0.12)",
    )
    parser.add_argument(
        "--segmentation-combine",
        choices=("union", "intersection", "mask-only"),
        default="union",
        help=(
            "Union/intersection with detector coverage, or mask-only to use "
            "the segmentation contour (failures still use geometry)"
        ),
    )
    parser.add_argument(
        "--sam2-model",
        default="facebook/sam2.1-hiera-base-plus",
        help="Local Transformers model directory or SAM 2.1 model ID",
    )
    parser.add_argument(
        "--sam2-refresh-interval",
        type=int,
        default=15,
        help="Face-detector correction interval for SAM tracking (default: 15)",
    )
    parser.add_argument(
        "--temporal-stabilization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Offline bidirectional mask stabilization for segmentation engines "
            "(slower; uses bounded temporary storage)"
        ),
    )
    parser.add_argument(
        "--backfill-frames",
        type=int,
        default=10,
        help="Maximum same-scene reverse repair window (default: 10)",
    )
    parser.add_argument(
        "--release-hold-frames",
        type=int,
        default=5,
        help="Frames to blend aligned mask changes and geometry (default: 5)",
    )
    parser.add_argument(
        "--scene-cut-sensitivity",
        type=float,
        default=0.55,
        help="Scene-cut sensitivity from 0 to 1 (default: 0.55)",
    )
    parser.add_argument(
        "--blur-strategy",
        choices=("adaptive", "fixed"),
        default="adaptive",
        help="Scale blur with face confidence, or use one fixed kernel",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=251,
        help="Maximum adaptive kernel, or kernel used by the fixed strategy",
    )
    parser.add_argument(
        "--blur-kernel-min",
        type=int,
        default=101,
        help="Minimum kernel used by the adaptive strategy",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="SAM compute device; auto uses CUDA when available, otherwise CPU",
    )
    parser.add_argument(
        "--temporal-storage-limit-mb",
        type=int,
        default=4096,
        help="Maximum temporary storage for two-pass masks (default: 4096 MiB)",
    )
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        help="FFmpeg executable (default: system FFmpeg on PATH)",
    )
    parser.add_argument(
        "--job-temp-dir",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--debug", action="store_true", help="Draw track coverage; do not blur"
    )
    parser.add_argument(
        "--mask-preview",
        action="store_true",
        help=(
            "Write a black diagnostic video with the exact final mask in blue; "
            "do not render source pixels"
        ),
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--lost-buffer", type=int, default=180)
    parser.add_argument("--smooth", type=float, default=0.7)
    parser.add_argument("--preset", choices=("quality", "fast"), default="quality")
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=30,
        help="Minimum detection width/height in pixels (default: 30)",
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
        help="Require all selected models to exist locally",
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
        detector=args.detector,
        model=args.model,
        threshold=args.thresh,
        time_thresholds=args.time_thresh,
        mask_scale=args.mask_scale,
        mask_shape=args.mask_shape,
        mask_engine=args.mask_engine,
        sam_mask_expansion=args.sam_mask_expansion,
        segmentation_combine=args.segmentation_combine,
        sam2_model=args.sam2_model,
        sam2_refresh_interval=args.sam2_refresh_interval,
        temporal_stabilization=args.temporal_stabilization,
        backfill_frames=args.backfill_frames,
        release_hold_frames=args.release_hold_frames,
        scene_cut_sensitivity=args.scene_cut_sensitivity,
        temporal_storage_limit_mb=args.temporal_storage_limit_mb,
        blur_strategy=args.blur_strategy,
        blur_kernel=args.blur_kernel,
        blur_kernel_min=args.blur_kernel_min,
        device=args.device,
        ffmpeg=args.ffmpeg,
        job_temp_dir=args.job_temp_dir,
        debug=args.debug,
        mask_preview=args.mask_preview,
        profile=args.profile,
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
