"""
blurface.cli — Argument parsing and config.
"""
import argparse


def parse_args():
    p = argparse.ArgumentParser(
        prog="blur-face",
        description="Face blur with tracking + optical flow — YOLO detection + custom tracker."
    )
    p.add_argument("input", help="Input video path")
    p.add_argument("-o", "--output", default="output_blur.mp4", help="Output path")
    p.add_argument("--model", default="yolov11m-face.pt",
        help="Model file (yolo26n-face.pt / yolo26m-face.pt / yolov11m-face.pt)")
    p.add_argument("--thresh", type=float, default=0.3,
        help="Detection threshold (0-1, lower = more sensitive)")
    p.add_argument("--time-thresh", type=str, default="",
        help='Time-based thresholds: "sec:thresh,sec:thresh"')
    p.add_argument("--mask-scale", type=float, default=1.35,
        help="Scale factor for blur region")
    p.add_argument("--blur-kernel", type=int, default=51,
        help="Gaussian blur kernel size (odd, larger = more blur)")
    p.add_argument("--device", default="cuda",
        help="cuda or cpu")
    p.add_argument("--debug", action="store_true",
        help="Review mode: colored boxes + IDs, no blur")
    p.add_argument("--profile", action="store_true",
        help="Show per-phase timing")
    p.add_argument("--exclude-ids", type=str, default="",
        help="Track IDs to skip blurring, e.g. '2,5,7'")
    p.add_argument("--lost-buffer", type=int, default=180,
        help="Frames to predict after detection lost (180 ≈ 6s @30fps)")
    p.add_argument("--smooth", type=float, default=0.7,
        help="Smoothing factor: 0=rigid, 1=no smoothing")
    p.add_argument("--preset", choices=("quality", "fast"), default="quality",
        help="Tracking preset: quality keeps full optical flow; fast limits optical-flow cost")
    p.add_argument("--max-face-height-ratio", type=float, default=0.4,
        help="Ignore detections taller than this fraction of frame height")
    p.add_argument("--flow-max-points", type=int, default=50,
        help="Max feature points per face for optical flow")
    p.add_argument("--flow-min-confirmations", type=int, default=3,
        help="Total detections before optical flow is enabled for a track")
    p.add_argument("--flow-max-missed", type=int, default=0,
        help="Only run optical flow for the first N missed frames per track (0=no limit)")
    p.add_argument("--no-flow", action="store_true",
        help="Disable optical-flow tracking (use prediction-only)")
    p.add_argument("--no-nvenc", action="store_true",
        help="Disable NVENC hardware encoding (use libx264 CPU)")
    p.add_argument("--version", action="store_true",
        help="Show version and exit")
    return p.parse_args()


def get_thresh(args, frame_idx: int, fps: float) -> float:
    if not args.time_thresh:
        return args.thresh
    sec = frame_idx / fps
    th = args.thresh
    for s, t in args.time_thresh:
        if sec >= s:
            th = t
    return th


def parse_time_thresh(raw: str):
    """Parse 'sec:thresh,sec:thresh' → [(sec, thresh), ...]"""
    if not raw:
        return []
    segments = []
    for part in raw.split(","):
        sec, th = part.strip().split(":")
        segments.append((float(sec), float(th)))
    segments.sort(key=lambda x: x[0])
    return segments


def parse_exclude_ids(raw: str) -> set:
    if not raw:
        return set()
    return {int(x.strip()) for x in raw.split(",") if x.strip()}
