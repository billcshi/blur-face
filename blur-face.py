#!/usr/bin/env python3
"""
blur-face — Face blur with tracking + optical flow.

Usage:
  python blur-face.py input.mov -o output.mp4
  python blur-face.py input.mov --debug --profile
  python blur-face.py input.mov --detect-interval 100
  python blur-face.py input.mov --model yolo26n-face.pt --thresh 0.15
"""
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

# Ensure the package is importable from script directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blurface.cli import parse_args, get_thresh, parse_time_thresh, parse_exclude_ids
from blurface.detector import FaceDetector
from blurface.tracker import FaceTracker
from blurface.renderer import apply_blur, draw_debug_box, HAS_CUDA
from blurface.encoder import FFmpegEncoder
from blurface.profiler import Profiler


def _detect_worker(detector, frame, conf):
    """Run detection in background thread; returns (boxes, gpu_seconds)."""
    t0 = time.time()
    boxes = detector.detect(frame, conf=conf)
    return boxes, time.time() - t0


def main():
    args = parse_args()
    args.time_thresh = parse_time_thresh(args.time_thresh)
    exclude_ids = parse_exclude_ids(args.exclude_ids)

    # ── Init ─────────────────────────────────────────
    detector = FaceDetector(args.model, device=args.device)
    if args.preset == "fast":
        if args.flow_max_points == 50:
            args.flow_max_points = 20
        if args.flow_max_missed == 0:
            args.flow_max_missed = 45

    flow_enabled = not getattr(args, 'no_flow', False)
    tracker = FaceTracker(
        lost_buffer=args.lost_buffer,
        smooth=args.smooth,
        flow_enabled=flow_enabled,
        flow_max_points=getattr(args, 'flow_max_points', 50),
        flow_min_confirmations=getattr(args, 'flow_min_confirmations', 3),
        flow_max_missed=getattr(args, 'flow_max_missed', 0),
    )
    prof = Profiler()

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total / fps if fps > 0 else 0

    kernel = args.blur_kernel if args.blur_kernel % 2 == 1 else args.blur_kernel + 1

    flow_str = "on" if flow_enabled else "off"
    render_str = "GPU (cv2.cuda)" if HAS_CUDA else "CPU"
    nvenc_enabled = not getattr(args, 'no_nvenc', False)
    print(f"GPU:     render={render_str}  |  encode={'NVENC' if nvenc_enabled else 'CPU'}")
    print(f"Tracker: lost-buffer={args.lost_buffer}  |  smooth={args.smooth}" +
          f"  |  flow={flow_str}" +
          (f"  |  preset={args.preset}" if flow_enabled else "") +
          (f"  |  flow-points={args.flow_max_points}" if flow_enabled else "") +
          (f"  |  flow-max-missed={args.flow_max_missed}" if flow_enabled and args.flow_max_missed > 0 else ""))
    if exclude_ids:
        print(f"Exclude IDs: {exclude_ids}")
    mode_str = "DEBUG (no blur)" if args.debug else "BLUR"
    print(f"Mode:    {mode_str}")
    print()

    encoder = FFmpegEncoder(args.output, w, h, fps, args.input,
                            use_nvenc=nvenc_enabled)

    wall_start = time.time()

    # ── Full-detect mode with async GPU pipeline ────
    pool = ThreadPoolExecutor(max_workers=1)

    with prof.phase("read"):
        ret, frame = cap.read()
    if not ret:
        return

    th = get_thresh(args, 0, fps)
    pending = pool.submit(_detect_worker, detector, frame, th)
    i = 0
    last_pct = -1

    while True:
        with prof.phase("read"):
            ret, next_frame = cap.read()

        if ret:
            next_th = get_thresh(args, i + 1, fps)
            next_pending = pool.submit(_detect_worker, detector, next_frame, next_th)

        boxes, detect_dt = pending.result()
        prof.record("detect", detect_dt)

        with prof.phase("track"):
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if flow_enabled else None
            tracks = tracker.update(boxes, frame_h=h, frame_w=w,
                                    frame_gray=frame_gray)

        with prof.phase("render"):
            for bbox, tid, missed, is_predicted in tracks:
                if args.debug:
                    draw_debug_box(frame, bbox, tid,
                                   is_predicted=is_predicted,
                                   is_excluded=(tid in exclude_ids))
                elif tid not in exclude_ids:
                    apply_blur(frame, bbox, kernel,
                               mask_scale=args.mask_scale,
                               frame_w=w, frame_h=h)

        with prof.phase("write"):
            encoder.write(frame.tobytes())

        i += 1
        pct = 100 * i // total
        if pct != last_pct and pct % 5 == 0:
            elapsed = time.time() - wall_start
            fps_now = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / fps_now if fps_now > 0 else 0
            n_tracks = tracker.active_count
            n_pred = tracker.predicted_count
            tinfo = f"tracks={n_tracks}"
            if n_pred > 0:
                tinfo += f"(pred={n_pred})"
            if tracker.flow_attempts > 0:
                tinfo += f"(flow={tracker.flow_successes}/{tracker.flow_attempts})"
            extra = f" th={th:.2f}" if args.time_thresh else ""
            line = f"  [{mode_str}]{extra}  {i}/{total} ({pct}%)  {fps_now:.1f}fps  ETA {eta:.0f}s  [{tinfo}]"
            if args.profile and i > 0:
                parts = []
                for ph in prof._order:
                    avg_ms = (prof.times[ph] / i) * 1000
                    parts.append(f"{ph}={avg_ms:.1f}ms")
                line += f"  |  {'  '.join(parts)}"
            print(line)
            last_pct = pct

        if not ret:
            break

        frame = next_frame
        pending = next_pending
        th = next_th

    wall_end = time.time()
    total_wall = wall_end - wall_start
    encoder.close()
    cap.release()
    pool.shutdown(wait=False)

    print(f"\n{'='*60}")
    print(f"Done → {args.output}")
    print(f"Total time:      {total_wall:.1f}s")
    print(f"Frames:          {i}")
    print(f"Avg FPS:         {i/total_wall:.1f}")
    print(f"Source FPS:      {fps:.2f}")
    if tracker.flow_attempts > 0:
        rate = 100 * tracker.flow_successes / tracker.flow_attempts
        print(f"Optical flow:    {tracker.flow_successes}/{tracker.flow_attempts} ({rate:.0f}% success)")
    if fps > 0 and i > 0:
        proc_fps = i / total_wall
        if proc_fps >= fps:
            print(f"Speed:           {proc_fps/fps:.1f}x real-time")
        else:
            print(f"Speed:           {fps/proc_fps:.1f}x slower than real-time")
    print()
    print(prof.summary(i, total_wall))
    if i > 0:
        total_phase = sum(prof.times[ph] for ph in prof._order)
        overlap_pct = (total_phase / total_wall * 100) - 100 if total_wall > 0 else 0
        if overlap_pct > 5:
            print(f"  → CPU/GPU overlap: {overlap_pct:.0f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
