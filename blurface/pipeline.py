"""End-to-end video processing orchestration."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import cv2

from .config import AppConfig
from .detector import FaceDetector
from .encoder import FFmpegEncoder
from .profiler import Profiler
from .renderer import HAS_CUDA, apply_blur, draw_debug_box
from .tracker import FaceTracker
from .video import VideoInputError, VideoSource


def _detect(detector: FaceDetector, frame, threshold: float):
    started = time.perf_counter()
    boxes = detector.detect(frame, conf=threshold)
    return boxes, time.perf_counter() - started


class VideoProcessor:
    """Own the validated processing lifecycle and all external resources."""

    def __init__(self, config: AppConfig):
        self.config = config.validate()
        self.profiler = Profiler()

    def _tracker(self) -> FaceTracker:
        config = self.config
        return FaceTracker(
            lost_buffer=config.lost_buffer,
            smooth=config.smooth,
            min_face_w=config.min_face_size,
            min_face_h=config.min_face_size,
            max_face_h_ratio=config.max_face_height_ratio,
            flow_enabled=config.flow_enabled,
            flow_max_points=config.flow_max_points,
            flow_min_confirmations=config.flow_min_confirmations,
            flow_max_missed=config.flow_max_missed,
        )

    def run(self) -> None:
        config = self.config
        wall_start = time.perf_counter()
        if config.offline:
            print("Model:   strict offline mode")
        else:
            print("Model:   local model preferred; a missing model may download now")
        print("Privacy: video frames are processed locally after model initialization")

        with VideoSource(config.input) as source:
            assert source.info is not None
            info = source.info
            # Model initialization happens before a frame is decoded. This is
            # the only stage at which our configuration permits a download.
            detector = FaceDetector(
                config.model,
                device=config.device,
                allow_download=not config.offline,
            )
            tracker = self._tracker()
            with self.profiler.phase("read"):
                read_ok, frame = source.read()
            if not read_ok:
                raise VideoInputError("input opened but did not yield a video frame")

            render_backend = "GPU (OpenCV CUDA)" if HAS_CUDA else "CPU"
            mode = "DEBUG" if config.debug else "BLUR"
            print(f"Render:  {render_backend}")
            print(f"Mode:    {mode}")
            if config.exclude_ids:
                print(
                    "[WARN] Track IDs are best-effort and can change when people cross; "
                    f"leaving IDs unblurred: {sorted(config.exclude_ids)}"
                )

            frame_count = 0
            last_percent = -1
            threshold = config.threshold_for(0, info.fps)
            with ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="face-detect"
            ) as pool:
                pending = pool.submit(_detect, detector, frame, threshold)
                with FFmpegEncoder(
                    config.output,
                    info.width,
                    info.height,
                    info.fps,
                    config.input,
                    use_nvenc=config.use_nvenc,
                    overwrite=config.overwrite,
                    ffmpeg_exe=config.ffmpeg,
                ) as encoder:
                    print(f"Encode:  {encoder.codec}")
                    while True:
                        with self.profiler.phase("read"):
                            next_ok, next_frame = source.read()
                        next_pending = None
                        next_threshold = threshold
                        if next_ok:
                            next_threshold = config.threshold_for(
                                frame_count + 1, info.fps
                            )
                            next_pending = pool.submit(
                                _detect, detector, next_frame, next_threshold
                            )

                        boxes, detect_seconds = pending.result()
                        self.profiler.record("detect", detect_seconds)
                        with self.profiler.phase("track"):
                            gray = (
                                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                if config.flow_enabled
                                else None
                            )
                            tracks = tracker.update(
                                boxes,
                                frame_h=info.height,
                                frame_w=info.width,
                                frame_gray=gray,
                            )
                        with self.profiler.phase("render"):
                            for track in tracks:
                                if config.debug:
                                    draw_debug_box(
                                        frame,
                                        track.box,
                                        track.track_id,
                                        is_predicted=track.is_predicted,
                                        is_excluded=track.track_id
                                        in config.exclude_ids,
                                    )
                                elif track.track_id not in config.exclude_ids:
                                    apply_blur(
                                        frame,
                                        track.box,
                                        config.blur_kernel,
                                        mask_scale=config.mask_scale,
                                    )
                        with self.profiler.phase("write"):
                            encoder.write(frame.tobytes())

                        frame_count += 1
                        self._show_progress(
                            frame_count,
                            info.total_frames,
                            wall_start,
                            tracker.active_count,
                            tracker.predicted_count,
                            threshold,
                            last_percent,
                        )
                        if info.total_frames:
                            percent = min(100, 100 * frame_count // info.total_frames)
                            if percent % 5 == 0:
                                last_percent = percent
                        if not next_ok:
                            break
                        frame = next_frame
                        pending = next_pending
                        threshold = next_threshold

        total_wall = time.perf_counter() - wall_start
        print(f"\nDone → {config.output}")
        print(
            f"Frames: {frame_count} | elapsed: {total_wall:.1f}s | "
            f"average: {frame_count / total_wall:.1f}fps"
        )
        summary = self.profiler.summary(frame_count, total_wall)
        if config.profile and summary:
            print(summary)

    @staticmethod
    def _show_progress(
        frame_count: int,
        total_frames: int | None,
        wall_start: float,
        tracks: int,
        predicted: int,
        threshold: float,
        last_percent: int,
    ) -> None:
        elapsed = time.perf_counter() - wall_start
        speed = frame_count / elapsed if elapsed else 0.0
        if total_frames:
            percent = min(100, 100 * frame_count // total_frames)
            if percent == last_percent or percent % 5:
                return
            remaining = max(0, total_frames - frame_count)
            eta = remaining / speed if speed else 0
            print(
                f"  {frame_count}/{total_frames} ({percent}%) "
                f"{speed:.1f}fps ETA {eta:.0f}s "
                f"tracks={tracks} predicted={predicted} threshold={threshold:.2f}"
            )
        elif frame_count % 100 == 0:
            print(
                f"  {frame_count} frames {speed:.1f}fps "
                f"tracks={tracks} predicted={predicted} threshold={threshold:.2f}"
            )
