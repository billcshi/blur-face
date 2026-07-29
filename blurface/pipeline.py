"""End-to-end video processing orchestration."""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .config import AppConfig
from .detector import FaceDetector
from .encoder import FFmpegEncoder
from .profiler import Profiler
from .renderer import (
    HAS_CUDA,
    apply_blur,
    apply_mask_preview,
    draw_debug_box,
)
from .sam2_segmenter import Sam2VideoSegmenter
from .tracker import FaceTracker
from .temporal import (
    FinalMaskStabilizer,
    SceneCutDetector,
    SAM_REVERSE_CACHE_LIMIT_BYTES,
    TemporalMaskStabilizer,
    TemporalStore,
    ambiguous_track_ids,
    box_iou,
    motion_proxy,
    resize_mask_to_source,
)
from .video import VideoInputError, VideoSource

SAM_MEMORY_MAX_FRAMES = 30
SAM_MEMORY_MAX_OBJECTS = 16
SAM_REVERSE_MAX_FRAMES = 16
SAM_MIN_BOX_COVERAGE = 0.50
SAM_MIN_AXIS_SPAN = 0.60
SAM_MIN_DENSE_AXIS_FRACTION = 0.45


def _detect(detector: FaceDetector, frame, threshold: float):
    started = time.perf_counter()
    boxes = detector.detect(frame, conf=threshold)
    return boxes, time.perf_counter() - started


def _mask_bounds(mask: np.ndarray) -> list[int] | None:
    """Return the tight exclusive bounds of a non-empty full-frame mask."""
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        return None
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return None
    return [
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    ]


def _add_box_to_mask(mask: np.ndarray, box) -> None:
    """Union one clipped tracker box into a full-frame mask."""
    height, width = mask.shape
    x1, y1, x2, y2 = (int(value) for value in box)
    x1, x2 = sorted(
        (max(0, min(width, x1)), max(0, min(width, x2)))
    )
    y1, y2 = sorted(
        (max(0, min(height, y1)), max(0, min(height, y2)))
    )
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 255


class VideoProcessor:
    """Own the validated processing lifecycle and all external resources."""

    def __init__(self, config: AppConfig):
        self.config = config.validate()
        self.profiler = Profiler()
        self._phase_progress: dict[str, tuple[int, float]] = {}

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
        if (
            config.temporal_stabilization
            and config.mask_engine == "sam2.1"
            and not config.debug
        ):
            self._run_temporal()
            return
        wall_start = time.perf_counter()
        if config.offline:
            print("Model:   strict offline mode")
        elif config.detector == "yolo":
            print("Model:   local YOLO weight required; no runtime download")
        else:
            print("Model:   local model preferred; a missing model may download now")
        print("Privacy: video frames are processed locally after model initialization")
        print(f"Detector: {config.detector}; model={config.model}")

        with VideoSource(config.input) as source:
            assert source.info is not None
            info = source.info
            # Model initialization happens before a frame is decoded. This is
            # the only stage at which our configuration permits a download.
            detector = FaceDetector(
                config.model,
                device=config.device,
                allow_download=not config.offline,
                backend=config.detector,
            )
            segmenter = None
            if not config.debug and config.mask_engine == "sam2.1":
                segmenter = Sam2VideoSegmenter(
                    config.sam2_model,
                    device=config.device,
                    offline=config.offline,
                )
            tracker = self._tracker()
            with self.profiler.phase("read"):
                read_ok, frame = source.read()
            if not read_ok:
                raise VideoInputError("input opened but did not yield a video frame")

            render_backend = "GPU (OpenCV CUDA)" if HAS_CUDA else "CPU"
            mode = (
                "DEBUG"
                if config.debug
                else "MASK PREVIEW"
                if config.mask_preview
                else "BLUR"
            )
            render_effect = (
                apply_mask_preview if config.mask_preview else apply_blur
            )
            print(f"Render:  {render_backend}")
            print(f"Mode:    {mode}")
            if segmenter is None:
                print(f"Mask:    geometric {config.mask_shape}")
            else:
                print(
                    "Mask:    SAM 2.1 video tracking "
                    f"(detector correction every "
                    f"{config.sam2_refresh_interval} frames)"
                )
            if config.blur_strategy == "adaptive":
                print(
                    "Blur:    confidence-adaptive "
                    f"{config.blur_kernel_min}-{config.blur_kernel}"
                )
            else:
                print(f"Blur:    fixed kernel {config.blur_kernel}")

            frame_count = 0
            last_percent = -1
            segmentation_warning_count = 0
            stream_cut_detector = (
                SceneCutDetector(config.scene_cut_sensitivity)
                if segmenter is not None
                and config.mask_engine == "sam2.1"
                else None
            )
            sam_ambiguous_ids: set[int] = set()
            sam_scene_frame = 0
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
                    include_audio=not config.mask_preview,
                    temporary_directory=config.job_temp_dir,
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
                            next_sam_scene_frame = sam_scene_frame + 1
                            if (
                                config.mask_engine == "sam2.1"
                                and next_sam_scene_frame
                                >= max(3, config.flow_min_confirmations)
                                and next_sam_scene_frame
                                % config.sam2_refresh_interval
                                != 0
                            ):
                                next_pending = pool.submit(
                                    lambda: (
                                        [],
                                        0.0,
                                    )
                                )
                            else:
                                next_pending = pool.submit(
                                    _detect,
                                    detector,
                                    next_frame,
                                    next_threshold,
                                )

                        boxes, detect_seconds = pending.result()
                        self.profiler.record("detect", detect_seconds)
                        if (
                            stream_cut_detector is not None
                            and stream_cut_detector.update(frame)
                        ):
                            # The next frame may already be running through the
                            # shared detector. Finish/cancel it before the
                            # synchronous cut-frame detection so the detector
                            # is never entered concurrently.
                            if next_pending is not None:
                                if not next_pending.cancel():
                                    next_pending.result()
                            boxes, cut_detect_seconds = _detect(
                                detector, frame, threshold
                            )
                            self.profiler.record(
                                "detect", cut_detect_seconds
                            )
                            if next_ok:
                                next_pending = pool.submit(
                                    _detect,
                                    detector,
                                    next_frame,
                                    next_threshold,
                                )
                            tracker.reset()
                            sam_ambiguous_ids.clear()
                            sam_scene_frame = 0
                            try:
                                segmenter.reset()
                            except Exception as exc:
                                if segmentation_warning_count < 3:
                                    print(
                                        "[WARN] SAM scene reset failed; "
                                        "using geometric coverage for the "
                                        f"remaining video: {exc}"
                                    )
                                segmentation_warning_count += 1
                                segmenter = None
                                stream_cut_detector = None
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
                            # SAM must always see original pixels. Earlier
                            # coverage boxes in the same frame may already have
                            # blurred overlapping regions in ``frame``.
                            segmentation_source = (
                                frame.copy() if segmenter is not None else None
                            )
                            if config.mask_preview:
                                frame[:] = 0
                            sam_masks = {}
                            ambiguous_ids = ambiguous_track_ids(tracks)
                            if (
                                segmenter is not None
                                and config.mask_engine == "sam2.1"
                            ):
                                if (
                                    ambiguous_ids
                                    and ambiguous_ids != sam_ambiguous_ids
                                ):
                                    # Discard all object memory at a crossing:
                                    # after separation SAM remains empty until
                                    # a fresh detector correction, so an identity
                                    # swap cannot survive the ambiguous span.
                                    try:
                                        segmenter.reset()
                                    except Exception as exc:
                                        if segmentation_warning_count < 3:
                                            print(
                                                "[WARN] SAM crossing reset "
                                                "failed; using geometric "
                                                f"coverage: {exc}"
                                            )
                                        segmentation_warning_count += 1
                                        segmenter = None
                                sam_ambiguous_ids = ambiguous_ids
                                correction_frame = (
                                    sam_scene_frame
                                    < max(3, config.flow_min_confirmations)
                                    or sam_scene_frame
                                    % config.sam2_refresh_interval
                                    == 0
                                )
                                prompts = (
                                    self._sam_prompt_boxes(
                                        tracks,
                                        ambiguous_ids,
                                        include_predicted=True,
                                    )
                                    if correction_frame
                                    else {}
                                )
                                if segmenter is not None:
                                    sam_masks = segmenter.track_frame(
                                        segmentation_source,
                                        prompts,
                                        config.sam_mask_expansion,
                                        # The video segmenter owns only the
                                        # SAM contour. The pipeline owns the
                                        # current Track geometry and must
                                        # apply union/intersection uniformly
                                        # on prompted and propagated frames.
                                        "mask-only",
                                    )
                                if (
                                    segmenter is not None
                                    and segmenter.last_error
                                    and segmentation_warning_count < 3
                                ):
                                    print(
                                        "[WARN] SAM 2 video tracking failed; "
                                        "using geometric coverage for this "
                                        f"frame: {segmenter.last_error}"
                                    )
                                    segmentation_warning_count += 1
                            for track in tracks:
                                if config.debug:
                                    draw_debug_box(
                                        frame,
                                        track.box,
                                        track.track_id,
                                        is_predicted=track.is_predicted,
                                        confidence=track.confidence,
                                    )
                                else:
                                    # Do not turn diagonally separated boxes into
                                    # one large ellipse: its corners can expose
                                    # the currently observed face. Cover the
                                    # smoothed/predicted and observed regions
                                    # independently.
                                    kernel = config.blur_kernel_for(
                                        track.confidence, threshold
                                    )
                                    if (
                                        segmenter is not None
                                        and config.mask_engine == "sam2.1"
                                    ):
                                        full_mask = sam_masks.get(
                                            track.track_id
                                        )
                                        if (
                                            full_mask is not None
                                            and full_mask.any()
                                            and track.track_id
                                            not in ambiguous_ids
                                            and self._sam_mask_matches_track(
                                                full_mask, track.box
                                            )
                                        ):
                                            render_mask = full_mask.copy()
                                            if (
                                                config.segmentation_combine
                                                == "union"
                                            ):
                                                for safety_box in (
                                                    track.coverage_boxes
                                                ):
                                                    _add_box_to_mask(
                                                        render_mask,
                                                        safety_box,
                                                    )
                                            elif (
                                                config.segmentation_combine
                                                == "intersection"
                                            ):
                                                current_geometry = np.zeros(
                                                    render_mask.shape,
                                                    dtype=np.uint8,
                                                )
                                                for current_box in (
                                                    track.coverage_boxes
                                                ):
                                                    _add_box_to_mask(
                                                        current_geometry,
                                                        current_box,
                                                    )
                                                render_mask = cv2.bitwise_and(
                                                    render_mask,
                                                    current_geometry,
                                                )
                                                geometry_area = (
                                                    np.count_nonzero(
                                                        current_geometry
                                                    )
                                                )
                                                if (
                                                    geometry_area
                                                    and np.count_nonzero(
                                                        render_mask
                                                    )
                                                    / geometry_area
                                                    < 0.25
                                                ):
                                                    # The current contour is
                                                    # no longer credible
                                                    # inside this Track. Fall
                                                    # through to immediate
                                                    # geometric coverage.
                                                    render_mask[:] = 0
                                            mask_box = _mask_bounds(
                                                render_mask
                                            )
                                            if mask_box is not None:
                                                x1, y1, x2, y2 = mask_box
                                                render_effect(
                                                    frame,
                                                    mask_box,
                                                    kernel,
                                                    mask_scale=1.0,
                                                    mask_shape=(
                                                        config.mask_shape
                                                    ),
                                                    coverage_mask=render_mask[
                                                        y1:y2, x1:x2
                                                    ],
                                                    segmentation_combine=(
                                                        "intersection"
                                                    ),
                                                )
                                                continue
                                    for coverage_box in track.coverage_boxes:
                                        render_effect(
                                            frame,
                                            coverage_box,
                                            kernel,
                                            mask_scale=config.mask_scale,
                                            mask_shape=config.mask_shape,
                                        )
                        with self.profiler.phase("write"):
                            encoder.write(frame.tobytes())

                        frame_count += 1
                        sam_scene_frame += 1
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
        print(f"\nDone -> {config.output}")
        print(
            f"Frames: {frame_count} | elapsed: {total_wall:.1f}s | "
            f"average: {frame_count / total_wall:.1f}fps"
        )
        summary = self.profiler.summary(frame_count, total_wall)
        if config.profile and summary:
            print(summary)

    @staticmethod
    def _mask_box_at_source(
        mask: np.ndarray, source_shape: tuple[int, int]
    ) -> list[int] | None:
        bounds = _mask_bounds(mask)
        if bounds is None:
            return None
        source_h, source_w = source_shape
        mask_h, mask_w = mask.shape
        return [
            round(bounds[0] * source_w / mask_w),
            round(bounds[1] * source_h / mask_h),
            round(bounds[2] * source_w / mask_w),
            round(bounds[3] * source_h / mask_h),
        ]

    @staticmethod
    def _sam_mask_matches_track(mask: np.ndarray, box) -> bool:
        """Reject sparse, fragmented, or drifted SAM object memory.

        Bounding-box overlap alone is unsafe: two tiny components at opposite
        corners can span a whole face box. Validate actual foreground pixels
        and require one coherent component through the face core.
        """
        if not isinstance(mask, np.ndarray) or mask.ndim != 2:
            return False
        foreground = np.ascontiguousarray(mask > 0, dtype=np.uint8)
        height, width = foreground.shape
        x1 = max(0, min(width, int(np.floor(box[0]))))
        y1 = max(0, min(height, int(np.floor(box[1]))))
        x2 = max(0, min(width, int(np.ceil(box[2]))))
        y2 = max(0, min(height, int(np.ceil(box[3]))))
        box_area = max(0, x2 - x1) * max(0, y2 - y1)
        total_area = int(np.count_nonzero(foreground))
        if box_area <= 0 or total_area <= 0:
            return False
        overlap_area = int(np.count_nonzero(foreground[y1:y2, x1:x2]))
        # A face-core stripe or eye-only mask can be coherent and centered
        # while still exposing most of the detector box. Successful
        # segmentation must cover a majority of the proposed face region;
        # otherwise geometry is the privacy-safe result.
        if overlap_area / box_area < SAM_MIN_BOX_COVERAGE:
            return False
        if overlap_area / total_area < 0.20 or total_area / box_area > 4.0:
            return False

        component_count, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(foreground, connectivity=8)
        )
        if component_count <= 1:
            return False
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        largest_area = int(stats[largest_label, cv2.CC_STAT_AREA])
        if largest_area / total_area < 0.60:
            return False

        component_in_box = labels[y1:y2, x1:x2] == largest_label
        component_rows, component_columns = np.nonzero(component_in_box)
        if not len(component_rows):
            return False
        box_height = y2 - y1
        box_width = x2 - x1
        horizontal_span = (
            int(component_columns.max()) - int(component_columns.min()) + 1
        )
        vertical_span = (
            int(component_rows.max()) - int(component_rows.min()) + 1
        )
        if (
            horizontal_span / box_width < SAM_MIN_AXIS_SPAN
            or vertical_span / box_height < SAM_MIN_AXIS_SPAN
        ):
            return False

        # Require two-dimensional support, not merely a long connected line.
        # Count rows/columns that contain meaningful coverage on the other
        # axis; vertical, horizontal, and cross-shaped slivers fail here.
        dense_rows = np.count_nonzero(
            np.count_nonzero(component_in_box, axis=1)
            >= max(1, round(box_width * 0.20))
        )
        dense_columns = np.count_nonzero(
            np.count_nonzero(component_in_box, axis=0)
            >= max(1, round(box_height * 0.20))
        )
        if (
            dense_rows / box_height < SAM_MIN_DENSE_AXIS_FRACTION
            or dense_columns / box_width < SAM_MIN_DENSE_AXIS_FRACTION
        ):
            return False

        core_x1 = x1 + round((x2 - x1) * 0.20)
        core_x2 = x1 + round((x2 - x1) * 0.80)
        core_y1 = y1 + round((y2 - y1) * 0.15)
        core_y2 = y1 + round((y2 - y1) * 0.85)
        if not np.any(
            labels[core_y1:core_y2, core_x1:core_x2] == largest_label
        ):
            return False

        mask_center = np.asarray(centroids[largest_label], dtype=float)
        box_center = np.array(
            ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        )
        diagonal = max(
            1.0, np.hypot(box[2] - box[0], box[3] - box[1])
        )
        return bool(np.linalg.norm(mask_center - box_center) <= diagonal * 0.55)

    @staticmethod
    def _sam_prompt_boxes(
        tracks, ambiguous: set[int], include_predicted: bool
    ) -> dict[int, tuple[int, int, int, int]]:
        """Never seed uncertain crossing identities into SAM memory."""
        return {
            track.track_id: tuple(track.observed_box or track.box)
            for track in tracks
            if track.track_id not in ambiguous
            and (track.observed_box is not None or include_predicted)
        }

    def _store_sam_reverse_masks(
        self,
        store: TemporalStore,
        segmenter,
        frame_window,
        scene_id: int,
        track,
        threshold: float,
        source_shape: tuple[int, int],
    ) -> int:
        """Prefer official SAM reverse propagation for a newly seeded face."""
        if (
            self.config.backfill_frames <= 0
            or not hasattr(segmenter, "reverse_propagate")
            or track.observed_box is None
        ):
            return 0
        window = [item for item in frame_window if item[1] == scene_id]
        if len(window) < 2:
            return 0
        masks = segmenter.reverse_propagate(
            [item[2] for item in window],
            track.track_id,
            tuple(track.observed_box),
            self.config.sam_mask_expansion,
        )
        previous_mask = masks.get(len(window) - 1)
        previous_box = list(track.observed_box)
        previous_pixels = (
            window[-1][2][previous_mask > 0]
            if previous_mask is not None
            and previous_mask.shape == window[-1][2].shape[:2]
            and np.any(previous_mask)
            else window[-1][2][
                previous_box[1] : previous_box[3],
                previous_box[0] : previous_box[2],
            ]
        )
        expected_index = len(window) - 2
        stored_count = 0
        for relative_index in sorted(masks, reverse=True):
            full_mask = masks[relative_index]
            if relative_index >= len(window) - 1:
                continue
            if relative_index != expected_index or not np.any(full_mask):
                break
            if not self._sam_mask_matches_track(full_mask, previous_box):
                break
            expected_index -= 1
            frame_index = window[relative_index][0]
            _stored_scene, gray, _stored_shape = store.frame(frame_index)
            mask = cv2.resize(
                full_mask,
                (gray.shape[1], gray.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            box = self._mask_box_at_source(mask, source_shape)
            if box is None:
                break
            old_area = max(
                1,
                (previous_box[2] - previous_box[0])
                * (previous_box[3] - previous_box[1]),
            )
            new_area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
            old_center = np.array(
                (
                    (previous_box[0] + previous_box[2]) / 2,
                    (previous_box[1] + previous_box[3]) / 2,
                )
            )
            new_center = np.array(
                ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            )
            old_diagonal = np.hypot(
                previous_box[2] - previous_box[0],
                previous_box[3] - previous_box[1],
            )
            if (
                not 0.45 <= new_area / old_area <= 2.2
                or np.linalg.norm(new_center - old_center)
                > max(12.0, old_diagonal * 1.5)
            ):
                break
            candidate_pixels = window[relative_index][2][full_mask > 0]
            if (
                not len(candidate_pixels)
                or not len(previous_pixels)
                or np.linalg.norm(
                    np.median(candidate_pixels, axis=0)
                    - np.median(previous_pixels, axis=0)
                )
                > 70
            ):
                break
            height, width = source_shape
            margin = max(2, round(min(height, width) * 0.025))
            if (
                box[0] <= margin
                or box[1] <= margin
                or box[2] >= width - margin
                or box[3] >= height - margin
            ):
                break
            others = [
                record
                for record in store.records_for_frame(frame_index)
                if record.track_id != track.track_id
            ]
            if any(box_iou(box, record.box) >= 0.20 for record in others):
                break
            store.add_track(
                frame_index,
                scene_id,
                track.track_id,
                box,
                (box,),
                track.confidence,
                threshold,
                False,
                False,
                False,
                mask,
                replace=True,
            )
            previous_box = box
            previous_pixels = candidate_pixels
            stored_count += 1
        return stored_count

    def _run_temporal(self) -> None:
        """Analyze, repair masks on disk, then reopen the input for rendering."""
        config = self.config
        wall_start = time.perf_counter()
        if config.offline:
            print("Model:   strict offline mode")
        elif config.detector == "yolo":
            print(
                "Model:   local YOLO weight required; "
                "a missing SAM model may download during initialization"
            )
        else:
            print("Model:   local model preferred; a missing model may download now")
        print("Privacy: video frames are processed locally after model initialization")
        print(f"Detector: {config.detector}; model={config.model}")
        print(
            "Temporal: offline two-pass stabilization "
            f"(backfill={config.backfill_frames}, "
            f"release-hold={config.release_hold_frames}, "
            f"temporary limit={config.temporal_storage_limit_mb} MiB)"
        )
        print(
            f"Mask:    {config.mask_engine}; "
            f"combine={config.segmentation_combine}; "
            "failures=geometric"
        )
        with VideoSource(config.input) as source:
            assert source.info is not None
            info = source.info
            duration = (
                info.total_frames / info.fps
                if info.total_frames is not None
                else None
            )
            print(
                f"Video:   {info.width}x{info.height} @ {info.fps:.3f} fps; "
                f"frames={info.total_frames or 'unknown'}"
                + (f"; duration={duration:.1f}s" if duration is not None else "")
            )
            init_started = time.perf_counter()
            detector = FaceDetector(
                config.model,
                device=config.device,
                allow_download=not config.offline,
                backend=config.detector,
            )
            segmenter = Sam2VideoSegmenter(
                config.sam2_model,
                device=config.device,
                offline=config.offline,
            )
            initialization_seconds = time.perf_counter() - init_started
            print(
                "SAM:     "
                f"model={getattr(segmenter, 'model_name', config.sam2_model)}; "
                f"device={getattr(segmenter, 'device', config.device)}; "
                f"initialization={initialization_seconds:.1f}s"
            )
            print(
                f"Analyze: {config.detector.upper()} correction every "
                f"{config.sam2_refresh_interval} frames; "
                "SAM memory propagates between corrections"
            )
            tracker = self._tracker()
            cut_detector = SceneCutDetector(config.scene_cut_sensitivity)
            scene_id = 0
            scene_start = 0
            warning_count = 0
            frame_window = deque(
                maxlen=min(
                    config.backfill_frames + 1, SAM_REVERSE_MAX_FRAMES
                )
            )
            frame_window_bytes = 0
            sam_quarantined_ids: set[int] = set()
            sam_over_capacity = False
            sam_mask_stats = {
                "accepted": 0,
                "missing_from_memory": 0,
                "empty_mask": 0,
                "low_object_score": 0,
                "invalid_object_score": 0,
                "quality_rejected": 0,
                "ambiguous": 0,
            }
            run_stats = {
                "scene_cuts": 0,
                "correction_frames": 0,
                "prompts": 0,
                "memory_resets": 0,
                "capacity_fallback_frames": 0,
                "reverse_backfilled_frames": 0,
                "segmenter_error_frames": 0,
                "max_concurrent_tracks": 0,
                "peak_reverse_cache_bytes": 0,
            }
            unique_tracks: set[tuple[int, int]] = set()
            with TemporalStore(
                parent=config.job_temp_dir,
                limit_bytes=config.temporal_storage_limit_mb * 1024**2
            ) as store:
                frame_index = 0
                analyze_started = time.perf_counter()
                while True:
                    with self.profiler.phase("read"):
                        ok, frame = source.read()
                    if not ok:
                        break
                    proxy = motion_proxy(frame)
                    is_cut = cut_detector.update(proxy)
                    if is_cut:
                        run_stats["scene_cuts"] += 1
                        scene_id += 1
                        scene_start = frame_index
                        tracker.reset()
                        if hasattr(segmenter, "reset"):
                            segmenter.reset()
                        frame_window.clear()
                        frame_window_bytes = 0
                        sam_quarantined_ids.clear()
                        sam_over_capacity = False
                        print(f"  scene cut at frame {frame_index}; temporal state reset")
                    store.add_frame(
                        frame_index, scene_id, proxy, frame.shape[:2]
                    )
                    threshold = config.threshold_for(frame_index, info.fps)
                    scene_offset = frame_index - scene_start
                    sam_window_reset = (
                        getattr(segmenter, "frame_count", 0)
                        >= SAM_MEMORY_MAX_FRAMES
                        or getattr(segmenter, "object_count", 0)
                        >= SAM_MEMORY_MAX_OBJECTS
                        or getattr(segmenter, "needs_reseed", False)
                    )
                    correction = (
                        sam_window_reset
                        or sam_over_capacity
                        or bool(sam_quarantined_ids)
                        or scene_offset < max(3, config.flow_min_confirmations)
                        or scene_offset % config.sam2_refresh_interval == 0
                    )
                    if correction:
                        run_stats["correction_frames"] += 1
                    with self.profiler.phase("detect"):
                        boxes = (
                            detector.detect(frame, conf=threshold)
                            if correction
                            else np.empty((0, 5), dtype=float)
                        )
                    gray = (
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        if config.flow_enabled
                        else None
                    )
                    with self.profiler.phase("track"):
                        tracks = tracker.update(
                            boxes,
                            frame_h=info.height,
                            frame_w=info.width,
                            frame_gray=gray,
                        )
                    run_stats["max_concurrent_tracks"] = max(
                        run_stats["max_concurrent_tracks"], len(tracks)
                    )
                    unique_tracks.update(
                        (scene_id, track.track_id) for track in tracks
                    )
                    ambiguous = ambiguous_track_ids(tracks)
                    active_ids = {track.track_id for track in tracks}
                    sam_quarantined_ids.intersection_update(active_ids)
                    sam_quarantined_ids.update(ambiguous)
                    sam_identity_reset = bool(ambiguous)
                    sam_masks = {}
                    if sam_window_reset or sam_identity_reset:
                        segmenter.reset()
                        run_stats["memory_resets"] += 1
                    prompts = (
                        self._sam_prompt_boxes(
                            tracks,
                            ambiguous,
                            sam_window_reset or sam_identity_reset,
                        )
                        if correction or sam_identity_reset
                        else {}
                    )
                    projected_ids = (
                        set(
                            getattr(
                                segmenter, "object_ids", frozenset()
                            )
                        )
                        | set(prompts)
                    )
                    sam_over_capacity = (
                        len(tracks) > SAM_MEMORY_MAX_OBJECTS
                        or len(projected_ids) > SAM_MEMORY_MAX_OBJECTS
                    )
                    if sam_over_capacity:
                        run_stats["capacity_fallback_frames"] += 1
                        segmenter.reset()
                    else:
                        sam_masks = segmenter.track_frame(
                            frame,
                            prompts,
                            config.sam_mask_expansion,
                            "mask-only",
                        )
                    if frame.nbytes > SAM_REVERSE_CACHE_LIMIT_BYTES:
                        frame_window.clear()
                        frame_window_bytes = 0
                    else:
                        cached_frame = frame.copy()
                        if (
                            frame_window.maxlen
                            and len(frame_window) == frame_window.maxlen
                        ):
                            frame_window_bytes -= frame_window[0][2].nbytes
                        frame_window.append(
                            (frame_index, scene_id, cached_frame)
                        )
                        frame_window_bytes += cached_frame.nbytes
                        run_stats["peak_reverse_cache_bytes"] = max(
                            run_stats["peak_reverse_cache_bytes"],
                            frame_window_bytes,
                        )
                        while (
                            len(frame_window) > 1
                            and frame_window_bytes
                            > SAM_REVERSE_CACHE_LIMIT_BYTES
                        ):
                                frame_window_bytes -= (
                                    frame_window.popleft()[2].nbytes
                                )
                    for track in tracks:
                        is_new_track = not store.has_track(
                            scene_id, track.track_id
                        )
                        if track.track_id in ambiguous:
                            sam_mask_stats["ambiguous"] += 1
                            raw_mask = None
                        elif track.track_id not in sam_masks:
                            sam_mask_stats["missing_from_memory"] += 1
                            raw_mask = None
                        else:
                            raw_mask = sam_masks[track.track_id]
                        raw_has_foreground = (
                            raw_mask is not None
                            and np.any(np.asarray(raw_mask) > 0)
                        )
                        if (
                            track.track_id not in ambiguous
                            and track.track_id in sam_masks
                            and not raw_has_foreground
                        ):
                            rejection = getattr(
                                segmenter, "last_rejections", {}
                            ).get(track.track_id)
                            if rejection == "low_object_score":
                                reason = "low_object_score"
                            elif rejection in {
                                "missing_object_score",
                                "invalid_object_score",
                            }:
                                reason = "invalid_object_score"
                            else:
                                reason = "empty_mask"
                            sam_mask_stats[reason] += 1
                            raw_mask = None
                        elif (
                            raw_mask is not None
                            and not self._sam_mask_matches_track(
                                raw_mask, track.box
                            )
                        ):
                            sam_mask_stats["quality_rejected"] += 1
                            raw_mask = None
                        elif raw_mask is not None:
                            sam_mask_stats["accepted"] += 1
                        if (
                            raw_mask is not None
                            and track.track_id in sam_quarantined_ids
                            and track.observed_box is not None
                        ):
                            sam_quarantined_ids.discard(track.track_id)
                        small_mask = (
                            cv2.resize(
                                raw_mask,
                                (proxy.shape[1], proxy.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                            if raw_mask is not None
                            and np.any(np.asarray(raw_mask) > 0)
                            else None
                        )
                        store.add_track(
                            frame_index,
                            scene_id,
                            track.track_id,
                            track.box,
                            track.coverage_boxes,
                            track.confidence,
                            threshold,
                            track.observed_box is not None,
                            track.track_id in ambiguous,
                            small_mask is None,
                            small_mask,
                        )
                        if (
                            is_new_track
                            and track.confidence >= threshold
                            and track.track_id not in ambiguous
                            and not sam_over_capacity
                        ):
                            run_stats["reverse_backfilled_frames"] += (
                                self._store_sam_reverse_masks(
                                    store,
                                    segmenter,
                                    frame_window,
                                    scene_id,
                                    track,
                                    threshold,
                                    frame.shape[:2],
                                )
                            )
                    if segmenter.last_error and warning_count < 3:
                        print(
                            "[WARN] Segmentation failed during analysis; "
                            f"geometry will cover affected tracks: {segmenter.last_error}"
                        )
                        warning_count += 1
                    if segmenter.last_error:
                        run_stats["segmenter_error_frames"] += 1
                    run_stats["prompts"] += len(prompts)
                    if frame_index % 32 == 0:
                        store.commit()
                    frame_index += 1
                    predicted_tracks = sum(
                        track.observed_box is None for track in tracks
                    )
                    fallback_observations = (
                        sum(sam_mask_stats.values())
                        - sam_mask_stats["accepted"]
                    )
                    self._show_analysis_progress(
                        frame_index,
                        info.total_frames,
                        "analyze",
                        overall_start=0,
                        overall_end=40,
                        started_at=analyze_started,
                        overall_started_at=analyze_started,
                        details=lambda: (
                            f"active={len(tracks)} "
                            f"detected={len(boxes)} "
                            f"predicted={predicted_tracks} "
                            f"corrections={run_stats['correction_frames']} "
                            f"prompts={run_stats['prompts']} "
                            f"masks(valid/fallback)="
                            f"{sam_mask_stats['accepted']}/"
                            f"{fallback_observations} "
                            f"temp={store.current_bytes / 1024**2:.1f}MiB"
                        ),
                    )
                if frame_index == 0:
                    raise VideoInputError(
                        "input opened but did not yield a video frame"
                    )
                store.commit()
                analyze_seconds = time.perf_counter() - analyze_started
                print(
                    "SAM mask observations: "
                    + ", ".join(
                        f"{name}={value}"
                        for name, value in sam_mask_stats.items()
                    )
                )
                (
                    analyzed_total,
                    analyzed_segmented,
                    analyzed_fallback,
                ) = store.mask_stats()
                print(
                    "Analyzed mask records: "
                    f"segmented={analyzed_segmented}, "
                    f"geometric_fallback={analyzed_fallback}, "
                    f"total={analyzed_total}"
                )
                accepted_percent = (
                    100.0 * analyzed_segmented / analyzed_total
                    if analyzed_total
                    else 0.0
                )
                print(
                    "SAM analysis summary: "
                    f"scenes={scene_id + 1}, tracks={len(unique_tracks)}, "
                    f"max_concurrent={run_stats['max_concurrent_tracks']}, "
                    f"correction_frames={run_stats['correction_frames']}, "
                    f"prompts={run_stats['prompts']}, "
                    f"memory_resets={run_stats['memory_resets']}, "
                    f"valid_masks={accepted_percent:.1f}%"
                )
                print(
                    "SAM recovery summary: "
                    f"reverse_backfilled={run_stats['reverse_backfilled_frames']}, "
                    f"capacity_fallback_frames="
                    f"{run_stats['capacity_fallback_frames']}, "
                    f"error_frames={run_stats['segmenter_error_frames']}, "
                    f"reverse_cache_peak="
                    f"{run_stats['peak_reverse_cache_bytes'] / 1024**2:.1f} MiB"
                )
                print("Temporal: stabilizing analyzed masks")
                stabilize_started = time.perf_counter()
                stabilize_tracks = max(1, store.track_key_count())

                def track_stabilize_progress(current, total):
                    if current <= stabilize_tracks:
                        pass_name = "reverse-repair"
                        pass_current = current
                    else:
                        pass_name = "forward-hysteresis"
                        pass_current = current - stabilize_tracks

                    def track_details():
                        records, segmented, fallback = store.mask_stats()
                        return (
                            f"pass={pass_name} "
                            f"tracks={pass_current}/{stabilize_tracks} "
                            f"records(segmented/fallback/total)="
                            f"{segmented}/{fallback}/{records} "
                            f"temp={store.current_bytes / 1024**2:.1f}MiB"
                        )

                    self._show_analysis_progress(
                        current,
                        total,
                        "stabilize",
                        overall_start=40,
                        overall_end=48,
                        started_at=stabilize_started,
                        overall_started_at=analyze_started,
                        rate_label="track-passes/s",
                        details=track_details,
                    )

                TemporalMaskStabilizer(
                    config.backfill_frames,
                    config.release_hold_frames,
                    config.mask_scale,
                ).process(
                    store,
                    progress=track_stabilize_progress,
                )
                stabilize_seconds = time.perf_counter() - stabilize_started
                total_records, segmented_records, fallback_records = (
                    store.mask_stats()
                )
                if any(store.mask_read_failures.values()):
                    print(
                        "Temporal mask read failures: "
                        + ", ".join(
                            f"{name}={value}"
                            for name, value in store.mask_read_failures.items()
                        )
                    )
                print(
                    "Temporal mask records: "
                    f"segmented={segmented_records}, "
                    f"geometric_fallback={fallback_records}, "
                    f"total={total_records}"
                )
                print("Temporal: composing and smoothing final frame masks")
                compose_started = time.perf_counter()

                def compose_progress(current, total):
                    self._show_analysis_progress(
                        current,
                        total,
                        "compose",
                        overall_start=48,
                        overall_end=50,
                        started_at=compose_started,
                        overall_started_at=analyze_started,
                        rate_label="frames/s",
                        details=lambda: (
                            f"merged-frames={current}/{total} "
                            f"temp={store.current_bytes / 1024**2:.1f}MiB"
                        ),
                    )

                self._compose_temporal_masks(
                    store,
                    frame_index,
                    progress=compose_progress,
                )
                compose_seconds = time.perf_counter() - compose_started
                final_started = time.perf_counter()

                def final_progress(current, total):
                    pass_frames = max(1, total // 2)
                    if current <= pass_frames:
                        pass_name = "reverse"
                        pass_current = current
                    else:
                        pass_name = "forward"
                        pass_current = current - pass_frames
                    self._show_analysis_progress(
                        current,
                        total,
                        "final-stabilize",
                        overall_start=50,
                        overall_end=70,
                        started_at=final_started,
                        overall_started_at=analyze_started,
                        rate_label="pass-frames/s",
                        details=lambda: (
                            f"pass={pass_name} "
                            f"frames={pass_current}/{pass_frames} "
                            f"temp={store.current_bytes / 1024**2:.1f}MiB"
                        ),
                    )

                FinalMaskStabilizer(
                    config.backfill_frames,
                    config.release_hold_frames,
                ).process(
                    store,
                    progress=final_progress,
                )
                final_seconds = time.perf_counter() - final_started
                print("Temporal: stabilization complete; starting render")
                render_started = time.perf_counter()
                self._render_temporal(
                    store,
                    info,
                    frame_index,
                    overall_started_at=analyze_started,
                )
                render_seconds = time.perf_counter() - render_started
                print(
                    "Temporary storage: "
                    f"current={store.current_bytes / 1024**2:.1f} MiB; "
                    f"peak={store.peak_bytes / 1024**2:.1f} MiB; "
                    f"image-data-written="
                    f"{store.image_bytes_written / 1024**2:.1f} MiB; "
                    f"limit={config.temporal_storage_limit_mb} MiB"
                )
                print(
                    "Phase timing: "
                    f"initialize={initialization_seconds:.1f}s, "
                    f"analyze={analyze_seconds:.1f}s, "
                    f"track-stabilize={stabilize_seconds:.1f}s, "
                    f"compose={compose_seconds:.1f}s, "
                    f"final-stabilize={final_seconds:.1f}s, "
                    f"render={render_seconds:.1f}s"
                )

        total_wall = time.perf_counter() - wall_start
        print(f"\nDone -> {config.output}")
        print(
            f"Frames: {frame_index} | elapsed: {total_wall:.1f}s | "
            f"average: {frame_index / total_wall:.1f}fps"
        )
        summary = self.profiler.summary(frame_index, total_wall)
        if config.profile and summary:
            print(summary)

    @staticmethod
    def _scale_box(
        box,
        source_shape: tuple[int, int],
        target_shape: tuple[int, int],
    ) -> list[int]:
        source_h, source_w = source_shape
        target_h, target_w = target_shape
        return [
            int(round(box[0] * target_w / source_w)),
            int(round(box[1] * target_h / source_h)),
            int(round(box[2] * target_w / source_w)),
            int(round(box[3] * target_h / source_h)),
        ]

    def _compose_temporal_masks(
        self,
        store,
        frame_count: int,
        progress=None,
    ) -> None:
        """Merge all Track IDs before a final scene-level smoothing pass."""
        config = self.config
        if progress is not None:
            progress(0, max(1, frame_count))
        for frame_index in range(frame_count):
            scene_id, proxy, source_shape = store.frame(frame_index)
            canvas = np.zeros((*proxy.shape[:2], 3), dtype=np.uint8)
            for record in store.records_for_frame(frame_index):
                mask = store.load_mask(record)
                if record.fallback or mask is None:
                    for box in record.coverage_boxes:
                        apply_mask_preview(
                            canvas,
                            self._scale_box(
                                box, source_shape, proxy.shape[:2]
                            ),
                            1,
                            mask_scale=config.mask_scale,
                            mask_shape=config.mask_shape,
                        )
                    continue
                proxy_box = self._scale_box(
                    record.box, source_shape, proxy.shape[:2]
                )
                if not self._sam_mask_matches_track(mask, proxy_box):
                    for box in record.coverage_boxes:
                        apply_mask_preview(
                            canvas,
                            self._scale_box(
                                box, source_shape, proxy.shape[:2]
                            ),
                            1,
                            mask_scale=config.mask_scale,
                            mask_shape=config.mask_shape,
                        )
                    continue
                render_mask = mask.copy()
                geometry = np.zeros(proxy.shape[:2], dtype=np.uint8)
                for box in record.coverage_boxes:
                    _add_box_to_mask(
                        geometry,
                        self._scale_box(
                            box, source_shape, proxy.shape[:2]
                        ),
                    )
                if config.segmentation_combine == "union":
                    render_mask = cv2.bitwise_or(render_mask, geometry)
                elif config.segmentation_combine == "intersection":
                    render_mask = cv2.bitwise_and(render_mask, geometry)
                    geometry_area = np.count_nonzero(geometry)
                    if (
                        geometry_area
                        and np.count_nonzero(render_mask) / geometry_area < 0.25
                    ):
                        render_mask[:] = 0
                bounds = _mask_bounds(render_mask)
                if bounds is None:
                    for box in record.coverage_boxes:
                        apply_mask_preview(
                            canvas,
                            self._scale_box(
                                box, source_shape, proxy.shape[:2]
                            ),
                            1,
                            mask_scale=config.mask_scale,
                            mask_shape=config.mask_shape,
                        )
                    continue
                x1, y1, x2, y2 = bounds
                apply_mask_preview(
                    canvas,
                    bounds,
                    1,
                    mask_scale=1.0,
                    coverage_mask=render_mask[y1:y2, x1:x2],
                    segmentation_combine="intersection",
                )
            store.replace_composite_mask(
                frame_index, scene_id, canvas[:, :, 0]
            )
            if progress is not None:
                progress(frame_index + 1, max(1, frame_count))
        store.commit()

    def _render_temporal(
        self,
        store,
        info,
        frame_count: int,
        *,
        overall_started_at: float | None = None,
    ) -> None:
        config = self.config
        with VideoSource(config.input) as source:
            with FFmpegEncoder(
                config.output,
                info.width,
                info.height,
                info.fps,
                config.input,
                use_nvenc=config.use_nvenc,
                overwrite=config.overwrite,
                ffmpeg_exe=config.ffmpeg,
                include_audio=not config.mask_preview,
                temporary_directory=config.job_temp_dir,
            ) as encoder:
                render_backend = "GPU (OpenCV CUDA)" if HAS_CUDA else "CPU"
                print(f"Render:  {render_backend}; encode={encoder.codec}")
                if config.mask_preview:
                    print(
                        "Mode:    MASK PREVIEW "
                        "(black=clear, blue=masked; no audio)"
                    )
                render_effect = (
                    apply_mask_preview if config.mask_preview else apply_blur
                )
                render_progress_started = time.perf_counter()
                for frame_index in range(frame_count):
                    with self.profiler.phase("read"):
                        ok, frame = source.read()
                    if not ok:
                        raise VideoInputError(
                            "input ended before the analyzed frame count"
                        )
                    if config.mask_preview:
                        frame[:] = 0
                    composite = store.load_composite_mask(frame_index)
                    if composite is not None:
                        render_mask = resize_mask_to_source(
                            composite, frame.shape[:2]
                        )
                        bounds = _mask_bounds(render_mask)
                        if bounds is not None:
                            x1, y1, x2, y2 = bounds
                            frame_records = store.records_for_frame(
                                frame_index
                            )
                            kernel = max(
                                (
                                    config.blur_kernel_for(
                                        record.confidence,
                                        record.threshold,
                                    )
                                    for record in frame_records
                                ),
                                default=config.blur_kernel_for(1.0, 0.0),
                            )
                            render_effect(
                                frame,
                                bounds,
                                kernel,
                                mask_scale=1.0,
                                coverage_mask=render_mask[
                                    y1:y2, x1:x2
                                ],
                                segmentation_combine="intersection",
                            )
                    else:
                        self._render_temporal_records(
                            frame, store, frame_index, render_effect
                        )
                    with self.profiler.phase("write"):
                        encoder.write(frame.tobytes())
                    self._show_analysis_progress(
                        frame_index + 1,
                        frame_count,
                        "render",
                        overall_start=70,
                        overall_end=100,
                        started_at=render_progress_started,
                        overall_started_at=overall_started_at,
                    )

    def _render_temporal_records(
        self, frame, store, frame_index: int, render_effect
    ) -> None:
        """Fail closed from per-track records if a composite file is unreadable."""
        config = self.config
        for record in store.records_for_frame(frame_index):
            kernel = config.blur_kernel_for(
                record.confidence, record.threshold
            )
            mask = store.load_mask(record)
            if record.fallback or mask is None:
                for box in record.coverage_boxes:
                    render_effect(
                        frame,
                        box,
                        kernel,
                        mask_scale=config.mask_scale,
                        mask_shape=config.mask_shape,
                    )
                continue
            render_mask = resize_mask_to_source(mask, frame.shape[:2])
            geometry = np.zeros(frame.shape[:2], dtype=np.uint8)
            for box in record.coverage_boxes:
                _add_box_to_mask(geometry, box)
            if config.segmentation_combine == "union":
                render_mask = cv2.bitwise_or(render_mask, geometry)
            elif config.segmentation_combine == "intersection":
                render_mask = cv2.bitwise_and(render_mask, geometry)
                geometry_area = np.count_nonzero(geometry)
                if (
                    geometry_area
                    and np.count_nonzero(render_mask) / geometry_area < 0.25
                ):
                    render_mask[:] = 0
            bounds = _mask_bounds(render_mask)
            if bounds is None:
                # Fail closed even for intersection.
                for box in record.coverage_boxes:
                    render_effect(
                        frame,
                        box,
                        kernel,
                        mask_scale=config.mask_scale,
                        mask_shape=config.mask_shape,
                    )
                continue
            x1, y1, x2, y2 = bounds
            render_effect(
                frame,
                bounds,
                kernel,
                mask_scale=1.0,
                mask_shape=config.mask_shape,
                coverage_mask=render_mask[y1:y2, x1:x2],
                segmentation_combine="intersection",
            )

    def _show_analysis_progress(
        self,
        current: int,
        total: int | None,
        phase: str,
        *,
        overall_start: int,
        overall_end: int,
        started_at: float | None = None,
        overall_started_at: float | None = None,
        rate_label: str = "fps",
        details=None,
    ) -> None:
        now = time.perf_counter()
        if total:
            phase_fraction = min(1.0, max(0.0, current / total))
            overall_value = overall_start + (
                phase_fraction * (overall_end - overall_start)
            )
            overall_percent = min(100, int(overall_value))
            # Across every phase, emit at roughly two overall percentage
            # points. This avoids dozens of duplicate lines for stages that
            # occupy only a small slice of the unified progress bar.
            bucket = min(50, int(overall_value // 2))
            previous = self._phase_progress.get(phase)
            # A ten-second heartbeat still updates a long-running unit without
            # flooding the log.
            if (
                previous is not None
                and bucket <= previous[0]
                and current < total
                and now - previous[1] < 10.0
            ):
                return
            self._phase_progress[phase] = (bucket, now)
            line = (
                f"  {current}/{total} ({overall_percent}%) phase={phase}"
            )
        else:
            previous = self._phase_progress.get(phase)
            if (
                current % 100
                and previous is not None
                and now - previous[1] < 10.0
            ):
                return
            self._phase_progress[phase] = (current // 100, now)
            line = f"  {current} frames phase={phase}"

        if started_at is not None:
            elapsed = max(0.0, now - started_at)
            speed = current / elapsed if elapsed else 0.0
            line += f" {speed:.1f}{rate_label}"
        if (
            total
            and overall_started_at is not None
            and overall_value > 0
        ):
            overall_elapsed = max(0.0, now - overall_started_at)
            overall_fraction = min(1.0, overall_value / 100.0)
            total_eta = (
                overall_elapsed * (1.0 - overall_fraction) / overall_fraction
            )
            line += f" total-ETA {total_eta:.0f}s"
        if details is not None:
            extra = details() if callable(details) else str(details)
            if extra:
                line += f" {extra}"
        print(line)

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
