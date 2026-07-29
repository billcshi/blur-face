import io
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from blurface.config import AppConfig
from blurface.pipeline import (
    VideoProcessor,
    _add_box_to_mask,
    _mask_bounds,
)
from blurface.renderer import apply_blur, apply_mask_preview
from blurface.temporal import ambiguous_track_ids


class FakeDetector:
    total_calls = 0

    def __init__(self, *_args, **_kwargs):
        self.index = 0

    def detect(self, _frame, conf=0.3):
        del conf
        self.__class__.total_calls += 1
        x1 = 10 + self.index * 2
        confidence = min(0.3 + self.index * 0.1, 0.95)
        self.index += 1
        return np.array([[x1, 10, x1 + 20, 30, confidence]], dtype=float)


class StaticDetector:
    def __init__(self, *_args, **_kwargs):
        pass

    def detect(self, _frame, conf=0.3):
        del conf
        return np.array([[10, 10, 30, 30, 0.95]], dtype=float)


class FakeSam:
    instances = []

    def __init__(self, *args, **kwargs):
        self.calls = 0
        self.last_error = None
        self.init_args = args
        self.init_kwargs = kwargs
        self.video_ids = set()
        self.video_boxes = {}
        self.__class__.instances.append(self)

    def build_mask(self, roi, inner, dilation, combine):
        del inner, dilation, combine
        self.calls += 1
        if self.calls == 1:
            return np.full(roi.shape[:2], 255, dtype=np.uint8)
        if self.calls == 2:
            return np.zeros(roi.shape[:2], dtype=np.uint8)
        self.last_error = "synthetic SAM miss"
        return None

    def track_frame(self, frame, prompts, dilation, combine):
        del dilation, combine
        self.calls += 1
        self.video_ids.update(prompts)
        self.video_boxes.update(prompts)
        masks = {}
        for track_id in self.video_ids:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            x1, y1, x2, y2 = self.video_boxes[track_id]
            mask[y1:y2, x1:x2] = 255
            masks[track_id] = mask
        return masks


class DriftingPropagationSam:
    instances = []

    def __init__(self, *_args, **_kwargs):
        self.last_error = None
        self.prompts = []
        self.combines = []
        self.__class__.instances.append(self)

    def track_frame(self, frame, prompts, dilation, combine):
        del dilation
        self.prompts.append(dict(prompts))
        self.combines.append(combine)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        # A single connected SAM object covers its correct Track and drifts
        # twelve pixels beyond the current geometry toward another person.
        mask[10:30, 10:42] = 255
        return {0: mask}

    def reset(self):
        pass


class LateDetector:
    def __init__(self, *_args, **_kwargs):
        self.index = 0

    def detect(self, _frame, conf=0.3):
        del conf
        index = self.index
        self.index += 1
        if index < 5:
            return np.empty((0, 5), dtype=float)
        return np.array([[28, 18, 48, 38, 0.95]], dtype=float)


class CutAwareDetector:
    def __init__(self, *_args, **_kwargs):
        pass

    def detect(self, frame, conf=0.3):
        del conf
        if float(frame.mean()) < 100:
            return np.empty((0, 5), dtype=float)
        return np.array([[20, 12, 40, 32, 0.95]], dtype=float)


class ResettableFakeSam(FakeSam):
    def reset(self):
        self.video_ids.clear()
        self.video_boxes.clear()


class StableVideoSam(FakeSam):
    def __init__(self, *_args, **_kwargs):
        super().__init__(*_args, **_kwargs)


class SparseVideoSam(FakeSam):
    def track_frame(self, frame, prompts, dilation, combine):
        del dilation, combine
        self.video_ids.update(prompts)
        masks = {}
        for track_id, box in prompts.items():
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            x1, y1, x2, y2 = box
            mask[y1 : y1 + 2, x1 : x1 + 2] = 255
            mask[y2 - 2 : y2, x2 - 2 : x2] = 255
            masks[track_id] = mask
        return masks


class ThinBarVideoSam(FakeSam):
    def track_frame(self, frame, prompts, dilation, combine):
        del dilation, combine
        self.video_ids.update(prompts)
        self.video_boxes.update(prompts)
        masks = {}
        for track_id in self.video_ids:
            x1, y1, x2, y2 = self.video_boxes[track_id]
            center = (x1 + x2) // 2
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            mask[y1:y2, center - 2 : center + 1] = 255
            masks[track_id] = mask
        return masks


class PipelineIntegrationTests(unittest.TestCase):
    def test_sam_progress_is_throttled_and_reports_useful_metrics(self):
        processor = VideoProcessor.__new__(VideoProcessor)
        processor._phase_progress = {}
        details_calls = 0

        def details():
            nonlocal details_calls
            details_calls += 1
            return (
                "active=2 detected=1 predicted=1 corrections=3 prompts=2 "
                "masks(valid/fallback)=20/1 temp=4.0MiB"
            )

        output = io.StringIO()
        with (
            patch("blurface.pipeline.time.perf_counter", return_value=10.0),
            redirect_stdout(output),
        ):
            for current in range(1, 16):
                processor._show_analysis_progress(
                    current,
                    1501,
                    "analyze",
                    overall_start=0,
                    overall_end=40,
                    started_at=0.0,
                    overall_started_at=0.0,
                    details=details,
                )
            processor._show_analysis_progress(
                76,
                1501,
                "analyze",
                overall_start=0,
                overall_end=40,
                started_at=0.0,
                overall_started_at=0.0,
                details=details,
            )

        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(details_calls, 2)
        self.assertIn("1/1501 (0%) phase=analyze", lines[0])
        self.assertIn("76/1501 (2%) phase=analyze", lines[1])
        self.assertIn("fps total-ETA", lines[1])
        self.assertIn("active=2 detected=1 predicted=1", lines[1])
        self.assertIn("masks(valid/fallback)=20/1", lines[1])

    def test_small_progress_phase_does_not_log_each_work_unit(self):
        processor = VideoProcessor.__new__(VideoProcessor)
        processor._phase_progress = {}
        output = io.StringIO()
        with (
            patch("blurface.pipeline.time.perf_counter", return_value=10.0),
            redirect_stdout(output),
        ):
            for current in range(29):
                processor._show_analysis_progress(
                    current,
                    28,
                    "stabilize",
                    overall_start=40,
                    overall_end=48,
                    overall_started_at=0.0,
                )
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 5)
        self.assertTrue(lines[0].startswith("  0/28 (40%)"))
        self.assertTrue(lines[-1].startswith("  28/28 (48%)"))
        self.assertTrue(all("total-ETA" in line for line in lines))

    def test_full_frame_mask_bounds_follow_sam_beyond_tracker_box(self):
        mask = np.zeros((60, 80), dtype=np.uint8)
        mask[35:45, 45:55] = 255
        self.assertEqual(_mask_bounds(mask), [45, 35, 55, 45])
        _add_box_to_mask(mask, [5, 10, 15, 20])
        self.assertEqual(_mask_bounds(mask), [5, 10, 55, 45])
        self.assertIsNone(_mask_bounds(mask[:, :, None]))

    def test_temporal_sam_memory_cannot_drift_to_another_track(self):
        mask = np.zeros((80, 120), dtype=np.uint8)
        mask[20:40, 20:40] = 255
        self.assertTrue(
            VideoProcessor._sam_mask_matches_track(
                mask, [22, 20, 42, 40]
            )
        )
        self.assertFalse(
            VideoProcessor._sam_mask_matches_track(
                mask, [80, 20, 100, 40]
            )
        )
        self.assertFalse(
            VideoProcessor._sam_mask_matches_track(
                mask, [38, 20, 58, 40]
            )
        )

    def test_sparse_components_cannot_fake_full_box_coverage(self):
        mask = np.zeros((80, 120), dtype=np.uint8)
        mask[20:22, 20:22] = 255
        mask[38:40, 38:40] = 255
        self.assertFalse(
            VideoProcessor._sam_mask_matches_track(
                mask, [20, 20, 40, 40]
            )
        )

    def test_centered_thin_strips_cannot_pass_sam_quality_gate(self):
        vertical = np.zeros((80, 120), dtype=np.uint8)
        vertical[20:40, 29:32] = 255
        horizontal = np.zeros_like(vertical)
        horizontal[29:32, 20:40] = 255
        for orientation, mask in (
            ("vertical", vertical),
            ("horizontal", horizontal),
        ):
            with self.subTest(orientation=orientation):
                self.assertFalse(
                    VideoProcessor._sam_mask_matches_track(
                        mask, [20, 20, 40, 40]
                    )
                )

    def test_ambiguous_crossing_ids_are_not_seeded_into_sam_memory(self):
        tracks = [
            SimpleNamespace(
                track_id=3,
                box=[10, 10, 30, 30],
                observed_box=[10, 10, 30, 30],
            ),
            SimpleNamespace(
                track_id=4,
                box=[25, 10, 45, 30],
                observed_box=[25, 10, 45, 30],
            ),
            SimpleNamespace(
                track_id=5,
                box=[60, 10, 80, 30],
                observed_box=[60, 10, 80, 30],
            ),
        ]
        prompts = VideoProcessor._sam_prompt_boxes(
            tracks, {3, 4}, include_predicted=True
        )
        self.assertEqual(set(prompts), {5})
        shared_mask = np.zeros((60, 90), dtype=np.uint8)
        shared_mask[15:30, 17:33] = 255
        self.assertFalse(
            VideoProcessor._sam_mask_matches_track(
                shared_mask, tracks[0].box
            )
        )
        self.assertFalse(
            VideoProcessor._sam_mask_matches_track(
                shared_mask, tracks[1].box
            )
        )
        self.assertEqual(ambiguous_track_ids(tracks[:2]), {3, 4})

    def test_synthetic_video_is_committed_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "nested" / "output.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48)
            )
            self.assertTrue(writer.isOpened())
            for _ in range(8):
                writer.write(np.full((48, 64, 3), 180, dtype=np.uint8))
            writer.release()

            config = AppConfig(
                input=source,
                output=output,
                model="unused.onnx",
                flow_enabled=False,
                use_nvenc=False,
                offline=True,
                min_face_size=1,
                profile=True,
            )
            output_log = io.StringIO()
            with (
                patch("blurface.pipeline.FaceDetector", FakeDetector),
                patch("blurface.pipeline.apply_blur", wraps=apply_blur) as blur,
                redirect_stdout(output_log),
            ):
                VideoProcessor(config).run()

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)
            self.assertEqual(list(output.parent.glob("*.partial*")), [])
            capture = cv2.VideoCapture(str(output))
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 8)
            capture.release()
            # Once smoothing lags a moving detection, both ellipses must be
            # rendered; one union ellipse can expose a face at its corners.
            self.assertGreater(blur.call_count, 8)
            kernels = {call.args[2] for call in blur.call_args_list}
            self.assertIn(101, kernels)
            self.assertIn(251, kernels)
            self.assertGreater(len(kernels), 2)
            # GitHub's Windows runner uses a legacy console encoding. Runtime
            # status text must remain printable there.
            output_log.getvalue().encode("cp1252")

    def test_mask_preview_contains_only_black_and_final_blue_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "mask-preview.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48)
            )
            for _ in range(4):
                writer.write(np.full((48, 64, 3), 180, dtype=np.uint8))
            writer.release()
            config = AppConfig(
                input=source,
                output=output,
                model="unused.onnx",
                mask_preview=True,
                flow_enabled=False,
                use_nvenc=False,
                offline=True,
                min_face_size=1,
            )
            output_log = io.StringIO()
            with (
                patch("blurface.pipeline.FaceDetector", FakeDetector),
                redirect_stdout(output_log),
            ):
                VideoProcessor(config).run()

            capture = cv2.VideoCapture(str(output))
            frames = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
            capture.release()
            self.assertEqual(len(frames), 4)
            for frame in frames:
                blue = (
                    (frame[:, :, 0] > 120)
                    & (frame[:, :, 0] > frame[:, :, 1] * 2)
                    & (frame[:, :, 0] > frame[:, :, 2] * 2)
                )
                self.assertTrue(blue.any())
                self.assertLess(float(frame[~blue].mean()), 12.0)
            self.assertIn("MASK PREVIEW", output_log.getvalue())

    def test_sam2_engine_uses_the_same_safe_mask_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "output.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48)
            )
            self.assertTrue(writer.isOpened())
            for _ in range(16):
                writer.write(
                    np.full((48, 64, 3), 180, dtype=np.uint8)
                )
            writer.release()
            config = AppConfig(
                input=source,
                output=output,
                model="unused.onnx",
                mask_engine="sam2.1",
                sam2_model="local-sam2",
                segmentation_combine="intersection",
                flow_enabled=False,
                use_nvenc=False,
                offline=True,
                min_face_size=1,
            )
            FakeSam.instances.clear()
            FakeDetector.total_calls = 0
            with (
                patch("blurface.pipeline.FaceDetector", FakeDetector),
                patch("blurface.pipeline.Sam2VideoSegmenter", FakeSam),
                patch(
                    "blurface.pipeline.apply_blur",
                    wraps=apply_blur,
                ) as blur,
                redirect_stdout(io.StringIO()),
            ):
                VideoProcessor(config).run()
            self.assertEqual(FakeSam.instances[0].init_args, ("local-sam2",))
            self.assertEqual(
                FakeSam.instances[0].init_kwargs,
                {"device": "auto", "offline": True},
            )
            self.assertEqual(FakeSam.instances[0].calls, 16)
            # The first three frames confirm the tracker; after that detection
            # runs only at the configured correction cadence.
            self.assertEqual(FakeDetector.total_calls, 4)
            self.assertTrue(
                any(
                    call.kwargs.get("coverage_mask") is not None
                    for call in blur.call_args_list
                )
            )

    def test_streaming_intersection_clips_unprompted_propagation_to_track(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "output.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48)
            )
            for _ in range(4):
                writer.write(np.full((48, 64, 3), 180, dtype=np.uint8))
            writer.release()
            config = AppConfig(
                input=source,
                output=output,
                model="unused.onnx",
                mask_engine="sam2.1",
                sam2_model="local-sam2",
                segmentation_combine="intersection",
                temporal_stabilization=False,
                sam2_refresh_interval=15,
                flow_enabled=False,
                use_nvenc=False,
                offline=True,
                min_face_size=1,
            )
            DriftingPropagationSam.instances.clear()
            with (
                patch("blurface.pipeline.FaceDetector", StaticDetector),
                patch(
                    "blurface.pipeline.Sam2VideoSegmenter",
                    DriftingPropagationSam,
                ),
                patch(
                    "blurface.pipeline.apply_blur", wraps=apply_blur
                ) as blur,
                redirect_stdout(io.StringIO()),
            ):
                VideoProcessor(config).run()

            segmenter = DriftingPropagationSam.instances[0]
            self.assertTrue(segmenter.prompts[0])
            self.assertEqual(segmenter.prompts[-1], {})
            self.assertEqual(segmenter.combines, ["mask-only"] * 4)
            self.assertEqual(blur.call_count, 4)
            for call in blur.call_args_list:
                x1, y1, x2, y2 = call.args[1]
                self.assertGreaterEqual(x1, 10)
                self.assertGreaterEqual(y1, 10)
                self.assertLessEqual(x2, 30)
                self.assertLessEqual(y2, 30)
                self.assertIsNotNone(
                    call.kwargs.get("coverage_mask")
                )

    def test_streaming_sparse_sam_mask_uses_geometric_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "output.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48)
            )
            writer.write(np.full((48, 64, 3), 180, dtype=np.uint8))
            writer.release()
            config = AppConfig(
                input=source,
                output=output,
                model="unused.onnx",
                mask_engine="sam2.1",
                sam2_model="local-sam2",
                segmentation_combine="mask-only",
                temporal_stabilization=False,
                flow_enabled=False,
                use_nvenc=False,
                offline=True,
                min_face_size=1,
            )
            with (
                patch("blurface.pipeline.FaceDetector", FakeDetector),
                patch(
                    "blurface.pipeline.Sam2VideoSegmenter", SparseVideoSam
                ),
                patch(
                    "blurface.pipeline.apply_blur", wraps=apply_blur
                ) as blur,
                redirect_stdout(io.StringIO()),
            ):
                VideoProcessor(config).run()
            self.assertTrue(blur.called)
            self.assertTrue(
                all(
                    call.kwargs.get("coverage_mask") is None
                    for call in blur.call_args_list
                )
            )

    def test_streaming_thin_bar_falls_back_for_non_union_modes(self):
        for combine in ("intersection", "mask-only"):
            with self.subTest(combine=combine), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source.mp4"
                output = root / "output.mp4"
                writer = cv2.VideoWriter(
                    str(source),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10,
                    (64, 48),
                )
                writer.write(np.full((48, 64, 3), 180, dtype=np.uint8))
                writer.release()
                config = AppConfig(
                    input=source,
                    output=output,
                    model="unused.onnx",
                    mask_engine="sam2.1",
                    sam2_model="local-sam2",
                    segmentation_combine=combine,
                    temporal_stabilization=False,
                    flow_enabled=False,
                    use_nvenc=False,
                    offline=True,
                    min_face_size=1,
                )
                with (
                    patch("blurface.pipeline.FaceDetector", FakeDetector),
                    patch(
                        "blurface.pipeline.Sam2VideoSegmenter",
                        ThinBarVideoSam,
                    ),
                    patch(
                        "blurface.pipeline.apply_blur", wraps=apply_blur
                    ) as blur,
                    redirect_stdout(io.StringIO()),
                ):
                    VideoProcessor(config).run()
                self.assertTrue(blur.called)
                self.assertTrue(
                    all(
                        call.kwargs.get("coverage_mask") is None
                        for call in blur.call_args_list
                    )
                )

    def test_streaming_sam_detects_the_first_frame_after_a_cut(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "output.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48)
            )
            for _ in range(5):
                writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            writer.write(np.full((48, 64, 3), 220, dtype=np.uint8))
            writer.release()
            config = AppConfig(
                input=source,
                output=output,
                model="unused.onnx",
                mask_engine="sam2.1",
                sam2_model="local-sam2",
                segmentation_combine="mask-only",
                temporal_stabilization=False,
                sam2_refresh_interval=15,
                flow_enabled=False,
                use_nvenc=False,
                offline=True,
                min_face_size=1,
            )
            with (
                patch("blurface.pipeline.FaceDetector", CutAwareDetector),
                patch(
                    "blurface.pipeline.Sam2VideoSegmenter",
                    ResettableFakeSam,
                ),
                patch(
                    "blurface.pipeline.apply_blur", wraps=apply_blur
                ) as blur,
                redirect_stdout(io.StringIO()),
            ):
                VideoProcessor(config).run()
            # Frame 5 is the final frame and is outside the global correction
            # cadence. A call therefore proves cut handling synchronously
            # detected and covered that first frame of the new scene.
            self.assertTrue(blur.called)
            self.assertTrue(
                any(call.args[0].mean() > 100 for call in blur.call_args_list)
            )

    def test_temporal_two_pass_repairs_late_detection_before_render(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "output.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (80, 60)
            )
            self.assertTrue(writer.isOpened())
            for index in range(6):
                value = 30 if index < 2 else 150
                frame = np.full((60, 80, 3), value, dtype=np.uint8)
                if index >= 2:
                    cv2.circle(frame, (38, 28), 9, (190, 190, 190), -1)
                writer.write(frame)
            writer.release()
            config = AppConfig(
                input=source,
                output=output,
                model="unused.onnx",
                mask_engine="sam2.1",
                sam2_model="local-sam2",
                sam2_refresh_interval=1,
                temporal_stabilization=True,
                backfill_frames=5,
                flow_enabled=False,
                use_nvenc=False,
                offline=True,
                min_face_size=1,
            )
            rendered_means = []

            def observed_blur(frame, *args, **kwargs):
                rendered_means.append(float(frame.mean()))
                return apply_blur(frame, *args, **kwargs)

            output_log = io.StringIO()
            with (
                patch("blurface.pipeline.FaceDetector", LateDetector),
                patch("blurface.pipeline.Sam2VideoSegmenter", StableVideoSam),
                patch(
                    "blurface.pipeline.apply_blur",
                    side_effect=observed_blur,
                ),
                redirect_stdout(output_log),
            ):
                VideoProcessor(config).run()
            self.assertTrue(output.is_file())
            # One detected frame plus frames 2-4 repaired before pass-2 render.
            self.assertGreaterEqual(len(rendered_means), 4)
            # The appearance gate stopped before the truly absent dark frames.
            self.assertTrue(all(value > 80 for value in rendered_means))
            log = output_log.getvalue()
            for phase in (
                "analyze",
                "stabilize",
                "compose",
                "final-stabilize",
                "render",
            ):
                self.assertIn(f"phase={phase}", log)
            overall_percentages = [
                int(value)
                for value in re.findall(
                    r"\((\d+)%\) phase=", log
                )
            ]
            self.assertEqual(
                overall_percentages, sorted(overall_percentages)
            )
            self.assertEqual(overall_percentages[-1], 100)
            phase_percentages = {}
            for percentage, phase in re.findall(
                r"\((\d+)%\) phase=([a-z-]+)", log
            ):
                phase_percentages.setdefault(phase, []).append(
                    int(percentage)
                )
            self.assertEqual(max(phase_percentages["analyze"]), 40)
            self.assertEqual(max(phase_percentages["stabilize"]), 48)
            self.assertEqual(max(phase_percentages["compose"]), 50)
            self.assertEqual(
                max(phase_percentages["final-stabilize"]), 70
            )
            self.assertIn("fps total-ETA", log)
            self.assertIn("masks(valid/fallback)=", log)
            self.assertIn("active=", log)
            self.assertIn("detected=", log)

    def test_temporal_sparse_sam_mask_falls_back_to_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "output.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48)
            )
            for _ in range(5):
                writer.write(np.full((48, 64, 3), 150, dtype=np.uint8))
            writer.release()
            config = AppConfig(
                input=source,
                output=output,
                model="unused.onnx",
                mask_engine="sam2.1",
                sam2_model="local-sam2",
                segmentation_combine="mask-only",
                temporal_stabilization=True,
                flow_enabled=False,
                use_nvenc=False,
                offline=True,
                min_face_size=1,
            )
            with (
                patch("blurface.pipeline.FaceDetector", FakeDetector),
                patch("blurface.pipeline.Sam2VideoSegmenter", SparseVideoSam),
                patch(
                    "blurface.pipeline.apply_blur", wraps=apply_blur
                ) as blur,
                redirect_stdout(io.StringIO()),
            ):
                VideoProcessor(config).run()
            segmented_calls = list(blur.call_args_list)
            self.assertTrue(segmented_calls)
            self.assertTrue(
                all(
                    (call.args[1][2] - call.args[1][0])
                    * (call.args[1][3] - call.args[1][1])
                    >= 400
                    for call in segmented_calls
                )
            )

    def test_temporal_thin_bar_falls_back_for_non_union_modes(self):
        for combine in ("intersection", "mask-only"):
            with self.subTest(combine=combine), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source.mp4"
                output = root / "output.mp4"
                writer = cv2.VideoWriter(
                    str(source),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10,
                    (64, 48),
                )
                for _ in range(3):
                    writer.write(
                        np.full((48, 64, 3), 150, dtype=np.uint8)
                    )
                writer.release()
                config = AppConfig(
                    input=source,
                    output=output,
                    model="unused.onnx",
                    mask_engine="sam2.1",
                    sam2_model="local-sam2",
                    segmentation_combine=combine,
                    temporal_stabilization=True,
                    flow_enabled=False,
                    use_nvenc=False,
                    offline=True,
                    min_face_size=1,
                )
                output_log = io.StringIO()
                with (
                    patch("blurface.pipeline.FaceDetector", FakeDetector),
                    patch(
                        "blurface.pipeline.Sam2VideoSegmenter",
                        ThinBarVideoSam,
                    ),
                    patch(
                        "blurface.pipeline.apply_blur", wraps=apply_blur
                    ) as blur,
                    redirect_stdout(output_log),
                ):
                    VideoProcessor(config).run()
                self.assertTrue(blur.called)
                self.assertTrue(
                    all(
                        call.kwargs.get("coverage_mask") is not None
                        and np.count_nonzero(
                            call.kwargs["coverage_mask"]
                        )
                        >= 400
                        for call in blur.call_args_list
                    )
                )
                self.assertIn(
                    "Analyzed mask records: segmented=0",
                    output_log.getvalue(),
                )

    def test_temporal_mask_preview_uses_stabilized_contour_on_black(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "preview.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48)
            )
            for _ in range(3):
                writer.write(np.full((48, 64, 3), 150, dtype=np.uint8))
            writer.release()
            config = AppConfig(
                input=source,
                output=output,
                model="unused.onnx",
                mask_engine="sam2.1",
                sam2_model="local-sam2",
                segmentation_combine="mask-only",
                temporal_stabilization=True,
                mask_preview=True,
                flow_enabled=False,
                use_nvenc=False,
                offline=True,
                min_face_size=1,
            )
            frames_before_paint = []

            def observed_preview(frame, *args, **kwargs):
                frames_before_paint.append(frame.copy())
                return apply_mask_preview(frame, *args, **kwargs)

            with (
                patch("blurface.pipeline.FaceDetector", FakeDetector),
                patch("blurface.pipeline.Sam2VideoSegmenter", StableVideoSam),
                patch(
                    "blurface.pipeline.apply_mask_preview",
                    side_effect=observed_preview,
                ) as preview,
                patch("blurface.pipeline.apply_blur") as blur,
                redirect_stdout(io.StringIO()),
            ):
                VideoProcessor(config).run()
            self.assertTrue(preview.called)
            blur.assert_not_called()
            self.assertTrue(
                all(not frame.any() for frame in frames_before_paint)
            )
            self.assertTrue(
                all(
                    call.kwargs.get("coverage_mask") is not None
                    and call.kwargs["segmentation_combine"] == "intersection"
                    for call in preview.call_args_list
                )
            )


if __name__ == "__main__":
    unittest.main()
