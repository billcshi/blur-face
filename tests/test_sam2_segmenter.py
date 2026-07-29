import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from blurface.sam2_segmenter import (
    Sam2Error,
    Sam2Segmenter,
    Sam2VideoSegmenter,
    build_safe_sam_mask,
)


class Sam2SegmenterTests(unittest.TestCase):
    def test_safe_mask_always_fills_detector_core(self):
        predicted = np.zeros((12, 16), dtype=np.uint8)
        predicted[1:4, 2:6] = 1
        mask = build_safe_sam_mask(
            predicted,
            (60, 80),
            (25, 20, 55, 50),
            dilation_ratio=0.1,
        )
        self.assertEqual(mask.shape, (60, 80))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(np.all(mask[20:50, 25:55] == 255))
        self.assertTrue(np.any(mask[:20, :25] == 255))

    def test_intersection_keeps_only_segmented_pixels_inside_detector_core(self):
        predicted = np.zeros((60, 80), dtype=np.uint8)
        predicted[10:40, 15:45] = 1
        mask = build_safe_sam_mask(
            predicted,
            (60, 80),
            (25, 20, 55, 50),
            dilation_ratio=0,
            combine="intersection",
        )
        self.assertEqual(mask[25, 30], 255)
        self.assertEqual(mask[45, 30], 0)
        self.assertEqual(mask[25, 20], 0)

    def test_inference_failure_returns_none_for_geometric_fallback(self):
        segmenter = Sam2Segmenter.__new__(Sam2Segmenter)
        segmenter.last_error = None
        with patch.object(
            Sam2Segmenter,
            "_predict",
            side_effect=RuntimeError("synthetic SAM failure"),
        ):
            mask = segmenter.build_mask(
                np.zeros((30, 40, 3), dtype=np.uint8),
                (5, 5, 35, 25),
            )
        self.assertIsNone(mask)
        self.assertIn("synthetic SAM failure", segmenter.last_error)

    def test_missing_optional_dependency_is_diagnostic(self):
        with patch(
            "blurface.sam2_segmenter._imports",
            side_effect=Sam2Error("run install-sam2.bat"),
        ):
            with self.assertRaisesRegex(Sam2Error, "install-sam2"):
                Sam2Segmenter("facebook/sam2.1-hiera-small")

    def test_video_propagation_returns_raw_contour_without_a_new_box(self):
        predicted = np.zeros((40, 50), dtype=np.uint8)
        predicted[10:30, 15:35] = 1
        mask = Sam2VideoSegmenter._clean_full_mask(
            predicted,
            box=None,
            dilation_ratio=0,
            combine="mask-only",
        )
        self.assertEqual(mask[20, 20], 255)
        self.assertEqual(mask[0, 0], 0)
        self.assertFalse(np.all(mask == 255))
        # Dilation is scaled to the tracked object, not the full frame.
        expanded = Sam2VideoSegmenter._clean_full_mask(
            predicted,
            box=None,
            dilation_ratio=0.12,
            combine="mask-only",
        )
        self.assertEqual(expanded[0, 0], 0)
        self.assertLess(np.count_nonzero(expanded), expanded.size // 2)
        self.assertTrue(np.any(expanded[8:10, 15:35] == 255))

    def test_video_propagation_rejects_policy_without_current_box(self):
        predicted = np.zeros((40, 50), dtype=np.uint8)
        predicted[10:30, 15:35] = 1
        for combine in ("union", "intersection"):
            with self.subTest(combine=combine):
                with self.assertRaisesRegex(
                    ValueError, "current tracker geometry"
                ):
                    Sam2VideoSegmenter._clean_full_mask(
                        predicted,
                        box=None,
                        dilation_ratio=0,
                        combine=combine,
                    )

    def test_video_multi_object_masks_drop_the_singleton_channel(self):
        masks = np.zeros((3, 1, 40, 50), dtype=np.float32)
        normalized = Sam2VideoSegmenter._normalise_mask_batch(masks)
        self.assertEqual(normalized.shape, (3, 40, 50))

    def test_public_mask_builder_rejects_internal_combine_modes(self):
        with self.assertRaisesRegex(ValueError, "union or intersection"):
            build_safe_sam_mask(
                np.ones((10, 10), dtype=np.uint8),
                (10, 10),
                (2, 2, 8, 8),
                combine="mask-only",
            )

    def test_reverse_video_api_uses_official_iterator_contract(self):
        class Tensor:
            def __init__(self, values):
                self.values = np.asarray(values)

            def detach(self):
                return self

            def to(self, *_args, **_kwargs):
                return self

            def float(self):
                return self

            def numpy(self):
                return self.values

        class Processor:
            def __init__(self):
                self.session_options = None
                self.prompt_options = None

            def init_video_session(self, **kwargs):
                self.session_options = kwargs
                return object()

            def add_inputs_to_inference_session(self, **kwargs):
                self.prompt_options = kwargs

            def post_process_masks(self, *_args, **_kwargs):
                mask = np.zeros((1, 1, 20, 30), dtype=np.float32)
                mask[:, :, 5:15, 10:20] = 1
                return [Tensor(mask)]

        class Model:
            def __init__(self):
                self.reverse_options = None

            def __call__(self, **_kwargs):
                return object()

            def propagate_in_video_iterator(self, **kwargs):
                self.reverse_options = kwargs
                for frame_index in (2, 1, 0):
                    yield SimpleNamespace(
                        frame_idx=frame_index,
                        object_ids=[7],
                        pred_masks=Tensor(np.ones((1, 1, 4, 4))),
                        object_score_logits=Tensor([1.0]),
                    )

        segmenter = Sam2VideoSegmenter.__new__(Sam2VideoSegmenter)
        segmenter._torch = SimpleNamespace(
            float32="float32",
            bfloat16="bfloat16",
            inference_mode=lambda: nullcontext(),
            backends=SimpleNamespace(
                mkldnn=SimpleNamespace(flags=lambda **_kwargs: nullcontext())
            ),
        )
        segmenter.device = SimpleNamespace(type="cpu")
        segmenter._processor = Processor()
        segmenter._model = Model()
        segmenter.last_error = None
        frames = [
            np.zeros((20, 30, 3), dtype=np.uint8) for _ in range(3)
        ]
        masks = segmenter.reverse_propagate(
            frames, 7, (10, 5, 20, 15), dilation_ratio=0
        )
        self.assertEqual(set(masks), {0, 1, 2})
        self.assertTrue(segmenter._model.reverse_options["reverse"])
        self.assertEqual(
            segmenter._model.reverse_options["start_frame_idx"], 2
        )
        self.assertEqual(
            segmenter._processor.prompt_options["input_boxes"],
            [[[10, 5, 20, 15]]],
        )
        self.assertIn("video", segmenter._processor.session_options)

    def test_video_rejects_low_object_score_before_accepting_mask(self):
        class Tensor:
            def __init__(self, values):
                self.values = np.asarray(values)

            def detach(self):
                return self

            def to(self, *_args, **_kwargs):
                return self

            def float(self):
                return self

            def numpy(self):
                return self.values

            def __getitem__(self, index):
                return Tensor(self.values[index])

        class Session:
            obj_ids = [9]

            @staticmethod
            def get_obj_num():
                return 1

        class Processor:
            @staticmethod
            def add_inputs_to_inference_session(**_kwargs):
                return None

            @staticmethod
            def post_process_masks(*_args, **_kwargs):
                mask = np.zeros((1, 1, 20, 30), dtype=np.float32)
                mask[:, :, 5:15, 10:20] = 1
                return [Tensor(mask)]

            def __call__(self, **_kwargs):
                return SimpleNamespace(
                    pixel_values=Tensor(
                        np.zeros((1, 3, 20, 30), dtype=np.float32)
                    ),
                    original_sizes=Tensor([[20, 30]]),
                )

        class Model:
            @staticmethod
            def __call__(**_kwargs):
                return SimpleNamespace(
                    pred_masks=Tensor(np.ones((1, 1, 4, 4))),
                    object_ids=[9],
                    object_score_logits=Tensor([0.49]),
                )

        segmenter = Sam2VideoSegmenter.__new__(Sam2VideoSegmenter)
        segmenter._torch = SimpleNamespace(
            bfloat16="bfloat16",
            inference_mode=lambda: nullcontext(),
            backends=SimpleNamespace(
                mkldnn=SimpleNamespace(flags=lambda **_kwargs: nullcontext())
            ),
        )
        segmenter.device = SimpleNamespace(type="cpu")
        segmenter._processor = Processor()
        segmenter._model = Model()
        segmenter._session = Session()
        segmenter._frame_index = 0
        segmenter._needs_reseed = False
        segmenter.last_error = None
        segmenter.last_rejections = {}

        masks = segmenter.track_frame(
            np.zeros((20, 30, 3), dtype=np.uint8),
            {9: (10, 5, 20, 15)},
            dilation_ratio=0,
            combine="mask-only",
        )

        self.assertIsNone(masks[9])
        self.assertEqual(
            segmenter.last_rejections[9], "low_object_score"
        )

    def test_object_score_must_exist_be_finite_and_reach_threshold(self):
        cases = (
            (np.asarray([], dtype=np.float32), 0, "missing_object_score"),
            (np.asarray([np.nan]), 0, "invalid_object_score"),
            (np.asarray([np.inf]), 0, "invalid_object_score"),
            (np.asarray([-np.inf]), 0, "invalid_object_score"),
            (np.asarray([0.49]), 0, "low_object_score"),
            (np.asarray([0.50]), 0, None),
        )
        for scores, index, expected in cases:
            with self.subTest(scores=scores):
                self.assertEqual(
                    Sam2VideoSegmenter._score_rejection(scores, index),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
