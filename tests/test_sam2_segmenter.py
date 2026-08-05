import unittest
from contextlib import nullcontext
import logging
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from blurface.sam2_segmenter import (
    Sam2Error,
    Sam2Segmenter,
    Sam2VideoSegmenter,
    _ExpectedImageConfigWarningFilter,
    _load_pretrained,
    build_safe_sam_mask,
)


class Sam2SegmenterTests(unittest.TestCase):
    def test_only_expected_image_subset_warning_is_suppressed(self):
        warning_filter = _ExpectedImageConfigWarningFilter()
        expected = logging.LogRecord(
            "transformers.configuration_utils",
            logging.WARNING,
            __file__,
            1,
            "You are using a model of type `sam2_video` to instantiate a "
            "model of type `sam2`. This may be expected if you are loading "
            "a checkpoint that shares a subset of the architecture (e.g., "
            "loading a `sam2_video` checkpoint into `Sam2Model`), but is "
            "otherwise not supported and can yield errors. Please verify "
            "that the checkpoint is compatible with the model you are "
            "instantiating.",
            (),
            None,
        )
        unrelated = logging.LogRecord(
            "transformers.configuration_utils",
            logging.WARNING,
            __file__,
            1,
            "checkpoint weights are missing",
            (),
            None,
        )
        extended = logging.LogRecord(
            "transformers.configuration_utils",
            logging.WARNING,
            __file__,
            1,
            expected.getMessage() + " Additional diagnostic.",
            (),
            None,
        )
        error = logging.LogRecord(
            "transformers.configuration_utils",
            logging.ERROR,
            __file__,
            1,
            expected.getMessage(),
            (),
            None,
        )
        self.assertFalse(warning_filter.filter(expected))
        self.assertTrue(warning_filter.filter(unrelated))
        self.assertTrue(warning_filter.filter(extended))
        self.assertTrue(warning_filter.filter(error))

    def test_hf_components_prefer_cache_before_remote_download(self):
        class LocalEntryNotFoundError(FileNotFoundError):
            pass

        class Loader:
            calls = []
            cached = False

            @classmethod
            def from_pretrained(cls, _source, **options):
                cls.calls.append(options["local_files_only"])
                if options["local_files_only"] and not cls.cached:
                    try:
                        raise LocalEntryNotFoundError("not cached")
                    except LocalEntryNotFoundError as exc:
                        raise OSError("cache lookup failed") from exc
                return "loaded"

        self.assertEqual(_load_pretrained(Loader, "org/model", False), "loaded")
        self.assertEqual(Loader.calls, [True, False])
        Loader.calls = []
        Loader.cached = True
        self.assertEqual(_load_pretrained(Loader, "org/model", False), "loaded")
        self.assertEqual(Loader.calls, [True])
        Loader.calls = []
        Loader.cached = False
        with self.assertRaisesRegex(OSError, "cache lookup failed"):
            _load_pretrained(Loader, "org/model", True)
        self.assertEqual(Loader.calls, [True])

    def test_hf_cache_error_does_not_trigger_remote_download(self):
        class Loader:
            calls = []

            @classmethod
            def from_pretrained(cls, _source, **options):
                cls.calls.append(options["local_files_only"])
                raise ValueError("invalid local configuration")

        with self.assertRaisesRegex(ValueError, "invalid local configuration"):
            _load_pretrained(Loader, "org/model", False)
        self.assertEqual(Loader.calls, [True])

    @staticmethod
    def _image_prediction_segmenter(mask, scores):
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

        class Inputs(dict):
            def to(self, _device):
                return self

        class Processor:
            def __call__(self, **_kwargs):
                return Inputs(original_sizes=Tensor([[8, 10]]))

            def post_process_masks(self, *_args, **_kwargs):
                return [np.asarray(mask)]

        class Model:
            def __call__(self, **_kwargs):
                return SimpleNamespace(
                    pred_masks=Tensor(np.asarray(mask)),
                    iou_scores=Tensor(np.asarray(scores)),
                )

        segmenter = Sam2Segmenter.__new__(Sam2Segmenter)
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
        return segmenter

    def test_single_image_mask_still_requires_one_valid_iou_score(self):
        roi = np.zeros((8, 10, 3), dtype=np.uint8)
        mask = np.ones((8, 10), dtype=np.float32)
        for scores in ([], [0.49], [np.nan], [0.9, 0.8]):
            with self.subTest(scores=scores):
                segmenter = self._image_prediction_segmenter(mask, scores)
                with self.assertRaisesRegex(ValueError, "scores"):
                    segmenter._predict(roi, (1, 1, 9, 7))
        segmenter = self._image_prediction_segmenter(mask, [0.9])
        np.testing.assert_array_equal(
            segmenter._predict(roi, (1, 1, 9, 7)), mask
        )

    def test_post_processed_image_mask_must_match_roi_shape(self):
        roi = np.zeros((60, 60, 3), dtype=np.uint8)
        segmenter = self._image_prediction_segmenter(
            np.array([[1, 1], [1, 0]], dtype=np.float32),
            [0.9],
        )
        contour = segmenter.build_contour(
            roi,
            (10, 10, 50, 50),
            dilation_ratio=0,
        )
        self.assertIsNone(contour)
        self.assertIn("does not match ROI shape", segmenter.last_error)

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

    def test_image_contour_api_does_not_apply_combination_policy(self):
        segmenter = Sam2Segmenter.__new__(Sam2Segmenter)
        segmenter.last_error = None
        predicted = np.zeros((30, 40), dtype=np.uint8)
        predicted[2:8, 3:10] = 1
        with patch.object(Sam2Segmenter, "_predict", return_value=predicted):
            contour = segmenter.build_contour(
                np.zeros((30, 40, 3), dtype=np.uint8),
                (15, 12, 35, 28),
                dilation_ratio=0,
            )
        self.assertEqual(contour[4, 5], 255)
        self.assertEqual(contour[20, 20], 0)

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
