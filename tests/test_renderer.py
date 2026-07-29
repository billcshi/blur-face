import unittest
from unittest.mock import patch

import cv2
import numpy as np

from blurface.renderer import (
    _coverage_mask,
    apply_blur_cpu,
    apply_mask_preview,
    draw_debug_box,
    prepare_mask_region,
)


class RendererValidationTests(unittest.TestCase):
    def test_rounded_rectangle_never_cuts_the_detection_box(self):
        inner = (20, 15, 80, 65)
        mask = _coverage_mask(100, 80, inner, "rounded-rect")
        self.assertTrue(np.all(mask[15:65, 20:80] == 255))
        self.assertEqual(mask[0, 0], 0)

    def test_rectangle_covers_the_full_expanded_region(self):
        mask = _coverage_mask(100, 80, (20, 15, 80, 65), "rectangle")
        self.assertTrue(np.all(mask == 255))

    def test_ellipse_remains_available_as_legacy_shape(self):
        mask = _coverage_mask(100, 80, (20, 15, 80, 65), "ellipse")
        self.assertEqual(mask[0, 0], 0)
        self.assertEqual(mask[40, 50], 255)

    def test_mask_preview_paints_only_final_coverage_blue(self):
        frame = np.zeros((40, 50, 3), dtype=np.uint8)
        coverage = np.zeros((20, 20), dtype=np.uint8)
        coverage[5:15, 4:16] = 255
        apply_mask_preview(
            frame,
            [10, 10, 30, 30],
            251,
            mask_scale=1.0,
            coverage_mask=coverage,
            segmentation_combine="mask-only",
        )
        self.assertTrue(np.all(frame[15:25, 14:26] == (255, 0, 0)))
        self.assertTrue(np.all(frame[:10] == 0))
        self.assertEqual(
            int(np.count_nonzero(np.any(frame != 0, axis=2))),
            int(np.count_nonzero(coverage)),
        )

    def test_custom_mask_cannot_remove_the_detector_core(self):
        frame = np.arange(80 * 100 * 3, dtype=np.uint8).reshape(80, 100, 3)
        before = frame.copy()
        bbox = [20, 20, 60, 60]
        bounds, inner = prepare_mask_region(frame, bbox, 1.5)
        bx1, by1, bx2, by2 = bounds
        custom = np.zeros((by2 - by1, bx2 - bx1), dtype=np.uint8)
        apply_blur_cpu(
            frame,
            bbox,
            51,
            mask_scale=1.5,
            coverage_mask=custom,
        )
        ix1, iy1, ix2, iy2 = inner
        core = frame[by1 + iy1 : by1 + iy2, bx1 + ix1 : bx1 + ix2]
        old_core = before[by1 + iy1 : by1 + iy2, bx1 + ix1 : bx1 + ix2]
        self.assertFalse(np.array_equal(core, old_core))
        self.assertTrue(np.array_equal(frame[:by1], before[:by1]))

    def test_intersection_mode_does_not_restore_the_detector_core(self):
        frame = np.arange(80 * 100 * 3, dtype=np.uint8).reshape(80, 100, 3)
        before = frame.copy()
        bbox = [20, 20, 60, 60]
        bounds, _inner = prepare_mask_region(frame, bbox, 1.5)
        bx1, by1, bx2, by2 = bounds
        custom = np.zeros((by2 - by1, bx2 - bx1), dtype=np.uint8)
        custom[20:30, 20:30] = 255
        apply_blur_cpu(
            frame,
            bbox,
            51,
            mask_scale=1.5,
            coverage_mask=custom,
            segmentation_combine="intersection",
        )
        self.assertTrue(np.array_equal(frame[by1 + 5, bx1 + 5], before[by1 + 5, bx1 + 5]))
        self.assertFalse(np.array_equal(frame[by1 + 25, bx1 + 25], before[by1 + 25, bx1 + 25]))

    def test_mask_only_preserves_contour_outside_detector_core(self):
        frame = np.arange(80 * 100 * 3, dtype=np.uint8).reshape(80, 100, 3)
        before = frame.copy()
        bbox = [30, 25, 60, 55]
        bounds, inner = prepare_mask_region(frame, bbox, 1.5)
        bx1, by1, bx2, by2 = bounds
        custom = np.zeros((by2 - by1, bx2 - bx1), dtype=np.uint8)
        ix1, iy1, _ix2, _iy2 = inner
        custom[max(0, iy1 - 5) : iy1, ix1 : ix1 + 5] = 255
        apply_blur_cpu(
            frame,
            bbox,
            51,
            mask_scale=1.5,
            coverage_mask=custom,
            segmentation_combine="mask-only",
        )
        self.assertFalse(
            np.array_equal(
                frame[by1 + iy1 - 3, bx1 + ix1 + 2],
                before[by1 + iy1 - 3, bx1 + ix1 + 2],
            )
        )
        self.assertTrue(
            np.array_equal(
                frame[by1 + iy1 + 8, bx1 + ix1 + 8],
                before[by1 + iy1 + 8, bx1 + ix1 + 8],
            )
        )

    def test_rejects_invalid_privacy_parameters(self):
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            apply_blur_cpu(frame, [10, 10, 40, 40], 0)
        with self.assertRaises(ValueError):
            apply_blur_cpu(frame, [10, 10, 40, 40], 51, mask_scale=0.5)
        with self.assertRaises(ValueError):
            apply_blur_cpu(frame, [10, np.nan, 40, 40], 51)

    def test_uses_actual_frame_bounds(self):
        frame = np.arange(30 * 30 * 3, dtype=np.uint8).reshape(30, 30, 3)
        apply_blur_cpu(
            frame,
            [-100, -100, 100, 100],
            10,
            frame_w=9999,
            frame_h=9999,
        )
        self.assertEqual(frame.shape, (30, 30, 3))

    def test_debug_label_shows_detection_confidence(self):
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        with patch(
            "blurface.renderer.cv2.putText", wraps=cv2.putText
        ) as put_text:
            draw_debug_box(frame, [10, 20, 40, 50], 2, confidence=0.876)
        self.assertIn("C:0.88", put_text.call_args_list[0].args[1])


if __name__ == "__main__":
    unittest.main()
