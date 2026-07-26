import unittest
from unittest.mock import patch

import cv2
import numpy as np

from blurface.renderer import _coverage_mask, apply_blur_cpu, draw_debug_box


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
