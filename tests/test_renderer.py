import unittest

import numpy as np

from blurface.renderer import apply_blur_cpu


class RendererValidationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
