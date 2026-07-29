"""Opt-in smoke test for the installed public Transformers SAM 2 video API."""

import os
from pathlib import Path
import unittest

import numpy as np

from blurface.sam2_segmenter import Sam2VideoSegmenter


_MODEL = os.environ.get("BLUR_FACE_REAL_SAM2_MODEL", "").strip()


@unittest.skipUnless(
    _MODEL,
    "set BLUR_FACE_REAL_SAM2_MODEL to a local SAM 2 model directory",
)
class RealSam2ContractTests(unittest.TestCase):
    def test_local_model_streams_prompt_then_memory_frame(self):
        model = Path(_MODEL).expanduser()
        self.assertTrue(model.is_dir(), model)
        segmenter = Sam2VideoSegmenter(model, device="cpu", offline=True)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[18:46, 18:46] = (128, 160, 192)

        prompted = segmenter.track_frame(
            frame,
            {17: (18, 18, 46, 46)},
            dilation_ratio=0,
            combine="mask-only",
        )
        propagated = segmenter.track_frame(
            frame,
            {},
            dilation_ratio=0,
            combine="mask-only",
        )

        self.assertIn(17, prompted)
        self.assertIn(17, propagated)
        self.assertIsNotNone(prompted[17])
        self.assertIsNotNone(propagated[17])
        self.assertGreater(np.count_nonzero(prompted[17]), 0)
        self.assertGreater(np.count_nonzero(propagated[17]), 0)
        self.assertIsNone(segmenter.last_error)
        self.assertEqual(segmenter.frame_count, 2)


if __name__ == "__main__":
    unittest.main()
