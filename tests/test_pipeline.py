import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from blurface.config import AppConfig
from blurface.pipeline import VideoProcessor


class FakeDetector:
    def __init__(self, *_args, **_kwargs):
        self.index = 0

    def detect(self, _frame, conf=0.3):
        del conf
        x1 = 10 + self.index * 2
        self.index += 1
        return np.array([[x1, 10, x1 + 20, 30]], dtype=float)


class PipelineIntegrationTests(unittest.TestCase):
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
                model="unused.pt",
                flow_enabled=False,
                use_nvenc=False,
                offline=True,
                min_face_size=1,
            )
            with patch("blurface.pipeline.FaceDetector", FakeDetector):
                VideoProcessor(config).run()

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)
            self.assertEqual(list(output.parent.glob("*.partial*")), [])
            capture = cv2.VideoCapture(str(output))
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 8)
            capture.release()


if __name__ == "__main__":
    unittest.main()
