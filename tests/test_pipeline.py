import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from blurface.config import AppConfig
from blurface.pipeline import VideoProcessor
from blurface.renderer import apply_blur


class FakeDetector:
    def __init__(self, *_args, **_kwargs):
        self.index = 0

    def detect(self, _frame, conf=0.3):
        del conf
        x1 = 10 + self.index * 2
        confidence = min(0.3 + self.index * 0.1, 0.95)
        self.index += 1
        return np.array([[x1, 10, x1 + 20, 30, confidence]], dtype=float)


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


if __name__ == "__main__":
    unittest.main()
