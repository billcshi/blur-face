import tempfile
import unittest
from pathlib import Path

from blurface.encoder import EncoderError, FFmpegEncoder


class EncoderFailureTests(unittest.TestCase):
    def test_ffmpeg_failure_is_raised_and_not_committed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "out.mp4"
            output.write_bytes(b"existing-safe-output")
            encoder = FFmpegEncoder(
                output,
                2,
                2,
                30,
                root / "missing-input.mov",
                use_nvenc=False,
                overwrite=True,
            )
            encoder.write(bytes(12))
            with self.assertRaises(EncoderError):
                encoder.close()
            self.assertEqual(output.read_bytes(), b"existing-safe-output")
            self.assertFalse(encoder.temp_path.exists())


if __name__ == "__main__":
    unittest.main()
