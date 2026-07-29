import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from blurface.encoder import EncoderError, FFmpegEncoder


class EncoderFailureTests(unittest.TestCase):
    def test_mask_preview_encoder_can_omit_source_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = MagicMock()
            process.stdin.closed = False
            with (
                patch("blurface.encoder._resolve_ffmpeg", return_value="ffmpeg"),
                patch("blurface.encoder._can_encode_nvenc", return_value=False),
                patch("blurface.encoder.subprocess.Popen", return_value=process),
            ):
                encoder = FFmpegEncoder(
                    root / "preview.mp4",
                    2,
                    2,
                    30,
                    root / "private.mov",
                    use_nvenc=False,
                    include_audio=False,
                )
            self.assertIn("-an", encoder.cmd)
            self.assertNotIn(str(root / "private.mov"), encoder.cmd)
            encoder.abort()

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
