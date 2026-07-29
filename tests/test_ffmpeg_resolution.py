import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from blurface.encoder import EncoderError, _resolve_ffmpeg


class FFmpegResolutionTests(unittest.TestCase):
    def test_explicit_ffmpeg_has_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "ffmpeg.exe"
            executable.touch()
            with patch("blurface.encoder.shutil.which", return_value="system"):
                self.assertEqual(_resolve_ffmpeg(executable), str(executable.resolve()))

    def test_system_ffmpeg_is_used(self):
        with patch("blurface.encoder.shutil.which", return_value="/system/ffmpeg"):
            self.assertEqual(_resolve_ffmpeg(), "/system/ffmpeg")

    def test_missing_system_ffmpeg_has_actionable_error(self):
        with (
            patch("blurface.encoder.shutil.which", return_value=None),
            self.assertRaisesRegex(EncoderError, "--ffmpeg"),
        ):
            _resolve_ffmpeg()


if __name__ == "__main__":
    unittest.main()
