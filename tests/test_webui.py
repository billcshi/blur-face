import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from blurface.webui import _UI_FILE, build_command, discover_local_models


class WebUiTests(unittest.TestCase):
    def test_build_command_uses_validated_local_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.mp4"
            source.touch()
            output = root / "output.mp4"
            command = build_command(
                {
                    "input": str(source),
                    "output": str(output),
                    "model": str(root / "face-model.pt"),
                    "overwrite": True,
                    "threshold": 0.3,
                    "mask_scale": 1.5,
                    "mask_shape": "rectangle",
                    "blur_strategy": "fixed",
                    "blur_kernel": 301,
                    "blur_kernel_min": 101,
                    "min_face_size": 30,
                    "preset": "quality",
                    "offline": True,
                    "no_nvenc": True,
                }
            )
            self.assertIn("--mask-shape", command)
            self.assertIn("--model", command)
            self.assertIn(str(root / "face-model.pt"), command)
            self.assertIn("rectangle", command)
            self.assertIn("--blur-strategy", command)
            self.assertIn("fixed", command)
            self.assertIn("--offline", command)
            self.assertIn("--no-nvenc", command)
            self.assertNotIn("shell=True", command)

    def test_build_command_rejects_unknown_mask_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.mp4"
            source.touch()
            with self.assertRaises(ValueError):
                build_command(
                    {
                        "input": str(source),
                        "output": str(Path(directory) / "output.mp4"),
                        "threshold": 0.3,
                        "mask_scale": 1.5,
                        "mask_shape": "cutout",
                        "blur_strategy": "adaptive",
                        "blur_kernel": 251,
                        "blur_kernel_min": 101,
                        "min_face_size": 30,
                        "preset": "quality",
                    }
                )

    def test_ui_has_no_remote_assets_or_upload_control(self):
        html = _UI_FILE.read_text(encoding="utf-8")
        self.assertNotIn("https://", html)
        self.assertNotIn('type="file"', html)
        self.assertIn("127.0.0.1 ONLY", html)
        self.assertIn("navigator.languages", html)
        self.assertIn('data-lang-mode="auto"', html)
        self.assertIn('id="model"', html)
        self.assertIn('data-help="thresholdHelp"', html)
        self.assertIn('idle: "Waiting"', html)

    def test_readmes_include_the_current_ui_screenshot(self):
        root = Path(__file__).resolve().parents[1]
        screenshot = root / "docs" / "blur-face-local-studio.png"
        self.assertTrue(screenshot.is_file())
        self.assertEqual(screenshot.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        for readme in ("README.md", "README.zh.md"):
            contents = (root / readme).read_text(encoding="utf-8")
            self.assertIn("docs/blur-face-local-studio.png", contents)

    def test_discovers_models_from_local_model_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.pt"
            second = root / "second.pt"
            first.touch()
            second.touch()
            (root / "ignore.txt").touch()
            with patch("blurface.webui.model_directories", return_value=(root,)):
                self.assertEqual(
                    discover_local_models(),
                    [str(first.resolve()), str(second.resolve())],
                )


if __name__ == "__main__":
    unittest.main()
