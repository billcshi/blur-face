import unittest
from pathlib import Path

from blurface.cli import parse_time_thresh
from blurface.config import AppConfig, ConfigError


class ConfigTests(unittest.TestCase):
    def test_time_thresholds_are_sorted(self):
        self.assertEqual(parse_time_thresh("10:0.4,0:0.2"), ((0.0, 0.2), (10.0, 0.4)))

    def test_rejects_fail_open_render_values(self):
        with self.assertRaises(ConfigError):
            AppConfig(Path("in.mp4"), mask_scale=0.5).validate()
        with self.assertRaises(ConfigError):
            AppConfig(Path("in.mp4"), lost_buffer=-1).validate()
        with self.assertRaises(ConfigError):
            AppConfig(Path("in.mp4"), blur_kernel=0).validate()

    def test_rejects_same_input_and_output(self):
        with self.assertRaises(ConfigError):
            AppConfig(Path("same.mp4"), output=Path("./same.mp4")).validate()

    def test_existing_output_requires_overwrite(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.mp4"
            output.touch()
            with self.assertRaises(ConfigError):
                AppConfig(Path("in.mp4"), output=output).validate()
            AppConfig(Path("in.mp4"), output=output, overwrite=True).validate()

    def test_exclusions_require_explicit_risk_acknowledgement(self):
        with self.assertRaises(ConfigError):
            AppConfig(Path("in.mp4"), exclude_ids=frozenset({1})).validate()
        AppConfig(
            Path("in.mp4"),
            exclude_ids=frozenset({1}),
            allow_unsafe_exclusions=True,
        ).validate()


if __name__ == "__main__":
    unittest.main()
