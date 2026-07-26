import unittest
from pathlib import Path

from blurface import __version__
from blurface.cli import parse_time_thresh
from blurface.config import AppConfig, ConfigError


class ConfigTests(unittest.TestCase):
    def test_stable_release_version(self):
        self.assertEqual(__version__, "1.0.0")

    def test_privacy_defaults_use_strong_blur(self):
        config = AppConfig(Path("in.mp4"))
        self.assertEqual(config.blur_strategy, "adaptive")
        self.assertEqual(config.blur_kernel_min, 101)
        self.assertEqual(config.blur_kernel, 251)
        self.assertEqual(config.mask_scale, 1.5)
        self.assertEqual(config.mask_shape, "rounded-rect")
        self.assertEqual(config.min_face_size, 30)

    def test_adaptive_blur_scales_with_confidence(self):
        config = AppConfig(Path("in.mp4"))
        self.assertEqual(config.blur_kernel_for(0.3, 0.3), 101)
        self.assertGreater(config.blur_kernel_for(0.6, 0.3), 101)
        self.assertLess(config.blur_kernel_for(0.6, 0.3), 251)
        self.assertEqual(config.blur_kernel_for(0.85, 0.3), 251)
        self.assertEqual(config.blur_kernel_for(0.99, 0.3), 251)

    def test_fixed_blur_ignores_confidence(self):
        config = AppConfig(
            Path("in.mp4"), blur_strategy="fixed", blur_kernel=51
        ).validate()
        self.assertEqual(config.blur_kernel_for(0.3, 0.3), 51)
        self.assertEqual(config.blur_kernel_for(0.99, 0.3), 51)

    def test_time_thresholds_are_sorted(self):
        self.assertEqual(parse_time_thresh("10:0.4,0:0.2"), ((0.0, 0.2), (10.0, 0.4)))

    def test_rejects_fail_open_render_values(self):
        with self.assertRaises(ConfigError):
            AppConfig(Path("in.mp4"), mask_scale=0.5).validate()
        with self.assertRaises(ConfigError):
            AppConfig(Path("in.mp4"), mask_shape="unsafe").validate()
        with self.assertRaises(ConfigError):
            AppConfig(Path("in.mp4"), model="").validate()
        with self.assertRaises(ConfigError):
            AppConfig(Path("in.mp4"), lost_buffer=-1).validate()
        with self.assertRaises(ConfigError):
            AppConfig(Path("in.mp4"), blur_kernel=0).validate()
        with self.assertRaises(ConfigError):
            AppConfig(
                Path("in.mp4"), blur_kernel=101, blur_kernel_min=151
            ).validate()

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
