import unittest
from pathlib import Path

from blurface import __version__
from blurface.cli import build_parser, parse_time_thresh
from blurface.config import AppConfig, ConfigError


class ConfigTests(unittest.TestCase):
    def test_release_version(self):
        self.assertEqual(__version__, "1.2.0")

    def test_release_version_is_consistent_in_metadata_and_documentation(self):
        root = Path(__file__).resolve().parents[1]
        expected = __version__
        checks = {
            "pyproject.toml": f'version = "{expected}"',
            "CHANGELOG.md": f"## {expected} —",
            "README.md": f"Version {expected}",
            "README.zh.md": f"{expected} 版",
            "docs/image-batch-processing.md": f"Version {expected}",
        }
        for relative, marker in checks.items():
            with self.subTest(path=relative):
                contents = (root / relative).read_text(encoding="utf-8")
                self.assertIn(marker, contents)

    def test_privacy_defaults_use_strong_blur(self):
        config = AppConfig(Path("in.mp4"))
        self.assertEqual(config.blur_strategy, "adaptive")
        self.assertEqual(config.blur_kernel_min, 101)
        self.assertEqual(config.blur_kernel, 251)
        self.assertEqual(config.mask_scale, 1.5)
        self.assertEqual(config.mask_shape, "rounded-rect")
        self.assertEqual(config.mask_engine, "geometric")
        self.assertEqual(config.detector, "yunet")
        self.assertEqual(config.sam_mask_expansion, 0.12)
        self.assertEqual(config.segmentation_combine, "union")
        self.assertEqual(
            config.sam2_model, "facebook/sam2.1-hiera-base-plus"
        )
        self.assertEqual(config.sam2_refresh_interval, 15)
        self.assertTrue(config.temporal_stabilization)
        self.assertEqual(config.device, "auto")
        self.assertFalse(config.mask_preview)
        self.assertEqual(config.backfill_frames, 10)
        self.assertEqual(config.release_hold_frames, 5)
        self.assertEqual(config.scene_cut_sensitivity, 0.55)
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

    def test_blur_strength_scales_above_a_1080p_short_edge(self):
        config = AppConfig(
            Path("in.mp4"), blur_strategy="fixed", blur_kernel=251
        )
        for shape in ((1080, 1920), (1920, 1080), (720, 1280)):
            with self.subTest(shape=shape):
                self.assertEqual(
                    config.blur_kernel_for_frame(0.9, 0.3, shape), 251
                )
        for shape in ((2160, 3840), (3840, 2160)):
            with self.subTest(shape=shape):
                self.assertEqual(
                    config.blur_kernel_for_frame(0.9, 0.3, shape), 503
                )

    def test_frame_relative_blur_rejects_invalid_dimensions(self):
        config = AppConfig(Path("in.mp4"))
        with self.assertRaisesRegex(ConfigError, "height and width"):
            config.blur_kernel_for_frame(0.9, 0.3, (1080,))
        with self.assertRaisesRegex(ConfigError, "positive"):
            config.blur_kernel_for_frame(0.9, 0.3, (0, 1920))

    def test_mask_only_segmentation_policy_is_valid(self):
        config = AppConfig(
            Path("in.mp4"), segmentation_combine="mask-only"
        ).validate()
        self.assertEqual(config.segmentation_combine, "mask-only")

    def test_time_thresholds_are_sorted(self):
        self.assertEqual(parse_time_thresh("10:0.4,0:0.2"), ((0.0, 0.2), (10.0, 0.4)))

    def test_rejects_fail_open_render_values(self):
        with self.assertRaises(ConfigError):
            AppConfig(Path("in.mp4"), mask_scale=0.5).validate()
        with self.assertRaises(ConfigError):
            AppConfig(Path("in.mp4"), mask_shape="unsafe").validate()
        with self.assertRaises(ConfigError):
            AppConfig(Path("in.mp4"), mask_engine="unsafe").validate()
        with self.assertRaises(ConfigError):
            AppConfig(Path("in.mp4"), mask_engine="face-parsing").validate()
        with self.assertRaises(ConfigError):
            AppConfig(
                Path("in.mp4"), mask_engine="sam2.1", sam2_model=""
            ).validate()
        with self.assertRaises(ConfigError):
            AppConfig(Path("in.mp4"), sam2_refresh_interval=0).validate()
        with self.assertRaises(ConfigError):
            AppConfig(Path("in.mp4"), backfill_frames=61).validate()
        with self.assertRaises(ConfigError):
            AppConfig(Path("in.mp4"), release_hold_frames=13).validate()
        with self.assertRaises(ConfigError):
            AppConfig(Path("in.mp4"), scene_cut_sensitivity=1.1).validate()
        with self.assertRaises(ConfigError):
            AppConfig(
                Path("in.mp4"), temporal_storage_limit_mb=63
            ).validate()
        with self.assertRaises(ConfigError):
            AppConfig(Path("in.mp4"), sam_mask_expansion=0.6).validate()
        with self.assertRaises(ConfigError):
            AppConfig(
                Path("in.mp4"), segmentation_combine="difference"
            ).validate()
        with self.assertRaises(ConfigError):
            AppConfig(Path("in.mp4"), model="").validate()
        with self.assertRaises(ConfigError):
            AppConfig(Path("in.mp4"), detector="remote").validate()
        with self.assertRaises(ConfigError):
            AppConfig(
                Path("in.mp4"), detector="yunet", model="face.pt"
            ).validate()
        with self.assertRaises(ConfigError):
            AppConfig(
                Path("in.mp4"), detector="yolo", model="face.onnx"
            ).validate()
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

    def test_debug_and_mask_preview_are_mutually_exclusive(self):
        with self.assertRaises(ConfigError):
            AppConfig(
                Path("in.mp4"), debug=True, mask_preview=True
            ).validate()

    def test_numeric_track_exclusion_options_are_removed(self):
        options = build_parser()._option_string_actions
        self.assertNotIn("--exclude-ids", options)
        self.assertNotIn("--allow-unsafe-exclusions", options)

    def test_cli_requires_explicit_yolo_backend(self):
        options = build_parser()._option_string_actions
        self.assertIn("--detector", options)
        self.assertEqual(options["--detector"].default, "yunet")

    def test_existing_output_requires_overwrite(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.mp4"
            output.touch()
            with self.assertRaises(ConfigError):
                AppConfig(Path("in.mp4"), output=output).validate()
            AppConfig(Path("in.mp4"), output=output, overwrite=True).validate()

if __name__ == "__main__":
    unittest.main()
