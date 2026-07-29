import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class OptionalYoloInstallTests(unittest.TestCase):
    def test_base_runtime_does_not_install_ultralytics(self):
        requirements = (ROOT / "requirements.lock").read_text(encoding="utf-8")
        self.assertNotIn("ultralytics", requirements.lower())
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        base_dependencies = pyproject.split("[project.optional-dependencies]", 1)[0]
        self.assertNotIn("ultralytics", base_dependencies.lower())

    def test_both_initializers_require_an_explicit_yolo_choice(self):
        windows = (ROOT / "init.bat").read_text(encoding="utf-8")
        unix = (ROOT / "init.sh").read_text(encoding="utf-8")
        for contents in (windows, unix):
            self.assertIn("BLUR_FACE_INSTALL_YOLO", contents)
            self.assertIn("BLUR_FACE_SKIP_YOLO_INSTALL", contents)
            self.assertIn("AGPL-3.0", contents)
            self.assertIn("akanametov/yolo-face 1.0.0", contents)
            self.assertIn("GPL-3.0", contents)
            self.assertIn("WIDER FACE", contents)
            self.assertIn("install-yolo", contents)

    def test_standalone_installers_require_license_acceptance(self):
        windows = (ROOT / "install-yolo.bat").read_text(encoding="utf-8")
        unix = (ROOT / "install-yolo.sh").read_text(encoding="utf-8")
        for contents in (windows, unix):
            self.assertIn("BLUR_FACE_YOLO_LICENSE_ACCEPTED", contents)
            self.assertIn("THIRD_PARTY_NOTICES.md", contents)
            self.assertIn("akanametov/yolo-face 1.0.0", contents)
            self.assertIn("GPL-3.0", contents)
            self.assertIn("WIDER FACE", contents)
            self.assertIn("requirements.yolo.lock", contents)
            self.assertIn("download_models.py", contents)
            self.assertIn("--yolo", contents)

    def test_notices_name_pinned_weight_license_and_training_data(self):
        for document in ("README.md", "README.zh.md", "THIRD_PARTY_NOTICES.md"):
            contents = (ROOT / document).read_text(encoding="utf-8")
            self.assertIn("akanametov/yolo-face", contents)
            self.assertIn("GPL-3.0", contents)
            self.assertIn("Enterprise", contents)
            self.assertIn("WIDER FACE", contents)


if __name__ == "__main__":
    unittest.main()
