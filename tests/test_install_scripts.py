import unittest
from pathlib import Path


class InstallScriptTests(unittest.TestCase):
    def test_project_is_installed_in_editable_mode(self):
        root = Path(__file__).resolve().parents[1]
        for script in ("init.bat", "init.sh"):
            contents = (root / script).read_text(encoding="utf-8")
            self.assertIn("install --no-deps --editable .", contents)

    def test_local_ui_launchers_use_the_project_environment(self):
        root = Path(__file__).resolve().parents[1]
        windows = (root / "start-ui.bat").read_text(encoding="utf-8")
        unix = (root / "start-ui.sh").read_text(encoding="utf-8")
        self.assertIn(".venv\\Scripts\\python.exe", windows)
        self.assertIn("-m blurface.webui", windows)
        self.assertIn(".venv/bin/python", unix)
        self.assertIn("-m blurface.webui", unix)

    def test_default_setup_installs_only_yunet_detector(self):
        root = Path(__file__).resolve().parents[1]
        contents = (root / "scripts" / "download_models.py").read_text(
            encoding="utf-8"
        )
        defaults = contents.split("DEFAULT_MODELS =", 1)[1].split(")", 1)[0]
        self.assertIn("face_detection_yunet_2023mar.onnx", defaults)
        self.assertNotIn("yolo", defaults.lower())

    def test_sam2_support_is_an_explicit_optional_install(self):
        root = Path(__file__).resolve().parents[1]
        windows = (root / "install-sam2.bat").read_text(encoding="utf-8")
        self.assertIn("requirements.sam2.lock", windows)
        self.assertNotIn("\npython -c", windows)
        self.assertIn('"%VENV_PYTHON%" -c "import torch', windows)
        self.assertIn(
            "requirements.sam2.lock",
            (root / "install-sam2.sh").read_text(encoding="utf-8"),
        )
        for script in ("init.bat", "init.sh"):
            self.assertNotIn(
                "requirements.sam2.lock",
                (root / script).read_text(encoding="utf-8"),
            )

    def test_windows_init_offers_explicit_sam_choice(self):
        root = Path(__file__).resolve().parents[1]
        windows = (root / "init.bat").read_text(encoding="utf-8")
        self.assertIn("Install optional SAM 2.1 support now?", windows)
        self.assertIn("BLUR_FACE_INSTALL_SAM", windows)
        self.assertIn("BLUR_FACE_SKIP_SAM_INSTALL", windows)
        self.assertIn("call install-sam2.bat", windows)
        self.assertIn('if /I "%CI%"=="true" goto SAM_SKIPPED', windows)

    def test_windows_sam_installer_does_not_depend_on_captured_timers(self):
        root = Path(__file__).resolve().parents[1]
        windows = (root / "install-sam2.bat").read_text(encoding="utf-8")
        self.assertNotIn("SETUP_START", windows)
        self.assertNotIn("STAGE_START", windows)
        self.assertIn("SAM setup complete", windows)

    def test_init_rebuilds_environment_and_explains_optional_detector(self):
        root = Path(__file__).resolve().parents[1]
        for script in ("init.bat", "init.sh"):
            contents = (root / script).read_text(encoding="utf-8")
            cleanup = contents.index("scripts")
            create = contents.index("-m venv .venv")
            self.assertLess(cleanup, create)
            self.assertIn("uninstall.py", contents)
            self.assertIn("Optional detector", contents)
            self.assertIn("install-yolo", contents)
            self.assertNotIn("MIT/Apache-compatible", contents)

    def test_windows_init_offers_explicit_external_ffmpeg_install(self):
        root = Path(__file__).resolve().parents[1]
        windows = (root / "init.bat").read_text(encoding="utf-8")
        unix = (root / "init.sh").read_text(encoding="utf-8")
        self.assertIn("choice /C YN", windows)
        self.assertIn("winget install --id Gyan.FFmpeg --exact", windows)
        self.assertIn("GPL-licensed", windows)
        self.assertIn("BLUR_FACE_SKIP_FFMPEG_INSTALL", windows)
        self.assertNotIn("winget install", unix)
        self.assertIn("package manager", unix)

    def test_ci_explicitly_provisions_external_ffmpeg(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (
            root / ".github" / "workflows" / "ci.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("apt-get install -y ffmpeg", workflow)
        self.assertIn("choco install ffmpeg --yes --no-progress", workflow)
        self.assertIn("ffmpeg -version", workflow)


if __name__ == "__main__":
    unittest.main()
