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


if __name__ == "__main__":
    unittest.main()
