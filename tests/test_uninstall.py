import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "uninstall.py"
SPEC = importlib.util.spec_from_file_location("blur_face_uninstall", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class UninstallTests(unittest.TestCase):
    def test_default_preserves_models_and_user_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").touch()
            (root / "blur-face.py").touch()
            (root / ".venv").mkdir()
            (root / "build").mkdir()
            (root / "blurface" / "__pycache__").mkdir(parents=True)
            (root / "blurface" / "__pycache__" / "module.pyc").touch()
            (root / "models").mkdir()
            (root / "models" / "custom.pt").write_bytes(b"model")
            (root / "output.mp4").write_bytes(b"video")

            MODULE.uninstall(root)

            self.assertFalse((root / ".venv").exists())
            self.assertFalse((root / "build").exists())
            self.assertFalse((root / "blurface" / "__pycache__").exists())
            self.assertTrue((root / "models" / "custom.pt").is_file())
            self.assertTrue((root / "output.mp4").is_file())

    def test_models_require_explicit_option(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").touch()
            (root / "blur-face.py").touch()
            (root / "models").mkdir()
            MODULE.uninstall(root, remove_models=True)
            self.assertFalse((root / "models").exists())


if __name__ == "__main__":
    unittest.main()
