import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from blurface.model_store import resolve_model


class ModelStoreTests(unittest.TestCase):
    def test_finds_model_in_project_models_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory) / "models"
            model_directory.mkdir()
            model = model_directory / "yolov11m-face.pt"
            model.write_bytes(b"local model")
            resolved = resolve_model(
                "yolov11m-face.pt",
                allow_download=False,
                search_directories=(model_directory,),
            )
            self.assertEqual(resolved, str(model.resolve()))

    def test_known_missing_model_uses_verified_downloader(self):
        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory) / "models"
            expected = model_directory / "yolov11m-face.pt"
            with patch(
                "blurface.model_store.download_model", return_value=expected
            ) as download:
                resolved = resolve_model(
                    "yolov11m-face.pt",
                    allow_download=True,
                    search_directories=(model_directory,),
                )
            download.assert_called_once_with("yolov11m-face.pt", model_directory)
            self.assertEqual(resolved, str(expected))

    def test_offline_mode_rejects_missing_model(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(FileNotFoundError),
        ):
            resolve_model(
                "yolov11m-face.pt",
                allow_download=False,
                search_directories=(Path(directory),),
            )


if __name__ == "__main__":
    unittest.main()
