"""
blurface.detector — YOLO face detector wrapper.
"""

import os
import time

import numpy as np


def _resolve_model(path: str, allow_download: bool = True) -> str:
    """Resolve a model locally, optionally leaving download to Ultralytics."""
    if os.path.isfile(path):
        return os.path.abspath(path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_model = os.path.abspath(
        os.path.join(script_dir, "..", "models", os.path.basename(path))
    )
    if os.path.isfile(project_model):
        return project_model
    if allow_download:
        return path
    raise FileNotFoundError(
        f"model not found locally: {path} (remove --offline to allow download)"
    )


class FaceDetector:
    """Wraps Ultralytics YOLO for face detection.

    Auto-searches models/ subdirectory if model not found at path.

    Usage:
        detector = FaceDetector("yolov11m-face.pt", device="cuda")
        boxes = detector.detect(frame, conf=0.2)  # -> np.ndarray (N,4) or empty
    """

    def __init__(
        self, model_path: str, device: str = "cuda", allow_download: bool = True
    ):
        t0 = time.time()
        resolved = _resolve_model(model_path, allow_download=allow_download)
        # Import lazily so `blur-face --help` works before optional ML
        # dependencies are installed.
        from ultralytics import YOLO

        self.model = YOLO(resolved)
        self.device = self._resolve_device(device)
        load_time = time.time() - t0
        print(f"Loaded {resolved} ({load_time:.1f}s)")

    def _resolve_device(self, device: str) -> str:
        if device != "cuda":
            return device
        try:
            import torch

            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
                print(f"[GPU] {gpu}")
                return "cuda"
        except Exception as exc:  # noqa: BLE001 - optional backend probe
            print(f"[WARN] CUDA probe failed: {exc}")
        print("[WARN] CUDA not available, falling back to CPU")
        return "cpu"

    def detect(self, frame, conf: float = 0.2):
        """Return (N, 4) array of [x1,y1,x2,y2], or empty array."""
        results = self.model.predict(
            frame, device=self.device, conf=conf, verbose=False
        )[0]
        if results.boxes is not None and len(results.boxes) > 0:
            return results.boxes.xyxy.cpu().numpy()
        return np.empty((0, 4))
