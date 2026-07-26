"""
blurface.detector — YOLO face detector wrapper.
"""

import time

import numpy as np

from .model_store import resolve_model


def _resolve_model(path: str, allow_download: bool = True) -> str:
    """Compatibility wrapper around the shared model store."""
    return resolve_model(path, allow_download=allow_download)


class FaceDetector:
    """Wraps Ultralytics YOLO for face detection.

    Auto-searches models/ subdirectory if model not found at path.

    Usage:
        detector = FaceDetector("yolov11m-face.pt", device="cuda")
        detections = detector.detect(frame, conf=0.2)  # -> [x1,y1,x2,y2,confidence]
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
                print(
                    f"[GPU] {gpu} "
                    f"(torch={torch.__version__}, runtime CUDA={torch.version.cuda})"
                )
                return "cuda"
            print(
                "[WARN] PyTorch cannot access CUDA "
                f"(torch={torch.__version__}, runtime CUDA={torch.version.cuda})"
            )
        except Exception as exc:  # noqa: BLE001 - optional backend probe
            print(f"[WARN] CUDA probe failed: {exc}")
        print("[WARN] CUDA not available, falling back to CPU")
        return "cpu"

    def detect(self, frame, conf: float = 0.2):
        """Return (N, 5) rows of [x1,y1,x2,y2,confidence], or empty."""
        results = self.model.predict(
            frame, device=self.device, conf=conf, verbose=False
        )[0]
        if results.boxes is not None and len(results.boxes) > 0:
            coordinates = results.boxes.xyxy.cpu().numpy()
            confidence = results.boxes.conf.cpu().numpy().reshape(-1, 1)
            return np.concatenate((coordinates, confidence), axis=1)
        return np.empty((0, 5))
