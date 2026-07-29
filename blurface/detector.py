"""Explicitly selected face-detector backends with one stable box API."""

import time

import cv2
import numpy as np

from .model_store import resolve_model


def _resolve_model(path: str, allow_download: bool = True) -> str:
    """Compatibility wrapper around the shared model store."""
    return resolve_model(path, allow_download=allow_download)


class DetectorError(RuntimeError):
    """Raised when an explicitly selected optional detector is unavailable."""


class _YuNetDetector:
    """Run the MIT-licensed YuNet model through OpenCV DNN."""

    def __init__(
        self, model_path: str, device: str = "auto", allow_download: bool = True
    ):
        t0 = time.time()
        resolved = _resolve_model(model_path, allow_download=allow_download)
        self.model = cv2.FaceDetectorYN.create(
            resolved,
            "",
            (320, 320),
            score_threshold=0.3,
            nms_threshold=0.3,
            top_k=5000,
        )
        self.device = "cpu"
        load_time = time.time() - t0
        print(f"Loaded OpenCV YuNet {resolved} ({load_time:.1f}s, CPU)")

    def detect(self, frame, conf: float = 0.2):
        """Return (N, 5) rows of [x1,y1,x2,y2,confidence], or empty."""
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("detector frame must be a BGR image")
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            return np.empty((0, 5))
        self.model.setInputSize((width, height))
        self.model.setScoreThreshold(float(conf))
        _retval, faces = self.model.detect(frame)
        if faces is not None and len(faces):
            faces = np.asarray(faces, dtype=float)
            coordinates = np.empty((len(faces), 4), dtype=float)
            coordinates[:, 0] = faces[:, 0]
            coordinates[:, 1] = faces[:, 1]
            coordinates[:, 2] = faces[:, 0] + faces[:, 2]
            coordinates[:, 3] = faces[:, 1] + faces[:, 3]
            return np.concatenate((coordinates, faces[:, -1:]), axis=1)
        return np.empty((0, 5))


class _YoloDetector:
    """Run an explicitly installed Ultralytics detector and local weight."""

    def __init__(
        self, model_path: str, device: str = "auto", allow_download: bool = True
    ):
        del allow_download
        t0 = time.time()
        try:
            # YOLO models are deliberately never downloaded during video
            # processing. The user must opt in through init/install-yolo and
            # thereby see the upstream AGPL/Enterprise license notice.
            resolved = _resolve_model(model_path, allow_download=False)
        except FileNotFoundError as exc:
            raise DetectorError(
                f"YOLO model not found locally: {model_path}. "
                "Run install-yolo.bat / ./install-yolo.sh, or select a local "
                ".pt model whose license you accept."
            ) from exc
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorError(
                "YOLO support is not installed. Run install-yolo.bat / "
                "./install-yolo.sh and accept the displayed upstream license."
            ) from exc

        self.model = YOLO(resolved)
        self.face_class_ids = self._validate_face_model()
        self.device = self._resolve_device(device)
        load_time = time.time() - t0
        print(
            f"Loaded optional Ultralytics YOLO {resolved} "
            f"({load_time:.1f}s, {self.device})"
        )

    def _validate_face_model(self) -> list[int]:
        task = str(getattr(self.model, "task", "")).lower()
        if task != "detect":
            raise DetectorError(
                f"YOLO model task must be detect, got {task or 'unknown'}"
            )
        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            entries = names.items()
        elif isinstance(names, (list, tuple)):
            entries = enumerate(names)
        else:
            entries = ()
        accepted_names = {"face", "human face", "human-face", "human_face"}
        face_class_ids = [
            int(class_id)
            for class_id, name in entries
            if str(name).strip().lower() in accepted_names
        ]
        if not face_class_ids:
            raise DetectorError(
                "Selected YOLO model has no explicit face class. "
                "Choose a face-detection weight; generic COCO or unrelated "
                "YOLO weights are unsafe for face anonymization."
            )
        return face_class_ids

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "cpu":
            return "cpu"
        try:
            import torch
        except ImportError as exc:
            raise DetectorError(
                "YOLO support requires PyTorch; rerun the optional YOLO installer."
            ) from exc

        if device in {"auto", "cuda"} or device.startswith("cuda:"):
            if torch.cuda.is_available():
                selected = "cuda:0" if device == "auto" else device
                device_index = (
                    int(selected.split(":", 1)[1])
                    if selected.startswith("cuda:")
                    else 0
                )
                print(
                    f"[GPU] {torch.cuda.get_device_name(device_index)} "
                    f"(torch={torch.__version__}, runtime CUDA={torch.version.cuda})"
                )
                return selected
            if device != "auto":
                print("[WARN] Requested CUDA is unavailable; YOLO is using CPU")
            return "cpu"
        if device == "mps":
            available = bool(
                getattr(getattr(torch, "backends", None), "mps", None)
                and torch.backends.mps.is_available()
            )
            if available:
                return "mps"
            print("[WARN] Requested MPS is unavailable; YOLO is using CPU")
            return "cpu"
        return device

    def detect(self, frame, conf: float = 0.2):
        """Return (N, 5) rows of [x1,y1,x2,y2,confidence], or empty."""
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("detector frame must be a BGR image")
        results = self.model.predict(
            frame,
            device=self.device,
            conf=float(conf),
            imgsz=1280,
            max_det=1000,
            classes=self.face_class_ids,
            verbose=False,
        )[0]
        boxes = getattr(results, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            coordinates = boxes.xyxy.cpu().numpy()
            confidence = boxes.conf.cpu().numpy().reshape(-1, 1)
            return np.concatenate((coordinates, confidence), axis=1)
        return np.empty((0, 5))


class FaceDetector:
    """Dispatch to the detector backend chosen explicitly in configuration."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        allow_download: bool = True,
        backend: str = "yunet",
    ):
        implementations = {
            "yunet": _YuNetDetector,
            "yolo": _YoloDetector,
        }
        try:
            implementation = implementations[backend]
        except KeyError as exc:
            raise DetectorError(f"unsupported detector backend: {backend}") from exc
        self._implementation = implementation(
            model_path,
            device=device,
            allow_download=allow_download,
        )
        self.device = self._implementation.device

    def detect(self, frame, conf: float = 0.2):
        return self._implementation.detect(frame, conf=conf)
