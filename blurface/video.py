"""Validated OpenCV video input."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


class VideoInputError(RuntimeError):
    """Raised when an input cannot provide processable video frames."""


@dataclass(frozen=True, slots=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    total_frames: int | None


class VideoSource:
    def __init__(self, path: Path):
        self.path = path
        self.capture: cv2.VideoCapture | None = None
        self.info: VideoInfo | None = None

    def __enter__(self) -> VideoSource:  # noqa: PYI034 - Python 3.10 support
        if not self.path.is_file():
            raise VideoInputError(f"input video does not exist: {self.path}")
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise VideoInputError(f"cannot open input video: {self.path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        raw_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            capture.release()
            raise VideoInputError("input video reports invalid dimensions")
        if fps <= 0:
            capture.release()
            raise VideoInputError("input video reports an invalid FPS")
        self.capture = capture
        self.info = VideoInfo(width, height, fps, raw_total if raw_total > 0 else None)
        return self

    def read(self):
        if self.capture is None:
            raise RuntimeError("video source is not open")
        return self.capture.read()

    def __exit__(self, *_exc) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = None
