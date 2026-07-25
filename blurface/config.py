"""Typed application configuration and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    """Raised when a user supplied configuration is unsafe or invalid."""


@dataclass(frozen=True, slots=True)
class AppConfig:
    input: Path
    output: Path = Path("output_blur.mp4")
    overwrite: bool = False
    model: str = "yolov11m-face.pt"
    threshold: float = 0.3
    time_thresholds: tuple[tuple[float, float], ...] = ()
    mask_scale: float = 1.35
    blur_kernel: int = 51
    device: str = "cuda"
    ffmpeg: Path | None = None
    debug: bool = False
    profile: bool = False
    exclude_ids: frozenset[int] = frozenset()
    allow_unsafe_exclusions: bool = False
    lost_buffer: int = 180
    smooth: float = 0.7
    preset: str = "quality"
    min_face_size: int = 8
    max_face_height_ratio: float = 1.0
    flow_max_points: int = 50
    flow_min_confirmations: int = 3
    flow_max_missed: int = 0
    flow_enabled: bool = True
    use_nvenc: bool = True
    offline: bool = False

    def validate(self) -> AppConfig:
        if not self.input:
            raise ConfigError("input path is required")
        if self.input.expanduser().resolve() == self.output.expanduser().resolve():
            raise ConfigError("input and output must be different files")
        if self.output.expanduser().exists() and not self.overwrite:
            raise ConfigError("output already exists; use --overwrite to replace it")
        if not 0.0 <= self.threshold <= 1.0:
            raise ConfigError("--thresh must be between 0 and 1")
        for second, threshold in self.time_thresholds:
            if second < 0:
                raise ConfigError("--time-thresh seconds must be non-negative")
            if not 0.0 <= threshold <= 1.0:
                raise ConfigError("--time-thresh values must be between 0 and 1")
        if self.mask_scale < 1.0:
            raise ConfigError("--mask-scale must be at least 1")
        if self.blur_kernel <= 0:
            raise ConfigError("--blur-kernel must be positive")
        if self.device not in {"cpu", "cuda", "mps"} and not self.device.startswith(
            "cuda:"
        ):
            raise ConfigError("--device must be cpu, cuda, cuda:N, or mps")
        if self.ffmpeg is not None and not self.ffmpeg.expanduser().is_file():
            raise ConfigError(f"--ffmpeg executable does not exist: {self.ffmpeg}")
        if self.lost_buffer < 0:
            raise ConfigError("--lost-buffer cannot be negative")
        if not 0.0 <= self.smooth <= 1.0:
            raise ConfigError("--smooth must be between 0 and 1")
        if self.min_face_size < 1:
            raise ConfigError("--min-face-size must be at least 1")
        if not 0.0 < self.max_face_height_ratio <= 1.0:
            raise ConfigError("--max-face-height-ratio must be in (0, 1]")
        if self.flow_max_points < 5:
            raise ConfigError("--flow-max-points must be at least 5")
        if self.flow_min_confirmations < 1:
            raise ConfigError("--flow-min-confirmations must be at least 1")
        if self.flow_max_missed < 0:
            raise ConfigError("--flow-max-missed cannot be negative")
        if any(track_id < 0 for track_id in self.exclude_ids):
            raise ConfigError("--exclude-ids cannot contain negative IDs")
        if self.exclude_ids and not self.allow_unsafe_exclusions:
            raise ConfigError(
                "--exclude-ids uses temporary motion IDs and can expose the wrong "
                "person; repeat with --allow-unsafe-exclusions after review"
            )
        return self

    def threshold_for(self, frame_index: int, fps: float) -> float:
        threshold = self.threshold
        if not self.time_thresholds:
            return threshold
        if fps <= 0:
            raise ConfigError("video FPS must be positive when --time-thresh is used")
        second = frame_index / fps
        for start, candidate in self.time_thresholds:
            if second >= start:
                threshold = candidate
            else:
                break
        return threshold
