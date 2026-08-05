"""Typed application configuration and validation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


class ConfigError(ValueError):
    """Raised when a user supplied configuration is unsafe or invalid."""


IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
)
BLUR_REFERENCE_SHORT_EDGE = 1080


@dataclass(frozen=True, slots=True)
class AppConfig:
    input: Path
    output: Path = Path("output_blur.mp4")
    overwrite: bool = False
    detector: str = "yunet"
    model: str = "face_detection_yunet_2023mar.onnx"
    threshold: float = 0.3
    time_thresholds: tuple[tuple[float, float], ...] = ()
    mask_scale: float = 1.5
    mask_shape: str = "rounded-rect"
    mask_engine: str = "geometric"
    sam_mask_expansion: float = 0.12
    segmentation_combine: str = "union"
    sam2_model: str = "facebook/sam2.1-hiera-base-plus"
    sam2_refresh_interval: int = 15
    temporal_stabilization: bool = True
    backfill_frames: int = 10
    release_hold_frames: int = 5
    scene_cut_sensitivity: float = 0.55
    temporal_storage_limit_mb: int = 4096
    blur_strategy: str = "adaptive"
    blur_kernel: int = 251
    blur_kernel_min: int = 101
    device: str = "auto"
    ffmpeg: Path | None = None
    job_temp_dir: Path | None = None
    debug: bool = False
    mask_preview: bool = False
    profile: bool = False
    lost_buffer: int = 180
    smooth: float = 0.7
    preset: str = "quality"
    min_face_size: int = 30
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
        if not self.model.strip():
            raise ConfigError("--model cannot be empty")
        if self.detector not in {"yunet", "yolo"}:
            raise ConfigError("--detector must be yunet or yolo")
        model_suffix = Path(self.model).suffix.lower()
        if self.detector == "yunet" and model_suffix != ".onnx":
            raise ConfigError("--detector yunet requires an ONNX --model")
        if self.detector == "yolo" and model_suffix != ".pt":
            raise ConfigError("--detector yolo requires a .pt --model")
        if not 0.0 <= self.threshold <= 1.0:
            raise ConfigError("--thresh must be between 0 and 1")
        for second, threshold in self.time_thresholds:
            if second < 0:
                raise ConfigError("--time-thresh seconds must be non-negative")
            if not 0.0 <= threshold <= 1.0:
                raise ConfigError("--time-thresh values must be between 0 and 1")
        if self.mask_scale < 1.0:
            raise ConfigError("--mask-scale must be at least 1")
        if self.mask_shape not in {"rounded-rect", "rectangle", "ellipse"}:
            raise ConfigError(
                "--mask-shape must be rounded-rect, rectangle, or ellipse"
            )
        if self.mask_engine not in {"geometric", "sam2.1"}:
            raise ConfigError(
                "--mask-engine must be geometric or sam2.1"
            )
        if self.mask_engine == "sam2.1" and not self.sam2_model.strip():
            raise ConfigError("--sam2-model is required by --mask-engine sam2.1")
        if self.sam2_refresh_interval <= 0:
            raise ConfigError("--sam2-refresh-interval must be positive")
        if not 0 <= self.backfill_frames <= 60:
            raise ConfigError("--backfill-frames must be between 0 and 60")
        if not 0 <= self.release_hold_frames <= 12:
            raise ConfigError("--release-hold-frames must be between 0 and 12")
        if not 0.0 <= self.scene_cut_sensitivity <= 1.0:
            raise ConfigError(
                "--scene-cut-sensitivity must be between 0 and 1"
            )
        if self.temporal_storage_limit_mb < 64:
            raise ConfigError(
                "--temporal-storage-limit-mb must be at least 64"
            )
        if not 0.0 <= self.sam_mask_expansion <= 0.5:
            raise ConfigError("--sam-mask-expansion must be between 0 and 0.5")
        if self.segmentation_combine not in {
            "union",
            "intersection",
            "mask-only",
        }:
            raise ConfigError(
                "--segmentation-combine must be union, intersection, or mask-only"
            )
        if self.blur_kernel <= 0:
            raise ConfigError("--blur-kernel must be positive")
        if self.blur_kernel_min <= 0:
            raise ConfigError("--blur-kernel-min must be positive")
        if (
            self.blur_strategy == "adaptive"
            and self.blur_kernel_min > self.blur_kernel
        ):
            raise ConfigError("--blur-kernel-min cannot exceed --blur-kernel")
        if self.blur_strategy not in {"adaptive", "fixed"}:
            raise ConfigError("--blur-strategy must be adaptive or fixed")
        if self.device not in {"auto", "cpu", "cuda", "mps"} and not self.device.startswith(
            "cuda:"
        ):
            raise ConfigError("--device must be auto, cpu, cuda, cuda:N, or mps")
        if self.ffmpeg is not None and not self.ffmpeg.expanduser().is_file():
            raise ConfigError(f"--ffmpeg executable does not exist: {self.ffmpeg}")
        if self.job_temp_dir is not None and not self.job_temp_dir.expanduser().is_dir():
            raise ConfigError(
                f"--job-temp-dir does not exist: {self.job_temp_dir}"
            )
        if self.debug and self.mask_preview:
            raise ConfigError("--debug and --mask-preview cannot be used together")
        if self.lost_buffer < 0:
            raise ConfigError("--lost-buffer cannot be negative")
        if not 0.0 <= self.smooth <= 1.0:
            raise ConfigError("--smooth must be between 0 and 1")
        if self.preset not in {"quality", "fast"}:
            raise ConfigError("--preset must be quality or fast")
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
        return self

    def blur_kernel_for(self, confidence: float, threshold: float) -> int:
        """Map detector confidence to an odd kernel within the configured range."""
        if self.blur_strategy == "fixed":
            return self._odd_kernel(self.blur_kernel)
        # Confidence above 0.85 is already a strong face detection. Saturating
        # there lets normal high-confidence faces receive the full privacy
        # kernel instead of reserving it for the practically unreachable 1.0.
        saturation = max(threshold, 0.85)
        if threshold >= saturation:
            ratio = 1.0
        else:
            ratio = (confidence - threshold) / (saturation - threshold)
            ratio = max(0.0, min(1.0, ratio))
        value = round(
            self.blur_kernel_min
            + ratio * (self.blur_kernel - self.blur_kernel_min)
        )
        return self._odd_kernel(value)

    def blur_kernel_for_frame(
        self,
        confidence: float,
        threshold: float,
        frame_shape: tuple[int, int],
    ) -> int:
        """Scale the shared 1080p blur strength for high-resolution media."""
        if len(frame_shape) != 2:
            raise ConfigError("frame shape must contain height and width")
        height, width = (int(value) for value in frame_shape)
        if height <= 0 or width <= 0:
            raise ConfigError("frame dimensions must be positive")
        base_kernel = self.blur_kernel_for(confidence, threshold)
        # UI/CLI kernel values remain a 1080p baseline. Scaling by the short
        # edge is orientation-independent: 1920x1080 and 1080x1920 retain the
        # same strength, while high-resolution images and videos do not become
        # perceptually weaker merely because they contain more pixels.
        resolution_scale = max(
            1.0, min(height, width) / BLUR_REFERENCE_SHORT_EDGE
        )
        return self._odd_kernel(round(base_kernel * resolution_scale))

    @staticmethod
    def _odd_kernel(value: int) -> int:
        return value if value % 2 else value + 1

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


@dataclass(frozen=True, slots=True)
class ImageBatchConfig:
    """Validated still-image sources and their deterministic destinations."""

    options: AppConfig
    inputs: tuple[Path, ...]
    outputs: tuple[Path, ...]
    output_directory: Path

    def validate(self) -> ImageBatchConfig:
        if not self.inputs or len(self.inputs) != len(self.outputs):
            raise ConfigError("image batch inputs and outputs must be non-empty pairs")
        self.options.validate()
        output_keys: set[str] = set()
        for source, output in zip(self.inputs, self.outputs):
            if not source.expanduser().is_file():
                raise ConfigError(f"input image does not exist: {source}")
            if source.suffix.lower() not in IMAGE_SUFFIXES:
                raise ConfigError(f"unsupported input image format: {source}")
            if output.suffix.lower() not in IMAGE_SUFFIXES:
                raise ConfigError(f"unsupported output image format: {output}")
            if source.expanduser().resolve() == output.expanduser().resolve():
                raise ConfigError(f"input and output must be different files: {source}")
            key = os.path.normcase(str(output.expanduser().resolve())).casefold()
            if key in output_keys:
                raise ConfigError(f"multiple inputs map to the same output: {output}")
            output_keys.add(key)
            if os.path.lexists(output.expanduser()) and not self.options.overwrite:
                raise ConfigError(
                    f"output already exists; use --overwrite to replace it: {output}"
                )
        if os.path.lexists(self.output_directory.expanduser()) and not (
            self.output_directory.expanduser().is_dir()
        ):
            raise ConfigError(
                f"image batch output directory is not a directory: "
                f"{self.output_directory}"
            )
        if self.options.time_thresholds:
            raise ConfigError("--time-thresh is available only for video input")
        return self
