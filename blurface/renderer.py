"""
blurface.renderer — Blur and debug-draw operations.
"""

import cv2
import numpy as np


def _has_opencv_cuda() -> bool:
    """Return True only when OpenCV has working CUDA support.

    The pip opencv-python wheels expose cv2.cuda stubs even when compiled
    without CUDA, so checking the namespace alone is not enough.
    """
    if not hasattr(cv2, "cuda") or not hasattr(cv2, "cuda_GpuMat"):
        return False

    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except (AttributeError, cv2.error):
        return False


HAS_CUDA = _has_opencv_cuda()
_CUDA_FALLBACK_WARNED = False

# 12 distinct colors for debug track IDs
PALETTE = [
    (255, 80, 80),
    (80, 255, 80),
    (80, 80, 255),
    (255, 255, 80),
    (255, 80, 255),
    (80, 255, 255),
    (255, 160, 40),
    (160, 40, 255),
    (40, 255, 160),
    (255, 120, 180),
    (120, 180, 255),
    (180, 255, 120),
]


def color_for(track_id: int):
    """Return a distinct BGR color for a given track ID."""
    return PALETTE[track_id % len(PALETTE)]


MASK_SHAPES = frozenset({"rounded-rect", "rectangle", "ellipse"})


def _prepare_region(frame: np.ndarray, bbox, kernel: int, mask_scale: float):
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR image")
    if kernel <= 0:
        raise ValueError("blur kernel must be positive")
    kernel = kernel if kernel % 2 else kernel + 1
    if mask_scale < 1.0:
        raise ValueError("mask scale must be at least 1")
    values = np.asarray(bbox, dtype=float)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("bounding box must contain four finite coordinates")
    x1, x2 = sorted((values[0], values[2]))
    y1, y2 = sorted((values[1], values[3]))
    width, height = x2 - x1, y2 - y1
    frame_h, frame_w = frame.shape[:2]
    bx1 = max(0, int(np.floor(x1 - width * (mask_scale - 1) / 2)))
    by1 = max(0, int(np.floor(y1 - height * (mask_scale - 1) / 2)))
    bx2 = min(frame_w, int(np.ceil(x2 + width * (mask_scale - 1) / 2)))
    by2 = min(frame_h, int(np.ceil(y2 + height * (mask_scale - 1) / 2)))
    inner = (
        max(0, int(np.floor(x1)) - bx1),
        max(0, int(np.floor(y1)) - by1),
        min(bx2 - bx1, int(np.ceil(x2)) - bx1),
        min(by2 - by1, int(np.ceil(y2)) - by1),
    )
    return bx1, by1, bx2, by2, kernel, inner


def _coverage_mask(
    width: int,
    height: int,
    inner: tuple[int, int, int, int],
    shape: str,
) -> np.ndarray:
    """Build a mask while keeping the full inner detection rectangle covered."""
    if shape not in MASK_SHAPES:
        raise ValueError(f"mask shape must be one of: {', '.join(sorted(MASK_SHAPES))}")
    mask = np.zeros((height, width), dtype=np.uint8)
    if width <= 0 or height <= 0:
        return mask
    if shape == "rectangle":
        mask[:] = 255
        return mask
    if shape == "ellipse":
        cv2.ellipse(
            mask,
            (width // 2, height // 2),
            (width // 2, height // 2),
            0,
            0,
            360,
            255,
            -1,
        )
        return mask

    ix1, iy1, ix2, iy2 = inner
    # Round only inside the expansion margin. If clipping removes a margin at
    # a frame edge, that side becomes square instead of cutting the face box.
    radius = min(
        round(min(width, height) * 0.2),
        ix1,
        iy1,
        max(0, width - ix2),
        max(0, height - iy2),
    )
    if radius <= 0:
        mask[:] = 255
        return mask
    cv2.rectangle(mask, (radius, 0), (width - radius - 1, height - 1), 255, -1)
    cv2.rectangle(mask, (0, radius), (width - 1, height - radius - 1), 255, -1)
    for center in (
        (radius, radius),
        (width - radius - 1, radius),
        (radius, height - radius - 1),
        (width - radius - 1, height - radius - 1),
    ):
        cv2.circle(mask, center, radius, 255, -1)
    # This is deliberately explicit: future changes to the rounded outline
    # must never make the model's original detection rectangle transparent.
    if ix2 > ix1 and iy2 > iy1:
        mask[iy1:iy2, ix1:ix2] = 255
    return mask


def apply_blur_cpu(
    frame: np.ndarray,
    bbox,
    kernel: int,
    mask_scale: float = 1.15,
    frame_w: int = 1920,
    frame_h: int = 1080,
    mask_shape: str = "rounded-rect",
) -> None:
    """Apply Gaussian blur to a region of the frame (mutates in place)."""
    del frame_w, frame_h  # retained for API compatibility
    bx1, by1, bx2, by2, kernel, inner = _prepare_region(
        frame, bbox, kernel, mask_scale
    )
    roi = frame[by1:by2, bx1:bx2]
    if roi.size > 0:
        h, w = roi.shape[:2]
        mask = _coverage_mask(w, h, inner, mask_shape)
        if kernel >= 31 and min(w, h) >= 40:
            # ── Gaussian with downscale optimisation ──
            small_w, small_h = w // 2, h // 2
            small = cv2.resize(roi, (small_w, small_h))
            small_blur = cv2.GaussianBlur(small, (kernel // 2 | 1, kernel // 2 | 1), 0)
            blurred = cv2.resize(small_blur, (w, h))
        else:
            # ── Gaussian (no downscale) ──
            blurred = cv2.GaussianBlur(roi, (kernel, kernel), 0)
        roi[:] = np.where(mask[:, :, None] == 255, blurred, roi)
        frame[by1:by2, bx1:bx2] = roi


def apply_blur_gpu(
    frame: np.ndarray,
    bbox,
    kernel: int,
    mask_scale: float = 1.15,
    frame_w: int = 1920,
    frame_h: int = 1080,
    mask_shape: str = "rounded-rect",
) -> None:
    """GPU-accelerated Gaussian blur via cv2.cuda (mutates in place)."""
    del frame_w, frame_h
    bx1, by1, bx2, by2, kernel, inner = _prepare_region(
        frame, bbox, kernel, mask_scale
    )
    roi = frame[by1:by2, bx1:bx2]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]
    mask = _coverage_mask(w, h, inner, mask_shape)

    if kernel >= 31 and min(w, h) >= 40:
        small_w, small_h = w // 2, h // 2
        # Upload ROI to GPU, downscale, blur, upscale, download
        gpu_roi = cv2.cuda_GpuMat()
        gpu_roi.upload(roi)
        gpu_small = cv2.cuda.resize(gpu_roi, (small_w, small_h))
        gpu_filter = cv2.cuda.createGaussianFilter(
            gpu_small.type(),
            -1,
            (kernel // 2 | 1, kernel // 2 | 1),
            0,
        )
        gpu_blur_small = gpu_filter.apply(gpu_small)
        gpu_blurred = cv2.cuda.resize(gpu_blur_small, (w, h))
        blurred = gpu_blurred.download()
    else:
        gpu_roi = cv2.cuda_GpuMat()
        gpu_roi.upload(roi)
        gpu_filter = cv2.cuda.createGaussianFilter(
            gpu_roi.type(),
            -1,
            (kernel, kernel),
            0,
        )
        blurred = gpu_filter.apply(gpu_roi).download()
    roi[:] = np.where(mask[:, :, None] == 255, blurred, roi)
    frame[by1:by2, bx1:bx2] = roi


# ── Dispatch: use GPU if available, else CPU ──
def apply_blur(
    frame: np.ndarray,
    bbox,
    kernel: int,
    mask_scale: float = 1.15,
    frame_w: int = 1920,
    frame_h: int = 1080,
    mask_shape: str = "rounded-rect",
) -> None:
    """Apply blur using GPU if available, falling back to CPU."""
    global HAS_CUDA, _CUDA_FALLBACK_WARNED

    if HAS_CUDA:
        try:
            apply_blur_gpu(
                frame,
                bbox,
                kernel,
                mask_scale,
                frame_w,
                frame_h,
                mask_shape,
            )
            return
        except (cv2.error, AttributeError) as exc:
            HAS_CUDA = False
            if not _CUDA_FALLBACK_WARNED:
                print(f"[Renderer] OpenCV CUDA failed, falling back to CPU: {exc}")
                _CUDA_FALLBACK_WARNED = True

    apply_blur_cpu(
        frame,
        bbox,
        kernel,
        mask_scale,
        frame_w,
        frame_h,
        mask_shape,
    )


def draw_debug_box(
    frame: np.ndarray,
    bbox,
    track_id: int,
    is_predicted: bool = False,
    is_excluded: bool = False,
    confidence: float | None = None,
) -> None:
    """Draw colored box + ID label on frame (mutates in place).
    Predicted tracks get dashed boxes with 'PRED' label.
    Excluded tracks get a green 'KEPT' label.
    """
    x1, y1, x2, y2 = bbox
    c = color_for(track_id)

    if is_predicted:
        # ── Dashed box for predicted positions ──
        for dx in range(x1, x2, 12):
            cv2.line(frame, (dx, y1), (min(dx + 6, x2), y1), c, 1)
        for dx in range(x1, x2, 12):
            cv2.line(frame, (dx, y2), (min(dx + 6, x2), y2), c, 1)
        for dy in range(y1, y2, 12):
            cv2.line(frame, (x1, dy), (x1, min(dy + 6, y2)), c, 1)
        for dy in range(y1, y2, 12):
            cv2.line(frame, (x2, dy), (x2, min(dy + 6, y2)), c, 1)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)

    label = f"ID:{track_id}"
    if confidence is not None:
        label += f" C:{confidence:.2f}"
    if is_predicted:
        label += " PRED"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), c, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 2), font, 0.5, (0, 0, 0), 1)

    if is_excluded:
        cv2.putText(frame, "KEPT", (x1, y2 + 14), font, 0.45, (0, 255, 0), 1)
