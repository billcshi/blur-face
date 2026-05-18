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
    (255, 80, 80), (80, 255, 80), (80, 80, 255),
    (255, 255, 80), (255, 80, 255), (80, 255, 255),
    (255, 160, 40), (160, 40, 255), (40, 255, 160),
    (255, 120, 180), (120, 180, 255), (180, 255, 120),
]


def color_for(track_id: int):
    """Return a distinct BGR color for a given track ID."""
    return PALETTE[track_id % len(PALETTE)]


def apply_blur_cpu(frame: np.ndarray, bbox, kernel: int,
                     mask_scale: float = 1.15, frame_w: int = 1920,
                     frame_h: int = 1080) -> None:
    """Apply Gaussian blur to a region of the frame (mutates in place)."""
    x1, y1, x2, y2 = bbox
    cw, ch = (x2 - x1), (y2 - y1)
    bx1 = max(0, int(x1 - cw * (mask_scale - 1) / 2))
    by1 = max(0, int(y1 - ch * (mask_scale - 1) / 2))
    bx2 = min(frame_w, int(x2 + cw * (mask_scale - 1) / 2))
    by2 = min(frame_h, int(y2 + ch * (mask_scale - 1) / 2))
    roi = frame[by1:by2, bx1:bx2]
    if roi.size > 0:
        h, w = roi.shape[:2]
        if kernel >= 31 and min(w, h) >= 40:
            # ── Gaussian with downscale optimisation ──
            small_w, small_h = w // 2, h // 2
            small = cv2.resize(roi, (small_w, small_h))
            small_blur = cv2.GaussianBlur(small, (kernel // 2 | 1, kernel // 2 | 1), 0)
            blurred = cv2.resize(small_blur, (w, h))
            # Downscaled mask
            mask = np.zeros((small_h, small_w), dtype=np.uint8)
            cv2.ellipse(mask, (small_w // 2, small_h // 2),
                        (small_w // 2, small_h // 2),
                        0, 0, 360, 255, -1)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            # ── Gaussian (no downscale) ──
            blurred = cv2.GaussianBlur(roi, (kernel, kernel), 0)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (w // 2, h // 2), (w // 2, h // 2),
                        0, 0, 360, 255, -1)
        roi[:] = np.where(mask[:, :, None] == 255, blurred, roi)
        frame[by1:by2, bx1:bx2] = roi


def apply_blur_gpu(frame: np.ndarray, bbox, kernel: int,
                   mask_scale: float = 1.15, frame_w: int = 1920,
                   frame_h: int = 1080) -> None:
    """GPU-accelerated Gaussian blur via cv2.cuda (mutates in place)."""
    x1, y1, x2, y2 = bbox
    cw, ch = (x2 - x1), (y2 - y1)
    bx1 = max(0, int(x1 - cw * (mask_scale - 1) / 2))
    by1 = max(0, int(y1 - ch * (mask_scale - 1) / 2))
    bx2 = min(frame_w, int(x2 + cw * (mask_scale - 1) / 2))
    by2 = min(frame_h, int(y2 + ch * (mask_scale - 1) / 2))
    roi = frame[by1:by2, bx1:bx2]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]

    if kernel >= 31 and min(w, h) >= 40:
        small_w, small_h = w // 2, h // 2
        # Upload ROI to GPU, downscale, blur, upscale, download
        gpu_roi = cv2.cuda_GpuMat()
        gpu_roi.upload(roi)
        gpu_small = cv2.cuda.resize(gpu_roi, (small_w, small_h))
        gpu_filter = cv2.cuda.createGaussianFilter(
            gpu_small.type(), -1,
            (kernel // 2 | 1, kernel // 2 | 1), 0,
        )
        gpu_blur_small = gpu_filter.apply(gpu_small)
        gpu_blurred = cv2.cuda.resize(gpu_blur_small, (w, h))
        blurred = gpu_blurred.download()
        # Mask on CPU (cv2.cuda has no drawing primitives)
        mask = np.zeros((small_h, small_w), dtype=np.uint8)
        cv2.ellipse(mask, (small_w // 2, small_h // 2),
                    (small_w // 2, small_h // 2),
                    0, 0, 360, 255, -1)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        gpu_roi = cv2.cuda_GpuMat()
        gpu_roi.upload(roi)
        gpu_filter = cv2.cuda.createGaussianFilter(
            gpu_roi.type(), -1, (kernel, kernel), 0,
        )
        blurred = gpu_filter.apply(gpu_roi).download()
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (w // 2, h // 2), (w // 2, h // 2),
                    0, 0, 360, 255, -1)

    roi[:] = np.where(mask[:, :, None] == 255, blurred, roi)
    frame[by1:by2, bx1:bx2] = roi


# ── Dispatch: use GPU if available, else CPU ──
def apply_blur(frame: np.ndarray, bbox, kernel: int,
               mask_scale: float = 1.15, frame_w: int = 1920,
               frame_h: int = 1080) -> None:
    """Apply blur using GPU if available, falling back to CPU."""
    global HAS_CUDA, _CUDA_FALLBACK_WARNED

    if HAS_CUDA:
        try:
            apply_blur_gpu(frame, bbox, kernel, mask_scale, frame_w, frame_h)
            return
        except cv2.error as exc:
            HAS_CUDA = False
            if not _CUDA_FALLBACK_WARNED:
                print(f"[Renderer] OpenCV CUDA failed, falling back to CPU: {exc}")
                _CUDA_FALLBACK_WARNED = True

    apply_blur_cpu(frame, bbox, kernel, mask_scale, frame_w, frame_h)


def draw_debug_box(frame: np.ndarray, bbox, track_id: int,
                   is_predicted: bool = False, is_excluded: bool = False) -> None:
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

    label = f"ID:{track_id}" + (" PRED" if is_predicted else "")
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), c, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 2), font, 0.5, (0, 0, 0), 1)

    if is_excluded:
        cv2.putText(frame, "KEPT", (x1, y2 + 14), font, 0.45, (0, 255, 0), 1)
