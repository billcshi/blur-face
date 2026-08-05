"""Optional SAM 2.1 box-prompted segmentation with privacy-safe output."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SAM_SESSION_MEMORY_LIMIT_BYTES = 512 * 1024**2
SAM_MIN_OBJECT_LOGIT = 0.5
SAM_MIN_IMAGE_SCORE = 0.5
_EXPECTED_IMAGE_CONFIG_WARNING = " ".join((
    "You are using a model of type `sam2_video` to instantiate a model of "
    "type `sam2`. This may be expected if you are loading a checkpoint that "
    "shares a subset of the architecture (e.g., loading a `sam2_video` "
    "checkpoint into `Sam2Model`), but is otherwise not supported and can "
    "yield errors. Please verify that the checkpoint is compatible with the "
    "model you are instantiating."
).split())


class Sam2Error(RuntimeError):
    """Raised when optional SAM 2.1 support cannot be initialized."""


def _imports():
    try:
        import torch
        from transformers import Sam2Model, Sam2Processor
    except (ImportError, OSError) as exc:
        raise Sam2Error(
            "SAM 2.1 support is not installed; run install-sam2.bat "
            "(Windows) or ./install-sam2.sh"
        ) from exc
    return torch, Sam2Model, Sam2Processor


def _video_imports():
    try:
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor
    except (ImportError, OSError) as exc:
        raise Sam2Error(
            "SAM 2 video support requires Transformers 5.14 or newer; "
            "run install-sam2.bat (Windows) or ./install-sam2.sh"
        ) from exc
    return torch, Sam2VideoModel, Sam2VideoProcessor


def _load_pretrained(component: Any, source: str, offline: bool):
    """Prefer a complete local HF snapshot, downloading only when absent."""
    options = {"local_files_only": True, "trust_remote_code": False}
    if offline or Path(source).exists():
        return component.from_pretrained(source, **options)
    try:
        return component.from_pretrained(source, **options)
    except Exception as exc:
        if not _is_hf_cache_miss(exc):
            raise
        return component.from_pretrained(
            source,
            local_files_only=False,
            trust_remote_code=False,
        )


def _is_hf_cache_miss(exc: BaseException) -> bool:
    """Recognize the Hub's explicit local-snapshot miss without hiding damage."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in {
            "IncompleteSnapshotError",
            "LocalEntryNotFoundError",
        }:
            return True
        current = current.__cause__ or current.__context__
    return False


class _ExpectedImageConfigWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = " ".join(record.getMessage().split())
        return not (
            record.levelno == logging.WARNING
            and message == _EXPECTED_IMAGE_CONFIG_WARNING
        )


@contextmanager
def _suppress_expected_image_config_warning():
    """Hide the documented video-checkpoint/image-subset compatibility warning."""
    logger = logging.getLogger("transformers.configuration_utils")
    warning_filter = _ExpectedImageConfigWarningFilter()
    logger.addFilter(warning_filter)
    try:
        yield
    finally:
        logger.removeFilter(warning_filter)


def _device(torch: Any, requested: str):
    value = str(requested).strip().lower()
    if value in {"", "auto"}:
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif value == "cuda":
        value = "cuda:0"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise Sam2Error(
            f"SAM 2.1 requested {requested!r}, but CUDA is unavailable"
        )
    try:
        return torch.device(value)
    except (RuntimeError, ValueError) as exc:
        raise Sam2Error(f"invalid SAM 2.1 device: {requested!r}") from exc


def _clean_sam_mask(
    predicted: np.ndarray,
    output_shape: tuple[int, int],
    inner: tuple[int, int, int, int],
    dilation_ratio: float = 0.12,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Clean and dilate a SAM mask using the supplied scale reference."""
    height, width = output_shape
    if height <= 0 or width <= 0:
        raise ValueError("output shape must be positive")
    mask = np.asarray(predicted)
    if mask.ndim != 2:
        raise ValueError("SAM 2.1 mask must be two-dimensional")
    if not np.isfinite(mask).all():
        raise ValueError("SAM 2.1 mask must contain finite values")
    if not np.isfinite(dilation_ratio) or not 0 <= dilation_ratio <= 0.5:
        raise ValueError("dilation ratio must be between 0 and 0.5")
    mask = cv2.resize(
        (mask > 0).astype(np.uint8) * 255,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    ix1, iy1, ix2, iy2 = (int(value) for value in inner)
    ix1, ix2 = sorted((max(0, ix1), min(width, ix2)))
    iy1, iy2 = sorted((max(0, iy1), min(height, iy2)))
    face_size = max(1, min(max(1, ix2 - ix1), max(1, iy2 - iy1)))

    close_size = max(3, round(face_size * 0.04) | 1)
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_size, close_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    dilation = round(face_size * dilation_ratio)
    if dilation > 0:
        kernel_size = dilation * 2 + 1
        dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = cv2.dilate(mask, dilation_kernel)
    return mask, (ix1, iy1, ix2, iy2)


def build_safe_sam_mask(
    predicted: np.ndarray,
    output_shape: tuple[int, int],
    inner: tuple[int, int, int, int],
    dilation_ratio: float = 0.12,
    combine: str = "union",
) -> np.ndarray:
    """Clean a SAM mask and combine it with the detector box."""
    if combine not in {"union", "intersection"}:
        raise ValueError("combine must be union or intersection")
    mask, (ix1, iy1, ix2, iy2) = _clean_sam_mask(
        predicted,
        output_shape,
        inner,
        dilation_ratio,
    )

    if combine == "union" and ix2 > ix1 and iy2 > iy1:
        mask[iy1:iy2, ix1:ix2] = 255
    elif combine == "intersection":
        core = np.zeros_like(mask)
        if ix2 > ix1 and iy2 > iy1:
            core[iy1:iy2, ix1:ix2] = 255
        mask = cv2.bitwise_and(mask, core)
    return np.ascontiguousarray(mask, dtype=np.uint8)


class Sam2Segmenter:
    """Use a Transformers SAM 2.1 model with a detector-box prompt."""

    def __init__(
        self,
        model: str | Path,
        device: str = "auto",
        offline: bool = False,
    ) -> None:
        self._torch, model_class, processor_class = _imports()
        self.device = _device(self._torch, device)
        self.model_name = str(model)
        if not self.model_name.strip():
            raise Sam2Error("SAM 2.1 model cannot be empty")
        source = (
            str(Path(self.model_name).expanduser().resolve())
            if Path(self.model_name).expanduser().exists()
            else self.model_name
        )
        try:
            self._processor = _load_pretrained(processor_class, source, offline)
            with _suppress_expected_image_config_warning():
                self._model = (
                    _load_pretrained(model_class, source, offline)
                    .to(self.device)
                    .eval()
                )
        except Exception as exc:
            mode = " from the local cache" if offline else ""
            raise Sam2Error(
                f"unable to load SAM 2.1 model {self.model_name!r}{mode}: {exc}"
            ) from exc
        self.last_error: str | None = None

    def _predict(
        self,
        roi_bgr: np.ndarray,
        inner: tuple[int, int, int, int],
    ) -> np.ndarray:
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        box = [[list(float(value) for value in inner)]]
        inputs = self._processor(
            images=rgb,
            input_boxes=box,
            return_tensors="pt",
        ).to(self.device)
        autocast = (
            self._torch.autocast("cuda", dtype=self._torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        backend_context = (
            self._torch.backends.mkldnn.flags(enabled=False)
            if self.device.type == "cpu"
            else nullcontext()
        )
        with self._torch.inference_mode(), autocast, backend_context:
            outputs = self._model(**inputs, multimask_output=True)
        masks = self._processor.post_process_masks(
            outputs.pred_masks.detach().to("cpu"),
            inputs["original_sizes"].detach().to("cpu"),
        )[0]
        masks_array = np.asarray(masks, dtype=np.float32)
        while masks_array.ndim > 3 and masks_array.shape[0] == 1:
            masks_array = masks_array[0]
        if masks_array.ndim == 2:
            masks_array = masks_array[np.newaxis, :, :]
        if masks_array.ndim != 3:
            raise ValueError(
                f"unexpected SAM 2.1 mask shape: {masks_array.shape}"
            )
        scores = (
            outputs.iou_scores.detach()
            .to("cpu")
            .float()
            .numpy()
            .reshape(-1)
        )
        if (
            not scores.size
            or scores.size != masks_array.shape[0]
            or not np.isfinite(scores).all()
            or float(np.max(scores)) < SAM_MIN_IMAGE_SCORE
        ):
            raise ValueError(
                "SAM 2.1 image mask scores are missing, mismatched, non-finite, or low"
            )
        index = int(np.argmax(scores))
        selected = masks_array[index]
        if selected.shape != roi_bgr.shape[:2]:
            raise ValueError(
                "SAM 2.1 post-processed image mask shape "
                f"{selected.shape} does not match ROI shape {roi_bgr.shape[:2]}"
            )
        return selected

    def build_contour(
        self,
        roi_bgr: np.ndarray,
        inner: tuple[int, int, int, int],
        dilation_ratio: float = 0.12,
    ) -> np.ndarray | None:
        """Return a cleaned image contour, or ``None`` for geometric fallback."""
        self.last_error = None
        try:
            if (
                not isinstance(roi_bgr, np.ndarray)
                or roi_bgr.ndim != 3
                or roi_bgr.shape[2] != 3
                or roi_bgr.size == 0
            ):
                raise ValueError("ROI must be a non-empty BGR image")
            predicted = self._predict(roi_bgr, inner)
            mask, _clipped_inner = _clean_sam_mask(
                predicted,
                roi_bgr.shape[:2],
                inner,
                dilation_ratio,
            )
            return mask
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    def build_mask(
        self,
        roi_bgr: np.ndarray,
        inner: tuple[int, int, int, int],
        dilation_ratio: float = 0.12,
        combine: str = "union",
    ) -> np.ndarray | None:
        """Return the compatible union/intersection mask contract."""
        if combine not in {"union", "intersection"}:
            self.last_error = "ValueError: combine must be union or intersection"
            return None
        contour = self.build_contour(roi_bgr, inner, dilation_ratio)
        if contour is None:
            return None
        ix1, iy1, ix2, iy2 = (int(value) for value in inner)
        height, width = contour.shape
        ix1, ix2 = sorted((max(0, ix1), min(width, ix2)))
        iy1, iy2 = sorted((max(0, iy1), min(height, iy2)))
        if combine == "union" and ix2 > ix1 and iy2 > iy1:
            contour[iy1:iy2, ix1:ix2] = 255
        elif combine == "intersection":
            core = np.zeros_like(contour)
            if ix2 > ix1 and iy2 > iy1:
                core[iy1:iy2, ix1:ix2] = 255
            contour = cv2.bitwise_and(contour, core)
        return np.ascontiguousarray(contour, dtype=np.uint8)


class Sam2VideoSegmenter:
    """Track detector-seeded objects through streamed frames with SAM 2."""

    def __init__(
        self,
        model: str | Path,
        device: str = "auto",
        offline: bool = False,
    ) -> None:
        self._torch, model_class, processor_class = _video_imports()
        self.device = _device(self._torch, device)
        self.model_name = str(model)
        if not self.model_name.strip():
            raise Sam2Error("SAM 2.1 model cannot be empty")
        source_path = Path(self.model_name).expanduser()
        source = (
            str(source_path.resolve())
            if source_path.exists()
            else self.model_name
        )
        try:
            self._processor = _load_pretrained(processor_class, source, offline)
            self._model = (
                _load_pretrained(model_class, source, offline)
                .to(self.device)
                .eval()
            )
        except Exception as exc:
            mode = " from the local cache" if offline else ""
            raise Sam2Error(
                f"unable to load SAM 2.1 video model "
                f"{self.model_name!r}{mode}: {exc}"
            ) from exc
        self._session = None
        self._frame_index = 0
        self._needs_reseed = False
        self.last_error: str | None = None
        self.last_rejections: dict[int, str] = {}

    def reset(self) -> None:
        """Start a fresh correction window and discard stale object memory."""
        dtype = (
            self._torch.bfloat16
            if self.device.type == "cuda"
            else self._torch.float32
        )
        self._session = self._processor.init_video_session(
            inference_device=self.device,
            inference_state_device=self.device,
            processing_device=self.device,
            video_storage_device="cpu",
            max_vision_features_cache_size=1,
            dtype=dtype,
        )
        self._frame_index = 0
        self._needs_reseed = False

    @property
    def frame_count(self) -> int:
        """Number of streamed frames retained by the current scene window."""
        return self._frame_index

    @property
    def object_count(self) -> int:
        """Number of object memories retained by the current session."""
        return self._session.get_obj_num() if self._session is not None else 0

    @property
    def object_ids(self) -> frozenset[int]:
        values = (
            getattr(self._session, "obj_ids", ())
            if self._session is not None
            else ()
        )
        return frozenset(int(value) for value in values)

    @property
    def needs_reseed(self) -> bool:
        return self._needs_reseed

    @staticmethod
    def _retained_tensor_bytes(value: Any, seen: set[int]) -> int:
        identity = id(value)
        if identity in seen:
            return 0
        seen.add(identity)
        if hasattr(value, "numel") and hasattr(value, "element_size"):
            try:
                return int(value.numel()) * int(value.element_size())
            except (TypeError, RuntimeError):
                return 0
        if isinstance(value, dict):
            return sum(
                Sam2VideoSegmenter._retained_tensor_bytes(item, seen)
                for item in value.values()
            )
        if isinstance(value, (list, tuple)):
            return sum(
                Sam2VideoSegmenter._retained_tensor_bytes(item, seen)
                for item in value
            )
        return 0

    def retained_session_bytes(self) -> int:
        """Estimate tensors retained by the streaming inference session."""
        if self._session is None:
            return 0
        seen: set[int] = set()
        values = (
            getattr(self._session, "processed_frames", None),
            getattr(self._session, "output_dict_per_obj", None),
            getattr(self._session, "point_inputs_per_obj", None),
            getattr(self._session, "mask_inputs_per_obj", None),
            getattr(getattr(self._session, "cache", None), "_vision_features", None),
        )
        return sum(self._retained_tensor_bytes(value, seen) for value in values)

    def _discard_oversized_session(self) -> None:
        if self.retained_session_bytes() <= SAM_SESSION_MEMORY_LIMIT_BYTES:
            return
        self.reset()
        self._needs_reseed = True

    @staticmethod
    def _clean_full_mask(
        predicted: np.ndarray,
        box: tuple[int, int, int, int] | None,
        dilation_ratio: float,
        combine: str,
    ) -> np.ndarray:
        height, width = predicted.shape
        if box is None:
            if combine != "mask-only":
                raise ValueError(
                    "propagated SAM masks require mask-only; apply the "
                    "combination policy with current tracker geometry"
                )
            foreground = np.asarray(predicted) > 0
            rows, columns = np.nonzero(foreground)
            if not len(rows):
                return np.zeros((height, width), dtype=np.uint8)
            inner = (
                int(columns.min()),
                int(rows.min()),
                int(columns.max()) + 1,
                int(rows.max()) + 1,
            )
            # Propagated frames have no detector box to combine with.
            # Derive the dilation scale from the object mask bounds rather
            # than the entire video frame, then keep the actual SAM contour.
        else:
            inner = box
        if box is None:
            mask, _bounds = _clean_sam_mask(
                predicted,
                (height, width),
                inner,
                dilation_ratio,
            )
            return np.ascontiguousarray(mask, dtype=np.uint8)
        if combine == "mask-only":
            mask, _bounds = _clean_sam_mask(
                predicted,
                (height, width),
                inner,
                dilation_ratio,
            )
            return np.ascontiguousarray(mask, dtype=np.uint8)
        return build_safe_sam_mask(
            predicted,
            (height, width),
            inner,
            dilation_ratio,
            combine,
        )

    @staticmethod
    def _normalise_mask_batch(masks: Any) -> np.ndarray:
        """Normalize SAM video masks to [objects, height, width]."""
        values = (
            masks.detach().to("cpu").float().numpy()
            if hasattr(masks, "detach")
            else np.asarray(masks, dtype=np.float32)
        )
        while values.ndim > 3 and values.shape[0] == 1:
            values = values[0]
        if values.ndim == 4 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim == 2:
            values = values[None]
        if values.ndim != 3:
            raise ValueError(
                f"unexpected SAM 2 video mask shape: {values.shape}"
            )
        return values

    @staticmethod
    def _score_rejection(scores: np.ndarray, index: int) -> str | None:
        """Return why an object score is unsafe, or ``None`` if accepted."""
        if index < 0 or index >= len(scores):
            return "missing_object_score"
        try:
            score = float(scores[index])
        except (TypeError, ValueError, OverflowError):
            return "invalid_object_score"
        if not np.isfinite(score):
            return "invalid_object_score"
        if score < SAM_MIN_OBJECT_LOGIT:
            return "low_object_score"
        return None

    def track_frame(
        self,
        frame_bgr: np.ndarray,
        prompts: dict[int, tuple[int, int, int, int]] | None = None,
        dilation_ratio: float = 0.12,
        combine: str = "mask-only",
    ) -> dict[int, np.ndarray | None]:
        """Return full-frame masks keyed by stable tracker ID."""
        self.last_error = None
        self.last_rejections = {}
        prompts = prompts or {}
        try:
            if (
                not isinstance(frame_bgr, np.ndarray)
                or frame_bgr.ndim != 3
                or frame_bgr.shape[2] != 3
                or frame_bgr.size == 0
            ):
                raise ValueError("frame must be a non-empty BGR image")
            if self._session is None:
                self.reset()
            if not prompts and self._session.get_obj_num() == 0:
                return {}

            height, width = frame_bgr.shape[:2]
            if prompts:
                object_ids = list(prompts)
                boxes = [[list(prompts[obj_id]) for obj_id in object_ids]]
                self._processor.add_inputs_to_inference_session(
                    inference_session=self._session,
                    frame_idx=self._frame_index,
                    obj_ids=object_ids,
                    input_boxes=boxes,
                    original_size=(height, width),
                )

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            inputs = self._processor(
                images=rgb,
                device=self.device,
                return_tensors="pt",
            )
            autocast = (
                self._torch.autocast("cuda", dtype=self._torch.bfloat16)
                if self.device.type == "cuda"
                else nullcontext()
            )
            backend_context = (
                self._torch.backends.mkldnn.flags(enabled=False)
                if self.device.type == "cpu"
                else nullcontext()
            )
            with self._torch.inference_mode(), autocast, backend_context:
                output = self._model(
                    inference_session=self._session,
                    frame=inputs.pixel_values[0],
                    frame_idx=self._frame_index,
                )
            masks = self._processor.post_process_masks(
                [output.pred_masks.detach().to("cpu")],
                original_sizes=(
                    inputs.original_sizes.detach().to("cpu")
                    if hasattr(inputs.original_sizes, "detach")
                    else inputs.original_sizes
                ),
                binarize=False,
            )[0]
            masks_array = self._normalise_mask_batch(masks)
            object_ids = list(output.object_ids)
            if len(object_ids) != len(masks_array):
                raise ValueError(
                    "SAM 2 video returned a different number of masks "
                    "than object IDs"
                )
            scores = (
                output.object_score_logits.detach()
                .to("cpu")
                .float()
                .numpy()
                .reshape(-1)
            )
            results = {}
            for index, obj_id in enumerate(object_ids):
                rejection = self._score_rejection(scores, index)
                if rejection is not None:
                    numeric_id = int(obj_id)
                    self.last_rejections[numeric_id] = rejection
                    results[numeric_id] = None
                    continue
                results[int(obj_id)] = self._clean_full_mask(
                    masks_array[index],
                    prompts.get(int(obj_id)),
                    dilation_ratio,
                    combine,
                )
            self._frame_index += 1
            self._discard_oversized_session()
            return results
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            try:
                self.reset()
            except Exception:
                self._session = None
            return {obj_id: None for obj_id in prompts}

    def reverse_propagate(
        self,
        frames_bgr: list[np.ndarray],
        object_id: int,
        box: tuple[int, int, int, int],
        dilation_ratio: float = 0.12,
    ) -> dict[int, np.ndarray]:
        """Propagate a last-frame prompt backward through a bounded window.

        This uses a separate asynchronous video session, so reverse repair
        cannot corrupt the live forward memory for the scene.
        """
        self.last_error = None
        if not frames_bgr:
            return {}
        try:
            height, width = frames_bgr[-1].shape[:2]
            if any(
                not isinstance(frame, np.ndarray)
                or frame.shape != frames_bgr[-1].shape
                for frame in frames_bgr
            ):
                raise ValueError("reverse SAM window contains invalid frames")
            rgb_frames = [
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr
            ]
            dtype = (
                self._torch.bfloat16
                if self.device.type == "cuda"
                else self._torch.float32
            )
            session = self._processor.init_video_session(
                video=rgb_frames,
                inference_device=self.device,
                inference_state_device=self.device,
                processing_device=self.device,
                video_storage_device="cpu",
                max_vision_features_cache_size=1,
                dtype=dtype,
            )
            prompt_index = len(frames_bgr) - 1
            self._processor.add_inputs_to_inference_session(
                inference_session=session,
                frame_idx=prompt_index,
                obj_ids=[int(object_id)],
                input_boxes=[[list(box)]],
                original_size=(height, width),
            )
            autocast = (
                self._torch.autocast("cuda", dtype=self._torch.bfloat16)
                if self.device.type == "cuda"
                else nullcontext()
            )
            backend_context = (
                self._torch.backends.mkldnn.flags(enabled=False)
                if self.device.type == "cpu"
                else nullcontext()
            )
            results: dict[int, np.ndarray] = {}
            with self._torch.inference_mode(), autocast, backend_context:
                # Materialize the prompted frame into memory before asking the
                # official iterator to walk in reverse.
                self._model(inference_session=session, frame_idx=prompt_index)
                outputs = self._model.propagate_in_video_iterator(
                    inference_session=session,
                    start_frame_idx=prompt_index,
                    max_frame_num_to_track=len(frames_bgr),
                    reverse=True,
                    show_progress_bar=False,
                )
                for output in outputs:
                    masks = self._processor.post_process_masks(
                        [output.pred_masks.detach().to("cpu")],
                        original_sizes=[[height, width]],
                        binarize=False,
                    )[0]
                    values = self._normalise_mask_batch(masks)
                    object_ids = [int(value) for value in output.object_ids]
                    if object_id not in object_ids:
                        continue
                    index = object_ids.index(object_id)
                    scores = (
                        output.object_score_logits.detach()
                        .to("cpu")
                        .float()
                        .numpy()
                        .reshape(-1)
                    )
                    if self._score_rejection(scores, index) is not None:
                        continue
                    results[int(output.frame_idx)] = self._clean_full_mask(
                        values[index],
                        None,
                        dilation_ratio,
                        "mask-only",
                    )
            return results
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return {}
