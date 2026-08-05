"""Bounded, atomic still-image face anonymization."""

from __future__ import annotations

from contextlib import ExitStack
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
import uuid
import zlib

import cv2
import numpy as np

from .config import ImageBatchConfig
from .detector import FaceDetector
from .pipeline import VideoProcessor
from .renderer import (
    apply_blur,
    apply_mask_preview,
    draw_debug_box,
    prepare_mask_region,
)
from .sam2_segmenter import Sam2Segmenter


class ImageInputError(RuntimeError):
    """Raised when an image cannot be decoded or safely encoded."""


_ALPHA_OUTPUT_SUFFIXES = frozenset({".png", ".webp", ".tif", ".tiff"})


def _print_status(message: str) -> None:
    """Print status text safely on legacy consoles with narrow encodings."""
    encoding = getattr(sys.stdout, "encoding", None)
    if encoding:
        try:
            message.encode(encoding)
        except (LookupError, UnicodeEncodeError):
            try:
                message = message.encode(
                    encoding, errors="backslashreplace"
                ).decode(encoding)
            except LookupError:
                message = message.encode(
                    "ascii", errors="backslashreplace"
                ).decode("ascii")
    print(message)


def _tiff_orientation(
    payload: bytes | bytearray,
    *,
    neutralize: bool = False,
) -> int:
    """Read a bounded EXIF Orientation value from one TIFF payload."""
    original = payload
    base_offset = 0
    if payload.startswith(b"Exif\x00\x00"):
        base_offset = 6
        payload = payload[6:]
    if len(payload) < 8 or bytes(payload[:2]) not in {b"II", b"MM"}:
        raise ImageInputError("invalid EXIF orientation metadata")
    byteorder = "little" if payload[:2] == b"II" else "big"
    if int.from_bytes(payload[2:4], byteorder) != 42:
        raise ImageInputError("invalid EXIF orientation metadata")
    ifd_offset = int.from_bytes(payload[4:8], byteorder)
    if ifd_offset > len(payload) - 2:
        raise ImageInputError("invalid EXIF orientation metadata")
    entry_count = int.from_bytes(payload[ifd_offset : ifd_offset + 2], byteorder)
    entries_end = ifd_offset + 2 + entry_count * 12
    if entries_end > len(payload):
        raise ImageInputError("invalid EXIF orientation metadata")
    for offset in range(ifd_offset + 2, entries_end, 12):
        tag = int.from_bytes(payload[offset : offset + 2], byteorder)
        if tag != 0x0112:
            continue
        value_type = int.from_bytes(payload[offset + 2 : offset + 4], byteorder)
        count = int.from_bytes(payload[offset + 4 : offset + 8], byteorder)
        if value_type != 3 or count != 1:
            raise ImageInputError("invalid EXIF orientation metadata")
        orientation = int.from_bytes(
            payload[offset + 8 : offset + 10], byteorder
        )
        if orientation not in range(1, 9):
            raise ImageInputError("invalid EXIF orientation metadata")
        if neutralize:
            if not isinstance(original, bytearray):
                raise TypeError("neutralizing TIFF orientation requires bytearray")
            value_offset = base_offset + offset + 8
            original[value_offset : value_offset + 2] = (1).to_bytes(
                2, byteorder
            )
        return orientation
    return 1


def _alpha_exif_orientation(encoded: bytes) -> int:
    """Extract orientation from alpha-capable formats without optional Pillow."""
    if encoded.startswith((b"II*\x00", b"MM\x00*")):
        return _tiff_orientation(encoded)
    if encoded.startswith(b"\x89PNG\r\n\x1a\n"):
        offset = 8
        orientation = 1
        seen_exif = False
        seen_image_data = False
        while offset <= len(encoded) - 12:
            size = int.from_bytes(encoded[offset : offset + 4], "big")
            chunk_end = offset + 12 + size
            if chunk_end > len(encoded):
                raise ImageInputError("invalid PNG metadata")
            chunk_type = encoded[offset + 4 : offset + 8]
            chunk_data = encoded[offset + 8 : offset + 8 + size]
            if chunk_type == b"IEND":
                if size != 0:
                    raise ImageInputError("invalid PNG metadata")
                return orientation
            if chunk_type == b"IDAT":
                seen_image_data = True
            elif chunk_type == b"eXIf":
                if seen_exif or seen_image_data:
                    raise ImageInputError("invalid PNG EXIF metadata placement")
                expected_crc = int.from_bytes(
                    encoded[offset + 8 + size : chunk_end], "big"
                )
                actual_crc = zlib.crc32(chunk_data, zlib.crc32(chunk_type))
                if actual_crc & 0xFFFFFFFF != expected_crc:
                    raise ImageInputError("invalid PNG EXIF metadata checksum")
                orientation = _tiff_orientation(chunk_data)
                seen_exif = True
            offset = chunk_end
        raise ImageInputError("invalid PNG metadata: missing IEND")
    if encoded.startswith(b"RIFF") and encoded[8:12] == b"WEBP":
        declared_size = int.from_bytes(encoded[4:8], "little")
        logical_end = min(len(encoded), 8 + declared_size)
        if logical_end < 12:
            raise ImageInputError("invalid WebP metadata")
        offset = 12
        orientation = 1
        seen_exif = False
        while offset <= logical_end - 8:
            size = int.from_bytes(encoded[offset + 4 : offset + 8], "little")
            chunk_end = offset + 8 + size
            if chunk_end > logical_end:
                raise ImageInputError("invalid WebP metadata")
            if encoded[offset : offset + 4] == b"EXIF":
                if seen_exif:
                    raise ImageInputError("duplicate WebP EXIF metadata")
                orientation = _tiff_orientation(encoded[offset + 8 : chunk_end])
                seen_exif = True
            offset = chunk_end + (size & 1)
        return orientation
    return 1


def _apply_exif_orientation(image: np.ndarray, orientation: int) -> np.ndarray:
    """Apply all eight EXIF orientation transforms to color and alpha together."""
    if orientation == 2:
        image = cv2.flip(image, 1)
    elif orientation == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 4:
        image = cv2.flip(image, 0)
    elif orientation == 5:
        image = cv2.transpose(image)
    elif orientation == 6:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        image = cv2.flip(cv2.transpose(image), -1)
    elif orientation == 8:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return np.ascontiguousarray(image)


def _directory_identity(path: Path) -> tuple[int, int]:
    """Return a non-following identity for an ordinary directory."""
    try:
        info = path.lstat()
    except OSError as exc:
        raise ImageInputError(f"cannot inspect output directory: {path}") from exc
    is_junction = getattr(path, "is_junction", None)
    file_attributes = getattr(info, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if (
        stat.S_ISLNK(info.st_mode)
        or (is_junction is not None and is_junction())
        or bool(file_attributes & reparse_flag)
        or not stat.S_ISDIR(info.st_mode)
    ):
        raise ImageInputError(f"output directory is not an ordinary directory: {path}")
    return info.st_dev, info.st_ino


def _validate_directory_identity(path: Path, expected: tuple[int, int]) -> None:
    if _directory_identity(path) != expected:
        raise ImageInputError(f"output directory changed during processing: {path}")


class _DirectoryAnchor:
    """Keep a directory stable and expose relative operations where supported."""

    def __init__(self, path: Path, expected: tuple[int, int]):
        self.dir_fd: int | None = None
        self._windows_handle: int | None = None
        if os.name == "nt":
            self._open_windows(path, expected)
            return
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise ImageInputError(f"cannot anchor output directory: {path}") from exc
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) != expected:
            os.close(descriptor)
            raise ImageInputError(
                f"directory changed while it was being anchored: {path}"
            )
        self.dir_fd = descriptor

    def _open_windows(self, path: Path, expected: tuple[int, int]) -> None:
        # Omitting FILE_SHARE_DELETE prevents the opened directory from being
        # renamed or replaced while path-based Windows commits are in flight.
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        create_file.restype = wintypes.HANDLE
        handle = create_file(
            str(path),
            0x0001,  # FILE_LIST_DIRECTORY
            0x0001 | 0x0002,  # share read/write, deliberately not delete
            None,
            3,  # OPEN_EXISTING
            0x02000000 | 0x00200000,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
            None,
        )
        invalid_handle = ctypes.c_void_p(-1).value
        if handle in {None, invalid_handle}:
            error = ctypes.get_last_error()
            raise ImageInputError(
                f"cannot anchor output directory: {path} (Windows error {error})"
            )
        self._windows_handle = int(handle)
        try:
            if _directory_identity(path) != expected:
                raise ImageInputError(
                    f"directory changed while it was being anchored: {path}"
                )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self.dir_fd is not None:
            os.close(self.dir_fd)
            self.dir_fd = None
        if self._windows_handle is not None:
            import ctypes
            from ctypes import wintypes

            close_handle = ctypes.WinDLL(
                "kernel32", use_last_error=True
            ).CloseHandle
            close_handle.argtypes = (wintypes.HANDLE,)
            close_handle.restype = wintypes.BOOL
            close_handle(self._windows_handle)
            self._windows_handle = None

    @classmethod
    def open_relative(
        cls,
        parent_fd: int,
        name: str,
        expected: tuple[int, int],
    ) -> _DirectoryAnchor:
        """Open a newly created POSIX child without resolving its parent path."""
        anchor = cls.__new__(cls)
        anchor.dir_fd = None
        anchor._windows_handle = None
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            anchor.dir_fd = os.open(name, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise ImageInputError(f"cannot anchor image job directory: {name}") from exc
        info = os.fstat(anchor.dir_fd)
        if (info.st_dev, info.st_ino) != expected:
            anchor.close()
            raise ImageInputError(
                f"image job directory changed while being anchored: {name}"
            )
        return anchor


def _open_directory_anchor(
    path: Path, expected: tuple[int, int]
) -> _DirectoryAnchor:
    return _DirectoryAnchor(path, expected)


def _cleanup_owned_job_temp(
    path: Path,
    job_anchor: _DirectoryAnchor,
    output_anchor: _DirectoryAnchor,
) -> None:
    """Remove only direct files from an anchored image-job directory."""
    try:
        if job_anchor.dir_fd is not None:
            for name in os.listdir(job_anchor.dir_fd):
                os.unlink(name, dir_fd=job_anchor.dir_fd)
        else:
            # The non-delete-sharing Windows handles keep both path components
            # stable until direct child cleanup has completed.
            for child in path.iterdir():
                child.unlink()
    finally:
        job_anchor.close()
    if output_anchor.dir_fd is not None:
        os.rmdir(path.name, dir_fd=output_anchor.dir_fd)
    else:
        path.rmdir()


def _create_owned_job_temp(
    output_directory: Path,
    output_anchor: _DirectoryAnchor,
) -> tuple[Path, _DirectoryAnchor]:
    """Create the private job directory through the stable output anchor."""
    if output_anchor.dir_fd is not None:
        for _attempt in range(100):
            name = f".blur-face-image-job-{uuid.uuid4().hex}"
            try:
                os.mkdir(name, 0o700, dir_fd=output_anchor.dir_fd)
            except FileExistsError:
                continue
            info = os.stat(
                name, dir_fd=output_anchor.dir_fd, follow_symlinks=False
            )
            if not stat.S_ISDIR(info.st_mode):
                raise ImageInputError(
                    f"new image job path is not an ordinary directory: {name}"
                )
            return (
                output_directory / name,
                _DirectoryAnchor.open_relative(
                    output_anchor.dir_fd,
                    name,
                    (info.st_dev, info.st_ino),
                ),
            )
        raise ImageInputError("cannot allocate a unique image job directory")
    # The Windows output handle denies delete sharing, so this parent path
    # cannot be renamed or replaced between anchor creation and mkdtemp.
    path = Path(
        tempfile.mkdtemp(prefix=".blur-face-image-job-", dir=output_directory)
    )
    return path, _open_directory_anchor(path, _directory_identity(path))


def _decode_image(path: Path) -> np.ndarray:
    """Decode a Unicode path without relying on OpenCV's path handling."""
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise ImageInputError(f"cannot read input image: {path}") from exc
    if encoded.size == 0:
        raise ImageInputError(f"input image is empty: {path}")
    unchanged = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if unchanged is None or unchanged.size == 0:
        raise ImageInputError(f"cannot decode input image: {path}")
    if unchanged.ndim == 3 and unchanged.shape[2] == 4:
        encoded_bytes = encoded.tobytes()
        if encoded_bytes.startswith((b"II*\x00", b"MM\x00*")):
            # OpenCV applies TIFF Orientation even under IMREAD_UNCHANGED.
            # Neutralize the tag in an in-memory copy, decode the stored pixel
            # order, then apply the original orientation exactly once to BGRA.
            neutralized = bytearray(encoded_bytes)
            orientation = _tiff_orientation(neutralized, neutralize=True)
            unchanged = cv2.imdecode(
                np.frombuffer(neutralized, dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
            if (
                unchanged is None
                or unchanged.ndim != 3
                or unchanged.shape[2] != 4
                or unchanged.size == 0
            ):
                raise ImageInputError(f"cannot decode input image: {path}")
        else:
            orientation = _alpha_exif_orientation(encoded_bytes)
        return _apply_exif_orientation(unchanged, orientation)
    # IMREAD_COLOR applies JPEG EXIF orientation in supported OpenCV builds.
    # Decode opaque/grayscale inputs that way to preserve the existing behavior.
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ImageInputError(f"cannot decode input image: {path}")
    return image


def _encode_image(image: np.ndarray, suffix: str) -> bytes:
    if image.ndim == 3 and image.shape[2] == 4 and suffix not in _ALPHA_OUTPUT_SUFFIXES:
        raise ImageInputError(
            f"output image format {suffix} cannot safely preserve transparency"
        )
    try:
        ok, encoded = cv2.imencode(suffix, image)
    except cv2.error as exc:
        raise ImageInputError(f"cannot encode output image format {suffix}") from exc
    if not ok or encoded is None or encoded.size == 0:
        raise ImageInputError(f"image encoder produced no {suffix} output")
    return encoded.tobytes()


def _atomic_write_image(
    image: np.ndarray,
    output: Path,
    job_temp: Path,
    overwrite: bool,
    directory_identity: tuple[int, int] | None = None,
    output_directory_fd: int | None = None,
    job_temp_directory_fd: int | None = None,
) -> None:
    """Commit one complete image without exposing partial output bytes."""
    payload = _encode_image(image, output.suffix.lower())
    temporary_name = f"{uuid.uuid4().hex}.partial"
    temporary = job_temp / temporary_name
    committed = False
    try:
        if job_temp_directory_fd is None:
            stream_context = temporary.open("xb")
        else:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=job_temp_directory_fd,
            )
            stream_context = os.fdopen(descriptor, "wb")
        with stream_context as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if directory_identity is not None:
            _validate_directory_identity(output.parent, directory_identity)
        if overwrite:
            if output_directory_fd is not None and job_temp_directory_fd is not None:
                os.replace(
                    temporary_name,
                    output.name,
                    src_dir_fd=job_temp_directory_fd,
                    dst_dir_fd=output_directory_fd,
                )
            else:
                os.replace(temporary, output)
            committed = True
        else:
            try:
                if output_directory_fd is not None and job_temp_directory_fd is not None:
                    os.link(
                        temporary_name,
                        output.name,
                        src_dir_fd=job_temp_directory_fd,
                        dst_dir_fd=output_directory_fd,
                        follow_symlinks=False,
                    )
                else:
                    os.link(temporary, output)
            except FileExistsError as exc:
                raise ImageInputError(
                    f"output appeared during processing and was not replaced: {output}"
                ) from exc
            except OSError as exc:
                raise ImageInputError(
                    "filesystem cannot atomically commit without overwrite; "
                    f"destination was left untouched: {output}"
                ) from exc
            committed = True
            try:
                if job_temp_directory_fd is not None:
                    os.unlink(temporary_name, dir_fd=job_temp_directory_fd)
                else:
                    temporary.unlink()
            except OSError:
                # The destination hard link already exposes the complete file.
                # The job-owned directory cleanup can reclaim this extra name.
                pass
    finally:
        if not committed:
            try:
                if job_temp_directory_fd is not None:
                    os.unlink(temporary_name, dir_fd=job_temp_directory_fd)
                else:
                    temporary.unlink(missing_ok=True)
            except FileNotFoundError:
                pass


class ImageProcessor:
    """Reuse the existing detector and renderer for one or more still images."""

    def __init__(self, config: ImageBatchConfig):
        output_directory = config.output_directory.expanduser().resolve()
        self.config = config.validate()
        self._output_directory = output_directory
        self._initial_output_identity = (
            _directory_identity(output_directory)
            if os.path.lexists(output_directory)
            else None
        )
        self._sources = tuple(
            path.expanduser().resolve(strict=True) for path in self.config.inputs
        )
        outputs: list[Path] = []
        for configured in self.config.outputs:
            expanded = configured.expanduser()
            if expanded.parent.resolve() != output_directory:
                raise ImageInputError(
                    f"image output is outside the anchored output directory: {configured}"
                )
            outputs.append(output_directory / expanded.name)
        for output in outputs:
            for source in self._sources:
                same_file = output == source
                if os.path.lexists(output):
                    try:
                        same_file = same_file or os.path.samefile(output, source)
                    except OSError as exc:
                        raise ImageInputError(
                            f"cannot verify image output identity: {output}"
                        ) from exc
                if same_file:
                    raise ImageInputError(
                        f"an image output would replace an input image: {source}"
                    )
        self._outputs = tuple(outputs)
        self._segmentation_warning_count = 0

    @staticmethod
    def _detections(boxes, shape, min_size: int, max_height_ratio: float):
        height, width = shape
        accepted: list[tuple[list[int], float]] = []
        if boxes is None:
            return accepted
        try:
            candidates = np.asarray(boxes)
        except (TypeError, ValueError) as exc:
            raise ImageInputError("detector returned malformed image boxes") from exc
        if np.iscomplexobj(candidates):
            raise ImageInputError("detector returned malformed image boxes")
        if candidates.size == 0:
            if candidates.ndim >= 1 and candidates.shape[0] == 0:
                return accepted
            raise ImageInputError("detector returned malformed image boxes")
        if candidates.ndim == 1:
            candidates = candidates.reshape(1, -1)
        elif candidates.ndim != 2:
            raise ImageInputError("detector returned malformed image boxes")
        for row in candidates:
            try:
                values = np.asarray(row, dtype=float).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise ImageInputError(
                    "detector returned malformed image boxes"
                ) from exc
            if values.size < 4:
                raise ImageInputError("detector returned incomplete image boxes")
            if not np.all(np.isfinite(values[:4])):
                raise ImageInputError(
                    "detector returned non-finite image coordinates"
                )
            x1, x2 = sorted((float(values[0]), float(values[2])))
            y1, y2 = sorted((float(values[1]), float(values[3])))
            x1 = max(0, min(width, int(np.floor(x1))))
            x2 = max(0, min(width, int(np.ceil(x2))))
            y1 = max(0, min(height, int(np.floor(y1))))
            y2 = max(0, min(height, int(np.ceil(y2))))
            if (
                x2 - x1 < min_size
                or y2 - y1 < min_size
                or (y2 - y1) / height > max_height_ratio
            ):
                continue
            confidence = (
                float(values[4])
                if values.size >= 5 and np.isfinite(values[4])
                else 1.0
            )
            accepted.append(
                ([x1, y1, x2, y2], max(0.0, min(1.0, confidence)))
            )
        return accepted

    def _process_one(self, source: Path, detector, segmenter) -> np.ndarray:
        config = self.config.options
        decoded = _decode_image(source)
        if decoded.shape[2] == 4:
            original = np.ascontiguousarray(decoded[:, :, :3])
            alpha = np.ascontiguousarray(decoded[:, :, 3])
        else:
            original = decoded
            alpha = None
        boxes = detector.detect(original, conf=config.threshold)
        detections = self._detections(
            boxes,
            original.shape[:2],
            config.min_face_size,
            config.max_face_height_ratio,
        )
        rendered = original.copy()
        if config.mask_preview:
            rendered[:] = 0
        for track_id, (box, confidence) in enumerate(detections):
            if config.debug:
                draw_debug_box(
                    rendered, box, track_id, confidence=confidence
                )
                continue
            coverage = None
            if segmenter is not None:
                bounds, inner = prepare_mask_region(
                    original, box, config.mask_scale
                )
                bx1, by1, bx2, by2 = bounds
                roi = original[by1:by2, bx1:bx2].copy()
                coverage = segmenter.build_contour(
                    roi, inner, config.sam_mask_expansion
                )
                if coverage is not None and (
                    not isinstance(coverage, np.ndarray)
                    or coverage.shape != roi.shape[:2]
                    or not np.isfinite(coverage).all()
                    or not VideoProcessor._sam_mask_matches_track(coverage, inner)
                ):
                    coverage = None
                if (
                    coverage is None
                    and segmenter.last_error
                    and self._segmentation_warning_count < 3
                ):
                    _print_status(
                        "[WARN] SAM 2 image segmentation failed; using geometric "
                        f"coverage for {source.name}: {segmenter.last_error}"
                    )
                    self._segmentation_warning_count += 1
            effect = apply_mask_preview if config.mask_preview else apply_blur
            effect(
                rendered,
                box,
                config.blur_kernel_for_frame(
                    confidence,
                    config.threshold,
                    original.shape[:2],
                ),
                mask_scale=config.mask_scale,
                mask_shape=config.mask_shape,
                coverage_mask=coverage,
                segmentation_combine=config.segmentation_combine,
            )
        if alpha is not None:
            return np.dstack((rendered, alpha))
        return rendered

    def run(self) -> None:
        batch = self.config
        config = batch.options
        output_directory = self._output_directory
        output_directory.mkdir(parents=True, exist_ok=True)
        output_identity = _directory_identity(output_directory)
        if (
            self._initial_output_identity is not None
            and output_identity != self._initial_output_identity
        ):
            raise ImageInputError(
                f"output directory changed after validation: {output_directory}"
            )
        total = len(batch.inputs)
        started = time.perf_counter()
        completed = 0
        with ExitStack() as anchors:
            output_anchor = _open_directory_anchor(
                output_directory, output_identity
            )
            anchors.callback(output_anchor.close)
            if config.job_temp_dir is not None:
                job_temp = config.job_temp_dir.expanduser().resolve()
                if job_temp.parent != output_directory:
                    raise ImageInputError(
                        "image job temporary directory must be inside the output directory"
                    )
                owned_job_temp = False
            else:
                job_temp, job_temp_anchor = _create_owned_job_temp(
                    output_directory, output_anchor
                )
                owned_job_temp = True
            if not owned_job_temp:
                job_temp_identity = _directory_identity(job_temp)
                job_temp_anchor = _open_directory_anchor(
                    job_temp, job_temp_identity
                )
            if owned_job_temp:
                anchors.callback(
                    _cleanup_owned_job_temp,
                    job_temp,
                    job_temp_anchor,
                    output_anchor,
                )
            else:
                anchors.callback(job_temp_anchor.close)
            if config.offline:
                _print_status("Model:   strict offline mode")
            _print_status(
                "Privacy: images are processed locally after model initialization"
            )
            _print_status(f"Detector: {config.detector}; model={config.model}")
            detector = FaceDetector(
                config.model,
                device=config.device,
                allow_download=not config.offline,
                backend=config.detector,
            )
            segmenter = None
            if config.mask_engine == "sam2.1" and not config.debug:
                segmenter = Sam2Segmenter(
                    config.sam2_model,
                    device=config.device,
                    offline=config.offline,
                )
            _print_status(f"  0/{total} (0%) phase=images")
            try:
                for source, output in zip(self._sources, self._outputs):
                    rendered = self._process_one(source, detector, segmenter)
                    _atomic_write_image(
                        rendered,
                        output,
                        job_temp,
                        config.overwrite,
                        output_identity,
                        output_anchor.dir_fd,
                        job_temp_anchor.dir_fd,
                    )
                    del rendered
                    completed += 1
                    percent = round(completed * 100 / total)
                    _print_status(
                        f"  {completed}/{total} ({percent}%) phase=images "
                        f"file={source.name}"
                    )
            except Exception:
                _print_status(
                    f"[ERROR] image batch stopped after {completed}/{total} "
                    f"completed outputs; file={source.name}"
                )
                raise
        _print_status(
            f"Completed {completed} image(s) in "
            f"{time.perf_counter() - started:.1f}s"
        )
