"""Command-line parsing for blur-face."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from .config import (
    IMAGE_SUFFIXES,
    AppConfig,
    ConfigError,
    ImageBatchConfig,
)

INPUT_MANIFEST_MAX_BYTES = 1024 * 1024
INPUT_MANIFEST_MAX_ITEMS = 10_000
UNSUPPORTED_IMAGE_SUFFIXES = frozenset(
    {
        ".avif",
        ".dng",
        ".exr",
        ".gif",
        ".heic",
        ".heif",
        ".ico",
        ".j2k",
        ".jp2",
        ".jxl",
        ".psd",
        ".raw",
        ".svg",
        ".svgz",
    }
)
_ISO_IMAGE_BRANDS = frozenset(
    {
        b"avif",
        b"avis",
        b"heic",
        b"heix",
        b"heim",
        b"heis",
        b"hevc",
        b"hevx",
        b"hevm",
        b"hevs",
        b"mif1",
        b"msf1",
    }
)
_MEDIA_SIGNATURE_LIMIT = 64 * 1024


def _xml_doctype_end(value: str, start: int) -> int | None:
    """Find a bounded DOCTYPE terminator, respecting quotes and subsets."""
    quote: str | None = None
    subset_depth = 0
    index = start
    while index < len(value):
        if quote is None and value.startswith("<!--", index):
            comment_end = value.find("-->", index + 4)
            if comment_end < 0:
                return None
            index = comment_end + 3
            continue
        current = value[index]
        if quote is not None:
            if current == quote:
                quote = None
            index += 1
            continue
        if current in {'"', "'"}:
            quote = current
        elif current == "[":
            subset_depth += 1
        elif current == "]" and subset_depth:
            subset_depth -= 1
        elif current == ">" and not subset_depth:
            return index + 1
        index += 1
    return None


def _xml_prefix_text(header: bytes) -> str:
    """Decode a bounded XML prefix with explicit UTF BOM handling."""
    value = header[:_MEDIA_SIGNATURE_LIMIT]
    if value.startswith(b"\xff\xfe"):
        return value[2:].decode("utf-16-le", errors="replace")
    if value.startswith(b"\xfe\xff"):
        return value[2:].decode("utf-16-be", errors="replace")
    if value.startswith(b"\xef\xbb\xbf"):
        value = value[3:]
    return value.decode("utf-8", errors="replace")


def _xml_root_local_name(header: bytes) -> tuple[bool, str | None]:
    """Return whether input starts as XML and its bounded root local name."""
    value = _xml_prefix_text(header).lstrip("\ufeff\t\r\n ")
    started_as_xml = value.startswith("<")
    while value:
        lowered = value.lower()
        if lowered.startswith("<?xml") or value.startswith("<?"):
            started_as_xml = True
            end = value.find("?>")
            if end < 0:
                return True, None
            value = value[end + 2 :].lstrip("\t\r\n ")
            continue
        if value.startswith("<!--"):
            started_as_xml = True
            end = value.find("-->", 4)
            if end < 0:
                return True, None
            value = value[end + 3 :].lstrip("\t\r\n ")
            continue
        if lowered.startswith("<!doctype"):
            started_as_xml = True
            end = _xml_doctype_end(value, len("<!doctype"))
            if end is None:
                return True, None
            value = value[end:].lstrip("\t\r\n ")
            continue
        break
    if not value.startswith("<"):
        return started_as_xml, None
    name_end = 1
    while (
        name_end < len(value)
        and value[name_end] not in "\x00\t\r\n />"
    ):
        name_end += 1
    qualified_name = value[1:name_end]
    name_parts = qualified_name.split(":")
    if (
        not qualified_name
        or len(name_parts) > 2
        or not all(name_parts)
    ):
        return True, None
    return True, name_parts[-1].lower()


def parse_time_thresh(raw: str) -> tuple[tuple[float, float], ...]:
    """Parse ``sec:threshold,sec:threshold`` into a sorted tuple."""
    if not raw:
        return ()
    segments: list[tuple[float, float]] = []
    try:
        for part in raw.split(","):
            fields = part.strip().split(":")
            if len(fields) != 2:
                raise ValueError
            segments.append((float(fields[0]), float(fields[1])))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            'expected "sec:threshold,sec:threshold"'
        ) from exc
    return tuple(sorted(segments))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blur-face",
        description="Local video and image face anonymization.",
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="*",
        help="Input video, image(s), or one image directory",
    )
    parser.add_argument(
        "--input-list",
        type=Path,
        help="UTF-8 JSON array of input paths (cannot be combined with inputs)",
    )
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing output after successful encoding",
    )
    parser.add_argument(
        "--detector",
        choices=("yunet", "yolo"),
        default="yunet",
        help=(
            "Face detector backend: bundled OpenCV YuNet, or explicitly "
            "installed optional Ultralytics YOLO"
        ),
    )
    parser.add_argument(
        "--model",
        default="face_detection_yunet_2023mar.onnx",
        help="Local detector model (.onnx for YuNet, .pt for YOLO)",
    )
    parser.add_argument("--thresh", type=float, default=0.3)
    parser.add_argument("--time-thresh", type=parse_time_thresh, default=())
    parser.add_argument("--mask-scale", type=float, default=1.5)
    parser.add_argument(
        "--mask-shape",
        choices=("rounded-rect", "rectangle", "ellipse"),
        default="rounded-rect",
        help="Coverage shape (default: rounded-rect)",
    )
    parser.add_argument(
        "--mask-engine",
        choices=("geometric", "sam2.1"),
        default="geometric",
        help="Fast geometric coverage or high-quality SAM 2.1",
    )
    parser.add_argument(
        "--sam-mask-expansion",
        type=float,
        default=0.12,
        help="Face-width fraction added around a SAM mask (default: 0.12)",
    )
    parser.add_argument(
        "--segmentation-combine",
        choices=("union", "intersection", "mask-only"),
        default="union",
        help=(
            "Union/intersection with detector coverage, or mask-only to use "
            "the segmentation contour (failures still use geometry)"
        ),
    )
    parser.add_argument(
        "--sam2-model",
        default="facebook/sam2.1-hiera-base-plus",
        help="Local Transformers model directory or SAM 2.1 model ID",
    )
    parser.add_argument(
        "--sam2-refresh-interval",
        type=int,
        default=15,
        help="Face-detector correction interval for SAM tracking (default: 15)",
    )
    parser.add_argument(
        "--temporal-stabilization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Offline bidirectional mask stabilization for segmentation engines "
            "(slower; uses bounded temporary storage)"
        ),
    )
    parser.add_argument(
        "--backfill-frames",
        type=int,
        default=10,
        help="Maximum same-scene reverse repair window (default: 10)",
    )
    parser.add_argument(
        "--release-hold-frames",
        type=int,
        default=5,
        help="Frames to blend aligned mask changes and geometry (default: 5)",
    )
    parser.add_argument(
        "--scene-cut-sensitivity",
        type=float,
        default=0.55,
        help="Scene-cut sensitivity from 0 to 1 (default: 0.55)",
    )
    parser.add_argument(
        "--blur-strategy",
        choices=("adaptive", "fixed"),
        default="adaptive",
        help="Scale blur with face confidence, or use one fixed kernel",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=251,
        help=(
            "Maximum adaptive kernel, or fixed kernel, at a 1080p baseline; "
            "high-resolution media scales it up"
        ),
    )
    parser.add_argument(
        "--blur-kernel-min",
        type=int,
        default=101,
        help=(
            "Minimum adaptive kernel at a 1080p baseline; high-resolution "
            "media scales it up"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="SAM compute device; auto uses CUDA when available, otherwise CPU",
    )
    parser.add_argument(
        "--temporal-storage-limit-mb",
        type=int,
        default=4096,
        help="Maximum temporary storage for two-pass masks (default: 4096 MiB)",
    )
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        help="FFmpeg executable (default: system FFmpeg on PATH)",
    )
    parser.add_argument(
        "--job-temp-dir",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--debug", action="store_true", help="Draw track coverage; do not blur"
    )
    parser.add_argument(
        "--mask-preview",
        action="store_true",
        help=(
            "Write a black diagnostic video with the exact final mask in blue; "
            "do not render source pixels"
        ),
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--lost-buffer", type=int, default=180)
    parser.add_argument("--smooth", type=float, default=0.7)
    parser.add_argument("--preset", choices=("quality", "fast"), default="quality")
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=30,
        help="Minimum detection width/height in pixels (default: 30)",
    )
    parser.add_argument(
        "--max-face-height-ratio",
        type=float,
        default=1.0,
        help="Maximum face height as a fraction of frame height (default: 1.0)",
    )
    parser.add_argument("--flow-max-points", type=int, default=50)
    parser.add_argument("--flow-min-confirmations", type=int, default=3)
    parser.add_argument("--flow-max-missed", type=int, default=0)
    parser.add_argument("--no-flow", action="store_true")
    parser.add_argument("--no-nvenc", action="store_true")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require all selected models to exist locally",
    )
    return parser


def read_input_manifest(path: Path) -> tuple[Path, ...]:
    """Read a bounded JSON path array used by large local UI batches."""
    manifest = path.expanduser()
    try:
        with manifest.open("rb") as stream:
            encoded = stream.read(INPUT_MANIFEST_MAX_BYTES + 1)
    except OSError as exc:
        raise ConfigError(f"cannot read input manifest: {manifest}") from exc
    if not encoded or len(encoded) > INPUT_MANIFEST_MAX_BYTES:
        raise ConfigError(
            f"input manifest must be between 1 and {INPUT_MANIFEST_MAX_BYTES} bytes"
        )
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"invalid UTF-8 JSON input manifest: {manifest}") from exc
    if not isinstance(payload, list) or not payload:
        raise ConfigError("input manifest must contain a non-empty JSON array")
    if len(payload) > INPUT_MANIFEST_MAX_ITEMS:
        raise ConfigError(
            f"input manifest cannot contain more than {INPUT_MANIFEST_MAX_ITEMS} paths"
        )
    if any(not isinstance(value, str) or not value.strip() for value in payload):
        raise ConfigError("every input manifest item must be a non-empty path string")
    return tuple(Path(value) for value in payload)


def _media_signature(path: Path) -> str:
    """Classify common media signatures without importing a decoder."""
    try:
        with path.expanduser().open("rb") as stream:
            header = stream.read(_MEDIA_SIGNATURE_LIMIT)
            stream.seek(0, 2)
            file_size = stream.tell()
    except OSError:
        return "unknown"
    if header.startswith(b"\xff\xd8\xff"):
        return "image"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image"
    if header.startswith(b"BM"):
        return "image"
    if header.startswith((b"II*\x00", b"MM\x00*")):
        return "image"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "unsupported-image"
    if header.startswith(b"\x00\x00\x01\x00"):
        return "unsupported-image"
    if header.startswith((b"8BPS", b"\x76\x2f\x31\x01")):
        return "unsupported-image"
    if header.startswith(b"\x00\x00\x00\x0cjP  \x0d\x0a\x87\x0a"):
        return "unsupported-image"
    if header.startswith((b"\xff\x0a", b"\x00\x00\x00\x0cJXL \x0d\x0a\x87\x0a")):
        return "unsupported-image"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image"
    if len(header) >= 8 and header[4:8] == b"ftyp":
        box_size = int.from_bytes(header[:4], "big")
        brand_offset = 8
        compatible_offset = 16
        if box_size == 1:
            if len(header) < 20:
                return "ambiguous"
            box_size = int.from_bytes(header[8:16], "big")
            brand_offset = 16
            compatible_offset = 24
        elif box_size == 0:
            box_size = file_size
        available_size = min(len(header), box_size)
        brands = set()
        if available_size >= brand_offset + 4:
            brands.add(header[brand_offset : brand_offset + 4].lower())
        if brands & _ISO_IMAGE_BRANDS:
            return "unsupported-image"
        if (
            box_size < compatible_offset
            or box_size > len(header)
            or (box_size - compatible_offset) % 4 != 0
        ):
            return "ambiguous"
        available_size = box_size
        if available_size >= compatible_offset + 4:
            brands.update(
                header[index : index + 4].lower()
                for index in range(
                    compatible_offset, available_size - 3, 4
                )
            )
        if brands & _ISO_IMAGE_BRANDS:
            return "unsupported-image"
        return "video"
    if header.startswith(b"\x1aE\xdf\xa3"):
        return "video"
    if header.startswith(b"FLV") or header.startswith(b"OggS"):
        return "video"
    if header.startswith(b"RIFF") and header[8:12] == b"AVI ":
        return "video"
    if header.startswith((b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3")):
        return "video"
    started_as_xml, xml_root = _xml_root_local_name(header)
    if xml_root == "svg":
        return "unsupported-image"
    if started_as_xml:
        return "ambiguous"
    return "unknown"


def _image_inputs(paths: tuple[Path, ...]) -> tuple[tuple[Path, ...], Path | None]:
    """Classify and expand image inputs, returning an optional source directory."""
    directories = [path for path in paths if path.expanduser().is_dir()]
    if directories:
        if len(paths) != 1:
            raise ConfigError("an image directory cannot be mixed with other inputs")
        directory = directories[0].expanduser()
        images = tuple(
            sorted(
                (
                    path
                    for path in directory.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                ),
                key=lambda path: (path.name.casefold(), path.name),
            )
        )
        if not images:
            raise ConfigError(f"input directory contains no supported images: {directory}")
        invalid = next(
            (
                path
                for path in images
                if _media_signature(path)
                in {"video", "unsupported-image", "ambiguous"}
            ),
            None,
        )
        if invalid is not None:
            raise ConfigError(f"file is not a supported image: {invalid}")
        return images, directory
    if len(paths) == 1:
        path = paths[0].expanduser()
        signature = _media_signature(path)
        if signature == "video":
            return (), None
        if signature == "unsupported-image":
            raise ConfigError(f"unsupported image format: {path}")
        if signature == "ambiguous":
            raise ConfigError(f"ambiguous or unsupported media format: {path}")
        if path.suffix.lower() in IMAGE_SUFFIXES:
            return (path,), None
        if signature == "image" or path.suffix.lower() in UNSUPPORTED_IMAGE_SUFFIXES:
            raise ConfigError(f"unsupported image filename or format: {path}")
        # Unknown single-file signatures retain the original decoder-based
        # video behavior for uncommon containers.
        return (), None
    if any(path.suffix.lower() not in IMAGE_SUFFIXES for path in paths):
        invalid = next(path for path in paths if path.suffix.lower() not in IMAGE_SUFFIXES)
        raise ConfigError(f"unsupported input format: {invalid}")
    expanded = tuple(path.expanduser() for path in paths)
    invalid = next(
        (
            path
            for path in expanded
            if _media_signature(path)
            in {"video", "unsupported-image", "ambiguous"}
        ),
        None,
    )
    if invalid is not None:
        raise ConfigError(f"file is not a supported image: {invalid}")
    return expanded, None


def _image_destinations(
    inputs: tuple[Path, ...],
    source_directory: Path | None,
    output: Path | None,
) -> tuple[tuple[Path, ...], Path]:
    batch = source_directory is not None or len(inputs) > 1
    if output is None:
        if batch:
            if source_directory is not None:
                output = source_directory.with_name(f"{source_directory.name}_blurred")
            else:
                output = inputs[0].parent / f"{inputs[0].stem}_blurred"
        else:
            source = inputs[0]
            output = source.with_name(f"{source.stem}_blurred{source.suffix}")
    output = output.expanduser()
    if batch:
        if output.suffix.lower() in IMAGE_SUFFIXES and not output.is_dir():
            raise ConfigError("multiple images require an output directory")
        return tuple(output / source.name for source in inputs), output
    source = inputs[0]
    if output.is_dir() or not output.suffix:
        return (output / source.name,), output
    if output.suffix.lower() not in IMAGE_SUFFIXES:
        raise ConfigError(f"unsupported output image format: {output}")
    return (output,), output.parent


def _app_config(args, input_path: Path, output_path: Path) -> AppConfig:
    flow_max_points = args.flow_max_points
    flow_max_missed = args.flow_max_missed
    if args.preset == "fast":
        if flow_max_points == 50:
            flow_max_points = 20
        if flow_max_missed == 0:
            flow_max_missed = 45
    return AppConfig(
        input=input_path,
        output=output_path,
        overwrite=args.overwrite,
        detector=args.detector,
        model=args.model,
        threshold=args.thresh,
        time_thresholds=args.time_thresh,
        mask_scale=args.mask_scale,
        mask_shape=args.mask_shape,
        mask_engine=args.mask_engine,
        sam_mask_expansion=args.sam_mask_expansion,
        segmentation_combine=args.segmentation_combine,
        sam2_model=args.sam2_model,
        sam2_refresh_interval=args.sam2_refresh_interval,
        temporal_stabilization=args.temporal_stabilization,
        backfill_frames=args.backfill_frames,
        release_hold_frames=args.release_hold_frames,
        scene_cut_sensitivity=args.scene_cut_sensitivity,
        temporal_storage_limit_mb=args.temporal_storage_limit_mb,
        blur_strategy=args.blur_strategy,
        blur_kernel=args.blur_kernel,
        blur_kernel_min=args.blur_kernel_min,
        device=args.device,
        ffmpeg=args.ffmpeg,
        job_temp_dir=args.job_temp_dir,
        debug=args.debug,
        mask_preview=args.mask_preview,
        profile=args.profile,
        lost_buffer=args.lost_buffer,
        smooth=args.smooth,
        preset=args.preset,
        min_face_size=args.min_face_size,
        max_face_height_ratio=args.max_face_height_ratio,
        flow_max_points=flow_max_points,
        flow_min_confirmations=args.flow_min_confirmations,
        flow_max_missed=flow_max_missed,
        flow_enabled=not args.no_flow,
        use_nvenc=not args.no_nvenc,
        offline=args.offline,
    )


def parse_args(
    argv: Sequence[str] | None = None,
) -> AppConfig | ImageBatchConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.input_list is not None and args.input:
            raise ConfigError("--input-list cannot be combined with positional inputs")
        paths = (
            read_input_manifest(args.input_list)
            if args.input_list is not None
            else tuple(args.input)
        )
        if not paths:
            raise ConfigError("at least one input path is required")
        image_inputs, source_directory = _image_inputs(paths)
        if not image_inputs:
            if len(paths) != 1:
                raise ConfigError("only one video can be processed at a time")
            output = args.output or Path("output_blur.mp4")
            return _app_config(args, paths[0], output).validate()
        outputs, output_directory = _image_destinations(
            image_inputs, source_directory, args.output
        )
        options = _app_config(args, image_inputs[0], outputs[0])
        return ImageBatchConfig(
            options=options,
            inputs=image_inputs,
            outputs=outputs,
            output_directory=output_directory,
        ).validate()
    except ConfigError as exc:
        parser.error(str(exc))


# Compatibility helpers for callers of the original module API.
def get_thresh(args, frame_idx: int, fps: float) -> float:
    if isinstance(args, AppConfig):
        return args.threshold_for(frame_idx, fps)
    if not args.time_thresh:
        return args.thresh
    if fps <= 0:
        raise ConfigError("video FPS must be positive when --time-thresh is used")
    threshold = args.thresh
    for second, candidate in args.time_thresh:
        if frame_idx / fps >= second:
            threshold = candidate
    return threshold
