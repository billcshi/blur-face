import base64
import json
import io
import os
import stat
import struct
import subprocess
from contextlib import redirect_stdout
from dataclasses import replace
import tempfile
import unittest
import weakref
import zlib
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from blurface.cli import (
    INPUT_MANIFEST_MAX_BYTES,
    INPUT_MANIFEST_MAX_ITEMS,
    _media_signature,
    parse_args,
    read_input_manifest,
)
from blurface.config import AppConfig, ImageBatchConfig
from blurface.image_pipeline import (
    ImageInputError,
    ImageProcessor,
    _apply_exif_orientation,
    _atomic_write_image,
    _cleanup_owned_job_temp,
    _create_owned_job_temp,
    _decode_image,
    _directory_identity,
    _open_directory_anchor,
)


class StaticDetector:
    instances = 0

    def __init__(self, *_args, **_kwargs):
        self.__class__.instances += 1

    def detect(self, _image, conf=0.3):
        del conf
        return np.array([[12, 10, 52, 42, 0.95]], dtype=float)


class NoFaceDetector(StaticDetector):
    def detect(self, _image, conf=0.3):
        del conf
        return np.empty((0, 5), dtype=float)


class EmptySam:
    instances = 0

    def __init__(self, *_args, **_kwargs):
        self.__class__.instances += 1
        self.last_error = None

    def build_contour(self, roi, inner, dilation):
        del inner, dilation
        return np.zeros(roi.shape[:2], dtype=np.uint8)


class ValidSam(EmptySam):
    def build_contour(self, roi, inner, dilation):
        del dilation
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = inner
        mask[y1:y2, x1:x2] = 255
        mask[y1:y2, max(0, x1 - 5) : x1] = 255
        return mask


class InvalidArraySam(EmptySam):
    def build_contour(self, roi, inner, dilation):
        del inner, dilation
        mask = np.ones((roi.shape[0], max(1, roi.shape[1] - 1)), dtype=float)
        mask[0, 0] = np.nan
        return mask


def _source_image(path: Path) -> np.ndarray:
    y, x = np.indices((52, 64))
    checker = ((x + y) % 2 * 255).astype(np.uint8)
    image = np.dstack((checker, 255 - checker, checker))
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError("test image encoding failed")
    path.write_bytes(encoded.tobytes())
    return image


def _oriented_rgba_tiff() -> bytes:
    """Build a minimal uncompressed little-endian RGBA TIFF, Orientation=6."""
    width, height = 3, 2
    entries = 12
    ifd_size = 2 + entries * 12 + 4
    bits_offset = 8 + ifd_size
    pixels_offset = bits_offset + 8

    def entry(tag, value_type, count, value):
        prefix = struct.pack("<HHI", tag, value_type, count)
        if value_type == 3 and count == 1:
            return prefix + struct.pack("<H", value) + b"\x00\x00"
        return prefix + struct.pack("<I", value)

    directory = b"".join(
        (
            entry(256, 4, 1, width),
            entry(257, 4, 1, height),
            entry(258, 3, 4, bits_offset),
            entry(259, 3, 1, 1),
            entry(262, 3, 1, 2),
            entry(273, 4, 1, pixels_offset),
            entry(274, 3, 1, 6),
            entry(277, 3, 1, 4),
            entry(278, 4, 1, height),
            entry(279, 4, 1, width * height * 4),
            entry(284, 3, 1, 1),
            entry(338, 3, 1, 2),
        )
    )
    pixels = bytes(
        (
            255, 0, 0, 10,
            0, 255, 0, 20,
            0, 0, 255, 30,
            255, 255, 0, 40,
            255, 0, 255, 50,
            0, 255, 255, 60,
        )
    )
    return (
        b"II*\x00\x08\x00\x00\x00"
        + struct.pack("<H", entries)
        + directory
        + b"\x00\x00\x00\x00"
        + struct.pack("<HHHH", 8, 8, 8, 8)
        + pixels
    )


def _batch(inputs, outputs, *, mask_engine="geometric", mask_preview=False):
    options = AppConfig(
        input=inputs[0],
        output=outputs[0],
        overwrite=False,
        model="unused.onnx",
        offline=True,
        flow_enabled=False,
        min_face_size=1,
        blur_strategy="fixed",
        blur_kernel=21,
        mask_engine=mask_engine,
        mask_preview=mask_preview,
    )
    return ImageBatchConfig(
        options=options,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        output_directory=outputs[0].parent,
    )


class ImageCliTests(unittest.TestCase):
    def test_single_video_command_remains_compatible(self):
        request = parse_args(["input.mp4", "-o", "output.mp4"])
        self.assertIsInstance(request, AppConfig)
        self.assertEqual(request.input, Path("input.mp4"))

    def test_video_signature_wins_over_misleading_image_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "misleading.jpg"
            source.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 20)
            request = parse_args([str(source), "-o", str(source.with_suffix(".mp4"))])
            self.assertIsInstance(request, AppConfig)

    def test_unknown_video_suffix_retains_decoder_based_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "clip.custom-container"
            source.write_bytes(b"unknown container header")
            request = parse_args([str(source), "-o", str(Path(directory) / "out.mp4")])
            self.assertIsInstance(request, AppConfig)

    def test_known_unsupported_image_does_not_fall_through_to_video(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "animated.gif"
            source.write_bytes(b"GIF89a" + b"\x00" * 20)
            with self.assertRaises(SystemExit):
                parse_args([str(source)])

    def test_iso_image_compatible_brand_is_not_misclassified_as_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = (
                (24).to_bytes(4, "big")
                + b"ftypisom"
                + b"\x00\x00\x00\x00"
                + b"mp42heic",
                (1).to_bytes(4, "big")
                + b"ftyp"
                + (32).to_bytes(8, "big")
                + b"isom"
                + b"\x00\x00\x00\x00"
                + b"mp42heic",
            )
            for index, payload in enumerate(payloads):
                with self.subTest(index=index):
                    source = root / f"compatible-{index}.bin"
                    source.write_bytes(payload)
                    self.assertEqual(_media_signature(source), "unsupported-image")
                    with self.assertRaises(SystemExit):
                        parse_args([str(source)])

    def test_incomplete_large_ftyp_does_not_route_known_heic_to_video(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "late-brand.mp4"
            box_size = 65_572
            payload = (
                box_size.to_bytes(4, "big")
                + b"ftypisom"
                + b"\x00\x00\x00\x00"
                + b"\x00" * (65_568 - 16)
                + b"heic"
            )
            source.write_bytes(payload)
            self.assertEqual(_media_signature(source), "ambiguous")
            with self.assertRaises(SystemExit):
                parse_args([str(source)])

            for suffix in (".bin", ".mp4"):
                with self.subTest(major_brand_suffix=suffix):
                    major_brand = Path(directory) / f"major-heic{suffix}"
                    major_brand.write_bytes(
                        box_size.to_bytes(4, "big")
                        + b"ftypheic"
                        + b"\x00\x00\x00\x00"
                        + b"\x00" * (box_size - 16)
                    )
                    self.assertEqual(
                        _media_signature(major_brand), "unsupported-image"
                    )
                    with self.assertRaises(SystemExit):
                        parse_args([str(major_brand)])

    def test_partial_ftyp_compatible_brand_is_ambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sizes = (*range(8, 16), 17, 18, 19, 21, 22, 23)
            for size in sizes:
                with self.subTest(size=size):
                    source = root / f"partial-{size}.mp4"
                    payload = (
                        size.to_bytes(4, "big")
                        + b"ftypisom"
                        + b"\x00\x00\x00\x00"
                        + b"heic"
                    )
                    source.write_bytes(payload[:size])
                    self.assertEqual(_media_signature(source), "ambiguous")
                    with self.assertRaises(SystemExit):
                        parse_args([str(source)])

    def test_heif_image_and_sequence_brands_do_not_route_to_video(self):
        with tempfile.TemporaryDirectory() as directory:
            for brand in (b"heim", b"heis", b"hevm", b"hevs"):
                with self.subTest(brand=brand):
                    source = Path(directory) / f"{brand.decode()}.mp4"
                    source.write_bytes(
                        b"\x00\x00\x00\x18ftyp" + brand + b"\x00" * 12
                    )
                    self.assertEqual(
                        _media_signature(source), "unsupported-image"
                    )
                    with self.assertRaises(SystemExit):
                        parse_args([str(source)])

    def test_binary_video_signature_wins_over_embedded_svg_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "metadata.mp4"
            source.write_bytes(
                (24).to_bytes(4, "big")
                + b"ftypisom"
                + b"\x00\x00\x00\x00mp42isom"
                + b"metadata <svg></svg>"
            )
            self.assertEqual(_media_signature(source), "video")
            request = parse_args(
                [str(source), "-o", str(Path(directory) / "output.mp4")]
            )
            self.assertIsInstance(request, AppConfig)

    def test_svg_and_jpeg_xl_signatures_are_known_unsupported_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            svg = root / "vector.unknown"
            svg.write_bytes(b"\xef\xbb\xbf  <?xml version='1.0'?><svg></svg>")
            jxl = root / "photo.unknown"
            jxl.write_bytes(b"\xff\x0a" + b"\x00" * 20)
            self.assertEqual(_media_signature(svg), "unsupported-image")
            self.assertEqual(_media_signature(jxl), "unsupported-image")
            nested = root / "metadata.unknown"
            nested.write_bytes(b"ordinary metadata containing <svg></svg>")
            self.assertEqual(_media_signature(nested), "unknown")
            prolog = root / "namespaced.unknown"
            prolog.write_bytes(
                b"<?xml version='1.0'?>\n"
                b"<!-- bounded leading comment -->\n"
                b"<!DOCTYPE svg:svg [<!ENTITY label 'a > b'>]>\n"
                b"<svg:svg xmlns:svg='http://www.w3.org/2000/svg'></svg:svg>"
            )
            self.assertEqual(_media_signature(prolog), "unsupported-image")
            with self.assertRaises(SystemExit):
                parse_args([str(prolog)])

    def test_utf16_and_commented_doctype_svg_do_not_route_to_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = "<?xml version='1.0'?><svg></svg>"
            fixtures = {
                "utf16-le.mp4": b"\xff\xfe" + document.encode("utf-16-le"),
                "utf16-be.mp4": b"\xfe\xff" + document.encode("utf-16-be"),
                "doctype.mp4": (
                    b"<!DOCTYPE svg [<!-- misleading ] > delimiter -->"
                    b"<!ENTITY label 'ok'>]><svg></svg>"
                ),
            }
            for name, payload in fixtures.items():
                with self.subTest(name=name):
                    source = root / name
                    source.write_bytes(payload)
                    self.assertEqual(
                        _media_signature(source), "unsupported-image"
                    )
                    with self.assertRaises(SystemExit):
                        parse_args([str(source)])

    def test_incomplete_xml_prefix_is_ambiguous_not_video(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "incomplete.mp4"
            source.write_bytes(b"<?xml version='1.0'?><!DOCTYPE svg [")
            self.assertEqual(_media_signature(source), "ambiguous")
            with self.assertRaises(SystemExit):
                parse_args([str(source)])

    def test_single_and_multiple_images_get_deterministic_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "第一张.png"
            second = root / "second.jpg"
            first.touch()
            second.touch()
            single = parse_args([str(first)])
            self.assertIsInstance(single, ImageBatchConfig)
            self.assertEqual(single.outputs, (root / "第一张_blurred.png",))
            batch = parse_args([str(first), str(second), "-o", str(root / "out")])
            self.assertEqual(batch.outputs, (root / "out" / first.name, root / "out" / second.name))

    def test_directory_expansion_is_non_recursive_and_sorted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.PNG").touch()
            (root / "A.jpg").touch()
            (root / "ignore.txt").touch()
            nested = root / "nested"
            nested.mkdir()
            (nested / "hidden.png").touch()
            request = parse_args([str(root), "-o", str(root.parent / "out")])
            self.assertEqual([path.name for path in request.inputs], ["A.jpg", "b.PNG"])

    def test_manifest_is_bounded_and_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "one.png"
            image.touch()
            manifest = root / "inputs.json"
            manifest.write_text(json.dumps([str(image)]), encoding="utf-8")
            self.assertEqual(read_input_manifest(manifest), (image,))
            with self.assertRaises(SystemExit):
                parse_args([str(image), "--input-list", str(manifest)])

            manifest.write_bytes(b"[" + b" " * INPUT_MANIFEST_MAX_BYTES + b"]")
            with self.assertRaisesRegex(ValueError, "between 1"):
                read_input_manifest(manifest)
            manifest.write_text(
                json.dumps(["x"] * (INPUT_MANIFEST_MAX_ITEMS + 1)),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "more than"):
                read_input_manifest(manifest)

    def test_empty_directory_and_mixed_media_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = root / "empty"
            empty.mkdir()
            with self.assertRaises(SystemExit):
                parse_args([str(empty)])
            image = root / "image.png"
            image.touch()
            video = root / "video.mp4"
            video.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 20)
            with self.assertRaises(SystemExit):
                parse_args([str(image), str(video), "-o", str(root / "out")])

    def test_duplicate_batch_basenames_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a" / "face.png"
            second = root / "b" / "FACE.PNG"
            first.parent.mkdir()
            second.parent.mkdir()
            first.touch()
            second.touch()
            with self.assertRaises(SystemExit):
                parse_args([str(first), str(second), "-o", str(root / "out")])

    def test_single_image_rejects_unsupported_output_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            source.touch()
            with self.assertRaises(SystemExit):
                parse_args([str(source), "-o", str(Path(directory) / "result.gif")])


class ImagePipelineTests(unittest.TestCase):
    def test_missing_or_nonfinite_confidence_keeps_finite_geometry(self):
        boxes = np.array(
            [
                [1, 2, 20, 22, np.nan],
                [2, 3, 21, 23, np.inf],
                [3, 4, 22, 24, -2],
                [4, 5, 23, 25, 2],
            ],
            dtype=float,
        )
        accepted = ImageProcessor._detections(boxes, (30, 30), 1, 1.0)
        self.assertEqual(len(accepted), 4)
        self.assertEqual([confidence for _box, confidence in accepted], [1.0, 1.0, 0.0, 1.0])
        missing = ImageProcessor._detections([[1, 2, 20, 22]], (30, 30), 1, 1.0)
        self.assertEqual(missing, [([1, 2, 20, 22], 1.0)])

    def test_malformed_detector_coordinates_fail_closed(self):
        invalid_outputs = (
            [[1, 2, 3]],
            [[np.nan, 2, 20, 22, 0.9]],
            [[1, np.inf, 20, 22, 0.9]],
            [[]],
            [[1 + 2j, 2, 20, 22, 0.9]],
            np.zeros((1, 1, 4), dtype=float),
        )
        for boxes in invalid_outputs:
            with self.subTest(boxes=np.asarray(boxes).shape):
                with self.assertRaisesRegex(ImageInputError, "detector"):
                    ImageProcessor._detections(boxes, (30, 30), 1, 1.0)

    def test_corrupt_nonempty_detector_output_does_not_commit_image(self):
        class CorruptDetector(StaticDetector):
            def detect(self, _image, conf=0.3):
                del conf
                return np.array([[np.nan, 2, 20, 22, 0.9]], dtype=float)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            output = root / "output.png"
            _source_image(source)
            with patch("blurface.image_pipeline.FaceDetector", CorruptDetector):
                with self.assertRaisesRegex(ImageInputError, "detector"):
                    ImageProcessor(_batch([source], [output])).run()
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".blur-face-image-job-*")), [])

    def test_directory_identity_rejects_junction_and_reparse_point(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with patch.object(
                Path, "is_junction", return_value=True, create=True
            ):
                with self.assertRaisesRegex(ImageInputError, "ordinary directory"):
                    _directory_identity(path)

            info = path.lstat()
            reparse_info = Mock(
                st_mode=info.st_mode,
                st_dev=info.st_dev,
                st_ino=info.st_ino,
                st_file_attributes=getattr(
                    stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
                ),
            )
            with (
                patch.object(
                    Path, "is_junction", return_value=False, create=True
                ),
                patch.object(Path, "lstat", return_value=reparse_info),
            ):
                with self.assertRaisesRegex(ImageInputError, "ordinary directory"):
                    _directory_identity(path)

    @unittest.skipUnless(os.name == "nt", "Windows junction regression")
    def test_directory_identity_rejects_real_windows_junction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            junction = root / "junction"
            target.mkdir()
            junction.mkdir()
            expected = _directory_identity(junction)
            junction.rename(root / "original")
            result = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(junction), str(target)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                self.skipTest(f"cannot create Windows junction: {result.stderr}")
            try:
                with self.assertRaisesRegex(ImageInputError, "ordinary directory"):
                    _directory_identity(junction)
                with self.assertRaisesRegex(ImageInputError, "ordinary directory"):
                    _open_directory_anchor(junction, expected)
            finally:
                if os.path.lexists(junction):
                    junction.rmdir()

    def test_jpeg_exif_orientation_is_applied_during_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oriented.jpg"
            image = np.zeros((10, 20, 3), dtype=np.uint8)
            ok, encoded = cv2.imencode(".jpg", image)
            self.assertTrue(ok)
            exif = (
                b"Exif\x00\x00II\x2a\x00\x08\x00\x00\x00\x01\x00"
                b"\x12\x01\x03\x00\x01\x00\x00\x00\x06\x00\x00\x00"
                b"\x00\x00\x00\x00"
            )
            jpeg = encoded.tobytes()
            app1 = b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif
            path.write_bytes(jpeg[:2] + app1 + jpeg[2:])
            decoded = _decode_image(path)
            self.assertEqual(decoded.shape[:2], (20, 10))

    def test_alpha_png_and_webp_apply_exif_orientation_to_all_channels(self):
        fixtures = {
            ".png": (
                "iVBORw0KGgoAAAANSUhEUgAAAAMAAAACCAYAAACddGYaAAAAGmVYSWZNTQ"
                "AqAAAACAABARIAAwAAAAEABgAAAAAAANZnS2kAAAAWSURBVHicY/zPwMDF"
                "AMUsDAwMcjAOABy1AVUomxkgAAAAAElFTkSuQmCC"
            ),
            ".webp": (
                "UklGRloAAABXRUJQVlA4WAoAAAAYAAAAAgAAAQAAVlA4TBkAAAAvAkAAEC8Q"
                "8x9ViABq2jZgQ51/OTWi/zG9AEVYSUYaAAAATU0AKgAAAAgAAQESAAMAAAAB"
                "AAYAAAAAAAA="
            ),
        }
        expected_alpha = np.array(
            [[40, 10], [50, 20], [60, 30]], dtype=np.uint8
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suffix, payload in fixtures.items():
                with self.subTest(suffix=suffix):
                    path = root / f"oriented{suffix}"
                    path.write_bytes(base64.b64decode(payload))
                    decoded = _decode_image(path)
                    self.assertEqual(decoded.shape, (3, 2, 4))
                    np.testing.assert_array_equal(decoded[:, :, 3], expected_alpha)

    def test_alpha_tiff_applies_exif_orientation_exactly_once(self):
        expected_alpha = np.array(
            [[40, 10], [50, 20], [60, 30]], dtype=np.uint8
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oriented.tiff"
            path.write_bytes(_oriented_rgba_tiff())
            decoded = _decode_image(path)
            self.assertEqual(decoded.shape, (3, 2, 4))
            np.testing.assert_array_equal(decoded[:, :, 3], expected_alpha)

    def test_alpha_decoders_ignore_exif_after_container_end(self):
        image = np.zeros((2, 3, 4), dtype=np.uint8)
        image[:, :, 3] = np.array([[10, 20, 30], [40, 50, 60]])
        exif = (
            b"MM\x00*\x00\x00\x00\x08\x00\x01"
            b"\x01\x12\x00\x03\x00\x00\x00\x01\x00\x06\x00\x00"
            b"\x00\x00\x00\x00"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ok, png = cv2.imencode(".png", image)
            self.assertTrue(ok)
            chunk_type = b"eXIf"
            chunk_crc = zlib.crc32(exif, zlib.crc32(chunk_type)) & 0xFFFFFFFF
            trailing_png = (
                png.tobytes()
                + len(exif).to_bytes(4, "big")
                + chunk_type
                + exif
                + chunk_crc.to_bytes(4, "big")
            )
            png_path = root / "trailing.png"
            png_path.write_bytes(trailing_png)
            self.assertEqual(_decode_image(png_path).shape, (2, 3, 4))

            ok, webp = cv2.imencode(".webp", image)
            self.assertTrue(ok)
            trailing_webp = (
                webp.tobytes()
                + b"EXIF"
                + len(exif).to_bytes(4, "little")
                + exif
            )
            webp_path = root / "trailing.webp"
            webp_path.write_bytes(trailing_webp)
            self.assertEqual(_decode_image(webp_path).shape, (2, 3, 4))

    def test_all_exif_orientation_transforms_are_supported(self):
        image = np.arange(6, dtype=np.uint8).reshape(2, 3)
        expected = {
            1: [[0, 1, 2], [3, 4, 5]],
            2: [[2, 1, 0], [5, 4, 3]],
            3: [[5, 4, 3], [2, 1, 0]],
            4: [[3, 4, 5], [0, 1, 2]],
            5: [[0, 3], [1, 4], [2, 5]],
            6: [[3, 0], [4, 1], [5, 2]],
            7: [[5, 2], [4, 1], [3, 0]],
            8: [[2, 5], [1, 4], [0, 3]],
        }
        for orientation, transformed in expected.items():
            with self.subTest(orientation=orientation):
                np.testing.assert_array_equal(
                    _apply_exif_orientation(image, orientation),
                    np.asarray(transformed, dtype=np.uint8),
                )

    def test_geometric_image_and_batch_reuse_one_detector(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = [root / "一.png", root / "two.png"]
            originals = [_source_image(path) for path in inputs]
            output_dir = root / "out"
            outputs = [output_dir / path.name for path in inputs]
            StaticDetector.instances = 0
            raw_console = io.BytesIO()
            console = io.TextIOWrapper(raw_console, encoding="cp1252")
            try:
                with (
                    patch("blurface.image_pipeline.FaceDetector", StaticDetector),
                    patch("blurface.image_pipeline.sys.stdout", console),
                ):
                    ImageProcessor(_batch(inputs, outputs)).run()
                console.flush()
                self.assertIn(b"file=\\u4e00.png", raw_console.getvalue())
            finally:
                console.detach()
            self.assertEqual(StaticDetector.instances, 1)
            for original, output in zip(originals, outputs):
                rendered = cv2.imdecode(np.fromfile(output, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertFalse(np.array_equal(rendered[15:38, 18:46], original[15:38, 18:46]))
            self.assertEqual(list(output_dir.glob(".blur-face-image-job-*")), [])

    def test_previous_rendered_image_is_released_before_next_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = [root / "first.png", root / "second.png"]
            for source in inputs:
                source.touch()
            outputs = [root / "out" / source.name for source in inputs]
            processor = ImageProcessor(_batch(inputs, outputs))
            references: list[weakref.ReferenceType[np.ndarray]] = []

            def process_one(_source, _detector, _segmenter):
                if references:
                    self.assertIsNone(references[-1]())
                rendered = np.zeros((32, 32, 3), dtype=np.uint8)
                references.append(weakref.ref(rendered))
                return rendered

            def discard_encoded_image(*_args, **_kwargs):
                return None

            with (
                patch("blurface.image_pipeline.FaceDetector", StaticDetector),
                patch.object(processor, "_process_one", side_effect=process_one),
                patch(
                    "blurface.image_pipeline._atomic_write_image",
                    new=discard_encoded_image,
                ),
            ):
                processor.run()

    def test_image_uses_resolution_scaled_shared_blur_kernel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            output = root / "output.png"
            _source_image(source)
            processor = ImageProcessor(_batch([source], [output]))
            decoded = np.zeros((216, 216, 3), dtype=np.uint8)
            with (
                patch("blurface.config.BLUR_REFERENCE_SHORT_EDGE", 108),
                patch("blurface.image_pipeline._decode_image", return_value=decoded),
                patch("blurface.image_pipeline.apply_blur") as blur,
            ):
                processor._process_one(source, StaticDetector(), None)
            self.assertEqual(blur.call_count, 1)
            self.assertEqual(blur.call_args.args[2], 43)

    def test_output_directory_symlink_retarget_cannot_overwrite_source(self):
        if not hasattr(os, "symlink"):
            self.skipTest("symbolic links are unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_output = root / "original-output"
            source_parent = root / "source-parent"
            original_output.mkdir()
            source_parent.mkdir()
            source = source_parent / "result.png"
            source_bytes = _source_image(source).tobytes()
            source_file_bytes = source.read_bytes()
            link = root / "selected-output"
            try:
                link.symlink_to(original_output, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"cannot create symbolic link: {exc}")
            output = link / source.name

            class RetargetingDetector(StaticDetector):
                def detect(self, image, conf=0.3):
                    del image, conf
                    link.unlink()
                    link.symlink_to(source_parent, target_is_directory=True)
                    return np.empty((0, 5), dtype=float)

            batch = _batch([source], [output])
            batch = replace(batch, options=replace(batch.options, overwrite=True))
            with patch("blurface.image_pipeline.FaceDetector", RetargetingDetector):
                ImageProcessor(batch).run()
            self.assertEqual(source.read_bytes(), source_file_bytes)
            self.assertTrue((original_output / source.name).is_file())
            self.assertEqual(source_bytes, _decode_image(source).tobytes())

    def test_existing_output_alias_of_source_is_rejected_by_file_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            _source_image(source)
            output_directory = root / "output"
            output_directory.mkdir()
            output = output_directory / "alias.png"
            try:
                os.link(source, output)
            except OSError as exc:
                self.skipTest(f"hard links are unavailable: {exc}")
            batch = _batch([source], [output])
            batch = replace(batch, options=replace(batch.options, overwrite=True))
            with self.assertRaisesRegex(ImageInputError, "replace an input"):
                ImageProcessor(batch)

    @unittest.skipIf(os.name == "nt", "POSIX directory-descriptor regression")
    def test_commit_is_anchored_across_post_check_directory_swap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_directory = root / "output"
            output_directory.mkdir()
            job = output_directory / ".blur-face-image-job-test"
            job.mkdir()
            victim = root / "victim"
            victim.mkdir()
            victim_output = victim / "face.png"
            victim_output.write_bytes(b"private source")
            moved = root / "approved-output-moved"
            output = output_directory / victim_output.name
            output_identity = _directory_identity(output_directory)
            job_identity = _directory_identity(job)
            output_anchor = _open_directory_anchor(output_directory, output_identity)
            job_anchor = _open_directory_anchor(job, job_identity)
            self.assertIsNotNone(output_anchor.dir_fd)
            self.assertIsNotNone(job_anchor.dir_fd)
            swapped = False

            def swap_after_check(_path, _identity):
                nonlocal swapped
                if not swapped:
                    output_directory.rename(moved)
                    output_directory.symlink_to(victim, target_is_directory=True)
                    swapped = True

            try:
                with (
                    patch("blurface.image_pipeline._encode_image", return_value=b"ANONYMIZED"),
                    patch(
                        "blurface.image_pipeline._validate_directory_identity",
                        side_effect=swap_after_check,
                    ),
                ):
                    _atomic_write_image(
                        np.zeros((2, 2, 3), dtype=np.uint8),
                        output,
                        job,
                        True,
                        output_identity,
                        output_anchor.dir_fd,
                        job_anchor.dir_fd,
                    )
            finally:
                job_anchor.close()
                output_anchor.close()
            self.assertEqual(victim_output.read_bytes(), b"private source")
            self.assertEqual((moved / victim_output.name).read_bytes(), b"ANONYMIZED")

    @unittest.skipIf(os.name == "nt", "POSIX directory-descriptor regression")
    def test_internal_job_creation_uses_anchored_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_directory = root / "output"
            output_directory.mkdir()
            moved = root / "approved-output-moved"
            victim = root / "victim"
            victim.mkdir()
            output_anchor = _open_directory_anchor(
                output_directory, _directory_identity(output_directory)
            )
            output_directory.rename(moved)
            output_directory.symlink_to(victim, target_is_directory=True)
            try:
                job_path, job_anchor = _create_owned_job_temp(
                    output_directory, output_anchor
                )
                self.assertTrue((moved / job_path.name).is_dir())
                self.assertEqual(list(victim.iterdir()), [])
                _cleanup_owned_job_temp(job_path, job_anchor, output_anchor)
                self.assertFalse((moved / job_path.name).exists())
            finally:
                output_anchor.close()

    @unittest.skipIf(os.name == "nt", "POSIX directory-descriptor regression")
    def test_internal_job_open_refuses_child_replaced_by_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_directory = root / "output"
            output_directory.mkdir()
            victim = root / "victim"
            victim.mkdir()
            private = victim / "private.txt"
            private.write_bytes(b"private")
            output_anchor = _open_directory_anchor(
                output_directory, _directory_identity(output_directory)
            )
            real_mkdir = os.mkdir

            def replace_child(path, mode=0o777, *, dir_fd=None):
                self.assertEqual(dir_fd, output_anchor.dir_fd)
                real_mkdir(path, mode, dir_fd=dir_fd)
                os.rename(
                    path,
                    f"{path}-original",
                    src_dir_fd=dir_fd,
                    dst_dir_fd=dir_fd,
                )
                os.symlink(
                    victim,
                    path,
                    target_is_directory=True,
                    dir_fd=dir_fd,
                )

            try:
                with patch(
                    "blurface.image_pipeline.os.mkdir",
                    side_effect=replace_child,
                ):
                    with self.assertRaisesRegex(
                        ImageInputError, "ordinary directory"
                    ):
                        _create_owned_job_temp(output_directory, output_anchor)
            finally:
                output_anchor.close()
            self.assertEqual(private.read_bytes(), b"private")

    @unittest.skipIf(os.name == "nt", "POSIX directory-descriptor regression")
    def test_directory_swap_cannot_redirect_internal_job_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            _source_image(source)
            output_directory = root / "output"
            output_directory.mkdir()
            output = output_directory / "result.png"
            moved = root / "approved-output-moved"
            victim = root / "victim"
            victim.mkdir()
            private: Path | None = None

            class SwapDirectoryDetector(StaticDetector):
                def detect(self, _image, conf=0.3):
                    nonlocal private
                    del conf
                    job = next(output_directory.glob(".blur-face-image-job-*"))
                    output_directory.rename(moved)
                    output_directory.symlink_to(victim, target_is_directory=True)
                    victim_job = victim / job.name
                    victim_job.mkdir()
                    private = victim_job / "unowned.txt"
                    private.write_bytes(b"private")
                    return np.empty((0, 5), dtype=float)

            batch = _batch([source], [output])
            batch = replace(batch, options=replace(batch.options, overwrite=True))
            with patch(
                "blurface.image_pipeline.FaceDetector", SwapDirectoryDetector
            ):
                with self.assertRaisesRegex(
                    ImageInputError, "ordinary directory"
                ):
                    ImageProcessor(batch).run()
            self.assertIsNotNone(private)
            self.assertEqual(private.read_bytes(), b"private")
            self.assertEqual(list(moved.glob(".blur-face-image-job-*")), [])

    def test_alpha_is_preserved_and_opaque_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            color = np.zeros((20, 24, 4), dtype=np.uint8)
            color[:, :, :3] = (20, 80, 220)
            color[:, :, 3] = np.arange(24, dtype=np.uint8)[None, :] * 10
            for suffix in (".png", ".webp", ".tif", ".tiff"):
                with self.subTest(suffix=suffix):
                    source = root / f"transparent{suffix}"
                    ok, encoded = cv2.imencode(suffix, color)
                    self.assertTrue(ok)
                    source.write_bytes(encoded.tobytes())
                    output = root / f"output{suffix}"
                    with patch("blurface.image_pipeline.FaceDetector", NoFaceDetector):
                        ImageProcessor(_batch([source], [output])).run()
                    decoded = cv2.imdecode(
                        np.fromfile(output, dtype=np.uint8), cv2.IMREAD_UNCHANGED
                    )
                    self.assertEqual(decoded.shape, color.shape)
                    np.testing.assert_array_equal(decoded[:, :, 3], color[:, :, 3])

            source = root / "transparent.png"
            ok, encoded = cv2.imencode(".png", color)
            self.assertTrue(ok)
            source.write_bytes(encoded.tobytes())
            opaque_output = root / "output.jpg"
            with patch("blurface.image_pipeline.FaceDetector", NoFaceDetector):
                with self.assertRaisesRegex(ImageInputError, "preserve transparency"):
                    ImageProcessor(_batch([source], [opaque_output])).run()
            self.assertFalse(opaque_output.exists())

    def test_supported_formats_round_trip_and_source_metadata_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suffix in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"):
                with self.subTest(suffix=suffix):
                    source = root / f"source{suffix}"
                    _source_image(source)
                    output = root / f"output{suffix}"
                    with patch("blurface.image_pipeline.FaceDetector", NoFaceDetector):
                        ImageProcessor(_batch([source], [output])).run()
                    self.assertIsNotNone(
                        cv2.imdecode(np.fromfile(output, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    )

            jpeg = root / "metadata.jpg"
            _source_image(jpeg)
            exif = b"Exif\x00\x00II\x2a\x00\x08\x00\x00\x00\x00\x00\x00\x00"
            payload = jpeg.read_bytes()
            jpeg.write_bytes(
                payload[:2] + b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif + payload[2:]
            )
            output = root / "metadata-output.jpg"
            with patch("blurface.image_pipeline.FaceDetector", NoFaceDetector):
                ImageProcessor(_batch([jpeg], [output])).run()
            self.assertNotIn(b"Exif\x00\x00", output.read_bytes())

    def test_mask_preview_is_black_and_blue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            _source_image(source)
            output = root / "preview.png"
            with patch("blurface.image_pipeline.FaceDetector", StaticDetector):
                ImageProcessor(_batch([source], [output], mask_preview=True)).run()
            rendered = cv2.imdecode(np.fromfile(output, dtype=np.uint8), cv2.IMREAD_COLOR)
            colors = np.unique(rendered.reshape(-1, 3), axis=0)
            self.assertTrue(any(np.array_equal(color, (255, 0, 0)) for color in colors))
            self.assertTrue(any(np.array_equal(color, (0, 0, 0)) for color in colors))

    def test_invalid_sam_contour_falls_back_to_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            original = _source_image(source)
            output = root / "sam.png"
            EmptySam.instances = 0
            with (
                patch("blurface.image_pipeline.FaceDetector", StaticDetector),
                patch("blurface.image_pipeline.Sam2Segmenter", EmptySam),
            ):
                ImageProcessor(_batch([source], [output], mask_engine="sam2.1")).run()
            rendered = cv2.imdecode(np.fromfile(output, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(EmptySam.instances, 1)
            self.assertFalse(np.array_equal(rendered[15:38, 18:46], original[15:38, 18:46]))

    def test_wrong_shape_nonfinite_sam_contour_falls_back_to_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            original = _source_image(source)
            output = root / "sam.png"
            with (
                patch("blurface.image_pipeline.FaceDetector", StaticDetector),
                patch("blurface.image_pipeline.Sam2Segmenter", InvalidArraySam),
            ):
                ImageProcessor(_batch([source], [output], mask_engine="sam2.1")).run()
            rendered = cv2.imdecode(
                np.fromfile(output, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            self.assertFalse(
                np.array_equal(rendered[15:38, 18:46], original[15:38, 18:46])
            )

    def test_sam_image_modes_reach_renderer_with_valid_contour(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            original = _source_image(source)
            for combine in ("union", "intersection", "mask-only"):
                output = root / f"{combine}.png"
                config = _batch([source], [output], mask_engine="sam2.1")
                config = replace(
                    config,
                    options=replace(config.options, segmentation_combine=combine),
                )
                with (
                    patch("blurface.image_pipeline.FaceDetector", StaticDetector),
                    patch("blurface.image_pipeline.Sam2Segmenter", ValidSam),
                ):
                    ImageProcessor(config).run()
                rendered = cv2.imdecode(
                    np.fromfile(output, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                self.assertFalse(
                    np.array_equal(rendered[20, 20], original[20, 20])
                )
                outside_changed = not np.array_equal(
                    rendered[20, 9], original[20, 9]
                )
                self.assertEqual(outside_changed, combine != "intersection")

    def test_no_overwrite_commit_rejects_destination_created_midflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = root / ".blur-face-image-job-test"
            job.mkdir()
            output = root / "output.png"

            def appeared(_image, _suffix):
                output.write_bytes(b"other process")
                return b"complete encoded image"

            with patch("blurface.image_pipeline._encode_image", side_effect=appeared):
                with self.assertRaisesRegex(ImageInputError, "appeared"):
                    _atomic_write_image(np.zeros((2, 2, 3), dtype=np.uint8), output, job, False)
            self.assertEqual(output.read_bytes(), b"other process")
            self.assertEqual(list(job.iterdir()), [])

    def test_wrong_filesystem_no_overwrite_failure_never_replaces(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = root / ".blur-face-image-job-test"
            job.mkdir()
            output = root / "output.png"
            with (
                patch("blurface.image_pipeline._encode_image", return_value=b"complete"),
                patch("blurface.image_pipeline.os.link", side_effect=OSError("EXDEV")),
            ):
                with self.assertRaisesRegex(ImageInputError, "left untouched"):
                    _atomic_write_image(np.zeros((2, 2, 3), dtype=np.uint8), output, job, False)
            self.assertFalse(os.path.lexists(output))
            self.assertEqual(list(job.iterdir()), [])

    def test_hard_link_commit_remains_successful_if_temp_unlink_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = root / ".blur-face-image-job-test"
            job.mkdir()
            output = root / "output.png"
            with (
                patch("blurface.image_pipeline._encode_image", return_value=b"complete"),
                patch("blurface.image_pipeline.Path.unlink", side_effect=OSError("busy")),
            ):
                _atomic_write_image(
                    np.zeros((2, 2, 3), dtype=np.uint8), output, job, False
                )
            self.assertEqual(output.read_bytes(), b"complete")

    def test_batch_failure_keeps_completed_file_and_not_current_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = [root / "first.png", root / "second.png"]
            for source in inputs:
                _source_image(source)
            output_dir = root / "out"
            outputs = [output_dir / source.name for source in inputs]
            encode_calls = 0

            def encode_or_fail(image, suffix):
                nonlocal encode_calls
                encode_calls += 1
                if encode_calls == 2:
                    raise ImageInputError("synthetic encoding failure")
                ok, encoded = cv2.imencode(suffix, image)
                self.assertTrue(ok)
                return encoded.tobytes()

            with (
                patch("blurface.image_pipeline.FaceDetector", StaticDetector),
                patch(
                    "blurface.image_pipeline._encode_image",
                    side_effect=encode_or_fail,
                ),
            ):
                logs = io.StringIO()
                with redirect_stdout(logs):
                    with self.assertRaisesRegex(ImageInputError, "synthetic"):
                        ImageProcessor(_batch(inputs, outputs)).run()
            self.assertTrue(outputs[0].is_file())
            self.assertFalse(outputs[1].exists())
            self.assertEqual(list(output_dir.glob(".blur-face-image-job-*")), [])
            self.assertIn("file=second.png", logs.getvalue())


if __name__ == "__main__":
    unittest.main()
