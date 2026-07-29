"""Disk-backed, scene-aware temporal mask stabilization.

The module deliberately has no model dependencies.  It operates on small
grayscale motion proxies and per-track segmentation masks while the original
video is reopened only for final rendering.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

MOTION_MAX_SIDE = 480
DEFAULT_TEMP_LIMIT_BYTES = 4 * 1024**3
SAM_REVERSE_CACHE_LIMIT_BYTES = 512 * 1024**2


class TemporalStoreError(RuntimeError):
    """Raised when bounded temporary analysis data cannot be persisted."""


@dataclass(frozen=True, slots=True)
class TemporalTrackRecord:
    frame_index: int
    scene_id: int
    track_id: int
    box: list[int]
    coverage_boxes: tuple[list[int], ...]
    confidence: float
    threshold: float
    observed: bool
    ambiguous: bool
    fallback: bool
    mask_path: str | None


def motion_proxy(frame: np.ndarray) -> np.ndarray:
    """Return a bounded color frame for motion and appearance validation."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("motion proxy source must be a BGR frame")
    height, width = frame.shape[:2]
    scale = min(1.0, MOTION_MAX_SIDE / max(height, width))
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return np.ascontiguousarray(frame)


class SceneCutDetector:
    """Conservative cut detector combining histogram and pixel evidence."""

    def __init__(self, sensitivity: float = 0.55) -> None:
        if not 0.0 <= sensitivity <= 1.0:
            raise ValueError("scene-cut sensitivity must be between 0 and 1")
        self.sensitivity = sensitivity
        self._previous: tuple[np.ndarray, np.ndarray | None] | None = None

    def update(self, frame: np.ndarray) -> bool:
        current = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        if current.ndim == 3:
            current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            current_chroma = cv2.cvtColor(current, cv2.COLOR_BGR2LAB)[..., 1:]
        else:
            current_gray = current
            current_chroma = None
        previous = self._previous
        self._previous = (current_gray, current_chroma)
        if previous is None:
            return False
        previous_gray, previous_chroma = previous
        hist_a = cv2.calcHist([previous_gray], [0], None, [32], [0, 256])
        hist_b = cv2.calcHist([current_gray], [0], None, [32], [0, 256])
        cv2.normalize(hist_a, hist_a, alpha=1.0, norm_type=cv2.NORM_L1)
        cv2.normalize(hist_b, hist_b, alpha=1.0, norm_type=cv2.NORM_L1)
        histogram_distance = cv2.compareHist(
            hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA
        )
        changed = np.mean(cv2.absdiff(previous_gray, current_gray) > 35)
        previous_std = float(previous_gray.std())
        current_std = float(current_gray.std())
        correlation = (
            float(
                np.corrcoef(
                    previous_gray.ravel(), current_gray.ravel()
                )[0, 1]
            )
            if previous_std > 1e-6 and current_std > 1e-6
            else 0.0
        )
        chroma_distance = (
            float(
                np.mean(
                    np.linalg.norm(
                        current_chroma.astype(np.float32)
                        - previous_chroma.astype(np.float32),
                        axis=2,
                    )
                )
            )
            if current_chroma is not None and previous_chroma is not None
            else 0.0
        )
        # Higher user sensitivity detects smaller changes. Requiring both
        # histogram and pixel signals avoids treating ordinary camera motion as
        # a cut. The structural branch catches cuts whose shots happen to have
        # similar global histograms.
        threshold = 0.72 - 0.34 * self.sensitivity
        return bool(
            (
                histogram_distance >= threshold
                and changed >= max(0.28, threshold * 0.72)
            )
            or (
                changed >= 0.82 - 0.25 * self.sensitivity
                and correlation < 0.15
            )
            or chroma_distance >= 34.0 - 18.0 * self.sensitivity
        )

    def reset(self) -> None:
        self._previous = None


class TemporalStore:
    """SQLite metadata plus bounded compressed mask/motion files."""

    def __init__(
        self,
        parent: Path | None = None,
        limit_bytes: int = DEFAULT_TEMP_LIMIT_BYTES,
    ) -> None:
        if limit_bytes <= 0:
            raise ValueError("temporary storage limit must be positive")
        self._temporary = tempfile.TemporaryDirectory(
            prefix="blur-face-temporal-",
            dir=str(parent) if parent is not None else None,
        )
        self.root = Path(self._temporary.name)
        self.limit_bytes = int(limit_bytes)
        self._image_current_bytes = 0
        self._peak_bytes = 0
        self.image_bytes_written = 0
        self._stable_mask_revision = 0
        self._composite_mask_revision = 0
        self.mask_read_failures = {
            "missing_or_unreadable": 0,
            "decode_failed": 0,
            "empty": 0,
        }
        self._connection = sqlite3.connect(self.root / "analysis.sqlite3")
        self._connection.execute(
            """
            CREATE TABLE frames (
                frame_index INTEGER PRIMARY KEY,
                scene_id INTEGER NOT NULL,
                gray_path TEXT NOT NULL,
                source_height INTEGER NOT NULL,
                source_width INTEGER NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE tracks (
                frame_index INTEGER NOT NULL,
                scene_id INTEGER NOT NULL,
                track_id INTEGER NOT NULL,
                box TEXT NOT NULL,
                coverage_boxes TEXT NOT NULL,
                confidence REAL NOT NULL,
                threshold REAL NOT NULL,
                observed INTEGER NOT NULL,
                ambiguous INTEGER NOT NULL,
                fallback INTEGER NOT NULL,
                mask_path TEXT,
                PRIMARY KEY (frame_index, scene_id, track_id)
            )
            """
        )
        self._connection.execute(
            "CREATE INDEX tracks_scene_track ON tracks(scene_id, track_id, frame_index)"
        )
        self._connection.execute(
            """
            CREATE TABLE composite_masks (
                frame_index INTEGER PRIMARY KEY,
                scene_id INTEGER NOT NULL,
                mask_path TEXT NOT NULL
            )
            """
        )
        self._connection.commit()
        self._check_capacity()
        self.closed = False

    def _database_bytes(self) -> int:
        return sum(
            path.stat().st_size
            for path in self.root.glob("analysis.sqlite3*")
            if path.is_file()
        )

    @property
    def current_bytes(self) -> int:
        """Current bytes for images plus SQLite and journal/WAL files."""
        return self._image_current_bytes + self._database_bytes()

    @property
    def peak_bytes(self) -> int:
        self._check_capacity()
        return self._peak_bytes

    def _check_capacity(self) -> None:
        current = self.current_bytes
        self._peak_bytes = max(self._peak_bytes, current)
        if current > self.limit_bytes:
            raise TemporalStoreError(
                "temporal analysis exceeded the configured temporary-storage "
                f"limit ({self.limit_bytes // 1024**2} MiB)"
            )

    def _account(self, path: Path) -> None:
        size = path.stat().st_size
        self._image_current_bytes += size
        self.image_bytes_written += size
        self._check_capacity()

    def _write_png(self, relative: str, image: np.ndarray) -> str:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            encoded_ok, encoded = cv2.imencode(
                ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9]
            )
        except cv2.error as exc:
            raise TemporalStoreError(
                f"unable to encode temporary mask: {path.name}"
            ) from exc
        if not encoded_ok or encoded is None or not encoded.size:
            raise TemporalStoreError(f"unable to write temporary image: {path.name}")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if (
            decoded is None
            or decoded.shape != image.shape
            or not np.array_equal(decoded, image)
        ):
            raise TemporalStoreError(
                f"temporary mask failed PNG round-trip validation: {path.name}"
            )
        try:
            path.write_bytes(encoded.tobytes())
        except OSError as exc:
            raise TemporalStoreError(
                f"unable to write temporary image: {path.name}"
            ) from exc
        self._account(path)
        return relative

    def _write_jpeg(self, relative: str, image: np.ndarray) -> str:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(
            str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 85]
        ):
            raise TemporalStoreError(
                f"unable to write temporary image: {path.name}"
            )
        self._account(path)
        return relative

    def add_frame(
        self,
        frame_index: int,
        scene_id: int,
        gray: np.ndarray,
        source_shape: tuple[int, int],
    ) -> None:
        relative = self._write_jpeg(
            f"motion/{frame_index:09d}.jpg",
            np.ascontiguousarray(gray, dtype=np.uint8),
        )
        self._connection.execute(
            "INSERT INTO frames VALUES (?, ?, ?, ?, ?)",
            (
                frame_index,
                scene_id,
                relative,
                int(source_shape[0]),
                int(source_shape[1]),
            ),
        )
        self._check_capacity()

    def add_track(
        self,
        frame_index: int,
        scene_id: int,
        track_id: int,
        box: list[int],
        coverage_boxes: tuple[list[int], ...],
        confidence: float,
        threshold: float,
        observed: bool,
        ambiguous: bool,
        fallback: bool,
        mask: np.ndarray | None,
        *,
        replace: bool = False,
    ) -> None:
        if replace:
            previous = self._connection.execute(
                "SELECT mask_path FROM tracks WHERE frame_index = ? "
                "AND scene_id = ? AND track_id = ?",
                (int(frame_index), int(scene_id), int(track_id)),
            ).fetchone()
            if previous is not None and previous[0]:
                previous_path = self.root / previous[0]
                try:
                    previous_size = previous_path.stat().st_size
                    previous_path.unlink()
                    self._image_current_bytes = max(
                        0, self._image_current_bytes - previous_size
                    )
                except FileNotFoundError:
                    pass
        mask_path = None
        if mask is not None and np.any(np.asarray(mask) > 0):
            mask_path = self._write_png(
                f"masks/{scene_id:06d}/{track_id:06d}/{frame_index:09d}.png",
                np.where(mask > 0, 255, 0).astype(np.uint8),
            )
        effective_fallback = bool(fallback or mask_path is None)
        operation = "INSERT OR REPLACE" if replace else "INSERT"
        self._connection.execute(
            f"{operation} INTO tracks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                frame_index,
                scene_id,
                track_id,
                json.dumps([int(value) for value in box]),
                json.dumps(
                    [[int(value) for value in item] for item in coverage_boxes]
                ),
                float(confidence),
                float(threshold),
                int(observed),
                int(ambiguous),
                int(effective_fallback),
                mask_path,
            ),
        )
        self._check_capacity()

    def commit(self) -> None:
        self._connection.commit()
        self._check_capacity()

    def frame(self, frame_index: int) -> tuple[int, np.ndarray, tuple[int, int]]:
        row = self._connection.execute(
            "SELECT scene_id, gray_path, source_height, source_width "
            "FROM frames WHERE frame_index = ?",
            (frame_index,),
        ).fetchone()
        if row is None:
            raise TemporalStoreError(f"missing analysis frame {frame_index}")
        proxy = cv2.imread(str(self.root / row[1]), cv2.IMREAD_UNCHANGED)
        if proxy is None:
            raise TemporalStoreError(f"unable to read analysis frame {frame_index}")
        return int(row[0]), proxy, (int(row[2]), int(row[3]))

    @staticmethod
    def _record(row) -> TemporalTrackRecord:
        return TemporalTrackRecord(
            frame_index=int(row[0]),
            scene_id=int(row[1]),
            track_id=int(row[2]),
            box=json.loads(row[3]),
            coverage_boxes=tuple(json.loads(row[4])),
            confidence=float(row[5]),
            threshold=float(row[6]),
            observed=bool(row[7]),
            ambiguous=bool(row[8]),
            fallback=bool(row[9]),
            mask_path=row[10],
        )

    def records_for_frame(self, frame_index: int) -> list[TemporalTrackRecord]:
        rows = self._connection.execute(
            "SELECT * FROM tracks WHERE frame_index = ? ORDER BY track_id",
            (frame_index,),
        )
        return [self._record(row) for row in rows]

    def has_track(self, scene_id: int, track_id: int) -> bool:
        return (
            self._connection.execute(
                "SELECT 1 FROM tracks WHERE scene_id = ? AND track_id = ? LIMIT 1",
                (scene_id, track_id),
            ).fetchone()
            is not None
        )

    def records_for_track(
        self, scene_id: int, track_id: int
    ):
        """Yield track history in bounded pages, safe while rows are replaced."""
        last_frame = -1
        while True:
            rows = self._connection.execute(
                "SELECT * FROM tracks WHERE scene_id = ? AND track_id = ? "
                "AND frame_index > ? ORDER BY frame_index LIMIT 256",
                (scene_id, track_id, last_frame),
            ).fetchall()
            if not rows:
                return
            for row in rows:
                record = self._record(row)
                last_frame = record.frame_index
                yield record

    def records_for_track_reverse(
        self, scene_id: int, track_id: int
    ):
        """Yield track history newest-first without loading a long video."""
        last_frame = self.max_frame_index() + 1
        while True:
            rows = self._connection.execute(
                "SELECT * FROM tracks WHERE scene_id = ? AND track_id = ? "
                "AND frame_index < ? ORDER BY frame_index DESC LIMIT 256",
                (scene_id, track_id, last_frame),
            ).fetchall()
            if not rows:
                return
            for row in rows:
                record = self._record(row)
                last_frame = record.frame_index
                yield record

    def track_keys(self):
        """Yield distinct identity keys without materializing the whole video."""
        last_scene = -1
        last_track = -1
        while True:
            rows = self._connection.execute(
                "SELECT DISTINCT scene_id, track_id FROM tracks "
                "WHERE scene_id > ? OR (scene_id = ? AND track_id > ?) "
                "ORDER BY scene_id, track_id LIMIT 256",
                (last_scene, last_scene, last_track),
            ).fetchall()
            if not rows:
                return
            for row in rows:
                if len(row) < 2:
                    raise TemporalStoreError(
                        "invalid temporal identity row"
                    )
                last_scene, last_track = int(row[0]), int(row[1])
                yield last_scene, last_track

    def track_key_count(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM "
            "(SELECT DISTINCT scene_id, track_id FROM tracks)"
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def mask_stats(self) -> tuple[int, int, int]:
        """Return total, segmented, and geometric-fallback record counts."""
        row = self._connection.execute(
            "SELECT COUNT(*), "
            "SUM(CASE WHEN fallback = 0 AND mask_path IS NOT NULL "
            "THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN fallback != 0 OR mask_path IS NULL "
            "THEN 1 ELSE 0 END) "
            "FROM tracks"
        ).fetchone()
        if row is None:
            return 0, 0, 0
        return tuple(int(value or 0) for value in row)

    def checkpoint_masks(self) -> None:
        """Preserve immutable analyzed-mask references for safe recovery."""
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS analyzed_mask_checkpoint ("
            "frame_index INTEGER NOT NULL, scene_id INTEGER NOT NULL, "
            "track_id INTEGER NOT NULL, fallback INTEGER NOT NULL, "
            "mask_path TEXT, "
            "PRIMARY KEY (frame_index, scene_id, track_id))"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS analyzed_checkpoint_scene_track "
            "ON analyzed_mask_checkpoint(scene_id, track_id, frame_index)"
        )
        self._connection.execute("DELETE FROM analyzed_mask_checkpoint")
        self._connection.execute(
            "INSERT INTO analyzed_mask_checkpoint "
            "SELECT frame_index, scene_id, track_id, fallback, mask_path "
            "FROM tracks"
        )
        self._check_capacity()

    def restore_missing_analyzed_masks(
        self, scene_id: int, track_id: int
    ) -> int:
        """Recover accepted raw masks erased by a track-level temporal error."""
        cursor = self._connection.execute(
            "UPDATE tracks SET "
            "fallback = (SELECT checkpoint.fallback "
            "FROM analyzed_mask_checkpoint AS checkpoint "
            "WHERE checkpoint.frame_index = tracks.frame_index "
            "AND checkpoint.scene_id = tracks.scene_id "
            "AND checkpoint.track_id = tracks.track_id), "
            "mask_path = (SELECT checkpoint.mask_path "
            "FROM analyzed_mask_checkpoint AS checkpoint "
            "WHERE checkpoint.frame_index = tracks.frame_index "
            "AND checkpoint.scene_id = tracks.scene_id "
            "AND checkpoint.track_id = tracks.track_id) "
            "WHERE scene_id = ? AND track_id = ? "
            "AND (fallback != 0 OR mask_path IS NULL) "
            "AND EXISTS (SELECT 1 "
            "FROM analyzed_mask_checkpoint AS checkpoint "
            "WHERE checkpoint.frame_index = tracks.frame_index "
            "AND checkpoint.scene_id = tracks.scene_id "
            "AND checkpoint.track_id = tracks.track_id "
            "AND checkpoint.fallback = 0 "
            "AND checkpoint.mask_path IS NOT NULL)",
            (scene_id, track_id),
        )
        self._check_capacity()
        return max(0, int(cursor.rowcount))

    def load_mask(self, record: TemporalTrackRecord) -> np.ndarray | None:
        if not record.mask_path:
            return None
        path = self.root / record.mask_path
        try:
            encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        except OSError:
            self.mask_read_failures["missing_or_unreadable"] += 1
            return None
        try:
            mask = (
                cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
                if encoded.size
                else None
            )
        except cv2.error:
            mask = None
        if mask is None or mask.ndim != 2:
            self.mask_read_failures["decode_failed"] += 1
            return None
        if not np.any(mask):
            self.mask_read_failures["empty"] += 1
            return None
        return mask

    def replace_mask(
        self,
        record: TemporalTrackRecord,
        mask: np.ndarray | None,
        *,
        fallback: bool,
    ) -> None:
        # Never overwrite a mask file that was just read. Besides preserving
        # the raw analysis artifact for diagnostics, this avoids a Windows
        # OpenCV failure mode where rewriting the same PNG can leave a valid
        # database row pointing at an empty/corrupt image.
        previous_path = (
            self.root / record.mask_path if record.mask_path is not None else None
        )
        mask_path = None
        if mask is not None and np.any(np.asarray(mask) > 0):
            self._stable_mask_revision += 1
            mask_path = self._write_png(
                "stable/"
                f"{record.scene_id:06d}/{record.track_id:06d}/"
                f"{record.frame_index:09d}-{self._stable_mask_revision:09d}.png",
                np.where(mask > 0, 255, 0).astype(np.uint8),
            )
        self._connection.execute(
            "UPDATE tracks SET fallback = ?, mask_path = ? "
            "WHERE frame_index = ? AND scene_id = ? AND track_id = ?",
            (
                int(fallback),
                mask_path,
                record.frame_index,
                record.scene_id,
                record.track_id,
            ),
        )
        if (
            previous_path is not None
            and record.mask_path is not None
            and Path(record.mask_path).parts[:1] == ("stable",)
            and previous_path != self.root / (mask_path or "")
        ):
            try:
                previous_size = previous_path.stat().st_size
                previous_path.unlink()
                self._image_current_bytes = max(
                    0, self._image_current_bytes - previous_size
                )
            except FileNotFoundError:
                pass
        self._check_capacity()

    def replace_composite_mask(
        self,
        frame_index: int,
        scene_id: int,
        mask: np.ndarray,
    ) -> None:
        """Persist one final, cross-track frame mask without retaining it in RAM."""
        previous = self._connection.execute(
            "SELECT mask_path FROM composite_masks WHERE frame_index = ?",
            (int(frame_index),),
        ).fetchone()
        if previous is not None:
            previous_path = self.root / previous[0]
            try:
                previous_size = previous_path.stat().st_size
                previous_path.unlink()
                self._image_current_bytes = max(
                    0, self._image_current_bytes - previous_size
                )
            except FileNotFoundError:
                pass
        self._composite_mask_revision += 1
        relative = self._write_png(
            "composite/"
            f"{scene_id:06d}/{frame_index:09d}-"
            f"{self._composite_mask_revision:09d}.png",
            np.where(mask > 0, 255, 0).astype(np.uint8),
        )
        self._connection.execute(
            "INSERT OR REPLACE INTO composite_masks VALUES (?, ?, ?)",
            (int(frame_index), int(scene_id), relative),
        )
        self._check_capacity()

    def composite_rows(self, *, reverse: bool = False):
        order = "DESC" if reverse else "ASC"
        rows = self._connection.execute(
            "SELECT frame_index, scene_id, mask_path "
            f"FROM composite_masks ORDER BY frame_index {order}"
        )
        for frame_index, scene_id, mask_path in rows:
            yield int(frame_index), int(scene_id), str(mask_path)

    def load_composite_mask(self, frame_index: int) -> np.ndarray | None:
        row = self._connection.execute(
            "SELECT mask_path FROM composite_masks WHERE frame_index = ?",
            (int(frame_index),),
        ).fetchone()
        if row is None:
            return None
        try:
            encoded = np.frombuffer(
                (self.root / row[0]).read_bytes(), dtype=np.uint8
            )
            mask = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        except (OSError, cv2.error):
            return None
        if mask is None or mask.ndim != 2:
            return None
        return np.where(mask > 0, 255, 0).astype(np.uint8)

    def max_frame_index(self) -> int:
        row = self._connection.execute("SELECT MAX(frame_index) FROM frames").fetchone()
        return int(row[0]) if row and row[0] is not None else -1

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._connection.close()
        self._temporary.cleanup()

    def __enter__(self) -> TemporalStore:  # noqa: PYI034
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def box_iou(first, second) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def box_overlap_over_smaller(first, second) -> float:
    """Return intersection divided by the smaller box's area."""
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(
        0, first[3] - first[1]
    )
    second_area = max(0, second[2] - second[0]) * max(
        0, second[3] - second[1]
    )
    smaller = min(first_area, second_area)
    return intersection / smaller if smaller else 0.0


def ambiguous_track_ids(records) -> set[int]:
    """Return IDs whose current boxes overlap enough to make history unsafe."""
    ambiguous: set[int] = set()
    for index, first in enumerate(records):
        for second in records[index + 1 :]:
            first_center = np.array(
                (
                    (first.box[0] + first.box[2]) / 2,
                    (first.box[1] + first.box[3]) / 2,
                )
            )
            second_center = np.array(
                (
                    (second.box[0] + second.box[2]) / 2,
                    (second.box[1] + second.box[3]) / 2,
                )
            )
            first_diagonal = np.hypot(
                first.box[2] - first.box[0],
                first.box[3] - first.box[1],
            )
            second_diagonal = np.hypot(
                second.box[2] - second.box[0],
                second.box[3] - second.box[1],
            )
            # Use the smaller face as the scale reference. The previous
            # larger-face 1.25x radius quarantined ordinary side-by-side faces
            # for an entire clip. A sub-face-width gap is still treated as an
            # imminent crossing, while actual overlap remains the primary
            # identity-risk signal.
            proximity = min(first_diagonal, second_diagonal) * 0.75
            if (
                box_iou(first.box, second.box) >= 0.05
                or box_overlap_over_smaller(
                    first.box, second.box
                )
                >= 0.10
                or np.linalg.norm(first_center - second_center)
                <= proximity
            ):
                ambiguous.update((first.track_id, second.track_id))
    return ambiguous


def _warp_mask(
    mask: np.ndarray,
    source_gray: np.ndarray,
    target_gray: np.ndarray,
) -> np.ndarray | None:
    """Motion-align a source mask into target coordinates, or reject it."""
    if (
        not isinstance(mask, np.ndarray)
        or mask.ndim != 2
        or not isinstance(source_gray, np.ndarray)
        or not isinstance(target_gray, np.ndarray)
        or source_gray.ndim not in {2, 3}
        or target_gray.ndim not in {2, 3}
        or source_gray.shape != target_gray.shape
        or mask.shape != source_gray.shape[:2]
    ):
        return None
    source_gray = (
        cv2.cvtColor(source_gray, cv2.COLOR_BGR2GRAY)
        if source_gray.ndim == 3
        else source_gray
    )
    target_gray = (
        cv2.cvtColor(target_gray, cv2.COLOR_BGR2GRAY)
        if target_gray.ndim == 3
        else target_gray
    )
    try:
        # Backward flow gives each target pixel its source sample coordinate.
        flow = cv2.calcOpticalFlowFarneback(
            target_gray,
            source_gray,
            None,
            0.5,
            3,
            21,
            3,
            5,
            1.2,
            0,
        )
        forward_flow = cv2.calcOpticalFlowFarneback(
            source_gray,
            target_gray,
            None,
            0.5,
            3,
            21,
            3,
            5,
            1.2,
            0,
        )
    except cv2.error:
        return None
    if (
        flow is None
        or forward_flow is None
        or not np.isfinite(flow).all()
        or not np.isfinite(forward_flow).all()
    ):
        return None
    height, width = mask.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    magnitude = np.linalg.norm(flow, axis=2)
    active = mask > 0
    if active.any() and np.median(magnitude[active]) > 0.28 * np.hypot(width, height):
        return None
    warped = cv2.remap(
        mask,
        xx + flow[..., 0],
        yy + flow[..., 1],
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    active_target = warped > 0
    if active_target.any():
        sampled_forward_x = cv2.remap(
            forward_flow[..., 0],
            xx + flow[..., 0],
            yy + flow[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,
        )
        sampled_forward_y = cv2.remap(
            forward_flow[..., 1],
            xx + flow[..., 0],
            yy + flow[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,
        )
        cycle_error = np.hypot(
            flow[..., 0] + sampled_forward_x,
            flow[..., 1] + sampled_forward_y,
        )
        valid_error = cycle_error[active_target]
        valid_error = valid_error[np.isfinite(valid_error)]
        if not len(valid_error) or np.median(valid_error) > 4.5:
            return None
    return np.where(warped > 0, 255, 0).astype(np.uint8)


def _warp_age(
    age: np.ndarray,
    source_gray: np.ndarray,
    target_gray: np.ndarray,
) -> np.ndarray | None:
    """Align a small hysteresis-age map with the same dense motion model."""
    if (
        not isinstance(age, np.ndarray)
        or age.ndim != 2
        or not isinstance(source_gray, np.ndarray)
        or not isinstance(target_gray, np.ndarray)
        or source_gray.ndim not in {2, 3}
        or target_gray.ndim not in {2, 3}
        or source_gray.shape != target_gray.shape
        or age.shape != source_gray.shape[:2]
    ):
        return None
    source_gray = (
        cv2.cvtColor(source_gray, cv2.COLOR_BGR2GRAY)
        if source_gray.ndim == 3
        else source_gray
    )
    target_gray = (
        cv2.cvtColor(target_gray, cv2.COLOR_BGR2GRAY)
        if target_gray.ndim == 3
        else target_gray
    )
    try:
        flow = cv2.calcOpticalFlowFarneback(
            target_gray,
            source_gray,
            None,
            0.5,
            3,
            21,
            3,
            5,
            1.2,
            0,
        )
    except cv2.error:
        return None
    if flow is None or not np.isfinite(flow).all():
        return None
    height, width = age.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    return cv2.remap(
        age,
        xx + flow[..., 0],
        yy + flow[..., 1],
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def aligned_iou(first: np.ndarray, second: np.ndarray) -> float:
    a, b = first > 0, second > 0
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / union if union else 1.0


def _appearance(proxy: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    pixels = proxy[mask > 0]
    if not len(pixels):
        return None
    values = np.median(pixels, axis=0)
    return np.atleast_1d(values).astype(np.float32)


def _coverage_mask(
    record: TemporalTrackRecord,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    scale: float = 1.0,
) -> np.ndarray:
    """Rasterize current fail-closed geometry at temporal-mask resolution."""
    source_h, source_w = source_shape
    target_h, target_w = target_shape
    result = np.zeros((target_h, target_w), dtype=np.uint8)
    for box in record.coverage_boxes:
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        half_width = max(0, box[2] - box[0]) * scale / 2
        half_height = max(0, box[3] - box[1]) * scale / 2
        x1 = max(
            0,
            min(
                target_w,
                int(np.floor((center_x - half_width) * target_w / source_w)),
            ),
        )
        y1 = max(
            0,
            min(
                target_h,
                int(np.floor((center_y - half_height) * target_h / source_h)),
            ),
        )
        x2 = max(
            0,
            min(
                target_w,
                int(np.ceil((center_x + half_width) * target_w / source_w)),
            ),
        )
        y2 = max(
            0,
            min(
                target_h,
                int(np.ceil((center_y + half_height) * target_h / source_h)),
            ),
        )
        if x2 > x1 and y2 > y1:
            result[y1:y2, x1:x2] = 255
    return result


def _seed_release_age(
    envelope: np.ndarray,
    raw: np.ndarray,
    release_hold_frames: int,
) -> np.ndarray:
    """Age far geometry sooner so fallback coverage contracts progressively."""
    age = np.zeros_like(raw, dtype=np.uint8)
    extra = (envelope > 0) & (raw == 0)
    if not extra.any() or release_hold_frames <= 0:
        return age
    distance = cv2.distanceTransform(
        np.where(raw == 0, 255, 0).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    farthest = float(distance[extra].max())
    if farthest > 0:
        age[extra] = np.floor(
            distance[extra] / farthest * release_hold_frames
        ).astype(np.uint8)
    return age


def _blend_toward_geometry(
    anchor: np.ndarray,
    expanded: np.ndarray,
    step: int,
    total_steps: int,
) -> np.ndarray:
    """Blend a validated propagated contour toward geometric coverage."""
    if total_steps <= 0 or step >= total_steps:
        return expanded
    margin = (expanded > 0) & (anchor == 0)
    if not margin.any():
        return anchor
    distance = cv2.distanceTransform(
        np.where(anchor == 0, 255, 0).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    farthest = float(distance[margin].max())
    if farthest <= 0:
        return anchor
    admitted = margin & (distance <= farthest * step / total_steps)
    return np.where((anchor > 0) | admitted, 255, 0).astype(np.uint8)


def _mask_matches_geometry(mask: np.ndarray, geometry: np.ndarray) -> bool:
    """Return whether a propagated mask still belongs to current geometry."""
    pixels = np.count_nonzero(mask)
    geometry_pixels = np.count_nonzero(geometry)
    if not pixels or not geometry_pixels:
        return False
    rows, columns = np.nonzero(mask)
    center_inside = bool(
        geometry[
            int(np.median(rows)),
            int(np.median(columns)),
        ]
        > 0
    )
    inside = np.count_nonzero((mask > 0) & (geometry > 0)) / pixels
    return bool(
        center_inside
        and inside >= 0.35
        and pixels <= geometry_pixels * 2.0
    )


def _filter_aligned_history(
    aligned: np.ndarray,
    raw: np.ndarray,
    current_geometry: np.ndarray,
) -> np.ndarray | None:
    """Drop disconnected history that cannot belong to the current face."""
    count, labels = cv2.connectedComponents(
        np.where(aligned > 0, 1, 0).astype(np.uint8),
        connectivity=8,
    )
    kept = np.zeros_like(aligned, dtype=np.uint8)
    for label in range(1, count):
        component = labels == label
        component_pixels = np.count_nonzero(component)
        if not component_pixels:
            continue
        raw_overlap = np.count_nonzero(component & (raw > 0))
        geometry_overlap = np.count_nonzero(
            component & (current_geometry > 0)
        )
        if (
            raw_overlap / component_pixels >= 0.05
            or geometry_overlap / component_pixels >= 0.35
        ):
            # A one-pixel bridge can connect a valid current lobe to a remote
            # stale face and defeat component-level filtering. Historical
            # pixels are therefore also clipped to the current expanded Track
            # domain. The current raw model mask is never clipped here.
            kept[component & (current_geometry > 0)] = 255
    return kept if np.any(kept) else None


class TemporalMaskStabilizer:
    """Backfill and stabilize masks without ever reassigning object identity."""

    def __init__(
        self,
        backfill_frames: int = 10,
        release_hold_frames: int = 5,
        geometry_scale: float = 1.0,
    ) -> None:
        if not 0 <= backfill_frames <= 60:
            raise ValueError("backfill frames must be between 0 and 60")
        if not 0 <= release_hold_frames <= 12:
            raise ValueError("release hold frames must be between 0 and 12")
        if not np.isfinite(geometry_scale) or geometry_scale < 1.0:
            raise ValueError("geometry scale must be at least 1")
        self.backfill_frames = backfill_frames
        self.release_hold_frames = release_hold_frames
        self.geometry_scale = float(geometry_scale)

    @staticmethod
    def _touches_edge(box, shape: tuple[int, int]) -> bool:
        height, width = shape
        margin = max(2, round(min(height, width) * 0.025))
        return (
            box[0] <= margin
            or box[1] <= margin
            or box[2] >= width - margin
            or box[3] >= height - margin
        )

    def _backfill_track(
        self,
        store: TemporalStore,
        records: list[TemporalTrackRecord],
    ) -> None:
        if not self.backfill_frames:
            return
        seed = next(
            (
                record
                for record in records
                if record.observed
                and not record.ambiguous
                and not record.fallback
                and record.mask_path
            ),
            None,
        )
        if seed is None:
            return
        _scene, seed_gray, source_shape = store.frame(seed.frame_index)
        if self._touches_edge(seed.box, source_shape):
            return
        current_mask = store.load_mask(seed)
        if current_mask is None:
            return
        current_gray = seed_gray
        low_h, low_w = current_mask.shape
        source_h, source_w = source_shape
        for frame_index in range(
            seed.frame_index - 1,
            max(-1, seed.frame_index - self.backfill_frames - 1),
            -1,
        ):
            scene_id, previous_gray, previous_source_shape = store.frame(frame_index)
            if scene_id != seed.scene_id or previous_source_shape != source_shape:
                break
            existing = next(
                (
                    item
                    for item in store.records_for_frame(frame_index)
                    if item.track_id == seed.track_id
                ),
                None,
            )
            if existing is not None:
                existing_mask = store.load_mask(existing)
                if (
                    existing.ambiguous
                    or existing.fallback
                    or existing_mask is None
                ):
                    break
                # A model-produced reverse mask is authoritative. Continue
                # farther backward from it without replacing it with flow.
                current_mask = existing_mask
                current_gray = previous_gray
                continue
            candidate = _warp_mask(current_mask, current_gray, previous_gray)
            if candidate is None or not np.any(candidate):
                break
            # Reject weak appearance support. This is deliberately permissive
            # enough for a weakly visible face but rejects background copying.
            candidate_appearance = _appearance(previous_gray, candidate)
            source_appearance = _appearance(current_gray, current_mask)
            if candidate_appearance is None or source_appearance is None:
                break
            if (
                np.linalg.norm(candidate_appearance - source_appearance)
                > (70 if len(candidate_appearance) == 3 else 48)
            ):
                break
            rows, columns = np.nonzero(candidate)
            low_box = [
                int(columns.min()),
                int(rows.min()),
                int(columns.max()) + 1,
                int(rows.max()) + 1,
            ]
            box = [
                round(low_box[0] * source_w / low_w),
                round(low_box[1] * source_h / low_h),
                round(low_box[2] * source_w / low_w),
                round(low_box[3] * source_h / low_h),
            ]
            if self._touches_edge(box, source_shape):
                break
            others = [
                item
                for item in store.records_for_frame(frame_index)
                if item.track_id != seed.track_id
            ]
            if any(box_iou(box, item.box) >= 0.20 for item in others):
                break
            inferred = TemporalTrackRecord(
                frame_index,
                seed.scene_id,
                seed.track_id,
                box,
                (box,),
                seed.confidence,
                seed.threshold,
                False,
                False,
                False,
                None,
            )
            store.add_track(
                inferred.frame_index,
                inferred.scene_id,
                inferred.track_id,
                inferred.box,
                inferred.coverage_boxes,
                inferred.confidence,
                inferred.threshold,
                inferred.observed,
                inferred.ambiguous,
                inferred.fallback,
                candidate,
                replace=True,
            )
            current_mask, current_gray = candidate, previous_gray

    def _refine_track_backward(
        self,
        store: TemporalStore,
        scene_id: int,
        track_id: int,
    ) -> None:
        """Use later complete contours to repair earlier under-segmentation."""
        if not self.backfill_frames:
            return
        future_record = None
        future_mask = None
        future_gray = None
        future_origin = None
        for record in store.records_for_track_reverse(scene_id, track_id):
            _scene, gray, source_shape = store.frame(record.frame_index)
            raw = store.load_mask(record)
            can_refine = bool(
                raw is not None
                and not record.fallback
                and not record.ambiguous
                and future_record is not None
                and future_mask is not None
                and future_gray is not None
                and future_origin is not None
                and future_record.frame_index == record.frame_index + 1
                and future_origin - record.frame_index
                <= self.backfill_frames
                and not future_record.fallback
                and not future_record.ambiguous
            )
            refined = raw
            added_future_pixels = False
            if can_refine:
                aligned = _warp_mask(future_mask, future_gray, gray)
                if aligned is not None and np.any(aligned):
                    raw_pixels = np.count_nonzero(raw)
                    aligned_pixels = np.count_nonzero(aligned)
                    intersection = np.count_nonzero(
                        (raw > 0) & (aligned > 0)
                    )
                    overlap_over_smaller = intersection / max(
                        1, min(raw_pixels, aligned_pixels)
                    )
                    geometry = _coverage_mask(
                        record, source_shape, raw.shape
                    )
                    geometry_pixels = np.count_nonzero(geometry)
                    inside_geometry = np.count_nonzero(
                        (aligned > 0) & (geometry > 0)
                    ) / max(1, aligned_pixels)
                    aligned_rows, aligned_columns = np.nonzero(aligned)
                    aligned_center_in_geometry = bool(
                        len(aligned_rows)
                        and geometry[
                            int(np.median(aligned_rows)),
                            int(np.median(aligned_columns)),
                        ]
                        > 0
                    )
                    raw_appearance = _appearance(gray, raw)
                    aligned_appearance = _appearance(gray, aligned)
                    appearance_ok = bool(
                        raw_appearance is not None
                        and aligned_appearance is not None
                        and np.linalg.norm(
                            raw_appearance - aligned_appearance
                        )
                        <= (
                            70
                            if len(aligned_appearance) == 3
                            else 48
                        )
                    )
                    if (
                        overlap_over_smaller >= 0.45
                        and inside_geometry >= 0.45
                        and aligned_center_in_geometry
                        and aligned_pixels <= max(1, raw_pixels) * 16.0
                        and aligned_pixels
                        <= max(1, geometry_pixels) * 1.5
                        and appearance_ok
                    ):
                        candidate = cv2.bitwise_or(raw, aligned)
                        if np.any((candidate > 0) & (raw == 0)):
                            refined = candidate
                            added_future_pixels = True
                            store.replace_mask(
                                record, refined, fallback=False
                            )
            if (
                raw is None
                or record.fallback
                or record.ambiguous
                or not added_future_pixels
            ):
                # Nothing from the future was accepted into this record; start
                # a new bounded provenance window from its own observation.
                future_origin = record.frame_index
            future_record = record
            future_mask = refined
            future_gray = gray
            if future_origin is None:
                future_origin = record.frame_index

    def _stabilize_track(
        self,
        store: TemporalStore,
        scene_id: int,
        track_id: int,
    ) -> None:
        previous_mask = None
        previous_gray = None
        hold_age = None
        previous_appearance = None
        previous_index = None
        previous_failed = False
        previous_ambiguous = False
        failure_age = 0
        for record in store.records_for_track(scene_id, track_id):
            _scene, gray, source_shape = store.frame(record.frame_index)
            raw = store.load_mask(record)
            discontinuity = (
                previous_index is None
                or record.frame_index != previous_index + 1
                or record.ambiguous
            )
            if discontinuity:
                previous_mask = hold_age = previous_gray = None
                previous_appearance = None
            aligned = (
                _warp_mask(previous_mask, previous_gray, gray)
                if previous_mask is not None and previous_gray is not None
                else None
            )
            if (
                aligned is not None
                and hold_age is not None
                and previous_gray is not None
            ):
                hold_age = _warp_age(hold_age, previous_gray, gray)
            current_appearance = (
                _appearance(gray, raw) if raw is not None else None
            )
            if (
                aligned is not None
                and previous_appearance is not None
                and current_appearance is not None
                and np.linalg.norm(
                    previous_appearance - current_appearance
                )
                > (70 if len(current_appearance) == 3 else 48)
            ):
                aligned = None
                hold_age = None
            if raw is None:
                if (
                    aligned is None
                    or self.release_hold_frames == 0
                    or record.ambiguous
                ):
                    stable = None
                    fallback = True
                    hold_age = None
                    failure_age += 1
                else:
                    failure_age += 1
                    expanded = _coverage_mask(
                        record,
                        source_shape,
                        aligned.shape,
                        self.geometry_scale,
                    )
                    core = _coverage_mask(
                        record, source_shape, aligned.shape, 1.0
                    )
                    aligned = _filter_aligned_history(
                        aligned,
                        np.zeros_like(aligned),
                        expanded,
                    )
                    if (
                        aligned is not None
                        and _mask_matches_geometry(aligned, core)
                    ):
                        stable = _blend_toward_geometry(
                            aligned,
                            expanded,
                            failure_age,
                            self.release_hold_frames,
                        )
                        fallback = False
                        hold_age = np.zeros_like(stable, dtype=np.uint8)
                    else:
                        stable = None
                        fallback = True
                        hold_age = None
            elif aligned is None:
                stable = raw
                fallback = False
                hold_age = np.zeros_like(raw, dtype=np.uint8)
            else:
                current_geometry = _coverage_mask(
                    record, source_shape, raw.shape, self.geometry_scale
                )
                aligned = _filter_aligned_history(
                    aligned, raw, current_geometry
                )
                if aligned is None:
                    stable = raw
                    fallback = False
                    hold_age = np.zeros_like(raw, dtype=np.uint8)
                    overlap = None
                else:
                    overlap = aligned_iou(aligned, raw)
                if overlap is None:
                    pass
                else:
                    raw_pixels = np.count_nonzero(raw)
                    aligned_pixels = np.count_nonzero(aligned)
                    area_ratio = raw_pixels / max(1, aligned_pixels)
                    intersection = np.count_nonzero(
                        (aligned > 0) & (raw > 0)
                    )
                    overlap_over_smaller = intersection / max(
                        1, min(raw_pixels, aligned_pixels)
                    )
                    if (
                        overlap_over_smaller < 0.45
                        and (
                            overlap < 0.12
                            or not 0.45 <= area_ratio <= 2.2
                        )
                    ):
                        # Real rapid motion/correction: respond immediately and
                        # discard history instead of creating a trail.
                        stable = raw
                        hold_age = np.zeros_like(raw, dtype=np.uint8)
                    else:
                        missing = (aligned > 0) & (raw == 0)
                        hold_age = (
                            np.zeros_like(raw, dtype=np.uint8)
                            if hold_age is None or hold_age.shape != raw.shape
                            else hold_age
                        )
                        hold_age = np.where(
                            raw > 0,
                            0,
                            np.minimum(
                                255,
                                hold_age + missing.astype(np.uint8),
                            ),
                        ).astype(np.uint8)
                        held = missing & (
                            hold_age <= self.release_hold_frames
                        )
                        stable = np.where(
                            (raw > 0) | held, 255, 0
                        ).astype(np.uint8)
                    fallback = False
            if (
                raw is not None
                and not record.ambiguous
                and previous_failed
                and self.release_hold_frames > 0
            ):
                expanded = _coverage_mask(
                    record,
                    source_shape,
                    raw.shape,
                    self.geometry_scale,
                )
                # For ordinary intermittent failures, continue from the
                # aligned staged envelope. After an ambiguous crossing no
                # history is reused; current geometry is the safe boundary.
                envelope = (
                    aligned
                    if (
                        aligned is not None
                        and not previous_ambiguous
                    )
                    else expanded
                )
                stable = cv2.bitwise_or(raw, envelope)
                hold_age = _seed_release_age(
                    stable, raw, self.release_hold_frames
                )
                fallback = False
            if raw is not None:
                failure_age = 0
            store.replace_mask(record, stable, fallback=fallback)
            current_failed = bool(
                record.ambiguous or raw is None or fallback
            )
            if record.ambiguous:
                # Do not use either side of a crossing as identity history.
                previous_mask = previous_gray = hold_age = None
                previous_appearance = None
                previous_index = None
            else:
                previous_mask = stable
                previous_gray = gray
                previous_appearance = current_appearance
                previous_index = record.frame_index
            previous_failed = current_failed
            previous_ambiguous = record.ambiguous

    @staticmethod
    def _fallback_track(
        store: TemporalStore, scene_id: int, track_id: int
    ) -> None:
        for record in store.records_for_track(scene_id, track_id):
            store.replace_mask(record, None, fallback=True)

    def process(
        self,
        store: TemporalStore,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Apply bounded reverse repair followed by forward hysteresis."""
        store.checkpoint_masks()
        # Account for checkpoint metadata before doing any further propagation
        # work, so a long video cannot temporarily exceed the configured cap.
        store.commit()
        track_count = store.track_key_count()
        total = max(1, track_count * 2)
        completed = 0
        if progress is not None:
            progress(completed, total)
        for scene_id, track_id in store.track_keys():
            try:
                self._backfill_track(
                    store, store.records_for_track(scene_id, track_id)
                )
                self._refine_track_backward(store, scene_id, track_id)
            except Exception as exc:  # noqa: BLE001 - per-track privacy boundary
                print(
                    "[WARN] Temporal backfill failed for "
                    f"scene={scene_id} track={track_id}; "
                    "using geometric coverage for affected frames: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._fallback_track(store, scene_id, track_id)
            completed += 1
            if progress is not None:
                progress(completed, total)
        store.commit()
        for scene_id, track_id in store.track_keys():
            try:
                self._stabilize_track(store, scene_id, track_id)
            except Exception as exc:  # noqa: BLE001 - per-track privacy boundary
                print(
                    "[WARN] Temporal stabilization failed for "
                    f"scene={scene_id} track={track_id}; "
                    "using geometric coverage for affected frames: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._fallback_track(store, scene_id, track_id)
            recovered_count = store.restore_missing_analyzed_masks(
                scene_id, track_id
            )
            if recovered_count:
                # A temporal-stage failure must not erase segmentation that
                # was already accepted for unaffected frames. Recover those
                # immutable observations; records without one remain geometry.
                print(
                    "[WARN] Temporal track lost valid analyzed masks for "
                    f"scene={scene_id} track={track_id}; recovered "
                    f"{recovered_count}, with failed frames using geometric "
                    "coverage"
                )
            completed += 1
            if progress is not None:
                progress(completed, total)
        store.commit()
        if completed < total and progress is not None:
            progress(total, total)


def _signed_mask_distance(mask: np.ndarray) -> np.ndarray:
    foreground = np.where(mask > 0, 255, 0).astype(np.uint8)
    background = cv2.bitwise_not(foreground)
    return cv2.distanceTransform(
        foreground, cv2.DIST_L2, 3
    ) - cv2.distanceTransform(background, cv2.DIST_L2, 3)


def interpolate_aligned_masks(
    aligned: np.ndarray,
    current: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Interpolate aligned contours rather than averaging binary pixels."""
    alpha = max(0.0, min(1.0, float(alpha)))
    if alpha <= 0:
        return np.where(aligned > 0, 255, 0).astype(np.uint8)
    if alpha >= 1:
        return np.where(current > 0, 255, 0).astype(np.uint8)
    field = (
        (1.0 - alpha) * _signed_mask_distance(aligned)
        + alpha * _signed_mask_distance(current)
    )
    return np.where(field >= 0, 255, 0).astype(np.uint8)


def _shrink_mask(mask: np.ndarray, step: int, total: int) -> np.ndarray:
    if total <= 0 or step > total:
        return np.zeros_like(mask)
    distance = cv2.distanceTransform(
        np.where(mask > 0, 255, 0).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    maximum = float(distance.max())
    if maximum <= 0:
        return np.zeros_like(mask)
    threshold = maximum * step / (total + 1)
    return np.where(distance > threshold, 255, 0).astype(np.uint8)


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    count, labels = cv2.connectedComponents(
        np.where(mask > 0, 1, 0).astype(np.uint8),
        connectivity=8,
    )
    return [
        np.where(labels == label, 255, 0).astype(np.uint8)
        for label in range(1, count)
    ]


def _match_mask_components(
    aligned: np.ndarray,
    current: np.ndarray,
) -> tuple[
    list[tuple[np.ndarray, np.ndarray]],
    list[np.ndarray],
    list[np.ndarray],
]:
    """Associate final-mask components without letting one face anchor another."""
    historical = _mask_components(aligned)
    observed = _mask_components(current)
    candidates = []
    for old_index, old in enumerate(historical):
        old_pixels = np.count_nonzero(old)
        for new_index, new in enumerate(observed):
            new_pixels = np.count_nonzero(new)
            intersection = np.count_nonzero((old > 0) & (new > 0))
            score = intersection / max(1, min(old_pixels, new_pixels))
            ratio = new_pixels / max(1, old_pixels)
            if (
                (score >= 0.35 and 0.30 <= ratio <= 3.3)
                or (score >= 0.65 and 0.10 <= ratio <= 10.0)
            ):
                candidates.append((score, old_index, new_index))
    pairs = []
    used_old = set()
    used_new = set()
    for _score, old_index, new_index in sorted(candidates, reverse=True):
        if old_index in used_old or new_index in used_new:
            continue
        used_old.add(old_index)
        used_new.add(new_index)
        pairs.append((historical[old_index], observed[new_index]))
    return (
        pairs,
        [
            component
            for index, component in enumerate(historical)
            if index not in used_old
        ],
        [
            component
            for index, component in enumerate(observed)
            if index not in used_new
        ],
    )


def _static_overlap_alignment(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray | None:
    """Use image coordinates only when two masks already strongly agree."""
    if (
        not isinstance(source, np.ndarray)
        or not isinstance(target, np.ndarray)
        or source.ndim != 2
        or target.ndim != 2
        or source.shape != target.shape
    ):
        return None
    source_mask = source > 0
    target_mask = target > 0
    source_pixels = int(np.count_nonzero(source_mask))
    target_pixels = int(np.count_nonzero(target_mask))
    if source_pixels == 0 or target_pixels == 0:
        return None
    intersection = int(np.count_nonzero(source_mask & target_mask))
    overlap = intersection / max(1, min(source_pixels, target_pixels))
    ratio = source_pixels / max(1, target_pixels)
    if overlap < 0.80 or not 0.50 <= ratio <= 2.0:
        return None

    source_binary = np.where(source_mask, 255, 0).astype(np.uint8)
    target_binary = np.where(target_mask, 255, 0).astype(np.uint8)
    pairs, unmatched_source, _unmatched_target = _match_mask_components(
        source_binary, target_binary
    )
    if not pairs or unmatched_source:
        return None

    source_y, source_x = np.nonzero(source_mask)
    target_y, target_x = np.nonzero(target_mask)
    scale = max(1.0, np.sqrt(min(source_pixels, target_pixels)))
    center_distance = np.hypot(
        float(source_x.mean() - target_x.mean()),
        float(source_y.mean() - target_y.mean()),
    )
    if center_distance > 0.20 * scale:
        return None

    # Reject a common large component that merely happens to overlap while a
    # remote lobe belongs to another person. Modest contour expansion and a
    # nearby geometric fallback remain eligible.
    for candidate, reference in (
        (source_mask, target_mask),
        (target_mask, source_mask),
    ):
        outside = candidate & ~reference
        if not np.any(outside):
            continue
        distance = cv2.distanceTransform(
            np.where(reference, 0, 255).astype(np.uint8),
            cv2.DIST_L2,
            3,
        )
        if float(np.max(distance[outside])) > 0.30 * scale:
            return None
    return source_binary


def _merge_component_transition(
    aligned: np.ndarray,
    current: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    pairs, unmatched_old, unmatched_new = _match_mask_components(
        aligned, current
    )
    # Current final coverage is a hard lower bound. Offline reverse repair
    # smooths growth into earlier frames; the forward pass must not delay a
    # newly visible face or erase current union/fallback coverage.
    result = np.where(current > 0, 255, 0).astype(np.uint8)
    for old, new in pairs:
        result = cv2.bitwise_or(
            result, interpolate_aligned_masks(old, new, alpha)
        )
    for new in unmatched_new:
        # A newly appearing face/component is never delayed by smoothing.
        result = cv2.bitwise_or(result, new)
    return result, unmatched_old, unmatched_new


class FinalMaskStabilizer:
    """Smooth the final scene mask after all Track IDs and policies are merged."""

    def __init__(self, backfill_frames: int, release_hold_frames: int) -> None:
        self.backfill_frames = max(0, int(backfill_frames))
        self.release_hold_frames = max(0, int(release_hold_frames))

    def _reverse(
        self,
        store: TemporalStore,
        progress: Callable[[int, int], None] | None = None,
        total: int = 0,
    ) -> None:
        future_mask = None
        future_gray = None
        future_scene = None
        future_age = 0
        completed = 0
        for frame_index, scene_id, _path in store.composite_rows(reverse=True):
            _scene, gray, _source_shape = store.frame(frame_index)
            current = store.load_composite_mask(frame_index)
            if current is None:
                future_mask = future_gray = future_scene = None
                future_age = 0
                completed += 1
                if progress is not None:
                    progress(completed, total)
                continue
            if (
                future_mask is not None
                and future_gray is not None
                and future_scene == scene_id
                and future_age < self.backfill_frames
                and np.any(current)
            ):
                aligned = _warp_mask(future_mask, future_gray, gray)
                pairs = []
                if aligned is not None:
                    pairs, unmatched_current, _unmatched_future = (
                        _match_mask_components(current, aligned)
                    )
                if not pairs:
                    static_aligned = _static_overlap_alignment(
                        future_mask, current
                    )
                    if static_aligned is not None:
                        aligned = static_aligned
                        pairs, unmatched_current, _unmatched_future = (
                            _match_mask_components(current, aligned)
                        )
                if aligned is not None and pairs:
                    distance = future_age + 1
                    # Ease strongly into the corrected contour near its
                    # observation frame while still tapering over the whole
                    # configured reverse window.
                    alpha = 1.0 - (
                        distance / max(1, self.backfill_frames + 1)
                    ) ** 2
                    # Future-only components are intentionally omitted: late
                    # face discovery is repaired by the guarded per-track
                    # stage, while this final stage only smooths matched shape.
                    # Reverse repair may add a matched future contour, but
                    # never removes coverage already present in the earlier
                    # frame.
                    transitioned = np.where(
                        current > 0, 255, 0
                    ).astype(np.uint8)
                    for old, new in pairs:
                        transitioned = cv2.bitwise_or(
                            transitioned,
                            interpolate_aligned_masks(old, new, alpha),
                        )
                    for component in unmatched_current:
                        transitioned = cv2.bitwise_or(
                            transitioned, component
                        )
                    current = transitioned
                    future_age = future_age + 1 if pairs else 0
                else:
                    future_age = 0
            else:
                future_age = 0
            store.replace_composite_mask(frame_index, scene_id, current)
            future_mask = current
            future_gray = gray
            future_scene = scene_id
            completed += 1
            if progress is not None:
                progress(completed, total)

    def _forward(
        self,
        store: TemporalStore,
        progress: Callable[[int, int], None] | None,
        completed: int,
        total: int,
    ) -> None:
        previous_mask = None
        previous_gray = None
        previous_scene = None
        empty_age = 0
        shape_age = 0
        previous_age = None
        for frame_index, scene_id, _path in store.composite_rows():
            _scene, gray, _source_shape = store.frame(frame_index)
            current = store.load_composite_mask(frame_index)
            if current is None:
                previous_mask = previous_gray = previous_scene = None
                empty_age = 0
                shape_age = 0
                previous_age = None
            elif previous_scene != scene_id:
                previous_mask = None
                previous_gray = None
                empty_age = 0
                shape_age = 0
                previous_age = None
            if (
                current is not None
                and previous_mask is not None
                and previous_gray is not None
                and self.release_hold_frames > 0
            ):
                aligned = _warp_mask(previous_mask, previous_gray, gray)
                used_static_alignment = False
                if np.any(current):
                    flow_pairs = (
                        _match_mask_components(aligned, current)[0]
                        if aligned is not None
                        else []
                    )
                    if not flow_pairs:
                        static_aligned = _static_overlap_alignment(
                            previous_mask, current
                        )
                        if static_aligned is not None:
                            aligned = static_aligned
                            used_static_alignment = True
                if aligned is not None:
                    if previous_age is None:
                        aligned_age = np.zeros_like(
                            aligned, dtype=np.uint8
                        )
                    elif used_static_alignment:
                        aligned_age = previous_age.copy()
                    else:
                        aligned_age = _warp_age(
                            previous_age, previous_gray, gray
                        )
                    if aligned_age is None:
                        aligned_age = np.zeros_like(aligned, dtype=np.uint8)
                    if np.any(current):
                        pairs, unmatched_old, unmatched_new = (
                            _match_mask_components(aligned, current)
                        )
                        if pairs and not unmatched_old and not unmatched_new:
                            pair_old = np.zeros_like(aligned)
                            pair_new = np.zeros_like(current)
                            for old, new in pairs:
                                pair_old = cv2.bitwise_or(pair_old, old)
                                pair_new = cv2.bitwise_or(pair_new, new)
                            nearly_equal = (
                                aligned_iou(pair_old, pair_new) >= 0.92
                            )
                        else:
                            nearly_equal = False
                        if nearly_equal:
                            shape_age = 0
                        else:
                            shape_age += 1
                            remaining = max(
                                1,
                                self.release_hold_frames - shape_age + 1,
                            )
                            current, unmatched_old, unmatched_new = (
                                _merge_component_transition(
                                    aligned,
                                    current,
                                    1.0 / remaining,
                                )
                            )
                        current_age = np.zeros_like(current, dtype=np.uint8)
                        # If a current component appears while an unrelated
                        # historical component disappears, treat that pair as
                        # disjoint motion: add the new face immediately and do
                        # not let another matched face authorize a stale trail.
                        if not unmatched_new:
                            for old in unmatched_old:
                                old_pixels = old > 0
                                age = min(
                                    255,
                                    int(np.median(aligned_age[old_pixels]))
                                    + 1,
                                )
                                released = _shrink_mask(
                                    old, age, self.release_hold_frames
                                )
                                current = cv2.bitwise_or(
                                    current, released
                                )
                                current_age[released > 0] = age
                        previous_age = current_age
                        empty_age = 0
                    elif np.any(aligned):
                        shape_age = 0
                        empty_age += 1
                        current = _shrink_mask(
                            aligned,
                            empty_age,
                            self.release_hold_frames,
                        )
                        previous_age = np.full_like(
                            current, empty_age, dtype=np.uint8
                        )
                    else:
                        empty_age = 0
                        shape_age = 0
                        previous_age = np.zeros_like(
                            current, dtype=np.uint8
                        )
                else:
                    empty_age = 0
                    shape_age = 0
                    previous_age = np.zeros_like(
                        current, dtype=np.uint8
                    )
            elif current is not None and np.any(current):
                empty_age = 0
                shape_age = 0
                previous_age = np.zeros_like(current, dtype=np.uint8)
            if current is not None:
                store.replace_composite_mask(frame_index, scene_id, current)
                previous_mask = current
                previous_gray = gray
                previous_scene = scene_id
            completed += 1
            if progress is not None:
                progress(completed, total)

    def process(
        self,
        store: TemporalStore,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        rows = sum(1 for _row in store.composite_rows())
        total = max(1, rows * 2)
        if progress is not None:
            progress(0, total)
        self._reverse(store, progress, total)
        store.commit()
        self._forward(store, progress, rows, total)
        store.commit()


def resize_mask_to_source(
    mask: np.ndarray, source_shape: tuple[int, int]
) -> np.ndarray:
    height, width = source_shape
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
