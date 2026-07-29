import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from blurface.temporal import (
    FinalMaskStabilizer,
    SceneCutDetector,
    TemporalMaskStabilizer,
    TemporalStore,
    TemporalStoreError,
    aligned_iou,
    ambiguous_track_ids,
)


class _Track:
    def __init__(self, track_id, box):
        self.track_id = track_id
        self.box = box


def _add_frame(store, index, scene, gray):
    store.add_frame(index, scene, gray, gray.shape)


def _add_mask(
    store,
    index,
    scene,
    track_id,
    box,
    mask,
    *,
    observed=True,
    ambiguous=False,
    fallback=False,
):
    store.add_track(
        index,
        scene,
        track_id,
        box,
        (box,),
        0.9,
        0.3,
        observed,
        ambiguous,
        fallback,
        mask,
    )


class TemporalMaskTests(unittest.TestCase):
    def test_final_stabilizer_reports_reverse_and_forward_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                proxy = np.tile(
                    np.arange(48, dtype=np.uint8), (36, 1)
                )
                mask = np.zeros((36, 48), dtype=np.uint8)
                mask[10:24, 16:32] = 255
                for index in range(6):
                    _add_frame(store, index, 0, proxy)
                    store.replace_composite_mask(index, 0, mask)
                store.commit()
                updates = []

                FinalMaskStabilizer(10, 5).process(
                    store,
                    progress=lambda current, total: updates.append(
                        (current, total)
                    ),
                )

                self.assertEqual(
                    updates,
                    [(current, 12) for current in range(13)],
                )

    def test_final_mask_smooths_shape_change_across_track_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                proxy = np.tile(
                    np.arange(48, dtype=np.uint8), (36, 1)
                )
                small = np.zeros((36, 48), dtype=np.uint8)
                small[14:22, 20:28] = 255
                large = np.zeros_like(small)
                large[8:28, 14:34] = 255
                for index in range(6):
                    _add_frame(store, index, 0, proxy)
                # The final stage deliberately sees only the merged result;
                # these contours may have originated from different Track IDs.
                for index in range(5):
                    store.replace_composite_mask(index, 0, small)
                store.replace_composite_mask(5, 0, large)
                store.commit()

                FinalMaskStabilizer(10, 5).process(store)

                areas = [
                    np.count_nonzero(store.load_composite_mask(index))
                    for index in range(6)
                ]
                self.assertEqual(areas, sorted(areas))
                self.assertGreater(areas[0], np.count_nonzero(small))
                self.assertLess(areas[0], np.count_nonzero(large))
                self.assertEqual(areas[-2], np.count_nonzero(large))
                self.assertEqual(areas[-1], np.count_nonzero(large))
                raw_jump = (
                    np.count_nonzero(large) - np.count_nonzero(small)
                )
                self.assertLessEqual(max(np.diff(areas)), raw_jump * 0.25)
                self.assertEqual(
                    len(list(store.root.glob("composite/**/*.png"))), 6
                )

    def test_final_mask_never_smooths_across_scene_cut(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                proxy = np.tile(
                    np.arange(48, dtype=np.uint8), (36, 1)
                )
                first = np.zeros((36, 48), dtype=np.uint8)
                first[10:22, 8:20] = 255
                second = np.zeros_like(first)
                second[8:28, 22:42] = 255
                _add_frame(store, 0, 0, proxy)
                _add_frame(store, 1, 1, proxy)
                store.replace_composite_mask(0, 0, first)
                store.replace_composite_mask(1, 1, second)
                store.commit()

                FinalMaskStabilizer(10, 5).process(store)

                self.assertTrue(
                    np.array_equal(store.load_composite_mask(0), first)
                )
                self.assertTrue(
                    np.array_equal(store.load_composite_mask(1), second)
                )

    def test_final_mask_responds_immediately_to_disjoint_fast_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                proxy = np.tile(
                    np.arange(60, dtype=np.uint8), (40, 1)
                )
                first = np.zeros((40, 60), dtype=np.uint8)
                first[12:24, 4:16] = 255
                second = np.zeros_like(first)
                second[12:24, 42:54] = 255
                for index in range(2):
                    _add_frame(store, index, 0, proxy)
                store.replace_composite_mask(0, 0, first)
                store.replace_composite_mask(1, 0, second)
                store.commit()

                with patch("blurface.temporal._warp_mask", return_value=None):
                    FinalMaskStabilizer(10, 5).process(store)

                self.assertTrue(
                    np.array_equal(store.load_composite_mask(1), second)
                )

    def test_final_mask_uses_overlap_alignment_when_flow_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                proxy = np.tile(
                    np.arange(256, dtype=np.uint8), (192, 1)
                )
                contour = np.zeros((192, 256), dtype=np.uint8)
                contour[64:128, 80:176] = 255
                fallback = np.zeros_like(contour)
                fallback[56:136, 72:184] = 255
                for index in range(6):
                    _add_frame(store, index, 0, proxy)
                    store.replace_composite_mask(
                        index, 0, contour if index < 5 else fallback
                    )
                store.commit()

                with patch("blurface.temporal._warp_mask", return_value=None):
                    FinalMaskStabilizer(10, 5).process(store)

                areas = [
                    np.count_nonzero(store.load_composite_mask(index))
                    for index in range(6)
                ]
                raw_jump = (
                    np.count_nonzero(fallback)
                    - np.count_nonzero(contour)
                )
                self.assertEqual(areas, sorted(areas))
                self.assertGreater(areas[0], np.count_nonzero(contour))
                self.assertLess(areas[0], np.count_nonzero(fallback))
                self.assertEqual(areas[-2], np.count_nonzero(fallback))
                self.assertEqual(areas[-1], np.count_nonzero(fallback))
                self.assertLessEqual(max(np.diff(areas)), raw_jump * 0.15)

    def test_final_mask_immediately_adds_a_new_face_component(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                proxy = np.tile(
                    np.arange(80, dtype=np.uint8), (40, 1)
                )
                face_a = np.zeros((40, 80), dtype=np.uint8)
                face_a[10:30, 8:28] = 255
                face_b = np.zeros_like(face_a)
                face_b[10:30, 52:72] = 255
                for index in range(2):
                    _add_frame(store, index, 0, proxy)
                store.replace_composite_mask(0, 0, face_a)
                store.replace_composite_mask(
                    1, 0, cv2.bitwise_or(face_a, face_b)
                )
                store.commit()

                FinalMaskStabilizer(0, 5).process(store)

                stabilized = store.load_composite_mask(1)
                self.assertTrue(np.all(stabilized[face_b > 0] == 255))

    def test_common_face_cannot_authorize_disjoint_component_history(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                proxy = np.tile(
                    np.arange(100, dtype=np.uint8), (40, 1)
                )
                face_a = np.zeros((40, 100), dtype=np.uint8)
                face_a[10:30, 4:24] = 255
                face_b = np.zeros_like(face_a)
                face_b[10:30, 38:58] = 255
                face_c = np.zeros_like(face_a)
                face_c[10:30, 76:96] = 255
                for index in range(2):
                    _add_frame(store, index, 0, proxy)
                store.replace_composite_mask(
                    0, 0, cv2.bitwise_or(face_a, face_b)
                )
                store.replace_composite_mask(
                    1, 0, cv2.bitwise_or(face_a, face_c)
                )
                store.commit()

                with patch("blurface.temporal._warp_mask", return_value=None):
                    FinalMaskStabilizer(10, 5).process(store)

                stabilized = store.load_composite_mask(1)
                self.assertTrue(np.all(stabilized[face_c > 0] == 255))
                self.assertEqual(
                    np.count_nonzero(stabilized[face_b > 0]), 0
                )

    def test_static_alignment_cannot_release_remote_old_component(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                proxy = np.tile(
                    np.arange(180, dtype=np.uint8), (140, 1)
                )
                current_a = np.zeros((140, 180), dtype=np.uint8)
                current_a[20:120, 10:110] = 255
                previous = np.zeros_like(current_a)
                previous[19:121, 9:111] = 255
                previous[125:128, 170:173] = 255
                for index in range(2):
                    _add_frame(store, index, 0, proxy)
                store.replace_composite_mask(0, 0, previous)
                store.replace_composite_mask(1, 0, current_a)
                store.commit()

                with patch("blurface.temporal._warp_mask", return_value=None):
                    FinalMaskStabilizer(0, 5).process(store)

                stabilized = store.load_composite_mask(1)
                self.assertTrue(np.all(stabilized[current_a > 0] == 255))
                self.assertEqual(
                    np.count_nonzero(stabilized[125:128, 170:173]), 0
                )

    def test_static_reverse_rejects_remote_lobe_behind_thin_bridge(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                proxy = np.tile(
                    np.arange(400, dtype=np.uint16), (200, 1)
                ).astype(np.uint8)
                current = np.zeros((200, 400), dtype=np.uint8)
                current[20:120, 20:120] = 255
                future = np.zeros_like(current)
                future[19:121, 19:121] = 255
                future[70, 121:340] = 255
                future[68:73, 340:345] = 255
                for index in range(2):
                    _add_frame(store, index, 0, proxy)
                store.replace_composite_mask(0, 0, current)
                store.replace_composite_mask(1, 0, future)
                store.commit()

                with patch("blurface.temporal._warp_mask", return_value=None):
                    FinalMaskStabilizer(10, 0).process(store)

                stabilized = store.load_composite_mask(0)
                self.assertTrue(np.all(stabilized[current > 0] == 255))
                self.assertEqual(
                    np.count_nonzero(stabilized[68:73, 340:345]), 0
                )

    def test_connected_crossing_never_removes_current_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                proxy = np.tile(
                    np.arange(70, dtype=np.uint8), (40, 1)
                )
                separated = np.zeros((40, 70), dtype=np.uint8)
                separated[10:30, 5:25] = 255
                separated[10:30, 35:55] = 255
                connected = separated.copy()
                connected[15:25, 25:35] = 255
                for index in range(2):
                    _add_frame(store, index, 0, proxy)
                store.replace_composite_mask(0, 0, separated)
                store.replace_composite_mask(1, 0, connected)
                store.commit()

                FinalMaskStabilizer(0, 5).process(store)

                stabilized = store.load_composite_mask(1)
                self.assertTrue(np.all(stabilized[connected > 0] == 255))

    def test_reverse_alignment_never_removes_current_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                proxy = np.tile(
                    np.arange(60, dtype=np.uint8), (40, 1)
                )
                current = np.zeros((40, 60), dtype=np.uint8)
                current[10:30, 15:35] = 255
                future = np.zeros_like(current)
                future[10:30, 20:40] = 255
                for index in range(2):
                    _add_frame(store, index, 0, proxy)
                store.replace_composite_mask(0, 0, current)
                store.replace_composite_mask(1, 0, future)
                store.commit()

                FinalMaskStabilizer(10, 0).process(store)

                stabilized = store.load_composite_mask(0)
                self.assertTrue(np.all(stabilized[current > 0] == 255))

    def test_track_failure_preserves_valid_observations_and_reports_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                gray = np.zeros((30, 40), dtype=np.uint8)
                _add_frame(store, 0, 0, gray)
                mask = np.zeros_like(gray)
                mask[8:20, 10:25] = 255
                _add_mask(
                    store, 0, 0, 4, [10, 8, 25, 20], mask
                )
                store.commit()
                updates = []
                stabilizer = TemporalMaskStabilizer(5, 3)
                with patch.object(
                    stabilizer,
                    "_backfill_track",
                    side_effect=ValueError(
                        "too many values to unpack (expected 2)"
                    ),
                ):
                    stabilizer.process(
                        store,
                        progress=lambda current, total: updates.append(
                            (current, total)
                        ),
                    )
                record = store.records_for_frame(0)[0]
                self.assertFalse(record.fallback)
                self.assertIsNotNone(store.load_mask(record))
                self.assertEqual(updates[0], (0, 2))
                self.assertEqual(updates[-1], (2, 2))
                self.assertEqual(store.mask_stats(), (1, 1, 0))

    def test_stabilization_never_overwrites_raw_mask_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                gray = np.full((30, 40), 100, dtype=np.uint8)
                _add_frame(store, 0, 0, gray)
                mask = np.zeros_like(gray)
                mask[8:20, 10:25] = 255
                _add_mask(store, 0, 0, 4, [10, 8, 25, 20], mask)
                store.commit()
                original = store.records_for_frame(0)[0]
                original_path = store.root / original.mask_path
                original_bytes = original_path.read_bytes()

                TemporalMaskStabilizer(0, 3).process(store)

                stabilized = store.records_for_frame(0)[0]
                self.assertNotEqual(stabilized.mask_path, original.mask_path)
                self.assertTrue(stabilized.mask_path.startswith("stable/"))
                self.assertEqual(original_path.read_bytes(), original_bytes)
                self.assertIsNotNone(store.load_mask(stabilized))
                self.assertEqual(store.mask_stats(), (1, 1, 0))

    def test_replacing_stable_mask_reclaims_old_revision_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                gray = np.full((30, 40), 100, dtype=np.uint8)
                _add_frame(store, 0, 0, gray)
                mask = np.zeros_like(gray)
                mask[8:20, 10:25] = 255
                _add_mask(store, 0, 0, 4, [10, 8, 25, 20], mask)
                raw = store.records_for_frame(0)[0]
                raw_path = store.root / raw.mask_path

                store.replace_mask(raw, mask, fallback=False)
                first_stable = store.records_for_frame(0)[0]
                current_before = store.current_bytes
                store.replace_mask(first_stable, mask, fallback=False)
                second_stable = store.records_for_frame(0)[0]

                self.assertTrue(raw_path.is_file())
                self.assertFalse((store.root / first_stable.mask_path).exists())
                self.assertTrue((store.root / second_stable.mask_path).is_file())
                self.assertEqual(
                    len(list(store.root.glob("stable/**/*.png"))), 1
                )
                self.assertEqual(store.current_bytes, current_before)
                actual_bytes = sum(
                    path.stat().st_size
                    for path in store.root.rglob("*")
                    if path.is_file()
                )
                self.assertEqual(store.current_bytes, actual_bytes)
                self.assertGreater(store.image_bytes_written, 0)
                self.assertGreaterEqual(store.peak_bytes, store.current_bytes)

    def test_storage_usage_includes_sqlite_and_all_current_files(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                gray = np.full((30, 40), 100, dtype=np.uint8)
                _add_frame(store, 0, 0, gray)
                mask = np.zeros_like(gray)
                mask[8:24, 10:30] = 255
                _add_mask(store, 0, 0, 4, [10, 8, 30, 24], mask)
                store.checkpoint_masks()
                store.commit()

                database_bytes = sum(
                    path.stat().st_size
                    for path in store.root.glob("analysis.sqlite3*")
                )
                actual_bytes = sum(
                    path.stat().st_size
                    for path in store.root.rglob("*")
                    if path.is_file()
                )
                self.assertGreater(database_bytes, 0)
                self.assertEqual(store.current_bytes, actual_bytes)
                self.assertGreater(
                    store.current_bytes, store.image_bytes_written
                )
                self.assertGreaterEqual(store.peak_bytes, actual_bytes)

    def test_mask_io_does_not_use_opencv_path_apis(self):
        with tempfile.TemporaryDirectory(prefix="遮罩-") as directory:
            with TemporalStore(Path(directory)) as store:
                mask = np.zeros((30, 40), dtype=np.uint8)
                mask[8:20, 10:25] = 255
                with (
                    patch(
                        "blurface.temporal.cv2.imwrite",
                        side_effect=AssertionError("path writer used"),
                    ),
                    patch(
                        "blurface.temporal.cv2.imread",
                        side_effect=AssertionError("path reader used"),
                    ),
                ):
                    _add_mask(
                        store, 0, 0, 4, [10, 8, 25, 20], mask
                    )
                    record = store.records_for_frame(0)[0]
                    loaded = store.load_mask(record)
                self.assertTrue(np.array_equal(loaded, mask))

    def test_negative_only_mask_is_not_persisted_as_valid_segmentation(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                logits = np.full((30, 40), -0.25, dtype=np.float32)
                _add_mask(
                    store, 0, 0, 4, [10, 8, 25, 20], logits
                )
                record = store.records_for_frame(0)[0]
                self.assertTrue(record.fallback)
                self.assertIsNone(record.mask_path)
                self.assertIsNone(store.load_mask(record))
                self.assertEqual(store.mask_stats(), (1, 0, 1))

    def test_corrupt_mask_file_is_counted_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                mask = np.zeros((30, 40), dtype=np.uint8)
                mask[8:20, 10:25] = 255
                _add_mask(store, 0, 0, 4, [10, 8, 25, 20], mask)
                record = store.records_for_frame(0)[0]
                (store.root / record.mask_path).write_bytes(b"not-a-png")

                self.assertIsNone(store.load_mask(record))
                self.assertEqual(
                    store.mask_read_failures["decode_failed"], 1
                )

    def test_stabilization_exception_recovers_only_valid_analyzed_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                gray = np.full((30, 40), 100, dtype=np.uint8)
                for index in range(2):
                    _add_frame(store, index, 0, gray)
                mask = np.zeros_like(gray)
                mask[8:20, 10:25] = 255
                _add_mask(store, 0, 0, 4, [10, 8, 25, 20], mask)
                _add_mask(
                    store,
                    1,
                    0,
                    4,
                    [10, 8, 25, 20],
                    None,
                    fallback=True,
                )
                store.commit()
                stabilizer = TemporalMaskStabilizer(0, 3)
                with patch.object(
                    stabilizer,
                    "_stabilize_track",
                    side_effect=RuntimeError("synthetic failure"),
                ):
                    stabilizer.process(store)

                first, second = (
                    store.records_for_frame(index)[0]
                    for index in range(2)
                )
                self.assertIsNotNone(store.load_mask(first))
                self.assertFalse(first.fallback)
                self.assertIsNone(store.load_mask(second))
                self.assertTrue(second.fallback)
                self.assertEqual(store.mask_stats(), (2, 1, 1))

    def test_checkpoint_recovery_is_scene_scoped_and_indexed(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                gray = np.full((30, 40), 100, dtype=np.uint8)
                first_mask = np.zeros_like(gray)
                first_mask[4:10, 5:12] = 255
                second_mask = np.zeros_like(gray)
                second_mask[18:25, 28:36] = 255
                for index, scene, mask in (
                    (0, 0, first_mask),
                    (1, 1, second_mask),
                ):
                    _add_frame(store, index, scene, gray)
                    _add_mask(
                        store,
                        index,
                        scene,
                        4,
                        [0, 0, 40, 30],
                        mask,
                    )
                store.commit()
                stabilizer = TemporalMaskStabilizer(0, 3)
                with patch.object(
                    stabilizer,
                    "_stabilize_track",
                    side_effect=RuntimeError("synthetic failure"),
                ):
                    stabilizer.process(store)

                recovered_first = store.load_mask(
                    store.records_for_frame(0)[0]
                )
                recovered_second = store.load_mask(
                    store.records_for_frame(1)[0]
                )
                self.assertTrue(np.array_equal(recovered_first, first_mask))
                self.assertTrue(np.array_equal(recovered_second, second_mask))
                query_plan = store._connection.execute(
                    "EXPLAIN QUERY PLAN SELECT frame_index "
                    "FROM analyzed_mask_checkpoint "
                    "WHERE scene_id = ? AND track_id = ?",
                    (1, 4),
                ).fetchall()
                self.assertIn(
                    "analyzed_checkpoint_scene_track",
                    " ".join(str(row) for row in query_plan),
                )

    def test_late_detection_backfills_visible_frames_but_not_before(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                for index in range(6):
                    gray = np.full(
                        (60, 80), 30 if index < 2 else 140, dtype=np.uint8
                    )
                    cv2.circle(gray, (40, 30), 10, 190, -1)
                    if index < 2:
                        cv2.circle(gray, (40, 30), 10, 30, -1)
                    _add_frame(store, index, 0, gray)
                mask = np.zeros((60, 80), dtype=np.uint8)
                cv2.circle(mask, (40, 30), 10, 255, -1)
                _add_mask(store, 5, 0, 7, [30, 20, 51, 41], mask)
                store.commit()

                TemporalMaskStabilizer(5, 3).process(store)

                self.assertFalse(store.records_for_frame(0))
                self.assertFalse(store.records_for_frame(1))
                for index in (2, 3, 4):
                    record = store.records_for_frame(index)[0]
                    self.assertEqual(record.track_id, 7)
                    self.assertIsNotNone(store.load_mask(record))

    def test_backfill_never_crosses_a_scene_cut(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                gray = np.full((50, 70), 120, dtype=np.uint8)
                for index in range(6):
                    _add_frame(store, index, 0 if index < 4 else 1, gray)
                mask = np.zeros_like(gray)
                mask[15:35, 25:45] = 255
                _add_mask(store, 5, 1, 3, [25, 15, 45, 35], mask)
                store.commit()
                TemporalMaskStabilizer(5, 3).process(store)
                self.assertFalse(store.records_for_frame(3))
                self.assertTrue(store.records_for_frame(4))

    def test_model_reverse_mask_is_not_overwritten_by_flow_backfill(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                gray = np.full((50, 70), 120, dtype=np.uint8)
                for index in range(3):
                    _add_frame(store, index, 0, gray)
                reverse_mask = np.zeros_like(gray)
                reverse_mask[10:20, 10:20] = 255
                seed_mask = np.zeros_like(gray)
                seed_mask[25:35, 35:45] = 255
                _add_mask(
                    store,
                    1,
                    0,
                    9,
                    [10, 10, 20, 20],
                    reverse_mask,
                    observed=False,
                )
                _add_mask(
                    store,
                    2,
                    0,
                    9,
                    [35, 25, 45, 35],
                    seed_mask,
                )
                store.commit()
                TemporalMaskStabilizer(2, 0).process(store)
                preserved = store.load_mask(store.records_for_frame(1)[0])
                self.assertEqual(preserved[15, 15], 255)
                self.assertEqual(preserved[30, 40], 0)

    def test_later_full_face_repairs_earlier_eye_only_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                frame = np.full((60, 80, 3), (120, 150, 180), dtype=np.uint8)
                for index in range(2):
                    _add_frame(store, index, 0, frame)
                eye_only = np.zeros((60, 80), dtype=np.uint8)
                eye_only[25:29, 32:48] = 255
                full_face = np.zeros_like(eye_only)
                cv2.ellipse(full_face, (40, 30), (13, 17), 0, 0, 360, 255, -1)
                _add_mask(
                    store,
                    0,
                    0,
                    7,
                    [25, 12, 56, 49],
                    eye_only,
                )
                _add_mask(
                    store,
                    1,
                    0,
                    7,
                    [25, 12, 56, 49],
                    full_face,
                )
                store.commit()

                TemporalMaskStabilizer(3, 3).process(store)

                repaired = store.load_mask(store.records_for_frame(0)[0])
                later = store.load_mask(store.records_for_frame(1)[0])
                self.assertGreater(
                    np.count_nonzero(repaired),
                    np.count_nonzero(eye_only) * 4,
                )
                self.assertGreater(aligned_iou(repaired, later), 0.8)

    def test_backward_refinement_stops_at_ambiguous_crossing(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                frame = np.full((60, 80, 3), 140, dtype=np.uint8)
                small = np.zeros((60, 80), dtype=np.uint8)
                small[25:30, 34:46] = 255
                large = np.zeros_like(small)
                cv2.circle(large, (40, 30), 14, 255, -1)
                for index, mask, ambiguous in (
                    (0, small, False),
                    (1, small, True),
                    (2, large, False),
                ):
                    _add_frame(store, index, 0, frame)
                    _add_mask(
                        store,
                        index,
                        0,
                        7,
                        [24, 14, 57, 47],
                        mask,
                        ambiguous=ambiguous,
                    )
                store.commit()

                TemporalMaskStabilizer(3, 3).process(store)

                first = store.load_mask(store.records_for_frame(0)[0])
                self.assertTrue(np.array_equal(first, small))

    def test_alternating_edge_pixel_is_held_in_aligned_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                gray = np.full((40, 50), 110, dtype=np.uint8)
                raw_masks = []
                for index in range(6):
                    _add_frame(store, index, 0, gray)
                    mask = np.zeros_like(gray)
                    mask[14:26, 18:30] = 255
                    if index % 2 == 0:
                        mask[20, 30] = 255
                    raw_masks.append(mask.copy())
                    _add_mask(store, index, 0, 1, [18, 14, 31, 26], mask)
                store.commit()
                TemporalMaskStabilizer(0, 3).process(store)
                stable = [
                    store.load_mask(store.records_for_frame(index)[0])
                    for index in range(6)
                ]
                raw_switches = sum(
                    raw_masks[index][20, 30]
                    != raw_masks[index - 1][20, 30]
                    for index in range(1, 6)
                )
                stable_switches = sum(
                    stable[index][20, 30] != stable[index - 1][20, 30]
                    for index in range(1, 6)
                )
                self.assertEqual(raw_switches, 5)
                self.assertLess(stable_switches, raw_switches)

    def test_disappearing_pixel_exits_after_configured_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                gray = np.full((40, 50), 110, dtype=np.uint8)
                for index in range(6):
                    _add_frame(store, index, 0, gray)
                    mask = np.zeros_like(gray)
                    mask[14:26, 18:30] = 255
                    if index == 0:
                        mask[20, 30] = 255
                    _add_mask(store, index, 0, 1, [18, 14, 31, 26], mask)
                store.commit()
                TemporalMaskStabilizer(0, 3).process(store)
                stable = [
                    store.load_mask(store.records_for_frame(index)[0])
                    for index in range(6)
                ]
                self.assertTrue(
                    all(mask[20, 30] == 255 for mask in stable[:4])
                )
                self.assertTrue(
                    all(mask[20, 30] == 0 for mask in stable[4:])
                )

    def test_geometric_fallback_releases_without_frame_to_frame_size_flash(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                gray = np.full((60, 80), 110, dtype=np.uint8)
                raw = np.zeros_like(gray)
                cv2.circle(raw, (40, 30), 8, 255, -1)
                for index in range(7):
                    _add_frame(store, index, 0, gray)
                    _add_mask(
                        store,
                        index,
                        0,
                        1,
                        [25, 15, 56, 46],
                        None if index == 1 else raw,
                        fallback=index == 1,
                    )
                store.commit()
                TemporalMaskStabilizer(0, 3, 1.5).process(store)
                records = [
                    store.records_for_frame(index)[0]
                    for index in range(7)
                ]
                masks = [store.load_mask(record) for record in records]
                raw_area = np.count_nonzero(raw)
                fallback_area = round((56 - 25) * 1.5) * round(
                    (46 - 15) * 1.5
                )

                self.assertFalse(records[1].fallback)
                self.assertIsNotNone(masks[1])
                # A validated propagated contour blends toward geometry
                # instead of hard-switching to the tracker rectangle.
                self.assertGreater(
                    np.count_nonzero(masks[1]), raw_area
                )
                self.assertLess(
                    np.count_nonzero(masks[1]), fallback_area
                )
                release_areas = [
                    np.count_nonzero(mask) for mask in masks[1:]
                ]
                self.assertEqual(release_areas, sorted(release_areas, reverse=True))
                self.assertEqual(release_areas[-1], raw_area)
                self.assertGreater(
                    len(set(release_areas)), 2
                )

    def test_fast_motion_after_fallback_never_reintroduces_old_history(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                gray = np.full((60, 120), 110, dtype=np.uint8)
                masks = []
                for center in (20, None, 90):
                    mask = None
                    if center is not None:
                        mask = np.zeros_like(gray)
                        cv2.circle(mask, (center, 30), 8, 255, -1)
                    masks.append(mask)
                for index, (center, mask) in enumerate(
                    zip((20, 75, 90), masks)
                ):
                    _add_frame(store, index, 0, gray)
                    _add_mask(
                        store,
                        index,
                        0,
                        2,
                        [center - 12, 18, center + 13, 43],
                        mask,
                        fallback=mask is None,
                    )
                store.commit()

                TemporalMaskStabilizer(0, 5, 1.5).process(store)

                self.assertTrue(
                    store.records_for_frame(1)[0].fallback
                )
                recovered = store.load_mask(
                    store.records_for_frame(2)[0]
                )
                self.assertEqual(recovered[30, 20], 0)
                self.assertEqual(recovered[30, 55], 0)
                self.assertEqual(recovered[30, 90], 255)

    def test_thin_bridge_cannot_carry_remote_history_through_missing_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                gray = np.full((60, 120), 110, dtype=np.uint8)
                bridged = np.zeros_like(gray)
                cv2.circle(bridged, (35, 30), 8, 255, -1)
                cv2.line(bridged, (43, 30), (92, 30), 255, 1)
                cv2.circle(bridged, (100, 30), 8, 255, -1)
                for index, mask in enumerate((bridged, None)):
                    _add_frame(store, index, 0, gray)
                    _add_mask(
                        store,
                        index,
                        0,
                        3,
                        [23, 18, 48, 43],
                        mask,
                        fallback=mask is None,
                    )
                store.commit()

                TemporalMaskStabilizer(0, 5, 1.5).process(store)

                missing = store.load_mask(
                    store.records_for_frame(1)[0]
                )
                self.assertIsNotNone(missing)
                self.assertEqual(missing[30, 100], 0)
                self.assertEqual(missing[30, 35], 255)

    def test_thin_bridge_cannot_carry_remote_history_into_recovered_raw(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                gray = np.full((60, 120), 110, dtype=np.uint8)
                bridged = np.zeros_like(gray)
                cv2.circle(bridged, (35, 30), 8, 255, -1)
                cv2.line(bridged, (43, 30), (92, 30), 255, 1)
                cv2.circle(bridged, (100, 30), 8, 255, -1)
                current = np.zeros_like(gray)
                cv2.circle(current, (35, 30), 8, 255, -1)
                for index, mask in enumerate((bridged, current)):
                    _add_frame(store, index, 0, gray)
                    _add_mask(
                        store,
                        index,
                        0,
                        3,
                        [23, 18, 48, 43],
                        mask,
                    )
                store.commit()

                TemporalMaskStabilizer(0, 5, 1.5).process(store)

                recovered = store.load_mask(
                    store.records_for_frame(1)[0]
                )
                self.assertEqual(recovered[30, 100], 0)
                self.assertEqual(recovered[30, 35], 255)

    def test_fast_motion_responds_without_a_stationary_trail(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                gray = np.zeros((60, 100), dtype=np.uint8)
                centers = (20, 42, 70)
                for index, center in enumerate(centers):
                    frame = gray.copy()
                    cv2.circle(frame, (center, 30), 8, 180, -1)
                    _add_frame(store, index, 0, frame)
                    mask = np.zeros_like(gray)
                    cv2.circle(mask, (center, 30), 8, 255, -1)
                    _add_mask(
                        store,
                        index,
                        0,
                        2,
                        [center - 8, 22, center + 9, 39],
                        mask,
                    )
                store.commit()
                TemporalMaskStabilizer(0, 3).process(store)
                last = store.load_mask(store.records_for_frame(2)[0])
                self.assertEqual(last[30, 70], 255)
                self.assertEqual(last[30, 20], 0)

    def test_real_edge_entry_is_not_backfilled(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                gray = np.full((50, 80), 120, dtype=np.uint8)
                for index in range(5):
                    _add_frame(store, index, 0, gray)
                mask = np.zeros_like(gray)
                mask[15:35, 0:14] = 255
                _add_mask(store, 4, 0, 4, [0, 15, 14, 35], mask)
                store.commit()
                TemporalMaskStabilizer(4, 3).process(store)
                self.assertFalse(store.records_for_frame(3))

    def test_crossing_tracks_are_marked_ambiguous_without_identity_merge(self):
        tracks = [_Track(10, [10, 10, 40, 40]), _Track(11, [20, 10, 50, 40])]
        self.assertEqual(ambiguous_track_ids(tracks), {10, 11})
        near_crossing = [
            _Track(20, [10, 10, 30, 30]),
            _Track(21, [31, 10, 51, 30]),
        ]
        self.assertEqual(ambiguous_track_ids(near_crossing), {20, 21})
        nearby_but_separate = [
            _Track(30, [10, 10, 30, 30]),
            _Track(31, [34, 10, 54, 30]),
            # A larger neighbor must not use its own diagonal to quarantine a
            # clearly separate smaller face.
            _Track(32, [60, 5, 105, 50]),
        ]
        self.assertEqual(
            ambiguous_track_ids(nearby_but_separate), set()
        )
        size_mismatched_overlap = [
            _Track(40, [0, 0, 200, 200]),
            _Track(41, [150, 150, 170, 170]),
        ]
        self.assertEqual(
            ambiguous_track_ids(size_mismatched_overlap), {40, 41}
        )
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                gray = np.full((60, 80), 100, dtype=np.uint8)
                for index in range(2):
                    _add_frame(store, index, 0, gray)
                    for track_id, start in ((10, 10), (11, 40)):
                        mask = np.zeros_like(gray)
                        mask[20:35, start : start + 15] = 255
                        _add_mask(
                            store,
                            index,
                            0,
                            track_id,
                            [start, 20, start + 15, 35],
                            mask,
                            ambiguous=index == 1,
                        )
                store.commit()
                TemporalMaskStabilizer(0, 3).process(store)
                left = store.load_mask(
                    next(
                        item
                        for item in store.records_for_frame(1)
                        if item.track_id == 10
                    )
                )
                right = store.load_mask(
                    next(
                        item
                        for item in store.records_for_frame(1)
                        if item.track_id == 11
                    )
                )
                self.assertEqual(left[25, 12], 255)
                self.assertEqual(left[25, 45], 0)
                self.assertEqual(right[25, 45], 255)

    def test_large_appearance_change_resets_history_before_identity_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                colors = ((20, 20, 220), (220, 180, 20))
                masks = []
                for index, color in enumerate(colors):
                    proxy = np.full((50, 70, 3), color, dtype=np.uint8)
                    _add_frame(store, index, 0, proxy)
                    mask = np.zeros((50, 70), dtype=np.uint8)
                    mask[15:35, 20 + index * 2 : 40 + index * 2] = 255
                    masks.append(mask)
                    _add_mask(
                        store,
                        index,
                        0,
                        12,
                        [20 + index * 2, 15, 40 + index * 2, 35],
                        mask,
                    )
                store.commit()
                TemporalMaskStabilizer(0, 3).process(store)
                second = store.load_mask(store.records_for_frame(1)[0])
                self.assertTrue(np.array_equal(second, masks[1]))

    def test_correction_contour_has_better_iou_than_raw_jump(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                gray = np.full((60, 80), 120, dtype=np.uint8)
                first = np.zeros_like(gray)
                second = np.zeros_like(gray)
                first[20:40, 25:45] = 255
                second[20:40, 28:48] = 255
                for index, mask in enumerate((first, second)):
                    _add_frame(store, index, 0, gray)
                    _add_mask(store, index, 0, 8, [25, 20, 48, 40], mask)
                store.commit()
                raw_iou = aligned_iou(first, second)
                TemporalMaskStabilizer(0, 3).process(store)
                stable_first = store.load_mask(store.records_for_frame(0)[0])
                stable_second = store.load_mask(store.records_for_frame(1)[0])
                self.assertGreater(aligned_iou(stable_first, stable_second), raw_iou)

    def test_empty_mask_and_flow_failure_remain_geometric_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            with TemporalStore(Path(directory)) as store:
                gray = np.full((30, 40), 100, dtype=np.uint8)
                _add_frame(store, 0, 0, gray)
                _add_mask(
                    store,
                    0,
                    0,
                    5,
                    [10, 8, 25, 23],
                    None,
                    fallback=True,
                )
                store.commit()
                TemporalMaskStabilizer(0, 3).process(store)
                record = store.records_for_frame(0)[0]
                self.assertTrue(record.fallback)
                self.assertIsNone(store.load_mask(record))
                self.assertEqual(record.coverage_boxes, ([10, 8, 25, 23],))

    def test_scene_cut_detector_resets_only_on_strong_combined_evidence(self):
        detector = SceneCutDetector(0.55)
        self.assertFalse(detector.update(np.zeros((90, 160), dtype=np.uint8)))
        self.assertTrue(detector.update(np.full((90, 160), 255, dtype=np.uint8)))
        detector.reset()
        checker = (
            (np.indices((90, 160)).sum(axis=0) % 2) * 255
        ).astype(np.uint8)
        self.assertFalse(detector.update(checker))
        # Same histogram, completely different spatial layout.
        self.assertTrue(detector.update(255 - checker))
        detector.reset()
        first_color = np.full((90, 160, 3), (0, 0, 255), dtype=np.uint8)
        second_color = np.full(
            (90, 160, 3), (255, 80, 0), dtype=np.uint8
        )
        self.assertEqual(
            cv2.cvtColor(first_color, cv2.COLOR_BGR2GRAY)[0, 0],
            cv2.cvtColor(second_color, cv2.COLOR_BGR2GRAY)[0, 0],
        )
        self.assertFalse(detector.update(first_color))
        self.assertTrue(detector.update(second_color))

    def test_temporary_storage_has_a_hard_limit_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            with TemporalStore(parent, limit_bytes=1024**2) as probe:
                database_floor = probe.current_bytes
            store = TemporalStore(
                parent, limit_bytes=database_floor + 100
            )
            root = store.root
            with self.assertRaises(TemporalStoreError):
                store.add_frame(
                    0,
                    0,
                    np.random.default_rng(4)
                    .integers(0, 256, (40, 40), dtype=np.uint8),
                    (40, 40),
                )
            store.close()
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
