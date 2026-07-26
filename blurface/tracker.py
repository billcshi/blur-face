"""
blurface.tracker — Lightweight multi-face tracker with smoothing + prediction.

No external tracker dependencies (no supervision, no ultralytics tracker).
Global motion-aware assignment with exponential smoothing.
When YOLO misses, uses Lucas-Kanade sparse optical flow to shift
the box instead of freezing it.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class TrackResult:
    """Tracking state and the regions that must be covered for one face."""

    box: list[int]
    track_id: int
    missed: int
    is_predicted: bool
    observed_box: list[int] | None = None
    confidence: float = 1.0

    def __iter__(self):
        # Preserve the original tuple-unpacking API.
        return iter((self.box, self.track_id, self.missed, self.is_predicted))

    @property
    def coverage_boxes(self) -> tuple[list[int], ...]:
        """Keep observed and smoothed regions separate for elliptical masks."""
        if self.observed_box is None or self.observed_box == self.box:
            return (self.box,)
        return self.box, self.observed_box


def _linear_sum_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    """Return a minimum-cost rectangular assignment using Hungarian matching."""
    if cost.size == 0:
        return []
    matrix = np.asarray(cost, dtype=float)
    transposed = matrix.shape[0] > matrix.shape[1]
    if transposed:
        matrix = matrix.T
    rows, cols = matrix.shape
    u = np.zeros(rows + 1)
    v = np.zeros(cols + 1)
    p = np.zeros(cols + 1, dtype=int)
    way = np.zeros(cols + 1, dtype=int)
    for row in range(1, rows + 1):
        p[0] = row
        col0 = 0
        min_values = np.full(cols + 1, np.inf)
        used = np.zeros(cols + 1, dtype=bool)
        while True:
            used[col0] = True
            row0 = p[col0]
            delta = np.inf
            col1 = 0
            for col in range(1, cols + 1):
                if used[col]:
                    continue
                current = matrix[row0 - 1, col - 1] - u[row0] - v[col]
                if current < min_values[col]:
                    min_values[col] = current
                    way[col] = col0
                if min_values[col] < delta:
                    delta = min_values[col]
                    col1 = col
            for col in range(cols + 1):
                if used[col]:
                    u[p[col]] += delta
                    v[col] -= delta
                else:
                    min_values[col] -= delta
            col0 = col1
            if p[col0] == 0:
                break
        while True:
            col1 = way[col0]
            p[col0] = p[col1]
            col0 = col1
            if col0 == 0:
                break
    pairs = [(p[col] - 1, col - 1) for col in range(1, cols + 1) if p[col]]
    if transposed:
        return [(col, row) for row, col in pairs]
    return pairs


class FaceTracker:
    """
    Tracks multiple faces across frames.

    - Assigns unique persistent IDs.
    - Smooths bounding box coordinates (exponential moving average).
    - Filters implausible detections by size and aspect ratio.
    - Optical-flow tracking when YOLO misses (with confirmation gating).
    - Drops tracks after lost_buffer consecutive misses.

    Usage:
        tracker = FaceTracker(lost_buffer=180, smooth=0.7)
        tracks = tracker.update(boxes, frame_h=1080, frame_w=1920)
        for box, tid, missed, is_predicted in tracks:
            ...
    """

    def __init__(
        self,
        lost_buffer: int = 180,
        smooth: float = 0.7,
        match_dist: float = 100,
        min_face_w: int = 30,
        min_face_h: int = 30,
        max_face_h_ratio: float = 1.0,
        max_aspect: float = 10.0,
        flow_enabled: bool = True,
        flow_max_points: int = 50,
        flow_quality: float = 0.01,
        flow_min_dist: int = 10,
        flow_min_confirmations: int = 3,
        flow_max_missed: int = 0,
        tentative_buffer: int = 1,
    ):
        self.tracks = {}  # id → state dict
        self.next_id = 0
        self.lost_buffer = lost_buffer  # frames before track is deleted
        self.smooth = smooth
        self.match_dist = match_dist  # px, centroid matching
        self.min_face_w = min_face_w
        self.min_face_h = min_face_h
        self.max_face_h_ratio = max_face_h_ratio  # relative to frame height
        self.max_aspect = max_aspect  # max(w/h, h/w)
        self.flow_enabled = flow_enabled
        self.flow_max_points = flow_max_points
        self.flow_quality = flow_quality
        self.flow_min_dist = flow_min_dist
        self.flow_min_confirmations = flow_min_confirmations
        self.flow_max_missed = flow_max_missed
        self.tentative_buffer = tentative_buffer
        self._frame_counter = 0
        # Optical flow statistics
        self.flow_attempts = 0
        self.flow_successes = 0

    def update(
        self,
        detections: np.ndarray,
        frame_h: int = 1080,
        frame_w: int = 1920,
        frame_gray: np.ndarray | None = None,
    ):
        """
        detections: (N, 4) boxes or (N, 5) rows ending in confidence.
        frame_gray: optional grayscale frame for optical-flow tracking.

        Returns: list of (box, track_id, missed_frames, is_predicted)
        """
        # ── Step 0: Filter implausible detections ──
        filtered, detection_confidences = self._filter_detections(
            detections, frame_h, frame_w
        )

        assigned_tracks = set()
        assignments: list[tuple[int, int]] = []
        for state in self.tracks.values():
            state["observed_box"] = None

        # ── Step 1: Globally match detections to motion-predicted tracks ──
        if len(filtered) > 0 and self.tracks:
            track_ids = sorted(self.tracks)
            real_cost = np.full((len(filtered), len(track_ids)), 2.0)
            for det_index, box in enumerate(filtered):
                center = np.asarray(self._center(box))
                for track_index, track_id in enumerate(track_ids):
                    state = self.tracks[track_id]
                    track_box = np.asarray(state["box"], dtype=float)
                    last_observed = np.asarray(
                        state.get("last_observed_box", track_box), dtype=float
                    )
                    velocity = np.asarray(state.get("velocity", (0.0, 0.0)))
                    predicted = np.asarray(self._center(last_observed)) + velocity
                    current_distance = float(
                        np.linalg.norm(center - np.asarray(self._center(track_box)))
                    )
                    predicted_distance = float(np.linalg.norm(center - predicted))
                    # A noisy velocity estimate must never make a nearby
                    # detection harder to match than the current position.
                    distance = min(current_distance, predicted_distance)
                    diagonal = float(
                        np.hypot(
                            track_box[2] - track_box[0],
                            track_box[3] - track_box[1],
                        )
                    )
                    gate = max(self.match_dist, diagonal * 1.5)
                    if distance <= gate:
                        iou = max(
                            self._iou(box, track_box),
                            self._iou(box, last_observed),
                        )
                        real_cost[det_index, track_index] = (
                            0.65 * distance / gate + 0.35 * (1.0 - iou)
                        )
            # One dummy column per detection allows every detection to remain
            # unmatched when all real candidates are outside their gates.
            cost = np.concatenate(
                [real_cost, np.ones((len(filtered), len(filtered)))], axis=1
            )
            for det_index, column in _linear_sum_assignment(cost):
                if column < len(track_ids) and real_cost[det_index, column] < 1.0:
                    track_id = track_ids[column]
                    assigned_tracks.add(track_id)
                    assignments.append((det_index, track_id))

        # ── Step 2: Smooth matched tracks ──
        for di, tid in assignments:
            det_box = filtered[di].astype(float)
            state = self.tracks[tid]
            old_box = np.array(state["box"], dtype=float)
            previous_observed = np.asarray(
                state.get("last_observed_box", old_box), dtype=float
            )
            measured_velocity = np.asarray(self._center(det_box)) - np.asarray(
                self._center(previous_observed)
            )
            old_velocity = np.asarray(state.get("velocity", (0.0, 0.0)))
            stable_velocity = 0.6 * old_velocity + 0.4 * measured_velocity
            new_box = self.smooth * det_box + (1 - self.smooth) * old_box
            state["box"] = new_box.astype(int).tolist()
            state["observed_box"] = det_box.astype(int).tolist()
            state["last_observed_box"] = det_box.astype(int).tolist()
            state["velocity"] = tuple(stable_velocity.tolist())
            state["confidence"] = float(detection_confidences[di])

            # Accumulate confirmation counter, extract flow points when ready
            if self.flow_enabled and frame_gray is not None:
                state["confirmed"] = state.get("confirmed", 0) + 1
                if state["confirmed"] >= self.flow_min_confirmations:
                    self._extract_flow_points(tid, frame_gray, det_box)
            else:
                state["confirmed"] = state.get("confirmed", 0) + 1

        # ── Step 3: New tracks for unmatched detections ──
        matched_det = {a[0] for a in assignments}
        for di, box in enumerate(filtered):
            if di not in matched_det:
                tid = self.next_id
                b = box.astype(int).tolist()
                self.tracks[tid] = {
                    "box": b,
                    "missed": 0,
                    "confirmed": 1,
                    "observed_box": b,
                    "last_observed_box": b,
                    "velocity": (0.0, 0.0),
                    "confidence": float(detection_confidences[di]),
                }
                # Only extract flow points if first detection is enough
                if (
                    self.flow_enabled
                    and frame_gray is not None
                    and self.flow_min_confirmations <= 1
                ):
                    self._extract_flow_points(tid, frame_gray, b)

                assigned_tracks.add(tid)
                self.next_id += 1

        # ── Step 4: Update missed counters ──
        for tid in self.tracks:
            if tid in assigned_tracks:
                self.tracks[tid]["missed"] = 0
            else:
                # Try optical flow before incrementing missed counter
                flow_box = None
                if (
                    self.flow_enabled
                    and frame_gray is not None
                    and self.tracks[tid].get("confirmed", 0)
                    >= self.flow_min_confirmations
                    and (
                        self.flow_max_missed <= 0
                        or self.tracks[tid]["missed"] < self.flow_max_missed
                    )
                ):
                    flow_box = self._optical_flow_track(
                        tid, frame_gray, frame_h, frame_w
                    )
                if flow_box is not None:
                    self.tracks[tid]["box"] = flow_box

                self.tracks[tid]["missed"] += 1

        # ── Step 5: Remove dead tracks ──
        dead = [
            tid
            for tid, state in self.tracks.items()
            if state["missed"] > self.lost_buffer
            or (
                state.get("confirmed", 1) < 2
                and state["missed"] > self.tentative_buffer
            )
        ]
        for tid in dead:
            del self.tracks[tid]

        self._frame_counter += 1

        # ── Return all active tracks ──
        results = []
        for tid, state in self.tracks.items():
            observed = state.get("observed_box")
            results.append(
                TrackResult(
                    list(state["box"]),
                    tid,
                    state["missed"],
                    state["missed"] > 0,
                    observed,
                    float(state.get("confidence", 1.0)),
                )
            )
        return results

    def _filter_detections(self, detections, frame_h, frame_w):
        """Remove detections that can't be faces (too big / wrong shape)."""
        if len(detections) == 0:
            return np.empty((0, 4)), np.empty(0)
        keep = []
        confidences = []
        max_h_px = int(frame_h * self.max_face_h_ratio)
        for detection in detections:
            if len(detection) < 4 or not np.all(np.isfinite(detection[:4])):
                continue
            raw_x1, raw_y1, raw_x2, raw_y2 = detection[:4]
            x1, x2 = sorted((raw_x1, raw_x2))
            y1, y2 = sorted((raw_y1, raw_y2))
            x1, x2 = max(0, x1), min(frame_w, x2)
            y1, y2 = max(0, y1), min(frame_h, y2)
            w, h = x2 - x1, y2 - y1
            if w < self.min_face_w or h < self.min_face_h:
                continue
            if h > max_h_px:
                continue
            if max(w, h) / max(min(w, h), 1) > self.max_aspect:
                continue
            keep.append(np.array([x1, y1, x2, y2], dtype=float))
            confidence = (
                float(detection[4])
                if len(detection) >= 5 and np.isfinite(detection[4])
                else 1.0
            )
            confidences.append(max(0.0, min(1.0, confidence)))
        if not keep:
            return np.empty((0, 4)), np.empty(0)
        return np.array(keep), np.array(confidences)

    @staticmethod
    def _center(box):
        return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

    @staticmethod
    def _iou(first, second) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    def _extract_flow_points(
        self, tid: int, frame_gray: np.ndarray, box=None
    ):
        """Extract feature points from the face ROI for optical flow tracking."""
        t = self.tracks[tid]
        x1, y1, x2, y2 = box if box is not None else t["box"]
        h, w = frame_gray.shape
        margin = 5
        rx1 = max(0, int(np.floor(x1 - margin)))
        ry1 = max(0, int(np.floor(y1 - margin)))
        rx2 = min(w, int(np.ceil(x2 + margin)))
        ry2 = min(h, int(np.ceil(y2 + margin)))
        roi = frame_gray[ry1:ry2, rx1:rx2]
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            self._clear_flow_state(t)
            return

        points = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=self.flow_max_points,
            qualityLevel=self.flow_quality,
            minDistance=self.flow_min_dist,
        )
        if points is None or len(points) == 0:
            self._clear_flow_state(t)
            return

        t["flow_points"] = points + [rx1, ry1]
        t["flow_prev_gray"] = frame_gray
        t["flow_frame"] = self._frame_counter

    def _optical_flow_track(
        self, tid: int, frame_gray: np.ndarray, frame_h: int, frame_w: int
    ):
        """Track a missed track via Lucas-Kanade optical flow.
        Returns new [x1,y1,x2,y2] box, or None if tracking failed."""
        t = self.tracks[tid]
        prev_gray = t.get("flow_prev_gray")
        points = t.get("flow_points")
        if prev_gray is None or points is None or len(points) == 0:
            return None
        self.flow_attempts += 1

        points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)

        try:
            new_points, status, _err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                frame_gray,
                points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            )
        except cv2.error:
            self._clear_flow_state(t)
            return None

        if new_points is None or status is None:
            self._clear_flow_state(t)
            return None
        flattened_new = new_points.reshape(-1, 2)
        mask = (status.flatten() == 1) & np.all(np.isfinite(flattened_new), axis=1)
        good_new = new_points[mask]
        good_old = points[mask]

        if len(good_new) < 5:
            self._clear_flow_state(t)
            return None

        gn = good_new.reshape(-1, 2)
        go = good_old.reshape(-1, 2)

        # Keep optical-flow prediction conservative. A face ROI usually needs
        # short-term translation between YOLO corrections; homography can expand
        # boxes badly when a few facial feature points drift.
        dx = np.median(gn[:, 0] - go[:, 0])
        dy = np.median(gn[:, 1] - go[:, 1])

        old_box = np.array(t["box"], dtype=float)
        new_box = old_box + [dx, dy, dx, dy]

        new_box[0] = max(0, new_box[0])
        new_box[1] = max(0, new_box[1])
        new_box[2] = min(frame_w, new_box[2])
        new_box[3] = min(frame_h, new_box[3])

        t["flow_points"] = good_new.reshape(-1, 1, 2)
        t["flow_prev_gray"] = frame_gray
        t["flow_frame"] = self._frame_counter

        self.flow_successes += 1
        return new_box.astype(int).tolist()

    @staticmethod
    def _clear_flow_state(state) -> None:
        state.pop("flow_points", None)
        state.pop("flow_prev_gray", None)
        state.pop("flow_frame", None)

    @property
    def active_count(self) -> int:
        return len(self.tracks)

    @property
    def predicted_count(self) -> int:
        return sum(1 for t in self.tracks.values() if t["missed"] > 0)

    @property
    def zombie_count(self) -> int:
        """Removed; kept for compatibility. Always returns 0."""
        return 0
