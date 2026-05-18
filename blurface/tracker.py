"""
blurface.tracker — Lightweight multi-face tracker with smoothing + prediction.

No external tracker dependencies (no supervision, no ultralytics tracker).
Custom centroid-based matching with exponential smoothing.
When YOLO misses, uses Lucas-Kanade sparse optical flow to shift
the box instead of freezing it.
"""
import cv2
import numpy as np


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

    def __init__(self, lost_buffer: int = 180, smooth: float = 0.7,
                 match_dist: float = 100,
                 min_face_w: int = 30, min_face_h: int = 30,
                 max_face_h_ratio: float = 0.4, max_aspect: float = 3.0,
                 flow_enabled: bool = True,
                 flow_max_points: int = 50,
                 flow_quality: float = 0.01,
                 flow_min_dist: int = 10,
                 flow_min_confirmations: int = 3,
                 flow_max_missed: int = 0):
        self.tracks = {}        # id → state dict
        self.next_id = 0
        self.lost_buffer = lost_buffer    # frames before track is deleted
        self.smooth = smooth
        self.match_dist = match_dist      # px, centroid matching
        self.min_face_w = min_face_w
        self.min_face_h = min_face_h
        self.max_face_h_ratio = max_face_h_ratio  # relative to frame height
        self.max_aspect = max_aspect      # max(w/h, h/w)
        self.flow_enabled = flow_enabled
        self.flow_max_points = flow_max_points
        self.flow_quality = flow_quality
        self.flow_min_dist = flow_min_dist
        self.flow_min_confirmations = flow_min_confirmations
        self.flow_max_missed = flow_max_missed
        self._frame_counter = 0
        # Optical flow statistics
        self.flow_attempts = 0
        self.flow_successes = 0

    def update(self, detections: np.ndarray,
               frame_h: int = 1080, frame_w: int = 1920,
               frame_gray: np.ndarray | None = None):
        """
        detections: (N, 4) array of [x1,y1,x2,y2], or empty.
        frame_gray: optional grayscale frame for optical-flow tracking.

        Returns: list of (box, track_id, missed_frames, is_predicted)
        """
        # ── Step 0: Filter implausible detections ──
        filtered = self._filter_detections(detections, frame_h, frame_w)

        assigned_tracks = set()
        assignments = []       # (det_idx, track_id)

        # ── Step 1: Match detections → existing tracks by centroid distance ──
        if len(filtered) > 0:
            for di, box in enumerate(filtered):
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                best_id, best_dist = None, self.match_dist
                for tid, t in self.tracks.items():
                    if tid in assigned_tracks:
                        continue
                    tcx, tcy = self._center(t['box'])
                    d = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
                    if d < best_dist:
                        best_dist, best_id = d, tid
                if best_id is not None:
                    assigned_tracks.add(best_id)
                    assignments.append((di, best_id))

        # ── Step 2: Smooth matched tracks ──
        for di, tid in assignments:
            det_box = filtered[di].astype(float)
            old_box = np.array(self.tracks[tid]['box'], dtype=float)
            new_box = self.smooth * det_box + (1 - self.smooth) * old_box
            self.tracks[tid]['box'] = new_box.astype(int).tolist()

            # Accumulate confirmation counter, extract flow points when ready
            if self.flow_enabled and frame_gray is not None:
                self.tracks[tid]['confirmed'] = self.tracks[tid].get('confirmed', 0) + 1
                if self.tracks[tid]['confirmed'] >= self.flow_min_confirmations:
                    self._extract_flow_points(tid, frame_gray)

        # ── Step 3: New tracks for unmatched detections ──
        matched_det = {a[0] for a in assignments}
        for di, box in enumerate(filtered):
            if di not in matched_det:
                tid = self.next_id
                b = box.astype(int).tolist()
                self.tracks[tid] = {
                    'box': b,
                    'missed': 0,
                    'confirmed': 1,
                }
                # Only extract flow points if first detection is enough
                if self.flow_enabled and frame_gray is not None \
                        and self.flow_min_confirmations <= 1:
                    self._extract_flow_points(tid, frame_gray)

                assigned_tracks.add(tid)
                self.next_id += 1

        # ── Step 4: Update missed counters ──
        for tid in self.tracks:
            if tid in assigned_tracks:
                self.tracks[tid]['missed'] = 0
            else:
                # Try optical flow before incrementing missed counter
                flow_box = None
                if (self.flow_enabled and frame_gray is not None
                        and self.tracks[tid].get('confirmed', 0) >= self.flow_min_confirmations
                        and (self.flow_max_missed <= 0
                             or self.tracks[tid]['missed'] < self.flow_max_missed)):
                    flow_box = self._optical_flow_track(tid, frame_gray,
                                                        frame_h, frame_w)
                if flow_box is not None:
                    self.tracks[tid]['box'] = flow_box

                self.tracks[tid]['missed'] += 1

        # ── Step 5: Remove dead tracks ──
        dead = [tid for tid, t in self.tracks.items()
                if t['missed'] > self.lost_buffer]
        for tid in dead:
            del self.tracks[tid]

        self._frame_counter += 1

        # ── Return all active tracks ──
        return [(t['box'], tid, t['missed'],
                 t['missed'] > 0)
                for tid, t in self.tracks.items()]

    def _filter_detections(self, detections, frame_h, frame_w):
        """Remove detections that can't be faces (too big / wrong shape)."""
        if len(detections) == 0:
            return detections
        keep = []
        max_h_px = int(frame_h * self.max_face_h_ratio)
        for box in detections:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if w < self.min_face_w or h < self.min_face_h:
                continue
            if h > max_h_px:
                continue
            if max(w, h) / max(min(w, h), 1) > self.max_aspect:
                continue
            keep.append(box)
        return np.array(keep) if keep else np.empty((0, 4))

    @staticmethod
    def _center(box):
        return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

    def _extract_flow_points(self, tid: int, frame_gray: np.ndarray):
        """Extract feature points from the face ROI for optical flow tracking."""
        t = self.tracks[tid]
        x1, y1, x2, y2 = t['box']
        h, w = frame_gray.shape
        margin = 5
        rx1 = max(0, x1 - margin)
        ry1 = max(0, y1 - margin)
        rx2 = min(w, x2 + margin)
        ry2 = min(h, y2 + margin)
        roi = frame_gray[ry1:ry2, rx1:rx2]
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            return

        points = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=self.flow_max_points,
            qualityLevel=self.flow_quality,
            minDistance=self.flow_min_dist,
        )
        if points is None or len(points) == 0:
            return

        t['flow_points'] = points + [rx1, ry1]
        t['flow_prev_gray'] = frame_gray

    def _optical_flow_track(self, tid: int, frame_gray: np.ndarray,
                            frame_h: int, frame_w: int):
        """Track a missed track via Lucas-Kanade optical flow.
        Returns new [x1,y1,x2,y2] box, or None if tracking failed."""
        self.flow_attempts += 1
        t = self.tracks[tid]
        prev_gray = t.get('flow_prev_gray')
        points = t.get('flow_points')
        if prev_gray is None or points is None or len(points) == 0:
            return None

        points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)

        try:
            new_points, status, _err = cv2.calcOpticalFlowPyrLK(
                prev_gray, frame_gray, points, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                          10, 0.03),
            )
        except cv2.error:
            return None

        mask = status.flatten() == 1
        good_new = new_points[mask]
        good_old = points[mask]

        if len(good_new) < 5:
            return None

        gn = good_new.reshape(-1, 2)
        go = good_old.reshape(-1, 2)

        # Keep optical-flow prediction conservative. A face ROI usually needs
        # short-term translation between YOLO corrections; homography can expand
        # boxes badly when a few facial feature points drift.
        dx = np.median(gn[:, 0] - go[:, 0])
        dy = np.median(gn[:, 1] - go[:, 1])

        old_box = np.array(t['box'], dtype=float)
        new_box = old_box + [dx, dy, dx, dy]

        new_box[0] = max(0, new_box[0])
        new_box[1] = max(0, new_box[1])
        new_box[2] = min(frame_w, new_box[2])
        new_box[3] = min(frame_h, new_box[3])

        t['flow_points'] = good_new.reshape(-1, 1, 2)
        t['flow_prev_gray'] = frame_gray

        self.flow_successes += 1
        return new_box.astype(int).tolist()

    @property
    def active_count(self) -> int:
        return len(self.tracks)

    @property
    def predicted_count(self) -> int:
        return sum(1 for t in self.tracks.values() if t['missed'] > 0)

    @property
    def zombie_count(self) -> int:
        """Removed; kept for compatibility. Always returns 0."""
        return 0
