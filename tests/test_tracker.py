import unittest

import cv2
import numpy as np

from blurface.tracker import FaceTracker


def tracker(**overrides):
    options = {
        "smooth": 0.7,
        "flow_enabled": False,
        "min_face_w": 1,
        "min_face_h": 1,
        "max_face_h_ratio": 1.0,
    }
    options.update(overrides)
    return FaceTracker(**options)


class TrackerPrivacyTests(unittest.TestCase):
    def test_detection_confidence_survives_short_term_prediction(self):
        subject = tracker()
        detected = subject.update(
            np.array([[10, 10, 30, 30, 0.82]], dtype=float), 100, 100
        )[0]
        self.assertAlmostEqual(detected.confidence, 0.82)
        predicted = subject.update(np.empty((0, 5)), 100, 100)[0]
        self.assertAlmostEqual(predicted.confidence, 0.82)

    def test_current_detection_has_its_own_coverage_region(self):
        subject = tracker()
        subject.update(np.array([[0, 0, 30, 30]], dtype=float), 200, 200)
        result = subject.update(np.array([[90, 0, 120, 30]], dtype=float), 200, 200)[0]
        self.assertEqual(result.observed_box, [90, 0, 120, 30])
        self.assertIn(result.observed_box, result.coverage_boxes)
        self.assertEqual(len(result.coverage_boxes), 2)

    def test_close_up_face_is_not_filtered_by_default(self):
        subject = FaceTracker(flow_enabled=False)
        results = subject.update(
            np.array([[100, 100, 700, 700]], dtype=float), 1080, 1920
        )
        self.assertEqual(len(results), 1)

    def test_tiny_detection_is_filtered_by_default(self):
        subject = FaceTracker(flow_enabled=False)
        results = subject.update(
            np.array([[10, 10, 25, 25]], dtype=float), 1080, 1920
        )
        self.assertEqual(results, [])

    def test_assignment_does_not_depend_on_detection_order(self):
        initial = np.array([[0, 0, 20, 20], [80, 0, 100, 20]], dtype=float)
        following = np.array([[5, 0, 25, 20], [40, 0, 60, 20]], dtype=float)
        mappings = []
        for detections in (following, following[::-1]):
            subject = tracker(smooth=1.0)
            subject.update(initial, 100, 200)
            results = subject.update(detections, 100, 200)
            mappings.append(
                {
                    tuple(result.observed_box): result.track_id
                    for result in results
                    if result.observed_box is not None
                }
            )
        self.assertEqual(mappings[0], mappings[1])

    def test_noisy_velocity_does_not_reject_nearby_detection(self):
        subject = tracker(smooth=1.0)
        subject.update(np.array([[0, 0, 20, 20]], dtype=float), 100, 300)
        subject.tracks[0]["velocity"] = (200.0, 0.0)
        result = subject.update(
            np.array([[5, 0, 25, 20]], dtype=float), 100, 300
        )[0]
        self.assertEqual(result.track_id, 0)
        self.assertEqual(subject.next_id, 1)

    def test_single_frame_false_positive_expires_quickly(self):
        subject = tracker(lost_buffer=180)
        subject.update(np.array([[10, 10, 30, 30]], dtype=float), 100, 100)
        self.assertEqual(subject.active_count, 1)
        subject.update(np.empty((0, 4)), 100, 100)
        self.assertEqual(subject.active_count, 1)
        results = subject.update(np.empty((0, 4)), 100, 100)
        self.assertEqual(results, [])
        self.assertEqual(subject.active_count, 0)

    def test_confirmed_face_keeps_the_full_lost_buffer(self):
        subject = tracker(lost_buffer=3)
        detection = np.array([[10, 10, 30, 30]], dtype=float)
        subject.update(detection, 100, 100)
        subject.update(detection, 100, 100)
        for _ in range(3):
            results = subject.update(np.empty((0, 4)), 100, 100)
            self.assertEqual(len(results), 1)
        results = subject.update(np.empty((0, 4)), 100, 100)
        self.assertEqual(results, [])

    def test_failed_feature_refresh_invalidates_old_flow_state(self):
        textured = np.zeros((120, 120), dtype=np.uint8)
        for x in range(25, 76, 10):
            for y in range(25, 76, 10):
                cv2.circle(textured, (x, y), 2, 255, -1)
        blank = np.zeros_like(textured)
        subject = tracker(
            flow_enabled=True,
            flow_min_confirmations=1,
            smooth=1.0,
        )
        subject.update(np.array([[20, 20, 80, 80]], float), 120, 120, textured)
        self.assertIn("flow_points", subject.tracks[0])
        subject.update(np.array([[22, 20, 82, 80]], float), 120, 120, blank)
        self.assertNotIn("flow_points", subject.tracks[0])
        self.assertNotIn("flow_prev_gray", subject.tracks[0])


if __name__ == "__main__":
    unittest.main()
