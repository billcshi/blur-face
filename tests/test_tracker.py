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
    def test_current_detection_is_fully_inside_render_box(self):
        subject = tracker()
        subject.update(np.array([[0, 0, 30, 30]], dtype=float), 200, 200)
        result = subject.update(np.array([[90, 0, 120, 30]], dtype=float), 200, 200)[0]
        self.assertLessEqual(result.box[0], 90)
        self.assertLessEqual(result.box[1], 0)
        self.assertGreaterEqual(result.box[2], 120)
        self.assertGreaterEqual(result.box[3], 30)

    def test_close_up_face_is_not_filtered_by_default(self):
        subject = FaceTracker(flow_enabled=False)
        results = subject.update(
            np.array([[100, 100, 700, 700]], dtype=float), 1080, 1920
        )
        self.assertEqual(len(results), 1)

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
