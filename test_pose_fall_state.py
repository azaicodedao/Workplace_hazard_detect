import unittest

import numpy as np

from pose_service.fall_state import TemporalFallClassifier


def keypoints(shoulders, hips):
    points = np.zeros((17, 3), dtype=np.float32)
    points[5] = [shoulders[0] - 5, shoulders[1], 0.9]
    points[6] = [shoulders[0] + 5, shoulders[1], 0.9]
    points[11] = [hips[0] - 5, hips[1], 0.9]
    points[12] = [hips[0] + 5, hips[1], 0.9]
    return points


class TemporalFallClassifierTests(unittest.TestCase):
    def test_vertical_person_remains_standing(self):
        classifier = TemporalFallClassifier()
        result = classifier.update(
            1,
            np.array([0, 0, 100, 200]),
            keypoints((50, 50), (50, 110)),
            now=0.0,
        )
        self.assertEqual(result["state"], "STANDING")
        self.assertLess(result["torso_angle"], 10)

    def test_rapid_horizontal_transition_reaches_fallen(self):
        classifier = TemporalFallClassifier()
        box = np.array([0, 0, 100, 200])
        classifier.update(7, box, keypoints((50, 40), (50, 70)), now=0.0)
        falling = classifier.update(7, box, keypoints((20, 105), (80, 125)), now=0.2)
        self.assertEqual(falling["state"], "FALLING")
        fallen = classifier.update(7, box, keypoints((20, 105), (80, 125)), now=0.8)
        self.assertEqual(fallen["state"], "FALLEN")

    def test_stale_track_is_removed(self):
        classifier = TemporalFallClassifier()
        classifier.update(
            3,
            np.array([0, 0, 100, 200]),
            keypoints((50, 50), (50, 110)),
            now=1.0,
        )
        classifier.expire(now=4.0, max_age_seconds=2.0)
        self.assertNotIn(3, classifier.tracks)


if __name__ == "__main__":
    unittest.main()
