import unittest

from app.inference.timeline import (
    COARSE_STRIDE_SECONDS,
    WINDOW_SECONDS,
    build_events,
    build_window_starts,
)


def prediction(start, end, status="Safe", confidence=0.1, actions=None):
    return {
        "window_id": f"coarse-{start}",
        "phase": "coarse",
        "start_time": start,
        "end_time": end,
        "status": status,
        "hazard_confidence": confidence,
        "detected_actions": actions or [],
    }


class VideoTimelineTests(unittest.TestCase):
    def test_coarse_windows_cover_video_end(self):
        self.assertEqual(WINDOW_SECONDS, 5.0)
        self.assertEqual(COARSE_STRIDE_SECONDS, 4.0)
        self.assertEqual(
            build_window_starts(12.0, COARSE_STRIDE_SECONDS),
            [0.0, 4.0, 7.0],
        )
        self.assertEqual(build_window_starts(3.0, COARSE_STRIDE_SECONDS), [0.0])

    def test_adjacent_hazard_windows_become_one_event(self):
        windows = [
            prediction(
                4.0,
                9.0,
                "Hazard",
                0.75,
                [{"action": "person falling down", "confidence": 0.7}],
            ),
            prediction(
                4.5,
                9.5,
                "Hazard",
                0.92,
                [{"action": "person falling down", "confidence": 0.88}],
            ),
            prediction(10.0, 15.0),
        ]
        events = build_events(windows)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["start_time"], 4.0)
        self.assertEqual(events[0]["end_time"], 9.5)
        self.assertEqual(events[0]["peak_time"], 7.0)
        self.assertEqual(events[0]["hazard_confidence"], 0.92)
        self.assertEqual(events[0]["detected_actions"][0]["confidence"], 0.88)

    def test_safe_windows_produce_no_events(self):
        self.assertEqual(build_events([prediction(0.0, 5.0)]), [])


if __name__ == "__main__":
    unittest.main()
