import unittest
from pathlib import Path

from app.config import (
    LIVE_FRAME_COUNT,
    LIVE_SAMPLE_INTERVAL_MS,
    LIVE_WINDOW_DURATION_MS,
)


class LiveWindowConfigurationTests(unittest.TestCase):
    def test_sixteen_frames_cover_exactly_five_seconds(self):
        timestamps = [index * LIVE_SAMPLE_INTERVAL_MS for index in range(LIVE_FRAME_COUNT)]
        self.assertEqual(len(timestamps), 16)
        self.assertEqual(timestamps[0], 0)
        self.assertAlmostEqual(timestamps[-1], LIVE_WINDOW_DURATION_MS)
        self.assertEqual(LIVE_WINDOW_DURATION_MS, 5000)

    def test_frontend_uses_latest_only_rolling_buffer(self):
        source = Path("frontend/js/app.js").read_text(encoding="utf-8")
        self.assertIn("const LIVE_STRIDE_MS = 4000", source)
        self.assertIn("const LIVE_CAPTURE_MAX_HEIGHT = 360", source)
        self.assertIn("frameBuffer: []", source)
        self.assertIn("if (live.analyzing)", source)
        self.assertNotIn("pendingWindow", source)


if __name__ == "__main__":
    unittest.main()
