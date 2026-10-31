"""Behavioral tests for the five-second live camera state machine."""

import unittest

from app.config import LIVE_FRAME_COUNT, LIVE_SAMPLE_INTERVAL_MS, LIVE_WINDOW_DURATION_MS


class LiveCameraState:
    REQUIRED_CONSECUTIVE_HAZARDS = 2

    def __init__(self):
        self.revision = 0
        self.active = False
        self.warming_up = False
        self.frames = []
        self.consecutive_hazards = 0

    def start(self):
        self.revision += 1
        self.active = True
        self.warming_up = True
        self.frames = []
        return self.revision

    def finish_warmup(self, revision):
        if self.active and revision == self.revision:
            self.warming_up = False

    def sample(self, revision, frame):
        if not self.active or self.warming_up or revision != self.revision:
            return None
        self.frames.append(frame)
        if len(self.frames) != LIVE_FRAME_COUNT:
            return None
        complete = self.frames
        self.frames = []
        return complete

    def apply_prediction(self, status):
        self.consecutive_hazards = (
            self.consecutive_hazards + 1 if status == "Hazard" else 0
        )
        return (
            "Hazard"
            if self.consecutive_hazards >= self.REQUIRED_CONSECUTIVE_HAZARDS
            else "Safe"
        )

    def stop(self):
        self.active = False
        self.revision += 1
        self.frames = []
        self.consecutive_hazards = 0


class WarmupAndSmoothingTests(unittest.TestCase):
    def setUp(self):
        self.camera = LiveCameraState()

    def test_warmup_frames_are_not_collected(self):
        revision = self.camera.start()
        self.assertIsNone(self.camera.sample(revision, "dark-frame"))
        self.assertEqual(self.camera.frames, [])
        self.camera.finish_warmup(revision)
        self.camera.sample(revision, "stable-frame")
        self.assertEqual(self.camera.frames, ["stable-frame"])

    def test_window_contains_exactly_sixteen_frames(self):
        revision = self.camera.start()
        self.camera.finish_warmup(revision)
        complete = None
        for index in range(LIVE_FRAME_COUNT):
            complete = self.camera.sample(revision, f"frame-{index}")
        self.assertEqual(len(complete), LIVE_FRAME_COUNT)
        self.assertEqual(self.camera.frames, [])

    def test_sampling_schedule_starts_at_zero_and_ends_at_five_seconds(self):
        timestamps = [index * LIVE_SAMPLE_INTERVAL_MS for index in range(LIVE_FRAME_COUNT)]
        self.assertEqual(timestamps[0], 0)
        self.assertAlmostEqual(timestamps[-1], LIVE_WINDOW_DURATION_MS)

    def test_two_consecutive_hazards_are_required(self):
        self.assertEqual(self.camera.apply_prediction("Hazard"), "Safe")
        self.assertEqual(self.camera.apply_prediction("Hazard"), "Hazard")
        self.assertEqual(self.camera.apply_prediction("Safe"), "Safe")
        self.assertEqual(self.camera.consecutive_hazards, 0)

    def test_stop_invalidates_old_revision_and_clears_frames(self):
        revision = self.camera.start()
        self.camera.finish_warmup(revision)
        self.camera.sample(revision, "frame")
        self.camera.stop()
        self.assertIsNone(self.camera.sample(revision, "stale-frame"))
        self.assertEqual(self.camera.frames, [])


if __name__ == "__main__":
    unittest.main()
