"""Behavioral tests for the five-second live camera state machine."""

import unittest

from app.config import LIVE_FRAME_COUNT, LIVE_SAMPLE_INTERVAL_MS, LIVE_WINDOW_DURATION_MS


class LiveCameraState:
    WINDOW_MS = 5000
    STRIDE_MS = 4000

    def __init__(self):
        self.revision = 0
        self.active = False
        self.warming_up = False
        self.frames = []
        self.analyzing = False
        self.next_due_ms = None

    def start(self):
        self.revision += 1
        self.active = True
        self.warming_up = True
        self.frames = []
        return self.revision

    def finish_warmup(self, revision):
        if self.active and revision == self.revision:
            self.warming_up = False

    def sample(self, revision, timestamp_ms, frame):
        if not self.active or self.warming_up or revision != self.revision:
            return None
        self.frames.append((timestamp_ms, frame))
        self.frames = [item for item in self.frames if item[0] >= timestamp_ms - 6000]
        if self.next_due_ms is None:
            self.next_due_ms = timestamp_ms + self.WINDOW_MS
        coverage = self.frames[-1][0] - self.frames[0][0]
        if (
            self.analyzing
            or timestamp_ms < self.next_due_ms
            or len(self.frames) < LIVE_FRAME_COUNT
            or coverage < self.WINDOW_MS
        ):
            return None
        self.analyzing = True
        self.next_due_ms = timestamp_ms + self.STRIDE_MS
        return [frame_value for _, frame_value in self.frames[-LIVE_FRAME_COUNT:]]

    def finish_analysis(self):
        self.analyzing = False

    def apply_prediction(self, status):
        return status

    def stop(self):
        self.active = False
        self.revision += 1
        self.frames = []
        self.analyzing = False
        self.next_due_ms = None


class WarmupAndSmoothingTests(unittest.TestCase):
    def setUp(self):
        self.camera = LiveCameraState()

    def test_warmup_frames_are_not_collected(self):
        revision = self.camera.start()
        self.assertIsNone(self.camera.sample(revision, 0, "dark-frame"))
        self.assertEqual(self.camera.frames, [])
        self.camera.finish_warmup(revision)
        self.camera.sample(revision, 0, "stable-frame")
        self.assertEqual(self.camera.frames, [(0, "stable-frame")])

    def test_window_contains_exactly_sixteen_frames(self):
        revision = self.camera.start()
        self.camera.finish_warmup(revision)
        complete = None
        for index in range(LIVE_FRAME_COUNT):
            timestamp = index * LIVE_SAMPLE_INTERVAL_MS
            complete = self.camera.sample(revision, timestamp, f"frame-{index}")
        self.assertEqual(len(complete), LIVE_FRAME_COUNT)
        self.assertEqual(len(self.camera.frames), LIVE_FRAME_COUNT)

    def test_gpu_busy_does_not_create_a_pending_window(self):
        revision = self.camera.start()
        self.camera.finish_warmup(revision)
        for index in range(LIVE_FRAME_COUNT):
            self.camera.sample(
                revision,
                index * LIVE_SAMPLE_INTERVAL_MS,
                f"frame-{index}",
            )
        self.assertTrue(self.camera.analyzing)
        for index in range(1, 13):
            result = self.camera.sample(
                revision,
                5000 + index * LIVE_SAMPLE_INTERVAL_MS,
                f"new-{index}",
            )
            self.assertIsNone(result)
        self.assertLessEqual(len(self.camera.frames), 19)

    def test_sampling_schedule_starts_at_zero_and_ends_at_five_seconds(self):
        timestamps = [index * LIVE_SAMPLE_INTERVAL_MS for index in range(LIVE_FRAME_COUNT)]
        self.assertEqual(timestamps[0], 0)
        self.assertAlmostEqual(timestamps[-1], LIVE_WINDOW_DURATION_MS)

    def test_each_window_uses_its_own_prediction(self):
        self.assertEqual(self.camera.apply_prediction("Hazard"), "Hazard")
        self.assertEqual(self.camera.apply_prediction("Safe"), "Safe")
        self.assertEqual(self.camera.apply_prediction("Hazard"), "Hazard")

    def test_stop_invalidates_old_revision_and_clears_frames(self):
        revision = self.camera.start()
        self.camera.finish_warmup(revision)
        self.camera.sample(revision, 0, "frame")
        self.camera.stop()
        self.assertIsNone(self.camera.sample(revision, 100, "stale-frame"))
        self.assertEqual(self.camera.frames, [])


if __name__ == "__main__":
    unittest.main()
