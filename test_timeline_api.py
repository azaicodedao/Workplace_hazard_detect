import unittest
import json
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from app.inference.timeline import VideoInfo
from app.main import app


class TimelineApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app, base_url="http://127.0.0.1:8000")

    def test_remote_timeline_returns_timestamped_event(self):
        class FakeRemoteClient:
            configured = True

            async def predict_frames(
                inner_self,
                frames,
                source_type,
                captured_duration_ms,
                window_id,
            ):
                self.assertEqual(len(frames), 16)
                self.assertEqual(captured_duration_ms, 5000)
                return {
                    "status": "Hazard",
                    "hazard_confidence": 0.9,
                    "detected_actions": [
                        {"action": "person falling down", "confidence": 0.85}
                    ],
                }

        fake_frames = [np.zeros((12, 16, 3), dtype=np.uint8) for _ in range(16)]

        def fake_sample(_path, start, duration):
            return fake_frames, min(duration, start + 5.0)

        encoded = [
            (f"frame_{index:02d}.jpg", b"jpeg", "image/jpeg")
            for index in range(16)
        ]
        with (
            patch("app.api.routes.INFERENCE_BACKEND", "remote"),
            patch("app.api.routes.remote_client", FakeRemoteClient()),
            patch("app.api.routes._save_upload"),
            patch("app.api.routes.probe_video", return_value=VideoInfo(6.0, 30.0, 180)),
            patch("app.api.routes.sample_window", side_effect=fake_sample),
            patch("app.api.routes.encode_remote_frames", return_value=encoded),
        ):
            response = self.client.post(
                "/predict/timeline/",
                files={"file": ("fall.mp4", b"fake-video", "video/mp4")},
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["status"], "Hazard")
        self.assertEqual(payload["video_duration"], 6.0)
        self.assertGreaterEqual(len(payload["events"]), 1)
        self.assertIn("peak_time", payload["events"][0])
        self.assertEqual(payload["events"][0]["detected_actions"][0]["action"], "person falling down")

    def test_stream_endpoint_emits_window_before_complete(self):
        class FakeRemoteClient:
            configured = True

            async def predict_frames(self, frames, **_kwargs):
                return {
                    "status": "Safe",
                    "hazard_confidence": 0.12,
                    "detected_actions": [],
                }

        fake_frames = [np.zeros((12, 16, 3), dtype=np.uint8) for _ in range(16)]
        encoded = [
            (f"frame_{index:02d}.jpg", b"jpeg", "image/jpeg")
            for index in range(16)
        ]
        with (
            patch("app.api.routes.INFERENCE_BACKEND", "remote"),
            patch("app.api.routes.remote_client", FakeRemoteClient()),
            patch("app.api.routes._save_upload"),
            patch("app.api.routes.probe_video", return_value=VideoInfo(4.0, 30.0, 120)),
            patch("app.api.routes.sample_window", return_value=(fake_frames, 4.0)),
            patch("app.api.routes.encode_remote_frames", return_value=encoded),
        ):
            response = self.client.post(
                "/predict/timeline/stream",
                files={"file": ("safe.mp4", b"fake-video", "video/mp4")},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("application/x-ndjson", response.headers["content-type"])
        events = [json.loads(line) for line in response.text.splitlines() if line]
        event_types = [event["type"] for event in events]
        self.assertEqual(event_types[0], "started")
        self.assertIn("window", event_types)
        self.assertEqual(event_types[-1], "complete")
        self.assertLess(event_types.index("window"), event_types.index("complete"))

    def test_pose_fall_makes_safe_videomae_upload_hazard(self):
        class FakeRemoteClient:
            configured = True

            async def predict_frames(self, frames, **_kwargs):
                return {
                    "status": "Safe",
                    "hazard_confidence": 0.1,
                    "detected_actions": [],
                }

        class FakePoseClient:
            configured = True

            async def analyze_video(self, _path, _filename):
                return {
                    "available": True,
                    "status": "Hazard",
                    "video_duration": 4.0,
                    "processed_frames": 40,
                    "target_fps": 10.0,
                    "effective_fps": 10.0,
                    "events": [{
                        "event_id": 1,
                        "track_id": 7,
                        "start_time": 1.2,
                        "end_time": 2.1,
                        "peak_time": 1.7,
                        "confidence": 0.88,
                        "final_state": "FALLEN",
                    }],
                }

        fake_frames = [np.zeros((12, 16, 3), dtype=np.uint8) for _ in range(16)]
        encoded = [
            (f"frame_{index:02d}.jpg", b"jpeg", "image/jpeg")
            for index in range(16)
        ]
        with (
            patch("app.api.routes.INFERENCE_BACKEND", "remote"),
            patch("app.api.routes.remote_client", FakeRemoteClient()),
            patch("app.api.routes.pose_video_client", FakePoseClient()),
            patch("app.api.routes.POSE_TRACKING_ENABLED", True),
            patch("app.api.routes._save_upload"),
            patch("app.api.routes.probe_video", return_value=VideoInfo(4.0, 30.0, 120)),
            patch("app.api.routes.sample_window", return_value=(fake_frames, 4.0)),
            patch("app.api.routes.encode_remote_frames", return_value=encoded),
        ):
            response = self.client.post(
                "/predict/timeline/",
                files={"file": ("fall.mp4", b"fake-video", "video/mp4")},
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["status"], "Hazard")
        self.assertEqual(payload["hazard_confidence"], 0.88)
        self.assertEqual(payload["pose_analysis"]["events"][0]["track_id"], 7)
        self.assertEqual(payload["pose_analysis"]["events"][0]["peak_time"], 1.7)


if __name__ == "__main__":
    unittest.main()
