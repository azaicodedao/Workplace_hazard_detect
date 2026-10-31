import unittest
from unittest.mock import patch

import cv2
import numpy as np

from fastapi.testclient import TestClient

from app.main import app


class CameraSessionApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app, base_url="http://127.0.0.1:8000")

    def test_create_qr_status_websocket_and_close(self):
        created = self.client.post("/api/camera-sessions")
        self.assertEqual(created.status_code, 200)
        session = created.json()
        self.assertIn("/static/phone.html?", session["join_url"])
        self.assertNotIn("127.0.0.1", session["join_url"])
        self.assertNotIn("localhost", session["join_url"])
        self.assertEqual(session["address_mode"], "auto-lan-ip")
        self.assertEqual(session["fallback_join_url"], session["join_url"])

        qr = self.client.get(session["qr_url"])
        self.assertEqual(qr.status_code, 200)
        self.assertEqual(qr.headers["content-type"], "image/png")

        status = self.client.get(
            f"/api/camera-sessions/{session['session_id']}",
            params={"host_token": session["host_token"]},
        )
        self.assertEqual(status.status_code, 200)

        path = (
            f"/api/camera-sessions/ws/{session['session_id']}"
            f"?role=host&token={session['host_token']}"
        )
        with self.client.websocket_connect(path) as websocket:
            connected = websocket.receive_json()
            self.assertEqual(connected["type"], "connected")
            self.assertEqual(connected["role"], "host")

        closed = self.client.delete(
            f"/api/camera-sessions/{session['session_id']}",
            params={"host_token": session["host_token"]},
        )
        self.assertEqual(closed.status_code, 204)

    def test_live_endpoint_accepts_sixteen_frames_from_five_second_window(self):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        encoded, buffer = cv2.imencode(".jpg", image)
        self.assertTrue(encoded)
        files = [
            ("frames", (f"frame_{index}.jpg", buffer.tobytes(), "image/jpeg"))
            for index in range(16)
        ]

        class FakePredictor:
            @staticmethod
            def predict_from_frames(frames):
                self.assertEqual(len(frames), 16)
                return {
                    "status": "Safe",
                    "hazard_confidence": 0.1,
                    "detected_actions": [],
                }

        with patch("app.api.routes.predictor", FakePredictor()):
            response = self.client.post(
                "/predict/live/",
                files=files,
                data={
                    "source_type": "phone_camera",
                    "captured_duration_ms": "5000",
                    "window_id": "1",
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "Safe")


if __name__ == "__main__":
    unittest.main()
