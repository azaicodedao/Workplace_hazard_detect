import unittest
from io import BytesIO
from unittest.mock import patch

import httpx

from app.inference.remote_client import RemoteInferenceClient, RemoteInferenceError


PREDICTION = {
    "status": "Safe",
    "hazard_confidence": 0.1,
    "detected_actions": [],
}


class RemoteInferenceClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_health_sends_bearer_token(self):
        async def handler(request):
            self.assertEqual(request.url.path, "/v1/health")
            self.assertEqual(request.headers["authorization"], "Bearer secret")
            self.assertEqual(request.headers["ngrok-skip-browser-warning"], "true")
            return httpx.Response(200, json={"status": "ready", "device": "cuda"})

        client = RemoteInferenceClient(
            "https://gpu.example",
            "secret",
            transport=httpx.MockTransport(handler),
        )
        result = await client.health()
        self.assertEqual(result["device"], "cuda")

    async def test_video_is_forwarded_as_multipart_file(self):
        async def handler(request):
            self.assertEqual(request.url.path, "/v1/predict/video")
            body = await request.aread()
            self.assertIn(b'name="file"', body)
            self.assertIn(b'filename="sample.mp4"', body)
            self.assertIn(b"video-bytes", body)
            return httpx.Response(200, json=PREDICTION)

        client = RemoteInferenceClient(
            "https://gpu.example",
            "secret",
            transport=httpx.MockTransport(handler),
        )

        with patch(
            "app.inference.remote_client.Path.open",
            return_value=BytesIO(b"video-bytes"),
        ):
            result = await client.predict_video(
                "sample.mp4",
                "sample.mp4",
                "video/mp4",
            )
        self.assertEqual(result, PREDICTION)

    async def test_sixteen_frames_are_forwarded_without_reencoding(self):
        async def handler(request):
            self.assertEqual(request.url.path, "/v1/predict/frames")
            self.assertEqual(request.extensions["timeout"]["read"], 20)
            body = await request.aread()
            self.assertEqual(body.count(b'name="files"'), 16)
            self.assertNotIn(b'name="frames"', body)
            self.assertIn(b"frame-content-15", body)
            self.assertIn(b'name="source_type"', body)
            return httpx.Response(200, json=PREDICTION)

        client = RemoteInferenceClient(
            "https://gpu.example",
            "secret",
            transport=httpx.MockTransport(handler),
        )
        frames = [
            (f"frame_{index}.jpg", f"frame-content-{index}".encode(), "image/jpeg")
            for index in range(16)
        ]
        result = await client.predict_frames(
            frames,
            source_type="computer_camera",
            captured_duration_ms=5000,
            window_id="7",
        )
        self.assertEqual(result, PREDICTION)

    async def test_live_timeout_becomes_gateway_timeout(self):
        async def handler(request):
            raise httpx.ReadTimeout("slow GPU API", request=request)

        client = RemoteInferenceClient(
            "https://gpu.example",
            "secret",
            live_timeout_seconds=20,
            transport=httpx.MockTransport(handler),
        )
        frames = [
            (f"frame_{index}.jpg", b"frame", "image/jpeg")
            for index in range(16)
        ]

        with self.assertRaises(RemoteInferenceError) as context:
            await client.predict_frames(
                frames,
                source_type="computer_camera",
                captured_duration_ms=5000,
                window_id="timeout-window",
            )

        self.assertEqual(context.exception.status_code, 504)

    async def test_remote_busy_status_is_preserved(self):
        async def handler(_request):
            return httpx.Response(429, json={"detail": "GPU busy"})

        client = RemoteInferenceClient(
            "https://gpu.example",
            "secret",
            transport=httpx.MockTransport(handler),
        )
        with self.assertRaises(RemoteInferenceError) as context:
            await client.health()
        self.assertEqual(context.exception.status_code, 429)
        self.assertEqual(context.exception.detail, "GPU busy")


if __name__ == "__main__":
    unittest.main()
