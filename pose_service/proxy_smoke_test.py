"""Exercise the main app's same-origin pose proxy against port 8001."""

import json

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app


def main():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Could not encode proxy smoke-test frame")

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        with client.websocket_connect("/ws/pose") as socket:
            ready = socket.receive_json()
            print(json.dumps(ready))
            socket.send_bytes(encoded.tobytes())
            result = socket.receive_json()
            print(json.dumps(result))
            if result.get("type") != "pose_result":
                raise RuntimeError(f"Unexpected proxy response: {result}")


if __name__ == "__main__":
    main()

