"""Send one synthetic JPEG through the running pose WebSocket service."""

import asyncio
import json

import cv2
import numpy as np
import websockets


async def main():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Could not encode smoke-test frame")

    async with websockets.connect("ws://127.0.0.1:8001/ws/pose") as socket:
        ready = json.loads(await socket.recv())
        print(json.dumps(ready))
        await socket.send(encoded.tobytes())
        result = json.loads(await socket.recv())
        print(json.dumps(result))
        if result.get("type") != "pose_result":
            raise RuntimeError(f"Unexpected response: {result}")


if __name__ == "__main__":
    asyncio.run(main())

