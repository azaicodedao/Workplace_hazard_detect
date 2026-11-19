"""Same-origin WebSocket proxy for the separately managed YOLO Pose service."""

import asyncio
import json

import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import POSE_SERVICE_WS_URL, POSE_TRACKING_ENABLED


router = APIRouter()


@router.websocket("/ws/pose")
async def pose_proxy(websocket: WebSocket):
    await websocket.accept()
    if not POSE_TRACKING_ENABLED:
        await websocket.send_json({"type": "pose_error", "detail": "Pose tracking is disabled."})
        await websocket.close(code=1008)
        return

    try:
        upstream = await websockets.connect(
            POSE_SERVICE_WS_URL,
            max_size=4 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
        )
    except Exception as error:
        await websocket.send_json(
            {
                "type": "pose_error",
                "detail": (
                    "Cannot connect to the YOLO Pose service on port 8001. "
                    f"Start pose_service/run.py first. ({error})"
                ),
            }
        )
        await websocket.close(code=1011)
        return

    async def browser_to_pose():
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                await upstream.send(message["bytes"])
            elif message.get("text") is not None:
                await upstream.send(message["text"])

    async def pose_to_browser():
        async for message in upstream:
            if isinstance(message, bytes):
                await websocket.send_bytes(message)
            else:
                await websocket.send_text(message)

    tasks = {
        asyncio.create_task(browser_to_pose()),
        asyncio.create_task(pose_to_browser()),
    }
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            exception = task.exception()
            if exception and not isinstance(exception, WebSocketDisconnect):
                raise exception
    except (WebSocketDisconnect, websockets.ConnectionClosed):
        pass
    except Exception as error:
        try:
            await websocket.send_text(
                json.dumps({"type": "pose_error", "detail": str(error)})
            )
        except Exception:
            pass
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await upstream.close()

