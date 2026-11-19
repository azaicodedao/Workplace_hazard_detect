"""FastAPI WebSocket service for real-time YOLO Pose tracking."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from time import monotonic

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from ultralytics import YOLO

from pose_service.fall_state import TemporalFallClassifier


load_dotenv()

MODEL_NAME = os.getenv("POSE_MODEL", "yolo26n-pose.pt")
TRACKER = os.getenv("POSE_TRACKER", "bytetrack.yaml")
DEVICE = os.getenv("POSE_DEVICE", "0" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = int(os.getenv("POSE_IMAGE_SIZE", "640"))
CONFIDENCE = float(os.getenv("POSE_CONFIDENCE", "0.35"))
VIDEO_TARGET_FPS = float(os.getenv("POSE_VIDEO_FPS", "10"))

app = FastAPI(title="YOLO Pose Fall Tracking Service")
model = YOLO(MODEL_NAME)
inference_lock = asyncio.Lock()


def _reset_tracker() -> None:
    model.predictor = None


def _track_frame(
    frame: np.ndarray,
    classifier: TemporalFallClassifier,
    now: float | None = None,
) -> dict:
    result = model.track(
        frame,
        persist=True,
        tracker=TRACKER,
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        verbose=False,
    )[0]
    people = []
    if result.boxes is not None and result.keypoints is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        keypoints = result.keypoints.data.detach().cpu().numpy()
        if result.boxes.id is None:
            track_ids = np.arange(len(boxes), dtype=int)
        else:
            track_ids = result.boxes.id.detach().cpu().numpy().astype(int)

        now = monotonic() if now is None else now
        for track_id, box, points in zip(track_ids, boxes, keypoints):
            metrics = classifier.update(int(track_id), box, points, now=now)
            people.append(
                {
                    "track_id": int(track_id),
                    "box": [round(float(value), 1) for value in box],
                    "keypoints": [
                        [
                            round(float(point[0]), 1),
                            round(float(point[1]), 1),
                            round(float(point[2]), 3),
                        ]
                        for point in points
                    ],
                    **metrics,
                }
            )
        classifier.expire(now)

    height, width = frame.shape[:2]
    return {
        "type": "pose_result",
        "frame_width": width,
        "frame_height": height,
        "device": DEVICE,
        "people": people,
    }


def _process_frame(jpeg_bytes: bytes, classifier: TemporalFallClassifier) -> dict:
    encoded = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid JPEG frame.")
    return _track_frame(frame, classifier)


def _analyze_video_path(video_path: str) -> dict:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("Could not open uploaded video for pose analysis.")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        capture.release()
        raise ValueError("Uploaded video has invalid FPS metadata.")
    duration = frame_count / fps if frame_count > 0 else 0.0
    frame_step = max(1, round(fps / max(1.0, VIDEO_TARGET_FPS)))
    effective_fps = fps / frame_step
    classifier = TemporalFallClassifier()
    active_events: dict[int, dict] = {}
    finished_events: list[dict] = []
    processed_frames = 0
    frame_index = 0

    def finish_event(track_id: int):
        event = active_events.pop(track_id, None)
        if event is not None:
            finished_events.append(event)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            current_index = frame_index
            frame_index += 1
            if current_index % frame_step != 0:
                continue

            timestamp = current_index / fps
            payload = _track_frame(frame, classifier, now=timestamp)
            processed_frames += 1
            visible_ids = set()
            for person in payload["people"]:
                track_id = int(person["track_id"])
                visible_ids.add(track_id)
                state = str(person["state"])
                if state in {"FALLING", "FALLEN"}:
                    event = active_events.setdefault(
                        track_id,
                        {
                            "track_id": track_id,
                            "start_time": round(timestamp, 3),
                            "end_time": round(timestamp, 3),
                            "peak_time": round(timestamp, 3),
                            "confidence": float(person["fall_score"]),
                            "final_state": state,
                            "last_seen": timestamp,
                        },
                    )
                    event["end_time"] = round(timestamp, 3)
                    event["last_seen"] = timestamp
                    event["final_state"] = state
                    if float(person["fall_score"]) >= float(event["confidence"]):
                        event["confidence"] = float(person["fall_score"])
                        event["peak_time"] = round(timestamp, 3)
                else:
                    finish_event(track_id)

            for track_id, event in list(active_events.items()):
                if track_id not in visible_ids and timestamp - float(event["last_seen"]) > 1.0:
                    finish_event(track_id)
    finally:
        capture.release()

    for track_id in list(active_events):
        finish_event(track_id)
    finished_events.sort(key=lambda item: float(item["start_time"]))
    for index, event in enumerate(finished_events, start=1):
        event.pop("last_seen", None)
        event["event_id"] = index
        event["confidence"] = round(float(event["confidence"]), 4)

    return {
        "status": "Hazard" if finished_events else "Safe",
        "video_duration": round(duration, 3),
        "processed_frames": processed_frames,
        "target_fps": VIDEO_TARGET_FPS,
        "effective_fps": round(effective_fps, 2),
        "events": finished_events,
    }


@app.get("/health")
async def health():
    return {
        "status": "ready",
        "api_version": "2.0.0",
        "model": MODEL_NAME,
        "device": DEVICE,
        "tracker": TRACKER,
        "capabilities": ["live_pose", "upload_video_fall_detection"],
    }


@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov"}:
        suffix = ".mp4"
    descriptor, temporary_path = tempfile.mkstemp(prefix="pose-upload-", suffix=suffix)
    os.close(descriptor)
    try:
        try:
            await asyncio.to_thread(_copy_upload, file.file, temporary_path)
        finally:
            await file.close()
        async with inference_lock:
            _reset_tracker()
            return await asyncio.to_thread(_analyze_video_path, temporary_path)
    finally:
        Path(temporary_path).unlink(missing_ok=True)


def _copy_upload(source, target_path: str) -> None:
    with open(target_path, "wb") as target:
        shutil.copyfileobj(source, target)


@app.websocket("/ws/pose")
async def pose_websocket(websocket: WebSocket):
    await websocket.accept()
    classifier = TemporalFallClassifier()
    async with inference_lock:
        _reset_tracker()
        await websocket.send_text(
            json.dumps(
                {
                    "type": "pose_ready",
                    "model": MODEL_NAME,
                    "device": DEVICE,
                    "tracker": TRACKER,
                }
            )
        )
        try:
            while True:
                frame_bytes = await websocket.receive_bytes()
                try:
                    payload = await asyncio.to_thread(_process_frame, frame_bytes, classifier)
                    await websocket.send_text(json.dumps(payload))
                except ValueError as error:
                    await websocket.send_text(
                        json.dumps({"type": "pose_error", "detail": str(error)})
                    )
        except WebSocketDisconnect:
            pass

