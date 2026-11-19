"""Utilities for overlapping-window temporal analysis of uploaded videos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


WINDOW_SECONDS = 5.0
COARSE_STRIDE_SECONDS = 4.0
FRAME_COUNT = 16
REMOTE_MAX_HEIGHT = 360
REMOTE_JPEG_QUALITY = 92


@dataclass(frozen=True)
class VideoInfo:
    duration_seconds: float
    fps: float
    frame_count: int


def probe_video(video_path: str) -> VideoInfo:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Khong the mo video de doc thoi luong.")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or frame_count <= 0:
            raise ValueError("Video khong co FPS hoac frame hop le.")
        duration = frame_count / fps
        if duration <= 0:
            raise ValueError("Thoi luong video khong hop le.")
        return VideoInfo(round(duration, 3), fps, frame_count)
    finally:
        cap.release()


def _deduplicate(values: Iterable[float]) -> list[float]:
    return sorted({round(max(0.0, value), 3) for value in values})


def build_window_starts(
    duration_seconds: float,
    stride_seconds: float,
    window_seconds: float = WINDOW_SECONDS,
) -> list[float]:
    """Cover the whole video and always include a window ending at EOF."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if stride_seconds <= 0 or window_seconds <= 0:
        raise ValueError("window and stride must be positive")
    if duration_seconds <= window_seconds:
        return [0.0]

    last_start = duration_seconds - window_seconds
    starts = list(np.arange(0.0, last_start + 1e-9, stride_seconds))
    starts.append(last_start)
    return _deduplicate(starts)


def sample_window(
    video_path: str,
    start_seconds: float,
    duration_seconds: float,
    frame_count: int = FRAME_COUNT,
    window_seconds: float = WINDOW_SECONDS,
) -> tuple[list[np.ndarray], float]:
    """Read uniformly spaced RGB frames from one timestamped window."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Khong the mo video de lay frame.")
    end_seconds = min(duration_seconds, start_seconds + window_seconds)
    sample_end = max(start_seconds, end_seconds - 0.001)
    timestamps = np.linspace(start_seconds, sample_end, frame_count)
    frames: list[np.ndarray] = []
    try:
        for timestamp in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                if frames:
                    frames.append(frames[-1].copy())
                    continue
                raise ValueError(f"Khong doc duoc frame tai giay {timestamp:.2f}.")
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames, round(end_seconds, 3)


def encode_remote_frames(frames: list[np.ndarray]) -> list[tuple[str, bytes, str]]:
    """Encode sampled RGB frames as compact JPEGs for the Kaggle frames API."""
    encoded: list[tuple[str, bytes, str]] = []
    for index, frame_rgb in enumerate(frames):
        height, width = frame_rgb.shape[:2]
        if height > REMOTE_MAX_HEIGHT:
            scale = REMOTE_MAX_HEIGHT / height
            frame_rgb = cv2.resize(
                frame_rgb,
                (max(1, round(width * scale)), REMOTE_MAX_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ok, buffer = cv2.imencode(
            ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, REMOTE_JPEG_QUALITY]
        )
        if not ok:
            raise ValueError(f"Khong the ma hoa frame thu {index + 1}.")
        encoded.append((f"frame_{index:02d}.jpg", buffer.tobytes(), "image/jpeg"))
    return encoded


def build_events(windows: list[dict]) -> list[dict]:
    """Merge adjacent positive windows and expose the highest-confidence timestamp."""
    hazards = sorted(
        (item for item in windows if str(item.get("status", "")).lower() == "hazard"),
        key=lambda item: float(item["start_time"]),
    )
    if not hazards:
        return []

    groups: list[list[dict]] = []
    for item in hazards:
        if not groups:
            groups.append([item])
            continue
        previous = groups[-1][-1]
        if float(item["start_time"]) <= float(previous["end_time"]):
            groups[-1].append(item)
        else:
            groups.append([item])

    events: list[dict] = []
    for event_id, group in enumerate(groups, start=1):
        peak = max(group, key=lambda item: float(item.get("hazard_confidence", 0.0)))
        actions: dict[str, float] = {}
        for window in group:
            for action in window.get("detected_actions", []):
                name = str(action.get("action", "Unknown"))
                actions[name] = max(actions.get(name, 0.0), float(action.get("confidence", 0.0)))
        events.append(
            {
                "event_id": event_id,
                "start_time": round(min(float(item["start_time"]) for item in group), 3),
                "end_time": round(max(float(item["end_time"]) for item in group), 3),
                "peak_time": round(
                    (float(peak["start_time"]) + float(peak["end_time"])) / 2.0, 3
                ),
                "hazard_confidence": round(float(peak["hazard_confidence"]), 4),
                "detected_actions": [
                    {"action": name, "confidence": round(confidence, 4)}
                    for name, confidence in sorted(actions.items(), key=lambda pair: pair[1], reverse=True)
                ],
                "window_count": len(group),
            }
        )
    return events
