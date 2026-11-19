"""Per-person temporal fall-state estimation from tracked COCO keypoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, degrees
from time import monotonic

import numpy as np


LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _joint_center(keypoints: np.ndarray, indices: tuple[int, int]) -> np.ndarray | None:
    valid = []
    for index in indices:
        point = keypoints[index]
        confidence = float(point[2]) if point.shape[0] > 2 else 1.0
        if confidence >= 0.25 and float(point[0]) > 0 and float(point[1]) > 0:
            valid.append(point[:2])
    if not valid:
        return None
    return np.mean(valid, axis=0)


@dataclass
class TrackState:
    state: str = "STANDING"
    last_seen: float = field(default_factory=monotonic)
    last_hip_y: float | None = None
    last_update: float | None = None
    falling_since: float | None = None
    low_posture_since: float | None = None


class TemporalFallClassifier:
    """Explainable state machine; thresholds must be validated per camera view."""

    def __init__(self):
        self.tracks: dict[int, TrackState] = {}

    def reset(self) -> None:
        self.tracks.clear()

    def expire(self, now: float | None = None, max_age_seconds: float = 2.0) -> None:
        now = monotonic() if now is None else now
        stale = [
            track_id
            for track_id, item in self.tracks.items()
            if now - item.last_seen > max_age_seconds
        ]
        for track_id in stale:
            self.tracks.pop(track_id, None)

    def update(
        self,
        track_id: int,
        box: np.ndarray,
        keypoints: np.ndarray,
        now: float | None = None,
    ) -> dict:
        now = monotonic() if now is None else now
        track = self.tracks.setdefault(track_id, TrackState(last_seen=now))
        x1, y1, x2, y2 = [float(value) for value in box]
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        aspect_ratio = width / height

        shoulders = _joint_center(keypoints, (LEFT_SHOULDER, RIGHT_SHOULDER))
        hips = _joint_center(keypoints, (LEFT_HIP, RIGHT_HIP))
        torso_angle = 0.0
        normalized_hip_y = None
        if shoulders is not None and hips is not None:
            delta = hips - shoulders
            torso_angle = degrees(
                atan2(abs(float(delta[0])), max(1e-6, abs(float(delta[1]))))
            )
            # Use image-space movement measured in current body heights. Measuring
            # relative to y1 would hide the downward motion because the box moves too.
            normalized_hip_y = float(hips[1]) / height

        downward_velocity = 0.0
        if (
            normalized_hip_y is not None
            and track.last_hip_y is not None
            and track.last_update is not None
        ):
            elapsed = max(1e-3, now - track.last_update)
            downward_velocity = (normalized_hip_y - track.last_hip_y) / elapsed

        if normalized_hip_y is not None:
            track.last_hip_y = normalized_hip_y
        track.last_update = now
        track.last_seen = now

        horizontal_score = _clamp((torso_angle - 35.0) / 45.0)
        aspect_score = _clamp((aspect_ratio - 0.55) / 0.65)
        velocity_score = _clamp(downward_velocity / 0.9)
        fall_score = 0.45 * velocity_score + 0.30 * horizontal_score + 0.25 * aspect_score
        low_posture = torso_angle >= 60.0 or aspect_ratio >= 0.90
        rapid_drop = downward_velocity >= 0.30

        if low_posture:
            if track.low_posture_since is None:
                track.low_posture_since = now
        else:
            track.low_posture_since = None

        if rapid_drop and low_posture and fall_score >= 0.52:
            if track.falling_since is None:
                track.falling_since = now
            track.state = "FALLING"

        falling_duration = now - track.falling_since if track.falling_since is not None else 0.0
        low_duration = now - track.low_posture_since if track.low_posture_since is not None else 0.0
        if low_posture and (falling_duration >= 0.45 or low_duration >= 1.20):
            track.state = "FALLEN"
        elif not low_posture and downward_velocity < 0.15:
            track.state = "STANDING"
            track.falling_since = None

        # The state machine can confirm FALLEN from sustained posture even when
        # instantaneous downward velocity has already returned to zero. Keep the
        # reported confidence consistent with the temporal decision.
        if track.state == "FALLEN":
            fall_score = max(fall_score, 0.65)
        elif track.state == "FALLING":
            fall_score = max(fall_score, 0.52)

        return {
            "state": track.state,
            "fall_score": round(_clamp(fall_score), 4),
            "torso_angle": round(torso_angle, 1),
            "aspect_ratio": round(aspect_ratio, 3),
            "downward_velocity": round(downward_velocity, 3),
        }

