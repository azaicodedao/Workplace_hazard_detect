from pydantic import BaseModel
from typing import List, Optional

class ActionItem(BaseModel):
    action: str
    confidence: float

class PredictionResponse(BaseModel):
    status: str
    hazard_confidence: float
    detected_actions: List[ActionItem]


class TimelineWindow(BaseModel):
    window_id: str
    phase: str
    start_time: float
    end_time: float
    status: str
    hazard_confidence: float
    detected_actions: List[ActionItem]


class TimelineEvent(BaseModel):
    event_id: int
    start_time: float
    end_time: float
    peak_time: float
    hazard_confidence: float
    detected_actions: List[ActionItem]
    window_count: int


class TimelineScanInfo(BaseModel):
    window_seconds: float
    coarse_stride_seconds: float
    coarse_windows: int
    total_inferences: int


class PoseFallEvent(BaseModel):
    event_id: int
    track_id: int
    start_time: float
    end_time: float
    peak_time: float
    confidence: float
    final_state: str


class PoseVideoAnalysis(BaseModel):
    available: bool
    status: str
    video_duration: float = 0.0
    processed_frames: int = 0
    target_fps: float = 0.0
    effective_fps: float = 0.0
    events: List[PoseFallEvent]
    error: Optional[str] = None


class TimelinePredictionResponse(BaseModel):
    status: str
    hazard_confidence: float
    video_duration: float
    events: List[TimelineEvent]
    windows: List[TimelineWindow]
    scan: TimelineScanInfo
    pose_analysis: PoseVideoAnalysis
