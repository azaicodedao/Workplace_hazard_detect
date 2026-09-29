from pydantic import BaseModel
from typing import List

class ActionItem(BaseModel):
    action: str
    confidence: float

class PredictionResponse(BaseModel):
    status: str
    hazard_confidence: float
    detected_actions: List[ActionItem]
