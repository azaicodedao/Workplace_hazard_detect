from typing import Any, Dict, List

from pydantic import BaseModel


class CameraSessionResponse(BaseModel):
    session_id: str
    host_token: str
    join_url: str
    fallback_join_url: str
    address_mode: str
    qr_url: str
    expires_at: float
    ice_servers: List[Dict[str, Any]]


class CameraSessionStatus(BaseModel):
    session_id: str
    connected_roles: List[str]
    expires_at: float
