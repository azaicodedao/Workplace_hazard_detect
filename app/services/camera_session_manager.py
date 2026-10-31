"""In-memory WebRTC signaling sessions for one desktop and one phone."""

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from fastapi import WebSocket

from app.config import CAMERA_SESSION_TTL_SECONDS


@dataclass
class CameraSession:
    session_id: str
    host_token: str
    phone_token: str
    created_at: float
    expires_at: float
    sockets: Dict[str, WebSocket] = field(default_factory=dict)


class CameraSessionManager:
    def __init__(self, ttl_seconds: int = CAMERA_SESSION_TTL_SECONDS):
        self.ttl_seconds = ttl_seconds
        self._sessions: Dict[str, CameraSession] = {}
        self._lock = asyncio.Lock()

    async def create(self) -> CameraSession:
        now = time.time()
        session = CameraSession(
            session_id=secrets.token_urlsafe(12),
            host_token=secrets.token_urlsafe(24),
            phone_token=secrets.token_urlsafe(24),
            created_at=now,
            expires_at=now + self.ttl_seconds,
        )
        async with self._lock:
            await self._remove_expired_locked(now)
            self._sessions[session.session_id] = session
        return session

    async def get(self, session_id: str) -> Optional[CameraSession]:
        async with self._lock:
            await self._remove_expired_locked(time.time())
            return self._sessions.get(session_id)

    async def authenticate(self, session_id: str, role: str, token: str) -> Optional[CameraSession]:
        if role not in {"host", "phone"}:
            return None
        session = await self.get(session_id)
        if not session:
            return None
        expected = session.host_token if role == "host" else session.phone_token
        return session if secrets.compare_digest(expected, token) else None

    async def connect(self, session: CameraSession, role: str, websocket: WebSocket) -> bool:
        async with self._lock:
            old_socket = session.sockets.get(role)
            session.sockets[role] = websocket
            other_socket = session.sockets.get(self._other_role(role))

        if old_socket and old_socket is not websocket:
            try:
                await old_socket.close(code=4001, reason="Replaced by a new connection")
            except Exception:
                pass

        return other_socket is not None

    async def relay(self, session: CameraSession, sender_role: str, message: dict) -> bool:
        other_socket = session.sockets.get(self._other_role(sender_role))
        if not other_socket:
            return False
        payload = dict(message)
        payload["from"] = sender_role
        return await self._safe_send(other_socket, payload)

    async def disconnect(self, session: CameraSession, role: str, websocket: WebSocket) -> None:
        other_socket = None
        async with self._lock:
            if session.sockets.get(role) is websocket:
                session.sockets.pop(role, None)
                other_socket = session.sockets.get(self._other_role(role))
        if other_socket:
            await self._safe_send(other_socket, {"type": "peer_left", "role": role})

    async def close(self, session_id: str, host_token: str) -> bool:
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session or not secrets.compare_digest(session.host_token, host_token):
                return False
            self._sessions.pop(session_id, None)
            sockets = list(session.sockets.values())
        for socket in sockets:
            try:
                await socket.close(code=1000, reason="Session closed")
            except Exception:
                pass
        return True

    async def status(self, session_id: str, host_token: str):
        session = await self.authenticate(session_id, "host", host_token)
        if not session:
            return None
        return {
            "session_id": session.session_id,
            "connected_roles": sorted(session.sockets.keys()),
            "expires_at": session.expires_at,
        }

    async def _remove_expired_locked(self, now: float) -> None:
        expired_ids = [
            session_id
            for session_id, session in self._sessions.items()
            if session.expires_at <= now
        ]
        for session_id in expired_ids:
            session = self._sessions.pop(session_id)
            for socket in session.sockets.values():
                try:
                    await socket.close(code=4002, reason="Session expired")
                except Exception:
                    pass

    @staticmethod
    def _other_role(role: str) -> str:
        return "phone" if role == "host" else "host"

    @staticmethod
    async def _safe_send(websocket: WebSocket, payload: dict) -> bool:
        try:
            await websocket.send_json(payload)
            return True
        except Exception:
            return False


camera_session_manager = CameraSessionManager()
