"""REST and WebSocket endpoints used to pair a phone camera with the dashboard."""

import asyncio
import io
from urllib.parse import urlencode

import qrcode
from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from app.config import PHONE_CAMERA_HOSTNAME, PUBLIC_BASE_URL, get_ice_servers
from app.schemas.camera_session import CameraSessionResponse, CameraSessionStatus
from app.services.camera_session_manager import camera_session_manager
from app.services.mdns_service import mdns_service
from app.services.network_address import get_lan_ipv4, is_loopback_host, replace_url_host


router = APIRouter(prefix="/api/camera-sessions", tags=["camera sessions"])
ALLOWED_SIGNAL_TYPES = {"offer", "answer", "ice_candidate", "ready", "ping"}


def _public_base_urls(request: Request):
    """Return the preferred and direct-LAN origins for a phone pairing link."""
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL, PUBLIC_BASE_URL, "configured"

    request_base = str(request.base_url).rstrip("/")
    request_host = request.url.hostname or ""
    # Resolve on every new session so a Wi-Fi/DHCP change needs no restart.
    lan_ip = get_lan_ipv4()
    lan_base = replace_url_host(request_base, lan_ip)

    # HTTPS certificates should target a stable hostname rather than a DHCP IP.
    if request.url.scheme == "https" and mdns_service.running:
        mdns_base = replace_url_host(request_base, PHONE_CAMERA_HOSTNAME)
        return mdns_base, lan_base, "mdns"

    if not is_loopback_host(request_host):
        return request_base, lan_base, "request-host"
    return lan_base, lan_base, "auto-lan-ip"


def _phone_join_urls(request: Request, session):
    primary_base, fallback_base, address_mode = _public_base_urls(request)
    query = urlencode({"session": session.session_id, "token": session.phone_token})
    return (
        f"{primary_base}/static/phone.html?{query}",
        f"{fallback_base}/static/phone.html?{query}",
        address_mode,
    )


@router.post("", response_model=CameraSessionResponse)
async def create_camera_session(request: Request):
    await asyncio.to_thread(mdns_service.refresh_network)
    session = await camera_session_manager.create()
    join_url, fallback_join_url, address_mode = _phone_join_urls(request, session)
    qr_url = (
        f"/api/camera-sessions/{session.session_id}/qr?"
        + urlencode({"host_token": session.host_token})
    )
    return CameraSessionResponse(
        session_id=session.session_id,
        host_token=session.host_token,
        join_url=join_url,
        fallback_join_url=fallback_join_url,
        address_mode=address_mode,
        qr_url=qr_url,
        expires_at=session.expires_at,
        ice_servers=get_ice_servers(),
    )


@router.get("/network-info")
async def get_phone_camera_network_info():
    return {
        "lan_ip": mdns_service.lan_ip or get_lan_ipv4(),
        "mdns_hostname": PHONE_CAMERA_HOSTNAME,
        "mdns_running": mdns_service.running,
        "mdns_error": mdns_service.last_error,
        "public_base_url_override": bool(PUBLIC_BASE_URL),
    }


@router.get("/{session_id}", response_model=CameraSessionStatus)
async def get_camera_session(session_id: str, host_token: str = Query(...)):
    status = await camera_session_manager.status(session_id, host_token)
    if not status:
        raise HTTPException(status_code=404, detail="Camera session not found or expired")
    return status


@router.delete("/{session_id}", status_code=204)
async def close_camera_session(session_id: str, host_token: str = Query(...)):
    closed = await camera_session_manager.close(session_id, host_token)
    if not closed:
        raise HTTPException(status_code=404, detail="Camera session not found or expired")


@router.get("/{session_id}/qr")
async def get_camera_session_qr(session_id: str, host_token: str, request: Request):
    await asyncio.to_thread(mdns_service.refresh_network)
    session = await camera_session_manager.authenticate(session_id, "host", host_token)
    if not session:
        raise HTTPException(status_code=404, detail="Camera session not found or expired")
    join_url, _, _ = _phone_join_urls(request, session)
    image = qrcode.make(join_url)
    output = io.BytesIO()
    image.save(output, format="PNG")
    output.seek(0)
    return StreamingResponse(output, media_type="image/png")


@router.websocket("/ws/{session_id}")
async def camera_signaling_websocket(
    websocket: WebSocket,
    session_id: str,
    role: str = Query(...),
    token: str = Query(...),
):
    session = await camera_session_manager.authenticate(session_id, role, token)
    if not session:
        await websocket.close(code=1008, reason="Invalid or expired camera session")
        return

    await websocket.accept()
    peer_connected = await camera_session_manager.connect(session, role, websocket)
    await websocket.send_json({
        "type": "connected",
        "role": role,
        "session_id": session.session_id,
        "ice_servers": get_ice_servers(),
    })
    if peer_connected:
        await websocket.send_json({
            "type": "peer_joined",
            "role": "phone" if role == "host" else "host",
        })
        await camera_session_manager.relay(
            session,
            role,
            {"type": "peer_joined", "role": role},
        )

    try:
        while True:
            message = await websocket.receive_json()
            message_type = message.get("type")
            if message_type not in ALLOWED_SIGNAL_TYPES:
                await websocket.send_json({"type": "error", "message": "Unsupported signal type"})
                continue
            if message_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            delivered = await camera_session_manager.relay(session, role, message)
            if not delivered and message_type != "ready":
                await websocket.send_json({"type": "peer_unavailable"})
    except WebSocketDisconnect:
        pass
    finally:
        await camera_session_manager.disconnect(session, role, websocket)
