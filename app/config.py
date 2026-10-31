"""Runtime configuration for live capture and phone camera pairing."""

import json
import os


LIVE_FRAME_COUNT = 16
LIVE_WINDOW_DURATION_MS = 5_000
LIVE_SAMPLE_INTERVAL_MS = LIVE_WINDOW_DURATION_MS / (LIVE_FRAME_COUNT - 1)
LIVE_DURATION_TOLERANCE_MS = 750

CAMERA_SESSION_TTL_SECONDS = int(os.getenv("CAMERA_SESSION_TTL_SECONDS", "300"))
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
PHONE_CAMERA_HOSTNAME = os.getenv("PHONE_CAMERA_HOSTNAME", "industrial-safety.local").strip().rstrip(".")
ENABLE_MDNS = os.getenv("ENABLE_MDNS", "true").lower() in {"1", "true", "yes", "on"}
APP_PORT = int(os.getenv("PORT", "8000"))


def get_ice_servers():
    """Return browser-compatible RTCIceServer dictionaries."""
    raw = os.getenv("RTC_ICE_SERVERS_JSON", "").strip()
    if raw:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("RTC_ICE_SERVERS_JSON must contain a JSON array")
        return parsed

    stun_url = os.getenv("STUN_URL", "stun:stun.l.google.com:19302").strip()
    return [{"urls": [stun_url]}] if stun_url else []
