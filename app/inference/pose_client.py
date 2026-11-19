"""HTTP client for uploaded-video analysis by the local YOLO Pose service."""

from pathlib import Path

import httpx


class PoseVideoClient:
    def __init__(self, base_url: str, timeout_seconds: float = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(timeout_seconds, connect=5.0)

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    async def analyze_video(self, file_path: str, filename: str) -> dict:
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            ) as client:
                with Path(file_path).open("rb") as video:
                    response = await client.post(
                        "/analyze/video",
                        files={"file": (filename, video, "application/octet-stream")},
                    )
        except (httpx.RequestError, OSError) as error:
            return {
                "available": False,
                "status": "Unavailable",
                "events": [],
                "error": str(error),
            }

        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            if response.status_code == 404:
                detail = (
                    "The running pose service is outdated and does not provide "
                    "POST /analyze/video. Stop port 8001 and restart py run.py."
                )
            return {
                "available": False,
                "status": "Unavailable",
                "events": [],
                "error": f"Pose service HTTP {response.status_code}: {detail}",
            }

        try:
            payload = response.json()
        except ValueError:
            return {
                "available": False,
                "status": "Unavailable",
                "events": [],
                "error": "Pose service returned invalid JSON.",
            }
        payload["available"] = True
        return payload
