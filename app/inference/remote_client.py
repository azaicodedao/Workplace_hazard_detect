"""Async client for the VideoMAE inference service running on Kaggle."""

from pathlib import Path
from typing import Any, Iterable

import httpx


class RemoteInferenceError(RuntimeError):
    """A normalized error returned to the local FastAPI proxy."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class RemoteInferenceClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        timeout_seconds: float = 180,
        live_timeout_seconds: float = 20,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = httpx.Timeout(timeout_seconds, connect=15.0)
        self.live_timeout = httpx.Timeout(live_timeout_seconds, connect=10.0)
        self.transport = transport

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.token)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "ngrok-skip-browser-warning": "true",
        }

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=self.timeout,
            follow_redirects=True,
            transport=self.transport,
        )

    @staticmethod
    def _remote_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text.strip() or f"Remote HTTP {response.status_code}"
        if isinstance(payload, dict):
            detail = payload.get("detail") or payload.get("message")
            if detail:
                return str(detail)
        return str(payload)

    @staticmethod
    def _validate_prediction(payload: Any) -> dict:
        required = {"status", "hazard_confidence", "detected_actions"}
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise RemoteInferenceError(
                502,
                "GPU API returned an invalid prediction response.",
            )
        return payload

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        if not self.configured:
            raise RemoteInferenceError(
                503,
                "Remote inference is not configured. Set REMOTE_INFERENCE_URL "
                "and REMOTE_INFERENCE_TOKEN.",
            )
        try:
            async with self._client() as client:
                response = await client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise RemoteInferenceError(504, "GPU API request timed out.") from exc
        except httpx.RequestError as exc:
            raise RemoteInferenceError(
                503,
                f"Cannot connect to GPU API: {exc}",
            ) from exc

        if response.is_error:
            status_code = response.status_code
            if status_code not in {400, 401, 403, 422, 429, 503, 504}:
                status_code = 502
            raise RemoteInferenceError(status_code, self._remote_detail(response))
        return response

    async def health(self) -> dict:
        response = await self._request("GET", "/v1/health")
        try:
            payload = response.json()
        except ValueError as exc:
            raise RemoteInferenceError(502, "GPU API health response is not JSON.") from exc
        if not isinstance(payload, dict):
            raise RemoteInferenceError(502, "GPU API health response is invalid.")
        return payload

    async def predict_video(
        self,
        file_path: str,
        filename: str,
        content_type: str,
    ) -> dict:
        with Path(file_path).open("rb") as video:
            response = await self._request(
                "POST",
                "/v1/predict/video",
                files={"file": (filename, video, content_type)},
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RemoteInferenceError(502, "GPU API prediction is not JSON.") from exc
        return self._validate_prediction(payload)

    async def predict_frames(
        self,
        frames: Iterable[tuple[str, bytes, str]],
        source_type: str,
        captured_duration_ms: int,
        window_id: str,
    ) -> dict:
        files = [
            ("files", (filename, content, content_type))
            for filename, content, content_type in frames
        ]
        response = await self._request(
            "POST",
            "/v1/predict/frames",
            files=files,
            timeout=self.live_timeout,
            data={
                "source_type": source_type,
                "captured_duration_ms": str(captured_duration_ms),
                "window_id": window_id,
            },
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RemoteInferenceError(502, "GPU API prediction is not JSON.") from exc
        return self._validate_prediction(payload)
