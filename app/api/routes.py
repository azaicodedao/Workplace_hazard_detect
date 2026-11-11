"""FastAPI inference routes with selectable local or remote GPU execution."""

import asyncio
import shutil
import threading
import traceback
import uuid
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.inference.remote_client import RemoteInferenceClient, RemoteInferenceError
from app.schemas.prediction import PredictionResponse
from app.config import (
    INFERENCE_BACKEND,
    LIVE_DURATION_TOLERANCE_MS,
    LIVE_FRAME_COUNT,
    LIVE_WINDOW_DURATION_MS,
    REMOTE_INFERENCE_TIMEOUT_SECONDS,
    REMOTE_INFERENCE_TOKEN,
    REMOTE_INFERENCE_URL,
    REMOTE_LIVE_TIMEOUT_SECONDS,
)

router = APIRouter()

# Only import/load PyTorch and VideoMAE when local inference is selected.
predictor = None
predictor_startup_error = None
if INFERENCE_BACKEND == "local":
    try:
        from app.inference.predictor import VideoPredictor

        predictor = VideoPredictor()
    except Exception as _e:
        predictor_startup_error = str(_e)
        print(f"Loi khoi tao mo hinh AI: {_e}")

remote_client = RemoteInferenceClient(
    base_url=REMOTE_INFERENCE_URL,
    token=REMOTE_INFERENCE_TOKEN,
    timeout_seconds=REMOTE_INFERENCE_TIMEOUT_SECONDS,
    live_timeout_seconds=REMOTE_LIVE_TIMEOUT_SECONDS,
)

print(f"Inference backend: {INFERENCE_BACKEND}")

# ---------------------------------------------------------------------------
# Inference lock -- threading.Lock vi inference chay trong worker thread
# ---------------------------------------------------------------------------
inference_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Thu muc luu video tam
# ---------------------------------------------------------------------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Live camera config (de thay doi de dang)
# ---------------------------------------------------------------------------
# Backward-compatible name used by existing tests and documentation.
LIVE_NUM_FRAMES: int = LIVE_FRAME_COUNT


# ---------------------------------------------------------------------------
# Worker functions -- chay trong thread pool, KHONG block event loop
# ---------------------------------------------------------------------------

def _run_video_inference(file_path: str) -> dict:
    """
    Chay inference cho video upload trong worker thread.
    Acquire lock voi blocking=True -- co the cho neu model dang ban.
    Giai phong lock trong finally block.
    """
    acquired = inference_lock.acquire(blocking=True)
    try:
        return predictor.predict(file_path)
    finally:
        if acquired:
            inference_lock.release()


def _run_live_inference(frames: List[np.ndarray]) -> dict:
    """
    Chay inference cho live camera trong worker thread.
    Thu acquire lock voi blocking=False. Neu that bai, raise RuntimeError("busy")
    de route handler tra 429.
    Giai phong lock trong finally block.
    """
    acquired = inference_lock.acquire(blocking=False)
    if not acquired:
        raise RuntimeError("busy")
    try:
        return predictor.predict_from_frames(frames)
    finally:
        inference_lock.release()


def _ensure_backend_ready() -> None:
    if INFERENCE_BACKEND == "local" and predictor is None:
        detail = "Mo hinh AI local chua duoc nap thanh cong."
        if predictor_startup_error:
            detail += f" {predictor_startup_error}"
        raise HTTPException(status_code=503, detail=detail)
    if INFERENCE_BACKEND == "remote" and not remote_client.configured:
        raise HTTPException(
            status_code=503,
            detail=(
                "GPU API chua duoc cau hinh. Hay dat REMOTE_INFERENCE_URL "
                "va REMOTE_INFERENCE_TOKEN trong file .env."
            ),
        )


def _raise_remote_error(error: RemoteInferenceError) -> None:
    raise HTTPException(status_code=error.status_code, detail=error.detail) from error


def _save_upload(source, target: Path) -> None:
    with target.open("wb") as buffer:
        shutil.copyfileobj(source, buffer)


# ---------------------------------------------------------------------------
# Helper: decode anh tu bytes -> RGB numpy array
# ---------------------------------------------------------------------------

def _decode_image_to_rgb(data: bytes, index: int) -> np.ndarray:
    """
    Giai ma bytes cua mot anh thanh numpy array RGB (H, W, 3) uint8.
    Raise ValueError neu du lieu khong hop le hoac khong giai ma duoc.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(
            f"Khong the giai ma anh thu {index + 1}. "
            "Dam bao la file anh hop le (JPEG, PNG, BMP, ...)."
        )
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


# ---------------------------------------------------------------------------
# GET /api/inference/health -- local view of the selected inference backend
# ---------------------------------------------------------------------------

@router.get("/api/inference/health")
async def inference_health():
    if INFERENCE_BACKEND == "local":
        return {
            "status": "ready" if predictor is not None else "unavailable",
            "backend": "local",
            "model_loaded": predictor is not None,
            "error": predictor_startup_error,
        }

    _ensure_backend_ready()
    try:
        remote_health = await remote_client.health()
    except RemoteInferenceError as error:
        _raise_remote_error(error)
    return {
        "status": remote_health.get("status", "ready"),
        "backend": "remote",
        "remote_url": REMOTE_INFERENCE_URL,
        "remote": remote_health,
    }


# ---------------------------------------------------------------------------
# POST /predict/ -- VIDEO UPLOAD
# ---------------------------------------------------------------------------

@router.post("/predict/", response_model=PredictionResponse)
async def predict_video(file: UploadFile = File(...)):
    """Analyze an uploaded video locally or forward it to the Kaggle GPU API."""
    _ensure_backend_ready()

    # Kiem tra dinh dang file
    allowed_extensions = [".mp4", ".avi", ".mov"]
    original_filename = Path(file.filename or "video.mp4").name
    ext = Path(original_filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Chi ho tro cac dinh dang video: {allowed_extensions}",
        )

    file_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    try:
        await asyncio.to_thread(_save_upload, file.file, file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Khong the luu video: {e}")
    finally:
        await file.close()

    try:
        if INFERENCE_BACKEND == "remote":
            result = await remote_client.predict_video(
                str(file_path),
                original_filename,
                file.content_type or "application/octet-stream",
            )
        else:
            result = await asyncio.to_thread(_run_video_inference, str(file_path))
    except RemoteInferenceError as error:
        _raise_remote_error(error)
    except Exception as e:
        print("========== INFERENCE ERROR ==========")
        traceback.print_exc()
        print("=====================================")
        raise HTTPException(
            status_code=500,
            detail=f"Loi trong qua trinh phan tich video: {str(e)}",
        )

    finally:
        try:
            file_path.unlink(missing_ok=True)
        except Exception as remove_err:
            print(f"Canh bao: Chua the xoa file tam {file_path}: {remove_err}")

    return result


# ---------------------------------------------------------------------------
# POST /predict/live/   -- LIVE CAMERA (endpoint moi)
# ---------------------------------------------------------------------------

@router.post("/predict/live/", response_model=PredictionResponse)
async def predict_live(
    frames: List[UploadFile] = File(...),
    source_type: str = Form("computer_camera"),
    captured_duration_ms: int = Form(LIVE_WINDOW_DURATION_MS),
    window_id: str = Form(""),
):
    """
    Phan tich 16 frame tu live camera.

    Request format:
        multipart/form-data
        field name: frames (multi-value)
        content:    16 file anh (JPEG/PNG/BMP), theo thu tu thoi gian

    Response: PredictionResponse (cung schema voi /predict/)

    HTTP 422: thieu hoac sai so luong frames, anh khong giai ma duoc
    HTTP 429: model dang chay inference cho request khac
    HTTP 500: loi he thong
    """
    _ensure_backend_ready()

    if source_type not in {"computer_camera", "phone_camera"}:
        raise HTTPException(
            status_code=422,
            detail="source_type phai la computer_camera hoac phone_camera.",
        )

    if abs(captured_duration_ms - LIVE_WINDOW_DURATION_MS) > LIVE_DURATION_TOLERANCE_MS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Cua so live phai dai {LIVE_WINDOW_DURATION_MS}ms "
                f"(+/- {LIVE_DURATION_TOLERANCE_MS}ms), nhan duoc {captured_duration_ms}ms."
            ),
        )

    # --- Kiem tra so luong frames ---
    if len(frames) != LIVE_NUM_FRAMES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Yeu cau dung {LIVE_NUM_FRAMES} frames, nhan duoc {len(frames)}. "
                f"(Cau hinh: {LIVE_NUM_FRAMES} frames trong "
                f"{LIVE_WINDOW_DURATION_MS / 1000:.1f}s cua so quan sat)"
            ),
        )

    # Keep original JPEG/PNG bytes for remote forwarding. Decode only in local mode.
    remote_frames: list[tuple[str, bytes, str]] = []
    decoded_frames: List[np.ndarray] = []
    for i, upload in enumerate(frames):
        try:
            data = await upload.read()
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Khong the doc frame thu {i + 1}: {e}",
            )
        finally:
            await upload.close()

        if len(data) == 0:
            raise HTTPException(
                status_code=422,
                detail=f"Frame thu {i + 1} rong (0 bytes).",
            )

        remote_frames.append(
            (
                Path(upload.filename or f"frame_{i:02d}.jpg").name,
                data,
                upload.content_type or "image/jpeg",
            )
        )

        if INFERENCE_BACKEND == "local":
            try:
                img_rgb = _decode_image_to_rgb(data, i)
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))
            decoded_frames.append(img_rgb)

    try:
        if INFERENCE_BACKEND == "remote":
            result = await remote_client.predict_frames(
                remote_frames,
                source_type=source_type,
                captured_duration_ms=captured_duration_ms,
                window_id=window_id,
            )
        else:
            result = await asyncio.to_thread(_run_live_inference, decoded_frames)
    except RemoteInferenceError as error:
        _raise_remote_error(error)
    except RuntimeError as e:
        if str(e) == "busy":
            raise HTTPException(
                status_code=429,
                detail=(
                    "Mo hinh dang xu ly mot yeu cau khac. "
                    "Vui long thu lai sau mot giay."
                ),
            )
        print("========== LIVE INFERENCE ERROR ==========")
        traceback.print_exc()
        print("==========================================")
        raise HTTPException(
            status_code=500,
            detail=f"Loi trong qua trinh phan tich live frame: {str(e)}",
        )
    except Exception as e:
        print("========== LIVE INFERENCE ERROR ==========")
        traceback.print_exc()
        print("==========================================")
        raise HTTPException(
            status_code=500,
            detail=f"Loi trong qua trinh phan tich live frame: {str(e)}",
        )

    return result
