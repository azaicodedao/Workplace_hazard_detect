"""
routes.py -- FastAPI router cho inference endpoints.

Lock strategy:
    - inference_lock: threading.Lock (khong phai asyncio.Lock)
      Vi inference (PyTorch forward pass) la synchronous va duoc chay trong
      worker thread thong qua asyncio.to_thread(). threading.Lock dam bao
      chi mot inference chay tren model/GPU tai mot thoi diem.

    - POST /predict/ (video upload):
        Goi asyncio.to_thread(_run_video_inference, ...).
        Worker thread acquire lock voi blocking=True (co the cho), giu nguyen
        trai nghiem cua chuc nang cu.

    - POST /predict/live/ (live camera):
        Goi asyncio.to_thread(_run_live_inference, ...).
        Worker thread thu acquire lock voi blocking=False. Neu that bai
        (model dang ban) -> raise RuntimeError("busy") -> route tra 429.
        Khong xep hang cho.

    Lock duoc giai phong trong finally block trong moi worker function,
    dam bao duoc giai phong ca khi inference sinh ra exception.

Live camera config:
    LIVE_NUM_FRAMES -- 16 frames required by VideoMAE.
    The browser captures frame 0 immediately and frame 15 at 5000ms,
    producing an exact five-second observation window.
"""

import os
import io
import shutil
import threading
import traceback
from typing import List

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.inference.predictor import VideoPredictor
from app.schemas.prediction import PredictionResponse
from app.config import (
    LIVE_DURATION_TOLERANCE_MS,
    LIVE_FRAME_COUNT,
    LIVE_WINDOW_DURATION_MS,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# Khoi tao model duy nhat -- khong tao them bat ky predictor nao khac
# ---------------------------------------------------------------------------
try:
    predictor = VideoPredictor()
except Exception as _e:
    print(f"Loi khoi tao mo hinh AI: {_e}")
    predictor = None

# ---------------------------------------------------------------------------
# Inference lock -- threading.Lock vi inference chay trong worker thread
# ---------------------------------------------------------------------------
inference_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Thu muc luu video tam
# ---------------------------------------------------------------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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
# POST /predict/   -- VIDEO UPLOAD (giu nguyen hoan toan)
# ---------------------------------------------------------------------------

@router.post("/predict/", response_model=PredictionResponse)
async def predict_video(file: UploadFile = File(...)):
    """
    Phan tich video upload. Endpoint nay KHONG thay doi so voi phien ban goc
    ngoai viec inference duoc chay qua asyncio.to_thread de tranh block event loop.
    """
    import asyncio

    if predictor is None:
        raise HTTPException(status_code=500, detail="Mo hinh AI chua duoc nap thanh cong.")

    # Kiem tra dinh dang file
    allowed_extensions = [".mp4", ".avi", ".mov"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Chi ho tro cac dinh dang video: {allowed_extensions}",
        )

    # Luu file tam vao dia
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Khong the luu video: {e}")

    # Chay inference trong worker thread (tranh block event loop)
    try:
        result = await asyncio.to_thread(_run_video_inference, file_path)
    except Exception as e:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
        print("========== INFERENCE ERROR ==========")
        traceback.print_exc()
        print("=====================================")
        raise HTTPException(
            status_code=500,
            detail=f"Loi trong qua trinh phan tich video: {str(e)}",
        )

    # Xoa file video sau khi phan tich xong de tiet kiem bo nho
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
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
    import asyncio

    if predictor is None:
        raise HTTPException(status_code=500, detail="Mo hinh AI chua duoc nap thanh cong.")

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

    # --- Doc va giai ma tung frame ---
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

        try:
            img_rgb = _decode_image_to_rgb(data, i)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        decoded_frames.append(img_rgb)

    # --- Chay inference trong worker thread ---
    try:
        result = await asyncio.to_thread(_run_live_inference, decoded_frames)
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
